import os
import httpx
import asyncio
import uuid
import re
import tempfile
import io
import json
import traceback
import base64
import wave
import audioop
from enum import Enum
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Form, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from pydub import AudioSegment

# Setup logging
def log(msg):
    safe_msg = str(msg).encode("ascii", errors="replace").decode("ascii")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {safe_msg}")

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def clean_error_text(text: str) -> str:
    """Remove ANSI color codes and normalize whitespace for UI display."""
    text = ANSI_ESCAPE_RE.sub("", text or "")
    text = text.replace("\r", "")
    return text.strip()

def friendly_twilio_error(error: Exception, phone_number: str) -> str:
    raw = clean_error_text(str(error))
    code = getattr(error, "code", None)
    lowered = raw.lower()

    # Twilio trial limitation: can only call verified numbers.
    if code == 21219 or "trial accounts may only make calls to verified numbers" in lowered:
        return (
            f"Twilio trial restriction: {phone_number} is not a verified number. "
            "Verify it in Twilio Console > Phone Numbers > Verified Caller IDs, "
            "or upgrade your Twilio account."
        )

    return raw or "Unknown Twilio error"

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
CAMB_API_KEY = os.getenv("CAMB_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000")

log(f"Starting server...")
log(f"PUBLIC_URL: {PUBLIC_URL}")
log(f"TWILIO_PHONE: {TWILIO_PHONE}")

_aai_transcriber = None

def get_aai_transcriber():
    """Load AssemblyAI lazily so webhook startup is not blocked by the SDK import."""
    global _aai_transcriber
    if _aai_transcriber is None:
        import assemblyai as aai

        aai.settings.api_key = ASSEMBLYAI_API_KEY
        config = aai.TranscriptionConfig(speech_models=["universal-2"], language_code="en_us")
        _aai_transcriber = aai.Transcriber(config=config)
    return _aai_transcriber

async def transcribe_audio_file(audio_path: str) -> str:
    """Use Deepgram for low-latency short utterances; fall back to AssemblyAI."""
    if DEEPGRAM_API_KEY:
        try:
            params = {
                "model": "nova-2",
                "language": "en",
                "smart_format": "false",
                "encoding": "linear16",
                "sample_rate": "8000",
                "channels": "1",
            }
            headers = {
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            }
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    "https://api.deepgram.com/v1/listen",
                    params=params,
                    headers=headers,
                    content=audio_bytes,
                )
            log(f"Deepgram STT status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                alternatives = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])
                transcript = alternatives[0].get("transcript", "") if alternatives else ""
                return transcript or ""
            log(f"Deepgram STT failed: {response.text[:200]}")
        except Exception as e:
            log(f"Deepgram STT error: {e}")

    transcript = await asyncio.to_thread(get_aai_transcriber().transcribe, audio_path)
    return transcript.text if transcript.text else ""

TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "riya_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)
log(f"Audio directory: {TEMP_AUDIO_DIR}")

# Twilio client
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID and TWILIO_TOKEN else None
if not twilio_client:
    log("WARNING: Twilio not configured!")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= SILENCE DETECTION CONFIG =================
# Twilio Media Streams sends µ-law audio: 8kHz, mono, ~160 bytes per 20ms frame
SILENCE_THRESHOLD = 300        # RMS below this = silence
SPEECH_MIN_FRAMES = 5          # Min speech frames before silence detection activates
SILENCE_DURATION_FRAMES = 25   # 25 frames x 20ms = 0.5 seconds of silence -> end of speech
NO_SPEECH_TIMEOUT_FRAMES = 500 # 10 seconds with zero speech → timeout
MAX_RECORDING_FRAMES = 1500    # 30 seconds max recording
USE_FAST_TWILIO_SAY_REPLIES = True

class ConversationState(str, Enum):
    GREETING = "greeting"
    INTEREST_CHECK = "interest_check"
    EXPERIENCE_CHECK = "experience_check"
    FRESHER_QUALIFICATION = "fresher_qualification"
    EXP_DETAILS = "exp_details"
    CUSTOMER_STORY = "customer_story"
    CUSTOMER_RETRY = "customer_retry"
    FESTIVAL_STORY = "festival_story"
    FESTIVAL_RETRY = "festival_retry"
    COMPLETED = "completed"
    REJECTED = "rejected"

class SessionData:
    def __init__(self, phone_number=None, public_url=None):
        self.conversation = []
        self.state = ConversationState.GREETING
        self.candidate_type = None
        self.retry_count = 0
        self.answers = {}
        self.phone_number = phone_number
        self.public_url = public_url
        self.current_audio_url = None
        self.recording_attempts = 0

def request_public_url(request: Request) -> str:
    """Use the incoming public host when the app is reached through a tunnel."""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    if host:
        base_url = f"{proto}://{host}".rstrip("/")
        if "localhost" not in host and "127.0.0.1" not in host:
            return base_url
    return PUBLIC_URL.rstrip("/")

# TTS cache: text -> public audio URL (avoids re-generating identical prompts)
TTS_CACHE = {}
GREETING_TEXT = (
    "Hi, this is Riya from Futuresoft Consultancy. We are hiring voice and chat profiles "
    "for companies like British Telecom, Teleperformance, and Wipro. Are you interested?"
)

async def generate_tts(text: str, session_id: str, public_url: str = None) -> str:
    """Generate TTS using CambAI and convert MP3 to WAV for Twilio."""
    log(f"=== TTS GENERATION START ===")
    log(f"Text: {text[:60]}...")
    
    try:
        audio_id = f"{session_id}_{uuid.uuid4().hex[:8]}.wav"
        audio_path = TEMP_AUDIO_DIR / audio_id
        
        if not CAMB_API_KEY:
            log("ERROR: CAMB_API_KEY not set!")
            return None
            
        url = "https://client.camb.ai/apis/tts-stream"
        headers = {
            "x-api-key": CAMB_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "language": "en-us",
            "voice_id": 147320,
            "speech_model": "mars-flash"
        }
        
        log(f"Calling CambAI API (async)...")
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, json=payload, headers=headers)
        
        log(f"CambAI Response Status: {response.status_code}")
        
        if response.status_code == 200:
            log(f"Converting MP3 to telephony WAV...")
            mp3_data = io.BytesIO(response.content)
            audio = AudioSegment.from_mp3(mp3_data)
            audio = audio.set_channels(1).set_frame_rate(8000).set_sample_width(2)
            if audio.dBFS != float("-inf") and audio.dBFS < -18:
                audio += min(8, -18 - audio.dBFS)
            audio.export(audio_path, format="wav")
            
            file_size = audio_path.stat().st_size
            log(f"SUCCESS: WAV file created: {audio_path} ({file_size} bytes)")
            
            if file_size < 1000:
                log(f"ERROR: File too small!")
                return None
                
            audio_url = f"{(public_url or PUBLIC_URL).rstrip('/')}/audio/{audio_id}"
            log(f"Audio URL: {audio_url}")
            return audio_url
        else:
            log(f"ERROR: CambAI returned status {response.status_code}")
            return None
            
    except Exception as e:
        log(f"CRITICAL TTS ERROR: {e}")
        traceback.print_exc()
        return None

async def get_or_generate_tts(text: str, session_id: str, public_url: str = None) -> str:
    """Return cached TTS audio URL if available, otherwise generate and cache."""
    cache_key = (public_url or PUBLIC_URL, text)
    if cache_key in TTS_CACHE:
        log(f"TTS CACHE HIT: '{text[:40]}...'")
        return TTS_CACHE[cache_key]
    
    audio_url = await generate_tts(text, session_id, public_url)
    if audio_url:
        TTS_CACHE[cache_key] = audio_url
    return audio_url

def check_story_quality(text: str) -> bool:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences) >= 8 or len(text.split()) >= 50

def get_reply(session: SessionData, user_input: str) -> str:
    user_lower = user_input.lower().strip()
    current_state = session.state
    
    if current_state == ConversationState.GREETING:
        session.state = ConversationState.INTEREST_CHECK
        return GREETING_TEXT
    
    elif current_state == ConversationState.INTEREST_CHECK:
        if any(word in user_lower for word in ['no', 'not', 'nah', 'nope']):
            session.state = ConversationState.REJECTED
            return "I understand. Thank you for your time. Have a great day!"
        elif any(word in user_lower for word in ['yes', 'yeah', 'sure', 'interested', 'ok']):
            session.state = ConversationState.EXPERIENCE_CHECK
            return "We will surely help you with the same. Could you please confirm me if you are Fresher OR Experienced?"
        else:
            return "Could you please confirm if you are interested? Just say Yes or No."
    
    elif current_state == ConversationState.EXPERIENCE_CHECK:
        if any(word in user_lower for word in ['fresher', 'fresh', 'student']):
            session.candidate_type = 'fresher'
            session.state = ConversationState.FRESHER_QUALIFICATION
            return "Now, what's your highest qualification like Graduate, Undergraduate, or Graduation drop-out?"
        elif any(word in user_lower for word in ['experience', 'experienced', 'worked']):
            session.candidate_type = 'experienced'
            session.state = ConversationState.EXP_DETAILS
            return "Now, please confirm your highest qualification and experience. Mention your job responsibility part clearly."
        else:
            return "Could you please clarify - are you a Fresher or Experienced?"
    
    elif current_state == ConversationState.FRESHER_QUALIFICATION:
        session.answers['qualification'] = user_input
        session.state = ConversationState.CUSTOMER_STORY
        return "That was very impressive. Could you please speak about any memorable interaction with customer within 10 to 12 sentences. You can start with, 'Once a customer called me for issue related to...' And your time starts now."
    
    elif current_state == ConversationState.EXP_DETAILS:
        session.answers['experience'] = user_input
        session.state = ConversationState.CUSTOMER_STORY
        return "That was very impressive. Could you please speak about any memorable interaction with customer within 10 to 12 sentences. You can start with, 'Once a customer called me for issue related to...' And your time starts now."
    
    elif current_state == ConversationState.CUSTOMER_STORY:
        is_valid = check_story_quality(user_input)
        if is_valid:
            session.answers['customer_story'] = user_input
            session.state = ConversationState.FESTIVAL_STORY
            return "Acknowledgment to statement. Could you please speak about any latest festival you celebrated like Diwali, Holi, Christmas or Eid in 10 to 12 sentences. Start with, 'I celebrated my last Diwali along with family...' And your time starts now."
        else:
            session.retry_count += 1
            if session.retry_count >= 2:
                session.state = ConversationState.REJECTED
                return "Sorry, we will not be able to help you with job as we hire candidates with good communication skills only."
            else:
                session.state = ConversationState.CUSTOMER_RETRY
                return "Sorry, you need to speak only 10 to 12 sentences on this topic. It can be done within 15 seconds only. Please speak on this topic now."
    
    elif current_state == ConversationState.CUSTOMER_RETRY:
        is_valid = check_story_quality(user_input)
        if is_valid:
            session.answers['customer_story'] = user_input
            session.state = ConversationState.FESTIVAL_STORY
            return "Acknowledgment to statement. Could you please speak about any latest festival you celebrated like Diwali, Holi, Christmas or Eid in 10 to 12 sentences. Start with, 'I celebrated my last Diwali along with family...' And your time starts now."
        else:
            session.state = ConversationState.REJECTED
            return "Sorry, we will not be able to help you with job as we hire candidates with good communication skills only."
    
    elif current_state == ConversationState.FESTIVAL_STORY:
        is_valid = check_story_quality(user_input)
        if is_valid:
            session.answers['festival'] = user_input
            session.state = ConversationState.COMPLETED
            return "That was amazing, now one of our HR Recruiter will connect you for your further interview process."
        else:
            session.retry_count += 1
            if session.retry_count >= 2:
                session.state = ConversationState.REJECTED
                return "Sorry, we will not be able to help you with job as we hire candidates with good communication skills only."
            else:
                session.state = ConversationState.FESTIVAL_RETRY
                return "Sorry, please speak clearly about the festival celebration for 10 to 12 sentences to proceed."
    
    elif current_state == ConversationState.FESTIVAL_RETRY:
        is_valid = check_story_quality(user_input)
        if is_valid:
            session.answers['festival'] = user_input
            session.state = ConversationState.COMPLETED
            return "That was amazing, now one of our HR Recruiter will connect you for your further interview process."
        else:
            session.state = ConversationState.REJECTED
            return "Sorry, we will not be able to help you with job as we hire candidates with good communication skills only."
    
    elif current_state in [ConversationState.REJECTED, ConversationState.COMPLETED]:
        return "Thank you for your time. Have a great day!"
    
    return "I'm sorry, could you please repeat that?"

# Store sessions
sessions = {}

@app.get("/", response_class=HTMLResponse)
def read_root():
    # Serve the built-in dashboard instead of a missing index.html file.
    return dashboard()

@app.get("/debug/files")
def debug_files():
    """Debug endpoint to check what audio files exist"""
    files = []
    try:
        for f in TEMP_AUDIO_DIR.iterdir():
            if f.is_file():
                files.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                })
    except Exception as e:
        return {"error": str(e), "directory": str(TEMP_AUDIO_DIR)}
    
    return {
        "directory": str(TEMP_AUDIO_DIR),
        "file_count": len(files),
        "files": files
    }

@app.get("/audio/{audio_id}")
def get_audio(audio_id: str):
    audio_path = TEMP_AUDIO_DIR / audio_id
    if audio_path.exists():
        return FileResponse(audio_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio not found")

# ================= TWILIO CALL HANDLING =================

@app.post("/initiate-call")
async def initiate_call(request: Request, phone_number: str = Form(...)):
    """Initiate a call to the specified phone number."""
    log(f"=== INITIATE CALL ===")
    log(f"Phone: {phone_number}")
    
    if not twilio_client:
        raise HTTPException(status_code=500, detail="Twilio not configured")
    
    public_url = request_public_url(request)
    session_id = str(uuid.uuid4())
    session = SessionData(phone_number=phone_number, public_url=public_url)
    sessions[session_id] = session
    log(f"Public URL for session: {public_url}")
    
    try:
        asyncio.create_task(asyncio.to_thread(get_aai_transcriber))
        session.current_audio_url = await get_or_generate_tts(GREETING_TEXT, session_id, public_url)
        log(f"Preloaded greeting audio: {session.current_audio_url}")

        call = twilio_client.calls.create(
            to=phone_number,
            from_=TWILIO_PHONE,
            url=f"{public_url}/twilio-webhook?session_id={session_id}",
            status_callback=f"{public_url}/call-status?session_id={session_id}",
            status_callback_event=["completed", "answered"]
        )
        
        return {
            "success": True, 
            "call_sid": call.sid, 
            "session_id": session_id,
            "message": f"Calling {phone_number}..."
        }
    except Exception as e:
        err = friendly_twilio_error(e, phone_number)
        log(f"Call initiation failed: {err}")
        return {"success": False, "error": err}

@app.post("/twilio-webhook")
async def twilio_webhook(request: Request):
    """Handle initial Twilio call — play greeting, then connect to Media Stream."""
    query_params = dict(request.query_params)
    session_id = query_params.get('session_id')

    log(f"\n{'='*60}")
    log(f"WEBHOOK CALLED — Session: {session_id}")
    log(f"{'='*60}")

    response = VoiceResponse()

    try:
        if not session_id or session_id not in sessions:
            log(f"ERROR: Session {session_id} not found!")
            response.say("Sorry, session expired.")
            return Response(content=str(response), media_type="application/xml")

        session = sessions[session_id]
        public_url = session.public_url or request_public_url(request)
        log(f"Current state: {session.state}")

        # Play greeting on first call
        if session.state == ConversationState.GREETING:
            reply = get_reply(session, "")
            audio_url = session.current_audio_url or await get_or_generate_tts(reply, session_id, public_url)
            if audio_url:
                response.play(audio_url)
            else:
                response.say(reply, voice="Polly.Joanna", language="en-US")
            response.say("Please say yes or no.", voice="Polly.Joanna")

        # Connect to Media Stream for real-time audio + silence detection
        if session.state not in [ConversationState.REJECTED, ConversationState.COMPLETED]:
            ws_url = public_url.replace('https://', 'wss://').replace('http://', 'ws://')
            connect = response.connect()
            stream = connect.stream(url=f"{ws_url}/media-stream")
            stream.parameter(name="session_id", value=session_id)
        else:
            response.hangup()

        return Response(content=str(response), media_type="application/xml")

    except Exception as e:
        log(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        response.say("Sorry, an error occurred.")
        return Response(content=str(response), media_type="application/xml")


# ================= MEDIA STREAM (SILENCE DETECTION) =================

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Receive real-time audio from Twilio, detect silence, then transcribe + respond."""
    await websocket.accept()

    session_id = websocket.query_params.get('session_id')
    session = sessions.get(session_id) if session_id else None
    audio_buffer = bytearray()
    speech_started = False
    speech_frames = 0
    silence_frames = 0
    total_frames = 0
    call_sid = None
    stream_sid = None

    log(f"Media stream opened - pending session={session_id}")

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            event = data.get('event')

            if event == 'connected':
                log("Media stream connected")

            elif event == 'start':
                stream_sid = data['start']['streamSid']
                call_sid = data['start']['callSid']
                custom_params = data['start'].get('customParameters') or {}
                session_id = session_id or custom_params.get('session_id')
                session = sessions.get(session_id) if session_id else None
                if not session:
                    log(f"Media stream: invalid session {session_id}")
                    await websocket.close()
                    return
                log(f"Stream started — SID: {stream_sid}, Call: {call_sid}")

            elif event == 'media':
                if not session:
                    continue

                chunk = base64.b64decode(data['media']['payload'])
                total_frames += 1

                # Decode µ-law → 16-bit PCM and compute RMS energy
                pcm_chunk = audioop.ulaw2lin(chunk, 2)
                rms = audioop.rms(pcm_chunk, 2)

                if rms > SILENCE_THRESHOLD:
                    speech_started = True
                    speech_frames += 1
                    silence_frames = 0
                    audio_buffer.extend(chunk)
                elif speech_started:
                    silence_frames += 1
                    audio_buffer.extend(chunk)  # keep trailing silence for STT quality

                # --- Decide if we should process ---
                should_process = False
                no_speech_timeout = False

                if speech_started and speech_frames >= SPEECH_MIN_FRAMES and silence_frames >= SILENCE_DURATION_FRAMES:
                    log(f"✅ Silence detected after {speech_frames} speech frames ({total_frames * 20}ms total)")
                    should_process = True
                elif total_frames >= MAX_RECORDING_FRAMES:
                    log(f"⏱️ Max recording duration reached")
                    should_process = True
                elif not speech_started and total_frames >= NO_SPEECH_TIMEOUT_FRAMES:
                    log(f"⏱️ No speech timeout")
                    no_speech_timeout = True

                # --- No speech timeout: prompt and reconnect ---
                if no_speech_timeout:
                    session.recording_attempts += 1
                    if session.recording_attempts >= 3:
                        session.state = ConversationState.REJECTED
                        reply = "Sorry, we will not be able to help you as we hire candidates with good communication skills only."
                    else:
                        reply = "I didn't hear you. Please speak clearly."

                    await _respond_and_reconnect(reply, session, session_id, call_sid, websocket)
                    break

                # --- Speech detected + silence → transcribe + reply ---
                if should_process:
                    session.recording_attempts = 0

                    # Convert µ-law buffer → WAV file
                    pcm_data = audioop.ulaw2lin(bytes(audio_buffer), 2)
                    tmp_path = tempfile.mktemp(suffix='.wav')
                    with wave.open(tmp_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(8000)
                        wf.writeframes(pcm_data)

                    log(f"Transcribing {len(audio_buffer)} bytes of audio...")
                    user_input = await transcribe_audio_file(tmp_path)
                    os.unlink(tmp_path)

                    log(f"✅ User said: '{user_input}'")

                    if not user_input.strip():
                        reply = "I didn't catch that. Could you please speak up?"
                    else:
                        reply = get_reply(session, user_input)
                        log(f"Reply: {reply[:60]}...")

                    await _respond_and_reconnect(reply, session, session_id, call_sid, websocket)
                    break

            elif event == 'stop':
                log("Stream stopped by Twilio")
                break

    except Exception as e:
        log(f"❌ Media stream error: {e}")
        traceback.print_exc()
    finally:
        log(f"Media stream closed — session={session_id}")


async def _respond_and_reconnect(reply: str, session: SessionData, session_id: str, call_sid: str, websocket: WebSocket):
    """Generate TTS, redirect the call to play response, and optionally reconnect stream."""
    public_url = (session.public_url or PUBLIC_URL).rstrip("/")
    audio_url = None if USE_FAST_TWILIO_SAY_REPLIES else await get_or_generate_tts(reply, session_id, public_url)

    new_response = VoiceResponse()
    if audio_url:
        new_response.play(audio_url)
    else:
        new_response.say(reply, voice="Polly.Joanna", language="en-US")

    if session.state not in [ConversationState.REJECTED, ConversationState.COMPLETED]:
        # Add a brief prompt before listening again
        if session.state in [ConversationState.INTEREST_CHECK, ConversationState.EXPERIENCE_CHECK]:
            new_response.say("Please respond now.", voice="Polly.Joanna")
        else:
            new_response.say("Please speak now.", voice="Polly.Joanna")

        # Reconnect to media stream for the next turn
        ws_url = public_url.replace('https://', 'wss://').replace('http://', 'ws://')
        connect = new_response.connect()
        stream = connect.stream(url=f"{ws_url}/media-stream")
        stream.parameter(name="session_id", value=session_id)
    else:
        log("Conversation ended")
        new_response.hangup()

    # Redirect the live call with new TwiML
    log(f"Redirecting call {call_sid}...")
    try:
        twilio_client.calls(call_sid).update(twiml=str(new_response))
    except TwilioRestException as e:
        log(f"Call redirect skipped: {clean_error_text(str(e))}")

@app.post("/call-status")
async def call_status(request: Request):
    """Handle call status callbacks."""
    form_data = await request.form()
    session_id = dict(request.query_params).get('session_id')
    call_status = form_data.get('CallStatus')
    log(f"Call status for {session_id}: {call_status}")
    return {"status": "ok"}

# ================= WEB INTERFACE =================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Riya Voice Agent</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; background: #f0f2f5; }
            .container { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #1a73e8; text-align: center; }
            input[type="tel"] { width: 100%; padding: 15px; font-size: 18px; border: 2px solid #ddd; border-radius: 10px; margin: 20px 0; box-sizing: border-box; }
            button { width: 100%; padding: 15px; background: #1a73e8; color: white; border: none; border-radius: 10px; font-size: 18px; cursor: pointer; }
            button:hover { background: #1557b0; }
            button:disabled { background: #ccc; }
            .status { margin-top: 20px; padding: 15px; border-radius: 8px; text-align: center; display: none; }
            .success { background: #e8f5e9; color: #2e7d32; }
            .error { background: #ffebee; color: #c62828; }
            .instructions { margin: 20px 0; padding: 15px; background: #e8f5e9; border-radius: 8px; border-left: 4px solid #4caf50; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Riya Voice Recruiter</h1>
            
            <div class="instructions">
                <strong>How to use:</strong>
                <ol>
                    <li>Enter your phone number</li>
                    <li>Click "Make Call"</li>
                    <li><strong>Answer the call immediately</strong></li>
                    <li><strong>Listen for the beep</strong> after Riya speaks</li>
                    <li><strong>Speak clearly after the beep</strong></li>
                </ol>
            </div>
            
            <input type="tel" id="phoneNumber" placeholder="+91XXXXXXXXXX" value="+91">
            <button onclick="makeCall()" id="callBtn">📞 Make Call</button>
            
            <div id="status" class="status"></div>
        </div>

        <script>
            async function makeCall() {
                const phone = document.getElementById('phoneNumber').value.trim();
                const btn = document.getElementById('callBtn');
                const status = document.getElementById('status');
                
                if (!phone || phone.length < 10) {
                    showStatus('Please enter a valid phone number', 'error');
                    return;
                }
                
                btn.disabled = true;
                btn.textContent = 'Calling...';
                showStatus('Initiating call...', 'info');
                
                try {
                    const formData = new FormData();
                    formData.append('phone_number', phone);
                    
                    const res = await fetch('/initiate-call', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await res.json();
                    
                    if (data.success) {
                        showStatus(`✅ Call initiated!<br><br>📞 <strong>Answer your phone NOW!</strong><br>🎤 Speak after the beep!`, 'success');
                    } else {
                        showStatus(`❌ Error: ${data.error}`, 'error');
                    }
                } catch (e) {
                    showStatus(`❌ Error: ${e.message}`, 'error');
                } finally {
                    btn.disabled = false;
                    btn.textContent = '📞 Make Call';
                }
            }
            
            function showStatus(msg, type) {
                const status = document.getElementById('status');
                status.innerHTML = msg;
                status.className = 'status ' + type;
                status.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
