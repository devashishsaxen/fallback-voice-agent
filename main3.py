import os
import requests
import uuid
import re
import tempfile
import io
import json
import traceback
from enum import Enum
from pathlib import Path
from datetime import datetime
import assemblyai as aai
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Record, Say, Pause

# Setup logging
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
CAMB_API_KEY = os.getenv("CAMB_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000")

# OPTIMIZATION: Use MP3 instead of WAV (smaller, faster)
USE_NATIVE_TTS = False  # Set True to use Twilio's instant TTS (fastest)
CACHE_TTS = True  # Cache common phrases

aai.settings.api_key = ASSEMBLYAI_API_KEY
TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "riya_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

twilio_client = Client(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID and TWILIO_TOKEN else None

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
    def __init__(self, phone_number=None):
        self.conversation = []
        self.state = ConversationState.GREETING
        self.candidate_type = None
        self.retry_count = 0
        self.answers = {}
        self.phone_number = phone_number
        self.recording_attempts = 0
        self.last_response_time = None

# Cache for TTS audio files
tts_cache = {}

def get_cache_key(text):
    """Generate cache key from text"""
    return hash(text) % 10000

def generate_tts(text: str, session_id: str, use_cache=True) -> str:
    """
    OPTIMIZED TTS with caching and MP3 support
    Returns URL or None (None = use Twilio native TTS)
    """
    if USE_NATIVE_TTS:
        return None  # Use Twilio's instant TTS
    
    try:
        # Check cache for common phrases
        cache_key = get_cache_key(text)
        if use_cache and cache_key in tts_cache:
            cached_path = tts_cache[cache_key]
            if os.path.exists(cached_path):
                log(f"TTS CACHE HIT: {text[:30]}...")
                filename = Path(cached_path).name
                return f"{PUBLIC_URL}/audio/{filename}"
        
        # Generate new audio
        audio_id = f"{session_id}_{uuid.uuid4().hex[:8]}.mp3"  # Use MP3 (smaller, faster)
        audio_path = TEMP_AUDIO_DIR / audio_id
        
        if not CAMB_API_KEY:
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
        
        log(f"TTS generating...")
        start_time = datetime.now()
        
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
        
        if response.status_code == 200:
            # Save MP3 directly (no conversion to WAV - saves 1-2 seconds)
            with open(audio_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            file_size = audio_path.stat().st_size
            
            if file_size > 1000:
                log(f"TTS done in {elapsed:.1f}s, size: {file_size} bytes")
                
                # Cache it
                if use_cache:
                    tts_cache[cache_key] = str(audio_path)
                
                return f"{PUBLIC_URL}/audio/{audio_id}"
        
        return None
            
    except Exception as e:
        log(f"TTS Error: {e}")
        return None

def check_story_quality(text: str) -> bool:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences) >= 8 or len(text.split()) >= 50

def get_reply(session: SessionData, user_input: str) -> str:
    user_lower = user_input.lower().strip()
    current_state = session.state
    
    if current_state == ConversationState.GREETING:
        session.state = ConversationState.INTEREST_CHECK
        return "Hi, this is Riya from Futuresoft Consultancy. We are hiring voice and chat profiles for companies like British Telecom, Teleperformance, and Wipro. Are you interested?"
    
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

sessions = {}

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.get("/audio/{audio_id}")
def get_audio(audio_id: str):
    audio_path = TEMP_AUDIO_DIR / audio_id
    if audio_path.exists():
        # Serve MP3 with correct MIME type
        return FileResponse(audio_path, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Audio not found")

@app.post("/initiate-call")
async def initiate_call(phone_number: str = Form(...)):
    if not twilio_client:
        raise HTTPException(status_code=500, detail="Twilio not configured")
    
    session_id = str(uuid.uuid4())
    session = SessionData(phone_number=phone_number)
    sessions[session_id] = session
    
    try:
        call = twilio_client.calls.create(
            to=phone_number,
            from_=TWILIO_PHONE,
            url=f"{PUBLIC_URL}/twilio-webhook?session_id={session_id}",
            status_callback=f"{PUBLIC_URL}/call-status?session_id={session_id}",
            status_callback_event=["completed", "answered"]
        )
        
        return {"success": True, "call_sid": call.sid, "session_id": session_id}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/twilio-webhook")
async def twilio_webhook(request: Request):
    start_time = datetime.now()
    form_data = await request.form()
    query_params = dict(request.query_params)
    
    session_id = query_params.get('session_id')
    recording_url = form_data.get('RecordingUrl')
    
    response = VoiceResponse()
    
    try:
        if not session_id or session_id not in sessions:
            response.say("Sorry, session expired.")
            return Response(content=str(response), media_type="application/xml")
        
        session = sessions[session_id]
        
        # Process recording
        user_input = ""
        if recording_url:
            try:
                auth = (TWILIO_SID, TWILIO_TOKEN)
                audio_response = requests.get(recording_url, auth=auth, timeout=10)  # Shorter timeout
                audio_content = audio_response.content
                
                if len(audio_content) > 1000:
                    # Use AssemblyAI with shorter timeout
                    config = aai.TranscriptionConfig(speech_models=["universal-2"], language_code="en_us")
                    transcriber = aai.Transcriber(config=config)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_content)
                        tmp_path = tmp.name
                    
                    transcript = transcriber.transcribe(tmp_path)
                    os.unlink(tmp_path)
                    
                    if transcript.text:
                        user_input = transcript.text
                        log(f"User said: {user_input[:50]}...")
            except Exception as e:
                log(f"Transcription error: {e}")
        
        # Get reply
        is_first_call = (not recording_url) and (session.state == ConversationState.GREETING)
        
        if is_first_call:
            reply = get_reply(session, "")
        elif not recording_url:
            session.recording_attempts += 1
            if session.recording_attempts >= 3:
                session.state = ConversationState.REJECTED
                reply = "Sorry, we will not be able to help you."
            else:
                reply = "I didn't hear you. Please speak after the beep."
        elif not user_input.strip():
            reply = "I didn't catch that. Could you please speak up?"
        else:
            reply = get_reply(session, user_input)
        
        # OPTIMIZED: Generate TTS (cached or native)
        audio_url = generate_tts(reply, session_id)
        
        if audio_url:
            response.play(audio_url)
        else:
            # FALLBACK: Instant Twilio TTS (zero latency)
            response.say(reply, voice="Polly.Joanna", language="en-US")
        
        # Continue conversation
        if session.state not in [ConversationState.REJECTED, ConversationState.COMPLETED]:
            if is_first_call:
                response.pause(length=1)
                response.say("Please say yes or no after the beep.", voice="Polly.Joanna")
            
            response.record(
                action=f"{PUBLIC_URL}/twilio-webhook?session_id={session_id}",
                max_length=20,  # Shorter max (was 60)
                play_beep=True,
                trim="trim-silence",
                timeout=10  # Shorter timeout (was 15)
            )
        else:
            response.hangup()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        log(f"Webhook processed in {elapsed:.2f}s")
        
        return Response(content=str(response), media_type="application/xml")
        
    except Exception as e:
        log(f"Error: {e}")
        response.say("Sorry, an error occurred.")
        return Response(content=str(response), media_type="application/xml")

@app.post("/call-status")
async def call_status(request: Request):
    form_data = await request.form()
    session_id = dict(request.query_params).get('session_id')
    log(f"Call status: {session_id} - {form_data.get('CallStatus')}")
    return {"status": "ok"}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Riya Voice Agent</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; background: #f0f2f5; }
            .container { background: white; padding: 30px; border-radius: 15px; }
            h1 { color: #1a73e8; text-align: center; }
            input { width: 100%; padding: 15px; font-size: 18px; border: 2px solid #ddd; border-radius: 10px; margin: 20px 0; box-sizing: border-box; }
            button { width: 100%; padding: 15px; background: #1a73e8; color: white; border: none; border-radius: 10px; font-size: 18px; cursor: pointer; }
            button:hover { background: #1557b0; }
            .status { margin-top: 20px; padding: 15px; border-radius: 8px; text-align: center; display: none; }
            .success { background: #e8f5e9; color: #2e7d32; }
            .error { background: #ffebee; color: #c62828; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Riya Voice Recruiter</h1>
            <input type="tel" id="phoneNumber" placeholder="+91XXXXXXXXXX" value="+91">
            <button onclick="makeCall()" id="callBtn">📞 Make Call</button>
            <div id="status" class="status"></div>
        </div>

        <script>
            async function makeCall() {
                const phone = document.getElementById('phoneNumber').value.trim();
                const status = document.getElementById('status');
                
                if (!phone || phone.length < 10) {
                    showStatus('Please enter a valid phone number', 'error');
                    return;
                }
                
                document.getElementById('callBtn').disabled = true;
                showStatus('Initiating call...', 'info');
                
                try {
                    const formData = new FormData();
                    formData.append('phone_number', phone);
                    
                    const res = await fetch('/initiate-call', { method: 'POST', body: formData });
                    const data = await res.json();
                    
                    if (data.success) {
                        showStatus(`✅ Call initiated! Answer your phone now!`, 'success');
                    } else {
                        showStatus(`❌ Error: ${data.error}`, 'error');
                    }
                } catch (e) {
                    showStatus(`❌ Error: ${e.message}`, 'error');
                } finally {
                    document.getElementById('callBtn').disabled = false;
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