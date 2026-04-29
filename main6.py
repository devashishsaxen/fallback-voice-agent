import os
import requests
import uuid
import re
import tempfile
import io
import json
import traceback
import asyncio
import websockets
import base64
import audioop
from enum import Enum
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Stream, Say
from pydub import AudioSegment

# Setup logging
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CAMB_API_KEY = os.getenv("CAMB_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000")

log(f"Starting server...")
log(f"PUBLIC_URL: {PUBLIC_URL}")
log(f"TWILIO_PHONE: {TWILIO_PHONE}")
log(f"Deepgram API Key present: {'Yes' if DEEPGRAM_API_KEY else 'NO - ERROR!'}")

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
        self.current_audio_url = None
        self.recording_attempts = 0
        self.greeting_played = False
        self.processing = False

def generate_tts(text: str, session_id: str) -> str:
    """Generate TTS using CambAI and save to file."""
    log(f"=== TTS GENERATION ===")
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
        
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        
        if response.status_code == 200:
            mp3_data = io.BytesIO(response.content)
            audio = AudioSegment.from_mp3(mp3_data)
            audio.export(audio_path, format="wav")
            
            file_size = audio_path.stat().st_size
            log(f"SUCCESS: {audio_path} ({file_size} bytes)")
            
            if file_size < 1000:
                return None
                
            return str(audio_path)
        else:
            log(f"ERROR: CambAI status {response.status_code}")
            return None
            
    except Exception as e:
        log(f"TTS ERROR: {e}")
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
        elif any(word in user_lower for word in ['yes', 'yeah', 'sure', 'interested', 'ok', 'yep']):
            session.state = ConversationState.EXPERIENCE_CHECK
            return "Great! Could you please tell me if you are a Fresher or Experienced?"
        else:
            return "Please say Yes if you're interested, or No if you're not."
    
    elif current_state == ConversationState.EXPERIENCE_CHECK:
        if any(word in user_lower for word in ['fresher', 'fresh', 'student', 'graduate']):
            session.candidate_type = 'fresher'
            session.state = ConversationState.FRESHER_QUALIFICATION
            return "What's your highest qualification? Graduate, Undergraduate, or Drop-out?"
        elif any(word in user_lower for word in ['experience', 'experienced', 'worked', 'job']):
            session.candidate_type = 'experienced'
            session.state = ConversationState.EXP_DETAILS
            return "Please share your highest qualification and years of experience briefly."
        else:
            return "Are you a Fresher or Experienced? Please clarify."
    
    elif current_state == ConversationState.FRESHER_QUALIFICATION:
        session.answers['qualification'] = user_input
        session.state = ConversationState.CUSTOMER_STORY
        return "Good. Now please describe a memorable customer interaction in 10 to 12 sentences. You can start with 'Once a customer called me regarding...'"
    
    elif current_state == ConversationState.EXP_DETAILS:
        session.answers['experience'] = user_input
        session.state = ConversationState.CUSTOMER_STORY
        return "Good. Now please describe a memorable customer interaction in 10 to 12 sentences. You can start with 'Once a customer called me regarding...'"
    
    elif current_state == ConversationState.CUSTOMER_STORY:
        is_valid = check_story_quality(user_input)
        if is_valid:
            session.answers['customer_story'] = user_input
            session.state = ConversationState.FESTIVAL_STORY
            return "Excellent. Now tell me about a recent festival you celebrated in 10 to 12 sentences. Start with 'I celebrated...'"
        else:
            session.retry_count += 1
            if session.retry_count >= 2:
                session.state = ConversationState.REJECTED
                return "Sorry, we need candidates with good communication skills. Thank you for your time."
            else:
                session.state = ConversationState.CUSTOMER_RETRY
                return "Please speak for at least 10 sentences about this topic. Go ahead."
    
    elif current_state == ConversationState.CUSTOMER_RETRY:
        is_valid = check_story_quality(user_input)
        if is_valid:
            session.answers['customer_story'] = user_input
            session.state = ConversationState.FESTIVAL_STORY
            return "Excellent. Now tell me about a recent festival you celebrated in 10 to 12 sentences."
        else:
            session.state = ConversationState.REJECTED
            return "Sorry, we need candidates with good communication skills. Thank you for your time."
    
    elif current_state == ConversationState.FESTIVAL_STORY:
        is_valid = check_story_quality(user_input)
        if is_valid:
            session.answers['festival'] = user_input
            session.state = ConversationState.COMPLETED
            return "Perfect! Our HR team will contact you soon for the next steps. Thank you!"
        else:
            session.retry_count += 1
            if session.retry_count >= 2:
                session.state = ConversationState.REJECTED
                return "Sorry, we need candidates with good communication skills. Thank you for your time."
            else:
                session.state = ConversationState.FESTIVAL_RETRY
                return "Please describe the festival celebration in 10 to 12 sentences."
    
    elif current_state == ConversationState.FESTIVAL_RETRY:
        is_valid = check_story_quality(user_input)
        if is_valid:
            session.answers['festival'] = user_input
            session.state = ConversationState.COMPLETED
            return "Perfect! Our HR team will contact you soon. Thank you!"
        else:
            session.state = ConversationState.REJECTED
            return "Sorry, we need candidates with good communication skills. Thank you for your time."
    
    elif current_state in [ConversationState.REJECTED, ConversationState.COMPLETED]:
        return "Thank you for your time. Have a great day!"
    
    return "I didn't catch that. Could you please repeat?"

# Store sessions
sessions = {}

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.get("/audio/{audio_id}")
def get_audio(audio_id: str):
    audio_path = TEMP_AUDIO_DIR / audio_id
    if audio_path.exists():
        return FileResponse(audio_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio not found")

@app.post("/initiate-call")
async def initiate_call(phone_number: str = Form(...)):
    """Initiate a call to the specified phone number."""
    log(f"=== INITIATE CALL ===")
    log(f"Phone: {phone_number}")
    
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
        
        return {
            "success": True, 
            "call_sid": call.sid, 
            "session_id": session_id,
            "message": f"Calling {phone_number}..."
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/twilio-webhook")
async def twilio_webhook(request: Request):
    """
    Return Stream immediately - don't wait for TTS generation.
    """
    query_params = dict(request.query_params)
    session_id = query_params.get('session_id')
    
    log(f"Webhook - Session: {session_id}")
    
    response = VoiceResponse()
    
    if not session_id or session_id not in sessions:
        response.say("Session error.")
        return Response(content=str(response), media_type="application/xml")
    
    # Connect to Media Stream immediately
    parsed_url = urlparse(PUBLIC_URL)
    ws_scheme = "wss" if parsed_url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed_url.netloc}/media-stream"
    
    connect = response.connect()
    stream = Stream(url=ws_url)
    stream.parameter(name="session_id", value=session_id)
    connect.append(stream)
    
    log(f"Returning Stream URL: {ws_url}")
    
    return Response(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Handle bidirectional stream with Deepgram Flux
    """
    await websocket.accept()
    log("=== MEDIA STREAM CONNECTED ===")
    
    if not DEEPGRAM_API_KEY:
        log("ERROR: No Deepgram API Key!")
        await websocket.close()
        return
    
    stream_sid = None
    call_sid = None
    session_id = None
    dg_ws = None
    session = None
    start_received = False
    audio_buffer = []
    
    try:
        # Phase 1: Wait for Twilio Start Event
        while not start_received:
            message = await websocket.receive_text()
            data = json.loads(message)
            event_type = data.get("event")
            
            if event_type == "connected":
                log("Twilio connected, waiting for start...")
                continue
                
            elif event_type == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"]["callSid"]
                custom_params = data["start"].get("customParameters", {})
                session_id = custom_params.get("session_id")
                
                log(f"Stream started: {stream_sid}, Session: {session_id}")
                
                if not session_id or session_id not in sessions:
                    log("Invalid session")
                    await websocket.close()
                    return
                
                session = sessions[session_id]
                start_received = True
                
            elif event_type == "media":
                audio_buffer.append(data["media"]["payload"])
                
        # Phase 2: Connect to Deepgram Flux with proper error handling
        ws_url = (
            "wss://api.deepgram.com/v1/listen?"  # Fixed: Use v1 instead of v2 for better compatibility
            "model=general&"  # Fixed: Use general model or nova-2 if flux fails
            "encoding=mulaw&"
            "sample_rate=8000&"
            "channels=1&"
            "endpointing=500&"  # 500ms silence detection
            "vad_turnoff=500&"  # Voice activity detection turnoff
            "interim_results=false&"  # Only final results
            "smart_format=true"
        )
        
        # Alternative Flux URL (if you specifically need Flux):
        # ws_url = (
        #     "wss://api.deepgram.com/v2/listen?"
        #     "model=flux-general-en&"
        #     "encoding=mulaw&"
        #     "sample_rate=8000&"
        #     "channels=1&"
        #     "eot_threshold=0.6&"
        #     "eager_eot_threshold=0.4"
        # )
        
        log(f"Connecting to Deepgram: {ws_url[:60]}...")
        
        # For websockets 13.1+, use extra_headers for auth (not additional_headers)
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        
        try:
            dg_ws = await websockets.connect(
                ws_url,
                extra_headers=headers
            )
            log("Deepgram WebSocket connected successfully")
        except Exception as e:
            log(f"Deepgram connection failed: {e}")
            # Fallback: try with Authorization in query params
            ws_url_with_auth = f"{ws_url}&token={DEEPGRAM_API_KEY}"
            dg_ws = await websockets.connect(ws_url_with_auth)
            log("Deepgram connected (fallback method)")
            
        # Flush buffered audio
        if audio_buffer:
            log(f"Flushing {len(audio_buffer)} buffered packets")
            for payload in audio_buffer:
                await dg_ws.send(base64.b64decode(payload))
            audio_buffer = []
        
        # Phase 3: Send Greeting
        if not session.greeting_played:
            original_state = session.state
            session.state = ConversationState.GREETING
            greeting_text = get_reply(session, "")
            
            audio_path = generate_tts(greeting_text, session_id)
            if audio_path:
                await send_audio_to_twilio(websocket, stream_sid, audio_path)
                session.greeting_played = True
                log("Greeting sent")
                await asyncio.sleep(0.5)
        
        # Phase 4: Bidirectional streaming
        async def twilio_to_deepgram():
            try:
                while True:
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    
                    if data["event"] == "media":
                        payload = base64.b64decode(data["media"]["payload"])
                        if dg_ws and dg_ws.open:
                            await dg_ws.send(payload)
                    
                    elif data["event"] == "stop":
                        log("Twilio stop received")
                        if dg_ws and dg_ws.open:
                            await dg_ws.close()
                        break
                    elif data["event"] == "mark":
                        continue
                        
            except WebSocketDisconnect:
                log("Twilio disconnected")
            except Exception as e:
                log(f"Twilio->Deepgram error: {e}")
        
        async def deepgram_to_twilio():
            nonlocal session, stream_sid
            
            try:
                async for message in dg_ws:
                    try:
                        data = json.loads(message)
                        
                        # Handle Deepgram responses
                        if data.get("type") == "Results":
                            # Standard Deepgram format
                            channel = data.get("channel", {})
                            alternatives = channel.get("alternatives", [])
                            if alternatives:
                                transcript = alternatives[0].get("transcript", "")
                                is_final = data.get("is_final", False)
                                
                                if is_final and transcript.strip():
                                    log(f"User said: {transcript}")
                                    
                                    if session.processing:
                                        continue
                                    
                                    session.processing = True
                                    reply = get_reply(session, transcript)
                                    log(f"AI: {reply[:60]}...")
                                    
                                    audio_path = generate_tts(reply, session_id)
                                    if audio_path:
                                        await send_audio_to_twilio(websocket, stream_sid, audio_path)
                                        
                                        if session.state in [ConversationState.REJECTED, ConversationState.COMPLETED]:
                                            log("Call ending")
                                            await asyncio.sleep(1)
                                            return
                                    
                                    session.processing = False
                        
                        elif data.get("type") == "TurnInfo":
                            # Flux format
                            event = data.get("event")
                            transcript = data.get("transcript", "")
                            
                            if event in ["EagerEndOfTurn", "EndOfTurn"]:
                                if session.processing or not transcript.strip():
                                    continue
                                
                                session.processing = True
                                log(f"User said: {transcript}")
                                
                                reply = get_reply(session, transcript)
                                audio_path = generate_tts(reply, session_id)
                                
                                if audio_path:
                                    await send_audio_to_twilio(websocket, stream_sid, audio_path)
                                    
                                    if session.state in [ConversationState.REJECTED, ConversationState.COMPLETED]:
                                        return
                                
                                session.processing = False
                                await dg_ws.send(json.dumps({"type": "Finalize"}))
                                
                    except Exception as e:
                        log(f"Message processing error: {e}")
                        
            except Exception as e:
                log(f"Deepgram loop error: {e}")
        
        # Run both directions
        await asyncio.gather(
            twilio_to_deepgram(),
            deepgram_to_twilio()
        )
            
    except Exception as e:
        log(f"Media stream fatal error: {e}")
        traceback.print_exc()
    finally:
        log("Cleaning up")
        if dg_ws and dg_ws.open:
            await dg_ws.close()
        try:
            await websocket.close()
        except:
            pass

async def send_audio_to_twilio(websocket, stream_sid, audio_path):
    """Convert WAV to mulaw and stream to Twilio"""
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(8000).set_channels(1)
        raw_data = audio.raw_data
        mulaw_data = audioop.lin2ulaw(raw_data, audio.sample_width)
        
        chunk_size = 160  # 20ms chunks
        for i in range(0, len(mulaw_data), chunk_size):
            chunk = mulaw_data[i:i+chunk_size]
            payload = base64.b64encode(chunk).decode('utf-8')
            
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload}
            })
        
        await websocket.send_json({
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {"name": "response_end"}
        })
        
        log(f"Audio sent: {len(mulaw_data)} bytes")
        
    except Exception as e:
        log(f"Send audio error: {e}")

@app.post("/call-status")
async def call_status(request: Request):
    form_data = await request.form()
    session_id = dict(request.query_params).get('session_id')
    call_status = form_data.get('CallStatus')
    log(f"Call status {session_id}: {call_status}")
    return {"status": "ok"}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Riya Voice Agent</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; background: #f0f2f5; }
        .container { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #1a73e8; text-align: center; }
        input[type="tel"] { width: 100%; padding: 15px; font-size: 18px; border: 2px solid #ddd; border-radius: 10px; margin: 20px 0; }
        button { width: 100%; padding: 15px; background: #1a73e8; color: white; border: none; border-radius: 10px; font-size: 18px; cursor: pointer; }
        button:hover { background: #1557b0; }
        button:disabled { background: #ccc; }
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
            const btn = document.getElementById('callBtn');
            const status = document.getElementById('status');
            
            if (!phone || phone.length < 10) {
                showStatus('Please enter a valid phone number', 'error');
                return;
            }
            
            btn.disabled = true;
            showStatus('Calling...', 'info');
            
            try {
                const formData = new FormData();
                formData.append('phone_number', phone);
                
                const res = await fetch('/initiate-call', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await res.json();
                
                if (data.success) {
                    showStatus('✅ Call initiated! Answer your phone.', 'success');
                } else {
                    showStatus('❌ Error: ' + data.error, 'error');
                }
            } catch (e) {
                showStatus('❌ Error: ' + e.message, 'error');
            } finally {
                btn.disabled = false;
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
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)