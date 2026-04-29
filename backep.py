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
from pydub import AudioSegment

# Setup logging
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
CAMB_API_KEY = os.getenv("CAMB_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000")

log(f"Starting server...")
log(f"PUBLIC_URL: {PUBLIC_URL}")
log(f"TWILIO_PHONE: {TWILIO_PHONE}")

# Setup AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY
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

def generate_tts(text: str, session_id: str) -> str:
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
        
        log(f"Calling CambAI API...")
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        
        log(f"CambAI Response Status: {response.status_code}")
        
        if response.status_code == 200:
            log(f"Converting MP3 to WAV...")
            mp3_data = io.BytesIO(response.content)
            audio = AudioSegment.from_mp3(mp3_data)
            audio.export(audio_path, format="wav")
            
            file_size = audio_path.stat().st_size
            log(f"SUCCESS: WAV file created: {audio_path} ({file_size} bytes)")
            
            if file_size < 1000:
                log(f"ERROR: File too small!")
                return None
                
            public_url = f"{PUBLIC_URL}/audio/{audio_id}"
            log(f"Audio URL: {public_url}")
            return public_url
        else:
            log(f"ERROR: CambAI returned status {response.status_code}")
            return None
            
    except Exception as e:
        log(f"CRITICAL TTS ERROR: {e}")
        traceback.print_exc()
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

# Store sessions
sessions = {}

@app.get("/")
def read_root():
    return FileResponse("index.html")

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
    Handle Twilio webhooks - FIXED to properly read form data and authenticate recording downloads
    """
    # Read all form data properly (Twilio sends POST data as form)
    form_data = await request.form()
    query_params = dict(request.query_params)
    
    # Get session_id from query params (it's in the URL)
    session_id = query_params.get('session_id')
    
    # Get RecordingUrl from form data (Twilio sends it here)
    recording_url = form_data.get('RecordingUrl')
    call_status = form_data.get('CallStatus')
    recording_duration = form_data.get('RecordingDuration')
    
    # DEBUG LOGGING
    log(f"\n{'='*60}")
    log(f"WEBHOOK CALLED")
    log(f"Session ID (from URL): {session_id}")
    log(f"RecordingUrl (from form): {recording_url}")
    log(f"RecordingDuration: {recording_duration}")
    log(f"All form keys: {list(form_data.keys())}")
    log(f"{'='*60}")
    
    response = VoiceResponse()
    
    try:
        if not session_id or session_id not in sessions:
            log(f"ERROR: Session {session_id} not found!")
            response.say("Sorry, session expired.")
            return Response(content=str(response), media_type="application/xml")
        
        session = sessions[session_id]
        log(f"Current state: {session.state}")
        
        # Process recording if present
        user_input = ""
        if recording_url:
            session.recording_attempts = 0
            try:
                log(f"Downloading recording from: {recording_url}")
                
                # CRITICAL FIX: Add Twilio authentication to download recording
                auth = (TWILIO_SID, TWILIO_TOKEN)
                audio_response = requests.get(recording_url, auth=auth, timeout=30)
                audio_content = audio_response.content
                
                log(f"Downloaded {len(audio_content)} bytes")
                
                # Check if we got real audio (should be > 1KB for any real recording)
                if len(audio_content) < 1000:
                    log(f"ERROR: Audio file too small ({len(audio_content)} bytes) - auth failed or empty recording")
                    log(f"Response status: {audio_response.status_code}")
                    user_input = ""
                else:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_content)
                        tmp_path = tmp.name
                    
                    log(f"Saved to: {tmp_path}")
                    log("Transcribing with AssemblyAI...")
                    
                    # Transcribe
                    config = aai.TranscriptionConfig(speech_models=["universal-2"], language_code="en_us")
                    transcriber = aai.Transcriber(config=config)
                    transcript = transcriber.transcribe(tmp_path)
                    
                    # Cleanup
                    os.unlink(tmp_path)
                    
                    if transcript.text:
                        user_input = transcript.text
                        log(f"✅ USER SAID: '{user_input}'")
                    else:
                        log("⚠️ Transcription returned empty")
                        
            except Exception as e:
                log(f"❌ Transcription error: {e}")
                traceback.print_exc()
        else:
            session.recording_attempts += 1
            log(f"No recording received (attempt #{session.recording_attempts})")
        
        # Determine reply
        is_first_call = (not recording_url) and (session.state == ConversationState.GREETING)
        log(f"Is first call: {is_first_call}")
        
        if is_first_call:
            log("Generating greeting...")
            reply = get_reply(session, "")
        elif not recording_url:
            # No recording received - give clearer instructions
            if session.recording_attempts == 1:
                reply = "I didn't hear you. Please speak after the beep."
            elif session.recording_attempts == 2:
                reply = "I still didn't hear you. Please speak loudly and clearly after the beep."
            else:
                session.state = ConversationState.REJECTED
                reply = "Sorry, we will not be able to help you as we hire candidates with good communication skills only."
        elif not user_input.strip():
            reply = "I didn't catch that. Could you please speak up?"
        else:
            reply = get_reply(session, user_input)
            log(f"Reply: {reply[:60]}...")
        
        # Generate TTS
        audio_url = generate_tts(reply, session_id)
        
        if audio_url:
            response.play(audio_url)
        else:
            response.say(reply, voice="Polly.Joanna", language="en-US")
        
        # Continue or hang up
        if session.state not in [ConversationState.REJECTED, ConversationState.COMPLETED]:
            if is_first_call:
                response.pause(length=1)
                response.say("Please say yes or no after the beep.", voice="Polly.Joanna")
            else:
                response.say("Please speak now after the beep.", voice="Polly.Joanna")
            
            response.pause(length=1)
            response.record(
                action=f"{PUBLIC_URL}/twilio-webhook?session_id={session_id}",
                max_length=30,
                play_beep=True,
                trim="do-not-trim",
                timeout=15,
                finish_on_key="#"
            )
        else:
            log("Conversation ended")
            response.hangup()
        
        return Response(content=str(response), media_type="application/xml")
        
    except Exception as e:
        log(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        response.say("Sorry, an error occurred.")
        return Response(content=str(response), media_type="application/xml")

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