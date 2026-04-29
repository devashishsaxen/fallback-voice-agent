import os
import requests
import uuid
import re
import tempfile
import io
import json
import traceback
import asyncio
import base64
import audioop
import time
from enum import Enum
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from collections import deque
from fastapi import FastAPI, HTTPException, Form, Request, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Stream, Say
from pydub import AudioSegment

# Import prompts from separate file
from prompts import ConversationFlow, IntentKeywords

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CAMB_API_KEY = os.getenv("CAMB_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000")

log(f"Starting server...")
log(f"PUBLIC_URL: {PUBLIC_URL}")

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
    FESTIVAL_STORY = "festival_story"
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
        self.audio_buffer = []
        self.lock = asyncio.Lock()
        self.last_speech_time = time.time()
        self.speech_history = deque(maxlen=10)
        self.total_silence_duration = 0

def generate_tts(text: str, session_id: str) -> str:
    log(f"TTS: {text[:50]}...")
    try:
        audio_id = f"{session_id}_{uuid.uuid4().hex[:8]}.wav"
        audio_path = TEMP_AUDIO_DIR / audio_id
        if not CAMB_API_KEY:
            log("TTS Error: CAMB_API_KEY not set")
            return None
            
        url = "https://client.camb.ai/apis/tts-stream"
        headers = {"x-api-key": CAMB_API_KEY, "Content-Type": "application/json"}
        payload = {
            "text": text, 
            "language": "en-us", 
            "voice_id": 147320, 
            "speech_model": "mars-flash"
        }
        
        log(f"TTS: Calling CAMB API...")
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        log(f"TTS: Response status {response.status_code}")
        
        if response.status_code == 200:
            mp3_data = io.BytesIO(response.content)
            audio = AudioSegment.from_mp3(mp3_data)
            audio.export(audio_path, format="wav")
            file_size = audio_path.stat().st_size
            log(f"TTS: Audio saved to {audio_path}, size: {file_size} bytes")
            return str(audio_path) if file_size > 1000 else None
        else:
            log(f"TTS: API error - {response.status_code}: {response.text[:200]}")
        return None
    except Exception as e:
        log(f"TTS Error: {e}")
        traceback.print_exc()
        return None

async def transcribe_with_deepgram(audio_data: bytes) -> str:
    if not DEEPGRAM_API_KEY:
        return ""
    
    try:
        url = (
            "https://api.deepgram.com/v1/listen?"
            "model=nova-2&encoding=mulaw&sample_rate=8000&channels=1&smart_format=true"
        )
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}", 
            "Content-Type": "audio/x-mulaw"
        }
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: requests.post(url, headers=headers, data=audio_data, timeout=10)
        )
        
        if response.status_code == 200:
            result = response.json()
            alternatives = result["results"]["channels"][0]["alternatives"]
            if alternatives:
                return alternatives[0]["transcript"]
        return ""
    except Exception as e:
        log(f"Transcription Error: {e}")
        return ""

def check_story_quality(text: str) -> bool:
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return len(sentences) >= 8 or len(text.split()) >= 50

def get_reply(session: SessionData, user_input: str) -> str:
    """Get AI reply based on current state and user input using imported prompts"""
    user_lower = user_input.lower().strip()
    
    if session.state == ConversationState.GREETING:
        session.state = ConversationState.INTEREST_CHECK
        return ConversationFlow.GREETING
    
    elif session.state == ConversationState.INTEREST_CHECK:
        if any(word in user_lower for word in IntentKeywords.NEGATIVE):
            session.state = ConversationState.REJECTED
            return ConversationFlow.INTEREST_NO
        elif any(word in user_lower for word in IntentKeywords.POSITIVE):
            session.state = ConversationState.EXPERIENCE_CHECK
            return ConversationFlow.INTEREST_YES
        else:
            return ConversationFlow.INTEREST_FALLBACK
    
    elif session.state == ConversationState.EXPERIENCE_CHECK:
        if any(word in user_lower for word in IntentKeywords.FRESHER):
            session.candidate_type = 'fresher'
            session.state = ConversationState.FRESHER_QUALIFICATION
            return ConversationFlow.FRESHER_PATH
        elif any(word in user_lower for word in IntentKeywords.EXPERIENCED):
            session.candidate_type = 'experienced'
            session.state = ConversationState.EXP_DETAILS
            return ConversationFlow.EXPERIENCED_PATH
        else:
            return ConversationFlow.EXPERIENCE_FALLBACK
    
    elif session.state == ConversationState.FRESHER_QUALIFICATION:
        session.answers['qualification'] = user_input
        session.state = ConversationState.CUSTOMER_STORY
        return ConversationFlow.CUSTOMER_STORY_PROMPT
    
    elif session.state == ConversationState.EXP_DETAILS:
        session.answers['experience'] = user_input
        session.state = ConversationState.CUSTOMER_STORY
        return ConversationFlow.CUSTOMER_STORY_PROMPT
    
    elif session.state == ConversationState.CUSTOMER_STORY:
        if check_story_quality(user_input):
            session.answers['customer_story'] = user_input
            session.state = ConversationState.FESTIVAL_STORY
            return ConversationFlow.CUSTOMER_STORY_SUCCESS
        else:
            session.retry_count += 1
            if session.retry_count >= 2:
                session.state = ConversationState.REJECTED
                return ConversationFlow.COMMUNICATION_REJECTION
            else:
                return ConversationFlow.CUSTOMER_STORY_RETRY
    
    elif session.state == ConversationState.FESTIVAL_STORY:
        if check_story_quality(user_input):
            session.answers['festival'] = user_input
            session.state = ConversationState.COMPLETED
            return ConversationFlow.FESTIVAL_STORY_SUCCESS
        else:
            session.retry_count += 1
            if session.retry_count >= 2:
                session.state = ConversationState.REJECTED
                return ConversationFlow.COMMUNICATION_REJECTION
            else:
                return ConversationFlow.FESTIVAL_STORY_RETRY
    
    elif session.state in [ConversationState.REJECTED, ConversationState.COMPLETED]:
        return ConversationFlow.FINAL_GOODBYE
    
    return ConversationFlow.REPEAT_REQUEST

sessions = {}

@app.get("/")
def root():
    return {"message": "Use /dashboard or /mic-test"}

@app.get("/audio/{audio_id}")
def get_audio(audio_id: str):
    audio_path = TEMP_AUDIO_DIR / audio_id
    if audio_path.exists():
        return FileResponse(audio_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio not found")

@app.post("/initiate-call")
async def initiate_call(phone_number: str = Form(...)):
    if not twilio_client:
        raise HTTPException(status_code=500, detail="Twilio not configured")
    
    session_id = str(uuid.uuid4())
    sessions[session_id] = SessionData(phone_number=phone_number)
    log(f"New session: {session_id}")
    
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
    query_params = dict(request.query_params)
    session_id = query_params.get('session_id')
    
    response = VoiceResponse()
    if not session_id or session_id not in sessions:
        response.say("Session error.")
        return Response(content=str(response), media_type="application/xml")
    
    parsed_url = urlparse(PUBLIC_URL)
    ws_scheme = "wss" if parsed_url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed_url.netloc}/media-stream"
    
    connect = response.connect()
    stream = Stream(url=ws_url)
    stream.parameter(name="session_id", value=session_id)
    connect.append(stream)
    
    return Response(content=str(response), media_type="application/xml")

def has_speech_activity(audio_chunk: bytes) -> bool:
    """Detect if audio chunk contains actual speech or just silence/comfort noise"""
    if len(audio_chunk) < 10:
        return False
    
    samples = list(audio_chunk)
    unique_values = len(set(samples))
    
    if unique_values < 5:
        return False
    
    try:
        linear = audioop.ulaw2lin(audio_chunk, 2)
        rms = audioop.rms(linear, 2)
        return rms > 200
    except:
        return unique_values > 20

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    log("=== STREAM CONNECTED ===")
    
    stream_sid = None
    session = None
    session_id = None
    start_received = False
    stop_received = False
    
    SILENCE_THRESHOLD = 1.0
    MIN_AUDIO_BYTES = 4000
    SPEECH_TIMEOUT = 30.0
    
    try:
        while not start_received and not stop_received:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                data = json.loads(message)
                
                if data["event"] == "connected":
                    continue
                elif data["event"] == "start":
                    stream_sid = data["start"]["streamSid"]
                    session_id = data["start"]["customParameters"].get("session_id")
                    
                    if not session_id or session_id not in sessions:
                        await websocket.close()
                        return
                    
                    session = sessions[session_id]
                    start_received = True
                    log(f"=== CALL STARTED === {stream_sid}")
                    
                elif data["event"] == "media":
                    if session:
                        chunk = base64.b64decode(data["media"]["payload"])
                        async with session.lock:
                            session.audio_buffer.append(chunk)
                            if has_speech_activity(chunk):
                                session.last_speech_time = time.time()
                                
                elif data["event"] == "stop":
                    stop_received = True
                    break
                    
            except asyncio.TimeoutError:
                log("Timeout waiting for start")
                return
        
        if not start_received or not session:
            return
            
        if not session.greeting_played:
            session.state = ConversationState.GREETING
            greeting = get_reply(session, "")
            audio_path = generate_tts(greeting, session_id)
            if audio_path:
                await send_audio_to_twilio(websocket, stream_sid, audio_path)
                session.greeting_played = True
                log("Greeting sent")
                session.last_speech_time = time.time()
                session.audio_buffer = []
        
        log("=== WAITING FOR SPEECH ===")
        last_status_log = time.time()
        last_processing_time = 0
        
        while not stop_received:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                data = json.loads(message)
                
                if data["event"] == "media":
                    chunk = base64.b64decode(data["media"]["payload"])
                    
                    async with session.lock:
                        session.audio_buffer.append(chunk)
                        if has_speech_activity(chunk):
                            session.last_speech_time = time.time()
                            
                elif data["event"] == "stop":
                    log("Stop received")
                    stop_received = True
                    break
                    
            except asyncio.TimeoutError:
                pass
            
            current_time = time.time()
            silence_duration = current_time - session.last_speech_time
            
            if current_time - last_status_log >= 2.0:
                async with session.lock:
                    buffer_bytes = sum(len(c) for c in session.audio_buffer)
                    log(f"[STATUS] Buffer: {len(session.audio_buffer)} chunks, {buffer_bytes} bytes, Silence: {silence_duration:.1f}s")
                last_status_log = current_time
            
            if not session.processing and session.audio_buffer:
                if silence_duration >= SILENCE_THRESHOLD:
                    async with session.lock:
                        total_bytes = sum(len(c) for c in session.audio_buffer)
                        
                        if current_time - last_processing_time < 3.0:
                            session.last_speech_time = current_time
                            continue
                        
                        if total_bytes >= MIN_AUDIO_BYTES:
                            log(f"\n>>> SILENCE DETECTED: {silence_duration:.1f}s <<<")
                            log(f">>> Processing {total_bytes} bytes <<<")
                            
                            audio_to_process = b''.join(session.audio_buffer)
                            session.audio_buffer = []
                            session.processing = True
                            last_processing_time = current_time
                        else:
                            session.last_speech_time = current_time
                            continue
                    
                    try:
                        transcript = await transcribe_with_deepgram(audio_to_process)
                        
                        if transcript and transcript.strip():
                            log(f"\n🎤 USER: {transcript}")
                            
                            reply = get_reply(session, transcript)
                            log(f"🤖 AI: {reply[:60]}...")
                            
                            audio_path = generate_tts(reply, session_id)
                            if audio_path:
                                await send_audio_to_twilio(websocket, stream_sid, audio_path)
                                session.last_speech_time = time.time()
                                
                                if session.state in [ConversationState.REJECTED, ConversationState.COMPLETED]:
                                    log("Conversation ended")
                                    await asyncio.sleep(1)
                                    stop_received = True
                                    break
                        else:
                            log("No transcript (empty)")
                            session.last_speech_time = time.time()
                            
                    except Exception as e:
                        log(f"Processing error: {e}")
                    finally:
                        session.processing = False
                        log("Ready for next input\n")
        
        if stop_received and session.audio_buffer and not session.processing:
            async with session.lock:
                if session.audio_buffer:
                    audio_data = b''.join(session.audio_buffer)
                    log(f"Processing final {len(audio_data)} bytes")
                    transcript = await transcribe_with_deepgram(audio_data)
                    if transcript:
                        log(f"Final: {transcript}")
        
        log("=== CALL ENDED ===")
        
    except Exception as e:
        log(f"Error: {e}")
        traceback.print_exc()
    finally:
        try:
            await websocket.close()
        except:
            pass

async def send_audio_to_twilio(websocket, stream_sid, audio_path):
    try:
        log(f"Sending audio: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        log(f"Audio loaded: {audio.frame_rate}Hz, {audio.channels}ch, {len(audio)}ms")
        audio = audio.set_frame_rate(8000).set_channels(1)
        raw_data = audio.raw_data
        mulaw_data = audioop.lin2ulaw(raw_data, audio.sample_width)
        log(f"Converted to mulaw: {len(mulaw_data)} bytes")
        
        chunk_count = 0
        for i in range(0, len(mulaw_data), 160):
            chunk = mulaw_data[i:i+160]
            payload = base64.b64encode(chunk).decode('utf-8')
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload}
            })
            chunk_count += 1
        
        await websocket.send_json({
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {"name": "end"}
        })
        log(f"Audio sent: {chunk_count} chunks")
        
    except Exception as e:
        log(f"Send error: {e}")
        traceback.print_exc()

@app.post("/call-status")
async def call_status(request: Request):
    form_data = await request.form()
    session_id = dict(request.query_params).get('session_id')
    log(f"Status {session_id[:8]}...: {form_data.get('CallStatus')}")
    return {"status": "ok"}

@app.get("/mic-test", response_class=HTMLResponse)
def mic_test():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Mic Test</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 50px auto; text-align: center; }
        button { width: 100px; height: 100px; border-radius: 50%; font-size: 40px; border: none; cursor: pointer; margin: 20px; }
        .rec { background: #e74c3c; color: white; }
        .stop { background: #2ecc71; }
        #result { margin-top: 20px; padding: 20px; background: #f0f0f0; border-radius: 10px; }
    </style>
</head>
<body>
    <h1>🎤 Mic Test</h1>
    <button id="btn" class="rec">🎤</button>
    <div id="status">Click to start</div>
    <div id="result"></div>
    
    <script>
        let rec, chunks = [], recording = false;
        const btn = document.getElementById('btn');
        const status = document.getElementById('status');
        const result = document.getElementById('result');
        
        btn.onclick = async () => {
            if (!recording) {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                rec = new MediaRecorder(stream);
                chunks = [];
                rec.ondataavailable = e => chunks.push(e.data);
                rec.onstop = async () => {
                    const blob = new Blob(chunks, { type: 'audio/webm' });
                    const fd = new FormData();
                    fd.append('audio', blob);
                    status.textContent = 'Processing...';
                    const r = await fetch('/transcribe', { method: 'POST', body: fd });
                    const d = await r.json();
                    result.innerHTML = '<strong>Transcript:</strong> ' + (d.transcript || d.error || 'No speech');
                    status.textContent = 'Done';
                };
                rec.start();
                recording = true;
                btn.className = 'stop';
                btn.textContent = '⏹';
                status.textContent = 'Recording...';
            } else {
                rec.stop();
                recording = false;
                btn.className = 'rec';
                btn.textContent = '🎤';
                status.textContent = 'Processing...';
            }
        };
    </script>
</body>
</html>"""

@app.post("/transcribe")
async def transcribe_endpoint(audio: UploadFile = File(...)):
    try:
        content = await audio.read()
        audio_seg = AudioSegment.from_file(io.BytesIO(content))
        audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
        buffer = io.BytesIO()
        audio_seg.export(buffer, format="wav")
        transcript = await transcribe_with_deepgram(buffer.getvalue())
        return {"transcript": transcript or "No speech detected"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Riya Voice Agent</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; background: #f0f2f5; }
        .container { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #1a73e8; text-align: center; }
        input { width: 100%; padding: 15px; font-size: 18px; border: 2px solid #ddd; border-radius: 10px; margin: 20px 0; }
        button { width: 100%; padding: 15px; background: #1a73e8; color: white; border: none; border-radius: 10px; font-size: 18px; cursor: pointer; }
        button:hover { background: #1557b0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Riya Voice Recruiter</h1>
        <form action="/initiate-call" method="post" enctype="multipart/form-data">
            <input type="tel" name="phone_number" placeholder="+91XXXXXXXXXX" value="+91" required>
            <button type="submit">📞 Make Call</button>
        </form>
        <p style="text-align: center; margin-top: 20px;"><a href="/mic-test">🎤 Test Microphone</a></p>
    </div>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)