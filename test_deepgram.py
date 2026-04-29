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
import httpx
import sqlite3
from enum import Enum
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from collections import deque
from fastapi import FastAPI, HTTPException, Form, Request, WebSocket, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, JSONResponse
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Stream
from pydub import AudioSegment

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CAMB_API_KEY = os.getenv("CAMB_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-70b-instruct")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000")

log(f"Starting server...")
log(f"PUBLIC_URL: {PUBLIC_URL}")

TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "riya_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)
DB_PATH = Path(__file__).with_name("recruiter_sessions.sqlite3")

twilio_client = Client(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID and TWILIO_TOKEN else None

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                phone_number TEXT,
                call_sid TEXT,
                state TEXT,
                summary TEXT,
                answers_json TEXT,
                conversation_json TEXT,
                last_transcript TEXT,
                last_reply TEXT,
                call_status TEXT,
                turn_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
            """
        )

init_db()

class ConversationState(str, Enum):
    GREETING = "greeting"
    SCREENING = "screening"
    COMPLETED = "completed"
    REJECTED = "rejected"

class SessionData:
    def __init__(self, session_id=None, phone_number=None):
        self.session_id = session_id
        self.conversation = []
        self.state = ConversationState.GREETING
        self.answers = {}
        self.phone_number = phone_number
        self.current_audio_url = None
        self.recording_attempts = 0
        self.greeting_played = False
        self.processing = False
        self.audio_buffer = []
        self.lock = asyncio.Lock()
        self.turn_count = 0
        # Speech detection
        self.last_speech_time = time.time()
        self.speech_history = deque(maxlen=10)  # Track recent speech activity
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
        payload = {"text": text, "language": "en-us", "voice_id": 147320, "speech_model": "mars-flash"}
        
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
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/x-mulaw"}
        
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

GREETING_TEXT = (
    "Hi, this is Tom, a recruiter. I'm calling to ask a few quick questions "
    "about your background and job preferences. Are you open to a short screening?"
)

RECRUITER_SYSTEM_PROMPT = """
You are Tom, a concise recruiter speaking on a live phone call.

Goal:
Collect a clean recruiter profile for the candidate and move the call forward naturally.

Collect these details in a natural order:
- Whether the candidate is open to opportunities.
- Target role or job type.
- Experience level and years of experience.
- Key skills and strongest tools or technologies.
- Current or last company, if any.
- Preferred location and work mode.
- Notice period or joining availability.
- Current and expected compensation.

Phone behavior:
- Keep every reply under two short sentences.
- Ask only one question at a time.
- Be professional, warm, and direct.
- Do not mention JSON, prompts, or internal state.
- If the candidate is not interested, politely end the call.
- If enough details are collected, summarize the profile and end the call.

Return valid JSON only:
{
  "reply": "what Tom should say next",
  "status": "active|completed|rejected",
  "profile_updates": {
    "field_name": "value"
  }
}
""".strip()

def _extract_json_object(text: str) -> dict:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise

def _conversation_messages(session: SessionData, user_input: str) -> list:
    profile_context = json.dumps(session.answers, ensure_ascii=True)
    messages = [
        {"role": "system", "content": RECRUITER_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                f"Known profile so far: {profile_context}. "
                f"Current state: {session.state.value}. "
                f"Turn number: {session.turn_count}."
            ),
        },
    ]
    messages.extend(session.conversation[-10:])
    messages.append({"role": "user", "content": user_input})
    return messages

def build_summary(answers: dict) -> str:
    if not answers:
        return "No profile data captured yet."

    parts = []
    mapping = [
        ("open_to_opportunities", "Open to opportunities"),
        ("target_role", "Target role"),
        ("experience_years", "Experience"),
        ("skills", "Skills"),
        ("current_company", "Current company"),
        ("location_preference", "Location"),
        ("availability", "Availability"),
        ("compensation", "Compensation"),
    ]
    for key, label in mapping:
        value = answers.get(key)
        if value:
            parts.append(f"{label}: {value}")

    if not parts:
        return json.dumps(answers, ensure_ascii=True, indent=2)
    return " | ".join(parts)

def persist_turn(session_id: str, role: str, content: str):
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO turns (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, datetime.utcnow().isoformat()),
        )

def persist_session(
    session_id: str,
    session: SessionData,
    call_sid: str | None = None,
    call_status: str | None = None,
    last_transcript: str | None = None,
    last_reply: str | None = None,
):
    summary = build_summary(session.answers)
    now = datetime.utcnow().isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, phone_number, call_sid, state, summary, answers_json,
                conversation_json, last_transcript, last_reply, call_status,
                turn_count, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                phone_number = excluded.phone_number,
                call_sid = COALESCE(excluded.call_sid, sessions.call_sid),
                state = excluded.state,
                summary = excluded.summary,
                answers_json = excluded.answers_json,
                conversation_json = excluded.conversation_json,
                last_transcript = COALESCE(excluded.last_transcript, sessions.last_transcript),
                last_reply = COALESCE(excluded.last_reply, sessions.last_reply),
                call_status = COALESCE(excluded.call_status, sessions.call_status),
                turn_count = excluded.turn_count,
                updated_at = excluded.updated_at
            """,
            (
                session_id,
                session.phone_number,
                call_sid,
                session.state.value,
                summary,
                json.dumps(session.answers, ensure_ascii=True),
                json.dumps(session.conversation, ensure_ascii=True),
                last_transcript,
                last_reply,
                call_status,
                session.turn_count,
                now,
                now,
            ),
        )

def fetch_sessions(limit: int = 50):
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT session_id, phone_number, call_sid, state, summary, answers_json,
                   conversation_json, last_transcript, last_reply, call_status,
                   turn_count, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]

def fetch_session_detail(session_id: str):
    with get_db_connection() as conn:
        session = conn.execute(
            """
            SELECT session_id, phone_number, call_sid, state, summary, answers_json,
                   conversation_json, last_transcript, last_reply, call_status,
                   turn_count, created_at, updated_at
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        turns = conn.execute(
            """
            SELECT role, content, created_at
            FROM turns
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
    if not session:
        return None
    data = dict(session)
    data["answers"] = json.loads(data["answers_json"] or "{}")
    data["conversation"] = json.loads(data["conversation_json"] or "[]")
    data["turns"] = [dict(row) for row in turns]
    return data

async def generate_llm_reply(session: SessionData, user_input: str) -> str | None:
    if not NVIDIA_API_KEY:
        log("LLM skipped: NVIDIA_API_KEY is not configured")
        return None

    payload = {
        "model": LLM_MODEL,
        "messages": _conversation_messages(session, user_input),
        "temperature": 0.2,
        "max_tokens": 180,
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(LLM_BASE_URL, json=payload, headers=headers)

        log(f"LLM status: {response.status_code}")
        if response.status_code >= 400:
            log(f"LLM failed: {response.text[:300]}")
            return None

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
        reply = str(parsed.get("reply", "")).strip()
        status = str(parsed.get("status", "active")).lower().strip()
        updates = parsed.get("profile_updates") or {}

        if isinstance(updates, dict):
            session.answers.update({k: v for k, v in updates.items() if v not in [None, ""]})

        if status == "completed":
            session.state = ConversationState.COMPLETED
        elif status == "rejected":
            session.state = ConversationState.REJECTED
        else:
            session.state = ConversationState.SCREENING

        if not reply:
            return None

        session.conversation.append({"role": "user", "content": user_input})
        session.conversation.append({"role": "assistant", "content": reply})
        session.turn_count += 1
        if session.session_id:
            persist_turn(session.session_id, "user", user_input)
            persist_turn(session.session_id, "assistant", reply)
            persist_session(session.session_id, session, last_transcript=user_input, last_reply=reply)
        return reply
    except Exception as e:
        log(f"LLM error: {e}")
        return None

def get_fallback_reply(session: SessionData, user_input: str) -> str:
    user_lower = user_input.lower().strip()

    if session.state == ConversationState.GREETING:
        session.state = ConversationState.SCREENING
        return GREETING_TEXT

    if any(word in user_lower for word in ["no", "not", "nah", "nope", "stop"]):
        session.state = ConversationState.REJECTED
        return "I understand. Thank you for your time. Have a great day!"

    if session.state == ConversationState.REJECTED:
        return "Thank you for your time. Have a great day!"

    if "target_role" not in session.answers:
        return "What role are you targeting right now?"
    if "experience_years" not in session.answers:
        return "How many years of experience do you have?"
    if "skills" not in session.answers:
        return "What are your strongest skills or tools?"
    if "location_preference" not in session.answers:
        return "What location and work mode do you prefer?"
    if "availability" not in session.answers:
        return "What is your notice period or joining availability?"

    session.state = ConversationState.COMPLETED
    return "Thanks. I have your profile details, and a recruiter can follow up with suitable opportunities."

async def get_reply(session: SessionData, user_input: str) -> str:
    if session.state == ConversationState.GREETING:
        session.state = ConversationState.SCREENING
        session.conversation.append({"role": "assistant", "content": GREETING_TEXT})
        if session.session_id:
            persist_turn(session.session_id, "assistant", GREETING_TEXT)
            persist_session(session.session_id, session, last_reply=GREETING_TEXT)
        return GREETING_TEXT

    llm_reply = await generate_llm_reply(session, user_input)
    if llm_reply:
        return llm_reply

    fallback_reply = get_fallback_reply(session, user_input)
    session.conversation.append({"role": "user", "content": user_input})
    session.conversation.append({"role": "assistant", "content": fallback_reply})
    session.turn_count += 1
    if session.session_id:
        persist_turn(session.session_id, "user", user_input)
        persist_turn(session.session_id, "assistant", fallback_reply)
        persist_session(session.session_id, session, last_transcript=user_input, last_reply=fallback_reply)
    return fallback_reply

sessions = {}

@app.get("/")
def root():
    return dashboard_page()

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
    sessions[session_id] = SessionData(session_id=session_id, phone_number=phone_number)
    persist_session(session_id, sessions[session_id])
    log(f"New session: {session_id}")
    
    try:
        call = twilio_client.calls.create(
            to=phone_number,
            from_=TWILIO_PHONE,
            url=f"{PUBLIC_URL}/twilio-webhook?session_id={session_id}",
            status_callback=f"{PUBLIC_URL}/call-status?session_id={session_id}",
            status_callback_event=["completed", "answered"]
        )
        persist_session(session_id, sessions[session_id], call_sid=call.sid)
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
    """
    Detect if audio chunk contains actual speech or just silence/comfort noise.
    Returns True if speech detected, False if silence.
    """
    if len(audio_chunk) < 10:
        return False
    
    # For mulaw audio, analyze the variance
    # Silence in mulaw tends to be all 0xFF or similar values
    # Speech will have variation
    
    # Calculate variance of sample values
    samples = list(audio_chunk)
    
    # Check if all values are the same (silence) or vary (speech)
    unique_values = len(set(samples))
    
    # If very few unique values, it's likely silence
    if unique_values < 5:
        return False
    
    # Check amplitude variation
    # Convert mulaw to linear for better analysis
    try:
        linear = audioop.ulaw2lin(audio_chunk, 2)
        # Calculate RMS (root mean square) energy
        rms = audioop.rms(linear, 2)
        
        # Threshold for speech (adjust if needed)
        # Typical silence is < 100, speech is > 500
        return rms > 200
        
    except:
        # Fallback: just check variance
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
    
    SILENCE_THRESHOLD = 1.0  # 1 second
    MIN_AUDIO_BYTES = 4000   # ~0.5 seconds
    SPEECH_TIMEOUT = 30.0    # Max time to wait for speech
    
    try:
        # Phase 1: Wait for start
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
                    # Early audio
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
            
        # Phase 2: Send greeting
        if not session.greeting_played:
            session.state = ConversationState.GREETING
            greeting = await get_reply(session, "")
            audio_path = generate_tts(greeting, session_id)
            if audio_path:
                await send_audio_to_twilio(websocket, stream_sid, audio_path)
                session.greeting_played = True
                log("Greeting sent")
                # Reset speech detection after greeting
                session.last_speech_time = time.time()
                session.audio_buffer = []
        
        # Phase 3: Main loop with speech detection
        log("=== WAITING FOR SPEECH ===")
        log(f"Will process after {SILENCE_THRESHOLD}s of silence")
        
        last_status_log = time.time()
        last_processing_time = 0
        
        while not stop_received:
            try:
                # Receive with short timeout
                message = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                data = json.loads(message)
                
                if data["event"] == "media":
                    chunk = base64.b64decode(data["media"]["payload"])
                    
                    async with session.lock:
                        session.audio_buffer.append(chunk)
                        
                        # Check if this chunk contains speech
                        if has_speech_activity(chunk):
                            session.last_speech_time = time.time()
                            session.total_silence_duration = 0
                        else:
                            # Accumulate silence time
                            pass  # Silence time calculated below
                        
                elif data["event"] == "stop":
                    log("Stop received")
                    stop_received = True
                    break
                    
            except asyncio.TimeoutError:
                # No data received, check for silence
                pass
            
            # Calculate current silence duration
            current_time = time.time()
            silence_duration = current_time - session.last_speech_time
            
            # Status log every 2 seconds
            if current_time - last_status_log >= 2.0:
                async with session.lock:
                    buffer_bytes = sum(len(c) for c in session.audio_buffer)
                    log(f"[STATUS] Buffer: {len(session.audio_buffer)} chunks, {buffer_bytes} bytes, Silence: {silence_duration:.1f}s, Processing: {session.processing}")
                last_status_log = current_time
            
            # Check if we should process (silence detected and have enough audio)
            if not session.processing and session.audio_buffer:
                
                # Check if silence threshold reached
                if silence_duration >= SILENCE_THRESHOLD:
                    async with session.lock:
                        total_bytes = sum(len(c) for c in session.audio_buffer)
                        
                        # Debounce: don't process too frequently
                        if current_time - last_processing_time < 3.0:
                            # Reset and wait more
                            session.last_speech_time = current_time
                            continue
                        
                        if total_bytes >= MIN_AUDIO_BYTES:
                            log(f"\n>>> SILENCE DETECTED: {silence_duration:.1f}s <<<")
                            log(f">>> Processing {total_bytes} bytes <<<")
                            
                            # Take audio
                            audio_to_process = b''.join(session.audio_buffer)
                            session.audio_buffer = []
                            session.processing = True
                            last_processing_time = current_time
                        else:
                            # Not enough audio, reset
                            session.last_speech_time = current_time
                            continue
                    
                    # Process outside lock
                    try:
                        transcript = await transcribe_with_deepgram(audio_to_process)
                        
                        if transcript and transcript.strip():
                            log(f"\n🎤 USER: {transcript}")
                            
                            reply = await get_reply(session, transcript)
                            log(f"🤖 AI: {reply[:60]}...")
                            
                            audio_path = generate_tts(reply, session_id)
                            if audio_path:
                                await send_audio_to_twilio(websocket, stream_sid, audio_path)
                                
                                # Reset speech timer after AI speaks
                                session.last_speech_time = time.time()
                                
                                if session.state in [ConversationState.REJECTED, ConversationState.COMPLETED]:
                                    log("Conversation ended")
                                    await asyncio.sleep(1)
                                    stop_received = True
                                    break
                        else:
                            log("No transcript (empty)")
                            # Reset timer to continue listening
                            session.last_speech_time = time.time()
                            
                    except Exception as e:
                        log(f"Processing error: {e}")
                    finally:
                        session.processing = False
                        log("Ready for next input\n")
        
        # Cleanup
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
    call_status_value = form_data.get('CallStatus')
    log(f"Status {session_id[:8]}...: {call_status_value}")
    session = sessions.get(session_id)
    if session:
        persist_session(session_id, session, call_status=call_status_value)
    else:
        with get_db_connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET call_status = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (call_status_value, datetime.utcnow().isoformat(), session_id),
            )
    return {"status": "ok"}

@app.get("/api/sessions")
def api_sessions():
    return {"sessions": fetch_sessions()}

@app.get("/api/sessions/{session_id}")
def api_session_detail(session_id: str):
    session = fetch_session_detail(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

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

def _dashboard_page_html() -> str:
    return """<!DOCTYPE html>
<html>
<head>
    <title>Tom Voice Recruiter Dashboard</title>
    <style>
        :root {
            --bg: #f5efe6;
            --panel: #fffaf3;
            --ink: #2a1f16;
            --muted: #6b5b4d;
            --accent: #b45309;
            --accent2: #1f6f78;
            --line: #e7d8c6;
        }
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at top left, rgba(180, 83, 9, 0.12), transparent 30%),
                radial-gradient(circle at top right, rgba(31, 111, 120, 0.12), transparent 25%),
                var(--bg);
        }
        .wrap { max-width: 1320px; margin: 0 auto; padding: 28px; }
        .hero {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            justify-content: space-between;
            align-items: end;
            margin-bottom: 18px;
        }
        h1 { margin: 0; font-size: 38px; color: #8a3f0a; }
        .subtitle { margin-top: 6px; color: var(--muted); }
        .grid { display: grid; grid-template-columns: 360px 1fr; gap: 18px; }
        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 12px 32px rgba(82, 57, 31, 0.08);
        }
        .card {
            background: #fff;
            border: 1px solid #eadac6;
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 14px;
        }
        .call-form { display: grid; gap: 12px; }
        input[type="tel"] {
            width: 100%;
            box-sizing: border-box;
            padding: 14px 16px;
            border-radius: 12px;
            border: 1px solid #d7c7b2;
            background: white;
            font: inherit;
        }
        button {
            border: 0;
            border-radius: 12px;
            padding: 13px 16px;
            background: linear-gradient(135deg, var(--accent), #d97706);
            color: white;
            cursor: pointer;
            font-weight: 700;
            font: inherit;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 999px;
            background: #f3e8d6;
            font-size: 12px;
        }
        .status { color: var(--muted); font-size: 14px; min-height: 20px; }
        .summary, .answers {
            background: #fff;
            border: 1px solid #eadac6;
            border-radius: 12px;
            padding: 12px;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid #eedfd0; vertical-align: top; }
        th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }
        tr.session-row { cursor: pointer; }
        tr.session-row:hover { background: rgba(180, 83, 9, 0.05); }
        .turn {
            border-left: 3px solid #d9c1a4;
            padding: 10px 12px;
            margin: 10px 0;
            background: #fffdf9;
            border-radius: 10px;
        }
        .turn.user { border-left-color: var(--accent2); }
        .turn.assistant { border-left-color: var(--accent); }
        .turn .meta { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
        .empty { color: var(--muted); padding: 16px 0; }
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="hero">
            <div>
                <h1>Tom Voice Recruiter</h1>
                <div class="subtitle">Start a call, store every turn in SQLite, and review summaries here.</div>
            </div>
            <div class="status" id="globalStatus"></div>
        </div>

        <div class="grid">
            <div class="panel">
                <div class="card">
                    <h2 style="margin-top:0;">Start a call</h2>
                    <form id="callForm" class="call-form">
                        <input type="tel" name="phone_number" placeholder="+91XXXXXXXXXX" value="+91" required>
                        <button type="submit">Call Number</button>
                    </form>
                    <div class="status" id="callStatus"></div>
                </div>

                <div class="card">
                    <h2 style="margin-top:0;">Sessions</h2>
                    <div id="sessionList" class="empty">Loading...</div>
                </div>
            </div>

            <div class="panel">
                <div class="card">
                    <h2 style="margin-top:0;">Selected Session</h2>
                    <div id="sessionDetail" class="empty">Pick a session to view the summary and responses.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const sessionList = document.getElementById('sessionList');
        const sessionDetail = document.getElementById('sessionDetail');
        const callStatus = document.getElementById('callStatus');
        const globalStatus = document.getElementById('globalStatus');
        const callForm = document.getElementById('callForm');

        function textHtml(value) {
            const div = document.createElement('div');
            div.textContent = value ?? '';
            return div.innerHTML;
        }

        async function loadSessions() {
            const res = await fetch('/api/sessions');
            const data = await res.json();
            const sessions = data.sessions || [];
            if (!sessions.length) {
                sessionList.innerHTML = '<div class="empty">No sessions yet.</div>';
                return;
            }
            sessionList.innerHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Phone</th>
                            <th>Status</th>
                            <th>Summary</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${sessions.map(s => `
                            <tr class="session-row" data-id="${textHtml(s.session_id)}">
                                <td>${textHtml(s.phone_number || '')}</td>
                                <td><span class="badge">${textHtml(s.state || '')}</span></td>
                                <td>${textHtml((s.summary || '').slice(0, 120))}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            sessionList.querySelectorAll('.session-row').forEach(row => {
                row.addEventListener('click', () => loadSessionDetail(row.dataset.id));
            });
        }

        async function loadSessionDetail(sessionId) {
            const res = await fetch(`/api/sessions/${sessionId}`);
            if (!res.ok) {
                sessionDetail.innerHTML = '<div class="empty">Session not found.</div>';
                return;
            }
            const s = await res.json();
            const turns = (s.turns || []).map(t => `
                <div class="turn ${textHtml(t.role)}">
                    <div class="meta">${textHtml(t.role)} | ${textHtml(t.created_at || '')}</div>
                    <div>${textHtml(t.content || '')}</div>
                </div>
            `).join('');
            sessionDetail.innerHTML = `
                <div class="card" style="margin-bottom:12px;">
                    <div><strong>Phone:</strong> ${textHtml(s.phone_number || '')}</div>
                    <div><strong>Status:</strong> ${textHtml(s.state || '')}</div>
                    <div><strong>Call SID:</strong> ${textHtml(s.call_sid || '')}</div>
                    <div><strong>Created:</strong> ${textHtml(s.created_at || '')}</div>
                    <div><strong>Updated:</strong> ${textHtml(s.updated_at || '')}</div>
                </div>
                <div class="card" style="margin-bottom:12px;">
                    <h3 style="margin-top:0;">Summary</h3>
                    <div class="summary">${textHtml(s.summary || '')}</div>
                </div>
                <div class="card" style="margin-bottom:12px;">
                    <h3 style="margin-top:0;">Captured Responses</h3>
                    <div class="answers">${textHtml(JSON.stringify(s.answers || {}, null, 2))}</div>
                </div>
                <div class="card">
                    <h3 style="margin-top:0;">Conversation</h3>
                    ${turns || '<div class="empty">No turns saved yet.</div>'}
                </div>
            `;
        }

        callForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            callStatus.textContent = 'Calling...';
            const fd = new FormData(callForm);
            const res = await fetch('/initiate-call', { method: 'POST', body: fd });
            const data = await res.json();
            if (data.success) {
                callStatus.textContent = `Call started. Session ${data.session_id.slice(0, 8)}...`;
                globalStatus.textContent = `Call created for ${fd.get('phone_number')}`;
                await loadSessions();
                await loadSessionDetail(data.session_id);
            } else {
                callStatus.textContent = data.error || 'Call failed.';
            }
        });

        loadSessions().catch(err => {
            sessionList.innerHTML = '<div class="empty">Could not load sessions.</div>';
            globalStatus.textContent = err.message;
        });

        setInterval(() => {
            loadSessions().catch(() => {});
        }, 10000);
    </script>
</body>
</html>"""

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page():
    return _dashboard_page_html()

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

