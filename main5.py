import os
import json
import asyncio
import websockets
import uuid
import re
from enum import Enum
from datetime import datetime
import assemblyai as aai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
TWILIO_SID      = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN    = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE    = os.getenv("TWILIO_PHONE_NUMBER")
ASSEMBLYAI_KEY  = os.getenv("ASSEMBLYAI_API_KEY")
PUBLIC_URL      = os.getenv("PUBLIC_URL", "").rstrip("/")

aai.settings.api_key = ASSEMBLYAI_KEY
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID else None

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ─── State Machine ────────────────────────────────────────────────────────────
class ConversationState(str, Enum):
    GREETING            = "greeting"
    INTEREST_CHECK      = "interest_check"
    EXPERIENCE_CHECK    = "experience_check"
    FRESHER_QUALIFICATION = "fresher_qualification"
    EXP_DETAILS         = "exp_details"
    CUSTOMER_STORY      = "customer_story"
    CUSTOMER_RETRY      = "customer_retry"
    FESTIVAL_STORY      = "festival_story"
    FESTIVAL_RETRY      = "festival_retry"
    COMPLETED           = "completed"
    REJECTED            = "rejected"


class SessionData:
    def __init__(self):
        self.state          = ConversationState.GREETING
        self.candidate_type = None
        self.customer_retries = 0   # FIX: separate retry counters
        self.festival_retries = 0
        self.answers        = {}
        self.call_sid       = None  # FIX: store Twilio call SID


# ─── In-memory store ──────────────────────────────────────────────────────────
sessions: dict[str, SessionData] = {}


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}", flush=True)


# ─── Sentence / word counter helper ──────────────────────────────────────────
def _long_enough(text: str) -> bool:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(sentences) >= 8 or len(text.split()) >= 50


# ─── Conversation logic ───────────────────────────────────────────────────────
def get_reply(session: SessionData, user_input: str) -> str:
    user_lower = user_input.lower().strip()

    if session.state == ConversationState.GREETING:
        session.state = ConversationState.INTEREST_CHECK
        return (
            "Hi, this is Riya from Futuresoft Consultancy. "
            "We are hiring voice and chat profiles for companies like "
            "British Telecom, Teleperformance, and Wipro. Are you interested?"
        )

    elif session.state == ConversationState.INTEREST_CHECK:
        if any(w in user_lower for w in ["no", "not", "nah", "nope"]):
            session.state = ConversationState.REJECTED
            return "I understand. Thank you for your time. Have a great day!"
        elif any(w in user_lower for w in ["yes", "yeah", "sure", "interested", "ok", "okay"]):
            session.state = ConversationState.EXPERIENCE_CHECK
            return "We will surely help you with the same. Could you please confirm if you are a Fresher or Experienced?"
        else:
            return "Could you please confirm if you are interested? Just say Yes or No."

    elif session.state == ConversationState.EXPERIENCE_CHECK:
        if any(w in user_lower for w in ["fresher", "fresh", "student", "no experience"]):
            session.candidate_type = "fresher"
            session.state = ConversationState.FRESHER_QUALIFICATION
            return "Now, what is your highest qualification — Graduate, Undergraduate, or Graduation drop-out?"
        elif any(w in user_lower for w in ["experience", "experienced", "worked", "working", "years"]):
            session.candidate_type = "experienced"
            session.state = ConversationState.EXP_DETAILS
            return (
                "Please confirm your highest qualification and total experience. "
                "Mention your job responsibilities clearly."
            )
        else:
            return "Could you please clarify — are you a Fresher or do you have Experience?"

    elif session.state == ConversationState.FRESHER_QUALIFICATION:
        session.answers["qualification"] = user_input
        session.state = ConversationState.CUSTOMER_STORY
        return (
            "That was very impressive. Could you please speak about any memorable interaction "
            "with a customer in 10 to 12 sentences? You can start with: "
            "'Once a customer called me for an issue related to...' Your time starts now."
        )

    elif session.state == ConversationState.EXP_DETAILS:
        session.answers["experience"] = user_input
        session.state = ConversationState.CUSTOMER_STORY
        return (
            "That was very impressive. Could you please speak about any memorable interaction "
            "with a customer in 10 to 12 sentences? You can start with: "
            "'Once a customer called me for an issue related to...' Your time starts now."
        )

    elif session.state == ConversationState.CUSTOMER_STORY:
        if _long_enough(user_input):
            session.answers["customer_story"] = user_input
            session.state = ConversationState.FESTIVAL_STORY
            return (
                "Wonderful! Could you now speak about a latest festival you celebrated — "
                "like Diwali, Holi, Christmas, or Eid — in 10 to 12 sentences? "
                "Start with: 'I celebrated my last Diwali along with my family...' Your time starts now."
            )
        else:
            session.customer_retries += 1
            if session.customer_retries >= 2:
                session.state = ConversationState.REJECTED
                return "Sorry, we will not be able to help you as we hire candidates with good communication skills only."
            session.state = ConversationState.CUSTOMER_RETRY
            return (
                "Sorry, you need to speak 10 to 12 sentences on this topic. "
                "Please try again now."
            )

    elif session.state == ConversationState.CUSTOMER_RETRY:
        if _long_enough(user_input):
            session.answers["customer_story"] = user_input
            session.state = ConversationState.FESTIVAL_STORY
            return (
                "Wonderful! Could you now speak about a latest festival you celebrated — "
                "like Diwali, Holi, Christmas, or Eid — in 10 to 12 sentences? "
                "Start with: 'I celebrated my last Diwali along with my family...' Your time starts now."
            )
        else:
            session.state = ConversationState.REJECTED
            return "Sorry, we will not be able to help you as we hire candidates with good communication skills only."

    elif session.state == ConversationState.FESTIVAL_STORY:
        if _long_enough(user_input):
            session.answers["festival"] = user_input
            session.state = ConversationState.COMPLETED
            return "That was amazing! One of our HR Recruiters will connect with you for your further interview process."
        else:
            session.festival_retries += 1
            if session.festival_retries >= 2:
                session.state = ConversationState.REJECTED
                return "Sorry, we will not be able to help you as we hire candidates with good communication skills only."
            session.state = ConversationState.FESTIVAL_RETRY
            return (
                "Sorry, please speak clearly about the festival celebration for 10 to 12 sentences to proceed."
            )

    elif session.state == ConversationState.FESTIVAL_RETRY:
        if _long_enough(user_input):
            session.answers["festival"] = user_input
            session.state = ConversationState.COMPLETED
            return "That was amazing! One of our HR Recruiters will connect with you for your further interview process."
        else:
            session.state = ConversationState.REJECTED
            return "Sorry, we will not be able to help you as we hire candidates with good communication skills only."

    elif session.state in (ConversationState.REJECTED, ConversationState.COMPLETED):
        return "Thank you for your time. Have a great day!"

    return "I'm sorry, could you please repeat that?"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _ws_base() -> str:
    """Convert PUBLIC_URL to wss:// base."""
    return PUBLIC_URL.replace("https://", "wss://").replace("http://", "ws://")


def _build_twiml(reply: str, session_id: str, reconnect: bool = True) -> str:
    """Build TwiML that speaks the reply.
    If reconnect=True, opens a new media stream for the next user turn.
    If reconnect=False (terminal states), just says the reply and hangs up."""
    response = VoiceResponse()
    response.say(reply, voice="Polly.Matthew", language="en-US")
    if reconnect:
        connect = Connect()
        connect.stream(url=f"{_ws_base()}/media-stream?session_id={session_id}")
        response.append(connect)
    else:
        response.hangup()
    return str(response)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.post("/initiate-call")
async def initiate_call(phone_number: str = Form(...)):
    if not twilio_client:
        raise HTTPException(status_code=500, detail="Twilio not configured")

    session_id = str(uuid.uuid4())
    sessions[session_id] = SessionData()

    call = twilio_client.calls.create(
        to=phone_number,
        from_=TWILIO_PHONE,
        url=f"{PUBLIC_URL}/twilio-stream?session_id={session_id}",
        method="POST",
    )

    sessions[session_id].call_sid = call.sid
    log(f"Call created: session={session_id}  call_sid={call.sid}")
    return {"success": True, "call_sid": call.sid, "session_id": session_id}


@app.post("/twilio-stream")
async def twilio_stream(request: Request):
    """Return TwiML that speaks the greeting and opens a Media Stream."""
    form_data   = await request.form()
    query_params = dict(request.query_params)

    session_id = form_data.get("session_id") or query_params.get("session_id")
    twilio_call_sid = form_data.get("CallSid")  # FIX: capture call SID from Twilio POST
    log(f"TwiML requested — session={session_id}  twilio_call_sid={twilio_call_sid}")

    if not session_id or session_id not in sessions:
        log("ERROR: session not found")
        r = VoiceResponse()
        r.say("Sorry, session error.")
        r.hangup()
        return HTMLResponse(content=str(r), media_type="application/xml")

    session = sessions[session_id]

    # Store real call SID if we didn't have it yet (initiate_call runs first,
    # but this is a safety net for cases like manual testing)
    if twilio_call_sid and not session.call_sid:
        session.call_sid = twilio_call_sid

    greeting = get_reply(session, "")   # triggers GREETING → INTEREST_CHECK
    twiml    = _build_twiml(greeting, session_id)
    log(f"Sending TwiML greeting for session {session_id}")
    return HTMLResponse(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Receive Twilio audio, stream to AssemblyAI, respond via call update."""
    await websocket.accept()

    query_params = dict(websocket.query_params)
    session_id   = query_params.get("session_id")
    log(f"Media stream opened — session={session_id}")

    if not session_id or session_id not in sessions:
        log(f"ERROR: unknown session_id={session_id}")
        await websocket.close()
        return

    session     = sessions[session_id]
    assembly_ws = None

    # ── Connect to AssemblyAI ────────────────────────────────────────────────
    aai_url = (
        "wss://api.assemblyai.com/v2/realtime/ws"
        "?sample_rate=8000&end_utterance_silence_threshold=1500"
    )
    try:
        assembly_ws = await websockets.connect(
            aai_url,
            extra_headers={"Authorization": ASSEMBLYAI_KEY},
            ping_interval=20,
            ping_timeout=10,
        )
        log("Connected to AssemblyAI (silence=1500ms)")
    except Exception as exc:
        log(f"FATAL: Cannot connect to AssemblyAI — {exc}")
        # Keep Twilio call alive even though STT is broken
        try:
            while True:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=120.0)
                if json.loads(raw).get("event") in ("stop",):
                    break
        except Exception:
            pass
        return

    # Flags shared between tasks
    replied   = False   # True once calls.update() succeeds → stop accepting more transcripts
    stop_evt  = asyncio.Event()  # Set when Twilio sends 'stop'

    # ── Task 1: Twilio audio → AssemblyAI ────────────────────────────────────
    async def receive_from_twilio():
        try:
            while True:
                raw   = await asyncio.wait_for(websocket.receive_text(), timeout=120.0)
                data  = json.loads(raw)
                event = data.get("event")

                if event == "media":
                    if not replied:
                        # FIX: use try/except instead of assembly_ws.open
                        # (.open does not exist in websockets >= 10.0)
                        try:
                            await assembly_ws.send(
                                json.dumps({"audio_data": data["media"]["payload"]})
                            )
                        except Exception:
                            pass  # AssemblyAI WS closed — ignore, keep reading Twilio

                elif event == "stop":
                    log("Twilio stop event received")
                    stop_evt.set()
                    break

                elif event == "connected":
                    log("Twilio stream connected")
                elif event == "start":
                    log(f"Twilio stream started: {data.get('start', {})}")

        except asyncio.TimeoutError:
            log("Twilio 120s silence timeout")
            stop_evt.set()
        except WebSocketDisconnect:
            log("Twilio WebSocket disconnected")
            stop_evt.set()
        except Exception as exc:
            log(f"receive_from_twilio error: {exc}")
            stop_evt.set()

    # ── Task 2: AssemblyAI transcripts → Twilio call update ──────────────────
    async def send_to_twilio():
        nonlocal replied
        try:
            while not stop_evt.is_set():
                try:
                    raw = await asyncio.wait_for(assembly_ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Short poll so we can check stop_evt regularly
                    continue

                result   = json.loads(raw)
                msg_type = result.get("message_type")

                if msg_type == "FinalTranscript":
                    transcript = result.get("text", "").strip()
                    log(f"USER: {transcript}")

                    # Guard: ignore empty transcripts or if we already replied this turn
                    if not transcript or replied:
                        continue

                    reply    = get_reply(session, transcript)
                    call_sid = session.call_sid
                    log(f"RIYA: {reply[:100]}")

                    if not call_sid:
                        log("ERROR: no call_sid on session")
                        continue

                    # Determine if this is a terminal state
                    terminal = session.state in (
                        ConversationState.COMPLETED,
                        ConversationState.REJECTED,
                    )

                    twiml = _build_twiml(reply, session_id, reconnect=not terminal)

                    # FIX: run blocking Twilio HTTP call in a thread so we don't
                    # freeze the asyncio event loop (which would prevent 'stop'
                    # from being processed and cause the call to drop)
                    try:
                        await asyncio.to_thread(
                            twilio_client.calls(call_sid).update, twiml=twiml
                        )
                        replied = True
                        log(f"Call {call_sid} updated OK — terminal={terminal}")
                        # Twilio will send 'stop' to this WebSocket once it processes
                        # the redirect.  receive_from_twilio will set stop_evt then.
                    except Exception as exc:
                        log(f"Call update failed: {exc}")

                elif msg_type == "PartialTranscript":
                    partial = result.get("text", "")
                    if partial:
                        log(f"  partial: {partial}")

                elif msg_type == "SessionBegins":
                    log(f"AssemblyAI session: {result.get('session_id')}")

                elif msg_type == "SessionTerminated":
                    log("AssemblyAI session terminated")
                    break

                elif result.get("error"):
                    log(f"AssemblyAI error: {result['error']}")
                    break

        except Exception as exc:
            log(f"send_to_twilio error: {exc}")

    # Run both tasks; use create_task so we can cancel properly
    t1 = asyncio.create_task(receive_from_twilio(), name="recv_twilio")
    t2 = asyncio.create_task(send_to_twilio(),      name="send_twilio")

    try:
        # Wait for the stop event (Twilio ended the stream)
        await stop_evt.wait()
    finally:
        # Cancel whichever task is still running
        for t in (t1, t2):
            if not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

        if assembly_ws:
            try:
                await assembly_ws.close()
            except Exception:
                pass
        log(f"Media stream closed — session={session_id}")


# ─── Dashboard ────────────────────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Riya Voice Recruiter</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 60px auto; padding: 20px; }
    h2   { color: #333; }
    input, button { padding: 10px 16px; font-size: 15px; margin: 6px 0; border-radius: 6px; }
    input  { border: 1px solid #ccc; width: 280px; }
    button { background: #4f46e5; color: #fff; border: none; cursor: pointer; }
    button:hover { background: #4338ca; }
    #status { margin-top: 16px; color: #555; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h2>🎙️ Riya — Voice Recruiter</h2>
  <p>Enter a phone number in E.164 format (e.g. +919876543210)</p>
  <input id="phone" type="tel" placeholder="+91XXXXXXXXXX" />
  <br>
  <button onclick="makeCall()">📞 Make Call</button>
  <div id="status"></div>

  <script>
    async function makeCall() {
      const phone  = document.getElementById('phone').value.trim();
      const status = document.getElementById('status');
      if (!phone) { status.textContent = 'Please enter a phone number.'; return; }
      status.textContent = 'Initiating call…';
      try {
        const fd = new FormData();
        fd.append('phone_number', phone);
        const res  = await fetch('/initiate-call', { method: 'POST', body: fd });
        const data = await res.json();
        status.textContent = data.success
          ? `✅ Call started!\nCall SID:    ${data.call_sid}\nSession ID: ${data.session_id}`
          : `❌ Error: ${JSON.stringify(data)}`;
      } catch (e) {
        status.textContent = `❌ Request failed: ${e}`;
      }
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)