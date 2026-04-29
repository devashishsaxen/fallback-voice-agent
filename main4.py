import os
import json
import asyncio
import websockets
import requests
import uuid
import re
from enum import Enum
from datetime import datetime
from pathlib import Path
import assemblyai as aai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from dotenv import load_dotenv

load_dotenv()

# Config
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE_NUMBER")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
PUBLIC_URL = os.getenv("PUBLIC_URL")

aai.settings.api_key = ASSEMBLYAI_API_KEY
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID else None

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
    def __init__(self):
        self.state = ConversationState.GREETING
        self.candidate_type = None
        self.retry_count = 0
        self.answers = {}

# Store sessions
sessions = {}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

def get_reply(session: SessionData, user_input: str) -> str:
    user_lower = user_input.lower().strip()
    
    if session.state == ConversationState.GREETING:
        session.state = ConversationState.INTEREST_CHECK
        return "Hi, this is Riya from Futuresoft Consultancy. We are hiring voice and chat profiles for companies like British Telecom, Teleperformance, and Wipro. Are you interested?"
    
    elif session.state == ConversationState.INTEREST_CHECK:
        if any(word in user_lower for word in ['no', 'not', 'nah', 'nope']):
            session.state = ConversationState.REJECTED
            return "I understand. Thank you for your time. Have a great day!"
        elif any(word in user_lower for word in ['yes', 'yeah', 'sure', 'interested', 'ok']):
            session.state = ConversationState.EXPERIENCE_CHECK
            return "We will surely help you with the same. Could you please confirm me if you are Fresher OR Experienced?"
        else:
            return "Could you please confirm if you are interested? Just say Yes or No."
    
    elif session.state == ConversationState.EXPERIENCE_CHECK:
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
    
    elif session.state == ConversationState.FRESHER_QUALIFICATION:
        session.answers['qualification'] = user_input
        session.state = ConversationState.CUSTOMER_STORY
        return "That was very impressive. Could you please speak about any memorable interaction with customer within 10 to 12 sentences. You can start with, 'Once a customer called me for issue related to...' And your time starts now."
    
    elif session.state == ConversationState.EXP_DETAILS:
        session.answers['experience'] = user_input
        session.state = ConversationState.CUSTOMER_STORY
        return "That was very impressive. Could you please speak about any memorable interaction with customer within 10 to 12 sentences. You can start with, 'Once a customer called me for issue related to...' And your time starts now."
    
    elif session.state == ConversationState.CUSTOMER_STORY:
        sentences = re.split(r'[.!?]+', user_input)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 8 or len(user_input.split()) >= 50:
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
    
    elif session.state == ConversationState.CUSTOMER_RETRY:
        sentences = re.split(r'[.!?]+', user_input)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 8 or len(user_input.split()) >= 50:
            session.answers['customer_story'] = user_input
            session.state = ConversationState.FESTIVAL_STORY
            return "Acknowledgment to statement. Could you please speak about any latest festival you celebrated like Diwali, Holi, Christmas or Eid in 10 to 12 sentences. Start with, 'I celebrated my last Diwali along with family...' And your time starts now."
        else:
            session.state = ConversationState.REJECTED
            return "Sorry, we will not be able to help you with job as we hire candidates with good communication skills only."
    
    elif session.state == ConversationState.FESTIVAL_STORY:
        sentences = re.split(r'[.!?]+', user_input)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 8 or len(user_input.split()) >= 50:
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
    
    elif session.state == ConversationState.FESTIVAL_RETRY:
        sentences = re.split(r'[.!?]+', user_input)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 8 or len(user_input.split()) >= 50:
            session.answers['festival'] = user_input
            session.state = ConversationState.COMPLETED
            return "That was amazing, now one of our HR Recruiter will connect you for your further interview process."
        else:
            session.state = ConversationState.REJECTED
            return "Sorry, we will not be able to help you with job as we hire candidates with good communication skills only."
    
    elif session.state in [ConversationState.REJECTED, ConversationState.COMPLETED]:
        return "Thank you for your time. Have a great day!"
    
    return "I'm sorry, could you please repeat that?"

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
        method="POST"
    )
    
    return {"success": True, "call_sid": call.sid, "session_id": session_id}

@app.post("/twilio-stream")
async def twilio_stream(request: Request):
    """Return TwiML to start Media Stream"""
    form_data = await request.form()
    query_params = dict(request.query_params)
    session_id = form_data.get('session_id') or query_params.get('session_id')
    
    log(f"TwiML requested for session: {session_id}")
    
    response = VoiceResponse()
    
    if session_id and session_id in sessions:
        reply = get_reply(sessions[session_id], "")
        response.say(reply, voice="Polly.Joanna", language="en-US")
        
        # Format WebSocket URL
        ws_url = PUBLIC_URL.replace("https://", "wss://").replace("http://", "wss://")
        stream_url = f"{ws_url}/media-stream?session_id={session_id}"
        
        log(f"Connecting stream to: {stream_url}")
        
        connect = Connect()
        connect.stream(url=stream_url)
        response.append(connect)
    else:
        log(f"ERROR: Session {session_id} not found")
        response.say("Sorry, session error.")
        response.hangup()
    
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket, session_id: str = None):
    """WebSocket for real-time audio streaming with AssemblyAI"""
    await websocket.accept()
    log(f"Media stream started for session: {session_id}")
    
    if not session_id or session_id not in sessions:
        log(f"ERROR: Invalid session_id: {session_id}")
        await websocket.close()
        return
    
    session = sessions[session_id]
    assembly_ws = None
    is_active = True
    
    try:
        # Connect to AssemblyAI
        assembly_ws = await websockets.connect(
            "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=8000",
            extra_headers={"Authorization": ASSEMBLYAI_API_KEY},
            ping_interval=20,
            ping_timeout=10
        )
        log("Connected to AssemblyAI")
        
        # CRITICAL FIX: 1500ms silence threshold (1.5 seconds)
        await assembly_ws.send(json.dumps({
            "audio_data": "",
            "end_utterance_silence_threshold": 1500  # Changed to 1.5 seconds
        }))
        log("AssemblyAI configured with 1500ms silence threshold")
        
        async def receive_from_twilio():
            """Receive audio from Twilio and send to AssemblyAI"""
            nonlocal is_active
            try:
                while is_active:
                    # 30 second timeout to prevent hanging
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    data = json.loads(message)
                    
                    if data["event"] == "media":
                        # Forward audio to AssemblyAI
                        await assembly_ws.send(json.dumps({
                            "audio_data": data["media"]["payload"]
                        }))
                    elif data["event"] == "stop":
                        log("Twilio stop event received")
                        is_active = False
                        break
                    elif data["event"] == "mark":
                        log(f"Mark event: {data.get('mark', {})}")
                        
            except asyncio.TimeoutError:
                log("Twilio receive timeout (30s)")
                is_active = False
            except WebSocketDisconnect:
                log("Twilio disconnected")
                is_active = False
            except Exception as e:
                log(f"Twilio error: {e}")
                is_active = False
        
        async def send_to_twilio():
            """Receive transcription from AssemblyAI and respond"""
            nonlocal is_active
            try:
                while is_active:
                    # 30 second timeout
                    message = await asyncio.wait_for(assembly_ws.recv(), timeout=30.0)
                    result = json.loads(message)
                    
                    if result.get("message_type") == "FinalTranscript":
                        transcript = result.get("text", "")
                        log(f"USER SAID: {transcript}")
                        
                        if transcript.strip():
                            # Get Riya's response
                            reply = get_reply(session, transcript)
                            log(f"RIYA REPLIES: {reply[:60]}...")
                            
                            # Update the live call
                            try:
                                twiml_response = VoiceResponse()
                                twiml_response.say(reply, voice="Polly.Joanna", language="en-US")
                                
                                # Reconnect stream for next input
                                connect = Connect()
                                ws_url = PUBLIC_URL.replace("https://", "wss://").replace("http://", "wss://")
                                connect.stream(url=f"{ws_url}/media-stream?session_id={session_id}")
                                twiml_response.append(connect)
                                
                                # Update call via Twilio API
                                twilio_client.calls(session_id).update(twiml=str(twiml_response))
                                log("Call updated successfully")
                                
                                # Don't close - let the new stream handle it
                                is_active = False
                                
                            except Exception as e:
                                log(f"Error updating call: {e}")
                                is_active = False
                                
                    elif result.get("message_type") == "PartialTranscript":
                        partial = result.get("text", "")
                        if partial:
                            log(f"Partial transcript: {partial}")
                            
            except asyncio.TimeoutError:
                log("AssemblyAI timeout (30s)")
            except Exception as e:
                log(f"AssemblyAI error: {e}")
            finally:
                is_active = False
        
        # Run both tasks
        await asyncio.gather(receive_from_twilio(), send_to_twilio())
        
    except Exception as e:
        log(f"Stream error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if assembly_ws:
            await assembly_ws.close()
        log(f"Media stream ended for {session_id}")

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Riya Voice Agent - Real-time</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; background: #f0f2f5; }
            .container { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #1a73e8; text-align: center; }
            input { width: 100%; padding: 15px; font-size: 18px; border: 2px solid #ddd; border-radius: 10px; margin: 20px 0; box-sizing: border-box; }
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
            <h1>🎙️ Riya Voice Recruiter<br><small style="color: green;">REAL-TIME MODE (1.5s silence detection)</small></h1>
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
                        showStatus(`✅ Call initiated!<br><br>📞 <strong>Answer your phone!</strong><br>🎤 <strong>Speak after the beep!</strong><br><br><small>System detects when you stop talking (1.5s pause)</small>`, 'success');
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