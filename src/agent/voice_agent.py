"""
Voice agent for CareProxy - handles real-time voice interactions with caregivers.

This module implements a LiveKit voice agent using the LiveKit Agents framework
with OpenAI integration for speech-to-text, text-to-speech, and conversation.

Run with: python src/agent/voice_agent.py dev
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from phoenix.otel import register

from triage import assess_conversation
from reports import generate_caregiver_report, generate_physician_report
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai, silero

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Load environment variables from .env file
# Expected variables: OPENAI_API_KEY, LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
load_dotenv()

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("careproxy")

# -----------------------------------------------------------------------------
# Phoenix Observability Setup
# -----------------------------------------------------------------------------
# Initialize Phoenix tracing to track all OpenAI calls
# View traces at http://localhost:6006
tracer_provider = register(
    project_name="careproxy",
    endpoint="http://localhost:6006/v1/traces",
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
logger.info("Phoenix tracing initialized - view at http://localhost:6006")

# Get tracer for custom spans
tracer = trace.get_tracer("careproxy")

# System prompt defining the agent's personality and behavior
INSTRUCTIONS = """You are CareProxy, a warm and caring healthcare navigation assistant.

Your role is to help people understand when and where to seek medical care - never diagnose, just guide.

CONVERSATION STYLE:
- Speak naturally and warmly, like a caring nurse
- Ask ONE clear question at a time
- Listen to their full answer before moving on
- Use their own words (if they say "dizzy spells", use "dizzy spells")
- Show empathy: "I understand that must be concerning"
- Be patient and reassuring

GATHERING INFORMATION:
When someone mentions a health concern, gently ask about:
- How severe it feels to them
- When it started
- Any other symptoms they've noticed
- Their relevant health history

After gathering enough information, provide clear guidance:
"Based on what you've shared, I recommend [action]. Here's why: [brief explanation]."

Then: "I'll create a summary for you and your doctor."

Remember: You're providing comfort and guidance, not just collecting data.

Start the conversation with: "Hello! I'm CareProxy. I'm here to help you navigate your health concerns. What's been going on?\""""


# -----------------------------------------------------------------------------
# Conversation Data Persistence
# -----------------------------------------------------------------------------


def save_conversation_data(transcript: str) -> dict:
    """
    Save conversation with triage assessment and reports.

    Creates two files:
    - conversations/latest.json: Most recent conversation (overwritten each time)
    - conversations/history.json: Last 10 conversations

    Args:
        transcript: The full conversation transcript

    Returns:
        dict: The saved conversation data
    """
    logger.info("Saving conversation data...")

    # Create conversations directory if it doesn't exist
    Path("conversations").mkdir(exist_ok=True)

    # Get triage assessment
    logger.info("Running triage assessment...")
    triage_data = assess_conversation(transcript)

    # Generate both reports
    logger.info("Generating reports...")
    caregiver_report = generate_caregiver_report(triage_data, transcript)
    physician_report = generate_physician_report(triage_data, transcript)

    # Create data structure
    conversation_data = {
        "timestamp": datetime.now().isoformat(),
        "transcript": transcript,
        "triage": triage_data,
        "caregiver_report": caregiver_report,
        "physician_report": physician_report,
    }

    # Save as latest.json (always overwrite)
    with open("conversations/latest.json", "w") as f:
        json.dump(conversation_data, f, indent=2)

    # Also append to history
    history_file = Path("conversations/history.json")
    try:
        with open(history_file, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    history.append(conversation_data)
    # Keep only last 10
    history = history[-10:]

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Conversation saved to conversations/latest.json")
    print(f"\n✅ Conversation saved to conversations/latest.json")
    print(f"   Urgency: {triage_data.get('urgency_emoji', '')} {triage_data.get('urgency_level', 'unknown').upper()}")

    return conversation_data


# -----------------------------------------------------------------------------
# Agent Setup
# -----------------------------------------------------------------------------


def prewarm_process(proc: JobProcess):
    """
    Prewarm function called when the worker process starts.

    This loads the VAD (Voice Activity Detection) model ahead of time
    so it's ready when a user connects, reducing latency.
    """
    logger.info("Prewarming: Loading Silero VAD model...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarming complete")


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for handling a LiveKit room connection.

    This function is called when a participant joins the room.
    It sets up the voice agent and starts the conversation.

    Args:
        ctx: JobContext containing room and participant information
    """
    logger.info(f"Agent connecting to room: {ctx.room.name}")

    # -------------------------------------------------------------------------
    # Step 1: Connect to the LiveKit room
    # -------------------------------------------------------------------------
    # AutoSubscribe.AUDIO_ONLY means we only subscribe to audio tracks,
    # not video, which is all we need for a voice agent
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info("Connected to room")

    # -------------------------------------------------------------------------
    # Step 2: Wait for a participant to join
    # -------------------------------------------------------------------------
    # We wait for the first participant that isn't the agent itself
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # -------------------------------------------------------------------------
    # Step 3: Create the Voice Agent
    # -------------------------------------------------------------------------
    # The Agent class orchestrates the full voice interaction:
    # - VAD (Voice Activity Detection): Detects when user is speaking
    # - STT (Speech-to-Text): Converts user's speech to text
    # - LLM (Language Model): Generates responses
    # - TTS (Text-to-Speech): Converts responses back to speech

    agent = Agent(
        instructions=INSTRUCTIONS,
        # Voice Activity Detection - uses prewarmed Silero model
        vad=ctx.proc.userdata["vad"],
        # Speech-to-Text - OpenAI Whisper
        stt=openai.STT(model="whisper-1"),
        # Language Model - GPT-4o for conversation
        llm=openai.LLM(model="gpt-4o"),
        # Text-to-Speech - OpenAI TTS with a friendly voice
        tts=openai.TTS(voice="alloy"),
    )

    # -------------------------------------------------------------------------
    # Step 4: Create and start the agent session
    # -------------------------------------------------------------------------
    # AgentSession manages the lifecycle of the voice interaction
    session = AgentSession()

    # -------------------------------------------------------------------------
    # Step 5: Set up transcript collection and conversation tracing
    # -------------------------------------------------------------------------
    # Collect all messages into a transcript for saving later
    transcript_lines: list[str] = []

    @session.on("user_input_transcribed")
    def on_user_input(event):
        """Called when user speech is transcribed to text."""
        # Add to transcript
        transcript_lines.append(f"User: {event.transcript}")

        # Trace for Phoenix observability
        with tracer.start_as_current_span("user_message") as span:
            span.set_attribute("message.role", "user")
            span.set_attribute("message.content", event.transcript)
            span.set_attribute("participant.identity", participant.identity)
            logger.info(f"User said: {event.transcript}")

    @session.on("agent_speech_committed")
    def on_agent_speech(event):
        """Called when agent finishes speaking a response."""
        # Add to transcript
        transcript_lines.append(f"Agent: {event.content}")

        # Trace for Phoenix observability
        with tracer.start_as_current_span("agent_message") as span:
            span.set_attribute("message.role", "assistant")
            span.set_attribute("message.content", event.content)
            logger.info(f"Agent said: {event.content}")

    @session.on("close")
    def on_session_close(event):
        """Called when the session ends - save conversation data."""
        logger.info(f"Session closed: {event.reason}")

        # Only save if we have conversation content
        if len(transcript_lines) > 0:
            full_transcript = "\n\n".join(transcript_lines)
            try:
                save_conversation_data(full_transcript)
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")
        else:
            logger.info("No conversation content to save")

    # Start the agent session with the room for audio I/O
    logger.info("Starting agent session...")
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )
    logger.info("Agent session started, listening for user speech...")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # The CLI handles:
    # - Parsing command line arguments (dev, start, etc.)
    # - Connecting to LiveKit server using env variables
    # - Managing worker lifecycle
    # - Dispatching jobs to the entrypoint function

    cli.run_app(
        WorkerOptions(
            # The entrypoint function to call for each room connection
            entrypoint_fnc=entrypoint,
            # Prewarm function to load models before handling requests
            prewarm_fnc=prewarm_process,
        ),
    )
