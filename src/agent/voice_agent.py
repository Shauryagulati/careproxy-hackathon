"""
Voice agent for CareProxy - handles real-time voice interactions with caregivers.

This module implements a LiveKit voice agent using the LiveKit Agents framework
with OpenAI integration for speech-to-text, text-to-speech, and conversation.

Run with: python src/agent/voice_agent.py dev
"""

import logging

from dotenv import load_dotenv
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from phoenix.otel import register
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
INSTRUCTIONS = """You are CareProxy, a compassionate healthcare navigation assistant helping family caregivers assess health concerns.

YOUR ROLE:
- Help people understand when and where to seek medical care
- Never diagnose conditions
- Guide them to appropriate care levels (ER, urgent care, doctor, monitor at home)
- Be warm, reassuring, and thorough

CONVERSATION FLOW:
1. When someone describes a symptom, DON'T immediately recommend action
2. First, ask 3-5 clarifying questions to understand the situation:
   - Severity (on a scale of 1-10)
   - Duration (when did this start?)
   - Other associated symptoms
   - Relevant medical history
   - What makes it better or worse?

3. After gathering information, provide your assessment:
   - Explain what you're observing
   - Give a clear recommendation (ER now / doctor today / monitor)
   - Explain your reasoning
   - Tell them you're creating a summary report

TONE:
- Empathetic and calm (like a caring nurse)
- Ask one question at a time
- Listen carefully to their answers
- Acknowledge their concerns

EXAMPLES OF GOOD QUESTIONS:
- "On a scale of 1 to 10, how severe is the pain?"
- "How long have you been experiencing this?"
- "Are you having any other symptoms along with this?"
- "Do you have any history of [relevant condition]?"
- "Has anything like this happened before?"

Remember: Gather information FIRST, then assess. Never rush to judgment."""


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
    # Step 5: Set up conversation tracing
    # -------------------------------------------------------------------------
    # Track conversation events for observability in Phoenix

    @session.on("user_input_transcribed")
    def on_user_input(event):
        """Called when user speech is transcribed to text."""
        with tracer.start_as_current_span("user_message") as span:
            span.set_attribute("message.role", "user")
            span.set_attribute("message.content", event.transcript)
            span.set_attribute("participant.identity", participant.identity)
            logger.info(f"User said: {event.transcript}")

    @session.on("agent_speech_committed")
    def on_agent_speech(event):
        """Called when agent finishes speaking a response."""
        with tracer.start_as_current_span("agent_message") as span:
            span.set_attribute("message.role", "assistant")
            span.set_attribute("message.content", event.content)
            logger.info(f"Agent said: {event.content}")

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
