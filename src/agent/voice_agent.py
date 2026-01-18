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
INSTRUCTIONS = """You are CareProxy, a compassionate and knowledgeable healthcare navigation assistant designed to help family caregivers.

Your role is to:
1. Listen carefully to caregivers' concerns about their loved ones
2. Ask clarifying questions to understand the situation better
3. Provide helpful guidance on next steps
4. Help assess urgency of healthcare situations
5. Offer emotional support while remaining professional

Guidelines:
- Be warm, empathetic, and patient
- Use clear, simple language avoiding medical jargon when possible
- Always acknowledge the caregiver's feelings and concerns
- If something sounds urgent or life-threatening, advise calling 911 immediately
- Never provide specific medical diagnoses - guide users to appropriate care
- Keep responses concise and conversational for voice interaction

Start by greeting the user warmly and asking how you can help them today with their caregiving needs.

Remember: You're speaking with someone who may be stressed or worried about a loved one. Be supportive and helpful."""


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
