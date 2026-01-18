"""
Triage module for CareProxy - analyzes conversation transcripts and generates
structured triage assessments using GPT-4o.

This module provides intelligent analysis of caregiver conversations to
determine urgency levels and appropriate care recommendations.
"""

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("careproxy.triage")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System prompt for triage analysis
TRIAGE_ANALYSIS_PROMPT = """You are a medical triage analyst. Analyze the following conversation between a caregiver and a healthcare navigation assistant.

Extract and assess the following information from the conversation:

1. CHIEF COMPLAINT: What is the main symptom or concern?
2. SEVERITY: Was a severity score mentioned (1-10 scale)?
3. DURATION: How long have symptoms been present?
4. ASSOCIATED SYMPTOMS: Any other symptoms mentioned?
5. RED FLAGS: Any concerning signs that indicate emergency (e.g., chest pain with shortness of breath, sudden severe headache, signs of stroke, difficulty breathing, high fever in infants)?
6. QUESTIONS ASKED: What clarifying questions did the agent ask?

Based on your analysis, determine the URGENCY LEVEL:
- "emergency": Life-threatening, call 911 immediately (e.g., stroke symptoms, severe chest pain, difficulty breathing, loss of consciousness)
- "urgent": Needs same-day medical attention (e.g., high fever, moderate pain, concerning symptoms)
- "routine": Can wait for regular doctor appointment (e.g., mild symptoms, chronic issues, minor concerns)
- "monitor": Watch and wait, no immediate action needed (e.g., very mild symptoms, improving condition)

Respond ONLY with valid JSON in this exact format:
{
  "urgency_level": "emergency" | "urgent" | "routine" | "monitor",
  "urgency_emoji": "🔴" | "🟡" | "🟢" | "⚪",
  "chief_complaint": "brief description of main concern",
  "key_symptoms": ["symptom1", "symptom2"],
  "severity_score": <number 1-10 or null if not mentioned>,
  "duration": "how long symptoms have been present or null",
  "red_flags": ["flag1", "flag2"] or [],
  "recommendation": "clear action the caregiver should take",
  "reasoning": "brief explanation of why this urgency level was assigned",
  "questions_asked": ["question1", "question2"]
}

Use these emoji mappings:
- emergency = 🔴
- urgent = 🟡
- routine = 🟢
- monitor = ⚪"""


def assess_conversation(transcript: str) -> dict:
    """
    Analyze a conversation transcript and generate a structured triage assessment.

    This function sends the conversation to GPT-4o for analysis and returns
    a structured assessment including urgency level, symptoms, and recommendations.

    Args:
        transcript: The full conversation transcript between caregiver and agent

    Returns:
        dict: Structured triage assessment with urgency level, symptoms,
              recommendations, and reasoning

    Raises:
        ValueError: If the transcript is empty
        RuntimeError: If the OpenAI API call fails
    """
    # Validate input
    if not transcript or not transcript.strip():
        logger.error("Empty transcript provided")
        raise ValueError("Transcript cannot be empty")

    logger.info("Starting triage assessment...")
    logger.debug(f"Transcript length: {len(transcript)} characters")

    try:
        # Call GPT-4o for analysis
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": TRIAGE_ANALYSIS_PROMPT},
                {"role": "user", "content": f"Analyze this conversation:\n\n{transcript}"},
            ],
            temperature=0.3,  # Lower temperature for more consistent analysis
            response_format={"type": "json_object"},  # Ensure JSON output
        )

        # Extract the response content
        response_text = response.choices[0].message.content
        logger.debug(f"Raw response: {response_text}")

        # Parse JSON response
        assessment = json.loads(response_text)

        # Validate required fields
        required_fields = [
            "urgency_level",
            "urgency_emoji",
            "chief_complaint",
            "key_symptoms",
            "recommendation",
            "reasoning",
        ]
        for field in required_fields:
            if field not in assessment:
                logger.warning(f"Missing required field: {field}")
                assessment[field] = None if field != "key_symptoms" else []

        # Ensure urgency_emoji matches urgency_level
        emoji_map = {
            "emergency": "🔴",
            "urgent": "🟡",
            "routine": "🟢",
            "monitor": "⚪",
        }
        if assessment.get("urgency_level") in emoji_map:
            assessment["urgency_emoji"] = emoji_map[assessment["urgency_level"]]

        logger.info(
            f"Assessment complete: {assessment['urgency_level']} - {assessment['chief_complaint']}"
        )
        return assessment

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse GPT response as JSON: {e}")
        raise RuntimeError(f"Invalid JSON response from GPT: {e}")

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise RuntimeError(f"Failed to analyze conversation: {e}")


def get_urgency_color(urgency_level: str) -> str:
    """
    Get the display color for an urgency level.

    Args:
        urgency_level: One of 'emergency', 'urgent', 'routine', 'monitor'

    Returns:
        str: Color name for display purposes
    """
    colors = {
        "emergency": "red",
        "urgent": "yellow",
        "routine": "green",
        "monitor": "gray",
    }
    return colors.get(urgency_level, "gray")


def format_assessment_summary(assessment: dict) -> str:
    """
    Format a triage assessment as a human-readable summary.

    Args:
        assessment: The assessment dict from assess_conversation()

    Returns:
        str: Formatted summary string
    """
    lines = [
        f"{assessment.get('urgency_emoji', '⚪')} URGENCY: {assessment.get('urgency_level', 'unknown').upper()}",
        f"",
        f"Chief Complaint: {assessment.get('chief_complaint', 'Not specified')}",
        f"",
    ]

    if assessment.get("key_symptoms"):
        lines.append("Key Symptoms:")
        for symptom in assessment["key_symptoms"]:
            lines.append(f"  • {symptom}")
        lines.append("")

    if assessment.get("severity_score"):
        lines.append(f"Severity: {assessment['severity_score']}/10")

    if assessment.get("duration"):
        lines.append(f"Duration: {assessment['duration']}")

    if assessment.get("red_flags"):
        lines.append("")
        lines.append("⚠️  Red Flags:")
        for flag in assessment["red_flags"]:
            lines.append(f"  • {flag}")

    lines.append("")
    lines.append(f"Recommendation: {assessment.get('recommendation', 'Consult a healthcare provider')}")
    lines.append("")
    lines.append(f"Reasoning: {assessment.get('reasoning', 'N/A')}")

    return "\n".join(lines)
