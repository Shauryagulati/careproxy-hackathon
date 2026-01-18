"""
Reports module for CareProxy - generates summaries and reports for care coordination.

This module provides two report formats:
1. Caregiver Report: Friendly, easy-to-read for family members
2. Physician Report: Medical-grade structured documentation
"""

import json
from datetime import datetime


def _get_value(data: dict, key: str, default: str = "Not specified") -> str:
    """Safely get a value from dict, returning default if missing or None."""
    value = data.get(key)
    if value is None or value == "":
        return default
    return str(value)


def _format_list(items: list | None, default: str = "None") -> str:
    """Format a list as bullet points or return default if empty."""
    if not items:
        return default
    return "\n".join(f"  - {item}" for item in items)


def _get_urgency_display(level: str) -> str:
    """Convert urgency level to display format with emoji."""
    mapping = {
        "emergency": "🔴 EMERGENCY",
        "urgent": "🟡 URGENT",
        "routine": "🟢 ROUTINE",
        "monitor": "⚪ MONITOR",
    }
    return mapping.get(level.lower(), f"⚪ {level.upper()}")


def generate_caregiver_report(triage_data: dict, transcript: str) -> str:
    """
    Generate a friendly, easy-to-read report for family caregivers.

    This report uses plain language, clear action items, and avoids
    medical jargon to help caregivers understand the assessment and
    know what steps to take.

    Args:
        triage_data: The triage assessment dict from assess_conversation()
        transcript: The full conversation transcript

    Returns:
        str: Formatted caregiver-friendly report
    """
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # Build the summary from available data
    chief_complaint = _get_value(triage_data, "chief_complaint", "your concern")
    duration = _get_value(triage_data, "duration", "not specified")
    severity = triage_data.get("severity_score")
    severity_display = f"{severity}/10" if severity else "Not specified"

    # Get symptoms list
    symptoms = triage_data.get("key_symptoms", [])
    symptoms_display = ", ".join(symptoms) if symptoms else "Not specified"

    # Build recommendation section
    recommendation = _get_value(triage_data, "recommendation", "Please consult with a healthcare provider")

    # Build warning signs section
    red_flags = triage_data.get("red_flags", [])
    warning_section = ""
    if red_flags:
        warning_section = f"""
WARNING SIGNS TO WATCH FOR:
{_format_list(red_flags)}
"""

    # Get urgency display
    urgency_level = _get_value(triage_data, "urgency_level", "unknown")
    urgency_display = _get_urgency_display(urgency_level)

    # Build plain English summary
    summary_parts = [f"We discussed {chief_complaint}."]
    if duration != "Not specified":
        summary_parts.append(f"This has been going on for {duration}.")
    if severity:
        summary_parts.append(f"The severity was rated {severity} out of 10.")

    plain_summary = " ".join(summary_parts)

    report = f"""
📋 CAREPROXY ASSESSMENT SUMMARY
Generated: {timestamp}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT WE DISCUSSED:
{plain_summary}

{urgency_display}

RECOMMENDATION:
{recommendation}
{warning_section}
WHAT WE LEARNED:
  - Chief concern: {chief_complaint.capitalize()}
  - Severity: {severity_display}
  - Duration: {duration}
  - Key symptoms: {symptoms_display}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  This assessment is for guidance only and does not replace medical advice.
    If symptoms worsen or you're concerned, seek medical attention immediately.
"""
    return report.strip()


def generate_physician_report(triage_data: dict, transcript: str) -> str:
    """
    Generate a medical-grade structured report for healthcare providers.

    This report follows clinical documentation standards and includes
    all relevant details for medical decision-making.

    Args:
        triage_data: The triage assessment dict from assess_conversation()
        transcript: The full conversation transcript

    Returns:
        str: Formatted physician report
    """
    timestamp = datetime.now().isoformat()
    timestamp_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Extract all relevant data
    chief_complaint = _get_value(triage_data, "chief_complaint")
    severity = triage_data.get("severity_score")
    severity_display = f"{severity}/10" if severity else "Not assessed"
    duration = _get_value(triage_data, "duration")
    urgency_level = _get_value(triage_data, "urgency_level", "Not determined")
    reasoning = _get_value(triage_data, "reasoning", "No reasoning provided")
    recommendation = _get_value(triage_data, "recommendation")

    # Format symptoms
    symptoms = triage_data.get("key_symptoms", [])
    symptoms_display = _format_list(symptoms, "None reported")

    # Format red flags
    red_flags = triage_data.get("red_flags", [])
    red_flags_display = _format_list(red_flags, "None identified")

    # Format questions asked
    questions = triage_data.get("questions_asked", [])
    questions_display = _format_list(questions, "Not documented")

    # Build history of present illness narrative
    hpi_parts = []
    if chief_complaint != "Not specified":
        hpi_parts.append(f"Patient/caregiver reports {chief_complaint}.")
    if duration != "Not specified":
        hpi_parts.append(f"Symptoms have been present for {duration}.")
    if severity:
        hpi_parts.append(f"Severity rated as {severity}/10 by patient/caregiver.")
    if symptoms:
        hpi_parts.append(f"Associated symptoms include: {', '.join(symptoms)}.")
    if not red_flags:
        hpi_parts.append("No red flag symptoms were identified during triage.")
    else:
        hpi_parts.append(f"Red flags identified: {', '.join(red_flags)}.")

    hpi_narrative = " ".join(hpi_parts) if hpi_parts else "Insufficient information obtained."

    report = f"""
PATIENT ENCOUNTER SUMMARY - CAREPROXY
Generated: {timestamp_display}
Document ID: CPX-{datetime.now().strftime('%Y%m%d%H%M%S')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHIEF COMPLAINT:
{chief_complaint}

HISTORY OF PRESENT ILLNESS:
{hpi_narrative}

SYMPTOM ASSESSMENT:
  - Severity: {severity_display}
  - Duration: {duration}
  - Onset: Not specified
  - Associated symptoms:
{symptoms_display}

RED FLAGS IDENTIFIED:
{red_flags_display}

TRIAGE ASSESSMENT:
  Level: {urgency_level.upper()}
  Reasoning: {reasoning}

CLINICAL QUESTIONS ASKED:
{questions_display}

RECOMMENDATIONS:
{recommendation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONVERSATION TRANSCRIPT:

{transcript.strip()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DISCLAIMER: This is an automated triage assessment generated by CareProxy.
Clinical correlation is required. This document does not constitute a
medical diagnosis or replace professional medical evaluation.
"""
    return report.strip()


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Test with sample data
    sample_triage = {
        "urgency_level": "urgent",
        "urgency_emoji": "🟡",
        "chief_complaint": "chest pain when breathing deeply",
        "key_symptoms": [
            "chest pain",
            "pain worsens with deep breaths",
            "intermittent discomfort",
        ],
        "severity_score": 4,
        "duration": "3 hours",
        "red_flags": [],
        "recommendation": "Schedule a same-day appointment with your primary care physician or visit an urgent care clinic. Given the recent history of bronchitis and the nature of the pain (worsening with breathing), this should be evaluated today but does not require emergency care.",
        "reasoning": "Chest pain rated 4/10, pleuritic in nature (worse with breathing), no cardiac symptoms (no arm/jaw pain, shortness of breath, or dizziness), recent bronchitis history suggests possible musculoskeletal or respiratory cause rather than cardiac.",
        "questions_asked": [
            "How severe is the pain on a scale of 1-10?",
            "How long has the pain been present?",
            "Is the pain constant or intermittent?",
            "Any associated symptoms like shortness of breath or dizziness?",
            "Any history of heart or lung problems?",
        ],
    }

    sample_transcript = """Agent: Hi, I'm CareProxy, your healthcare navigation assistant. What's concerning you today?

User: My dad has been having chest pain since this morning.

Agent: I'm sorry to hear that. Chest pain can be concerning. On a scale of 1 to 10, how severe would you say the pain is?

User: He says it's about a 4 out of 10.

Agent: Thank you. How long has he been experiencing this pain?

User: It started about 3 hours ago.

Agent: Is the pain constant, or does it come and go?

User: It comes and goes. It gets worse when he takes deep breaths.

Agent: I see. Is he experiencing any other symptoms along with the chest pain? Things like shortness of breath, dizziness, nausea, or pain in his arm or jaw?

User: No, none of those. Just the chest pain when breathing.

Agent: Does your dad have any history of heart problems or lung conditions?

User: No heart problems. He did have bronchitis last month.

Agent: That's helpful context. Based on what you've told me - the pain being a 4/10, worsening with deep breaths, no other cardiac symptoms, and his recent bronchitis - this could be related to his respiratory system rather than his heart. I recommend scheduling a same-day appointment with his doctor or visiting urgent care to get this evaluated today."""

    print("=" * 70)
    print("CAREGIVER REPORT")
    print("=" * 70)
    print(generate_caregiver_report(sample_triage, sample_transcript))

    print("\n\n")

    print("=" * 70)
    print("PHYSICIAN REPORT")
    print("=" * 70)
    print(generate_physician_report(sample_triage, sample_transcript))
