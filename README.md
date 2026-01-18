# CareProxy

A voice agent for family caregivers.

## Overview

CareProxy is an AI-powered voice assistant designed to help family caregivers manage their caregiving responsibilities. It provides real-time voice interactions, intelligent triage for care-related questions, and generates helpful reports.

## Features

- **Voice Agent**: Real-time voice conversations powered by LiveKit and OpenAI
- **Intelligent Triage**: Helps prioritize and address caregiving concerns
- **Reports**: Generate summaries and reports for care coordination

## Project Structure

```
careproxy/
├── src/
│   ├── agent/
│   │   ├── voice_agent.py   # Main voice agent implementation
│   │   ├── triage.py        # Care triage logic
│   │   └── reports.py       # Report generation
│   └── ui/                  # React frontend (coming soon)
├── requirements.txt
├── .env                     # Environment variables (not in git)
└── README.md
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env`:
   - OpenAI API key
   - LiveKit credentials

4. Run the agent:
   ```bash
   python -m src.agent.voice_agent
   ```

## Observability

This project uses Arize Phoenix for tracing and observability. Run Phoenix locally at `localhost:6006` to view traces.

## License

MIT
