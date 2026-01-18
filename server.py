from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).resolve().parent
LATEST = BASE / "conversations" / "latest.json"
HISTORY = BASE / "conversations" / "history.json"

@app.get("/api/latest")
def latest():
    if not LATEST.exists():
        return {}
    return json.loads(LATEST.read_text())

@app.get("/api/history")
def history():
    if not HISTORY.exists():
        return []
    return json.loads(HISTORY.read_text())

