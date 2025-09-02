"""Simple configuration using environment variables."""

import os
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # dotenv is optional - continue without it
    pass

# Configuration from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_TARGET_MODEL = os.getenv("DEFAULT_TARGET_MODEL", "google/gemini-2.5-flash")
DEFAULT_JUDGE_MODEL = os.getenv("DEFAULT_JUDGE_MODEL", "openai/o4-mini")