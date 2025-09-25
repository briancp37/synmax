"""Configuration and path management for data agent."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DATA_AGENT_CACHE_TTL_HOURS = int(os.getenv("DATA_AGENT_CACHE_TTL_HOURS", "24"))

# Default dataset path
DATA_PATH = DATA_DIR / "pipeline_data.parquet"

# Rules thresholds (configurable)
RULES_CONFIG: dict[str, Any] = {
    "zero_quantity_streak_days": 7,
    "imbalance_threshold_pct": 5.0,
    "ramp_risk_window_days": 30,
}

# Performance settings
USE_DUCKDB_FOR_HEAVY_GROUPBY = True
DEFAULT_PLANNER = "deterministic"


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for directory in [DATA_DIR, ARTIFACTS_DIR, CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_cache_dir() -> Path:
    """Get cache directory, creating if needed."""
    ensure_directories()
    return CACHE_DIR


def get_artifacts_dir() -> Path:
    """Get artifacts directory, creating if needed."""
    ensure_directories()
    return ARTIFACTS_DIR


def get_data_dir() -> Path:
    """Get data directory, creating if needed."""
    ensure_directories()
    return DATA_DIR
