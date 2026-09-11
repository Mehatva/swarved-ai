"""
SwarVed AI Backend — Central Configuration
==========================================
All tuneable constants live here so no magic strings are buried in
routes or database modules.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
DATABASE_PATH: Path = DATA_DIR / "swarved_incidents.db"

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_V1_PREFIX: str = "/api/v1"
APP_TITLE: str = "SwarVed AI — National DPI Interoperability Backend"
APP_DESCRIPTION: str = (
    "FastAPI backend for automated cyber incident registration "
    "and live WebSocket threat analytics."
)
APP_VERSION: str = "1.0.0"

# I4C reference number format: I4C-YYYY-XXXXXXXX
I4C_REF_PREFIX: str = "I4C"
I4C_REF_HEX_LENGTH: int = 8

# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
# Maximum simultaneous WebSocket connections to the live threat feed
WS_MAX_CONNECTIONS: int = 50

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Restrict in production to the actual frontend origin
CORS_ALLOW_ORIGINS: list[str] = ["*"]

# ---------------------------------------------------------------------------
# Server (used when running main.py directly)
# ---------------------------------------------------------------------------
SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 8000
SERVER_RELOAD: bool = False
