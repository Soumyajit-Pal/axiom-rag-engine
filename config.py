import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Fallback to Streamlit Community Cloud Secrets if deployed
try:
    import streamlit as st
    if not GEMINI_API_KEY and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Models
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"

# Embedding & Retrieval Hyperparameters
EMBEDDING_DIM = 3072
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5
EMBED_BATCH_SIZE = 100
