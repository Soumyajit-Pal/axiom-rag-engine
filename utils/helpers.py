import hashlib
import logging
import re
from pathlib import Path
from typing import Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("numpy_rag")


def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from PDF:
    - Joins hyphenated line-breaks (e.g., 'computa-\ntion' -> 'computation')
    - Collapses multiple whitespace characters and linebreaks
    - Strips leading/trailing whitespace
    """
    if not text:
        return ""
    # Merge hyphenated words split across lines
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Replace line breaks with spaces
    text = re.sub(r"\r?\n+", " ", text)
    # Normalize multiple whitespace characters
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def compute_file_hash(file_source: Union[str, Path, bytes]) -> str:
    """
    Computes a SHA-256 hash from either a file path or raw bytes.
    Used for caching embeddings and chunks.
    """
    hasher = hashlib.sha256()
    if isinstance(file_source, (str, Path)):
        with open(file_source, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    elif isinstance(file_source, bytes):
        hasher.update(file_source)
    else:
        raise ValueError("Unsupported file source type for hashing")
    return hasher.hexdigest()
