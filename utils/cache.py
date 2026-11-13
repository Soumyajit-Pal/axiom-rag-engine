from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from config import CACHE_DIR
from utils.helpers import logger


def get_cache_paths(file_hash: str) -> Tuple[Path, Path]:
    """Returns the matrix npz path and metadata json path for a given file hash."""
    matrix_path = CACHE_DIR / f"{file_hash}_matrix.npz"
    metadata_path = CACHE_DIR / f"{file_hash}_metadata.json"
    return matrix_path, metadata_path


def has_cache(file_hash: str) -> bool:
    """Checks whether valid cached embeddings and metadata exist."""
    matrix_path, metadata_path = get_cache_paths(file_hash)
    return matrix_path.is_file() and metadata_path.is_file()


def save_cache(file_hash: str, df: pd.DataFrame, matrix: np.ndarray) -> None:
    """
    Saves the embedding matrix as compressed NumPy (.npz)
    and the chunk metadata as JSON.
    """
    matrix_path, metadata_path = get_cache_paths(file_hash)
    try:
        np.savez_compressed(matrix_path, embeddings=matrix)
        df.to_json(metadata_path, orient="records", indent=2)
        logger.info(f"Successfully cached embeddings to {matrix_path.name}")
    except Exception as e:
        logger.warning(f"Failed to cache data for {file_hash}: {e}")


def load_cache(file_hash: str) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    """
    Loads cached DataFrame and embedding matrix if they exist.
    Returns (DataFrame, ndarray) or None.
    """
    if not has_cache(file_hash):
        return None

    matrix_path, metadata_path = get_cache_paths(file_hash)
    try:
        with np.load(matrix_path) as data:
            matrix = data["embeddings"]
        df = pd.read_json(metadata_path)
        logger.info(f"Loaded cached index for hash: {file_hash[:8]}... ({len(df)} chunks)")
        return df, matrix
    except Exception as e:
        logger.warning(f"Failed to load cache for {file_hash}: {e}")
        return None
