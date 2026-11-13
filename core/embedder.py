import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

from config import (
    EMBED_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
)
from utils.helpers import logger


def get_api_key(override_key: Optional[str] = None) -> str:
    """Resolves API key from override parameter or config/env."""
    key = (override_key or GEMINI_API_KEY).strip()
    if not key:
        raise ValueError(
            "Gemini API Key is not set! Please provide it in your .env file "
            "or enter it in the Streamlit sidebar."
        )
    return key


def embed_texts(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    api_key: Optional[str] = None,
    batch_size: int = EMBED_BATCH_SIZE
) -> np.ndarray:
    """
    Batches and converts a list of texts into dense vectors using 
    the Gemini REST API batchEmbedContents endpoint.
    
    Args:
        texts: List of strings to embed
        task_type: 'RETRIEVAL_DOCUMENT' for docs or 'RETRIEVAL_QUERY' for questions
        api_key: Optional Gemini API key override
        batch_size: Maximum texts per batch API call (Gemini limit is 100)
        
    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM) with dtype float32
    """
    resolved_key = get_api_key(api_key)
    endpoint = f"{GEMINI_BASE_URL}/models/{EMBEDDING_MODEL}:batchEmbedContents?key={resolved_key}"
    
    all_vectors = []
    total_texts = len(texts)
    
    for i in range(0, total_texts, batch_size):
        batch = texts[i : i + batch_size]
        
        # Build payload according to Gemini REST API specs
        payload = {
            "requests": [
                {
                    "model": f"models/{EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                }
                for t in batch
            ]
        }
        
        # Exponential backoff retry loop
        retries = 3
        backoff = 1.0
        success = False
        
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(endpoint, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    raw_embeddings = data.get("embeddings", [])
                    if len(raw_embeddings) != len(batch):
                        raise ValueError(
                            f"Mismatch: Sent {len(batch)} items, received {len(raw_embeddings)} embeddings"
                        )
                    for item in raw_embeddings:
                        all_vectors.append(item["values"])
                    success = True
                    break
                elif response.status_code == 429:
                    logger.warning(
                        f"Rate limit (429) hit on batch {i // batch_size + 1}. Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    error_msg = response.text
                    logger.error(f"API Error ({response.status_code}): {error_msg}")
                    if attempt == retries:
                        raise RuntimeError(f"Gemini API returned {response.status_code}: {error_msg}")
                    time.sleep(backoff)
                    backoff *= 2
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error on attempt {attempt}: {e}")
                if attempt == retries:
                    raise e
                time.sleep(backoff)
                backoff *= 2
                
        if not success:
            raise RuntimeError(f"Failed to embed batch starting at index {i}")

        # Minor pause between batches to prevent quota bursts
        if i + batch_size < total_texts:
            time.sleep(0.3)
            
    matrix = np.array(all_vectors, dtype=np.float32)
    logger.info(f"Generated embedding matrix with shape: {matrix.shape}")
    return matrix


def embed_single_query(
    query: str,
    api_key: Optional[str] = None
) -> np.ndarray:
    """
    Embeds a single user query string using taskType='RETRIEVAL_QUERY'.
    
    Returns:
        1D np.ndarray of shape (EMBEDDING_DIM,) with dtype float32
    """
    resolved_key = get_api_key(api_key)
    endpoint = f"{GEMINI_BASE_URL}/models/{EMBEDDING_MODEL}:embedContent?key={resolved_key}"
    
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": query}]},
        "taskType": "RETRIEVAL_QUERY"
    }
    
    response = requests.post(endpoint, json=payload, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Gemini Query Embed Error ({response.status_code}): {response.text}")
        
    data = response.json()
    vector = np.array(data["embedding"]["values"], dtype=np.float32)
    return vector


def build_vector_store(
    chunks: List[Dict],
    embeddings: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Structures the chunks into a metadata DataFrame and ensures alignment
    with the dense embedding matrix.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"Length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
        
    df = pd.DataFrame(chunks)
    return df, embeddings
