from typing import Optional
import numpy as np
import pandas as pd

from core.embedder import embed_single_query
from utils.helpers import logger


def calculate_cosine_similarity(
    query_vector: np.ndarray,
    doc_matrix: np.ndarray,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Computes vectorized cosine similarities between a single query vector 
    and all document vectors in pure NumPy using BLAS dot products.
    
    Formula:
        cos_sim(q, d_i) = (d_i . q) / (||d_i|| * ||q|| + eps)
        
    Args:
        query_vector: 1D array of shape (d,)
        doc_matrix: 2D array of shape (N, d)
        eps: Small epsilon to prevent division by zero
        
    Returns:
        1D array of similarity scores of shape (N,)
    """
    # Step 1: Vectorized dot products across all rows (BLAS dgemv / sgemv)
    dot_products = np.dot(doc_matrix, query_vector)  # shape: (N,)
    
    # Step 2: Compute L2 norms
    query_norm = np.linalg.norm(query_vector)         # scalar
    doc_norms = np.linalg.norm(doc_matrix, axis=1)    # shape: (N,)
    
    # Step 3: Compute element-wise cosine similarity with zero-division safeguard
    denominator = doc_norms * query_norm + eps
    similarities = dot_products / denominator
    
    return similarities


def retrieve_top_k(
    query: str,
    df: pd.DataFrame,
    matrix: np.ndarray,
    k: int = 5,
    api_key: Optional[str] = None,
    min_score_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Embeds the user query, evaluates cosine similarity against the entire
    in-memory matrix, and returns the top-k highest scoring chunks.
    
    Args:
        query: User question string
        df: Pandas DataFrame containing chunk metadata
        matrix: NumPy array of shape (N, d)
        k: Number of chunks to retrieve
        api_key: Optional Gemini API key override
        min_score_threshold: Optional lower bound filter for relevance
        
    Returns:
        Sorted DataFrame containing the top-k chunks with 'similarity_score'
    """
    if len(df) == 0 or matrix.shape[0] == 0:
        logger.warning("Empty vector store provided to retriever")
        return pd.DataFrame()
        
    # Cap k to the number of available chunks
    k = min(k, len(df))
    
    # 1. Embed query with RETRIEVAL_QUERY task type
    query_vec = embed_single_query(query, api_key=api_key)
    
    # 2. Compute similarity scores
    scores = calculate_cosine_similarity(query_vec, matrix)
    
    # 3. Fast Top-K extraction using NumPy argsort
    # argsort returns indices in ascending order, so [-k:] selects top k,
    # and [::-1] puts the highest score first.
    top_k_indices = np.argsort(scores)[-k:][::-1]
    top_k_scores = scores[top_k_indices]
    
    # 4. Filter by minimum threshold if desired
    valid_mask = top_k_scores >= min_score_threshold
    final_indices = top_k_indices[valid_mask]
    final_scores = top_k_scores[valid_mask]
    
    if len(final_indices) == 0:
        return pd.DataFrame()
        
    # 5. Extract metadata rows and attach scores
    top_df = df.iloc[final_indices].copy()
    top_df["similarity_score"] = np.round(final_scores, 4)
    
    logger.info(
        f"Retrieved {len(top_df)} chunks for query: '{query[:30]}...' "
        f"(Top score: {final_scores[0]:.4f})"
    )
    return top_df
