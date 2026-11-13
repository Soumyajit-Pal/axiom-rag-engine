from typing import Optional
import pandas as pd
import requests

from config import GEMINI_BASE_URL, GENERATION_MODEL
from core.embedder import get_api_key
from utils.helpers import logger


SYSTEM_INSTRUCTION = (
    "You are an analytical AI assistant answering questions strictly based on the provided document context.\n"
    "Guidelines:\n"
    "1. Answer using ONLY the information presented in the context chunks.\n"
    "2. If the context does not contain sufficient details to answer, state clearly: "
    "'I do not have enough information in the provided document to answer this question.'\n"
    "3. Explicitly cite the page number(s) (e.g., [Page X]) where relevant facts were found.\n"
    "4. Do not make up facts or extrapolate beyond what is documented."
)


def build_rag_prompt(query: str, retrieved_chunks: pd.DataFrame) -> str:
    """
    Assembles a grounded prompt combining the retrieved context chunks and the user query.
    """
    if retrieved_chunks.empty:
        context_str = "No relevant context found in the document."
    else:
        formatted_chunks = []
        for rank, (_, row) in enumerate(retrieved_chunks.iterrows(), start=1):
            pages = row.get("page_numbers", [])
            page_label = ", ".join(map(str, pages)) if isinstance(pages, list) else str(pages)
            score = row.get("similarity_score", 0.0)
            text = row.get("text", "")
            
            chunk_block = (
                f"[Source Chunk #{rank} | Page(s): {page_label} | Cosine Similarity: {score:.4f}]\n"
                f"{text}"
            )
            formatted_chunks.append(chunk_block)
            
        context_str = "\n\n---\n\n".join(formatted_chunks)

    prompt = f"""{SYSTEM_INSTRUCTION}

CONTEXT FROM DOCUMENT:
----------------------------------------
{context_str}
----------------------------------------

USER QUESTION:
{query}

ANSWER:"""
    return prompt


def generate_rag_response(
    prompt: str,
    api_key: Optional[str] = None,
    temperature: float = 0.2
) -> str:
    """
    Calls the Gemini REST API generateContent endpoint to produce a grounded response.
    
    Args:
        prompt: Assembled prompt string with context and question
        api_key: Optional Gemini API key override
        temperature: Sampling temperature (low for factual adherence)
        
    Returns:
        Generated text response string
    """
    resolved_key = get_api_key(api_key)
    endpoint = f"{GEMINI_BASE_URL}/models/{GENERATION_MODEL}:generateContent?key={resolved_key}"
    
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 1024,
            "topP": 0.95
        }
    }
    
    try:
        response = requests.post(endpoint, json=payload, timeout=45)
        
        if response.status_code != 200:
            error_details = response.text
            logger.error(f"Gemini Generation API Error ({response.status_code}): {error_details}")
            return f"⚠️ API Error ({response.status_code}): {error_details}"
            
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            feedback = data.get("promptFeedback", {})
            return f"⚠️ No response generated. Feedback: {feedback}"
            
        first_candidate = candidates[0]
        content_parts = first_candidate.get("content", {}).get("parts", [])
        if not content_parts:
            return "⚠️ Empty response returned by model."
            
        return content_parts[0].get("text", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during text generation: {e}")
        return f"⚠️ Network error while communicating with Gemini API: {e}"
