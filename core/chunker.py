import io
import re
from pathlib import Path
from typing import BinaryIO, Dict, List, Union
import pdfplumber

from utils.helpers import clean_text, logger


def extract_text_from_pdf(file_source: Union[str, Path, BinaryIO]) -> List[Dict]:
    """
    Parses a PDF file and extracts text page-by-page.
    
    Args:
        file_source: File path (str/Path) or file-like binary stream.
        
    Returns:
        List of dicts: [{"page": int, "text": str}]
    """
    pages_data = []
    
    try:
        # If file_source is bytes or BytesIO, pdfplumber accepts BytesIO directly
        with pdfplumber.open(file_source) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text()
                if raw_text:
                    cleaned = clean_text(raw_text)
                    if cleaned:
                        pages_data.append({
                            "page": page_idx,
                            "text": cleaned
                        })
    except Exception as e:
        logger.error(f"Error parsing PDF with pdfplumber: {e}")
        # Secondary fallback: try PyMuPDF if available
        try:
            import fitz  # PyMuPDF
            if isinstance(file_source, (str, Path)):
                doc = fitz.open(file_source)
            elif hasattr(file_source, "read"):
                file_source.seek(0)
                doc = fitz.open(stream=file_source.read(), filetype="pdf")
            else:
                raise e

            pages_data = []
            for page_idx, page in enumerate(doc, start=1):
                raw_text = page.get_text()
                if raw_text:
                    cleaned = clean_text(raw_text)
                    if cleaned:
                        pages_data.append({"page": page_idx, "text": cleaned})
            logger.info("Successfully extracted text using PyMuPDF fallback")
        except Exception as fallback_err:
            logger.error(f"Fallback extraction also failed: {fallback_err}")
            raise e

    logger.info(f"Extracted {len(pages_data)} pages of text from PDF")
    return pages_data


def find_optimal_split_point(text: str, target_end: int, min_split_point: int) -> int:
    """
    Snaps the chunk boundary to the closest sentence punctuation (., !, ?) 
    within the allowed window [min_split_point, target_end].
    If no sentence boundary is found, falls back to the nearest word boundary.
    """
    sub_text = text[min_split_point:target_end]
    
    # Check for sentence endings: '.', '!', '?' followed by whitespace or end
    sentence_matches = list(re.finditer(r"[.!?](?:\s+|$)", sub_text))
    if sentence_matches:
        # Pick the last sentence boundary in the search window
        last_match = sentence_matches[-1]
        return min_split_point + last_match.end()
    
    # Fallback: find the nearest word boundary (space)
    space_matches = list(re.finditer(r"\s+", sub_text))
    if space_matches:
        last_space = space_matches[-1]
        return min_split_point + last_space.end()
    
    # If no boundary found, return target_end
    return target_end


def sliding_window_chunker(
    pages: List[Dict],
    chunk_size: int = 500,
    overlap: int = 100,
    source_name: str = "document.pdf"
) -> List[Dict]:
    """
    Builds overlapping chunks across the entire document text with 
    sentence-boundary snapping and exact page mapping.
    
    Args:
        pages: List of {"page": int, "text": str}
        chunk_size: Target characters per chunk
        overlap: Character overlap between consecutive chunks
        source_name: Source document name for metadata
        
    Returns:
        List of chunk dictionaries with metadata
    """
    if not pages:
        return []
        
    if overlap >= chunk_size:
        overlap = chunk_size // 4
        logger.warning(f"Overlap must be < chunk_size. Adjusted overlap to {overlap}")

    # Step 1: Concatenate document text while tracking page character ranges
    master_text_parts = []
    page_intervals = []  # tuples of (start_char, end_char, page_num)
    current_char_offset = 0

    for page_item in pages:
        page_num = page_item["page"]
        text_val = page_item["text"]
        start_pos = current_char_offset
        end_pos = start_pos + len(text_val)
        
        master_text_parts.append(text_val)
        page_intervals.append((start_pos, end_pos, page_num))
        
        # Add a single separator space between pages
        current_char_offset = end_pos + 1

    master_text = " ".join(master_text_parts)
    total_length = len(master_text)
    
    if total_length == 0:
        return []

    # Step 2: Sliding window extraction
    chunks = []
    chunk_id = 0
    start_index = 0

    while start_index < total_length:
        ideal_end = start_index + chunk_size
        
        if ideal_end >= total_length:
            actual_end = total_length
        else:
            # Search within the last 25% of the chunk window for a clean break
            search_window_start = max(start_index + int(chunk_size * 0.75), start_index + 1)
            actual_end = find_optimal_split_point(master_text, ideal_end, search_window_start)
            
        chunk_text = master_text[start_index:actual_end].strip()
        
        if len(chunk_text) >= 20:  # Ignore trivial noise chunks
            # Find which page(s) overlap with [start_index, actual_end]
            spanned_pages = [
                p_num for p_start, p_end, p_num in page_intervals
                if max(start_index, p_start) < min(actual_end, p_end)
            ]
            if not spanned_pages:
                spanned_pages = [page_intervals[0][2]]

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page_numbers": sorted(list(set(spanned_pages))),
                "char_start": start_index,
                "char_end": actual_end,
                "source_file": source_name,
                "token_estimate": len(chunk_text.split())
            })
            chunk_id += 1

        # Advance window
        if actual_end >= total_length:
            break
            
        step = actual_end - overlap
        if step <= start_index:
            # Ensure strictly forward progress
            start_index = actual_end
        else:
            start_index = step

    logger.info(f"Generated {len(chunks)} chunks from {len(pages)} pages")
    return chunks
