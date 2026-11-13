import io
import time
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_DIM,
    GEMINI_API_KEY,
)
from core.chunker import extract_text_from_pdf, sliding_window_chunker
from core.embedder import build_vector_store, embed_texts
from core.retriever import retrieve_top_k
from core.generator import build_rag_prompt, generate_rag_response
from utils.cache import load_cache, save_cache
from utils.helpers import compute_file_hash, logger

# Configure Streamlit page
st.set_page_config(
    page_title="Axiom RAG Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern portfolio aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-tag {
        font-size: 0.95rem;
        color: #7e8c9b;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        padding: 12px;
        border-radius: 8px;
        background-color: #f0f2f6;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .source-card {
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        background-color: #f8f9fa;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df_chunks" not in st.session_state:
    st.session_state.df_chunks = None
if "embedding_matrix" not in st.session_state:
    st.session_state.embedding_matrix = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "active_file_hash" not in st.session_state:
    st.session_state.active_file_hash = None


# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("⚡ Control Panel")
    st.caption("Framework-Free NumPy RAG Engine")
    
    st.markdown("---")
    
    # 1. PDF Upload
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    # 3. Hyperparameters
    with st.expander("⚙️ Engine Hyperparameters", expanded=False):
        chunk_size = st.slider(
            "Chunk Size (Characters)",
            min_value=200,
            max_value=1200,
            value=DEFAULT_CHUNK_SIZE,
            step=50,
            help="Length of each sliding context chunk"
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (Characters)",
            min_value=20,
            max_value=300,
            value=DEFAULT_CHUNK_OVERLAP,
            step=10,
            help="Overlap between consecutive chunks to maintain semantic flow"
        )
        top_k = st.slider(
            "Top-K Chunks to Retrieve",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            help="Number of most relevant chunks passed to the LLM"
        )
        temperature = st.slider(
            "Generation Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Lower values yield strictly grounded factual responses"
        )

    # 4. Ingest Button
    index_btn = st.button("🚀 Ingest & Index Document", use_container_width=True, type="primary")

    if index_btn:
        if not GEMINI_API_KEY:
            st.error("⚠️ GEMINI_API_KEY is not configured in .env. Please set it to proceed.")
        elif uploaded_file is None:
            st.warning("Please upload a PDF document first.")
        else:
            with st.spinner("Processing document..."):
                try:
                    pdf_bytes = uploaded_file.getvalue()
                    file_hash = compute_file_hash(pdf_bytes)
                    cached_data = load_cache(file_hash)
                    
                    if cached_data is not None:
                        # Load from persistent cache
                        df_chunks, matrix = cached_data
                        st.session_state.df_chunks = df_chunks
                        st.session_state.embedding_matrix = matrix
                        st.session_state.active_file_hash = file_hash
                        st.session_state.doc_stats = {
                            "filename": uploaded_file.name,
                            "pages": int(df_chunks["page_numbers"].explode().nunique()),
                            "chunks": len(df_chunks),
                            "cached": True
                        }
                        st.success(f"⚡ Loaded {len(df_chunks)} chunks from local cache!")
                    else:
                        # Fresh ingestion pipeline
                        t0 = time.time()
                        
                        # Phase 1: Ingestion & Sliding Window Chunking
                        stream = io.BytesIO(pdf_bytes)
                        pages_data = extract_text_from_pdf(stream)
                        if not pages_data:
                            st.error("No readable text found in PDF. Make sure it is not a scanned image.")
                            st.stop()
                            
                        chunks = sliding_window_chunker(
                            pages_data,
                            chunk_size=chunk_size,
                            overlap=chunk_overlap,
                            source_name=uploaded_file.name
                        )
                        
                        # Phase 2: Vectorization via Gemini REST API
                        texts_to_embed = [c["text"] for c in chunks]
                        with st.status(f"Vectorizing {len(texts_to_embed)} chunks via Gemini API...", expanded=True) as status:
                            matrix = embed_texts(
                                texts_to_embed,
                                task_type="RETRIEVAL_DOCUMENT"
                            )
                            df_chunks, matrix = build_vector_store(chunks, matrix)
                            save_cache(file_hash, df_chunks, matrix)
                            status.update(label="Vectorization complete & cached!", state="complete")
                            
                        st.session_state.df_chunks = df_chunks
                        st.session_state.embedding_matrix = matrix
                        st.session_state.active_file_hash = file_hash
                        st.session_state.doc_stats = {
                            "filename": uploaded_file.name,
                            "pages": len(pages_data),
                            "chunks": len(df_chunks),
                            "cached": False,
                            "time_taken": round(time.time() - t0, 2)
                        }
                        st.success(f"Indexed {len(df_chunks)} chunks in {st.session_state.doc_stats['time_taken']}s!")
                except Exception as err:
                    logger.exception("Indexing failed")
                    st.error(f"Failed to process document: {err}")

    # 5. Live Document Stats
    if st.session_state.doc_stats:
        st.markdown("---")
        st.subheader("📊 Document Index Stats")
        stats = st.session_state.doc_stats
        st.write(f"**Document:** `{stats.get('filename')}`")
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Total Pages", stats.get("pages", 0))
        col_s2.metric("Total Chunks", stats.get("chunks", 0))
        st.write(f"**Dense Dimension:** `{EMBEDDING_DIM}d`")
        status_tag = "🟢 Cached Index" if stats.get("cached") else "🔵 Freshly Built"
        st.caption(f"Status: {status_tag}")
        
        if st.button("🗑️ Clear Session & Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.df_chunks = None
            st.session_state.embedding_matrix = None
            st.session_state.doc_stats = {}
            st.rerun()


# --- MAIN AREA ---
st.markdown('<div class="main-title">⚡ Axiom RAG Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-tag">Pure Mathematical Vector Retrieval (<code>np.dot</code>) &bull; Zero Frameworks &bull; Zero Vector Databases &bull; Gemini REST API</div>',
    unsafe_allow_html=True
)

# Architectural highlight banner
with st.expander("📐 Technical Architecture & Implementation Details", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**1. Ingestion & Splitter**")
        st.caption(
            "Sliding character window with sentence-boundary snapping and exact page interval mapping."
        )
    with col_b:
        st.markdown("**2. Vector Retrieval**")
        st.caption(
            "Vectorized Cosine Similarity via BLAS matrix-vector product `np.dot(M, q)` with `np.argsort` top-k."
        )
    with col_c:
        st.markdown("**3. Grounded Synthesis**")
        st.caption(
            "Context assembly into structured prompts with citation metadata and anti-hallucination guardrails."
        )

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📎 Retrieved Sources & Cosine Scores"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**Rank {s['rank']}** &bull; Page(s): `{s['pages']}` &bull; Score: `{s['score']:.4f}`"
                    )
                    st.caption(s["text"])

# Chat input handle
prompt_input = st.chat_input("Ask a factual question about your uploaded document...")

if prompt_input:
    # Validate preconditions
    if not GEMINI_API_KEY:
        st.warning("⚠️ GEMINI_API_KEY is not configured in .env.")
        st.stop()
        
    if st.session_state.df_chunks is None or st.session_state.embedding_matrix is None:
        st.warning("⚠️ Please upload and index a PDF document before asking questions.")
        st.stop()

    # 1. Render user message
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # 2. Execute retrieval & generation
    with st.chat_message("assistant"):
        with st.spinner("Calculating vector similarities & generating response..."):
            try:
                t_start = time.time()
                
                # Phase 3: Pure NumPy Vector Retrieval
                retrieved_df = retrieve_top_k(
                    query=prompt_input,
                    df=st.session_state.df_chunks,
                    matrix=st.session_state.embedding_matrix,
                    k=top_k
                )
                
                retrieval_latency = round((time.time() - t_start) * 1000, 2)
                
                # Phase 4: Prompt Assembly & Text Generation
                rag_prompt = build_rag_prompt(prompt_input, retrieved_df)
                response_text = generate_rag_response(
                    rag_prompt,
                    temperature=temperature
                )
                
                st.markdown(response_text)
                
                # Format sources for inspection
                sources_data = []
                if not retrieved_df.empty:
                    with st.expander(f"📎 Retrieved {len(retrieved_df)} Context Chunks (Search Latency: {retrieval_latency}ms)"):
                        for rank, (_, row) in enumerate(retrieved_df.iterrows(), start=1):
                            pages = row.get("page_numbers", [])
                            page_str = ", ".join(map(str, pages)) if isinstance(pages, list) else str(pages)
                            score = float(row.get("similarity_score", 0.0))
                            text_snippet = row.get("text", "")
                            
                            sources_data.append({
                                "rank": rank,
                                "pages": page_str,
                                "score": score,
                                "text": text_snippet
                            })
                            
                            st.markdown(
                                f"**Rank #{rank}** | Page(s): `{page_str}` | Cosine Similarity: `{score:.4f}`"
                            )
                            st.info(text_snippet)
                            
                    with st.expander("🔬 View Raw Assembled Prompt sent to Gemini"):
                        st.code(rag_prompt, language="text")

                # Store message with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources_data
                })
                
            except Exception as e:
                logger.exception("Query execution failed")
                error_msg = f"An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
