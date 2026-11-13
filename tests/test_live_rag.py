import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from core.chunker import sliding_window_chunker
from core.embedder import embed_texts, build_vector_store
from core.retriever import retrieve_top_k
from core.generator import build_rag_prompt, generate_rag_response

# 1. Sample document pages
sample_pages = [
    {
        "page": 1,
        "text": "The NumPy RAG engine is a framework-free retrieval system designed for low-latency in-memory search. It utilizes raw matrix-vector multiplications via np.dot for cosine similarity."
    },
    {
        "page": 2,
        "text": "For embedding generation, the system connects directly to the Gemini REST API using gemini-embedding-001 with 3072 dense dimensions. Answers are synthesized with gemini-2.5-flash."
    }
]

print("[1] Chunking document...")
chunks = sliding_window_chunker(sample_pages, chunk_size=200, overlap=30, source_name="live_test.pdf")
print(f"    Generated {len(chunks)} chunks.")

print("[2] Embedding chunks via Gemini REST API...")
texts = [c["text"] for c in chunks]
matrix = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
print(f"    Embedding matrix shape: {matrix.shape}")

df, matrix = build_vector_store(chunks, matrix)

print("[3] Performing NumPy Vector Retrieval...")
query = "Which embedding model and dimension does the NumPy RAG engine use?"
top_k_df = retrieve_top_k(query, df, matrix, k=2)
print("    Top retrieved chunks:")
for _, row in top_k_df.iterrows():
    print(f"    - Score {row['similarity_score']:.4f} (Page {row['page_numbers']}): {row['text']}")

print("[4] Generating response via Gemini REST API...")
prompt = build_rag_prompt(query, top_k_df)
response = generate_rag_response(prompt)
print("\n=== GENERATED RESPONSE ===")
print(response)
print("===========================")
