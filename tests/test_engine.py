import unittest
import numpy as np
import pandas as pd
from core.chunker import find_optimal_split_point, sliding_window_chunker
from core.retriever import calculate_cosine_similarity
from utils.cache import save_cache, load_cache, has_cache


class TestNumPyRAGEngine(unittest.TestCase):

    def test_sentence_boundary_snapping(self):
        sample_text = "This is sentence one. This is sentence two. This is sentence three."
        target_end = 35
        min_split = 20
        split_point = find_optimal_split_point(sample_text, target_end, min_split)
        # Should snap after "This is sentence one." (index 21) or sentence two
        self.assertIn(sample_text[split_point - 1], [".", " "])

    def test_sliding_window_chunker(self):
        pages = [
            {"page": 1, "text": "Deep learning models require computational power. Linear algebra is the foundation."},
            {"page": 2, "text": "Retrieval augmented generation connects LLMs with proprietary knowledge bases."}
        ]
        chunks = sliding_window_chunker(pages, chunk_size=60, overlap=15, source_name="test.pdf")
        self.assertGreater(len(chunks), 0)
        self.assertIn("chunk_id", chunks[0])
        self.assertIn("page_numbers", chunks[0])
        self.assertIn("text", chunks[0])

    def test_cosine_similarity_math(self):
        # Query: [1, 0, 0]
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Doc 1: [1, 0, 0] (identical -> sim = 1.0)
        # Doc 2: [0, 1, 0] (orthogonal -> sim = 0.0)
        # Doc 3: [-1, 0, 0] (opposite -> sim = -1.0)
        doc_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0]
        ], dtype=np.float32)

        sims = calculate_cosine_similarity(q, doc_matrix)
        self.assertAlmostEqual(sims[0], 1.0, places=5)
        self.assertAlmostEqual(sims[1], 0.0, places=5)
        self.assertAlmostEqual(sims[2], -1.0, places=5)

    def test_cache_serialization(self):
        dummy_hash = "test_hash_12345"
        dummy_df = pd.DataFrame([{"chunk_id": 0, "text": "test", "page_numbers": [1]}])
        dummy_matrix = np.random.randn(1, 768).astype(np.float32)

        save_cache(dummy_hash, dummy_df, dummy_matrix)
        self.assertTrue(has_cache(dummy_hash))

        loaded_df, loaded_matrix = load_cache(dummy_hash)
        self.assertEqual(len(loaded_df), 1)
        self.assertEqual(loaded_matrix.shape, (1, 768))


if __name__ == "__main__":
    unittest.main()
