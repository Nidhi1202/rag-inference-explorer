"""
Tests for RAG Inference Explorer backend.

Acceptance criteria (from 0-feature-implementation-evidence.md):
  - pip install -r requirements.txt succeeds
  - python app.py starts Gradio on localhost
  - PDF upload → chunks → ChromaDB indexed
  - RoPE question → cited answer returned
  - Inference panel shows real values, not mocks
  - Missing GROQ_API_KEY → graceful error, not stack trace
"""

import math
import os
import textwrap
import types
from unittest.mock import MagicMock, patch

import pytest

# Llama 3.1 8B architecture constants — single source of truth for tests
_LLAMA_8B = dict(layers=32, kv_heads=8, head_dim=128)


# ---------------------------------------------------------------------------
# rag.processor
# ---------------------------------------------------------------------------

class TestChunkText:
    def setup_method(self):
        from rag.processor import chunk_text
        self.chunk_text = chunk_text

    def test_returns_nonempty_list_for_nonempty_input(self):
        chunks = self.chunk_text("The quick brown fox jumped over the lazy dog. " * 50)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_each_chunk_is_nonempty_string(self):
        chunks = self.chunk_text("word " * 200)
        assert all(isinstance(c, str) and c.strip() for c in chunks)

    def test_empty_input_returns_empty_list(self):
        chunks = self.chunk_text("")
        assert chunks == []

    def test_short_text_produces_single_chunk(self):
        text = "Short sentence."
        chunks = self.chunk_text(text)
        assert len(chunks) == 1
        assert text in chunks[0]

    def test_long_text_is_split_into_multiple_chunks(self):
        # ~3000 chars — must exceed chunk_size=512 chars and produce >1 chunk
        text = "This is a sentence about transformer attention mechanisms. " * 60
        chunks = self.chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        text = "word " * 2000
        chunks = self.chunk_text(text, chunk_size=200, overlap=20)
        # no chunk should vastly exceed the chunk_size (allow 2x headroom for word boundaries)
        for chunk in chunks:
            assert len(chunk) <= 600, f"Chunk too large: {len(chunk)} chars"

    def test_whitespace_only_input_returns_empty_list(self):
        chunks = self.chunk_text("   \n\n\t  ")
        assert chunks == []


class TestBuildCollection:
    def setup_method(self):
        from rag.processor import build_collection
        self.build_collection = build_collection

    def test_returns_chromadb_collection(self):
        import chromadb
        chunks = ["Attention is all you need.", "Transformers use multi-head attention."]
        collection = self.build_collection(chunks, collection_name="test_col_basic")
        assert hasattr(collection, "query"), "Expected a ChromaDB Collection object"

    def test_collection_count_matches_chunk_count(self):
        chunks = [f"Chunk number {i} about KV-cache compression." for i in range(5)]
        collection = self.build_collection(chunks, collection_name="test_col_count")
        assert collection.count() == 5

    def test_empty_chunks_returns_empty_collection(self):
        collection = self.build_collection([], collection_name="test_col_empty")
        assert collection.count() == 0

    def test_collection_name_is_used(self):
        chunks = ["Latent attention reduces KV-cache by 16x."]
        collection = self.build_collection(chunks, collection_name="my_unique_col")
        assert collection.name == "my_unique_col"

    def test_duplicate_texts_are_stored(self):
        chunks = ["Same chunk."] * 3
        collection = self.build_collection(chunks, collection_name="test_col_dup")
        # dedup is acceptable but must not crash; count >= 1
        assert collection.count() >= 1


# ---------------------------------------------------------------------------
# rag.retriever
# ---------------------------------------------------------------------------

class TestRetrieve:
    def setup_method(self):
        from rag.processor import build_collection
        from rag.retriever import retrieve
        self.retrieve = retrieve

        chunks = [
            "TransMLA introduces multi-head latent attention to compress KV-cache.",
            "The RoPE format mismatch causes incorrect positional encoding in MLA.",
            "Benchmark results show 16x KV-cache reduction with TransMLA on SmolLM.",
            "LoRA fine-tuning adapts pre-trained models with minimal trainable parameters.",
            "Flash attention reduces memory usage by tiling computation on GPU SRAM.",
        ]
        self.collection = build_collection(chunks, collection_name="test_retriever")

    def test_returns_list_of_dicts(self):
        results = self.retrieve("KV-cache compression", self.collection, k=2)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_each_result_has_required_keys(self):
        results = self.retrieve("RoPE mismatch", self.collection, k=2)
        for r in results:
            assert "text" in r, "Missing 'text' key"
            assert "score" in r, "Missing 'score' key"

    def test_returns_at_most_k_results(self):
        results = self.retrieve("attention", self.collection, k=3)
        assert len(results) <= 3

    def test_scores_are_numeric(self):
        results = self.retrieve("latent attention", self.collection, k=2)
        for r in results:
            assert isinstance(r["score"], (int, float)), f"Score is not numeric: {r['score']}"

    def test_most_relevant_result_mentions_query_topic(self):
        results = self.retrieve("RoPE positional encoding", self.collection, k=1)
        assert len(results) == 1
        # The RoPE chunk should be retrieved — it's the most semantically relevant
        assert "RoPE" in results[0]["text"] or "positional" in results[0]["text"].lower()

    def test_text_field_is_nonempty_string(self):
        results = self.retrieve("transformer", self.collection, k=2)
        for r in results:
            assert isinstance(r["text"], str) and r["text"].strip()


# ---------------------------------------------------------------------------
# rag.llm — KV-cache formula (the core differentiator)
# ---------------------------------------------------------------------------

class TestKvCacheFormula:
    """
    Formula: kv_mb = 2 * layers * kv_heads * head_dim * tokens * 2 / (1024 * 1024)
    Validated in prototype: Llama 3.1 8B, 1289 tokens → 161.1 MB
      2 * 32 * 8 * 128 * 1289 * 2 / 1048576 = 161.125 MB
    """

    def setup_method(self):
        from rag.llm import compute_kv_cache_mb
        self.compute_kv_cache_mb = compute_kv_cache_mb

    def test_llama_31_8b_at_1289_tokens(self):
        # This value was validated against the prototype formula — must match exactly
        result = self.compute_kv_cache_mb(**_LLAMA_8B, tokens=1289)
        expected = (2 * 32 * 8 * 128 * 1289 * 2) / (1024 * 1024)  # 161.125 MB
        assert math.isclose(result, expected, rel_tol=1e-6), (
            f"KV-cache formula wrong: got {result:.3f} MB, expected {expected:.3f} MB"
        )

    def test_zero_tokens_returns_zero(self):
        result = self.compute_kv_cache_mb(**_LLAMA_8B, tokens=0)
        assert result == 0.0

    def test_result_scales_linearly_with_tokens(self):
        base = self.compute_kv_cache_mb(**_LLAMA_8B, tokens=100)
        double = self.compute_kv_cache_mb(**_LLAMA_8B, tokens=200)
        assert math.isclose(double, 2 * base, rel_tol=1e-9)

    def test_llama_33_70b_larger_than_8b(self):
        result_70b = self.compute_kv_cache_mb(layers=80, kv_heads=8, head_dim=128, tokens=1000)
        result_8b = self.compute_kv_cache_mb(**_LLAMA_8B, tokens=1000)
        assert result_70b > result_8b

    def test_returns_float(self):
        result = self.compute_kv_cache_mb(**_LLAMA_8B, tokens=500)
        assert isinstance(result, float)


class TestInferenceMetrics:
    def setup_method(self):
        from rag.llm import build_inference_metrics
        self.build_inference_metrics = build_inference_metrics

    def _make_mock_response(self, prompt_tokens=50, completion_tokens=120):
        resp = MagicMock()
        resp.usage.prompt_tokens = prompt_tokens
        resp.usage.completion_tokens = completion_tokens
        resp.usage.completion_time = None  # prevent MagicMock auto-attribute from fooling isinstance check
        return resp

    def test_returns_dict_with_required_keys(self):
        resp = self._make_mock_response()
        metrics = self.build_inference_metrics(
            response=resp,
            ttft_ms=145.0,
            total_ms=360.0,
            **_LLAMA_8B,
        )
        required = {"ttft_ms", "total_ms", "tokens_per_sec", "kv_cache_mb",
                    "prompt_tokens", "completion_tokens", "total_tokens"}
        assert required.issubset(metrics.keys()), (
            f"Missing keys: {required - metrics.keys()}"
        )

    def test_tokens_per_sec_is_positive(self):
        resp = self._make_mock_response(completion_tokens=100)
        metrics = self.build_inference_metrics(
            response=resp, ttft_ms=100.0, total_ms=500.0,
            **_LLAMA_8B,
        )
        assert metrics["tokens_per_sec"] > 0

    def test_kv_cache_mb_uses_total_tokens(self):
        resp = self._make_mock_response(prompt_tokens=50, completion_tokens=50)
        metrics = self.build_inference_metrics(
            response=resp, ttft_ms=100.0, total_ms=400.0,
            **_LLAMA_8B,
        )
        expected_kv = (2 * 32 * 8 * 128 * 100 * 2) / (1024 * 1024)
        assert math.isclose(metrics["kv_cache_mb"], expected_kv, rel_tol=1e-6)

    def test_total_tokens_is_sum(self):
        resp = self._make_mock_response(prompt_tokens=40, completion_tokens=80)
        metrics = self.build_inference_metrics(
            response=resp, ttft_ms=100.0, total_ms=300.0,
            **_LLAMA_8B,
        )
        assert metrics["total_tokens"] == 120
        assert metrics["prompt_tokens"] == 40
        assert metrics["completion_tokens"] == 80

    def test_ttft_and_total_ms_are_preserved(self):
        resp = self._make_mock_response()
        metrics = self.build_inference_metrics(
            response=resp, ttft_ms=123.4, total_ms=567.8,
            **_LLAMA_8B,
        )
        assert math.isclose(metrics["ttft_ms"], 123.4, rel_tol=1e-6)
        assert math.isclose(metrics["total_ms"], 567.8, rel_tol=1e-6)


class TestBuildLlm:
    def setup_method(self):
        from rag.llm import build_llm
        self.build_llm = build_llm

    def test_missing_api_key_raises_value_error(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                self.build_llm("llama-3.1-8b-instant")

    def test_returns_callable_with_api_key_set(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_fake_key_for_unit_test"}):
            llm = self.build_llm("llama-3.1-8b-instant")
            # Must be callable (ChatGroq or a wrapper) — not None, not a string
            assert llm is not None
            assert callable(llm) or hasattr(llm, "invoke"), (
                "build_llm must return a callable or LangChain LLM with .invoke()"
            )


# ---------------------------------------------------------------------------
# Integration — processor → retriever pipeline (no Groq call)
# ---------------------------------------------------------------------------

class TestProcessorRetrieverIntegration:
    def test_full_index_and_retrieve_pipeline(self):
        from rag.processor import chunk_text, build_collection
        from rag.retriever import retrieve

        paper_text = textwrap.dedent("""
            TransMLA: Multi-head Latent Attention for Large Language Models

            Abstract: We present TransMLA, a method for converting standard multi-head attention
            (MHA) to multi-head latent attention (MLA), enabling significant KV-cache compression.

            Section 3.2 — RoPE Compatibility:
            A key challenge in implementing TransMLA is the RoPE format mismatch. The original
            MLA architecture applies RoPE to only a portion of each head's key dimension. When
            converting from MHA, the RoPE rotary embeddings must be reformatted to avoid incorrect
            positional encodings that degrade model quality.

            Section 4 — Experiments:
            We evaluate TransMLA on SmolLM-135M and Qwen3-4B. The results show a 16x reduction
            in KV-cache memory footprint with less than 0.5% degradation on MMLU benchmarks.
        """).strip()

        chunks = chunk_text(paper_text, chunk_size=200, overlap=20)
        assert len(chunks) > 0, "Chunking produced no output"

        collection = build_collection(chunks, collection_name="integration_test")
        assert collection.count() > 0

        results = retrieve("RoPE format mismatch positional encoding", collection, k=2)
        assert len(results) >= 1

        top_text = results[0]["text"].lower()
        assert "rope" in top_text or "positional" in top_text or "mismatch" in top_text, (
            f"Top result doesn't mention RoPE context: {results[0]['text'][:100]}"
        )

    def test_file_size_warning_for_large_input(self):
        from rag.processor import chunk_text
        # 21MB of text should trigger a warning (not an exception) — checked via return or log
        # We just verify it doesn't crash; the warning behavior is tested manually
        large_text = "word " * 4_000_000  # ~20MB
        try:
            chunks = chunk_text(large_text)
            assert isinstance(chunks, list)
        except MemoryError:
            pytest.skip("System ran out of memory — environment constraint, not a code bug")
