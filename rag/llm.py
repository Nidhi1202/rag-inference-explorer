import os
import time

from groq import Groq

MODEL_CONFIGS: dict[str, dict] = {
    "llama-3.1-8b-instant": {"layers": 32, "kv_heads": 8, "head_dim": 128, "max_ctx": 131072},
    "llama-3.3-70b-versatile": {"layers": 80, "kv_heads": 8, "head_dim": 128, "max_ctx": 131072},
    "mixtral-8x7b-32768": {"layers": 32, "kv_heads": 8, "head_dim": 128, "max_ctx": 32768},
}


def compute_kv_cache_mb(layers: int, kv_heads: int, head_dim: int, tokens: int) -> float:
    """KV-cache size in MB: 2 × L × H × D × T × 2 bytes / 1024²"""
    return (2 * layers * kv_heads * head_dim * tokens * 2) / (1024 * 1024)


def build_inference_metrics(
    response,
    ttft_ms: float,
    total_ms: float,
    layers: int,
    kv_heads: int,
    head_dim: int,
    **kwargs,
) -> dict:
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = prompt_tokens + completion_tokens

    # Prefer Groq's server-side completion time for accurate tok/s
    completion_time_s = getattr(usage, "completion_time", None)
    if completion_time_s and isinstance(completion_time_s, (int, float)) and completion_time_s > 0:
        tokens_per_sec = round(completion_tokens / completion_time_s, 1)
    else:
        gen_ms = max(total_ms - ttft_ms, 1.0)
        tokens_per_sec = round(completion_tokens / (gen_ms / 1000), 1)

    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "tokens_per_sec": tokens_per_sec,
        "kv_cache_mb": compute_kv_cache_mb(layers, kv_heads, head_dim, total_tokens),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def build_llm(model_name: str):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Add it to .env locally or via HF Spaces → Settings → Secrets."
        )
    client = Groq(api_key=api_key)

    def invoke(messages: list[dict], **kw):
        return client.chat.completions.create(model=model_name, messages=messages, **kw)

    return invoke


def chat_with_context(
    question: str,
    context_chunks: list[dict],
    model_name: str,
    llm_fn,
) -> tuple[str, dict]:
    cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["llama-3.1-8b-instant"])

    context = "\n\n".join(
        f"[{i + 1}] {chunk['text']}" for i, chunk in enumerate(context_chunks)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer questions using the provided context passages. "
                "Cite sources with [1], [2], etc. matching the numbered passages. "
                "If you cannot answer from the context, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    t0 = time.perf_counter()
    response = llm_fn(messages=messages)
    total_ms = (time.perf_counter() - t0) * 1000

    # Use Groq's server-side total_time if available for a cleaner TTFT estimate
    usage = response.usage
    server_total_s = getattr(usage, "total_time", None)
    server_completion_s = getattr(usage, "completion_time", None)

    if (
        isinstance(server_total_s, (int, float))
        and isinstance(server_completion_s, (int, float))
        and server_total_s > 0
    ):
        ttft_ms = (server_total_s - server_completion_s) * 1000
        total_ms = server_total_s * 1000
    else:
        ttft_ms = total_ms  # non-streaming fallback

    content = response.choices[0].message.content
    metrics = build_inference_metrics(
        response=response,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        layers=cfg["layers"],
        kv_heads=cfg["kv_heads"],
        head_dim=cfg["head_dim"],
    )

    return content, metrics
