---
title: RAG Inference Explorer
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
short_description: Ask questions about research papers with live LLM inference insights
---

# RAG Inference Explorer

Upload a research paper (PDF) → ask questions → get cited answers with live LLM inference metrics.

**Live demo:** [HF Spaces link — coming soon]

## What makes this different from a generic RAG chatbot

The **Inference Insights panel** shows real metrics after every query:

| Metric | What it shows |
|---|---|
| **TTFT** | Time to first token — Groq's speed advantage |
| **KV-cache size** | `2 × L × H × D × T × 2 bytes / 1024²` — computed from real token counts |
| **Context used** | % of the model's context window consumed |
| **Tokens/sec** | Generation throughput from Groq's server-side timing |

The KV-cache formula is the same one used in the [TransMLA paper](https://arxiv.org/abs/2502.07864) to motivate multi-head latent attention — this app exists to make that math interactive.

## Architecture

```
app.py                  # Gradio UI (HF Spaces entry point)
rag/
  processor.py          # PDF → text → chunks → ChromaDB (BAAI/bge-small-en-v1.5)
  retriever.py          # query → top-k chunks (cosine similarity)
  llm.py                # Groq API call + real inference metrics
```

## Local setup

```bash
git clone <repo-url>
cd rag-inference-explorer
pip install -r requirements.txt
cp .env.example .env      # add your GROQ_API_KEY
python app.py             # opens at http://localhost:7860
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Suggested demo questions

- "What problem does this paper solve?"
- "How does KV-cache compression work?"
- "What is the RoPE format issue?"
- "What are the benchmark results?"

## Tech stack

| Component | Technology |
|---|---|
| UI / Deployment | Gradio 4.x on Hugging Face Spaces |
| LLM | Groq API — llama-3.1-8b-instant / llama-3.3-70b-versatile / mixtral-8x7b |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers |
| Vector DB | ChromaDB in-memory (per session, no persistence) |
| PDF parsing | pymupdf |
| Chunking | LangChain RecursiveCharacterTextSplitter (512 chars, 64 overlap) |
