import gradio as gr
from dotenv import load_dotenv

from rag.llm import MODEL_CONFIGS, build_llm, chat_with_context
from rag.processor import build_collection, chunk_text, extract_text
from rag.retriever import retrieve

load_dotenv()

MODELS = list(MODEL_CONFIGS.keys())


def _format_inference_panel(metrics: dict | None) -> str:
    if metrics is None:
        return "_Upload a paper and ask a question to see live inference metrics._"

    model = metrics.get("model", MODELS[0])
    cfg = MODEL_CONFIGS.get(model, MODEL_CONFIGS[MODELS[0]])
    max_ctx = cfg["max_ctx"]
    ctx_pct = round(metrics["total_tokens"] / max_ctx * 100, 2)

    return (
        "### ⚡ Inference Insights\n\n"
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| **TTFT** | {metrics['ttft_ms']:.0f} ms |\n"
        f"| **Total latency** | {metrics['total_ms']:.0f} ms |\n"
        f"| **Tokens / sec** | {metrics['tokens_per_sec']:.0f} |\n"
        f"| **KV-cache size** | {metrics['kv_cache_mb']:.1f} MB |\n"
        f"| **Prompt tokens** | {metrics['prompt_tokens']:,} |\n"
        f"| **Completion tokens** | {metrics['completion_tokens']:,} |\n"
        f"| **Context used** | {ctx_pct}% of {max_ctx:,} tokens |\n\n"
        "<details><summary>How is KV-cache size calculated?</summary>\n\n"
        f"`2 × L × H × D × T × 2 bytes / 1024²`  \n"
        f"L={cfg['layers']} layers · H={cfg['kv_heads']} KV heads · "
        f"D={cfg['head_dim']} head dim · T={metrics['total_tokens']} tokens\n\n"
        "</details>"
    )


def process_paper(file, model_name):
    if file is None:
        return None, "⚠️ Please upload a PDF first.", _format_inference_panel(None)

    try:
        text = extract_text(file)

        if len(text.encode("utf-8")) > 20 * 1024 * 1024:
            return None, "⚠️ File exceeds 20 MB of text — upload a smaller PDF.", _format_inference_panel(None)

        chunks = chunk_text(text)
        if not chunks:
            return None, "⚠️ No text could be extracted. Try a text-based PDF (not a scan).", _format_inference_panel(None)

        collection = build_collection(chunks, collection_name="paper_session")
        status = f"✅ Ready — indexed **{collection.count()} chunks** ({len(text):,} chars). Ask a question below."
        return collection, status, _format_inference_panel(None)

    except ValueError as e:
        return None, f"⚠️ {e}", _format_inference_panel(None)
    except Exception as e:
        return None, f"❌ Error processing PDF: {e}", _format_inference_panel(None)


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def respond(message, history, collection, model_name):
    if not message.strip():
        return history, _format_inference_panel(None)

    if collection is None:
        return (
            list(history) + [_msg("user", message), _msg("assistant", "⚠️ Please upload and process a paper first.")],
            _format_inference_panel(None),
        )

    try:
        llm_fn = build_llm(model_name)
    except ValueError as e:
        return (
            list(history) + [_msg("user", message), _msg("assistant", f"⚠️ {e}")],
            _format_inference_panel(None),
        )

    try:
        chunks = retrieve(message, collection, k=4)
        answer, metrics = chat_with_context(message, chunks, model_name, llm_fn)
        metrics["model"] = model_name

        if chunks:
            sources = "\n\n---\n**Sources retrieved:**\n" + "\n".join(
                f"[{i + 1}] _{chunk['text'][:120].strip()}…_ (score: {chunk['score']:.2f})"
                for i, chunk in enumerate(chunks[:3])
            )
            answer = answer + sources

        return (
            list(history) + [_msg("user", message), _msg("assistant", answer)],
            _format_inference_panel(metrics),
        )

    except Exception as e:
        return (
            list(history) + [_msg("user", message), _msg("assistant", f"❌ Error: {e}")],
            _format_inference_panel(None),
        )


CUSTOM_CSS = """
body { font-family: 'Inter', sans-serif; }
#inference-panel { border-radius: 8px; padding: 4px 12px; }
.suggestion-row { gap: 8px; flex-wrap: wrap; }
"""

SUGGESTIONS = [
    "What problem does this paper solve?",
    "How does KV-cache compression work?",
    "What is the RoPE format issue?",
    "What are the benchmark results?",
]

with gr.Blocks(title="RAG Inference Explorer") as demo:
    collection_state = gr.State(None)

    gr.Markdown(
        "# RAG Inference Explorer\n"
        "*Upload a research paper → ask questions → see live LLM inference metrics.*"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📄 Paper Setup")
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            model_radio = gr.Radio(choices=MODELS, value=MODELS[0], label="Model")
            process_btn = gr.Button("⚙️ Process Paper", variant="primary")
            status_md = gr.Markdown("_Upload a PDF to get started._")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=420, label="Chat")
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask about your paper…",
                    label="",
                    scale=5,
                    show_label=False,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            inference_md = gr.Markdown(
                _format_inference_panel(None),
                elem_id="inference-panel",
            )

    with gr.Row(elem_classes="suggestion-row"):
        for label in SUGGESTIONS:
            chip = gr.Button(label, size="sm", variant="secondary")
            chip.click(fn=lambda q=label: q, outputs=msg_input)

    # Wire up events
    process_btn.click(
        fn=process_paper,
        inputs=[file_input, model_radio],
        outputs=[collection_state, status_md, inference_md],
    )

    submit_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot, collection_state, model_radio],
        outputs=[chatbot, inference_md],
    ).then(fn=lambda: "", outputs=msg_input)

    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot, collection_state, model_radio],
        outputs=[chatbot, inference_md],
    ).then(fn=lambda: "", outputs=msg_input)


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Monochrome(primary_hue="violet"),
        css=CUSTOM_CSS,
    )
