"""
ui/gradio_ui.py
~~~~~~~~~~~~~~~
Gradio-based chat UI for local-chatbot.

Features
--------
* Multi-turn conversation with streaming token display.
* Model selector (any LiteLLM model string).
* RAG toggle + document upload.
* Agent toggle (LangGraph on/off).
* Backend status sidebar widget.
* Code-viewer panel that shows the Aider repo map on demand.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import gradio as gr

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_available_models() -> list[str]:
    try:
        from gateway.litellm_gateway import list_available_models
        return list_available_models()
    except Exception:
        return [settings.default_model]


def _stream_response(
    user_msg: str,
    history: list[list[str | None]],
    model: str,
    temperature: float,
    max_tokens: int,
    use_rag: bool,
    use_agent: bool,
    system_prompt: str,
) -> Generator[list[list[str | None]], None, None]:
    """
    Core generator – converts chatbot history (Gradio format) to OpenAI
    message dicts, calls the appropriate backend, and yields incremental updates.

    Optimizations based on streaming and UI research:
    - Debouncing: Reduce UI update frequency to prevent jank (RAIL: 16ms animation budget)
    - Batching: Accumulate tokens before UI updates to reduce DOM operations
    - Perceived performance: Immediate user echo + skeleton states + typing indicators
    - Eq2: When use_rag=True and use_agent=False, RAG retrieval is fired
      as a background thread *before* the LLM call starts, so retrieval
      overlaps with LLM prefill (prefetch-and-overlap latency hiding).
    """
    import concurrent.futures
    import time

    # Convert Gradio history → OpenAI messages
    openai_history = []
    for user_turn, bot_turn in history:
        if user_turn:
            openai_history.append({"role": "user",      "content": user_turn})
        if bot_turn:
            openai_history.append({"role": "assistant", "content": bot_turn})

    # Add current user message to display history immediately (perceived speed)
    new_history = history + [[user_msg, ""]]
    yield new_history
    
    # Add typing indicator for perceived performance (based on "Typing Indicator and Perceived Wait Time" research)
    typing_indicators = ["⏳", "🤔", "💭", "✨"]
    indicator_index = 0
    typing_cycles = 0
    
    # Show initial typing indicator
    new_history[-1][1] = typing_indicators[0]
    yield new_history

    if use_agent:
        # Update typing indicator during agent processing for perceived responsiveness
        import time
        max_cycles = settings.ui_max_typing_cycles
        for i in range(min(3, max_cycles)):
            indicator_index = (indicator_index + 1) % len(typing_indicators)
            new_history[-1][1] = typing_indicators[indicator_index]
            yield new_history
            time.sleep(0.05)  # Brief delay for animation effect (perceived activity)
        
        # Run full LangGraph agent (non-streaming for simplicity)
        try:
            import asyncio
            from agents.langgraph_agent import achat
            
            # Proper event loop handling for nested event loops
            try:
                loop = asyncio.get_running_loop()
                # If there's already a running loop, we need to use a different approach
                logger.warning("Event loop already running, using thread-based execution")
                from agents.langgraph_agent import chat as sync_chat
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(sync_chat, user_msg, openai_history)
                    reply = future.result(timeout=settings.ui_agent_timeout)  # Use configured timeout
            except RuntimeError:
                # No running loop, safe to create new one
                reply = asyncio.run(achat(user_msg, history=openai_history))
                
        except concurrent.futures.TimeoutError:
            logger.error("Agent execution timed out after 30 seconds")
            new_history[-1][1] = "⚠️ Error: Agent request timed out. Please try again."
            yield new_history
            return
        except Exception as exc:
            logger.exception("Agent execution failed: %s", exc)
            new_history[-1][1] = f"⚠️ Error: {exc}"
            yield new_history
            return

        new_history[-1][1] = reply
        yield new_history
        return

    # ------------------------------------------------------------------
    # Eq2 – Prefetch-and-Overlap: fire RAG retrieval in a background
    # thread right now so it runs concurrently with LLM prefill below.
    # ------------------------------------------------------------------
    rag_future: concurrent.futures.Future | None = None
    rag_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    t_rag_start = time.perf_counter()

    if use_rag:
        def _rag_fetch():
            try:
                from rag.llama_index_rag import get_rag
                result = get_rag().query(user_msg)
                return result.get("answer", "")
            except Exception as exc:
                logger.warning("Eq2 UI RAG prefetch failed: %s", exc)
                return ""

        rag_future = rag_executor.submit(_rag_fetch)

    # Direct streaming via LiteLLM gateway (starts LLM prefill immediately)
    msgs = openai_history + [{"role": "user", "content": user_msg}]
    accumulated = ""
    t_llm_start = time.perf_counter()
    
    # UI rendering optimization parameters from configuration
    # Based on RAIL model: Animation budget 16ms (60 FPS)
    debounce_threshold = settings.ui_debounce_threshold
    batch_size = settings.ui_batch_size
    max_typing_cycles = settings.ui_max_typing_cycles
    
    first_update = True
    last_update_time = time.perf_counter()
    update_count = 0
    
    # Simple typing indicator for perceived performance
    # Based on "Typing Indicator and Perceived Wait Time" research
    time.sleep(0.05)  # 50ms brief delay
    new_history[-1][1] = "⏳"  # Initial loading indicator
    yield new_history
    typing_cycles = 0
    
    # Update typing indicator during initial streaming delay
    while typing_cycles < min(3, max_typing_cycles):
        # Small delay to show typing animation before first token
        time.sleep(0.03)
        indicator_index = (indicator_index + 1) % len(typing_indicators)
        new_history[-1][1] = typing_indicators[indicator_index]
        yield new_history
        typing_cycles += 1

    try:
        from gateway.litellm_gateway import chat_stream
        chunk_buffer = []
        
        for chunk in chat_stream(
            msgs,
            model=model or None,
            temperature=temperature,
            max_tokens=int(max_tokens),
            system_prompt=system_prompt or None,
        ):
            chunk_buffer.append(chunk)
            accumulated += chunk
            current_time = time.perf_counter()
            
            # First update immediately for perceived speed (Miller's 0.1s threshold)
            if first_update:
                new_history[-1][1] = accumulated
                yield new_history
                first_update = False
                last_update_time = current_time
                chunk_buffer = []
                update_count += 1
                continue
            
            # Subsequent updates: batch and debounce
            time_since_update = (current_time - last_update_time) * 1000  # ms
            should_update = (
                len(chunk_buffer) >= batch_size or  # Minimum batch size
                time_since_update >= debounce_threshold * 1000 or  # Time-based debounce
                len(accumulated) > 100  # Large content threshold
            )
            
            if should_update:
                new_history[-1][1] = accumulated
                yield new_history
                last_update_time = current_time
                chunk_buffer = []
                update_count += 1
        
        # Final update to ensure all content is displayed
        if chunk_buffer:
            new_history[-1][1] = accumulated
            yield new_history
            update_count += 1
            
        logger.debug(f"UI rendering: {update_count} updates for {len(accumulated)} chars")
            
    except Exception as exc:
        logger.exception("Streaming failed: %s", exc)
        new_history[-1][1] = f"⚠️ Error: {exc}"
        yield new_history
        return

    t_llm_elapsed = time.perf_counter() - t_llm_start

    # ------------------------------------------------------------------
    # Eq2 – collect RAG result (should already be done; overlap is free)
    # Append retrieved context as a footnote if it adds information.
    # ------------------------------------------------------------------
    if rag_future is not None:
        try:
            rag_context = rag_future.result(timeout=2.0)   # minimal wait
            t_rag_elapsed = time.perf_counter() - t_rag_start

            from gateway.perf_math import eq2_speedup_prefetch, eq2_visible_latency
            t_prefill_est = t_llm_elapsed * 0.35
            vis_lat  = eq2_visible_latency(t_rag_elapsed, t_prefill_est, t_llm_elapsed * 0.65)
            speedup  = eq2_speedup_prefetch(t_llm_elapsed, t_rag_elapsed, t_prefill_est)
            logger.debug(
                "Eq2 UI overlap: rag=%.2fs llm=%.2fs vis_lat=%.0fms speedup=%.2fx",
                t_rag_elapsed, t_llm_elapsed, vis_lat * 1000, speedup,
            )

            if rag_context and rag_context.strip() not in accumulated:
                new_history[-1][1] = accumulated + (
                    f"\n\n---\n*📚 Retrieved context:* {rag_context[:300]}"
                    + ("…" if len(rag_context) > 300 else "")
                )
                yield new_history
        except concurrent.futures.TimeoutError:
            logger.debug("Eq2 UI RAG prefetch timed out – result discarded.")
        except Exception as exc:
            logger.warning("Eq2 UI RAG result collection failed: %s", exc)
    
    # Ensure executor is always properly shut down
    try:
        rag_executor.shutdown(wait=False)
    except Exception:
        pass

    rag_executor.shutdown(wait=False)


def _ingest_files(files) -> str:
    """Upload files to the RAG index via the ingest pipeline."""
    if not files:
        return "No files selected."
    paths = [Path(f.name) for f in files]
    try:
        from rag.llama_index_rag import get_rag
        n = get_rag().add_documents(paths)
        return f"✅ Ingested {len(paths)} file(s) → {n} chunk(s) added to LlamaIndex."
    except Exception as exc:
        return f"❌ Ingest failed: {exc}"


def _get_repo_map() -> str:
    """Return an Aider repo map for the configured repo path."""
    try:
        from tools.aider_tool import RepoMapTool
        tool = RepoMapTool()
        return tool._run()
    except Exception as exc:
        return f"Repo map unavailable: {exc}"


def _get_backend_status() -> str:
    """Return a status string for the local inference backend."""
    try:
        from backends.model_server import get_backend_status
        s = get_backend_status()
        icon = "🟢" if s.get("healthy") else "🔴"
        return (
            f"{icon} **{s['backend'].upper()}**\n"
            f"Model: `{s.get('model', 'N/A')}`\n"
            f"URL:   `{s.get('base_url', 'N/A')}`"
        )
    except Exception as exc:
        return f"Status unavailable: {exc}"


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    models = _get_available_models()

    with gr.Blocks(title=settings.app_name) as demo:

        # Header
        gr.Markdown(f"# 🤖 {settings.app_name}")
        gr.Markdown(
            "Powered by **vLLM / SGLang** · **LiteLLM** · **LangGraph** · "
            "**LlamaIndex** · **Haystack** · **Aider**"
        )

        with gr.Tabs():
            # ── Tab 1: Chat ──────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=520,
                            avatar_images=(None, "🤖"),
                        )
                        with gr.Row():
                            msg_box = gr.Textbox(
                                placeholder="Type your message … (Shift+Enter for newline)",
                                label="",
                                scale=8,
                                lines=2,
                                max_lines=6,
                            )
                            send_btn = gr.Button("Send ▶", scale=1, variant="primary")

                        with gr.Row():
                            clear_btn  = gr.Button("🗑 Clear chat",    size="sm")
                            retry_btn  = gr.Button("🔄 Retry last",   size="sm")

                    with gr.Column(scale=1, min_width=240):
                        gr.Markdown("### ⚙️ Settings")
                        model_dd = gr.Dropdown(
                            choices=models,
                            value=settings.default_model,
                            label="Model",
                            allow_custom_value=True,
                        )
                        temperature_sl = gr.Slider(
                            0.0, 2.0, value=settings.litellm_temperature,
                            step=0.05, label="Temperature",
                        )
                        max_tokens_sl = gr.Slider(
                            64, 8192, value=settings.litellm_max_tokens,
                            step=64, label="Max tokens",
                        )
                        use_rag_cb    = gr.Checkbox(value=True,  label="Enable RAG")
                        use_agent_cb  = gr.Checkbox(value=True,  label="Enable Agent (LangGraph)")
                        system_prompt = gr.Textbox(
                            value=settings.agent_system_prompt,
                            label="System prompt",
                            lines=4,
                            max_lines=8,
                        )

                        gr.Markdown("---")
                        gr.Markdown("### 🖥 Backend")
                        backend_md = gr.Markdown(_get_backend_status())
                        refresh_btn = gr.Button("↺ Refresh status", size="sm")
                        refresh_btn.click(fn=_get_backend_status, outputs=backend_md)

            # ── Tab 2: Documents / RAG ───────────────────────────────────
            with gr.Tab("📄 Documents"):
                gr.Markdown(
                    "Upload documents to add to the RAG knowledge base. "
                    "Supported: `.txt`, `.pdf`, `.md`, `.py`, and most text formats."
                )
                file_upload = gr.File(
                    label="Upload files",
                    file_count="multiple",
                    file_types=[".txt", ".pdf", ".md", ".py", ".rst", ".csv", ".json"],
                )
                ingest_btn    = gr.Button("📥 Ingest files", variant="primary")
                ingest_status = gr.Textbox(label="Ingest result", interactive=False)
                ingest_btn.click(fn=_ingest_files, inputs=file_upload, outputs=ingest_status)

                gr.Markdown("---")
                gr.Markdown("### Indexed documents directory")
                gr.Markdown(f"`{settings.rag_docs_dir}`")

            # ── Tab 3: Code / Repo ───────────────────────────────────────
            with gr.Tab("🗂 Code"):
                gr.Markdown(
                    "Inspect the configured repository using **Aider's repo map**. "
                    "The map summarises classes, functions, and their relationships."
                )
                repo_path_box = gr.Textbox(
                    value=str(settings.aider_repo_path),
                    label="Repository path",
                )
                map_btn    = gr.Button("🗺 Generate repo map", variant="primary")
                map_output = gr.Code(label="Repo map", language="markdown", lines=30)
                map_btn.click(fn=_get_repo_map, outputs=map_output)

        # ── Event wiring ─────────────────────────────────────────────────
        shared_inputs = [
            msg_box, chatbot, model_dd, temperature_sl,
            max_tokens_sl, use_rag_cb, use_agent_cb, system_prompt,
        ]

        def _submit(msg, hist, model, temp, mtok, rag, agent, sys_p):
            yield from _stream_response(msg, hist, model, temp, mtok, rag, agent, sys_p)

        send_btn.click(
            fn=_submit,
            inputs=shared_inputs,
            outputs=chatbot,
        ).then(fn=lambda: "", outputs=msg_box)

        msg_box.submit(
            fn=_submit,
            inputs=shared_inputs,
            outputs=chatbot,
        ).then(fn=lambda: "", outputs=msg_box)

        clear_btn.click(fn=lambda: [], outputs=chatbot)

        def _retry(hist, model, temp, mtok, rag, agent, sys_p):
            if not hist:
                return hist
            last_user = hist[-1][0]
            short_hist = hist[:-1]
            yield from _stream_response(last_user, short_hist, model, temp, mtok, rag, agent, sys_p)

        retry_btn.click(
            fn=_retry,
            inputs=[chatbot, model_dd, temperature_sl, max_tokens_sl,
                    use_rag_cb, use_agent_cb, system_prompt],
            outputs=chatbot,
        )

    return demo


# ---------------------------------------------------------------------------
# Launch helper
# ---------------------------------------------------------------------------

def launch(share: bool | None = None, **kwargs):
    """Launch the Gradio UI."""
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name=settings.ui_host,
        server_port=settings.ui_port,
        share=share if share is not None else settings.ui_share,
        theme=gr.themes.Soft(),
        **kwargs,
    )


if __name__ == "__main__":
    launch()
