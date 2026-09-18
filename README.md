# 🤖 Local Chatbot — CLI

A fully local AI chatbot run from the terminal. It talks to a local Ollama
model through a **balanced mode** that routes trivial questions to a fast
model, hard reasoning to a strong reasoning model, caches repeat questions,
and time-boxes slow generations so a request never hangs. No API keys, no
cloud, no GPU required.

---

## Quick start

### 1. Install Ollama + a model pair

Download Ollama from https://ollama.com, then pull the recommended pair:

```bash
ollama pull qwen3:4b    # strong small reasoning model (for hard questions)
ollama pull phi3:mini   # fast model for trivia (auto-detected)
```

Only pulling one model is fine too — simple chat will just use that model.

### 2. Install dependencies

```bash
cd local-chatbot
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
```

In `.env` ensure:

```
DEFAULT_MODEL=ollama/qwen3:4b
PERFORMANCE_MODE=balanced
```

That is all — balanced mode is the default and needs nothing else.

---

## CLI usage

```
python main.py <command>

Commands:
  chat       Ask a single question (terminal)
  status     Show config and backend health
  benchmark  Benchmark hardware and get model recommendations
```

### Chat

```bash
python main.py chat "How many days are in a week?"
python main.py chat "Prove that sqrt(2) is irrational"
python main.py chat --model qwen3:4b "Why is the sky blue?"   # use a specific model
python main.py chat --use-agent "Tell me about LangGraph"     # LangGraph agent (tools/RAG)
python main.py chat --no-rag "..."                            # skip RAG retrieval
```

Each `chat` call runs the balanced gateway: trivial questions answer in
seconds on a fast model, hard reasoning is time-boxed to ~90s, and the answer
is the model's real final reply.

### Status

```bash
python main.py status
```

### Benchmark

```bash
python main.py benchmark                        # CPU recommendations
python main.py benchmark --gpu "RTX 4090"        # simulate a GPU
python main.py benchmark --gpu "RTX 4090" --top 3 --speed fast
python main.py benchmark --json                  # JSON output
```

Run `python main.py --help` for the full list of options.

---

## How balanced mode answers questions

Every question makes ONE adaptive call to the local model (no fallback
chains), classified and routed on the fly:

| Query type | Example | Model | Latency |
|---|---|---|---|
| Trivial / factual | "How many days are in a week?" | fast "simple" model | ~5-10s |
| Reasoning | "Prove sqrt(2) is irrational" | selected reasoning model, time-boxed | max ~90s |
| Repeat | any question you already asked | answer cache | ~0s |

Model routing tiers (the first **installed** candidate of the right tier is
used):

| Tier | Models |
|---|---|
| `simple` | `phi3:mini`, `gemma2:2b`, `qwen2.5:0.5b`, `tinyllama` |
| `medium` | `phi3:3.8b`, `qwen2.5:3b`, `gemma2:9b` |
| `complex` (selected) | `qwen2.5:7b`, `qwen2.5:14b`, `llama3.2`, `deepseek-llm:7b` |

Hard reasoning is streamed and cut off after ~90s of wall-clock instead of
hanging; a real answer cut mid-sentence gets one short continuation pass, and
a model that only produced reasoning gets returned as-is. When a call fails it
degrades gracefully — optimized → plain retry → raw generator → a neutral
`"I couldn't finish, please retry"`. Answers are always the model's real final
`response`, not its hidden reasoning text.

---

## Tests

```bash
pytest
```

---

## Licence

MIT