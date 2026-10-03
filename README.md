# Company Policy AI Agent (RAG Chatbot)
Speaker's Linkedin : https://www.linkedin.com/in/khushiiiyadav/

A Streamlit chatbot that answers employee questions about company policies and their leave balance. It combines **RAG** (retrieval-augmented generation) with **tool calling**, orchestrated by **LangGraph**, using a **Groq**-hosted LLM.

Ask things like:
- "What is the leave carry-forward limit?"
- "Can I work from home?"
- "How many leaves do I have left, and what is the leave policy?"
- 📎 Attach a screenshot of a leave request and ask "Is this leave request valid?"

## How it works

```
User question (+ optional image)
     │
     ▼
┌──────────┐
│  vision  │  only if an image is attached: Groq vision model
│          │  (qwen/qwen3.8-27b) extracts text, dates, numbers
└──────────┘
     │
     ▼
┌──────────┐   tool call?   ┌───────────────────────────────┐
│  agent   │ ─────────────▶ │  tools                        │
│ (Groq    │                │  • search_company_policies    │
│  LLM)    │ ◀───────────── │    (FAISS vector search)      │
└──────────┘    results     │  • get_leave_balance          │
     │                      └───────────────────────────────┘
     ▼ no more tool calls
 Final answer + sources
```

1. **Documents**: policy files in `data/` (leave policy and workplace policy for a fictional "ACME Corporation") are loaded and split into chunks.
2. **Embeddings**: chunks are embedded locally with `sentence-transformers/all-MiniLM-L6-v2` (no API key needed) and stored in an in-memory **FAISS** vector store.
3. **Tools**:
   - `search_company_policies`: retrieves the 3 most relevant policy chunks.
   - `get_leave_balance`: returns the current employee's leave balance (demo data: 14 days).
4. **Vision (optional)**: if the user attaches an image, a vision node uses `qwen/qwen3.8-27b` on Groq to extract its details as text. The agent then checks them against policy with the same tools.
5. **Agent loop (LangGraph)**: the LLM (`openai/gpt-oss-120b` on Groq) decides which tools to call, reads the results, and loops until it can answer.
6. **UI**: Streamlit shows the answer, the tools used, and the source documents.
7. **Tracing (optional)**: every run can be traced in **LangSmith**.

## Tech stack

| Part | Tool |
|---|---|
| UI | Streamlit |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Vision | Groq (`qwen/qwen3.8-27b`) |
| Agent orchestration | LangGraph |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (runs locally) |
| Vector store | FAISS |
| Observability | LangSmith |

## How to run

### 1. Prerequisites
- Python 3.9+
- A free Groq API key from https://console.groq.com/keys
- (Optional) A LangSmith API key from https://smith.langchain.com for tracing

### 2. Clone and install

```bash
git clone https://github.com/Khushi-ydv/Rag_Chatbot.git
cd Rag_Chatbot

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Add your API keys

Create a file named `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key

# Optional: LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=rag-chatbot
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

If `LANGSMITH_API_KEY` is empty, tracing is turned off automatically.

### 4. Start the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. The first start takes a little longer because the embedding model is downloaded.

## Viewing traces in LangSmith

With tracing enabled, open the **rag-chatbot** project at https://smith.langchain.com. Each question is recorded as a `rag-chatbot` run (tagged `streamlit`, `rag`). It shows the agent steps, LLM calls, tool calls, and retrieved documents.

## Project structure

```
Rag_Chatbot/
├── app.py              # Streamlit app: documents, vector store, tools, LangGraph agent, UI
├── data/
│   ├── leave_policy.txt
│   └── workplace_policy.txt
├── requirements.txt
└── .env                # your API keys (not committed)
```

> **Note:** `app.py` regenerates the two demo policy files in `data/` on every start. To use your own documents, remove the demo-file section in `app.py` (section 2) and put your `.txt` files in `data/`.
