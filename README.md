cat > README.md << 'EOF'
---
title: LLM Arbitration System
emoji: ⚖
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# LLM Arbitration System

A multi-agent system that takes any LLM-generated output, routes it to three specialized critic agents running on different model architectures, detects disagreements between critics, and synthesizes a confidence-scored verdict with confirmed issues, dismissed flags, and inline annotations.

## Architecture

Three critic agents run in parallel via LangGraph fan-out, each evaluating a different dimension:

| Agent | Model | Dimension |
|---|---|---|
| Factual Accuracy Critic | Llama 3.3 70b (Groq) | Verifiable claims |
| Logical Consistency Critic | Qwen3 32b (Groq) | Reasoning validity |
| Completeness Critic | Llama 4 Scout (Groq) | Coverage gaps |
| Adjudicator | Llama 3.3 70b (Groq) | Conflict resolution |

The adjudicator receives all three reports plus detected disagreements and synthesizes a final verdict with an overall quality score, confidence level, confirmed issues, and dismissed flags.

## Stack

| Component | Tool |
|---|---|
| Agent framework | LangGraph |
| LLM inference | Groq API |
| Structured outputs | Pydantic + instructor |
| REST API | FastAPI + OpenAPI |
| UI | Streamlit |
| MCP tool server | mcp Python library |
| Local inference fallback | Ollama |
| Audit trail | SQLite |
| Experiment tracking | W&B |
| Deployment | Docker + HuggingFace Spaces |

## Verdict structure

Every arbitration produces:
- Overall quality score (1-10)
- Confidence level (low, medium, high)
- Confirmed issues with exact quote, severity, evidence, and which critics flagged it
- Dismissed flags with adjudicator reasoning
- One paragraph summary

## MCP tool server

The arbitration engine is exposed as an MCP-compatible tool server. Claude Desktop and Cursor can call it natively:

```json
{
  "mcpServers": {
    "llm-arbitration": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/mcp_server/server.py"]
    }
  }
}
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /v1/arbitrate | Arbitrate a single LLM output |
| POST | /v1/arbitrate/batch | Arbitrate multiple outputs |
| GET | /v1/arbitrations/{id} | Retrieve a past verdict |
| GET | /v1/analytics | Critic behavior stats |
| GET | /docs | Interactive OpenAPI documentation |

## Local setup

```bash
git clone https://github.com/sanjana231100/llm-arbitration
cd llm-arbitration
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add GROQ_API_KEY to .env

# terminal 1 - API
uvicorn api.main:app --reload --port 8000

# terminal 2 - UI
streamlit run ui/app.py

# terminal 3 - MCP server
python mcp_server/server.py
```

## Docker

```bash
docker compose up
```

## W&B analytics

Tracks across arbitrations:
- Critic score distributions
- Disagreement rates between critics
- Which critic flags the most issues
- Confidence level distribution over time
EOF
