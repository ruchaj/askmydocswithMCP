import asyncio
import json
import os
from pathlib import Path
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).parent / ".env")

load_dotenv(Path(__file__).with_name(".env"))

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "No Anthropic API key found. Add a `.env` file next to askmydocs_core.py "
        "containing `ANTHROPIC_API_KEY=sk-ant-...`, or set the environment variable."
    )

MODEL = os.getenv("MODEL", "claude-sonnet-4-6")
if not MODEL:
    raise RuntimeError(
        "No model specified. Add a `.env` file next to askmydocs_core.py "
        "containing `MODEL=your-model-name`, or set the environment variable."
    )
MAX_ROUNDS = 3
SERVER_PATH = str(Path(__file__).with_name("askmydocs_mcp_server.py"))

class AgentState(TypedDict):
    question: str
    doc_set_id: str
    queries: list[str]
    observations: list[str]
    rounds: int
    verdict: str
    answer: str
    trace: Annotated[list[str], lambda a, b: a + b]

def _llm():
    return ChatAnthropic(api_key=API_KEY, model=MODEL)
def _json_list(text: str, fallback: list[str]) -> list[str]:
    try:
        start, end = text.find("["), text.rfind("]") + 1
        parsed = json.loads(text[start:end])
        return [str(q) for q in parsed][:4] or fallback
    except Exception as E:
        return fallback

async def plan(state: AgentState) -> dict:
    prompt = (
            "You are planning retrieval over a set of uploaded documents.\n"
    f"Question: {state['question']}\n\n"
    "Write 2 to 4 search queries that together would surface the evidence "
    "needed to answer it. Vary the phrasing: documents rarely use the same "
    "words as the question. Reply with a JSON array of strings and nothing else."
        )
    resp  = await AgentState._llm().achat(prompt)
    queries = AgentState._json_list(resp.text, fallback=[])
    return {
        "queries": queries,
        "rounds": 0,
        "observations": [],
        "trace": [f"Planned queries: {queries}"]
    }
async def act(state: AgentState, tools: dict) -> dict:
    search = tools["search_documents"]
    found = []
    for query in state["queries"]:
        results = await search.ainvoke({"query": query, "doc_set_id": state["doc_set_id"], "k": 3})
        found.append(f"Query: {query}, Results: {results}")
    return {
        "observations": state["observations"] + found,
        "rounds": state["rounds"] + 1,
        "trace": state["trace"] + [f"Found {len(found)} documents for queries: {state['queries']} via MCP"],
    }

async def reflect(state: AgentState) -> dict:
    evidence = "\n".join(state["observations"])
    prompt = (
        "You are reflecting on the evidence gathered for a question.\n"
        f"Question: {state['question']}\n"
        f"Evidence:\n{evidence}\n\n"
        "Can this question be answered from the evidence, citing it?\n"
        "If yes, replyexactly: {\"verdict\": \"sufficient\", \"answer\": \"<your answer>\"}"
        "If no, reply exactly: {\"verdict\": \"insufficient\", \"answer\": \"<your explanation>\"}"
        "with your explanation targeting only what is still missing from the evidence."
        "Do not repeat queries that already returned what you need from the evidence."
        "Ensure your response is a valid JSON object and nothing else."
    )
    resp = await _llm().ainvoke(prompt)
    text = resp.content
    verdict = "sufficient" if '"verdict": "sufficient"' in text else "insufficient"
    queries = _json_list(text, fallback=[]) if verdict == "insufficient" else []
    return {
        "verdict": verdict,
        "queries": queries or state["queries"],
        "trace": [f"reflect: {verdict}" + (f", next {queries}" if queries else "")],
    }

async def compose(state: AgentState) -> dict:
    prompt = (
        "You are generating a final answer based on the gathered evidence.\n"
        f"Question: {state['question']}\n"
        f"Evidence:\n{'\n'.join(state['observations'])}\n\n"
        "Provide a concise and accurate answer citing the evidence. "
        "Reply with a JSON object: {\"verdict\": \"sufficient\", \"answer\": \"<your answer>\"}"
    )
    resp = await _llm().ainvoke(prompt)
    return {"answer": resp.content, "trace": [f"compose: grounded answer written"]}

def route(state: AgentState) -> str:
    if state["verdict"] == "sufficient" or state["rounds"] >= MAX_ROUNDS:
        return "compose"
    return "act"

def build_graph(tools: dict):
    g = StateGraph(AgentState)
    g.add_node("plan", plan)
    g.add_node("act", lambda state: act(state, tools))
    g.add_node("reflect", reflect)
    g.add_node("compose", compose)

    g.add_edge(START, "plan")
    g.add_edge("plan", "act")
    g.add_edge("act", "reflect")
    g.add_conditional_edges("reflect", route, {"act": "act", "compose": "compose"}),
    g.add_edge("compose", END)
    return g.compile()