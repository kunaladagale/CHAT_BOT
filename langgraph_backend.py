"""
langgraph_backend.py
====================
The agent graph.

    START ──► chat_node ──(tools_condition)──► tools ──┐
                  │                                     │
                  │ no tool calls                       │  result fed back
                  ▼                                     │
                 END  ◄───────────────────────────────  ┘

chat_node asks the LLM. The LLM either answers in plain text (-> END) or emits
one or more tool_calls (-> tools). The ToolNode runs them, appends a ToolMessage
for each, and loops back into chat_node so the model can read the results and
either call more tools or write the final answer. That loop is the whole
difference between a chatbot and an agent.
"""

from dotenv import load_dotenv
load_dotenv()

import sqlite3
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from chat_db import DB_PATH, init_chats_table   # share the same SQLite file
from tools import TOOLS                          # <-- all tools live in tools.py

# --- LangSmith @traceable, with a no-op fallback ---------------------------
try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*d_args, **d_kwargs):          # type: ignore[misc]
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        return lambda fn: fn


# ===========================================================================
# State
# ===========================================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ===========================================================================
# Model + tool binding
# ===========================================================================
# bind_tools() is the line that makes the model tool-aware: it sends the JSON
# schema of every tool (built from the type hints + docstrings in tools.py)
# along with each request, so the model can reply with tool_calls instead of
# text. Without it the graph would loop forever doing nothing.
#
# NOTE: the model must actually support tool calling. qwen3 does. If you swap
# to a model that doesn't (e.g. plain llama2), tool_calls will always be empty.
llm = ChatOllama(
    model="qwen3:8b",
    reasoning=False,      # clean, direct answers (no thinking monologue)
    keep_alive="30m",     # keep the model warm for fast replies
    temperature=0,        # deterministic tool-argument extraction
)

llm_with_tools = llm.bind_tools(TOOLS)


SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant with access to tools.\n"
    "\n"
    "Rules:\n"
    "- For ANY arithmetic, use the `calculator` tool. Never do mental math.\n"
    "- For ANY weather question, use `get_weather`. Never answer from memory.\n"
    "- For ANY share price, index level or crypto price, use `get_stock_price`. "
    "Never answer from memory.\n"
    "- You may call several tools in a row: read each result, then decide "
    "whether you need another tool or are ready to answer.\n"
    "- If a tool returns a line starting with 'Error:', do not invent the "
    "answer. Either retry with corrected arguments (e.g. a different ticker "
    "suffix) or tell the user plainly what went wrong.\n"
    "- Once you have what you need, reply in clear natural language and state "
    "the units and the as-of time for any live data."
))


# ===========================================================================
# Nodes
# ===========================================================================
@traceable(run_type="chain", name="chat_node_reasoning")
def _decide(messages):
    """The actual LLM call, as its own named LangSmith span.

    Kept separate from the node function because @traceable injects a
    `config=None` kwarg into the signature it exposes, and LangGraph inspects
    node signatures to decide what to pass in. Wrapping the body instead of the
    node keeps both happy.
    """
    return llm_with_tools.invoke([SYSTEM_PROMPT] + list(messages))


def chat_node(state: ChatState):
    """Ask the LLM. It returns either a normal answer or a set of tool_calls.

    The system prompt is prepended at call time rather than stored in state,
    so it never gets checkpointed and duplicated on every turn.
    """
    response = _decide(state["messages"])
    return {"messages": [response]}


# ToolNode is the prebuilt executor: it reads the tool_calls off the last
# AIMessage, runs each matching function from TOOLS, and appends one
# ToolMessage per call (carrying the result and the tool_call_id).
# handle_tool_errors=True means an exception becomes a readable ToolMessage
# instead of crashing the graph -- a second safety net on top of the
# try/except blocks inside tools.py.
tool_node = ToolNode(TOOLS, handle_tool_errors=True)


# ===========================================================================
# Persistence
# ===========================================================================
# PERSISTENT memory in a local SQLite file (no server, no password).
# SqliteSaver saves every turn keyed by thread_id and, on the next run,
# feeds the whole history back to the LLM automatically.
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_conn)
checkpointer.setup()          # creates checkpoint tables the first time (idempotent)
init_chats_table()            # creates our chat-metadata table the first time


# ===========================================================================
# Graph wiring
# ===========================================================================
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

# tools_condition inspects the last message:
#   has .tool_calls  -> returns "tools"
#   otherwise        -> returns END
graph.add_conditional_edges(
    "chat_node",
    tools_condition,
    {"tools": "tools", END: END},
)

# after the tools run, ALWAYS go back to the LLM with the results -> the loop
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# ===========================================================================
# Handy for the UI / debugging
# ===========================================================================
TOOL_NAMES = [t.name for t in TOOLS]

if __name__ == "__main__":
    # Print the graph so you can eyeball the wiring:  python langgraph_backend.py
    try:
        print(chatbot.get_graph().draw_ascii())
    except Exception:  # noqa: BLE001 -- draw_ascii needs the `grandalf` package
        print("Nodes:", list(chatbot.get_graph().nodes))
        print("Tools:", TOOL_NAMES)