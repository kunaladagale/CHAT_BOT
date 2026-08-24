"""
langgraph_backend.py
====================
Agent graph, now with the email tool coming from an MCP SERVER instead of a
local @tool.

    START ─► chat_node ─┬─(email tool call)──► review_email ─(approve)─► tools ─► chat_node
                        │                            └─(reject/revise)──────────► chat_node
                        ├─(other tool call)───────────────────────► tools ─► chat_node
                        └─(no tool call)───────────────────────────────────► END

Local tools (calculator, weather, stock) still live in tools.py.
The email tool is loaded from mcp_email_server.py over MCP.

Human-in-the-loop: because the email tool now runs in a SEPARATE process, the
approval pause can't live inside it. Instead, `review_email` is a node in THIS
graph that interrupt()s for approval BEFORE the email tool is allowed to run.
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import sqlite3
import asyncio
import threading
from pathlib import Path
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command

from langchain_mcp_adapters.client import MultiServerMCPClient

from chat_db import DB_PATH, init_chats_table
from tools import TOOLS as LOCAL_TOOLS          # calculator, weather, stock, RAG (local)
from rag import RAG                             # to make the prompt document-aware

EMAIL_TOOL_NAME = "send_email"


# ===========================================================================
# 1. Load the email tool from the MCP server
# ===========================================================================
# We launch mcp_email_server.py as a subprocess and talk to it over stdio.
# get_tools() is async, so we run it once here at import with asyncio.run().
_server_path = str(Path(__file__).parent / "mcp_email_server.py")

mcp_client = MultiServerMCPClient({
    "email": {
        "command": sys.executable,      # same Python/venv we're running in
        "args": [_server_path],
        "transport": "stdio",
    }
})

_RAW_MCP_TOOLS = asyncio.run(mcp_client.get_tools())   # async-only tools

# --- make the async MCP tools callable SYNCHRONOUSLY ----------------------
# Streamlit runs the graph with the sync chatbot.stream(), but MCP tools are
# async-only ("StructuredTool does not support sync invocation"). We run a
# private event loop in a background thread and bridge each call to it.
_bg_loop = asyncio.new_event_loop()
threading.Thread(target=_bg_loop.run_forever, daemon=True).start()


def _wrap_sync(async_tool):
    """Return a copy of an async MCP tool that ALSO works when called sync."""
    async def _acall(**kwargs):
        return await async_tool.ainvoke(kwargs)

    def _scall(**kwargs):
        fut = asyncio.run_coroutine_threadsafe(_acall(**kwargs), _bg_loop)
        return fut.result()

    return StructuredTool.from_function(
        func=_scall,               # sync path (what ToolNode uses here)
        coroutine=_acall,          # async path (kept for completeness)
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
    )


MCP_TOOLS = [_wrap_sync(t) for t in _RAW_MCP_TOOLS]

# everything the model can call = local tools + MCP tools
ALL_TOOLS = list(LOCAL_TOOLS) + list(MCP_TOOLS)


# ===========================================================================
# 2. State + model
# ===========================================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOllama(
    model="qwen3:8b",
    reasoning=False,
    keep_alive="30m",
    temperature=0,
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)


def build_system_prompt() -> SystemMessage:
    """Built fresh each turn so it can tell the model which documents are
    currently uploaded — this is what makes the agent actually call the RAG
    tool instead of answering from memory."""
    text = (
        "You are a helpful assistant with access to tools.\n"
        "- For arithmetic use `calculator`. For weather use `get_weather`. "
        "For any price use `get_stock_price`. For sending email use `send_email`.\n"
        "- When drafting an email body, write clean PLAIN TEXT — short paragraphs "
        "and simple '-' bullets. Never use HTML unless explicitly asked.\n"
        "- If the user's message contains a YouTube link/URL, FIRST call "
        "`add_youtube_video` with that URL to load its transcript, then answer "
        "their question using `search_documents`.\n"
        "- If the user asks for the answer as audio/voice or to listen/hear it, "
        "first compose your answer, then call `text_to_speech` with that answer.\n"
        "- For current news or latest events, use `get_news` and summarize the "
        "results, mentioning the sources.\n"
    )
    if RAG.has_documents:
        text += (
            "\nIMPORTANT — the user has UPLOADED these documents: "
            + ", ".join(RAG.sources()) + ".\n"
            "For ANY question that could relate to their content (people, terms, "
            "products, facts named in them), you MUST call `search_documents` "
            "FIRST and answer from its results, citing the [S#] tags. Do NOT answer "
            "from your own knowledge and do NOT say a term is unknown before you "
            "have searched the documents.\n"
        )
    else:
        text += "- No documents are uploaded; do not use `search_documents`.\n"
    text += "Once you have what you need, reply in clear natural language."
    return SystemMessage(content=text)


# ===========================================================================
# 3. Nodes
# ===========================================================================
def chat_node(state: ChatState):
    response = llm_with_tools.invoke([build_system_prompt()] + list(state["messages"]))
    return {"messages": [response]}


def review_email(state: ChatState):
    """HUMAN-IN-THE-LOOP gate. Runs when the model wants to send an email.
    Pauses for approval, then routes based on the human's decision."""
    last = state["messages"][-1]                      # AIMessage with tool_calls

    #Last Message must look like
    """AIMessage(
    content="",
    tool_calls=[
        {
            "name": "send_email",
            "id": "call_123",
            "args": {
                "to": "rahul@gmail.com",
                "subject": "Meeting",
                "body": "Hello Rahul..."
            }
        }
    ]
)"""
    email_call = next(tc for tc in last.tool_calls if tc["name"] == EMAIL_TOOL_NAME)
    args = email_call.get("args", {})

    # PAUSE — hand the draft to the UI (same payload shape the frontend expects)
    decision = interrupt({
        "type": "email_approval",
        "to": args.get("to", ""),
        "subject": args.get("subject", ""),
        "body": args.get("body", ""),
    })
    if not isinstance(decision, dict):
        decision = {"action": str(decision)}
    action = decision.get("action", "reject")

    if action == "approve":
        # apply any edits the user made in the UI, then let the tool run
        new_calls = []
        for c in last.tool_calls:
            if c["id"] == email_call["id"]:
                c = {**c, "args": {
                    "to": decision.get("to", args.get("to")),
                    "subject": decision.get("subject", args.get("subject")),
                    "body": decision.get("body", args.get("body")),
                }}
            new_calls.append(c)
        edited = AIMessage(content=last.content, tool_calls=new_calls, id=last.id)
        return Command(goto="tools", update={"messages": [edited]})

    if action == "revise":
        note = ToolMessage(
            content=("The user did NOT approve. Requested changes: "
                     + decision.get("feedback", "")
                     + " Rewrite the email as clean plain text and call send_email again."),
            tool_call_id=email_call["id"],
        )
        return Command(goto="chat_node", update={"messages": [note]})

    # reject
    note = ToolMessage(
        content="The user rejected the email. It was NOT sent.",
        tool_call_id=email_call["id"],
    )
    return Command(goto="chat_node", update={"messages": [note]})


#Take the tool calls generated by the LLM and actually execute the corresponding tools. this are the example of tool calls, ToolNode receives this tool call and executes the actual send_email function.
"""{
    "name": "send_email",
    "args": {
        "to": "rahul@gmail.com",
        "subject": "Meeting",
        "body": "Meeting is at 10 AM."
    }
}  """


tool_node = ToolNode(ALL_TOOLS, handle_tool_errors=True)


def route_after_chat(state: ChatState):
    """Decide where to go after the model speaks."""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if not tool_calls:
        return END
    names = [tc["name"] for tc in tool_calls]
    if EMAIL_TOOL_NAME in names:      # email needs approval first
        return "review_email"
    return "tools"                    # other tools run directly


# ===========================================================================
# 4. Persistence
# ===========================================================================
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_conn)
checkpointer.setup()
init_chats_table()


# ===========================================================================
# 5. Graph wiring
# ===========================================================================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("review_email", review_email)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    route_after_chat,
    {"review_email": "review_email", "tools": "tools", END: END},
)
graph.add_edge("tools", "chat_node")
# review_email routes itself with Command(goto=...), so no static edge needed.

chatbot = graph.compile(checkpointer=checkpointer)

TOOL_NAMES = [t.name for t in ALL_TOOLS]
