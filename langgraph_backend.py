import sqlite3
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

from chat_db import DB_PATH, init_chats_table   # share the same SQLite file


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOllama(
    model="qwen3:8b",
    reasoning=False,      # clean, direct answers (no thinking monologue)
    keep_alive="30m",     # keep the model warm for fast replies
)


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# PERSISTENT memory in a local SQLite file (no server, no password).
# SqliteSaver saves every turn keyed by thread_id and, on the next run,
# feeds the whole history back to the LLM automatically.
# ---------------------------------------------------------------------------
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_conn)
checkpointer.setup()          # creates checkpoint tables the first time (idempotent)
init_chats_table()            # creates our chat-metadata table the first time

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)