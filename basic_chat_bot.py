# from logging import config
#
# from langgraph.graph import StateGraph, START, END
# from typing import TypedDict, Annotated
# from langchain_core.messages import BaseMessage, HumanMessage
# from langchain_ollama import ChatOllama
# from langgraph.checkpoint.memory import MemorySaver
# from langsmith._openapi_client.types import thread
#
# from langgraph.graph.message import add_messages
#
# class ChatState(TypedDict):
#
#     messages: Annotated[list[BaseMessage], add_messages]  # this add msg is the reducer function
#
# llm = ChatOllama(model="qwen3:8b")
#
#
# def chat_node(state: ChatState):
#
#     # take user query from state
#     messages = state['messages']
#
#     # send to llm
#     response = llm.invoke(messages)
#
#     # response store state
#     return {'messages': [response]}
#
# checkpointer = MemorySaver()
# graph = StateGraph(ChatState)
#
# # add nodes
# graph.add_node('chat_node', chat_node)
#
# graph.add_edge(START, 'chat_node')
# graph.add_edge('chat_node', END)
#
# chatbot = graph.compile(checkpointer=checkpointer)
#
#
#
# initial_state = {
#     'messages': [HumanMessage(content='What is the capital of india')]
# }
#
# print(chatbot.invoke(initial_state)['messages'][-1].content)
#
#
# thread_id = 1
# while True:
#     user_message = input('Type here')
#     print("user_message:", user_message)
#
#     if user_message.strip().lower() in ["exit", "quit"]:
#         break
#
#     config = {'configurable': {'thread_id': thread_id}}
#
#     response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)
#     print("AI", response['messages'][-1].content)
#
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOllama(model="qwen3:8b")


def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------------
# Conversation configuration
# -------------------------

thread_id = "1"

config = {
    "configurable": {
        "thread_id": thread_id
    }
}


# -------------------------
# First message
# -------------------------

initial_state = {
    "messages": [
        HumanMessage(content="What is the capital of India?")
    ]
}

response = chatbot.invoke(
    initial_state,
    config=config
)

print("AI:", response["messages"][-1].content)


# -------------------------
# Continue conversation
# -------------------------

while True:

    user_message = input("Type here: ")

    if user_message.strip().lower() in ["exit", "quit"]:
        break

    response = chatbot.invoke(
        {
            "messages": [
                HumanMessage(content=user_message)
            ]
        },
        config=config
    )

    print("AI:", response["messages"][-1].content)