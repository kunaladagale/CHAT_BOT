# load environment variables FIRST — before importing langgraph_backend,
# which imports LangChain. This is what turns LangSmith tracing on.
from dotenv import load_dotenv
load_dotenv()

import os
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import wait_for_all_tracers

from langgraph_backend import chatbot
import chat_db as db   # SQLite metadata layer (create/list/rename/delete chats)

# One-time confirmation in the TERMINAL that tracing is on (remove later).
print("[LangSmith] tracing:", os.getenv("LANGCHAIN_TRACING_V2"),
      "| project:", os.getenv("LANGCHAIN_PROJECT"))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def generate_thread_id():
    """Generate a unique ID for every chat."""
    return str(uuid.uuid4())


def new_chat():
    """Create a brand-new conversation, persisted in SQLite immediately."""
    tid = generate_thread_id()
    db.create_chat(tid)                      # saved in the DB right away
    st.session_state['thread_id'] = tid
    st.session_state['message_history'] = []
    st.session_state['generating'] = False


def load_history_from_db(thread_id):
    """Rebuild the on-screen history from the checkpointer (the database is the
    source of truth, NOT session_state)."""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', []) if state.values else []
    history = []
    for m in messages:
        role = 'user' if isinstance(m, HumanMessage) else 'assistant'
        history.append({'role': role, 'content': m.content})
    return history


def open_chat(thread_id):
    """Switch to an existing conversation and load it from the database."""
    st.session_state['thread_id'] = thread_id
    st.session_state['generating'] = False
    st.session_state['message_history'] = load_history_from_db(thread_id)


# ---------------------------------------------------------------------------
# session state  (UI state only — the real data lives in the database)
# ---------------------------------------------------------------------------

if 'generating' not in st.session_state:
    st.session_state['generating'] = False

if 'thread_id' not in st.session_state:
    new_chat()

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = load_history_from_db(st.session_state['thread_id'])

# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

st.sidebar.title('LangGraph Chat Bot')

if st.sidebar.button('➕ New Chat', use_container_width=True):
    new_chat()
    st.rerun()

st.sidebar.header('Chats')

for chat in db.list_chats():
    tid = chat['thread_id']
    title = chat['title']

    is_current = (tid == st.session_state['thread_id'])
    open_col, menu_col = st.sidebar.columns([0.8, 0.2])

    if open_col.button(
        ('👉 ' if is_current else '') + (title or 'New Chat'),
        key=f'open_{tid}',
        use_container_width=True,
    ):
        open_chat(tid)
        st.rerun()

    with menu_col.popover('⋮'):
        new_title = st.text_input('Rename chat', value=title, key=f'rn_{tid}')
        if st.button('💾 Save', key=f'save_{tid}'):
            db.rename_chat(tid, new_title)
            st.rerun()

        st.divider()
        confirm = st.checkbox('Confirm delete', key=f'cf_{tid}')
        if st.button('🗑️ Delete', key=f'del_{tid}', disabled=not confirm):
            db.delete_chat(tid)
            if tid == st.session_state['thread_id']:
                new_chat()
            st.rerun()

# ---------------------------------------------------------------------------
# streaming generator (always uses the CURRENT thread id)
# ---------------------------------------------------------------------------

def stream_reply(user_text):
    tid = st.session_state['thread_id']
    config = {
        'configurable': {'thread_id': tid},   # LangGraph checkpointer (memory)
        'metadata': {'thread_id': tid},        # LangSmith groups runs into a THREAD by this
        'run_name': 'chat_turn',               # nicer label for each turn in LangSmith
    }
    for chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content=user_text)]},
        config=config,
        stream_mode="messages",
    ):
        if chunk.content:
            yield chunk.content

# ---------------------------------------------------------------------------
# main chat area
# ---------------------------------------------------------------------------

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here', disabled=st.session_state['generating'])

# Step 1: new message -> name the chat (if first), store it, lock input, rerun
if user_input and not st.session_state['generating']:
    tid = st.session_state['thread_id']
    db.set_title_if_default(tid, user_input[:40])
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    st.session_state['generating'] = True
    st.rerun()

# guard: only generate when the last message is a pending user message
if st.session_state['generating'] and (
    not st.session_state['message_history']
    or st.session_state['message_history'][-1]['role'] != 'user'
):
    st.session_state['generating'] = False

# Step 2: generating -> stream the answer live (also traced to LangSmith and
# saved to the database by the checkpointer), then unlock
if st.session_state['generating']:
    last_user = st.session_state['message_history'][-1]['content']
    with st.chat_message('assistant'):
        placeholder = st.empty()
        ai_message = ""
        try:
            # drain the stream FULLY (to StopIteration) so LangGraph closes
            # cleanly — avoids the GeneratorExit that st.write_stream + rerun cause
            for token in stream_reply(last_user):
                ai_message += token
                placeholder.markdown(ai_message)
        except Exception as e:
            ai_message = f"⚠️ Error: {e}"
            placeholder.markdown(ai_message)

    wait_for_all_tracers()   # push the trace to LangSmith right away
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    db.touch_chat(st.session_state['thread_id'])
    st.session_state['generating'] = False
    st.rerun()