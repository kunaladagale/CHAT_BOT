"""
streamlit_frontend.py
=====================
Chat UI with live tool visibility.

The interesting part is stream_reply(): it asks LangGraph for TWO stream modes
at once --

    "messages" -> token-by-token text as the LLM writes it
    "updates"  -> a dict per node as each node finishes

-- and turns them into a single flat event stream the UI can render:

    {"type": "token",      "text": ...}
    {"type": "tool_start", "name": ..., "args": ...}
    {"type": "tool_end",   "name": ..., "result": ...}

That's what lets the user watch "🌤️ Weather — running…" turn into
"✅ Weather" with the raw result inside it, before the answer appears.
"""

# load environment variables FIRST — before importing langgraph_backend,
# which imports LangChain. This is what turns LangSmith tracing on.
from dotenv import load_dotenv
load_dotenv()

import os
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tracers.langchain import wait_for_all_tracers

from langgraph_backend import chatbot
from tools import TOOL_LABELS
import chat_db as db   # SQLite metadata layer (create/list/rename/delete chats)

# One-time confirmation in the TERMINAL that tracing is on (remove later).
print("[LangSmith] tracing:", os.getenv("LANGCHAIN_TRACING_V2"),
      "| project:", os.getenv("LANGCHAIN_PROJECT"))

# Stop a confused model from looping chat -> tools -> chat forever.
# Each round trip costs 2 steps, so 12 allows ~5 tool rounds per turn.
RECURSION_LIMIT = 12


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


def _text_of(content) -> str:
    """Message content is usually a str, but some providers send a list of
    content blocks. Flatten either into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


def load_history_from_db(thread_id):
    """Rebuild the on-screen history from the checkpointer (the database is the
    source of truth, NOT session_state).

    The checkpointed message list now contains four kinds of message:
        HumanMessage                       -> a user bubble
        AIMessage with .tool_calls         -> the model asking for a tool
        ToolMessage                        -> the tool's result
        AIMessage with text                -> the answer bubble

    Only the first and last become bubbles; the middle two are collected into
    the `tools` list attached to the answer bubble that follows them.
    """
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', []) if state.values else []

    history = []
    pending_tools = []     # tool activity seen since the last answer bubble
    by_call_id = {}        # tool_call_id -> the dict in pending_tools

    for m in messages:
        if isinstance(m, HumanMessage):
            history.append({'role': 'user', 'content': _text_of(m.content), 'tools': []})

        elif isinstance(m, ToolMessage):
            entry = by_call_id.get(getattr(m, 'tool_call_id', None))
            if entry is not None:
                entry['result'] = _text_of(m.content)
            else:  # result without a matching request (shouldn't happen)
                pending_tools.append({'name': getattr(m, 'name', 'tool') or 'tool',
                                      'args': {}, 'result': _text_of(m.content)})

        elif isinstance(m, AIMessage):
            tool_calls = getattr(m, 'tool_calls', None) or []
            for tc in tool_calls:
                entry = {'name': tc.get('name', 'tool'),
                         'args': tc.get('args', {}) or {},
                         'result': None}
                pending_tools.append(entry)
                if tc.get('id'):
                    by_call_id[tc['id']] = entry

            text = _text_of(m.content)
            # An AIMessage that only asks for tools is not an answer bubble.
            if not tool_calls and text.strip():
                history.append({'role': 'assistant', 'content': text,
                                'tools': pending_tools})
                pending_tools, by_call_id = [], {}

    return history


def open_chat(thread_id):
    """Switch to an existing conversation and load it from the database."""
    st.session_state['thread_id'] = thread_id
    st.session_state['generating'] = False
    st.session_state['message_history'] = load_history_from_db(thread_id)


def render_tool_activity(tools):
    """Redraw past tool calls (collapsed) when history is re-rendered."""
    for t in tools or []:
        label = TOOL_LABELS.get(t['name'], f"🔧 {t['name']}")
        with st.expander(f"✅ {label}", expanded=False):
            st.markdown("**Input**")
            st.json(t.get('args') or {})
            st.markdown("**Result**")
            st.code(t.get('result') or '(no result)', language=None)


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

st.sidebar.title('LangGraph Agent')

if st.sidebar.button('➕ New Chat', use_container_width=True):
    new_chat()
    st.rerun()

with st.sidebar.expander('🔧 Available tools', expanded=False):
    st.markdown(
        "- 🧮 **calculator** — any arithmetic\n"
        "- 🌤️ **get_weather** — live weather, any city/district\n"
        "- 📈 **get_stock_price** — live stock, index & crypto prices"
    )

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
    """Run one agent turn and yield flat UI events.

    stream_mode is a LIST, so LangGraph yields (mode, payload) tuples:
      ("messages", (chunk, metadata))  -> live tokens
      ("updates",  {node_name: state}) -> a node just finished
    """
    tid = st.session_state['thread_id']
    config = {
        'configurable': {'thread_id': tid},   # LangGraph checkpointer (memory)
        'metadata': {'thread_id': tid},        # LangSmith groups runs into a THREAD by this
        'run_name': 'chat_turn',               # nicer label for each turn in LangSmith
        'recursion_limit': RECURSION_LIMIT,    # hard stop on runaway tool loops
    }

    for mode, payload in chatbot.stream(
        {'messages': [HumanMessage(content=user_text)]},
        config=config,
        stream_mode=['updates', 'messages'],
    ):
        # ---- live tokens from the LLM ------------------------------------
        if mode == 'messages':
            chunk, metadata = payload
            if metadata.get('langgraph_node') != 'chat_node':
                continue
            text = _text_of(getattr(chunk, 'content', ''))
            if text:
                yield {'type': 'token', 'text': text}

        # ---- a node finished: look for tool requests / tool results ------
        elif mode == 'updates':
            for node, update in (payload or {}).items():
                if not isinstance(update, dict):
                    continue
                for m in update.get('messages', []) or []:

                    if node == 'chat_node':
                        for tc in getattr(m, 'tool_calls', None) or []:
                            yield {
                                'type': 'tool_start',
                                'id': tc.get('id'),
                                'name': tc.get('name', 'tool'),
                                'args': tc.get('args', {}) or {},
                            }

                    elif node == 'tools':
                        yield {
                            'type': 'tool_end',
                            'id': getattr(m, 'tool_call_id', None),
                            'name': getattr(m, 'name', 'tool') or 'tool',
                            'result': _text_of(getattr(m, 'content', '')),
                        }


# ---------------------------------------------------------------------------
# main chat area
# ---------------------------------------------------------------------------

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        if message['role'] == 'assistant':
            render_tool_activity(message.get('tools'))
        st.markdown(message['content'])

user_input = st.chat_input('Type here', disabled=st.session_state['generating'])

# Step 1: new message -> name the chat (if first), store it, lock input, rerun
if user_input and not st.session_state['generating']:
    tid = st.session_state['thread_id']
    db.set_title_if_default(tid, user_input[:40])
    st.session_state['message_history'].append(
        {'role': 'user', 'content': user_input, 'tools': []}
    )
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
        # tool boxes render ABOVE the answer because this container is
        # created first
        tools_area = st.container()
        placeholder = st.empty()

        ai_message = ""
        tool_log = []          # what we persist so it redraws after the rerun
        open_boxes = {}        # tool_call_id -> (status_widget, body_slot, entry)

        try:
            # drain the stream FULLY (to StopIteration) so LangGraph closes
            # cleanly — avoids the GeneratorExit that st.write_stream + rerun cause
            for event in stream_reply(last_user):

                # ---------- live text ----------------------------------
                if event['type'] == 'token':
                    ai_message += event['text']
                    placeholder.markdown(ai_message)

                # ---------- the model asked for a tool -----------------
                elif event['type'] == 'tool_start':
                    # Anything the model typed before deciding to call a tool
                    # is preamble noise; drop it so only the final answer shows.
                    ai_message = ""
                    placeholder.empty()

                    label = TOOL_LABELS.get(event['name'], f"🔧 {event['name']}")
                    with tools_area:
                        box = st.status(f"{label} — running…",
                                        state="running", expanded=True)
                    with box:
                        st.markdown("**Input**")
                        st.json(event['args'])
                        st.markdown("**Result**")
                        body = st.empty()
                        body.caption("waiting…")

                    entry = {'name': event['name'], 'args': event['args'],
                             'result': None}
                    tool_log.append(entry)
                    key = event.get('id') or f"{event['name']}#{len(tool_log)}"
                    open_boxes[key] = (box, body, label, entry)

                # ---------- the tool came back -------------------------
                elif event['type'] == 'tool_end':
                    key = event.get('id')
                    slot = open_boxes.pop(key, None)
                    if slot is None:  # fall back to matching on name
                        for k, v in list(open_boxes.items()):
                            if v[3]['name'] == event['name']:
                                slot = open_boxes.pop(k)
                                break
                    if slot is None:  # a result with no box; make one
                        label = TOOL_LABELS.get(event['name'], f"🔧 {event['name']}")
                        with tools_area:
                            box = st.status(label, state="complete", expanded=False)
                        with box:
                            body = st.empty()
                        entry = {'name': event['name'], 'args': {}, 'result': None}
                        tool_log.append(entry)
                        slot = (box, body, label, entry)

                    box, body, label, entry = slot
                    result = event['result']
                    entry['result'] = result

                    failed = result.strip().lower().startswith('error')
                    body.code(result or '(empty)', language=None)
                    box.update(
                        label=f"{'⚠️' if failed else '✅'} {label}",
                        state="error" if failed else "complete",
                        expanded=False,
                    )

        except Exception as e:
            ai_message = f"⚠️ Error: {e}"
            placeholder.markdown(ai_message)

        # any box still open means the run died mid-tool
        for box, body, label, entry in open_boxes.values():
            box.update(label=f"⚠️ {label} — interrupted", state="error", expanded=False)

        if not ai_message.strip():
            ai_message = "_(no text response)_"
            placeholder.markdown(ai_message)

    wait_for_all_tracers()   # push the trace to LangSmith right away
    st.session_state['message_history'].append(
        {'role': 'assistant', 'content': ai_message, 'tools': tool_log}
    )
    db.touch_chat(st.session_state['thread_id'])
    st.session_state['generating'] = False
    st.rerun()