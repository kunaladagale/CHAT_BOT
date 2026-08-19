# load environment variables FIRST — before importing langgraph_backend,
# which imports LangChain. This is what turns LangSmith tracing on.
from dotenv import load_dotenv
load_dotenv()

import os
import re
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.types import Command
from langgraph.errors import GraphInterrupt

from langgraph_backend import chatbot
import chat_db as db          # SQLite metadata layer (create/list/rename/delete chats)
from tools import TOOL_LABELS  # {tool_name: "🧮 Calculator", ...} for the status line
from rag import RAG            # document index for the Agentic RAG tool

print("[LangSmith] tracing:", os.getenv("LANGCHAIN_TRACING_V2"),
      "| project:", os.getenv("LANGCHAIN_PROJECT"))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def generate_thread_id():
    """Generate a unique ID for every chat."""
    return str(uuid.uuid4())


def current_config():
    """The config for the CURRENT thread — used for streaming and resuming."""
    tid = st.session_state['thread_id']
    return {
        'configurable': {'thread_id': tid},   # LangGraph checkpointer (memory)
        'metadata': {'thread_id': tid},        # LangSmith groups runs into a THREAD
        'run_name': 'chat_turn',
    }


def new_chat():
    """Create a brand-new conversation, persisted in SQLite immediately."""
    tid = generate_thread_id()
    db.create_chat(tid)
    st.session_state['thread_id'] = tid
    st.session_state['message_history'] = []
    st.session_state['generating'] = False
    st.session_state.pop('pending_email', None)   # clear any pending approval


def load_history_from_db(thread_id):
    """Rebuild on-screen history from the checkpointer. Skips tool-call plumbing
    (ToolMessages and empty tool-call AIMessages) so only real turns show."""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', []) if state.values else []
    history = []
    for m in messages:
        if isinstance(m, HumanMessage):
            history.append({'role': 'user', 'content': m.content})
        elif isinstance(m, AIMessage) and m.content:   # skip tool-call-only AIMessages
            history.append({'role': 'assistant', 'content': m.content})
        # ToolMessage -> skipped
    return history


def open_chat(thread_id):
    """Switch to an existing conversation and load it from the database."""
    st.session_state['thread_id'] = thread_id
    st.session_state['generating'] = False
    st.session_state.pop('pending_email', None)
    st.session_state['message_history'] = load_history_from_db(thread_id)


def get_pending_interrupt():
    """If the graph is paused on an interrupt() (awaiting human approval),
    return that interrupt's payload; otherwise None."""
    snap = chatbot.get_state(current_config())
    if not snap.next:                      # graph is not paused
        return None
    # newer LangGraph exposes snapshot.interrupts; older nests under tasks
    intr = getattr(snap, "interrupts", None)
    if intr:
        return intr[0].value
    for task in getattr(snap, "tasks", ()):
        t_intr = getattr(task, "interrupts", None)
        if t_intr:
            return t_intr[0].value
    return None


# ---------------------------------------------------------------------------
# session state
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
# sidebar: document upload for Agentic RAG
# ---------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.header('📄 Documents')
uploaded = st.sidebar.file_uploader(
    'Upload documents to ask questions about',
    type=['pdf', 'docx', 'txt', 'md', 'csv'],
    accept_multiple_files=True,
    key='rag_uploader',
)
if uploaded and st.sidebar.button('📥 Index files', use_container_width=True):
    import os
    import tempfile
    updir = os.path.join(tempfile.gettempdir(), 'rag_uploads')
    os.makedirs(updir, exist_ok=True)
    paths = []
    for f in uploaded:
        p = os.path.join(updir, f.name)
        with open(p, 'wb') as out:
            out.write(f.getbuffer())
        paths.append(p)
    with st.spinner('Indexing documents…'):
        msg = RAG.ingest(paths)
    st.sidebar.success(msg)

if RAG.has_documents:
    # pick which source questions are answered from (persists in the RAG store)
    options = ['All sources'] + RAG.sources()
    current = RAG.active_source or 'All sources'
    idx = options.index(current) if current in options else 0
    picked = st.sidebar.selectbox('Answer questions from:', options, index=idx)
    RAG.set_active(None if picked == 'All sources' else picked)

# ---------------------------------------------------------------------------
# streaming helpers (always use the CURRENT thread id)
# ---------------------------------------------------------------------------

def render_stream(stream_iter):
    """Consume a LangGraph messages-stream and drive the UI:
      - show a '🔧 Using <tool>…' line whenever the model calls a tool
      - stream the assistant's answer token by token
    Returns the final assistant text.
    """
    status_area = st.container()   # tool-usage lines appear here, above the answer
    text_box = st.empty()          # the streamed answer appears here
    text = ""
    seen_tools = set()

    for chunk, metadata in stream_iter:
        if not isinstance(chunk, AIMessageChunk):
            continue

        # --- detect tool calls the model is making ---
        tool_calls = chunk.tool_call_chunks or []
        if not tool_calls and getattr(chunk, "tool_calls", None):
            tool_calls = chunk.tool_calls   # some providers send whole calls, not deltas
        for tc in tool_calls:
            name = tc.get("name")
            if name and name not in seen_tools:
                seen_tools.add(name)
                status_area.info(f"🔧 Using {TOOL_LABELS.get(name, name)} …")

        # --- stream the answer text ---
        if chunk.content:
            text += chunk.content
            text_box.markdown(text)

    return text


def new_turn_stream(user_text):
    return chatbot.stream(
        {'messages': [HumanMessage(content=user_text)]},
        config=current_config(),
        stream_mode="messages",
    )


def resume_stream(decision):
    return chatbot.stream(
        Command(resume=decision),
        config=current_config(),
        stream_mode="messages",
    )

# ---------------------------------------------------------------------------
# main chat area
# ---------------------------------------------------------------------------

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# ---- HUMAN-IN-THE-LOOP: email approval gate -------------------------------
# If the graph paused asking to send an email, show the draft for approval
# instead of the normal input.
if st.session_state.get('pending_email'):
    draft = st.session_state['pending_email']
    with st.chat_message('assistant'):
        st.warning("✋ Please review this email before it is sent.")
        appr_to = st.text_input('To', draft.get('to', ''), key='appr_to')
        appr_subject = st.text_input('Subject', draft.get('subject', ''), key='appr_subject')
        appr_body = st.text_area('Body', draft.get('body', ''), height=220, key='appr_body')

        c1, c2 = st.columns(2)
        approve = c1.button('✅ Approve & Send', use_container_width=True)
        reject = c2.button('❌ Reject', use_container_width=True)

        st.divider()
        st.caption("Not quite right? Tell the assistant what to change and it will rewrite it:")
        feedback = st.text_area(
            'Requested changes',
            key='appr_feedback',
            placeholder='e.g. make it plain text, shorter, and more formal',
        )
        revise = st.button('✏️ Request changes', use_container_width=True)

    if approve or reject or revise:
        if revise:
            decision = {
                'action': 'revise',
                'feedback': feedback,
                'to': appr_to, 'subject': appr_subject, 'body': appr_body,
            }
        else:
            decision = {
                'action': 'approve' if approve else 'reject',
                'to': appr_to, 'subject': appr_subject, 'body': appr_body,
            }

        with st.chat_message('assistant'):
            try:
                render_stream(resume_stream(decision))
            except GraphInterrupt:
                pass                       # revision produced a NEW draft -> handled below
            except Exception as e:
                st.markdown(f"⚠️ Error: {e}")
        wait_for_all_tracers()

        # did the assistant produce a fresh draft to approve? (revise path)
        pending = get_pending_interrupt()
        if pending and isinstance(pending, dict) and pending.get('type') == 'email_approval':
            st.session_state['pending_email'] = pending        # show the new draft
        else:
            st.session_state.pop('pending_email', None)         # sent or rejected -> done

        # reset the form widgets so they re-load from the (possibly new) draft
        for k in ('appr_to', 'appr_subject', 'appr_body', 'appr_feedback'):
            st.session_state.pop(k, None)

        db.touch_chat(st.session_state['thread_id'])
        st.session_state['message_history'] = load_history_from_db(st.session_state['thread_id'])
        st.rerun()

# ---- normal chat input (disabled while generating OR awaiting approval) ----
input_disabled = st.session_state['generating'] or bool(st.session_state.get('pending_email'))
user_input = st.chat_input('Type here', disabled=input_disabled)

# regex to spot a YouTube link anywhere in the message
_YT_URL = re.compile(r'(https?://\S*(?:youtube\.com/\S+|youtu\.be/\S+))')

# Step 1: new message -> name the chat (if first), store it, lock input, rerun
if user_input and not st.session_state['generating']:
    tid = st.session_state['thread_id']

    # DETERMINISTIC YouTube handling: if the message has a YouTube URL, load its
    # transcript now (don't rely on the small model to call the tool).
    yt = _YT_URL.search(user_input)
    if yt:
        with st.spinner('▶️ Loading YouTube transcript…'):
            info = RAG.ingest_youtube(yt.group(1))
        st.sidebar.info(info)

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

# Step 2: generating -> stream the answer live, then check if the graph paused
# for email approval
if st.session_state['generating']:
    last_user = st.session_state['message_history'][-1]['content']
    with st.chat_message('assistant'):
        try:
            ai_message = render_stream(new_turn_stream(last_user))
        except GraphInterrupt:
            ai_message = ""          # paused for approval — handled below, not an error
        except Exception as e:
            ai_message = f"⚠️ Error: {e}"
            st.markdown(ai_message)

    wait_for_all_tracers()

    # did the graph pause asking to send an email?
    pending = get_pending_interrupt()
    if pending and isinstance(pending, dict) and pending.get('type') == 'email_approval':
        st.session_state['pending_email'] = pending

    db.touch_chat(st.session_state['thread_id'])
    st.session_state['message_history'] = load_history_from_db(st.session_state['thread_id'])
    st.session_state['generating'] = False
    st.rerun()