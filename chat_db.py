"""
SQLite layer for the chatbot.

No server, no password, no install — SQLite is built into Python and stores
everything in a single file (chatbot.db) in your project folder.

Two kinds of data live in that file:
  1. Conversation messages -> stored by LangGraph's SqliteSaver
                              (tables: checkpoints / writes)
  2. Chat metadata         -> stored here in our own `chats` table
                              (thread_id, title, created_at, updated_at)
"""
import os
import sqlite3
# ---------------------------------------------------------------------------
# database file
# ---------------------------------------------------------------------------
# One file on disk, next to your code. Change the path/name if you like.
DB_PATH = os.getenv("CHAT_DB_PATH", "chatbot.db")

# check_same_thread=False  -> Streamlit uses background threads
# isolation_level=None     -> autocommit, so we never forget to commit
_conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_conn.execute("PRAGMA busy_timeout = 5000;")   # wait, don't error, if briefly locked

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
def init_chats_table():
    """Create the chat-metadata table if it doesn't exist. Safe to call every start."""
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            thread_id  TEXT PRIMARY KEY,
            title      TEXT NOT NULL DEFAULT 'New Chat',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )


# ---------------------------------------------------------------------------
# chat metadata operations
# ---------------------------------------------------------------------------
def create_chat(thread_id, title="New Chat"):
    """Register a new (empty) conversation. No-op if it already exists."""
    _conn.execute(
        "INSERT OR IGNORE INTO chats (thread_id, title) VALUES (?, ?);",
        (thread_id, title),
    )


def list_chats():
    """Return all chats, newest activity first: [{'thread_id':..., 'title':...}, ...]."""
    rows = _conn.execute(
        "SELECT thread_id, title FROM chats ORDER BY updated_at DESC;"
    ).fetchall()
    return [{"thread_id": r[0], "title": r[1]} for r in rows]


def rename_chat(thread_id, new_title):
    """Change a chat's title (persisted)."""
    _conn.execute(
        "UPDATE chats SET title = ?, updated_at = datetime('now') WHERE thread_id = ?;",
        (new_title, thread_id),
    )


def set_title_if_default(thread_id, title):
    """Set the title only if it's still the default — used to name a chat by its
    first message without overwriting a name the user chose."""
    _conn.execute(
        "UPDATE chats SET title = ? WHERE thread_id = ? AND title = 'New Chat';",
        (title, thread_id),
    )


def touch_chat(thread_id):
    """Bump updated_at so the chat floats to the top of the list."""
    _conn.execute(
        "UPDATE chats SET updated_at = datetime('now') WHERE thread_id = ?;",
        (thread_id,),
    )


def delete_chat(thread_id):
    """Permanently delete a conversation: its messages (checkpointer tables)
    AND its metadata row."""
    # remove the stored conversation state written by SqliteSaver
    for table in ("writes", "checkpoints"):
        try:
            _conn.execute(f"DELETE FROM {table} WHERE thread_id = ?;", (thread_id,))
        except sqlite3.OperationalError:
            # table may not exist yet on a brand-new database — safe to ignore
            pass
    # remove our metadata
    _conn.execute("DELETE FROM chats WHERE thread_id = ?;", (thread_id,))