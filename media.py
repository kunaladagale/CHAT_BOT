"""
media.py
========
Tiny shared hand-off between the text_to_speech tool (which runs deep inside the
graph) and the Streamlit UI (which shows the audio player). The tool drops the
path of the audio it just created here; the frontend picks it up and renders it.
"""

_LAST_AUDIO = {"path": None}


def set_last_audio(path: str) -> None:
    _LAST_AUDIO["path"] = path


def pop_last_audio():
    """Return the most recent audio path once, then clear it."""
    path = _LAST_AUDIO["path"]
    _LAST_AUDIO["path"] = None
    return path