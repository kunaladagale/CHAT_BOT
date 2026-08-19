"""
rag.py
======
Advanced, fully-local Agentic RAG pipeline.

Pipeline (all local, no API keys):
    upload -> load (pdf/docx/txt/md/csv) -> chunk + metadata
           -> embed (Ollama: nomic-embed-text) -> Chroma (persistent)
    query  -> rewrite (qwen3) -> HYBRID retrieve (Chroma dense + BM25 keyword)
           -> RERANK (Flashrank) -> build context + citations

Exposed via a single object `RAG` with two methods the rest of the app uses:
    RAG.ingest(file_paths)  -> add documents to the index
    RAG.search(query, ...)  -> return (answer_context, citations)

Everything runs inside the LangGraph tool, so it is traced in LangSmith; the
@traceable spans below make each stage (rewrite / retrieve / rerank) show up as
a nested step.

Install:
    pip install langchain-chroma chromadb rank-bm25 flashrank \
                pypdf docx2txt langchain-community
    ollama pull nomic-embed-text
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_chroma import Chroma

# --- LangSmith @traceable, with a no-op fallback ---------------------------
try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda fn: fn


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
PERSIST_DIR = os.getenv("RAG_DIR", "rag_store")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DENSE_K = 10        # candidates from each retriever before reranking
RERANK_TOP_N = 5    # chunks kept after reranking (fed to the LLM)


# ---------------------------------------------------------------------------
# YouTube helpers
# ---------------------------------------------------------------------------
_YT_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> str | None:
    """Pull the 11-char video id out of any common YouTube URL form."""
    m = _YT_RE.search(url or "")
    if m:
        return m.group(1)
    s = (url or "").strip()
    return s if re.fullmatch(r"[A-Za-z0-9_-]{11}", s) else None


def _fmt_ts(seconds) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _fetch_transcript(video_id: str) -> list[dict]:
    """Fetch a transcript, working across youtube-transcript-api versions."""
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id)          # v1.x API
        return [{"text": s.text, "start": s.start} for s in fetched]
    except AttributeError:
        return YouTubeTranscriptApi.get_transcript(video_id)      # older API


def _loader_for(path: str):
    """Pick the right loader by file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(path)                 # per-page docs -> page metadata
    if ext == ".docx":
        return Docx2txtLoader(path)
    if ext == ".csv":
        return CSVLoader(path)
    return TextLoader(path, encoding="utf-8")     # .txt, .md, fallback


class RagStore:
    """Holds the vector index + keyword index. One instance for the app."""

    def __init__(self):
        self._lock = threading.Lock()
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        )
        # small, deterministic model just for query rewriting
        self._rewriter = ChatOllama(model="qwen3:8b", reasoning=False,
                                    temperature=0, keep_alive="30m")
        # persistent dense store
        self.vectorstore = Chroma(
            collection_name="documents",
            embedding_function=self.embeddings,
            persist_directory=PERSIST_DIR,
        )
        self._chunks: list[Document] = self._load_existing_chunks()
        # the source follow-up questions are scoped to (None = search everything)
        self.active_source: str | None = None

    # ---- persistence helpers ---------------------------------------------
    def _load_existing_chunks(self) -> list[Document]:
        """Rebuild the in-memory chunk list (for BM25) from what's already in
        Chroma, so keyword search survives an app restart."""
        try:
            data = self.vectorstore.get()          # all stored docs
            texts = data.get("documents") or []
            metas = data.get("metadatas") or []
            return [Document(page_content=t, metadata=m or {})
                    for t, m in zip(texts, metas)]
        except Exception:
            return []

    @property
    def has_documents(self) -> bool:
        return len(self._chunks) > 0

    def sources(self) -> list[str]:
        return sorted({c.metadata.get("source", "?") for c in self._chunks})

    def set_active(self, source: str | None):
        """Scope follow-up questions to one source (None = search all)."""
        self.active_source = source

    # ---- ingest ----------------------------------------------------------
    @traceable(run_type="chain", name="rag_ingest")
    def ingest(self, file_paths: list[str]) -> str:
        """Load, chunk (with metadata) and index the given files."""
        new_chunks: list[Document] = []
        for path in file_paths:
            name = Path(path).name
            try:
                docs = _loader_for(path).load()
            except Exception as exc:  # noqa: BLE001
                return f"Error loading {name}: {exc}"

            chunks = self.splitter.split_documents(docs)
            for i, ch in enumerate(chunks):
                ch.metadata["source"] = name
                ch.metadata["chunk_id"] = i
                ch.metadata.setdefault("page", ch.metadata.get("page"))
            new_chunks.extend(chunks)

        if not new_chunks:
            return "No text could be extracted from the uploaded file(s)."

        with self._lock:
            self.vectorstore.add_documents(new_chunks)   # embeds + persists
            self._chunks.extend(new_chunks)

        uploaded = sorted({Path(p).name for p in file_paths})
        # a single uploaded file becomes the active context; several -> search all
        self.active_source = uploaded[0] if len(uploaded) == 1 else None
        return f"Indexed {len(new_chunks)} chunks from: {', '.join(uploaded)}"

    # ---- ingest a YouTube transcript ------------------------------------
    @traceable(run_type="chain", name="rag_ingest_youtube")
    def ingest_youtube(self, url: str) -> str:
        """Fetch a video's transcript and index it into the SAME store, so it
        gets the full hybrid+rerank pipeline and is answered by search_documents."""
        vid = extract_video_id(url)
        if not vid:
            return f"Error: could not find a YouTube video id in '{url}'."
        try:
            entries = _fetch_transcript(vid)
        except Exception as exc:  # noqa: BLE001
            return (f"Error: could not fetch transcript for video {vid} -- {exc}. "
                    f"The video may have transcripts disabled.")
        if not entries:
            return f"Error: no transcript is available for video {vid}."

        source = f"YouTube:{vid}"
        # group transcript lines into blocks that each carry a start timestamp
        docs, buf, buf_start = [], [], None
        for e in entries:
            if buf_start is None:
                buf_start = e.get("start", 0)
            buf.append(e.get("text", ""))
            if sum(len(x) for x in buf) >= 500:
                docs.append(Document(page_content=" ".join(buf),
                                     metadata={"source": source, "video_id": vid,
                                               "timestamp": _fmt_ts(buf_start)}))
                buf, buf_start = [], None
        if buf:
            docs.append(Document(page_content=" ".join(buf),
                                 metadata={"source": source, "video_id": vid,
                                           "timestamp": _fmt_ts(buf_start or 0)}))

        chunks = self.splitter.split_documents(docs)
        for i, ch in enumerate(chunks):
            ch.metadata["chunk_id"] = i
        with self._lock:
            self.vectorstore.add_documents(chunks)
            self._chunks.extend(chunks)
        self.active_source = source        # scope follow-ups to this video
        return (f"Loaded transcript from {source} ({len(chunks)} chunks). "
                f"You can now ask questions about the video.")

    # ---- query transformation -------------------------------------------
    @traceable(run_type="chain", name="rag_query_rewrite")
    def _rewrite(self, query: str) -> str:
        """Rewrite a chat-style question into a keyword-rich retrieval query."""
        try:
            prompt = (
                "Rewrite the user's question into a concise, keyword-rich search "
                "query for document retrieval. Keep all key entities. Return ONLY "
                f"the rewritten query, nothing else.\n\nQuestion: {query}"
            )
            out = self._rewriter.invoke(prompt).content.strip()
            return out or query
        except Exception:  # never let rewrite failure break retrieval
            return query

    # ---- retrieval + rerank ---------------------------------------------
    @traceable(run_type="retriever", name="rag_hybrid_retrieve")
    def _retrieve(self, query: str, source: str | None):
        """Hybrid (dense + BM25) retrieval, then cross-encoder reranking."""
        search_kwargs = {"k": DENSE_K}
        if source:                                   # metadata filtering
            search_kwargs["filter"] = {"source": source}
        dense = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        pool = self._chunks
        if source:
            pool = [c for c in self._chunks if c.metadata.get("source") == source]
        bm25 = BM25Retriever.from_documents(pool or self._chunks)
        bm25.k = DENSE_K

        hybrid = EnsembleRetriever(retrievers=[dense, bm25], weights=[0.5, 0.5])

        reranked = ContextualCompressionRetriever(
            base_compressor=FlashrankRerank(top_n=RERANK_TOP_N),
            base_retriever=hybrid,
        )
        return reranked.invoke(query)

    @traceable(run_type="tool", name="rag_search")
    def search(self, query: str, source: str | None = None) -> str:
        """Return reranked context with inline citations, ready for the LLM."""
        if not self.has_documents:
            return ("No documents have been uploaded yet. Ask the user to upload "
                    "a document first.")

        effective_source = source or self.active_source   # default to current context
        rewritten = self._rewrite(query)
        docs = self._retrieve(rewritten, effective_source)
        if not docs:
            return "No relevant passages were found in the uploaded documents."

        blocks, citations = [], []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "?")
            page = d.metadata.get("page")
            ts = d.metadata.get("timestamp")
            if isinstance(page, int):
                loc = f" p.{page + 1}"
            elif ts:
                loc = f" @ {ts}"            # YouTube transcript timestamp
            else:
                loc = ""
            tag = f"[S{i}: {src}{loc}]"
            blocks.append(f"{tag}\n{d.page_content.strip()}")
            citations.append(tag)

        context = "\n\n".join(blocks)
        return (
            "Use ONLY the passages below to answer, and cite the [S#] tags you "
            "used.\n\n" + context +
            "\n\nSources: " + ", ".join(citations)
        )


# one shared instance for the whole app
RAG = RagStore()