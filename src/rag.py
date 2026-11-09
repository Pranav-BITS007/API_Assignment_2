"""
ZG549 Assignment II - RAG component (Section III).

Corpus -> chunks -> ChromaDB index -> retrieval. Pure logic, no Streamlit.
The embedding function is injectable: the app uses sentence-transformers
(all-MiniLM-L6-v2); tests inject a deterministic hash embedder so the
plumbing is verifiable without model downloads.

Corpus: public NSE/SEBI circulars and related market documents (PDF or
plain text) dropped into data/corpus/.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

import chromadb

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
COLLECTION = "zg549_corpus"


# ----------------------------------------------------------------------
# Document loading
# ----------------------------------------------------------------------

def _read_pdf(path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for a PDF."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def _read_txt(path: str) -> list[tuple[int, str]]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [(1, f.read())]


def load_corpus(corpus_dir: str) -> list[dict]:
    """Load every .pdf/.txt in corpus_dir into page records."""
    records = []
    for name in sorted(os.listdir(corpus_dir)):
        path = os.path.join(corpus_dir, name)
        if name.lower().endswith(".pdf"):
            pages = _read_pdf(path)
        elif name.lower().endswith(".txt"):
            pages = _read_txt(path)
        else:
            continue
        for page_no, text in pages:
            records.append({"doc": name, "page": page_no, "text": text})
    return records


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Character-window chunking with overlap, snapped to whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def build_chunks(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        for j, chunk in enumerate(chunk_text(rec["text"])):
            out.append({
                "id": f"{rec['doc']}:p{rec['page']}:c{j}",
                "text": chunk,
                "metadata": {"doc": rec["doc"], "page": rec["page"]},
            })
    return out


# ----------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------

class HashEmbedder(chromadb.EmbeddingFunction):
    """Deterministic bag-of-words hash embedding. TEST USE ONLY —
    lets the Chroma round-trip be verified without network access."""

    def __init__(self, dims: int = 128):
        self.dims = dims

    def __call__(self, input: list[str]) -> list[list[float]]:  # chroma iface
        vecs = []
        for text in input:
            v = [0.0] * self.dims
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dims] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vecs.append([x / norm for x in v])
        return vecs

    def name(self) -> str:  # chromadb >= 0.5 asks for this
        return "hash-test-embedder"


def default_embedder():
    """sentence-transformers MiniLM — the real embedder for the app."""
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ----------------------------------------------------------------------
# Index + retrieval
# ----------------------------------------------------------------------

@dataclass
class Retrieved:
    text: str
    doc: str
    page: int
    distance: float


class RagIndex:
    def __init__(self, persist_dir: str, embedder=None):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = embedder or default_embedder()

    def _collection(self):
        return self.client.get_or_create_collection(
            COLLECTION, embedding_function=self.embedder
        )

    def rebuild(self, corpus_dir: str) -> int:
        """Drop and re-ingest the corpus. Returns chunk count."""
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = self._collection()
        chunks = build_chunks(load_corpus(corpus_dir))
        if not chunks:
            return 0
        B = 100
        for i in range(0, len(chunks), B):
            batch = chunks[i:i + B]
            col.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )
        return len(chunks)

    def count(self) -> int:
        return self._collection().count()

    def query(self, question: str, k: int = 4) -> list[Retrieved]:
        col = self._collection()
        if col.count() == 0:
            return []
        res = col.query(query_texts=[question], n_results=min(k, col.count()))
        out = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(Retrieved(text=text, doc=meta["doc"],
                                 page=int(meta["page"]), distance=dist))
        return out


# ----------------------------------------------------------------------
# Prompt assembly (used by the generator in models.py)
# ----------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a research assistant for an equity research desk covering "
    "NIFTY-50 stocks. Answer ONLY from the provided context extracts. "
    "Cite sources inline as [doc, page N]. If the context does not "
    "contain the answer, say so plainly instead of guessing."
)


def build_rag_prompt(question: str, retrieved: list[Retrieved]) -> str:
    ctx = "\n\n".join(
        f"[{r.doc}, page {r.page}]\n{r.text}" for r in retrieved
    )
    return (
        f"Context extracts:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        "Answer (grounded in the context, with citations):"
    )
