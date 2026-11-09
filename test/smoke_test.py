"""Smoke test for the RAG plumbing - runs WITHOUT model downloads.

Verifies: corpus loading (.txt), chunking, Chroma ingest, retrieval
relevance (hash embedder), prompt assembly, and the generator fallback
selection logic. Run the real stack via the Streamlit app.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import models
import rag

tmp = tempfile.mkdtemp()
corpus = os.path.join(tmp, "corpus")
os.makedirs(corpus)

DOCS = {
    "sebi_margin_circular.txt": (
        "SEBI circular on margin requirements for the equity derivatives "
        "segment. Initial margin shall be collected upfront from clients. "
        "The margin framework covers SPAN margin, exposure margin and "
        "peak margin reporting obligations for trading members. " * 6
    ),
    "nse_circuit_filters.txt": (
        "NSE circular describing index circuit breakers. Trading halts "
        "are triggered at 10 percent, 15 percent and 20 percent moves in "
        "the NIFTY 50 index, with halt durations depending on the time "
        "of day the limit is breached. " * 6
    ),
    "rbi_repo_policy.txt": (
        "Monetary policy statement. The repo rate is the rate at which "
        "the central bank lends to commercial banks against government "
        "securities. Changes in the repo rate transmit to bond yields "
        "and equity valuations. " * 6
    ),
}
for name, text in DOCS.items():
    with open(os.path.join(corpus, name), "w") as f:
        f.write(text)

# chunking
chunks = rag.chunk_text("word " * 2000)
assert all(len(c) <= rag.CHUNK_CHARS + 50 for c in chunks)
assert len(chunks) > 3
print(f"chunking: {len(chunks)} chunks from 2000 words")

records = rag.load_corpus(corpus)
assert len(records) == 3
built = rag.build_chunks(records)
assert all({"doc", "page"} <= set(c["metadata"]) for c in built)
print(f"corpus: {len(records)} docs -> {len(built)} chunks")

# index + retrieval with deterministic embedder
index = rag.RagIndex(os.path.join(tmp, "chroma"), embedder=rag.HashEmbedder())
n = index.rebuild(corpus)
assert n == index.count() == len(built)

hits = index.query("What are the margin requirements for derivatives?", k=2)
assert hits and hits[0].doc == "sebi_margin_circular.txt", hits[0].doc
hits2 = index.query("When does a trading halt trigger on the index?", k=2)
assert hits2[0].doc == "nse_circuit_filters.txt", hits2[0].doc
hits3 = index.query("What is the repo rate?", k=2)
assert hits3[0].doc == "rbi_repo_policy.txt", hits3[0].doc
print("retrieval: 3/3 queries hit the right document")

# prompt assembly
prompt = rag.build_rag_prompt("What is the repo rate?", hits3)
assert "[rbi_repo_policy.txt, page 1]" in prompt and "What is the repo rate?" in prompt
print("prompt assembly: citations present")

# generator routing (no network, no key -> local path selected)
os.environ.pop("GROQ_API_KEY", None)
assert "fallback" in models.generator_info().lower()
os.environ["GROQ_API_KEY"] = "gsk_dummy"
assert "Groq" in models.generator_info()
os.environ.pop("GROQ_API_KEY", None)
assert not models.finetuned_available()
print("generator routing: key/no-key paths correct")

shutil.rmtree(tmp, ignore_errors=True)  # chroma holds files open on Windows
print("\nSMOKE TEST PASSED")
