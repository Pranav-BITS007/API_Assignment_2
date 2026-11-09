"""
ZG549 Assignment II - model layer (Sections I, II, III).

Three NLP sub-tasks for the equity research desk assistant:
  1. Sentiment of financial headlines  -> FinBERT (pretrained SLM) and,
     when present, the fine-tuned DistilBERT from Section II
     (models/finetuned-sentiment, produced by the Colab notebook).
  2. Summarization of news/circulars   -> DistilBART CNN.
  3. Grounded QA (the RAG chatbot)     -> Groq-hosted Llama via the
     OpenAI-compatible API, with a local FLAN-T5 fallback when no
     GROQ_API_KEY is set.

All transformers imports are lazy so the module imports cheaply and the
RAG smoke test never pulls model weights.
"""

from __future__ import annotations

import os

FINBERT = "ProsusAI/finbert"
SUMMARIZER = "sshleifer/distilbart-cnn-12-6"
LOCAL_GENERATOR = "google/flan-t5-base"
FINETUNED_DIR = os.path.join("models", "finetuned-sentiment")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE = "https://api.groq.com/openai/v1"

_cache: dict = {}


def _pipeline(task: str, model: str):
    key = (task, model)
    if key not in _cache:
        from transformers import pipeline

        _cache[key] = pipeline(task, model=model)
    return _cache[key]


# ----------------------------------------------------------------------
# Sub-task 1: sentiment
# ----------------------------------------------------------------------

def finetuned_available() -> bool:
    return os.path.isdir(FINETUNED_DIR) and any(
        f.startswith("model") or f.endswith(".safetensors")
        for f in os.listdir(FINETUNED_DIR)
    )


def sentiment(texts: list[str], use_finetuned: bool = False) -> list[dict]:
    """Return [{'label': 'positive|negative|neutral', 'score': float}]."""
    model = FINETUNED_DIR if (use_finetuned and finetuned_available()) else FINBERT
    pipe = _pipeline("text-classification", model)
    out = []
    for res in pipe(texts, truncation=True):
        out.append({"label": res["label"].lower(), "score": round(res["score"], 4)})
    return out


# ----------------------------------------------------------------------
# Sub-task 2: summarization
# ----------------------------------------------------------------------

def summarize(text: str, max_words_hint: int = 120) -> str:
    pipe = _pipeline("summarization", SUMMARIZER)
    n_tokens = max(48, min(int(max_words_hint * 1.4), 256))
    res = pipe(text[:4000], max_length=n_tokens, min_length=max(24, n_tokens // 3),
               do_sample=False)
    return res[0]["summary_text"].strip()


# ----------------------------------------------------------------------
# Sub-task 3: generation for RAG (hosted API with local fallback)
# ----------------------------------------------------------------------

def generator_info() -> str:
    if os.environ.get("GROQ_API_KEY"):
        return f"Groq hosted API ({GROQ_MODEL})"
    return f"Local fallback ({LOCAL_GENERATOR}) - set GROQ_API_KEY for better answers"


def generate(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        try:
            return _generate_groq(api_key, system_prompt, user_prompt, max_tokens)
        except Exception as e:  # rate limit, model rename, network
            return (f"[Groq call failed: {e}. Falling back to local model.]\n\n"
                    + _generate_local(system_prompt, user_prompt, max_tokens))
    return _generate_local(system_prompt, user_prompt, max_tokens)


def _generate_groq(api_key: str, system_prompt: str, user_prompt: str,
                   max_tokens: int) -> str:
    """OpenAI-compatible chat completion against Groq."""
    import requests

    resp = requests.post(
        f"{GROQ_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _generate_local(system_prompt: str, user_prompt: str,
                    max_tokens: int) -> str:
    pipe = _pipeline("text2text-generation", LOCAL_GENERATOR)
    prompt = f"{system_prompt}\n\n{user_prompt}"
    res = pipe(prompt[:3500], max_new_tokens=min(max_tokens, 256))
    return res[0]["generated_text"].strip()
