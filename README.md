# ZG549 Assignment II - Equity Research Desk Assistant

**Group 32 | AIMLZG549 API-driven Cloud Native Solutions**

An API-based AI application for an equity research desk covering NIFTY-50
stocks. Three cohesive NLP sub-tasks serve one workflow, the desk's morning
triage of news and regulatory circulars:

| Section | Component | Model |
|---|---|---|
| I - Sub-task 1 | Headline sentiment triage | FinBERT (`ProsusAI/finbert`, finance SLM) |
| I - Sub-task 2 | Desk-ready summarization | DistilBART (`sshleifer/distilbart-cnn-12-6`) |
| I - Sub-task 3 + III | RAG chatbot over NSE/SEBI circulars | MiniLM embeddings + ChromaDB + Groq-hosted Llama (local FLAN-T5 fallback) |
| II - Fine-tuning | Sentiment model fine-tuned on Financial PhraseBank | `distilbert-base-uncased` -> `models/finetuned-sentiment` |

Continuity with Assignment I: A1 built the quantitative pipeline for this
desk (volatility regime classification from prices); A2 adds the text layer
(news and regulation intelligence) for the same desk.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

python test\smoke_test.py          # verifies RAG plumbing, no model downloads
```

**Groq API key (free, no card):** create a key at console.groq.com/keys, then

```powershell
$env:GROQ_API_KEY = "gsk_..."      # per session, before streamlit run
# optional: $env:GROQ_MODEL = "llama-3.3-70b-versatile"
```

Without the key the app still works using a local FLAN-T5 generator
(noticeably weaker answers; fine for development, use Groq for the demo).

## Corpus (Section III)

Download 3-6 public PDFs into `data/corpus/`, e.g.:
- SEBI master circular for stock brokers (sebi.gov.in -> Legal -> Master Circulars)
- NSE circulars on margin / circuit breakers (nseindia.com -> Circulars)
- An RBI monetary policy statement (optional macro context)

Then in the app's Tab 3, open *Corpus / index management* -> **Rebuild index**.

## Fine-tuning (Section II)

Open `finetune/finetune_phrasebank.ipynb` in Google Colab (T4 GPU runtime),
Run All (~10 min). It fine-tunes DistilBERT on Financial PhraseBank
(`sentences_allagree`), compares against the FinBERT baseline, and downloads
`finetuned-sentiment.zip`. Unzip into `models/finetuned-sentiment/`; the
Sentiment tab then offers the fine-tuned model in the model picker.

## Run the app

```bash
streamlit run app.py
```

First use of each tab downloads its model from Hugging Face (one-time,
~250-500 MB each); subsequent runs are cached.

## Screenshot checklist (rubric-mapped)

- [ ] I.7  App home with the three tabs visible
- [ ] I.3a Sentiment tab: scored headlines table + positive/neutral/negative metrics
- [ ] I.3b Summarize tab: article in, desk note out
- [ ] II   Colab: training log, baseline-vs-fine-tuned accuracy, confusion matrix
- [ ] II   Sentiment tab with the fine-tuned model selected in the picker
- [ ] III  RAG tab: corpus listed + chunk count after Rebuild index
- [ ] III  RAG answer with inline [doc, page] citations + expanded retrieved-context evidence
- [ ] IV   Representative workflow run end-to-end (headlines -> summary -> regulation query)
