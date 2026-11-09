"""
ZG549 Assignment II - Equity Research Desk Assistant (Streamlit app).

Three cohesive NLP sub-tasks for a NIFTY-50 research desk:
  Tab 1  Sentiment triage of headlines (FinBERT / fine-tuned DistilBERT)
  Tab 2  Desk-ready summaries of news and circulars
  Tab 3  Ask the Regulations - RAG chatbot over NSE/SEBI circulars

Run:  streamlit run app.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import streamlit as st

import models
import rag

CORPUS_DIR = "data/corpus"
INDEX_DIR = "data/chroma"

st.set_page_config(page_title="Equity Research Desk Assistant",
                   page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("Equity Research Desk Assistant")
st.caption("ZG549 Assignment II - Group 32 | NLP: sentiment triage, "
           "summarization, and regulation-grounded QA for NIFTY-50 coverage")

SAMPLE_HEADLINES = [
    "Reliance Industries Q2 profit beats estimates on strong refining margins",
    "HDFC Bank asset quality deteriorates as gross NPAs rise sharply",
    "Infosys retains FY27 revenue guidance; announces share buyback",
    "SEBI imposes penalty on brokerage for margin reporting lapses",
    "Tata Steel announces capacity expansion at Kalinganagar plant",
]

tab1, tab2, tab3 = st.tabs(
    ["1 - Sentiment Triage", "2 - Summarize", "3 - Ask the Regulations (RAG)"]
)

# ----------------------------------------------------------------------
# Tab 1: sentiment
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Headline sentiment triage")
    st.write("Scores overnight headlines so the desk can prioritise coverage "
             "before market open.")

    choices = ["FinBERT (pretrained)"]
    if models.finetuned_available():
        choices.append("Fine-tuned DistilBERT (Section II)")
    model_choice = st.radio("Model", choices, horizontal=True)
    use_ft = model_choice.startswith("Fine-tuned")

    text_in = st.text_area(
        "Headlines (one per line)", value="\n".join(SAMPLE_HEADLINES), height=150
    )
    if st.button("Score sentiment", type="primary"):
        lines = [l.strip() for l in text_in.splitlines() if l.strip()]
        with st.spinner("Scoring..."):
            results = models.sentiment(lines, use_finetuned=use_ft)
        df = pd.DataFrame({
            "headline": lines,
            "sentiment": [r["label"] for r in results],
            "confidence": [r["score"] for r in results],
        })
        st.dataframe(df, width='stretch')
        counts = df["sentiment"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", int(counts.get("positive", 0)))
        c2.metric("Neutral", int(counts.get("neutral", 0)))
        c3.metric("Negative", int(counts.get("negative", 0)))

# ----------------------------------------------------------------------
# Tab 2: summarization
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Desk-ready summary")
    st.write("Condenses a news article or circular into a note the desk can "
             "read in thirty seconds.")
    article = st.text_area("Paste article / circular text", height=260,
                           placeholder="Paste a news article or an exchange "
                                       "circular here...")
    length = st.slider("Target length (words)", 60, 200, 120, step=20)
    if st.button("Summarize", type="primary"):
        if len(article.split()) < 40:
            st.warning("Paste at least a paragraph of text.")
        else:
            with st.spinner("Summarizing..."):
                summary = models.summarize(article, max_words_hint=length)
            st.success(summary)

# ----------------------------------------------------------------------
# Tab 3: RAG chatbot
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Ask the Regulations")
    st.write("Answers desk questions from the ingested NSE/SEBI circular "
             "corpus, with citations. Generator: " + models.generator_info())

    @st.cache_resource
    def get_index():
        return rag.RagIndex(INDEX_DIR)

    index = get_index()

    with st.expander("Corpus / index management"):
        docs = [f for f in sorted(os.listdir(CORPUS_DIR))
                if f.lower().endswith((".pdf", ".txt"))] if os.path.isdir(CORPUS_DIR) else []
        st.write(f"Documents in `{CORPUS_DIR}`: **{len(docs)}** | "
                 f"Indexed chunks: **{index.count()}**")
        if docs:
            st.caption(", ".join(docs))
        if st.button("Rebuild index from corpus"):
            with st.spinner("Ingesting and embedding corpus..."):
                n = index.rebuild(CORPUS_DIR)
            st.success(f"Indexed {n} chunks.")
            st.rerun()

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(content)

    question = st.chat_input("e.g. What are the margin requirements for "
                             "equity derivatives?")
    if question:
        st.session_state.chat.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            if index.count() == 0:
                answer = ("The corpus index is empty. Add PDFs to "
                          f"`{CORPUS_DIR}` and click *Rebuild index*.")
                st.markdown(answer)
            else:
                with st.spinner("Retrieving and generating..."):
                    hits = index.query(question, k=4)
                    prompt = rag.build_rag_prompt(question, hits)
                    answer = models.generate(rag.SYSTEM_PROMPT, prompt)
                st.markdown(answer)
                with st.expander("Retrieved context (evidence)"):
                    for h in hits:
                        st.markdown(f"**[{h.doc}, page {h.page}]** "
                                    f"(distance {h.distance:.3f})")
                        st.caption(h.text[:600] + ("..." if len(h.text) > 600 else ""))
        st.session_state.chat.append(("assistant", answer))
