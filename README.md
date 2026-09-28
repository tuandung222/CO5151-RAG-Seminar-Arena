---
title: CO5151 RAG Paradigm Inspector
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ⚖️ CO5151 RAG Paradigm Inspector

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space%20Live-yellow.svg)](https://huggingface.co/spaces/tuandunghcmut/CO5151-RAG-Seminar-Arena)
[![GitHub Repo](https://img.shields.io/badge/GitHub-tuandung222%2FCO5151--RAG--Seminar--Arena-blue?logo=github)](https://github.com/tuandung222/CO5151-RAG-Seminar-Arena)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=Streamlit&logoColor=white)](https://streamlit.io)

Interactive empirical benchmark and diagnostic arena comparing **5 RAG paradigms** on authentic legal reasoning (Vietnamese Labor Code 2019 vs. obsolete 2012 distractors).

Built for a student seminar from **CO5151 (Advanced Agentic AI)** at **HCMUT (ĐHBK ĐHQG-HCM)**.

---

## 🎯 5 RAG Paradigms Evaluated

| Paradigm | Key Mechanism | Paper Citation |
|---|---|---|
| **Pure LLM** | Direct generation without external knowledge grounding | Baseline |
| **Naive RAG** | Top-k dense semantic vector retrieval | Lewis et al. (NeurIPS 2020) |
| **Modular RAG** | Dense + BM25 reciprocal rank fusion + Cross-Encoder reranking | Gao et al. (2023) |
| **Self-RAG** | Self-reflection critique tokens (`[Retrieve]`, `[IsREL]`, `[IsSUP]`, `[IsUSE]`) | Asai et al. (ICLR 2024) |
| **GraphRAG** | Knowledge Graph extraction & hierarchical community summary synthesis | Edge et al. (Microsoft 2024) |
| **FLARE** | Forward-looking active retrieval triggered by low token confidence | Jiang et al. (EMNLP 2023) |

---

## 🔗 Quick Links

- 🌐 **Live Web App**: [tuandunghcmut-co5151-rag-seminar-arena.hf.space](https://tuandunghcmut-co5151-rag-seminar-arena.hf.space)
- 🤗 **Hugging Face Space**: [tuandunghcmut/CO5151-RAG-Seminar-Arena](https://huggingface.co/spaces/tuandunghcmut/CO5151-RAG-Seminar-Arena)
- 📑 **Slide & Technical Report**: [`SEMINAR_DEMO_TECHNICAL_REPORT.md`](./SEMINAR_DEMO_TECHNICAL_REPORT.md)

---

## 🚀 Quickstart

### Local Setup
```bash
git clone https://github.com/tuandung222/CO5151-RAG-Seminar-Arena.git
cd CO5151-RAG-Seminar-Arena
pip install -r requirements.txt
streamlit run app.py
```

### Docker
```bash
docker compose up -d
```
Access the application at `http://localhost:8501`.

---
