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
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=Streamlit&logoColor=white)](https://streamlit.io)

An empirical testbed evaluating **5 scientific RAG paradigms** (Pure LLM, Naive RAG, Self-RAG, GraphRAG, FLARE) against authentic statutory reasoning (Vietnamese Labor Code 2019 vs. obsolete 2012 distractors).

> **Master's Student Seminar Project** — Course **CO5151: Advanced Agentic AI**, Ho Chi Minh City University of Technology (HCMUT).  
> **Presenter:** Dung Vo ([@tuandung222](https://github.com/tuandung222))

---

## 🎯 5 RAG Paradigms Evaluated

| Paradigm | Core Operational Mechanism | Primary Failure Mode Addressed | Paper / Reference |
|---|---|---|---|
| **Pure LLM** | Parametric weights only | Baseline (zero external retrieval) | Baseline |
| **Naive RAG** | Top-k dense semantic vector retrieval | In-context knowledge injection | Lewis et al. (NeurIPS 2020) |
| **Self-RAG** | Self-reflection critic tokens (`[Retrieve]`, `[IsREL]`, `[IsSUP]`) | Context poisoning from outdated distractors | Asai et al. (ICLR 2024) |
| **GraphRAG** | Statutory Knowledge Graph & Modularity Map-Reduce | Local blindness across statutory chapters | Edge et al. (Microsoft 2024) |
| **FLARE** | Active retrieval triggered by low token confidence | Excessive retrieval latency & overhead | Jiang et al. (EMNLP 2023) |

---

## 🔗 Quick Links

- 🌐 **Live Web Application**: [tuandunghcmut-co5151-rag-seminar-arena.hf.space](https://tuandunghcmut-co5151-rag-seminar-arena.hf.space)
- 🤗 **Hugging Face Space**: [tuandunghcmut/CO5151-RAG-Seminar-Arena](https://huggingface.co/spaces/tuandunghcmut/CO5151-RAG-Seminar-Arena)
- 📑 **Technical Report & Blueprint**: [`SEMINAR_DEMO_TECHNICAL_REPORT.md`](./SEMINAR_DEMO_TECHNICAL_REPORT.md)

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
Access the application at `http://localhost:8501` (or port `7860` on Hugging Face Spaces).

---

## 👤 Author & Academic Context

- **Presenter:** Dung Vo ([@tuandung222](https://github.com/tuandung222))
- **Program:** Master's in Computer Science, Ho Chi Minh City University of Technology (HCMUT)
- **Course:** CO5151 Advanced Agentic AI
