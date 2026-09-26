#!/bin/bash
# Runner script for CO5151 Seminar Arena Demo
cd "$(dirname "$0")"

echo "============================================================"
echo "🚀 Launching CO5151 Seminar Arena: RAG & Knowledge Grounding"
echo "============================================================"
echo "Corpus: Bộ luật Lao động 2019 (Real Corpus)"
echo "Retriever: BAAI/bge-m3 (Dense SOTA) + BM25 (Sparse) + RRF"
echo "LLM: Qualgo Gateway (gpt-4o-mini) / Local Ollama"
echo "============================================================"

streamlit run app.py --server.port 8501 --server.headless false
