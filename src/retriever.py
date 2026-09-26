import json
import re
import urllib.request
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import numpy as np

from huggingface_hub import InferenceClient
from .config import (
    CORPUS_PATH,
    CORPUS_EMBEDDINGS_PATH,
    CORPUS_EMBEDDINGS_META_PATH,
    HF_TOKEN,
    HF_ROUTER_URL,
    USER_AGENT,
)


def safe_log(msg: str):
    """Safely log messages without crashing on headless/detached stdout broken pipes."""
    try:
        print(msg, flush=True)
    except Exception:
        pass


def tokenize_vietnamese(text: str) -> List[str]:
    """Lightweight whitespace and punctuation tokenizer for Vietnamese words."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if len(w) > 0]


class LegalRetriever:
    """
    State-of-the-art Production Hybrid Retriever:
    1. Offline Pre-computed Dense Matrix (BAAI/bge-m3, 1024-dim, L2-normalized)
    2. Online Single-Query Embedding & Ultra-low-latency Cosine Dot Product (< 0.1ms)
    3. Sparse Lexical BM25 (Okapi)
    4. Reciprocal Rank Fusion (RRF, Cormack et al. 2009)
    """

    def __init__(self, corpus_path=CORPUS_PATH, api_key: str = HF_TOKEN):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus: List[Dict[str, Any]] = json.load(f)

        self.api_key = api_key or HF_TOKEN
        self.doc_ids = [doc["article_id"] for doc in self.corpus]
        self.doc_map = {doc["article_id"]: doc for doc in self.corpus}
        self.raw_texts = [
            f"{doc.get('title', '')}: {doc.get('content', '')}" for doc in self.corpus
        ]

        # 1. Initialize BM25
        self.tokenized_corpus = [tokenize_vietnamese(txt) for txt in self.raw_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 2. Initialize Hugging Face Inference Client
        self.hf_client = InferenceClient(api_key=self.api_key) if self.api_key else None

        # 3. Load Offline Pre-computed Dense Embeddings Matrix (15 x 1024)
        self.corpus_embeddings = None
        self.corpus_embeddings_meta = {}
        if CORPUS_EMBEDDINGS_PATH.exists():
            try:
                self.corpus_embeddings = np.load(CORPUS_EMBEDDINGS_PATH)
                safe_log(f"[Retriever] Loaded offline pre-computed embeddings: {self.corpus_embeddings.shape}")
            except Exception as e:
                safe_log(f"[Retriever Warning] Failed loading pre-computed embeddings: {e}")

        if CORPUS_EMBEDDINGS_META_PATH.exists():
            try:
                with open(CORPUS_EMBEDDINGS_META_PATH, "r", encoding="utf-8") as f:
                    self.corpus_embeddings_meta = json.load(f)
            except Exception:
                pass

    def update_api_key(self, api_key: str):
        """Update Hugging Face token dynamically across clients."""
        if api_key:
            self.api_key = api_key
            self.hf_client = InferenceClient(api_key=api_key)

    def search_bm25(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Exact keyword lexical search via Okapi BM25."""
        tokenized_q = tokenize_vietnamese(query)
        if not tokenized_q:
            return []
        scores = self.bm25.get_scores(tokenized_q)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self.doc_ids[idx], float(scores[idx])))
        return results

    def search_dense_bge_m3(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Production-grade Dense Semantic Search:
        - Query is encoded into 1024-dim vector using BAAI/bge-m3.
        - Dot product against offline pre-indexed matrix in RAM (< 0.1ms).
        - Falls back gracefully to router or BM25 if offline.
        """
        if not query.strip():
            return []

        # Option A: Fast Vector Dot Product using Pre-computed Matrix
        if self.corpus_embeddings is not None and self.hf_client is not None:
            try:
                q_vec = self.hf_client.feature_extraction(query, model="BAAI/bge-m3")
                q_vec = np.array(q_vec, dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm

                # Instant cosine similarity via matrix-vector multiplication (N, d) @ (d,)
                scores = np.dot(self.corpus_embeddings, q_vec)
                top_indices = np.argsort(scores)[::-1][:top_k]
                return [
                    (self.doc_ids[idx], float(scores[idx]))
                    for idx in top_indices
                ]
            except Exception as e:
                safe_log(f"[Retriever Warning] Single-query feature extraction failed ({e}), falling back to router...")

        # Option B: Router Pairwise Fallback
        payload = json.dumps({
            "inputs": {
                "source_sentence": query,
                "sentences": self.raw_texts,
            }
        }).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        try:
            req = urllib.request.Request(
                HF_ROUTER_URL, data=payload, headers=headers
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                scores = json.loads(resp.read().decode("utf-8"))
                if isinstance(scores, list) and len(scores) == len(self.doc_ids):
                    top_indices = np.argsort(scores)[::-1][:top_k]
                    return [
                        (self.doc_ids[idx], float(scores[idx]))
                        for idx in top_indices
                    ]
        except Exception as e:
            safe_log(f"[Retriever Warning] HF BGE-M3 router failed ({e}), falling back to BM25...")

        # Option C: Local fallback based on BM25 normalized
        bm25_res = self.search_bm25(query, top_k=top_k)
        if bm25_res:
            max_s = max(s for _, s in bm25_res) or 1.0
            return [(d, s / max_s) for d, s in bm25_res]
        return []

    def search_hybrid_rrf(
        self, query: str, top_k: int = 5, rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion combining BM25 and Dense BGE-M3.
        RRF Score(d) = 1/(k + rank_bm25) + 1/(k + rank_dense)
        """
        bm25_hits = self.search_bm25(query, top_k=len(self.doc_ids))
        dense_hits = self.search_dense_bge_m3(query, top_k=len(self.doc_ids))

        rrf_scores: Dict[str, float] = {doc_id: 0.0 for doc_id in self.doc_ids}
        bm25_rank_map = {doc_id: r for r, (doc_id, _) in enumerate(bm25_hits)}
        dense_rank_map = {doc_id: r for r, (doc_id, _) in enumerate(dense_hits)}

        for doc_id in self.doc_ids:
            if doc_id in bm25_rank_map:
                rrf_scores[doc_id] += 1.0 / (rrf_k + bm25_rank_map[doc_id] + 1)
            if doc_id in dense_rank_map:
                rrf_scores[doc_id] += 1.0 / (rrf_k + dense_rank_map[doc_id] + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in ranked:
            doc = self.doc_map[doc_id]
            results.append({
                "article_id": doc_id,
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "rrf_score": round(score, 4),
                "bm25_rank": bm25_rank_map.get(doc_id, -1) + 1,
                "dense_rank": dense_rank_map.get(doc_id, -1) + 1,
            })
        return results

    def get_article(self, article_id: str) -> Dict[str, Any]:
        return self.doc_map.get(article_id, {})
