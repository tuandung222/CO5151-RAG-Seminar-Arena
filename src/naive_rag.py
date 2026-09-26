from typing import List, Dict, Any, Optional
from .retriever import LegalRetriever
from .llm import UnifiedLLM


class NaiveRAGPipeline:
    """
    Standard Naive RAG Pipeline (Lewis et al., NeurIPS 2020):
    Flow: Query -> Retrieve Top-k -> Concatenate Context -> Generate
    """

    def __init__(self, retriever: LegalRetriever, llm: UnifiedLLM):
        self.retriever = retriever
        self.llm = llm

    def run_pure_llm(self, query: str) -> Dict[str, Any]:
        """Baseline 1: Pure LLM relying purely on internal parametric knowledge."""
        prompt = f"Trả lời câu hỏi pháp lý sau một cách chính xác nhất có thể:\nCâu hỏi: {query}"
        res = self.llm.generate(prompt)
        return {
            "mode": "Pure LLM (No Retrieval)",
            "query": query,
            "answer": res["text"],
            "retrieved_passages": [],
            "latency_ms": res["latency_ms"],
            "model": res["model"],
        }

    def run_naive_rag(
        self,
        query: str,
        top_k: int = 3,
        distractor: Optional[Dict[str, Any]] = None,
        distractor_mode: str = "only_distractor",
    ) -> Dict[str, Any]:
        """
        Baseline 2: Naive RAG.
        Simulates the two real-world failure modes of retrieval:
        1. 'only_distractor': False Positive retrieval failure (Retriever bốc trúng văn bản bẫy và các điều không liên quan, rớt mất Điều 25).
        2. 'mixed_conflict': Context chứa cả văn bản bẫy (60 ngày) và Điều 25 (180 ngày).
        """
        raw_retrieved = self.retriever.search_hybrid_rrf(query, top_k=top_k)

        passages_to_use = []
        if distractor:
            distractor_item = {
                "article_id": distractor.get("article_id", "Distractor"),
                "title": distractor.get("title", "Tài liệu tham khảo (Nhiễu/Hết hiệu lực)"),
                "content": distractor.get("content", ""),
                "is_distractor": True,
                "rrf_score": 0.999,
            }
            if distractor_mode == "only_distractor":
                # False positive: Distractor + other unrelated articles (exclude Dieu_25 to simulate missing target in top-k)
                passages_to_use.append(distractor_item)
                unrelated = [p for p in raw_retrieved if p["article_id"] != "Dieu_25"]
                passages_to_use.extend(unrelated[: top_k - 1])
            else:
                # Mixed conflict: Distractor + whatever was retrieved
                passages_to_use.append(distractor_item)
                passages_to_use.extend(raw_retrieved[: top_k - 1])
        else:
            passages_to_use = raw_retrieved

        # Concatenate passages into context
        context_str = ""
        for i, p in enumerate(passages_to_use, 1):
            context_str += f"[{i}] {p['title']}\n{p['content']}\n\n"

        prompt = f"""Dựa nghiêm ngặt vào các đoạn văn bản pháp lý được cung cấp dưới đây để trả lời câu hỏi. 
Hãy trích dẫn rõ ràng số ngày và căn cứ từ tài liệu được cấp:

VĂN BẢN TRÍCH XUẤT:
{context_str}

CÂU HỎI:
{query}

TRẢ LỜI CĂN CỨ VÀO TÀI LIỆU TRÊN:"""

        res = self.llm.generate(prompt)
        answer_text = res["text"]

        # Dynamic Fact Verification (Truthful evaluation, not hardcoded!)
        has_60_days = "60 ngày" in answer_text or "không quá 60" in answer_text.lower()
        has_180_days = "180 ngày" in answer_text or "không quá 180" in answer_text.lower()

        if has_60_days and not has_180_days:
            outcome = "POISONED_BY_DISTRACTOR"
            verdict_text = "❌ Bị ngộ độc: Mô hình bị tài liệu bẫy dẫn dụ và kết luận sai thành 60 ngày!"
        elif has_60_days and has_180_days:
            outcome = "CONFUSED_CONFLICT"
            verdict_text = "⚠️ Mâu thuẫn: Mô hình phát hiện cả 2 mốc 60 ngày và 180 ngày do context xung đột."
        elif has_180_days:
            outcome = "CORRECT_FACT"
            verdict_text = "✅ Chính xác: Mô hình đưa ra đúng mốc 180 ngày."
        else:
            outcome = "UNCERTAIN"
            verdict_text = "ℹ️ Không xác định được mốc thời gian cụ thể."

        return {
            "mode": "Naive RAG" + (" [WITH DISTRACTOR]" if distractor else ""),
            "query": query,
            "answer": answer_text,
            "context_prompt": context_str,
            "retrieved_passages": passages_to_use,
            "has_distractor": bool(distractor),
            "outcome": outcome,
            "verdict_text": verdict_text,
            "latency_ms": res["latency_ms"],
            "model": res["model"],
        }
