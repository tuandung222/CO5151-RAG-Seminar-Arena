import json
import time
from typing import List, Dict, Any, Tuple
from .retriever import LegalRetriever
from .llm import UnifiedLLM


class FLARERAGPipeline:
    """
    Active Retrieval / FLARE Pipeline (Jiang et al., EMNLP 2023):
    Forward-Looking Active REtrieval:
    - LLM drafts the next proposition / sentence.
    - Inspect confidence / factual certainty of the proposition.
    - HIGH confidence (general discourse / logic) -> Keep draft, NO retrieval.
    - LOW confidence (specific legal quantitative thresholds, days, penalties) -> Trigger on-demand retrieval -> Regenerate with grounding.
    """

    def __init__(self, retriever: LegalRetriever, llm: UnifiedLLM):
        self.retriever = retriever
        self.llm = llm

    def run_flare(self, query: str, theta: float = 0.5) -> Dict[str, Any]:
        """Execute FLARE active retrieval loop on query with confidence threshold theta."""
        start_flare_t = time.time()
        # Step 1: Generate initial draft sentences
        draft_prompt = f"""Hãy chia nhỏ câu trả lời cho câu hỏi sau thành 2-3 câu văn ngắn gọn:
CÂU HỎI: {query}

ĐỊNH DẠNG: Mỗi câu trên 1 dòng, không đánh số:"""
        draft_res = self.llm.generate(draft_prompt, temperature=0.1)
        raw_sentences = [s.strip() for s in draft_res["text"].split("\n") if s.strip()]

        trace_steps = []
        final_sentences = []
        retrieval_calls = 0

        for i, sentence in enumerate(raw_sentences, 1):
            # Check if sentence contains specific quantitative thresholds, sanctions, or factual claims
            check_prompt = f"""Đánh giá xem câu văn sau đây có chứa khẳng định về con số cụ thể, thời hạn, tỷ lệ phần trăm hoặc mức bồi thường pháp luật mà cần phải kiểm chứng qua văn bản pháp quy hay không:
CÂU VĂN: "{sentence}"

Trả về JSON:
{{
  "requires_retrieval": true hoặc false,
  "confidence_score": số từ 0.0 đến 1.0 (trên 0.8 là rất chắc chắn, dưới 0.8 là cần tra cứu),
  "search_query": "từ khóa chính xác cần tra cứu nếu cần",
  "reason": "lý do"
}}
Chỉ trả về JSON:"""

            eval_res = self.llm.generate(check_prompt, temperature=0.0, max_tokens=150)
            try:
                cleaned = eval_res["text"].replace("```json", "").replace("```", "").strip()
                eval_data = json.loads(cleaned)
                req_retrieval = eval_data.get("requires_retrieval", False)
                confidence = float(eval_data.get("confidence_score", 0.9))
                sub_query = eval_data.get("search_query", query)
            except Exception:
                req_retrieval = ("ngày" in sentence or "%" in sentence or "bồi thường" in sentence)
                confidence = 0.6 if req_retrieval else 0.95
                sub_query = query

            if req_retrieval and confidence < theta:
                # Active retrieval triggered!
                retrieval_calls += 1
                evidence_passages = self.retriever.search_hybrid_rrf(sub_query, top_k=2)
                evidence_context = "\n".join([f"{p['title']}: {p['content']}" for p in evidence_passages])

                rewrite_prompt = f"""Viết lại câu văn sau đây cho chuẩn xác và có căn cứ trích dẫn rõ ràng dựa trên văn bản pháp lý:
CÂU GỐC: {sentence}
VĂN BẢN TRÍCH XUẤT:
{evidence_context}

CHỈ VIẾT LẠI ĐÚNG 01 CÂU HOÀN CHỈNH:"""
                rewritten_res = self.llm.generate(rewrite_prompt, temperature=0.0)
                verified_sentence = rewritten_res["text"].strip().strip('"')

                trace_steps.append({
                    "step": i,
                    "draft_sentence": sentence,
                    "confidence": confidence,
                    "status": "TRIGGERED_RETRIEVAL",
                    "search_query": sub_query,
                    "evidence_used": [p["title"] for p in evidence_passages],
                    "final_sentence": verified_sentence,
                })
                final_sentences.append(verified_sentence)
            else:
                # High confidence, keep draft directly without wasting retrieval
                trace_steps.append({
                    "step": i,
                    "draft_sentence": sentence,
                    "confidence": confidence,
                    "status": "CONFIDENT_NO_RETRIEVAL",
                    "search_query": None,
                    "evidence_used": [],
                    "final_sentence": sentence,
                })
                final_sentences.append(sentence)

        full_answer = " ".join(final_sentences)

        return {
            "mode": "Active Retrieval / FLARE (Forward-Looking Uncertainty Triggered)",
            "query": query,
            "total_sentences": len(raw_sentences),
            "retrieval_calls_made": retrieval_calls,
            "trace_steps": trace_steps,
            "final_answer": full_answer,
            "latency_ms": round((time.time() - start_flare_t) * 1000, 2),
        }
