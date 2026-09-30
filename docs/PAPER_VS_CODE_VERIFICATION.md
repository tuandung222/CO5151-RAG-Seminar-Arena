# 🔬 Báo Cáo Đối Chiếu: Mã Nguồn Demo vs. Papers Gốc

**Ngày**: 30/09/2026 | **4 agents đã đọc kỹ cả paper lẫn code**

---

## Tổng Quan Nhanh

| Pipeline | Mức độ trung thành | Bản chất |
|:---|:---:|:---|
| **Naive RAG** | ⚠️ Surrogate | Prompt-based context injection, không phải RAG-Sequence/Token |
| **GraphRAG** | ⚠️ Toy Demo | Graph hand-crafted, Greedy Modularity thay Leiden, map-reduce đơn giản |
| **Self-RAG** | ⚠️ In-Context Surrogate | Zero-shot prompting thay trained critic, có disclaimer ✅ |
| **FLARE** | ❌ Post-hoc Verifier | Verbalized confidence thay token logprobs, **UI ghi sai** ❌ |

---

## 1. Naive RAG vs. Lewis et al. 2020

| Component | Paper | Code | Verdict |
|:---|:---|:---|:---:|
| Retriever | DPR bi-encoder | BM25 + BGE-M3 Hybrid RRF | ❌ |
| Generator | BART-large (seq2seq) | Generic LLM via `UnifiedLLM` | ❌ |
| RAG-Seq/Token | 2 variants with marginalization | Simple context concatenation | ❌ |
| Marginalization | $P(y\|x) = \sum_z P(y\|x,z)P(z\|x)$ | Không có | ❌ |
| Training | Joint retriever + generator | Zero training | ❌ |
| Top-k | k=5 or k=10 | k=3 | ⚠️ |
| Distractor injection | Không có | 2 modes: only_distractor, mixed_conflict | 📚 |
| Verdict logic | Không có | Keyword matching (60/180 ngày) | 📚 |

> **Kết luận**: Demo minh họa **concept** "retrieve → concatenate → generate" đúng tinh thần, nhưng không reproduce paper. Distractor injection là **sáng tạo sư phạm** để demo "When Retrieval Hurts".

---

## 2. GraphRAG vs. Edge et al. 2024

| Component | Paper | Code | Verdict |
|:---|:---|:---|:---:|
| Text Chunking | 600-token overlapping chunks | Dùng thẳng legal articles (pre-chunked) | ⚠️ |
| Entity/Rel Extraction | LLM + domain prompts + self-reflection | **Hardcoded** 15 edges tĩnh | ❌ |
| Graph Construction | LLM-extracted → NetworkX/graspologic | NetworkX nhưng hand-crafted | ⚠️ |
| Community Detection | **Leiden** (hierarchical) | **Greedy Modularity** (non-hierarchical) | ❌ |
| Community Summarization | Pre-generated at indexing time | Skipped — sinh on-the-fly tại query time | ❌ |
| Map-Reduce | Parallel map + helpfulness scoring + reduce | Sequential map, **no scoring/filtering** | ⚠️ |
| Query-focused filtering | Score 0-100, filter out 0 | Process ALL communities | ❌ |

> **Kết luận**: Demo minh họa đúng **pattern** map-reduce trên graph communities, nhưng bỏ qua hầu hết pipeline (entity extraction, Leiden, indexing-time summaries, helpfulness scoring).

---

## 3. Self-RAG vs. Asai et al. ICLR 2024

| Component | Paper | Code | Verdict |
|:---|:---|:---|:---:|
| [Retrieve] gate | Trained model natively predicts token | Zero-shot LLM prompt → `retrieve_probability` | ⚠️ |
| [IsREL] critic | Fine-tuned on NLI data | Separate LLM call → JSON `is_rel_token` | ⚠️ |
| [IsSUP] check | Trained critic (semantic evaluation) | **Keyword matching**: `article_id in text` | ❌ |
| [IsUSE] score | Model natively predicts 1-5 | **Hardcoded**: 5 if supported, 3 if not, 2 if rejected | ❌ |
| Scoring formula | $logP(y) + w \cdot logP(\text{tokens})$ | Không có logprobs, không có weighted scoring | ❌ |
| Tree decoding / beam | Parallel segment generation, beam search | Single-pass sequential generation | ❌ |
| Training | 150K examples, GPT-4 critic labels | Zero training | ❌ |
| **UI Disclaimer** | — | ✅ Có caption "In-Context Surrogate" rõ ràng | ✅ |

> **Kết luận**: Surrogate đúng **workflow** (retrieve → critique → filter → generate) nhưng dùng prompting thay trained model. **UI đã có disclaimer** — đây là điểm tốt.

---

## 4. FLARE vs. Jiang et al. EMNLP 2023

| Component | Paper | Code | Verdict |
|:---|:---|:---|:---:|
| Confidence | Real token logprobs $\min_t P(w_t)$ | **Verbalized**: LLM self-report JSON score | ❌ |
| Threshold θ | Applied to real probabilities | Applied to self-reported score (0.3/0.5/0.7) | ⚠️ |
| Query formulation | Mask low-conf tokens / generate question | LLM outputs `search_query` in JSON | ❌ |
| Sentence-level processing | Generate 1 sentence → check → continue | **Generate entire draft** → post-hoc iterate | ❌ |
| Regeneration | Continue LM conditioned on full context | Rewrite sentence in isolation | ⚠️ |
| FLARE variant | FLARE_direct or FLARE_instruct | Neither — custom post-hoc verifier | ❌ |
| Fallback | N/A (uses API logprobs) | Heuristic: "ngày"/"%"/"bồi thường" → conf=0.4 | ❌ |
| **UI labeling** | — | ❌ **GHI SAI**: "độ tự tin token" / "token confidence" | ❌ |

> **Kết luận**: Code thực tế là **"LLM-as-Judge Sentence Verifier"**, không phải FLARE. Và **UI vẫn ghi sai** là "token confidence".

---

## 🔴 Lỗi UI Cần Sửa NGAY

### Lỗi FLARE: UI ghi "token confidence" nhưng thực tế là verbalized

**Vị trí** (từ FLARE agent): `app.py` line ~1724
- Hiện: `"Đang đánh giá độ tự tin token so với ngưỡng theta..."` 
- Sửa: `"Đang đánh giá Verbalized Confidence (LLM tự khai) so với ngưỡng theta..."`

Cũng cần check các chỗ khác trong Tab 4 ghi "token" confidence.

---

## Đánh Giá Tổng Thể Cho Seminar

### Điểm mạnh ✅
- Demo minh họa **đúng các concept cốt lõi** (retrieval gating, reflection, active retrieval, community map-reduce)
- Self-RAG có disclaimer "In-Context Surrogate" rõ ràng
- Comparison matrix phân biệt rõ 4 paradigms
- Context poisoning / "When Retrieval Hurts" là sáng tạo sư phạm hiệu quả

### Rủi ro khi bị hỏi ⚠️
1. **"Các bạn dùng Leiden hay Greedy Modularity?"** → Trả lời thẳng: Greedy Modularity cho simplicity, Leiden cho production.
2. **"FLARE dùng token logprobs thật không?"** → Cần sửa UI trước. Trả lời: Verbalized confidence (LLM tự khai), không phải API logprobs.
3. **"Self-RAG IsSUP dùng trained critic?"** → Đã có disclaimer. Trả lời: In-context surrogate với keyword attribution check.
4. **"Graph của các bạn chỉ có 15 edges hand-crafted?"** → Trả lời: Simplified cho 30-phút demo, full GraphRAG cần LLM entity extraction.
