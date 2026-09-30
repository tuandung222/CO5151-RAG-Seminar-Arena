# 🎯 Đối Chiếu Demo vs. Yêu Cầu Seminar G8 (S1-4 RAG)

**Deadline**: 18:00 ngày mai 30/9/2026 | **Phòng**: B4-305

---

## 3 Yêu Cầu Thầy Nêu Rõ Cho G8

Trích email thầy Lê Xuân Bách:

> G8 (RAG):
> 1. **Vì sao naive RAG thất bại với câu hỏi "global" và GraphRAG xử lý thế nào**
> 2. **Ít nhất MỘT trường hợp cụ thể retrieval làm giảm độ chính xác, có số liệu**
> 3. **Retrieval là một tool trong vòng quyết định của agent, không phải một stage pipeline riêng**

---

## YÊU CẦU 1: Naive RAG thất bại với câu hỏi "global" → GraphRAG

### Đánh giá: ✅ **ĐÃ ĐÁP ỨNG TỐT**

Demo hiện có **Tab 2: GraphRAG vs. Naive RAG** trình bày:

| Thành phần | Có trong demo? | Vị trí |
|---|:--:|---|
| Giải thích Local Blindness của Vector Search | ✅ | Context card (app.py:1207-1225) |
| Ví dụ cụ thể: Điều 37 (Chương III) vs Điều 122 (Chương VIII) | ✅ | Context card + community visualization |
| Graph topology + cross-chapter bridges | ✅ | Mermaid diagrams (3 tabs) |
| Map-Reduce workflow giải thích | ✅ | Expander + công thức Modularity |
| Live demo chạy GraphRAG vs Naive | ✅ | "Run Live" button |
| Công thức Modularity Q | ✅ | LaTeX inline |

**Lưu ý khi trình bày**: Nên nhấn mạnh rằng graph trong demo là **hand-crafted** (15 nodes, 15 edges) để minh họa pattern Map-Reduce. Nếu bị hỏi "Entity extraction tự động ở đâu?", trả lời thẳng: "Đây là demo minh họa pattern Map-Reduce trên graph cấu trúc sẵn, tương tự simplified version. Full GraphRAG sẽ dùng LLM để trích xuất entity/relation tự động."

---

## YÊU CẦU 2: Ít nhất 1 trường hợp retrieval làm GIẢM độ chính xác, có số liệu

### Đánh giá: ⚠️ **CÓ NHƯNG CẦN KIỂM TRA LẠI SAU FIX B3**

Demo hiện có **Tab 1: When Retrieval Hurts** trình bày context poisoning:
- Distractor (BLLĐ 2012, 60 ngày) được chèn vào context → Naive RAG trả lời sai
- Context card giải thích cơ chế + dẫn paper (Neeman et al. 2023, Wu et al. 2024)

### 🔴 VẤN ĐỀ CRITICAL SAU FIX B3:

Trước đây, chế độ `only_distractor` cắt `Dieu_25` khỏi Naive RAG (rigging). Sau fix `73d1a62`, code `only_distractor` và `mixed_conflict` **giờ giống hệt nhau**:

```python
# only_distractor (sau fix):
passages_to_use.append(distractor_item)
passages_to_use.extend(raw_retrieved[: top_k - 1])

# mixed_conflict:
passages_to_use.append(distractor_item) 
passages_to_use.extend(raw_retrieved[: top_k - 1])
```

**Hệ quả**: Cả hai mode giờ cho cùng kết quả. Và vì retriever sẽ truy xuất `Dieu_25` (nó là kết quả top cho câu hỏi về thử việc), Naive RAG **có thể trả lời đúng 180 ngày** — nghĩa là demo "When Retrieval Hurts" có thể **không thể hiện được poisoning nữa**.

### Giải pháp đề xuất:

**Option A (Khuyến nghị)**: Đổi `only_distractor` thành chèn distractor **thay thế** slot đầu tiên nhưng giữ `top_k=2` thay vì `top_k=3`, để distractor chiếm 50% context thay vì 33%:

```python
if distractor_mode == "only_distractor":
    # Distractor overwhelms context: only 1 real doc alongside distractor
    passages_to_use.append(distractor_item)
    non_distractor = [p for p in raw_retrieved if p["article_id"] != distractor.get("article_id")]
    passages_to_use.extend(non_distractor[:1])  # Only 1 real passage
```

**Option B**: Tăng số distractor (chèn 2 distractor passages khác nhau từ BLLĐ 2012).

**Option C**: Giữ nguyên nhưng nói rõ trong demo: "Điểm mấu chốt không phải Naive RAG luôn sai, mà là nó KHÔNG CÓ CƠ CHẾ KIỂM ĐỊNH. Khi distractor nằm cùng context, kết quả phụ thuộc vào prompt sensitivity — không có gì bảo vệ."

> **⚠️ QUAN TRỌNG**: Cần chạy live test ngay để xem Naive RAG có bị poisoned hay không sau fix B3. Nếu nó vẫn trả lời sai (do distractor ở rank 1 có `rrf_score: 0.999`), thì demo vẫn hoạt động. Nếu nó trả lời đúng, cần chọn Option A.

---

## YÊU CẦU 3: Retrieval là tool trong vòng quyết định của agent

### Đánh giá: 🟡 **CÓ NGẦM NHƯNG CHƯA NÓI RÕ**

Demo hiện có:
- **Self-RAG `[Retrieve]` gate** (Tab 1 & 3): Model quyết định có truy xuất hay không → ✅ đây chính là "retrieval as a tool in agent's decision loop"
- **FLARE active retrieval** (Tab 4): Model sinh nháp → đánh giá confidence → quyết định có gọi retriever → ✅ retrieval là tool được gọi theo nhu cầu

**Nhưng**: UI **chưa bao giờ nói rõ bằng tiếng người** rằng đây là pattern "retrieval as a tool in agent decision loop" theo nghĩa thầy muốn. Thầy nhấn mạnh:

> "retrieval là một **tool** trong **vòng quyết định của agent**, không phải một **stage pipeline riêng**"

Đây là góc nhìn agentic (CoALA framework) — retrieval không phải bước cố định mà là action mà agent chọn gọi (hoặc không gọi) dựa trên state hiện tại.

### Cần bổ sung:

1. **Thêm 1 callout/context card** ở đầu Tab 1 hoặc Tab 5 (comparison matrix) nói rõ:
   - Naive RAG: retrieval là **fixed pipeline stage** (luôn retrieve, luôn concatenate)
   - Self-RAG: retrieval là **conditional tool** — `[Retrieve]` gate quyết định có gọi hay không
   - FLARE: retrieval là **on-demand tool** — chỉ gọi khi confidence thấp
   - GraphRAG: retrieval là **structured tool** — truy vấn theo graph topology thay vì vector similarity

2. **Liên hệ với CoALA framework** (Sumers 2024) — đây là core reading thầy yêu cầu cả lớp đọc.

---

## CHECKLIST TỔNG HỢP TRƯỚC SEMINAR

| # | Hạng mục | Trạng thái | Hành động |
|---|----------|:--:|---|
| 1 | Naive RAG thất bại với global query → GraphRAG | ✅ | Sẵn sàng demo |
| 2 | Retrieval làm giảm accuracy (context poisoning) | ⚠️ | **Cần test live** xem B3 fix có phá demo không |
| 3 | Retrieval as agentic tool (CoALA framing) | 🟡 | **Cần thêm 1 callout** nói rõ paradigm shift |
| 4 | Demo chạy trên máy chiếu | ❓ | Test trên máy chiếu phòng B4-305 |
| 5 | `only_distractor` vs `mixed_conflict` giờ giống nhau | 🔴 | Cần differentiate lại hoặc merge thành 1 mode |
| 6 | Cached results label = "Curated Illustration" | ✅ | Đã sửa, trung thực |
| 7 | Self-RAG = In-context Surrogate (có disclaimer) | ✅ | Đã có caption ⚠️ |
| 8 | FLARE = verbalized confidence (không phải logprobs) | ✅ | Đã sửa formula + notation |
| 9 | Chuẩn bị đối đáp Q&A từ G7 + thầy | ❓ | Xem phần dưới |

---

## CÂU HỎI DỰ ĐOÁN TỪ G7 / THẦY

### Câu hỏi thầy có thể hỏi (dựa trên rubric hints):

1. **"So sánh `only_distractor` vs `mixed_conflict` khác nhau thế nào?"**
   → Sau fix B3, code giờ giống nhau. Cần giải thích hoặc sửa trước.

2. **"Self-RAG của các bạn có dùng fine-tuned model không? Hay chỉ là prompting?"**
   → Đã có disclaimer "In-Context Surrogate". Trả lời thẳng: surrogate, nhưng pattern đúng paper.

3. **"FLARE dùng token logprobs hay verbalized confidence?"**
   → Đã sửa label. Trả lời: verbalized confidence (LLM tự khai).

4. **"Làm sao chứng minh GraphRAG tốt hơn khi graph chỉ có 15 nodes hand-crafted?"**
   → "Demo minh họa pattern Map-Reduce. Full GraphRAG sẽ dùng LLM entity extraction. Corpus nhỏ giúp verify từng bước trong 30 phút."

5. **"Retrieval là tool hay pipeline stage? Các bạn thể hiện điểm này ở đâu?"**
   → Đây là câu hỏi quan trọng nhất. Cần có callout rõ ràng trong demo.

6. **"Số liệu accuracy drop cụ thể khi retrieval gây hại?"**
   → Tab 1 có outcome: POISONED (60 ngày) vs CORRECT (180 ngày). Nhưng cần chạy live để có số liệu thực.

### Câu hỏi từ G7 (phản biện, có dẫn chứng paper):

7. **"RAG paper gốc (Lewis 2020) dùng DPR + BART, còn các bạn dùng BGE-M3 + Qwen. Baseline có fair không?"**
   → "Chúng tôi so sánh pattern (blind concatenation vs reflective critique), không so sánh model cụ thể."

8. **"GraphRAG paper (Edge 2024) dùng Leiden community detection, các bạn dùng Greedy Modularity. Kết quả có khác?"**
   → "Trên graph nhỏ 15 nodes, cả hai converge. Demo chọn `greedy_modularity_communities` vì sẵn trong NetworkX, không cần thêm dependency."
