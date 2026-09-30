# 🎤 SCRIPT THUYẾT TRÌNH — Demo Walkthrough

> **30 phút | 19:00–19:30 | B4-305**
> Mở app trên projector: https://tuandunghcmut-co5151-rag-seminar-arena.hf.space

---

## TRƯỚC KHI BẮT ĐẦU (setup)

1. Mở app → **Sidebar trái** → Chọn **🇻🇳 Tiếng Việt**
2. Provider: đã set sẵn (HF Inference API)
3. App hiển thị **kết quả Gold (cached)** mặc định → không cần chạy live ngay
4. Thu gọn sidebar (bấm X) để màn hình rộng hơn

---

## PHÚT 0:00–1:30 — MỞ ĐẦU

### 👀 Đang nhìn: Header + Bảng tổng quan

**Nói:**
> "Chào thầy và các bạn. Nhóm G8 trình bày chủ đề Retrieval-Augmented Agents. Thay vì slide, nhóm sẽ demo trực tiếp trên app — mọi thứ mình nói đều có thể bấm chạy kiểm chứng."

**👆 Chỉ vào** tiêu đề app: *"RAG Evaluation Laboratory"*

> "App có 5 tabs, mỗi tab minh họa một paradigm RAG khác nhau, đối chứng trên Bộ luật Lao động Việt Nam."

**👆 Mở expander** "📋 Hướng dẫn tổng quan" → **chỉ nhanh vào bảng 5 hàng**:

> "Tổng quan 5 paradigm: Pure LLM, Naive RAG, Self-RAG, GraphRAG, FLARE. Mỗi cái giải quyết một failure mode khác nhau. Mình sẽ đi từng cái."

**⏱️ Đóng expander lại** — không đọc chi tiết, chỉ giới thiệu nhanh.

---

## PHÚT 1:30–7:00 — TAB 1: "KHI RETRIEVAL GÂY HẠI" ⚔️
### → Rubric Q1 (cơ chế RAG) + Q5 (retrieval giảm accuracy)

### 👀 Đang nhìn: Tab 1, kết quả Gold 3 cột

**👆 Click vào Tab 1** (đã mặc định ở đây)

**Nói:**
> "Đây là thí nghiệm cốt lõi. Tình huống: doanh nghiệp hỏi **'Thời gian thử việc CEO tối đa bao lâu?'** — đáp án đúng là **180 ngày** theo Bộ luật Lao động 2019."

**👆 Chỉ lần lượt 3 cột kết quả:**

**Cột 1 — Pure LLM (xanh dương):**
> "Pure LLM — không retrieve gì, trả lời từ parametric memory. Đây là baseline."

**Cột 2 — Naive RAG (đỏ/vàng):**
> "Naive RAG — retrieve top-3 passages bằng BM25 + BGE-M3, nối thô vào prompt. **Kết quả: bị poisoned** — trả lời 60 ngày vì nhồi phải luật 2012 đã bãi bỏ."

**👆 Mở expander** "Các đoạn trích trong ngữ cảnh đưa vào prompt":
> "Xem đây: passage #1 là Điều 27 BLLĐ 2012 (bãi bỏ), passage #2 là Điều 25 BLLĐ 2019 (hiện hành). Naive RAG nhồi cả hai, model bị confuse."

**Cột 3 — Self-RAG (xanh lá):**
> "Self-RAG — dùng token `[IsREL]` thẩm định: luật 2012 → IRRELEVANT, loại bỏ. Chỉ giữ luật 2019 → trả lời đúng 180 ngày."

**👆 Chỉ vào dòng** `Token [IsSUP]: FULLY_SUPPORTED | Utility: 5/5`

> "Đây chính là hiện tượng **'When Retrieval Hurts'** — retrieval thêm noise thay vì giúp. Naive RAG **không có cơ chế lọc**, còn Self-RAG **tự đánh giá** và loại bỏ."

### 📌 Nếu có thời gian — chỉ vào bảng so sánh bên dưới 3 cột
> "Bảng tóm tắt: Naive RAG bị poisoned, Self-RAG tự sửa."

**⚠️ SKIP:** Không mở expander "Bối cảnh thực tế & Kiến trúc" — quá chi tiết.

---

## PHÚT 7:00–12:00 — TAB 2: GRAPHRAG 🌐
### → Rubric Q2 (Naive fails global → GraphRAG)

**👆 Click Tab 2**

**Nói:**
> "Failure mode thứ hai: **Local Blindness**. Câu hỏi kiểu 'Tóm tắt tất cả quyền lợi khi sa thải trái luật' — nằm rải rác nhiều chương."

**👆 Chỉ vào kết quả 2 cột:**

**Cột trái — Naive RAG (warning vàng):**
> "Naive RAG chỉ retrieve top-3 → bỏ sót. Trả lời thiếu."

**Cột phải — GraphRAG (success xanh):**
> "GraphRAG: (1) xây Knowledge Graph, (2) phân cụm cộng đồng — mỗi community là một nhóm điều khoản liên quan, (3) Map-Reduce qua tất cả communities → bao quát hết."

**👆 Mở expander "Community Details"** nếu có:
> "Mỗi community được tóm tắt bằng LLM, sau đó reduce thành câu trả lời tổng thể."

**👆 Mở expander kiến trúc** → chỉ Mermaid diagram GraphRAG:
> "Flowchart: Query → Tất cả Communities → Map (mỗi community sinh summary) → Reduce (tổng hợp) → Final Answer."

**📌 Giải thích nhanh Modularity:**
> "Community detection dùng Newman's Modularity Q — tối đa hóa mật độ liên kết trong community, tối thiểu hóa liên kết giữa communities. Paper gốc Edge et al. dùng Leiden algorithm, demo dùng Greedy Modularity — cùng nguyên lý."

**⚠️ SKIP:** Không mở expander toán học chi tiết.

---

## PHÚT 12:00–18:00 — TAB 3: REFLECTION TOKENS 🔬
### → Rubric Q3 (Self-RAG reflection tokens chi tiết)

**👆 Click Tab 3**

**Nói:**
> "Tab 3 bóc tách chi tiết cơ chế Self-RAG. Asai et al. (ICLR 2024) đưa vào 4 special tokens."

**👆 Chỉ lần lượt vào kết quả — mỗi passage có verdict:**

> "Passage 1: Điều 27 BLLĐ 2012 → `[IsREL]: IRRELEVANT` — phát hiện luật hết hiệu lực, **loại bỏ**."

> "Passage 2: Điều 25 BLLĐ 2019 → `[IsREL]: RELEVANT` — luật hiện hành, **giữ lại**."

> "Sau khi lọc, model sinh câu trả lời → `[IsSUP]: FULLY_SUPPORTED` — câu trả lời có căn cứ từ tài liệu đã giữ."

> "`[IsUSE]: 5/5` — đánh giá mức độ hữu dụng cao nhất."

**👆 Nếu có slider τ — kéo demo:**
> "Slider τ kiểm soát trade-off: τ cao → ưu tiên critique chặt (lọc nhiều hơn), τ thấp → ưu tiên fluency."

**📌 QUAN TRỌNG — nói rõ:**
> "**Lưu ý**: Demo dùng In-Context Surrogate — zero-shot prompting thay vì trained critic như paper. Paper gốc train trên 150K examples với GPT-4 labels. Demo đã ghi rõ disclaimer trên giao diện."

**👆 Chỉ vào caption disclaimer** trên UI.

**⚠️ SKIP:** Không mở Deep Dive expander — quá dài.

---

## PHÚT 18:00–23:00 — TAB 4: FLARE ⚡
### → Rubric Q4 (FLARE active retrieval)

**👆 Click Tab 4**

**Nói:**
> "FLARE (Jiang et al., EMNLP 2023) — câu hỏi: **khi nào cần retrieve?** Naive RAG luôn retrieve. FLARE chỉ retrieve khi cần."

**👆 Chỉ vào kết quả FLARE — từng câu:**

> "FLARE sinh nháp, rồi kiểm tra confidence từng câu:
> - Câu 1: confidence cao → **skip retrieval** (tiết kiệm latency)
> - Câu 2: confidence thấp → **trigger retrieval** → viết lại với evidence
> - Câu 3: confidence thấp → **trigger retrieval**"

**👆 Chỉ vào dòng** `Trigger Rule: Conf_LLM < θ`:
> "Ngưỡng θ quyết định khi nào trigger. Demo dùng Verbalized Confidence — LLM tự khai mức tự tin — thay vì token logprobs trong paper gốc."

**👆 Mở expander kiến trúc** → chỉ Mermaid diagram:
> "Flow: Query → Sinh nháp đầy đủ → Xác minh từng câu → Confidence < θ? → Có: retrieve + viết lại. Không: giữ nguyên."

**📌 So sánh với Self-RAG:**
> "Self-RAG đánh giá **sau** khi retrieve (lọc passage). FLARE đánh giá **trước** khi retrieve (quyết định có retrieve không). Hai chiến lược bổ trợ nhau."

**⚠️ SKIP:** Không mở Deep Dive expander.

---

## PHÚT 23:00–26:00 — QUAY VỀ HEADER: RETRIEVAL AS TOOL
### → Rubric Q6 (Retrieval trong agent decision loop)

**👆 Cuộn lên đầu trang** → mở expander "📋 Hướng dẫn tổng quan"

**👆 Chỉ vào bảng thứ 2** (Retrieval: Từ Pipeline → Tool):

**Nói:**
> "Tổng kết sự tiến hóa:"

> "- **Naive RAG**: retrieval là stage cố định — luôn retrieve, không lựa chọn
> - **Self-RAG**: retrieval là tool có điều kiện — `[Retrieve]` gate quyết định
> - **FLARE**: retrieval là tool theo yêu cầu — trigger khi confidence thấp
> - **GraphRAG**: retrieval là tool cấu trúc — truy vấn theo graph topology"

> "Theo framework CoALA (Sumers 2024), Language Agent có Action Space gồm nhiều tools — retrieval chỉ là **một action** mà agent **chọn gọi hoặc bỏ qua** dựa trên trạng thái hiện tại. Không phải bước bắt buộc."

---

## PHÚT 26:00–28:00 — TAB 5: DATA EXPLORER (nhanh)
### → Rubric Q7 (Extensions) + bonus

**👆 Click Tab 5**

> "Tab 5 cho phép inspect hạ tầng: corpus, embedding space, knowledge graph."

**👆 Lướt nhanh** — chỉ vào:
- Bảng corpus (các điều luật)
- Knowledge graph visualization (nếu có)

> "Hạn chế và mở rộng:
> 1. Self-RAG cần trained critic → hướng: LLM-as-judge zero-shot
> 2. GraphRAG tốn chi phí graph construction → hướng: incremental update
> 3. FLARE cần logprobs → hướng: verbalized confidence (như demo)
> 4. Tất cả đều single-hop → hướng: multi-hop chain-of-retrieval
> 5. Chỉ text → hướng: multimodal RAG"

---

## PHÚT 28:00–30:00 — KẾT LUẬN

**👆 Cuộn lên bảng tổng quan** hoặc đứng trước 3 cột Tab 1:

> "Tóm lại: Retrieval là con dao hai lưỡi.
> - Corpus sạch → Naive RAG đủ
> - Corpus có conflict → Self-RAG lọc
> - Câu hỏi global → GraphRAG phủ sóng
> - Sinh dài → FLARE tiết kiệm
> - Tất cả → Agent tự quyết định"

> "Mọi thứ trên app đều live, các bạn có thể truy cập link trên Hugging Face Space để thử. Xin cảm ơn, nhóm sẵn sàng nhận câu hỏi."

---

## 🗺️ TÓM TẮT: CLICK GÌ, SKIP GÌ

| Thứ tự | UI Element | Hành động | Thời gian |
|:---:|---|:---:|:---:|
| 1 | Header expander "Hướng dẫn tổng quan" | **MỞ NHANH** → chỉ bảng → đóng | 30s |
| 2 | Tab 1: 3 cột kết quả Gold | **CHỈ VÀO** từng cột | 3 phút |
| 3 | Tab 1: Expander "Passages injected" | **MỞ** xem 2 passages | 30s |
| 4 | Tab 1: Expander "Bối cảnh & Kiến trúc" | **❌ SKIP** | — |
| 5 | Tab 2: 2 cột kết quả | **CHỈ VÀO** | 2 phút |
| 6 | Tab 2: Expander kiến trúc GraphRAG | **MỞ** xem diagram | 1 phút |
| 7 | Tab 2: Expander toán học | **❌ SKIP** | — |
| 8 | Tab 3: Kết quả reflection per-passage | **CHỈ VÀO** từng verdict | 3 phút |
| 9 | Tab 3: Deep Dive expander | **❌ SKIP** | — |
| 10 | Tab 4: Kết quả FLARE per-sentence | **CHỈ VÀO** confidence | 2 phút |
| 11 | Tab 4: Expander kiến trúc FLARE | **MỞ** xem diagram | 1 phút |
| 12 | Tab 4: Deep Dive expander | **❌ SKIP** | — |
| 13 | Header: bảng "Retrieval as Tool" | **MỞ** cho Q6 | 2 phút |
| 14 | Tab 5: Data Explorer | **LƯỚT NHANH** | 1 phút |

### ❌ TUYỆT ĐỐI KHÔNG MỞ:
- Deep Dive expanders (quá dài, lạc đề)
- Toán học chi tiết expanders (giám khảo sẽ hỏi nếu muốn)
- Sidebar settings (đã setup sẵn)
- Nút "Run Live" (dùng Gold cached, tránh rủi ro timeout/error)

### ✅ NÊN MỞ:
- Mermaid diagrams (trực quan, 5 giây hiểu)
- Passages retrieved (chứng minh poisoning)
- Bảng tổng quan (Q1 + Q6)

---

## 🛡️ Q&A DỰ BỊ (15 phút)

**Nếu G7 hỏi "Demo dùng surrogate, khác paper thế nào?":**
> "Paper train 150K labels. Demo dùng zero-shot prompting — approximation. Đã ghi disclaimer trên UI."

**Nếu hỏi "Score 0.03 là sao?":**
> "Đó là RRF score = 1/(60+rank). Max = 0.033. Không phải similarity score."

**Nếu hỏi "GraphRAG hand-craft graph, fair không?":**
> "Corpus nhỏ 5 điều luật. Mục đích demo community detection + map-reduce, không phải entity extraction."

**Nếu hỏi "Lost in the Middle số liệu?":**
> "Liu 2024: GPT-3.5 giảm 56→44% khi gold passage ở giữa context."

**Nếu thầy hỏi "Khi nào dùng paradigm nào?":**
> Chỉ vào bảng 5 hàng ở header expander — cột cuối "Khi Nào Nên Triển Khai".
