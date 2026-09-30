# 🎤 SCRIPT THUYẾT TRÌNH — Demo Walkthrough Chi Tiết

> **30 phút | 19:00–19:30 | B4-305**  
> Demo: https://tuandunghcmut-co5151-rag-seminar-arena.hf.space  
> Sidebar: chọn 🇻🇳 Tiếng Việt, thu gọn sidebar trước khi bắt đầu

---

## PHÚT 0:00–1:30 — MỞ ĐẦU

**👀 Màn hình: Header app + 5 tabs**

> "Chào thầy và các bạn. Nhóm G8 trình bày chủ đề **Retrieval-Augmented Agents**."

> "Câu hỏi cốt lõi: LLM có kiến thức sẵn trong trọng số (parametric memory), nhưng kiến thức đó bị freeze tại thời điểm pre-training. RAG giải quyết bằng cách **tra cứu thêm tài liệu bên ngoài** khi trả lời. Nhưng retrieval không phải lúc nào cũng tốt — **khi nào giúp, khi nào hại?**"

> "App demo có 5 tabs, mỗi tab là một thí nghiệm đối chứng trực tiếp. Tất cả chạy trên **Bộ luật Lao động Việt Nam** — domain có tài liệu cũ (luật 2012) và mới (luật 2019), tạo ra xung đột thực tế."

**👆 Mở nhanh expander "📋 Hướng dẫn tổng quan"** → chỉ bảng 5 hàng → đóng lại (15 giây).

---

## PHÚT 1:30–8:00 — TAB 1: KHI RETRIEVAL GÂY HẠI

### Giải pháp là gì?
> "Tab 1 so sánh 3 paradigm: **Pure LLM** (không retrieve), **Naive RAG** (retrieve thụ động), và **Self-RAG** (retrieve + tự đánh giá). Mục đích: chứng minh retrieval có thể **làm giảm** accuracy nếu không có cơ chế lọc."

### Context & Query
> "Tình huống: Doanh nghiệp hỏi **'Thời gian thử việc CEO tối đa bao lâu?'**"

> "Đáp án đúng: **180 ngày** theo Điều 25, Bộ luật Lao động 2019 (hiện hành từ 01/01/2021). Tuy nhiên, luật cũ 2012 (đã bãi bỏ) quy định chỉ **60 ngày**. Nếu retriever truy xuất phải luật cũ, model sẽ bị 'đầu độc'."

**👆 Chỉ vào query box:**
> "Đây là câu hỏi — có thể chỉnh sửa trước khi chạy. Kết quả đang hiển thị là pre-computed."

### Giải thích từng thông tin trên UI

**👆 Chỉ Cột 1 — Pure LLM (khối xanh dương `st.info`):**
> "**Model 1: Pure LLM (Parametric Baseline)** — LLM trả lời từ trọng số nội tại, không tra cứu bất kỳ tài liệu nào."
> 
> "Dòng mô tả: *'Internal parameters only; zero external context'*."
> 
> "Kết quả: Model trả lời dúng 180 ngày — vì Qwen 72B đã học được luật 2019 trong pre-training."

**👆 Chỉ dưới vạch kẻ `---`:**
> "Phía dưới là **metadata hệ thống** — phân tách khỏi output model:"
> - "`Latency: 1420 ms`" — thời gian inference
> - "`✅ Nhận diện mốc 180 ngày từ bộ nhớ trong`" — hệ thống verify output có chứa đáp án đúng

---

**👆 Chỉ Cột 2 — Naive RAG (khối đỏ `st.error`):**
> "**Model 2: Naive RAG (Lewis et al., 2020)** — retrieve top-k passages bằng BM25 + BGE-M3, nối thô vào prompt."
> 
> "Mô tả: *'Blind in-context injection of top-k passages'* — 'blind' vì nhồi thụ động, không kiểm tra tài liệu."
> 
> "Kết quả hiện **khối đỏ** = bị poisoned. Model trả lời **60 ngày** — dẫn sai luật 2012."

**👆 Chỉ dưới vạch kẻ:**
> "Verdict: *'❌ Bị nhiễm độc: Mô hình bị tài liệu gây nhiễu dẫn dụ và kết luận sai thành 60 ngày!'*"
> "Latency: giờ tốn thêm thời gian retrieve."

**👆 Mở expander "Các đoạn trích trong ngữ cảnh đưa vào prompt":**
> "Đây là bằng chứng — 2 passages được nhồi vào prompt:"
> - "`#1 — Điều 27 BLLĐ 2012 (Bãi bỏ)` — RRF: 0.0331" — **đây là thủ phạm**, luật cũ nhưng similarity score cao hơn
> - "`#2 — Điều 25 BLLĐ 2019 (Hiện hành)` — RRF: 0.0315" — luật đúng nhưng xếp sau
> 
> "RRF là Reciprocal Rank Fusion — công thức `1/(60+rank)`, max ~0.033. Đây là ranking score, không phải similarity."
> 
> "Naive RAG nhận **cả hai** vào prompt nhưng **không biết cái nào hiện hành** → bị luật cũ chi phối."

**Đóng expander.**

---

**👆 Chỉ Cột 3 — Self-RAG (khối xanh lá `st.success`):**
> "**Model 3: Self-RAG (Asai et al., ICLR 2024)** — cùng retrieve nhưng thêm cơ chế tự đánh giá."
> 
> "Mô tả: *'Self-reflection critic loop [IsREL] & verification [IsSUP]'*."
> 
> "Kết quả **xanh lá** = đúng. Trả lời 180 ngày, trích dẫn Khoản 1 Điều 25 BLLĐ 2019."

**👆 Chỉ metadata bên dưới:**
> - "`✅ Đã xác thực căn cứ; tài liệu gây nhiễu đã bị loại trừ`" — Self-RAG đã **phát hiện và loại** luật 2012
> - "`Token [IsSUP]: FULLY_SUPPORTED`" — câu trả lời được **xác thực có căn cứ** từ tài liệu giữ lại
> - "`Utility [IsUSE]: 5/5`" — đánh giá hữu dụng cao nhất

> "**Đây chính là hiện tượng 'When Retrieval Hurts':** cùng một retriever, cùng kết quả retrieve, nhưng Naive RAG bị poisoned vì nhồi thụ động, Self-RAG sống sót vì có cơ chế lọc."

---

## PHÚT 8:00–13:00 — TAB 2: GRAPHRAG

### Giải pháp là gì?
> "GraphRAG (Edge et al., 2024) giải quyết failure mode thứ hai: **Local Blindness**. Naive RAG chỉ retrieve top-k chunks gần nhất — bỏ sót khi câu hỏi trải dài nhiều chương."

### Context & Query

**👆 Click Tab 2. Chỉ vào query box:**
> "Câu hỏi: *'Tổng hợp toàn diện các trường hợp NSDLĐ KHÔNG ĐƯỢC hoặc BỊ HẠN CHẾ quyền chấm dứt hợp đồng và xử lý kỷ luật?'*"
> 
> "Câu hỏi này là **global/synthesis query** — đáp án nằm rải rác ở Điều 37, 122, 123, 125, 36... không cluster trong một đoạn. Top-3 vector search không đủ."

### Giải thích từng thông tin trên UI

**👆 Chỉ Cột trái — Naive RAG (khối warning vàng):**
> "Naive RAG retrieve top-3 → trả lời **thiếu**, chỉ nêu được 1-2 trường hợp."

**👆 Chỉ Cột phải — GraphRAG (khối success xanh):**
> "GraphRAG liệt kê **đầy đủ** — bao quát tất cả điều khoản liên quan."
> 
> "Cách hoạt động:
> 1. **Xây Knowledge Graph** — mỗi node là thực thể pháp lý (điều luật, khái niệm), mỗi edge là quan hệ
> 2. **Community Detection** — phân cụm graph thành communities (nhóm điều khoản liên quan). Paper dùng Leiden algorithm, demo dùng Greedy Modularity — cùng nguyên lý tối đa hóa Newman's Modularity Q
> 3. **Map phase** — LLM tóm tắt từng community liên quan đến query
> 4. **Reduce phase** — tổng hợp tất cả community summaries thành câu trả lời cuối"

**👆 Mở expander kiến trúc → chỉ Mermaid diagram GraphRAG:**
> "Flowchart: Query → Tất cả Communities → Map (mỗi community sinh summary) → Reduce (tổng hợp) → Final Answer. Khác biệt cốt lõi: GraphRAG **duyệt qua tất cả communities**, không chỉ top-k."

**Đóng expander. SKIP expander toán học.**

---

## PHÚT 13:00–19:00 — TAB 3: SELF-RAG REFLECTION TOKENS

### Giải pháp là gì?
> "Self-RAG (Asai et al., ICLR 2024) đưa vào 4 **special tokens** — gọi là Reflection Tokens — để kiểm soát toàn bộ vòng đời: có cần retrieve không, tài liệu có liên quan không, câu trả lời có căn cứ không, có hữu dụng không."

### Context & Query

**👆 Click Tab 3. Chỉ vào query:**
> "Cùng câu hỏi thử việc CEO — nhưng tab này **bóc tách chi tiết** từng bước Self-RAG làm gì."

### Giải thích từng thông tin trên UI

**👆 Chỉ phần trên — Retrieve Decision:**
> "Đầu tiên: **`[Retrieve]: NEED_RETRIEVAL`** — Self-RAG quyết định câu hỏi này CẦN tra cứu (câu hỏi về mốc thời gian cụ thể, không thể trả lời từ parametric memory alone)."
> 
> "Reasoning: *'Câu hỏi đòi hỏi xác định mốc thời gian thử việc tối đa... cần kích hoạt tra cứu.'*"

**👆 Chỉ phần giữa — Per-passage critique:**
> "Sau khi retrieve, Self-RAG **đánh giá từng passage**:"
> 
> "Passage 1: **Điều 27 BLLĐ 2012 (Hết hiệu lực)** → `[IsREL]: IRRELEVANT` — phát hiện luật đã bãi bỏ, **loại bỏ**."
> 
> "Passage 2: **Điều 25 BLLĐ 2019 (Hiện hành)** → `[IsREL]: RELEVANT` — luật hiện hành, **giữ lại**."
> 
> "Đây là bước **quyết định** — Naive RAG không có bước này, nên nhồi cả hai."

**👆 Chỉ phần dưới — Generation + Verification:**
> "Sau khi lọc, model sinh câu trả lời chỉ từ tài liệu RELEVANT:"
> 
> "`[IsSUP]: FULLY_SUPPORTED` — câu trả lời **có căn cứ** từ Điều 25."
> 
> "`[IsUSE]: 5/5` — mức hữu dụng cao nhất."
> 
> "Critique summary: *'Đã loại bỏ văn bản đã hết hiệu lực (BLLĐ 2012); chấp nhận căn cứ hiện hành Điều 25 BLLĐ 2019. Khẳng định mốc 180 ngày.'*"

**📌 Lưu ý quan trọng — nói rõ:**
> "**Disclaimer**: Paper gốc **train** critic model trên 150K examples với GPT-4 labels. Demo dùng **In-Context Surrogate** — zero-shot prompting. Giao diện đã ghi rõ disclaimer này."

**SKIP Deep Dive expander.**

---

## PHÚT 19:00–24:00 — TAB 4: FLARE

### Giải pháp là gì?
> "FLARE (Jiang et al., EMNLP 2023) — **Forward-Looking Active REtrieval**. Thay vì retrieve ngay từ đầu (Naive RAG) hay đánh giá sau khi retrieve (Self-RAG), FLARE **sinh trước rồi kiểm tra**: nếu model tự tin → không retrieve; nếu model bất định → trigger retrieve + viết lại."

### Context & Query

**👆 Click Tab 4. Chỉ vào query:**
> "Câu hỏi: *'Khi NLĐ đơn phương chấm dứt HĐLĐ trái pháp luật thì có được nhận trợ cấp thôi việc không và phải bồi thường gì?'*"
> 
> "Câu hỏi này có phần **nguyên tắc chung** (model tự tin) và phần **số liệu cụ thể** (cần tra cứu)."

### Giải thích từng thông tin trên UI

**👆 Chỉ phần `Trigger Rule: Conf_LLM < θ`:**
> "Quy tắc kích hoạt: Khi confidence của LLM cho câu đang sinh < ngưỡng θ (mặc định 0.5), FLARE trigger retrieval."

**👆 Chỉ trace từng câu — 3 bước:**

> "**Câu 1**: Nguyên tắc chung ('đơn phương trái luật → không được trợ cấp')
> - Confidence: **0.94** (rất cao) → **SKIP retrieval** ✅
> - Model tự tin vì đây là kiến thức phổ thông pháp luật"

> "**Câu 2**: Số liệu bồi thường cụ thể ('nửa tháng tiền lương')
> - Confidence: **0.38** (thấp hơn θ=0.5) → **TRIGGER retrieval** 🔍
> - Model bất định về con số chính xác → retrieve Điều 40-41 → viết lại"

> "**Câu 3**: Thêm chi tiết ('chi phí đào tạo')
> - Confidence: **0.91** → **SKIP retrieval** ✅"

> "Tổng kết hiển thị: **3 câu, chỉ 1 lần retrieval** — tiết kiệm so với Naive RAG (luôn retrieve)."

**👆 Mở expander kiến trúc → chỉ Mermaid diagram:**
> "Flow: Query → Sinh nháp đầy đủ → Xác minh từng câu → Confidence < θ? → Có: retrieve + viết lại. Không: giữ nguyên."

**📌 So sánh 3 paradigm:**
> "- **Naive RAG**: luôn retrieve → lãng phí khi model đã biết
> - **Self-RAG**: đánh giá **sau** retrieve → lọc passage xấu
> - **FLARE**: đánh giá **trước** retrieve → quyết định có cần retrieve không
> 
> Ba chiến lược **bổ trợ** nhau, không thay thế."

**Đóng expander. SKIP Deep Dive.**

---

## PHÚT 24:00–27:00 — RETRIEVAL AS TOOL + TỔNG KẾT

**👆 Cuộn lên header → mở expander "📋 Hướng dẫn tổng quan"**

> "Nhìn lại toàn cảnh: Sự tiến hóa từ **Retrieval-as-Pipeline** sang **Retrieval-as-Tool**:"
> 
> "- **Naive RAG**: retrieval là bước cố định, luôn chạy — giống một stage trong pipeline
> - **Self-RAG**: agent có `[Retrieve]` gate — **tự quyết** có retrieve không
> - **FLARE**: agent **giám sát** confidence — trigger khi bất định
> - **GraphRAG**: agent chọn **cách** retrieve — graph traversal thay vì vector search"
> 
> "Theo framework CoALA (Sumers 2024): retrieval chỉ là **một action** trong Action Space của agent. Agent có Decision Procedure để chọn action nào — retrieve, generate, hay dùng tool khác."

**Đóng expander.**

---

## PHÚT 27:00–29:00 — TAB 5 LƯỚT NHANH + HẠN CHẾ

**👆 Click Tab 5 — lướt nhanh:**
> "Tab 5 cho phép inspect hạ tầng: corpus 15 điều luật, knowledge graph, embedding space."

**Hạn chế và hướng mở rộng (nói nhanh):**
> "1. Self-RAG cần trained critic → hướng: LLM-as-judge zero-shot
> 2. GraphRAG tốn chi phí graph construction → hướng: incremental update  
> 3. FLARE cần logprobs (API hạn chế) → hướng: verbalized confidence
> 4. Tất cả single-hop → hướng: multi-hop chain-of-retrieval
> 5. Chỉ text → hướng: multimodal RAG"

---

## PHÚT 28:00–29:00 — TAB 6: KHO MÃ GIẢ & THUẬT TOÁN (VŨ KHÍ ĐẶC BIỆT KHI Q&A)

**👆 Click Tab 6 (Algorithms) — Giới thiệu tổng quan:**
> "Để phục vụ nghiên cứu và trả lời chất vấn kỹ thuật chuyên sâu của hội đồng, nhóm đã tổng hợp toàn bộ mã giả suy luận (Inference Pseudocode) và sơ đồ luồng vector cho cả 4 cơ chế RAG cốt lõi: Naive RAG, Self-RAG, GraphRAG, và FLARE.
> Giao diện hỗ trợ linh hoạt 3 chế độ hiển thị: Song song (Split 50:50), Toàn màn hình Mã giả (100% Width), hoặc Xếp dọc toàn diện, kèm nút Phóng to Modal toàn màn hình để hội trường có thể quan sát từng dòng mã rõ ràng nhất.
> Mỗi mô hình đều có đầy đủ 3 phần:
> 1. **Sơ đồ luồng vector tương tác**
> 2. **Công thức toán học từ bài báo gốc** (Marginalization, Modularity Q, Reflection Scoring, Uncertainty Threshold)
> 3. **Mã giả Pythonic** có chú thích từng bước thực thi và đánh số dòng chi tiết.
> Bất kỳ ai nhìn vào Tab 6 cũng có thể nắm bắt và tái lập lại 100% thuật toán."

---

## PHÚT 29:00–30:00 — KẾT LUẬN

**👆 Về Tab 1 — chỉ 3 cột:**
> "Retrieval là con dao hai lưỡi. Corpus sạch → Naive RAG đủ. Corpus có conflict → Self-RAG lọc. Câu hỏi global → GraphRAG phủ sóng. Sinh dài → FLARE tiết kiệm. Và agent tự quyết định dùng cái nào."

> "Demo vẫn live trên Hugging Face, mọi người thử được. Xin cảm ơn."

---

## 🗺️ TÓM TẮT: CLICK GÌ, SKIP GÌ

| # | UI Element | Hành động | Phút |
|:---:|---|:---:|:---:|
| 1 | Header expander "Hướng dẫn tổng quan" | **MỞ 15s** → đóng | 0:30 |
| 2 | Tab 1: Query box | **CHỈ** — đọc câu hỏi | 1:30 |
| 3 | Tab 1: Cột 1 Pure LLM (info xanh) | **CHỈ** output + metadata | 3:00 |
| 4 | Tab 1: Cột 2 Naive RAG (error đỏ) | **CHỈ** output + verdict | 4:00 |
| 5 | Tab 1: Expander "Passages injected" | **MỞ** — chỉ 2 passages | 5:00 |
| 6 | Tab 1: Cột 3 Self-RAG (success xanh) | **CHỈ** output + IsSUP + IsUSE | 6:30 |
| 7 | Tab 1: Expander "Bối cảnh & Kiến trúc" | **❌ SKIP** | — |
| 8 | Tab 2: 2 cột kết quả | **CHỈ** Naive thiếu vs Graph đủ | 9:00 |
| 9 | Tab 2: Expander kiến trúc GraphRAG | **MỞ** — diagram 30s | 11:00 |
| 10 | Tab 2: Expander toán học | **❌ SKIP** | — |
| 11 | Tab 3: Retrieve Decision | **CHỈ** NEED_RETRIEVAL | 14:00 |
| 12 | Tab 3: Per-passage IsREL | **CHỈ** IRRELEVANT vs RELEVANT | 16:00 |
| 13 | Tab 3: IsSUP + IsUSE | **CHỈ** FULLY_SUPPORTED + 5/5 | 17:30 |
| 14 | Tab 3: Deep Dive | **❌ SKIP** | — |
| 15 | Tab 4: Trigger Rule + trace 3 câu | **CHỈ** confidence per sentence | 21:00 |
| 16 | Tab 4: Expander kiến trúc FLARE | **MỞ** — diagram 20s | 23:00 |
| 17 | Tab 4: Deep Dive | **❌ SKIP** | — |
| 18 | Header expander "Hướng dẫn" lần 2 | **MỞ** cho Q6 (agent loop) | 25:00 |
| 19 | Tab 5: Data Explorer | **LƯỚT 1 phút** | 28:00 |
| 20 | Tab 1: Quay lại 3 cột | **CHỈ** kết luận | 29:00 |

### ❌ KHÔNG MỞ:
- Deep Dive expanders (dành cho Q&A nếu bị hỏi)
- Toán học chi tiết  
- Sidebar settings
- Nút "Run Live" (dùng Gold, tránh timeout)

---

## 🛡️ Q&A DỰ BỊ

**"Demo dùng surrogate, khác paper?"**
> Paper train 150K labels, demo zero-shot prompting. Đã disclaimer trên UI.

**"Score 0.03 nghĩa gì?"**
> RRF = 1/(60+rank). Max ~0.033. Ranking score, không phải similarity.

**"GraphRAG hand-craft graph?"**
> Corpus nhỏ 5 điều luật. Demo minh họa community detection + map-reduce.

**"Lost in the Middle số liệu?"**
> Liu 2024: GPT-3.5 giảm 56→44% khi gold passage ở giữa 20 documents.

**"FLARE verbalized confidence có đáng tin?"**
> Paper dùng token logprobs. Verbalized là approximation, nhưng LLMs self-calibrate hợp lý (Kadavath 2022).

**"Khi nào dùng paradigm nào?"**
> Chỉ bảng 5 hàng header: corpus sạch→Naive, conflict→Self-RAG, global→GraphRAG, multi-sentence→FLARE.
