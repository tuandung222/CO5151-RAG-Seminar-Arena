"""
Internationalization (i18n) Module for RAG Seminar Arena.
Supports seamless switching between English and Vietnamese across the entire UI.
"""

I18N = {
    "en": {
        "page_title": "RAG Evaluation Laboratory: Scientific Paradigms Benchmark",
        "main_header": "CO5151: RAG Scientific Paradigms & Grounding Arena",
        "sub_header": "Empirical Benchmark: Pure LLM vs. Naive RAG vs. Self-RAG vs. GraphRAG vs. FLARE against Authentic Vietnamese Labor Code (Law No. 45/2019/QH14 vs. Repealed Distractor Law No. 10/2012/QH13).",
        "lang_switch_label": "Display Language / Ngôn ngữ hiển thị:",
        
        # Sidebar
        "sidebar_engine": "Inference Engine",
        "sidebar_provider": "Select Provider & Architecture:",
        "sidebar_token_label": "Hugging Face User Access Token (Optional):",
        "sidebar_token_help": "Required to run LIVE cluster verification benchmarks. Pre-computed gold runs do not require any token.",
        "sidebar_retrieval_topology": "Retrieval Topology & Indices",
        "sidebar_dense": "Dense Encoder:",
        "sidebar_sparse": "Sparse Lexical:",
        "sidebar_hybrid": "Hybrid Fusion:",
        "sidebar_corpus_count": "Corpus Articles:",
        "sidebar_knowledge_scope": "Knowledge Cutoff & Scope",
        "sidebar_active_law": "Active Statute: Labor Code 2019 (No. 45/2019/QH14, eff. Jan 1, 2021)",
        "sidebar_distractor_law": "Distractor: Labor Code 2012 (No. 10/2012/QH13, repealed)",
        "sidebar_author": "Researcher / Presenter:",
        "sidebar_supervisor": "Academic Supervisor:",
        
        # Master Educational Guide
        "master_guide_expander": "Master Educational & Visual Guide: How the 5 Grounding Paradigms Actually Work",
        "master_guide_header": "Scientific Architecture & Mechanism Dissection (ICLR, NeurIPS, EMNLP)",
        "master_guide_intro": "This interactive laboratory is engineered for graduate students and researchers studying **Topic S1-4: Retrieval-Augmented & Knowledge-Grounded Agents**. Below is the formal operational breakdown of each paradigm:",
        "master_guide_tbl_header": "TAXONOMY OF THE 5 GROUNDING PARADIGMS",
        "paradigm_pure_title": "1. Pure Parametric LLM (Baseline)",
        "paradigm_pure_desc": "Generates text exclusively by sampling from frozen transformer weights without consulting external references.",
        "paradigm_naive_title": "2. Naive RAG (Lewis et al., NeurIPS 2020)",
        "paradigm_naive_desc": "Linearly concatenates the top-k nearest semantic chunks into the generation prompt without validation or filtering.",
        "paradigm_self_title": "3. Self-RAG (Asai et al., ICLR 2024)",
        "paradigm_self_desc": "Employs an adaptive reflection critic loop with special tokens ([Retrieve], [IsREL], [IsSUP], [IsUSE]) to actively prune irrelevant or outdated distractors.",
        "paradigm_graph_title": "4. GraphRAG (Edge et al., Microsoft Research 2024)",
        "paradigm_graph_desc": "Constructs a statutory knowledge graph, partitions it via Modularity community detection, and executes a Hierarchical Map-Reduce summary to overcome Local Blindness.",
        "paradigm_flare_title": "5. Active Retrieval / FLARE (Jiang et al., EMNLP 2023)",
        "paradigm_flare_desc": "Drafts sentence-by-sentence. When token confidence drops below threshold θ, it initiates targeted on-demand tool calls, saving up to 67% redundant retrieval overhead.",

        # Tabs
        "tab1_title": "1. 3-Model Arena (Outdated Distractor)",
        "tab2_title": "2. GraphRAG vs. Naive (Community Summary)",
        "tab3_title": "3. Self-RAG Reflection Inspector",
        "tab4_title": "4. Active Retrieval (FLARE)",
        "tab5_title": "5. Data Topology & Vector Space",

        # Tab 1
        "tab1_header": "Comparative Benchmark: Pure LLM vs. Naive RAG vs. Self-RAG",
        "tab1_desc": "**Scientific Premise:** Naive RAG uncritically accepts retrieved chunks based solely on semantic embedding similarity. When outdated or repealed statutory passages enter the context, Naive RAG gets **poisoned** and produces authoritative false legal advice. Self-RAG uses an active `[IsREL]` critic to detect and prune expired statutes.",
        "tab1_q_label": "Probing Question (Executive Probation Limit under Current Law):",
        "tab1_btn_run": "Run Live Verification Benchmark",
        "status_cached_label": "Pre-computed Gold Run (Cached)",
        "status_live_label": "Live Cluster Execution (Verified)",
        "cached_telemetry_note": "Results loaded instantaneously from verified cluster evaluation. To independently verify live, click 'Run Live Verification Benchmark' above.",
        "model1_title": "Model 1: Pure LLM (Parametric Baseline)",
        "model1_caption": "Internal parameters only; zero external context",
        "model2_title": "Model 2: Naive RAG (Lewis et al., 2020)",
        "model2_caption": "Blind in-context injection of top-k passages",
        "model3_title": "Model 3: Self-RAG (Asai et al., ICLR 2024)",
        "model3_caption": "Self-reflection critic loop [IsREL] & verification [IsSUP]",
        "matrix_title": "Executive Comparative Matrix (3-Paradigm Evaluation)",
        "matrix_dim": "Evaluation Dimension",
        "matrix_m1": "Model 1: Pure LLM",
        "matrix_m2": "Model 2: Naive RAG",
        "matrix_m3": "Model 3: Self-RAG",
        "matrix_mech": "Operational Mechanism",
        "matrix_mech_pure": "Parametric weights only",
        "matrix_mech_naive": "Blind top-k context concatenation",
        "matrix_mech_self": "Reflective critic [IsREL] & verification [IsSUP]",
        "matrix_lat": "Execution Latency",
        "matrix_claim": "Concluded Duration Claim",
        "matrix_distractor": "Distractor Robustness",
        "matrix_domain": "Domain Reliability",

        # Tab 2
        "tab2_header": "Hierarchical GraphRAG vs. Naive RAG (Overcoming Local Blindness)",
        "tab2_desc": "**Scientific Premise:** Naive RAG retrieves only top-k nearest semantic chunks, suffering from **Local Blindness** on holistic synthesis queries across multiple legal chapters. GraphRAG partitions the statutory corpus into thematic communities via **Newman's Modularity ($Q$)** and executes a **Hierarchical Map-Reduce** to achieve 100% comprehensive coverage.",
        "tab2_q_label": "Global Synthesis Question across Labor Code Chapters:",
        "tab2_naive_col": "Naive RAG (Local Chunk Blindness)",
        "tab2_naive_caption": "Notice how Naive RAG captures only 1-2 local articles, missing broader statutory provisions across the corpus.",
        "tab2_graph_col": "GraphRAG (Hierarchical Global Synthesis)",
        "tab2_map_summaries_title": "Intermediate Community Map Summaries (Map Phase):",

        # Tab 3
        "tab3_header": "Dissection of Self-RAG Special Reflection Tokens",
        "tab3_desc": "**Scientific Premise:** Self-RAG (Asai et al., ICLR 2024) introduces four discrete reflection tokens to control the entire generation lifecycle: `[Retrieve]`, `[IsREL]` (Relevance), `[IsSUP]` (Support/Attribution), and `[IsUSE]` (Utility).",
        "tab3_q_label": "Probe Question for Reflection Gate:",
        "tab3_gate_title": "Passage Filtering Gate [IsREL]:",
        "tab3_verif_title": "Attribution & Grounding Verification [IsSUP]:",
        "tab3_ans_title": "Final Grounded Response:",
        "tab3_deepdive_title": "Theoretical & Engineering Deep Dive: Does Self-RAG use Token Probabilities (Logprobs)?",

        # Tab 4
        "tab4_header": "Forward-Looking Active Retrieval (FLARE on-demand)",
        "tab4_desc": "**Scientific Premise:** Instead of passive retrieval upfront, FLARE (Jiang et al., EMNLP 2023) drafts sentence-by-sentence. When generation confidence drops below a threshold ($Confidence < \\theta$) or factual claims emerge, the system actively issues a targeted search query and rewrites the sentence with exact citations.",
        "tab4_q_label": "Multi-Faceted Statutory Probe Question:",
        "tab4_ans_title": "Final Synthesized Output:",
        "tab4_deepdive_title": "FLARE Uncertainty Trigger & Token Confidence Math",

        # Tab 5
        "tab5_header": "Data Topology, Vector Space & Knowledge Graph Deep Inspector",
        "tab5_desc": "**Academic Objective:** Inspect the dataset schema, dense vector tensor dimensions (BAAI/bge-m3 1024-d), production offline pre-computation vs online serving lifecycle, and statutory knowledge graph community modularity.",
        "tab5_sec1_title": "1. Production Serving Lifecycle: Offline Indexing vs. Real-Time Online Serving",
        "tab5_sec2_title": "2. Dense Vector Space & Pre-computed Tensor Matrix",
        "tab5_sec3_title": "3. Statutory Knowledge Graph Topology (NetworkX & Modularity Communities)",
    },
    
    "vi": {
        "page_title": "Phòng Thí Nghiệm Đánh Giá RAG: Đối Chứng Các Cơ Chế Khoa Học",
        "main_header": "CO5151: Đấu Trường Đánh Giá Các Cơ Chế RAG Khoa Học",
        "sub_header": "Thực nghiệm đối chứng: Pure LLM vs. Naive RAG vs. Self-RAG vs. GraphRAG vs. FLARE trên dữ liệu thực tế Bộ luật Lao động 2019 (Luật 45/2019/QH14 đối chứng tài liệu lỗi thời Luật 10/2012/QH13).",
        "lang_switch_label": "Display Language / Ngôn ngữ hiển thị:",
        
        # Sidebar
        "sidebar_engine": "Động Cơ Suy Luận (Inference Engine)",
        "sidebar_provider": "Chọn Nhà Cung Cấp & Mô Hình:",
        "sidebar_token_label": "Hugging Face User Access Token (Tùy chọn):",
        "sidebar_token_help": "Bắt buộc khi chạy kiểm chứng TRỰC TIẾP trên cluster. Kết quả chuẩn tiền tính toán không cần nhập token.",
        "sidebar_retrieval_topology": "Cấu Trúc Chỉ Mục & Topo Truy Xuất",
        "sidebar_dense": "Mã hóa Dense:",
        "sidebar_sparse": "Truy xuất Từ khóa:",
        "sidebar_hybrid": "Hợp nhất Xếp hạng:",
        "sidebar_corpus_count": "Số lượng Điều luật:",
        "sidebar_knowledge_scope": "Phạm Vi Tri Thức & Cắt Ngắn",
        "sidebar_active_law": "Luật hiện hành: Bộ luật Lao động 2019 (Luật số 45/2019/QH14, có hiệu lực 01/01/2021)",
        "sidebar_distractor_law": "Tài liệu gây nhiễu: Bộ luật Lao động 2012 (Luật số 10/2012/QH13, đã hết hiệu lực)",
        "sidebar_author": "Học viên / Tác giả:",
        "sidebar_supervisor": "Giảng viên hướng dẫn:",
        
        # Master Educational Guide
        "master_guide_expander": "Cẩm Nang Trực Quan & Giáo Khoa: Toàn Bộ Cơ Chế Hoạt Động Của 5 Phương Pháp RAG",
        "master_guide_header": "Phân Tích Kiến Trúc Khoa Học & Cơ Chế Hoạt Động (ICLR, NeurIPS, EMNLP)",
        "master_guide_intro": "Phòng thực nghiệm này được thiết kế phục vụ môn học Cao học **CO5151 - Advanced Agentic AI** (Chủ đề S1-4: Retrieval-Augmented & Knowledge-Grounded Agents). Dưới đây là phân tích toán học và cơ chế chi tiết của từng phương pháp:",
        "master_guide_tbl_header": "PHÂN LOẠI 5 CƠ CHẾ RAG KHOA HỌC",
        "paradigm_pure_title": "1. Pure Parametric LLM (Baseline Bộ nhớ tham số mô hình)",
        "paradigm_pure_desc": "Sinh văn bản hoàn toàn dựa trên xác suất từ trọng số đã đóng băng, không tra cứu ngữ cảnh ngoài.",
        "paradigm_naive_title": "2. Naive RAG (Lewis et al., NeurIPS 2020)",
        "paradigm_naive_desc": "Nối trực tiếp top-k các đoạn trích trong ngữ cảnh có độ tương đồng cosine cao nhất vào prompt mà không qua khâu tự đánh giá hay kiểm chứng.",
        "paradigm_self_title": "3. Self-RAG (Asai et al., ICLR 2024)",
        "paradigm_self_desc": "Sử dụng vòng lặp tự đánh giá với các token đặc biệt ([Retrieve], [IsREL], [IsSUP], [IsUSE]) để chủ động phát hiện và loại bỏ tài liệu gây nhiễu hoặc sai lệch.",
        "paradigm_graph_title": "4. GraphRAG (Edge et al., Microsoft Research 2024)",
        "paradigm_graph_desc": "Xây dựng đồ thị tri thức pháp lý, Phân cụm cộng đồng theo Modularity Q và chạy Tổng hợp phân cấp Map-Reduce để khắc phục Điểm mù cục bộ của Vector Search.",
        "paradigm_flare_title": "5. Active Retrieval / FLARE (Jiang et al., EMNLP 2023)",
        "paradigm_flare_desc": "Sinh nháp dự phóng từng câu liên tiếp. Chỉ kích hoạt truy xuất chủ động theo độ bất định token tại chỗ khi độ tự tin rơi xuống dưới ngưỡng θ, tiết kiệm tới 67% chi phí tính toán.",

        # Tabs
        "tab1_title": "1. Đấu Trường 3 Mô Hình (Tài Liệu Gây Nhiễu)",
        "tab2_title": "2. GraphRAG vs. Naive (Cụm Cộng Đồng)",
        "tab3_title": "3. Token tự đánh giá Self-RAG (Inspector)",
        "tab4_title": "4. Truy Xuất Chủ Động (FLARE)",
        "tab5_title": "5. Topo Dữ Liệu & Đồ Thị Tri Thức",

        # Tab 1
        "tab1_header": "Thực Nghiệm Đối Chứng: Pure LLM vs. Naive RAG vs. Self-RAG",
        "tab1_desc": "**Cơ sở khoa học:** Naive RAG tiếp nhận tài liệu thụ động chỉ dựa trên độ tương đồng ngữ nghĩa. Khi kho dữ liệu chứa văn bản hết hiệu lực hoặc tài liệu gây nhiễu, Naive RAG bị **nhiễm độc ngữ cảnh** và đưa ra kết luận hoàn toàn sai lệch. Self-RAG sử dụng Bộ lọc tài liệu liên quan `[IsREL]` để phát hiện và loại trừ tài liệu gây nhiễu hoặc hết hiệu lực.",
        "tab1_q_label": "Câu hỏi kiểm chứng (Thời gian thử việc tối đa của Giám đốc điều hành theo luật hiện hành):",
        "tab1_btn_run": "Chạy Thực Nghiệm Kiểm Chứng Trực Tiếp",
        "status_cached_label": "Kết Quả Tiền Tính Toán (Cached)",
        "status_live_label": "Thực Nghiệm Cluster Trực Tiếp (Verified)",
        "cached_telemetry_note": "Kết quả được tải tức thì từ phiên kiểm chứng chuẩn trên cluster. Để tự kiểm chứng độc lập trực tiếp, hãy nhấn nút 'Chạy Thực Nghiệm Kiểm Chứng Trực Tiếp' ở trên.",
        "model1_title": "Mô hình 1: Pure LLM (Baseline Bộ nhớ tham số mô hình)",
        "model1_caption": "Chỉ dựa vào bộ nhớ trong; không có ngữ cảnh ngoài",
        "model2_title": "Mô hình 2: Naive RAG (Lewis et al., 2020)",
        "model2_caption": "Nối thô top-k các đoạn trích trong ngữ cảnh vào prompt",
        "model3_title": "Mô hình 3: Self-RAG (Asai et al., ICLR 2024)",
        "model3_caption": "Vòng tự đánh giá [IsREL] lọc tài liệu & kiểm định [IsSUP]",
        "matrix_title": "Bảng Ma Trận So Sánh Đối Chứng (Đánh Giá 3 Cơ Chế)",
        "matrix_dim": "Tiêu Chí Đánh Giá",
        "matrix_m1": "Mô hình 1: Pure LLM",
        "matrix_m2": "Mô hình 2: Naive RAG",
        "matrix_m3": "Mô hình 3: Self-RAG",
        "matrix_mech": "Cơ Chế Vận Hành",
        "matrix_mech_pure": "Bộ nhớ tham số mô hình",
        "matrix_mech_naive": "Nối thô top-k đoạn văn bản đưa vào prompt",
        "matrix_mech_self": "Bộ lọc tài liệu liên quan [IsREL] & Kiểm định căn cứ trích dẫn [IsSUP]",
        "matrix_lat": "Độ Trễ Thực Thi",
        "matrix_claim": "Kết Luận Về Thời Hạn",
        "matrix_distractor": "Khả Năng Chống Tài Liệu Gây Nhiễu",
        "matrix_domain": "Độ Tin Cậy Pháp Lý",

        # Tab 2
        "tab2_header": "GraphRAG Phân Cấp vs. Naive RAG (Khắc Phục Điểm Mù Cục Bộ Của Vector Search)",
        "tab2_desc": "**Cơ sở khoa học:** Naive RAG chỉ bốc được top-k các đoạn trích trong ngữ cảnh gần nhất, dẫn đến hiện tượng **Điểm mù cục bộ của Vector Search** khi gặp câu hỏi tổng hợp toàn diện trải dài qua nhiều chương luật. GraphRAG phân hoạch đồ thị thành các cụm cộng đồng thông qua **Phân cụm cộng đồng theo Modularity ($Q$)** và chạy **Tổng hợp phân cấp Map-Reduce** để bao quát 100% ngữ liệu.",
        "tab2_q_label": "Câu hỏi tổng hợp toàn diện trải dài qua các chương luật lao động:",
        "tab2_naive_col": "Naive RAG (Điểm Mù Đoạn Cục Bộ)",
        "tab2_naive_caption": "Lưu ý: Naive RAG chỉ bốc được 1-2 điều luật cục bộ, bỏ sót hoàn toàn các chế định quan trọng khác trong bộ luật.",
        "tab2_graph_col": "GraphRAG (Tổng Hợp Toàn Cục Phân Cấp)",
        "tab2_map_summaries_title": "Báo Cáo Tóm Tắt Từng Cụm Cộng Đồng (Pha Map):",

        # Tab 3
        "tab3_header": "Bóc Tách Chi Tiết Reflection Tokens Của Self-RAG",
        "tab3_desc": "**Cơ sở khoa học:** Self-RAG (Asai et al., ICLR 2024) đưa vào 4 token tự đánh giá chuyên biệt để kiểm soát toàn bộ chu trình sinh: `[Retrieve]` (Cổng truy xuất), `[IsREL]` (Bộ lọc tài liệu liên quan), `[IsSUP]` (Kiểm định căn cứ trích dẫn / Đối chiếu ngữ cảnh), và `[IsUSE]` (Đánh giá mức độ hữu dụng).",
        "tab3_q_label": "Câu hỏi đa ý kiểm chứng Cổng truy xuất:",
        "tab3_gate_title": "Bộ lọc tài liệu liên quan [IsREL]:",
        "tab3_verif_title": "Kiểm định căn cứ trích dẫn / Đối chiếu ngữ cảnh [IsSUP]:",
        "tab3_ans_title": "Câu Trả Lời Xác Thực Cuối Cùng:",
        "tab3_deepdive_title": "Phân Tích Chuyên Sâu: Self-RAG Có Dùng Xác Suất Token (Logprobs) Không?",

        # Tab 4
        "tab4_header": "Truy Xuất Chủ Động Theo Nhu Cầu (FLARE Active Retrieval)",
        "tab4_desc": "**Cơ sở khoa học:** Thay vì truy xuất thụ động ngay từ đầu, FLARE (Jiang et al., EMNLP 2023) sinh nháp dự phóng từng câu một. Khi độ tự tin của câu rơi xuống dưới ngưỡng ($Confidence < \\theta$), hệ thống mới kích hoạt truy xuất chủ động theo độ bất định token đúng trọng tâm và viết lại câu chuẩn xác theo điều luật.",
        "tab4_q_label": "Câu hỏi đa khía cạnh về chế tài pháp lý:",
        "tab4_ans_title": "Kết Quả Tổng Hợp Hoàn Chỉnh:",
        "tab4_deepdive_title": "Cơ Chế Kích Hoạt Độ Không Chắc Chắn & Toán Học Tự Tin Token Của FLARE",

        # Tab 5
        "tab5_header": "Kiểm Tra Sâu Topo Dữ Liệu, Không Gian Vector & Đồ Thị Tri Thức",
        "tab5_desc": "**Mục tiêu học thuật:** Khảo sát cấu trúc schema dữ liệu, ma trận tensor dense vector (BAAI/bge-m3 1024 chiều), quy trình Indexing Ngoại tuyến vs Phục vụ Trực tuyến, và cấu trúc Modularity đồ thị tri thức pháp lý.",
        "tab5_sec1_title": "1. Chu Trình Phục Vụ Thực Tế: Pha Ngoại Tuyến (Offline) vs Trực Tuyến (Online Real-time)",
        "tab5_sec2_title": "2. Không Gian Vector Dense & Ma Trận Tensor Đã Tính Sẵn",
        "tab5_sec3_title": "3. Cấu Trúc Đồ Thị Tri Thức Pháp Lý (NetworkX & Các Cụm Modularity)",
    }
}


def t(key: str, lang: str = "en") -> str:
    """Retrieve translated text string for the given key and language."""
    pack = I18N.get(lang, I18N["en"])
    return pack.get(key, I18N["en"].get(key, key))
