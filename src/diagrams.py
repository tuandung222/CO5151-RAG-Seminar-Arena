"""
Mermaid Architecture Diagrams for RAG Paradigms (Bilingual: English & Vietnamese).
Rendered as responsive vector SVGs with interactive toolbar in Streamlit.
"""

DIAGRAMS_EN = {
    "naive_rag": """
flowchart LR
    Q["User Query"] --> RET["Hybrid Retriever<br/>(BM25 / BGE-M3)"]
    RET --> CHUNKS["Top-k Retrieved Chunks<br/>(Local Context)"]
    CHUNKS --> CONCAT["Blind Context Injection Prompt"]
    Q --> CONCAT
    CONCAT --> LLM["Generator LLM<br/>(Instruct Model)"]
    LLM --> ANS["Unverified Response<br/>(Vulnerable to Distractor Poisoning)"]
    
    style RET fill:#f1f5f9,stroke:#475569,stroke-width:2px
    style CHUNKS fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style ANS fill:#fee2e2,stroke:#dc2626,stroke-width:2px
""",
    "self_rag": """
flowchart LR
    Q["User Query"] --> DECIDE{"[Retrieve] Token?<br/>External Lookup"}
    DECIDE -->|No| PARAM["Parametric Generation<br/>(Internal Memory)"]
    DECIDE -->|Yes| RET["Dense BGE-M3 + BM25<br/>(Passage Retrieval)"]
    RET --> CRITIC{"[IsREL] Critic<br/>Passage Filter"}
    CRITIC -->|Invalid / Expired| REJECT["Prune Distractor<br/>(IsREL: IRRELEVANT)"]
    CRITIC -->|Valid| GROUNDED["Verified Context<br/>(IsREL: RELEVANT)"]
    GROUNDED --> GEN["Grounded Generation<br/>(In-Context Synthesis)"]
    GEN --> SUP{"[IsSUP] Critic<br/>Attribution"}
    SUP --> OUT["Final Verified Response<br/>Utility: 5/5 [IsUSE]"]
    
    style DECIDE fill:#f8fafc,stroke:#64748b,stroke-width:2px
    style CRITIC fill:#ffedd5,stroke:#ea580c,stroke-width:2px
    style REJECT fill:#fee2e2,stroke:#dc2626,stroke-width:2px
    style OUT fill:#ecfdf5,stroke:#059669,stroke-width:2px
""",
    "graphrag": """
flowchart TD
    CORPUS["Statutory Legal Corpus<br/>(15 Labor Code Articles)"] --> EXTRACT["Entity & Cross-Reference Extraction"]
    EXTRACT --> GRAPH["Knowledge Graph Network<br/>(NetworkX G = V, E)"]
    GRAPH --> CLUSTER["Hierarchical Community Detection<br/>(Newman Modularity Q)"]
    
    CLUSTER --> COMM1["Community 1: Labor Contracts & Probation"]
    CLUSTER --> COMM2["Community 2: Termination & Severance Compensation"]
    CLUSTER --> COMM3["Community 3: Discipline & Special Protections"]
    
    COMM1 --> MAP1["MAP: Community Summary 1"]
    COMM2 --> MAP2["MAP: Community Summary 2"]
    COMM3 --> MAP3["MAP: Community Summary 3"]
    
    MAP1 --> REDUCE["REDUCE: Global Multi-Community Synthesis"]
    MAP2 --> REDUCE
    MAP3 --> REDUCE
    
    REDUCE --> GLOBAL_ANS["Comprehensive Global Synthesis Response<br/>(Zero Blind Spots Across Chapters)"]
    
    style GRAPH fill:#f1f5f9,stroke:#475569,stroke-width:2px
    style REDUCE fill:#ffedd5,stroke:#c2410c,stroke-width:2px
    style GLOBAL_ANS fill:#ecfdf5,stroke:#059669,stroke-width:2px
""",
    "flare": """
flowchart LR
    Q["User Query"] --> DRAFT["Sentence-by-Sentence Forward Drafting"]
    DRAFT --> CONF{"Token Confidence Check<br/>P(token) &lt; Threshold &theta; ?"}
    CONF -->|High Confidence| KEEP["Retain Drafted Sentence<br/>(Zero Retrieval Overhead)"]
    CONF -->|Low Confidence| SEARCH["Active Tool Call:<br/>Search(Sub-Query)"]
    SEARCH --> RET["Targeted BGE-M3 Retrieval"]
    RET --> REWRITE["Fact-Grounded Sentence Rewriting"]
    KEEP --> NEXT["Sentence Concatenation"]
    REWRITE --> NEXT
    
    style CONF fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style SEARCH fill:#ffedd5,stroke:#ea580c,stroke-width:2px
    style REWRITE fill:#ecfdf5,stroke:#059669,stroke-width:2px
""",
    "offline_vs_online": """
flowchart TD
    subgraph OFFLINE["1. Offline Indexing Phase (One-Time Ingestion Pipeline)"]
        CORPUS["Labor Code Corpus (15 Articles)"] --> CHUNKING["Chunking & Metadata Extraction"]
        CHUNKING --> EMB_GEN["Dense Neural Encoder<br/>(BAAI/bge-m3: d=1024, L2 Normalized)"]
        CHUNKING --> BM25_IDX["Sparse BM25 Inverted Index"]
        CHUNKING --> KG_BUILD["Statutory Cross-Reference Graph Construction"]
        KG_BUILD --> COMM_DET["Greedy Modularity Community Detection"]
        EMB_GEN --> NPY_FILE[("corpus_embeddings.npy<br/>Matrix: 15 x 1024 float32")]
    end

    subgraph ONLINE["2. Online Serving Phase (Real-Time Query Pipeline)"]
        USER_Q["User Query (q)"] --> Q_ENC["Encode Single Query<br/>(BAAI/bge-m3: 1 x 1024)"]
        Q_ENC --> DOT_PROD["RAM Dot Product Similarity (&lt; 0.1ms)<br/>Scores = NPY_FILE · q_vec"]
        USER_Q --> BM25_SCORE["BM25 Lexical Scoring"]
        DOT_PROD --> RRF["Reciprocal Rank Fusion (RRF: k=60)"]
        BM25_SCORE --> RRF
        RRF --> TOPK["Top-k Relevant Chunks"]
        TOPK --> LLM_GEN["LLM Generator / Self-RAG Reflection Loop"]
        LLM_GEN --> FINAL_ANS["Final Grounded Response"]
    end

    style OFFLINE fill:#f8fafc,stroke:#475569,stroke-width:2px
    style ONLINE fill:#f8fafc,stroke:#0284c7,stroke-width:2px
    style NPY_FILE fill:#ecfdf5,stroke:#059669,stroke-width:2px
    style DOT_PROD fill:#ffedd5,stroke:#ea580c,stroke-width:2px
""",
    "knowledge_graph_full": """
flowchart TB
    subgraph C1["Community 1: Labor Contracts & Probation"]
        D13["Art 13: Labor Contract"]
        D20["Art 20: Contract Types"]
        D24["Art 24: Probation Agreement"]
        D25["Art 25: Maximum Probation Duration"]
        D26["Art 26: Probation Wage"]
        D27["Art 27: Conclusion of Probation"]
        
        D13 -->|contract type| D20
        D13 -->|agreement| D24
        D24 -->|duration limit| D25
        D24 -->|wage criteria| D26
        D24 -->|evaluation outcome| D27
    end

    subgraph C2["Community 2: Termination & Severance Compensation"]
        D34["Art 34: Termination Grounds"]
        D35["Art 35: Unilateral Employee"]
        D36["Art 36: Unilateral Employer"]
        D37["Art 37: Unilateral Restrictions"]
        D40["Art 40: Illegal Employee Breach"]
        D41["Art 41: Illegal Employer Damages"]
        D46["Art 46: Severance Allowance"]
        
        D20 -.->|expiration| D34
        D34 -->|employee rights| D35
        D34 -->|employer rights| D36
        D36 -->|statutory restriction| D37
        D35 -->|damages liability| D40
        D36 -->|damages liability| D41
        D34 -->|allowance duty| D46
    end

    subgraph C3["Community 3: Discipline & Special Protections"]
        D122["Art 122: Disciplinary Principles"]
        D125["Art 125: Dismissal Measures"]
        
        D37 -.->|maternity protection| D122
        D122 -->|highest sanction| D125
        D36 -.->|unauthorized absence| D125
    end

    style C1 fill:#f0f9ff,stroke:#0284c7,stroke-width:2px
    style C2 fill:#fff7ed,stroke:#ea580c,stroke-width:2px
    style C3 fill:#faf5ff,stroke:#9333ea,stroke-width:2px
    style D25 fill:#ecfdf5,stroke:#059669,stroke-width:2px
"""
}

DIAGRAMS_VI = {
    "naive_rag": """
flowchart LR
    Q["Câu hỏi của Người dùng"] --> RET["Bộ truy xuất lai<br/>(BM25 / BGE-M3)"]
    RET --> CHUNKS["Top-k Đoạn văn bản bốc được<br/>(Ngữ cảnh cục bộ)"]
    CHUNKS --> CONCAT["Prompt nối ngữ cảnh thô<br/>(Không phản tư)"]
    Q --> CONCAT
    CONCAT --> LLM["Mô hình LLM sinh phản hồi<br/>(Instruct Model)"]
    LLM --> ANS["Câu trả lời chưa kiểm chứng<br/>(Dễ bị ngộ độc tài liệu bẫy)"]
    
    style RET fill:#f1f5f9,stroke:#475569,stroke-width:2px
    style CHUNKS fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style ANS fill:#fee2e2,stroke:#dc2626,stroke-width:2px
""",
    "self_rag": """
flowchart LR
    Q["Câu hỏi của Người dùng"] --> DECIDE{"Cổng [Retrieve]?<br/>Có cần tra cứu ngoài?"}
    DECIDE -->|Không| PARAM["Sinh từ bộ nhớ trong<br/>(Parametric Weights)"]
    DECIDE -->|Có| RET["Truy xuất BGE-M3 + BM25<br/>(Đoạn luật ứng viên)"]
    RET --> CRITIC{"Bộ phản tư [IsREL]<br/>Lọc đoạn văn bản"}
    CRITIC -->|Hết hạn / Lỗi thời| REJECT["Loại bỏ văn bản bẫy<br/>(IsREL: IRRELEVANT)"]
    CRITIC -->|Hợp lệ| GROUNDED["Ngữ cảnh căn cứ chuẩn<br/>(IsREL: RELEVANT)"]
    GROUNDED --> GEN["Sinh phản hồi có căn cứ<br/>(Grounded Generation)"]
    GEN --> SUP{"Bộ kiểm định [IsSUP]<br/>Đối chiếu chứng cứ"}
    SUP --> OUT["Câu trả lời xác thực hoàn tất<br/>Độ hữu dụng: 5/5 [IsUSE]"]
    
    style DECIDE fill:#f8fafc,stroke:#64748b,stroke-width:2px
    style CRITIC fill:#ffedd5,stroke:#ea580c,stroke-width:2px
    style REJECT fill:#fee2e2,stroke:#dc2626,stroke-width:2px
    style OUT fill:#ecfdf5,stroke:#059669,stroke-width:2px
""",
    "graphrag": """
flowchart TD
    CORPUS["Kho ngữ liệu luật lao động<br/>(15 Điều luật BLLĐ 2019)"] --> EXTRACT["Trích xuất thực thể & Dẫn chiếu chéo"]
    EXTRACT --> GRAPH["Mạng lưới Đồ thị tri thức<br/>(NetworkX G = V, E)"]
    GRAPH --> CLUSTER["Phát hiện cấu trúc cộng đồng phân cấp<br/>(Tối đa hóa Modularity Q)"]
    
    CLUSTER --> COMM1["Cụm 1: Giao kết & Chế định Thử việc"]
    CLUSTER --> COMM2["Cụm 2: Chấm dứt HĐLĐ & Bồi thường"]
    CLUSTER --> COMM3["Cụm 3: Kỷ luật & Bảo vệ đặc thù"]
    
    COMM1 --> MAP1["MAP: Báo cáo tóm tắt Cụm 1"]
    COMM2 --> MAP2["MAP: Báo cáo tóm tắt Cụm 2"]
    COMM3 --> MAP3["MAP: Báo cáo tóm tắt Cụm 3"]
    
    MAP1 --> REDUCE["REDUCE: Tổng hợp đa cụm toàn cục"]
    MAP2 --> REDUCE
    MAP3 --> REDUCE
    
    REDUCE --> GLOBAL_ANS["Câu trả lời bao quát toàn diện<br/>(Khắc phục triệt để điểm mù cục bộ)"]
    
    style GRAPH fill:#f1f5f9,stroke:#475569,stroke-width:2px
    style REDUCE fill:#ffedd5,stroke:#c2410c,stroke-width:2px
    style GLOBAL_ANS fill:#ecfdf5,stroke:#059669,stroke-width:2px
""",
    "flare": """
flowchart LR
    Q["Câu hỏi của Người dùng"] --> DRAFT["Soạn thảo dự phóng từng câu liên tiếp"]
    DRAFT --> CONF{"Đánh giá độ tự tin Token<br/>P(token) &lt; Ngưỡng &theta; ?"}
    CONF -->|Tự tin cao| KEEP["Giữ nguyên câu đã soạn<br/>(Tiết kiệm 67% chi phí truy xuất)"]
    CONF -->|Tự tin thấp| SEARCH["Kích hoạt Tool Call chủ động:<br/>Search(Ý_con_cần_tra)"]
    SEARCH --> RET["Truy xuất BGE-M3 đúng trọng tâm"]
    RET --> REWRITE["Viết lại câu dựa trên căn cứ luật"]
    KEEP --> NEXT["Ghép nối câu hoàn chỉnh"]
    REWRITE --> NEXT
    
    style CONF fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style SEARCH fill:#ffedd5,stroke:#ea580c,stroke-width:2px
    style REWRITE fill:#ecfdf5,stroke:#059669,stroke-width:2px
""",
    "offline_vs_online": """
flowchart TD
    subgraph OFFLINE["1. Pha Đánh chỉ mục Ngoại tuyến (Offline Pipeline - 1 lần duy nhất)"]
        CORPUS["Ngữ liệu Bộ luật Lao động (15 Điều luật)"] --> CHUNKING["Phân mảnh & Trích xuất siêu dữ liệu"]
        CHUNKING --> EMB_GEN["Bộ mã hóa Dense Vector<br/>(BAAI/bge-m3: d=1024, Chuẩn hóa L2)"]
        CHUNKING --> BM25_IDX["Chỉ mục đảo thưa BM25 (Sparse Index)"]
        CHUNKING --> KG_BUILD["Xây dựng Đồ thị dẫn chiếu pháp lý"]
        KG_BUILD --> COMM_DET["Phân cụm cấu trúc cộng đồng Modularity"]
        EMB_GEN --> NPY_FILE[("corpus_embeddings.npy<br/>Ma trận: 15 x 1024 float32")]
    end

    subgraph ONLINE["2. Pha Phục vụ Trực tuyến (Online Serving - Real-time dưới 0.1ms)"]
        USER_Q["Câu hỏi người dùng (q)"] --> Q_ENC["Mã hóa 1 vector câu hỏi<br/>(BAAI/bge-m3: 1 x 1024)"]
        Q_ENC --> DOT_PROD["Tích vô hướng trực tiếp trên RAM (&lt; 0.1ms)<br/>Điểm tương đồng = Ma_trận · q_vec"]
        USER_Q --> BM25_SCORE["Tính điểm từ khóa BM25"]
        DOT_PROD --> RRF["Hợp nhất xếp hạng (RRF: k=60)"]
        BM25_SCORE --> RRF
        RRF --> TOPK["Top-k Đoạn văn bản phù hợp nhất"]
        TOPK --> LLM_GEN["Mô hình LLM / Vòng phản tư Self-RAG"]
        LLM_GEN --> FINAL_ANS["Câu trả lời xác thực chuẩn xác"]
    end

    style OFFLINE fill:#f8fafc,stroke:#475569,stroke-width:2px
    style ONLINE fill:#f8fafc,stroke:#0284c7,stroke-width:2px
    style NPY_FILE fill:#ecfdf5,stroke:#059669,stroke-width:2px
    style DOT_PROD fill:#ffedd5,stroke:#ea580c,stroke-width:2px
""",
    "knowledge_graph_full": """
flowchart TB
    subgraph C1["Cụm 1: Giao kết & Chế định Thử việc"]
        D13["Điều 13: Hợp đồng lao động"]
        D20["Điều 20: Các loại hợp đồng"]
        D24["Điều 24: Thỏa thuận thử việc"]
        D25["Điều 25: Thời hạn thử việc tối đa"]
        D26["Điều 26: Tiền lương thử việc"]
        D27["Điều 27: Kết thúc thời gian thử việc"]
        
        D13 -->|loại hợp đồng| D20
        D13 -->|thỏa thuận| D24
        D24 -->|giới hạn thời hạn| D25
        D24 -->|tiêu chí tiền lương| D26
        D24 -->|kết quả đánh giá| D27
    end

    subgraph C2["Cụm 2: Chấm dứt HĐLĐ & Trợ cấp bồi thường"]
        D34["Điều 34: Các trường hợp chấm dứt"]
        D35["Điều 35: NLĐ đơn phương chấm dứt"]
        D36["Điều 36: NSDLĐ đơn phương chấm dứt"]
        D37["Điều 37: Hạn chế quyền đơn phương"]
        D40["Điều 40: Nghĩa vụ khi NLĐ vi phạm"]
        D41["Điều 41: Nghĩa vụ bồi thường của NSDLĐ"]
        D46["Điều 46: Trợ cấp thôi việc"]
        
        D20 -.->|hết hạn HĐLĐ| D34
        D34 -->|quyền của NLĐ| D35
        D34 -->|quyền của NSDLĐ| D36
        D36 -->|giới hạn pháp lý| D37
        D35 -->|nghĩa vụ bồi thường| D40
        D36 -->|nghĩa vụ bồi thường| D41
        D34 -->|nghĩa vụ trợ cấp| D46
    end

    subgraph C3["Cụm 3: Kỷ luật lao động & Bảo vệ đặc thù"]
        D122["Điều 122: Nguyên tắc xử lý kỷ luật"]
        D125["Điều 125: Áp dụng kỷ luật sa thải"]
        
        D37 -.->|bảo vệ thai sản & nuôi con nhỏ| D122
        D122 -->|hình thức xử lý cao nhất| D125
        D36 -.->|xử lý tự ý bỏ việc 05 ngày| D125
    end

    style C1 fill:#f0f9ff,stroke:#0284c7,stroke-width:2px
    style C2 fill:#fff7ed,stroke:#ea580c,stroke-width:2px
    style C3 fill:#faf5ff,stroke:#9333ea,stroke-width:2px
    style D25 fill:#ecfdf5,stroke:#059669,stroke-width:2px
"""
}


def get_diagram(name: str, lang: str = "en") -> str:
    """Retrieve Mermaid diagram code according to active language ('en' or 'vi')."""
    if lang == "vi":
        return DIAGRAMS_VI.get(name, DIAGRAMS_EN.get(name, ""))
    return DIAGRAMS_EN.get(name, "")


# Backwards compatibility constants (default to EN)
DIAGRAM_NAIVE_RAG = DIAGRAMS_EN["naive_rag"]
DIAGRAM_SELF_RAG = DIAGRAMS_EN["self_rag"]
DIAGRAM_GRAPHRAG = DIAGRAMS_EN["graphrag"]
DIAGRAM_FLARE = DIAGRAMS_EN["flare"]
DIAGRAM_OFFLINE_VS_ONLINE = DIAGRAMS_EN["offline_vs_online"]
DIAGRAM_KNOWLEDGE_GRAPH_FULL = DIAGRAMS_EN["knowledge_graph_full"]
