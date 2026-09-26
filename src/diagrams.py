"""
Mermaid Architecture Diagrams for RAG Paradigms.
Rendered as responsive vector SVGs inside Streamlit.
"""

DIAGRAM_NAIVE_RAG = """
flowchart LR
    Q["User Query"] --> RET["Hybrid Retriever (BM25 / BGE-M3)"]
    RET --> CHUNKS["Top-k Retrieved Chunks (Local Context)"]
    CHUNKS --> CONCAT["Blind Context Injection Prompt"]
    Q --> CONCAT
    CONCAT --> LLM["Generator LLM (Instruct Model)"]
    LLM --> ANS["Unverified Response (Vulnerable to Distractor Poisoning)"]
    
    style RET fill:#f1f5f9,stroke:#475569,stroke-width:2px
    style CHUNKS fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style ANS fill:#fee2e2,stroke:#dc2626,stroke-width:2px
"""

DIAGRAM_SELF_RAG = """
flowchart LR
    Q["User Query"] --> DECIDE{"[Retrieve]?<br/>External Lookup"}
    DECIDE -- No --> PARAM["Parametric Generation"]
    DECIDE -- Yes --> RET["Dense BGE-M3 + BM25"]
    RET --> CRITIC{"[IsREL] Critic<br/>Passage Filter"}
    CRITIC -- Invalid/Expired --> REJECT["Prune Distractor"]
    CRITIC -- Valid --> GROUNDED["Verified Context"]
    GROUNDED --> GEN["Grounded Generation"]
    GEN --> SUP{"[IsSUP]<br/>Attribution"}
    SUP --> OUT["Final Verified Response<br/>([IsUSE: 5/5])"]
    
    style DECIDE fill:#f8fafc,stroke:#64748b,stroke-width:2px
    style CRITIC fill:#ffedd5,stroke:#ea580c,stroke-width:2px
    style REJECT fill:#fee2e2,stroke:#dc2626,stroke-width:2px
    style OUT fill:#ecfdf5,stroke:#059669,stroke-width:2px
"""

DIAGRAM_GRAPHRAG = """
flowchart TD
    CORPUS["Statutory Legal Corpus"] --> EXTRACT["Entity & Cross-Reference Extraction"]
    EXTRACT --> GRAPH["Knowledge Graph (NetworkX)"]
    GRAPH --> CLUSTER["Hierarchical Community Detection (Modularity Q)"]
    
    CLUSTER --> COMM1["Community 1: Labor Contracts & Probation"]
    CLUSTER --> COMM2["Community 2: Termination & Severance Compensation"]
    CLUSTER --> COMM3["Community 3: Discipline & Special Protections"]
    
    COMM1 --> MAP1["MAP: Community Summary 1"]
    COMM2 --> MAP2["MAP: Community Summary 2"]
    COMM3 --> MAP3["MAP: Community Summary 3"]
    
    MAP1 --> REDUCE["REDUCE: Global Multi-Community Synthesis"]
    MAP2 --> REDUCE
    MAP3 --> REDUCE
    
    REDUCE --> GLOBAL_ANS["Comprehensive Global Synthesis Response"]
    
    style GRAPH fill:#f1f5f9,stroke:#475569,stroke-width:2px
    style REDUCE fill:#ffedd5,stroke:#c2410c,stroke-width:2px
    style GLOBAL_ANS fill:#ecfdf5,stroke:#059669,stroke-width:2px
"""

DIAGRAM_FLARE = """
flowchart LR
    Q["User Query"] --> DRAFT["Sentence-by-Sentence Forward Drafting"]
    DRAFT --> CONF{"Token Confidence Assessment<br/>P(token) < Threshold θ ?"}
    CONF -- High Confidence --> KEEP["Retain Drafted Sentence (Zero Retrieval Overhead)"]
    CONF -- Low Confidence --> SEARCH["Active Tool Call: Search(Sub-Query)"]
    SEARCH --> RET["Targeted BGE-M3 Retrieval"]
    RET --> REWRITE["Fact-Grounded Sentence Rewriting"]
    KEEP --> NEXT["Sentence Concatenation"]
    REWRITE --> NEXT
    
    style CONF fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style SEARCH fill:#ffedd5,stroke:#ea580c,stroke-width:2px
    style REWRITE fill:#ecfdf5,stroke:#059669,stroke-width:2px
"""

DIAGRAM_OFFLINE_VS_ONLINE = """
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
"""

DIAGRAM_KNOWLEDGE_GRAPH_FULL = """
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

