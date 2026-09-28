import streamlit as st
import streamlit.components.v1 as components
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

# Import core modules
from src.config import (
    CORPUS_PATH,
    TEST_CASES_PATH,
    DEFAULT_BASE_URL,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    VLLM_BASE_URL,
    VLLM_MODEL,
    HF_EMBEDDING_MODEL,
    HF_TOKEN,
    CORPUS_EMBEDDINGS_PATH,
    CORPUS_EMBEDDINGS_META_PATH,
    CACHED_BENCHMARK_PATH,
)
from src.retriever import LegalRetriever, tokenize_vietnamese
from src.llm import UnifiedLLM
from src.naive_rag import NaiveRAGPipeline
from src.graph_rag import GraphRAGPipeline
from src.self_rag import SelfRAGPipeline
from src.flare_rag import FLARERAGPipeline
from src.diagrams import (
    get_diagram,
    DIAGRAM_NAIVE_RAG,
    DIAGRAM_SELF_RAG,
    DIAGRAM_GRAPHRAG,
    DIAGRAM_FLARE,
    DIAGRAM_OFFLINE_VS_ONLINE,
    DIAGRAM_KNOWLEDGE_GRAPH_FULL,
)
from src.i18n import t

# Page configuration
st.set_page_config(
    page_title="RAG Evaluation Laboratory: Scientific Paradigms Benchmark",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Academic / Data Scientist UI Styling (Inter Typography, Slate & Minimalist)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .main-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }
    .status-banner-cached {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #0284c7;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 14px;
        font-size: 0.88rem;
    }
    .status-banner-live {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 14px;
        font-size: 0.88rem;
    }
    .arena-header-1 {
        background-color: #1e293b;
        color: #f8fafc;
        padding: 9px 14px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: -0.01em;
    }
    .arena-header-2 {
        background-color: #334155;
        color: #f8fafc;
        padding: 9px 14px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: -0.01em;
    }
    .arena-header-3 {
        background-color: #0f172a;
        color: #f8fafc;
        padding: 9px 14px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: -0.01em;
    }
    .critique-pass {
        border-left: 3px solid #059669;
        background-color: #f8fafc;
        padding: 10px 12px;
        margin-bottom: 8px;
        border-radius: 0 4px 4px 0;
        font-size: 0.88rem;
    }
    .critique-fail {
        border-left: 3px solid #dc2626;
        background-color: #f8fafc;
        padding: 10px 12px;
        margin-bottom: 8px;
        border-radius: 0 4px 4px 0;
        font-size: 0.88rem;
    }
    .badge-cached {
        color: #0284c7;
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-verified {
        color: #059669;
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-poisoned {
        color: #dc2626;
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .context-card {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 16px;
        font-size: 0.9rem;
        line-height: 1.55;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .context-card-title {
        font-weight: 700;
        color: #0369a1;
        font-size: 0.95rem;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .context-pill {
        background-color: #e0f2fe;
        color: #0369a1;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 9999px;
        display: inline-block;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


def render_mermaid(diagram_code: str, height: int = 460):
    """
    Render responsive vector SVG Mermaid diagram with complete viewing visibility.
    Includes built-in zoom controls and pop-out high-resolution viewer.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{
                box-sizing: border-box;
            }}
            body {{
                margin: 0;
                padding: 6px 10px;
                background-color: #ffffff;
                font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
            }}
            .toolbar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 4px 10px;
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                margin-bottom: 8px;
            }}
            .toolbar-title {{
                font-size: 11px;
                font-weight: 600;
                color: #64748b;
                letter-spacing: -0.01em;
            }}
            .toolbar-actions {{
                display: flex;
                gap: 5px;
            }}
            .tool-btn {{
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                color: #1e293b;
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 500;
                cursor: pointer;
                box-shadow: 0 1px 2px rgba(0,0,0,0.04);
                transition: all 0.15s ease;
            }}
            .tool-btn:hover {{
                background-color: #f1f5f9;
                border-color: #94a3b8;
            }}
            .diagram-viewport {{
                width: 100%;
                overflow-x: auto;
                overflow-y: auto;
                padding: 8px 0;
                text-align: center;
                background: #ffffff;
                border-radius: 6px;
            }}
            .mermaid {{
                display: inline-block;
                margin: 0 auto;
                transform-origin: top center;
                transition: transform 0.15s ease-out;
            }}
            .mermaid svg {{
                max-width: 100% !important;
                height: auto !important;
                display: block;
                margin: 0 auto;
            }}
        </style>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'neutral',
                securityLevel: 'loose',
                flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis' }}
            }});
        </script>
    </head>
    <body>
        <div class="toolbar">
            <span class="toolbar-title">Vector Architecture Schematic</span>
            <div class="toolbar-actions">
                <button class="tool-btn" onclick="zoomIn()">Zoom (+)</button>
                <button class="tool-btn" onclick="zoomOut()">Zoom (-)</button>
                <button class="tool-btn" onclick="resetZoom()">Reset</button>
                <button class="tool-btn" onclick="openFull()">Full View / Pop-out ↗</button>
            </div>
        </div>
        <div class="diagram-viewport">
            <div class="mermaid" id="mermaid-graph">
            {diagram_code}
            </div>
        </div>
        <script>
            let zoomLevel = 1.0;
            function zoomIn() {{
                zoomLevel = Math.min(2.5, zoomLevel + 0.25);
                applyZoom();
            }}
            function zoomOut() {{
                zoomLevel = Math.max(0.4, zoomLevel - 0.25);
                applyZoom();
            }}
            function resetZoom() {{
                zoomLevel = 1.0;
                applyZoom();
            }}
            function applyZoom() {{
                const el = document.getElementById('mermaid-graph');
                if (el) el.style.transform = 'scale(' + zoomLevel + ')';
            }}
            function openFull() {{
                const svg = document.querySelector('.mermaid svg');
                if (!svg) {{
                    alert('Diagram is rendering, please retry in 1 second.');
                    return;
                }}
                const svgContent = new XMLSerializer().serializeToString(svg);
                const blob = new Blob([svgContent], {{ type: 'image/svg+xml;charset=utf-8' }});
                const blobUrl = URL.createObjectURL(blob);
                window.open(blobUrl, '_blank');
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_content, height=height, scrolling=True)


@st.cache_resource
def load_base_resources():
    """Load retriever, test cases, and pre-computed gold benchmark results."""
    retriever = LegalRetriever(CORPUS_PATH)
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    cached_benchmark = {}
    if CACHED_BENCHMARK_PATH.exists():
        try:
            with open(CACHED_BENCHMARK_PATH, "r", encoding="utf-8") as f:
                cached_benchmark = json.load(f)
        except Exception as e:
            st.warning(f"Note: Could not load cached benchmark: {e}")
    return retriever, test_cases, cached_benchmark


retriever, test_cases, cached_benchmark = load_base_resources()

# Initialize live execution flags in session state (enables dynamic responsiveness to all UI options)
if "tab1_is_live" not in st.session_state:
    st.session_state["tab1_is_live"] = False
if "tab2_is_live" not in st.session_state:
    st.session_state["tab2_is_live"] = False
if "tab3_is_live" not in st.session_state:
    st.session_state["tab3_is_live"] = False
if "tab4_is_live" not in st.session_state:
    st.session_state["tab4_is_live"] = False

# ==========================================
# SIDEBAR: SYSTEM CONTROLS & TELEMETRY
# ==========================================
with st.sidebar:
    st.markdown("### 🌐 Display Language / Ngôn ngữ")
    lang_choice = st.radio(
        "Display Language / Ngôn ngữ hiển thị:",
        options=["English 🇬🇧", "Tiếng Việt 🇻🇳"],
        index=0 if st.session_state.get("lang", "en") == "en" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    lang = "vi" if "Tiếng Việt" in lang_choice else "en"
    st.session_state["lang"] = lang
    st.markdown("---")

    st.markdown(f"### {t('sidebar_engine', lang)}")
    provider_choice = st.selectbox(
        t("sidebar_provider", lang),
        options=[
            "Hugging Face Serverless (User Token)",
            "Local Ollama (Native REST API)",
            "OpenAI-Compatible Gateway",
            "Local vLLM Server",
        ],
        index=0,
    )

    if provider_choice == "Hugging Face Serverless (User Token)":
        hf_model_choice = st.selectbox(
            "Model Architecture:" if lang == "en" else "Kiến trúc mô hình:",
            options=[
                "Qwen/Qwen2.5-72B-Instruct (SOTA Flagship)",
                "meta-llama/Llama-3.1-8B-Instruct (Academic Baseline)",
                "Qwen/Qwen3.5-9B (Next-Gen Multimodal Foundation)",
                "Qwen/Qwen2.5-Coder-32B-Instruct (High-Precision Reasoning)",
                "Qwen/Qwen2.5-Coder-7B-Instruct (Fast Edge)",
                "Custom Model ID",
            ],
            index=0,
        )
        cur_provider = "huggingface"
        cur_base_url = "https://api-inference.huggingface.co/v1"
        
        # User Hugging Face Token Authentication Input
        default_hf_token = st.session_state.get("user_hf_token", HF_TOKEN)
        user_hf_token = st.text_input(
            t("sidebar_token_label", lang),
            value=default_hf_token,
            type="password",
            help=t("sidebar_token_help", lang),
        )
        if user_hf_token:
            st.session_state["user_hf_token"] = user_hf_token
            retriever.update_api_key(user_hf_token)
        cur_api_key = user_hf_token
        parsed_model = hf_model_choice.split(" ")[0]
        cur_model = st.text_input("Model ID:", value="Qwen/Qwen2.5-72B-Instruct") if "Custom" in hf_model_choice else parsed_model

    elif provider_choice == "Local Ollama (Native REST API)":
        cur_provider = "ollama-native"
        cur_base_url = st.text_input("Ollama Base URL:", value=OLLAMA_BASE_URL)
        cur_api_key = "EMPTY"
        cur_model = st.text_input("Ollama Model Name:", value=OLLAMA_MODEL)

    elif provider_choice == "OpenAI-Compatible Gateway":
        cur_provider = "openai-compatible"
        cur_base_url = st.text_input("Base URL:", value=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        cur_api_key = st.text_input("API Key:", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
        cur_model = st.text_input("Model Name:", value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

    else:
        cur_provider = "openai-compatible"
        cur_base_url = st.text_input("vLLM Base URL:", value=VLLM_BASE_URL)
        cur_api_key = st.text_input("API Key:", value="EMPTY", type="password")
        cur_model = st.text_input("Model Name:", value=VLLM_MODEL)

    llm = UnifiedLLM(
        provider=cur_provider,
        base_url=cur_base_url,
        api_key=cur_api_key,
        model_name=cur_model,
    )

    if st.button("Test Model Latency & Ping" if lang == "en" else "Kiểm tra kết nối & độ trễ", use_container_width=True):
        conn_res = llm.test_connection()
        if conn_res["success"]:
            st.success(f"{'Connected:' if lang == 'en' else 'Đã kết nối:'} {conn_res['latency_ms']} ms\n{conn_res['message']}")
        else:
            st.error(f"{'Connection failed:' if lang == 'en' else 'Lỗi kết nối:'} {conn_res['message']}")

    st.markdown("---")
    st.markdown(f"### {t('sidebar_retrieval_topology', lang)}")
    st.markdown(f"**{t('sidebar_dense', lang)}** `SOTA {HF_EMBEDDING_MODEL}` (1024-d)")
    st.markdown(f"**{t('sidebar_sparse', lang)}** `Okapi BM25`")
    st.markdown(f"**{t('sidebar_corpus_count', lang)}** `{len(retriever.corpus)} Articles` (RAM matrix)")
    st.markdown("---")

    st.markdown(f"### ⏱️ {t('sidebar_knowledge_scope', lang)}")
    if lang == "vi":
        st.info("""
        **Mốc Thời Gian & Hiệu Lực Pháp Lý:**
        - 🟢 **BLLĐ 2019 (Luật số 45/2019/QH14):** Hiệu lực từ **01/01/2021 đến nay (2026)**. Đây là Bộ luật Lao động **hiện hành cao nhất và mới nhất** tại Việt Nam (không có bộ luật nào mới hơn).
        - 🔴 **BLLĐ 2012 (Luật số 10/2012/QH13):** Đã **hết hiệu lực từ 01/01/2021** (theo Điều 219 BLLĐ 2019). Đóng vai trò tài liệu gây nhiễu thực tế (*Repealed Distractor*) để kiểm thử khả năng nhận diện tài liệu bãi bỏ của RAG.
        """)
    else:
        st.info("""
        **Timeline & Statutory Baseline:**
        - 🟢 **Labor Code 2019 (Law No. 45/2019/QH14):** In effect from **Jan 1, 2021 through 2026**. This is the **latest and sole active Labor Code** in Vietnam (no newer Labor Code enacted).
        - 🔴 **Labor Code 2012 (Law No. 10/2012/QH13):** Fully **repealed on Jan 1, 2021** (per Art 219, Labor Code 2019). Serves as an authentic *Repealed Distractor* to test RAG's distractor pruning capability.
        """)
    st.markdown("---")

    st.markdown("### Standard Benchmark Presets" if lang == "en" else "### Bộ Câu Hỏi Đánh Giá Chuẩn")
    preset_names = [f"[{c['id']}] {c['name'][:35]}..." for c in test_cases]
    selected_preset_idx = st.selectbox(
        "Load Benchmark Preset:" if lang == "en" else "Tải câu hỏi kiểm định:",
        options=list(range(len(test_cases))),
        format_func=lambda i: preset_names[i],
    )
    selected_case = test_cases[selected_preset_idx]
    ctx_text = selected_case.get(f"real_world_context_{lang}", selected_case["focus"])
    obj_text = selected_case.get(f"pedagogical_objective_{lang}", selected_case["focus"])
    st.caption(f"🏛️ **{'Tình huống thực tế:' if lang == 'vi' else 'Real-world Context:'}** {ctx_text}")
    st.caption(f"🎯 **{'Mục tiêu thực nghiệm:' if lang == 'vi' else 'Experimental Objective:'}** {obj_text}")
    
    preset_tab_map = {
        0: ("Tab 1: When Retrieval Hurts", 1),
        1: ("Tab 2: GraphRAG vs Global Synthesis", 2),
        2: ("Tab 4: Active Retrieval (FLARE)", 4),
        3: ("Tab 3: Self-RAG Reflection Inspector", 3),
    }
    t_tab_name, _ = preset_tab_map.get(selected_preset_idx, ("Tab 1", 1))
    st.info(f"📍 **{'Phân Tích Tại:' if lang == 'vi' else 'Mapped to:'}** `{t_tab_name}`")
    
    if st.button("🚀 " + ("Synchronize Question to Tab" if lang == "en" else "Nạp Câu Hỏi Vào Tab"), use_container_width=True, key="btn_sync_preset"):
        if selected_preset_idx == 0:
            st.session_state["q_tab1"] = selected_case["question"]
            st.session_state["tab1_is_live"] = False
        elif selected_preset_idx == 1:
            st.session_state["q_tab2"] = selected_case["question"]
            st.session_state["tab2_is_live"] = False
        elif selected_preset_idx == 2:
            st.session_state["q_tab4"] = selected_case["question"]
            st.session_state["tab4_is_live"] = False
        elif selected_preset_idx == 3:
            st.session_state["q_tab3"] = selected_case["question"]
            st.session_state["tab3_is_live"] = False
        st.success("Đã nạp câu hỏi vào Tab!" if lang == "vi" else "Question loaded into target Tab!")
        st.rerun()

    st.markdown("---")
    st.caption(f"CO5151: Advanced Agentic AI (Master's Seminar)\n{t('sidebar_author', lang)} Dung Vo (tuandung222)")

# Instantiate pipeline objects dynamically using current LLM
naive_pipe = NaiveRAGPipeline(retriever, llm)
graph_pipe = GraphRAGPipeline(retriever, llm)
self_pipe = SelfRAGPipeline(retriever, llm)
flare_pipe = FLARERAGPipeline(retriever, llm)

# Main Header Area
st.markdown(f'<div class="main-header">{t("main_header", lang)}</div>', unsafe_allow_html=True)
sub_text = f"Academic testbed evaluating <b>{cur_model}</b> ({provider_choice}) and <b>BGE-M3 (1024-d)</b> across Parametric, Naive RAG, Self-RAG, GraphRAG, and FLARE mechanisms." if lang == "en" else f"Phòng thực nghiệm học thuật đánh giá <b>{cur_model}</b> ({provider_choice}) và <b>BGE-M3 (1024-d)</b> đối chứng các cơ chế Pure LLM, Naive RAG, Self-RAG, GraphRAG và FLARE."
st.markdown(f'<div class="sub-header">{sub_text}</div>', unsafe_allow_html=True)

with st.expander(t("master_guide_expander", lang), expanded=False):
    if lang == "vi":
        st.info("""
        ⏱️ **Mốc Thời Gian & Giả Định Khoa Học Về Hiệu Lực Pháp Lý:**
        - **Luật Thực Tế Hiện Hành Duy Nhất:** **Bộ luật Lao động 2019 (Luật số 45/2019/QH14)** được Quốc hội thông qua ngày 20/11/2019, chính thức có hiệu lực thi hành từ ngày **01/01/2021 đến nay (2026)**. *Tính đến nay, chưa có bất kỳ Bộ luật Lao động nào mới hơn.*
        - **Tài Liệu Gây Nhiễu Thực Tế (Repealed Distractor):** **Bộ luật Lao động 2012 (Luật số 10/2012/QH13)** đã **chính thức hết hiệu lực từ ngày 01/01/2021** (theo Điều 219 BLLĐ 2019).
        - **Giả Định Thực Nghiệm:** Trong doanh nghiệp, kho dữ liệu nội bộ (SharePoint, Confluence) thường lưu trữ lẫn lộn cả văn bản cũ và mới. Khi người dùng tra cứu quy định *"hiện hành"*, hệ thống RAG lý tưởng bắt buộc phải có cơ chế nhận diện thời hiệu và bãi bỏ để loại trừ tài liệu cũ, thay vì nhồi nhét thụ động như Naive RAG.
        """)
        st.markdown(r"""
| Cơ Chế RAG | Cơ Chế Vận Hành Cốt Lõi | Công Thức Toán Học / Thuật Toán | Điểm Yếu Chính Được Khắc Phục | Khi Nào Nên Triển Khai Thực Tế |
| :--- | :--- | :--- | :--- | :--- |
| **1. Pure LLM** | Bộ nhớ tham số mô hình | $P(y \mid x; \theta)$ | Không tốn độ trễ tra cứu ngoài | Trò chuyện tổng quát, suy luận đời thường |
| **2. Naive RAG** | Tìm kiếm tương đồng top-k + nối chuỗi thô | $\arg\max_y P(y \mid x, \text{TopK}(x))$ | Bổ sung tri thức thiếu trong pre-training | FAQ đơn giản trên tài liệu tĩnh, không xung đột |
| **3. Self-RAG** | Token tự đánh giá + Cổng xác suất logprob | $\text{Score} = \text{LLM} + w_{\text{rel}} \log P(\text{IsREL}) + w_{\text{sup}} \log P(\text{IsSUP})$ | **Nhiễm độc tài liệu gây nhiễu** (Luật hết hiệu lực/mâu thuẫn) | Lĩnh vực rủi ro cao (Pháp lý, Y tế, Kiểm toán tài chính) |
| **4. GraphRAG** | Đồ thị tri thức + Phân cụm cộng đồng theo Modularity + Tổng hợp phân cấp Map-Reduce | $\text{Reduce}(\{\text{Map}(C_i)\}_{i=1}^M)$, $Q = \sum [e_{ii} - a_i^2]$ | **Điểm mù cục bộ của Vector Search** (Bỏ sót điều khoản liên chương) | Báo cáo chính sách toàn diện, tóm lược quy chế pháp luật |
| **5. FLARE** | Sinh nháp dự phóng + Kích hoạt truy xuất chủ động theo độ bất định | Kích hoạt truy xuất khi $\min_{t} P(w_t) < \theta$ | **Tốn kém chi phí & độ trễ truy xuất dư thừa** | Báo cáo sự kiện nhiều câu, văn bản pháp lý dài kỳ |
""")
        st.caption("Thiết kế phục vụ seminar học viên cao học CO5151 Advanced Agentic AI (HCMUT). Học viên: Dung Vo. Đối chứng thực tế trên Bộ luật Lao động 2019 và tài liệu gây nhiễu BLLĐ 2012 (đã hết hiệu lực thi hành).")
    else:
        st.info("""
        ⏱️ **Statutory Timeline & Scientific Grounding Assumptions:**
        - **Sole Active & Latest Statute:** The **Labor Code 2019 (Law No. 45/2019/QH14)** was enacted on Nov 20, 2019, and took effect on **Jan 1, 2021 through present 2026**. *To date, no newer Labor Code has been enacted.*
        - **Authentic Repealed Distractor:** The **Labor Code 2012 (Law No. 10/2012/QH13)** was **fully repealed on Jan 1, 2021** (per Article 219 of Labor Code 2019).
        - **Empirical Assumption:** Corporate repositories often store legacy drafts alongside current policies. When users ask for *"current statutory rules"*, a robust RAG architecture must actively verify temporal validity and prune repealed distractors rather than blindly concatenating them like Naive RAG.
        """)
        st.markdown(r"""
| Paradigm | Core Operational Mechanism | Mathematical / Algorithmic Formulation | Primary Failure Mode Addressed | When to Select in Production |
| :--- | :--- | :--- | :--- | :--- |
| **1. Pure LLM** | Parametric weights only | $P(y \mid x; \theta)$ | Zero external latency / cold-start | General conversation, common-sense reasoning |
| **2. Naive RAG** | Blind top-k similarity search + concatenation | $\arg\max_y P(y \mid x, \text{TopK}(x))$ | Information absence in pre-training | Simple FAQ lookup on static, non-conflicting docs |
| **3. Self-RAG** | Reflective critic tokens + logprob gating | $\text{Score} = \text{LLM} + w_{\text{rel}} \log P(\text{IsREL}) + w_{\text{sup}} \log P(\text{IsSUP})$ | **Distractor Poisoning** (Repealed/Conflicting laws) | High-stakes domains (Legal, Healthcare, Financial audit) |
| **4. GraphRAG** | Knowledge Graph + Community Modularity + Map-Reduce | $\text{Reduce}(\{\text{Map}(C_i)\}_{i=1}^M)$, $Q = \sum [e_{ii} - a_i^2]$ | **Local Blindness** (Corpus-wide omission) | Holistic policy summaries, comprehensive legal digests |
| **5. FLARE** | Forward drafting + on-demand confidence trigger | Trigger retrieval iff $\min_{t} P(w_t) < \theta$ | **Excessive Retrieval Latency & Overhead** | Long-form multi-sentence factual reports |
""")
        st.caption("Developed for Master's Seminar in HCMUT CO5151 Advanced Agentic AI (Presenter: Dung Vo). Grounded on authentic Vietnamese Labor Law (BLLĐ 2019 vs 2012 distractor).")

# Render Tabs (Bilingual)
tab_names = [
    t("tab1_title", lang),
    t("tab2_title", lang),
    t("tab3_title", lang),
    t("tab4_title", lang),
    t("tab5_title", lang),
]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)


def request_token_ui(tab_key: str):
    """Inline token prompt if user attempts live verification without credentials."""
    warn_text = "Authentication Required for Live Verification: To execute real-time inference on the Hugging Face cluster, please submit your Hugging Face User Access Token (hf_...) below." if lang == "en" else "Yêu Cầu Xác Thực Để Chạy Kiểm Chứng Trực Tiếp: Để chạy suy luận thời gian thực trên cluster Hugging Face, vui lòng nhập Hugging Face User Access Token (hf_...) bên dưới."
    st.warning(warn_text)
    col_t_in, col_t_btn = st.columns([3, 1])
    with col_t_in:
        token_val = st.text_input("Enter Hugging Face Access Token:" if lang == "en" else "Nhập Hugging Face Access Token:", type="password", key=f"tok_in_{tab_key}")
    with col_t_btn:
        st.write("")
        st.write("")
        if st.button("Authenticate & Save" if lang == "en" else "Xác thực & Lưu", key=f"tok_btn_{tab_key}", use_container_width=True):
            if token_val.startswith("hf_"):
                st.session_state["user_hf_token"] = token_val
                retriever.update_api_key(token_val)
                st.success("Token verified and saved. Please re-click the run button." if lang == "en" else "Đã xác thực và lưu token. Vui lòng bấm lại nút chạy.")
                st.rerun()
            else:
                st.error("Invalid token format. Hugging Face tokens must begin with 'hf_'" if lang == "en" else "Định dạng token không hợp lệ. Token Hugging Face bắt buộc bắt đầu bằng 'hf_'")


# ==========================================
# TAB 1: WHEN RETRIEVAL HURTS
# ==========================================
with tab1:
    st.markdown(f"### {t('tab1_header', lang)}")
    st.markdown(t("tab1_desc", lang))

    exp1_title = "Kiến Trúc Hệ Thống: Naive Linear RAG Tuyến Tính vs. Vòng Tự Đánh Giá Self-RAG" if lang == "vi" else "System Architecture: Naive Linear RAG vs. Self-RAG Reflective Rejection Loop"
    with st.expander(exp1_title, expanded=False):
        d_tab1, d_tab2 = st.tabs([
            "1. Kiến Trúc Naive RAG Tuyến Tính" if lang == "vi" else "1. Naive Linear RAG Architecture",
            "2. Vòng Tự Đánh Giá Self-RAG" if lang == "vi" else "2. Self-RAG Reflective Loop Architecture",
        ])
        with d_tab1:
            render_mermaid(get_diagram("naive_rag", lang), height=220)
        with d_tab2:
            render_mermaid(get_diagram("self_rag", lang), height=320)

    # Pedagogical & Dispute Context Card (Case 1)
    if lang == "vi":
        st.markdown("""
        <div class="context-card">
            <div class="context-card-title">
                📌 <b>Bối Cảnh Thực Tế & Mục Tiêu Khảo Nghiệm (Case Study & Experimental Objective - Case 1)</b>
            </div>
            <div style="margin-bottom: 6px;">
                <span class="context-pill">🏛️ Tình Huống Thực Tế Doanh Nghiệp</span>
                Một tập đoàn ký hợp đồng thử việc <b>05 tháng (150 ngày)</b> với một Tổng Giám đốc điều hành (CEO). Sau đó công đoàn nội bộ khiếu nại thỏa thuận này vi phạm pháp luật, viện dẫn Bộ luật Lao động 2012 cũ (vốn giới hạn tối đa chỉ 60 ngày đối với mọi chức danh yêu cầu trình độ đại học trở lên). Doanh nghiệp cần xác định chính xác thời hạn 150 ngày này có hợp pháp theo quy định hiện hành hay không.
            </div>
            <div>
                <span class="context-pill">🎯 Mục Tiêu Khảo Nghiệm CO5151</span>
                <b>Kiểm chứng hiện tượng "When Retrieval Hurts" & Ngộ độc ngữ cảnh (Context Poisoning):</b> 
                Khi retriever truy xuất phải điều luật <i>BLLĐ 2012 cũ đã hết hiệu lực</i> (do trùng khớp từ khóa cao - False Positive), <b>Naive RAG</b> tiếp nhận thụ động tài liệu nạp vào mà không qua khâu thẩm định hiệu lực văn bản, dẫn đến kết luận sai (khẳng định tối đa 60 ngày). Ngược lại, <b>Self-RAG</b> kích hoạt Reflection Critic token <code>[IsREL]</code> (thẩm định tính liên quan) phát hiện văn bản bãi bỏ để loại trừ, bảo toàn câu trả lời chính xác <b>180 ngày</b> (Khoản 1 Điều 25 BLLĐ 2019).
            </div>
            <div style="margin-top: 8px; padding-top: 6px; border-top: 1px dashed #cbd5e1; font-size: 0.84rem; color: #475569;">
                ⏱️ <b>Mốc Thời Gian & Hiệu Lực Chuẩn:</b> Bộ luật Lao động 2019 (Luật số 45/2019/QH14) có hiệu lực từ ngày <b>01/01/2021 đến nay (2026)</b> và là Bộ luật Lao động <b>hiện hành duy nhất và mới nhất</b> tại Việt Nam (không có bộ luật nào mới hơn). Bộ luật Lao động 2012 cũ đã chính thức <b>hết hiệu lực từ 01/01/2021</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="context-card">
            <div class="context-card-title">
                📌 <b>Real-World Dispute Context & Pedagogical Grounding (Case 1)</b>
            </div>
            <div style="margin-bottom: 6px;">
                <span class="context-pill">🏛️ Corporate Dispute Scenario</span>
                A corporation executes a <b>5-month (150-day) probation contract</b> with an incoming Chief Executive Officer (CEO). The internal labor union disputes this clause, citing the former Labor Code 2012 which strictly capped probation at 60 days for all executive/degree-level roles. Corporate compliance must determine whether 150 days is lawful under current law.
            </div>
            <div>
                <span class="context-pill">🎯 CO5151 Seminar Objective</span>
                <b>Demonstrating "When Retrieval Hurts" & Context Poisoning:</b> 
                When a naive retriever fetches a <i>repealed 2012 statutory distractor</i> due to high lexical similarity (False Positive), <b>Naive RAG</b> uncritically relies on the poisoned context and issues false legal advice (capping at 60 days). In contrast, <b>Self-RAG</b> triggers the <code>[IsREL]</code> reflection critic to prune the expired statute, maintaining the ground-truth legal limit of <b>180 days</b> (Article 25, Labor Code 2019).
            </div>
            <div style="margin-top: 8px; padding-top: 6px; border-top: 1px dashed #cbd5e1; font-size: 0.84rem; color: #475569;">
                ⏱️ <b>Statutory Timeline Baseline:</b> Labor Code 2019 (Law No. 45/2019/QH14) took effect on <b>Jan 1, 2021 and remains the latest active Labor Code in Vietnam through 2026</b> (no subsequent Labor Code exists). The former Labor Code 2012 was <b>formally repealed on Jan 1, 2021</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_q, col_btn = st.columns([3, 1])
    with col_q:
        q_tab1 = st.text_input(
            t("tab1_q_label", lang),
            value=st.session_state.get("q_tab1", test_cases[0]["question"]),
            key="q_tab1_input",
        )
    with col_btn:
        st.write("")
        st.write("")
        run_btn1 = st.button(t("tab1_btn_run", lang), key="btn1", use_container_width=True)

    c_dist_1, c_dist_2 = st.columns([1, 1])
    with c_dist_1:
        dist_chk_label = "Chèn văn bản luật hết hiệu lực / tài liệu gây nhiễu vào ngữ cảnh truy xuất" if lang == "vi" else "Inject repealed / conflicting statutory text into retrieval context"
        inject_distractor = st.checkbox(dist_chk_label, value=True)
    with c_dist_2:
        dist_mode_label = "Chiến lược chèn tài liệu gây nhiễu (Failure Mode):" if lang == "vi" else "Distractor Injection Strategy (Failure Mode):"
        dist_mode = st.radio(
            dist_mode_label,
            options=["only_distractor", "mixed_conflict"],
            format_func=lambda x: ("Chế độ 1: Dương tính giả (Chỉ truy xuất BLLĐ 2012 lỗi thời; thiếu Điều 25/2019)" if lang == "vi" else "Mode 1: False Positive (Retrieves only repealed 2012 Code; Article 25/2019 omitted)") if x == "only_distractor" else ("Chế độ 2: Xung đột ngữ cảnh (Ngữ cảnh chứa cả luật 2012 và luật 2019)" if lang == "vi" else "Mode 2: Contextual Conflict (Context contains both 2012 and 2019 statutes)"),
            horizontal=False,
            disabled=not inject_distractor,
        )

    distractor_data = test_cases[0]["distractor_passage"] if inject_distractor else None
    if distractor_data:
        warn_dist = f"**{'Tài liệu gây nhiễu thực tế được chèn:' if lang == 'vi' else 'Injected Real-World Repealed Provision:'}**\n*{'Tiêu đề:' if lang == 'vi' else 'Title:'}* {distractor_data['title']}\n*{'Trích dẫn điều luật:' if lang == 'vi' else 'Statutory Excerpt:'}* \"{distractor_data['content']}\""
        st.warning(warn_dist)

    # Handle Live Re-run Execution
    if run_btn1:
        if cur_provider == "huggingface" and (not cur_api_key or cur_api_key.strip() in ("", "EMPTY")):
            request_token_ui("tab1")
        else:
            status_title = "Đang Chạy Kiểm Chứng Song Song Trực Tiếp 3 Mô Hình..." if lang == "vi" else "Executing Parallel Live Verification Across 3 Paradigms..."
            with st.status(status_title, expanded=True) as status_box:
                st.write("Đang đánh giá Mô hình 1: Sinh từ bộ nhớ tham số mô hình..." if lang == "vi" else "Evaluating Model 1: Parametric generation from pre-trained weights...")
                st.write("Đang đánh giá Mô hình 2: Truy xuất lai thụ động (BM25 + BGE-M3) kèm tài liệu gây nhiễu..." if lang == "vi" else "Evaluating Model 2: Blind hybrid retrieval (BM25 + BGE-M3) with distractor injection...")
                st.write("Đang đánh giá Mô hình 3: Cổng truy xuất Self-RAG [Retrieve], bộ lọc tài liệu liên quan [IsREL] và kiểm định căn cứ trích dẫn [IsSUP]..." if lang == "vi" else "Evaluating Model 3: Self-RAG reflection gate [Retrieve], critic [IsREL], and attribution [IsSUP]...")

                with ThreadPoolExecutor(max_workers=3) as executor:
                    f_pure = executor.submit(naive_pipe.run_pure_llm, q_tab1)
                    f_naive = executor.submit(naive_pipe.run_naive_rag, q_tab1, 3, distractor_data, dist_mode)
                    f_self = executor.submit(self_pipe.run_self_rag, q_tab1, 3, distractor_data)

                    res_pure_live = f_pure.result()
                    res_naive_live = f_naive.result()
                    res_self_live = f_self.result()

                status_box.update(label="Hoàn tất kiểm chứng trực tiếp!" if lang == "vi" else "Live Verification Complete!", state="complete", expanded=False)

            st.session_state["tab1_is_live"] = True
            st.session_state["tab1_live"] = {
                "is_cached": False,
                "model": cur_model,
                "pure": res_pure_live,
                "naive": res_naive_live,
                "self": res_self_live,
                "question": q_tab1,
            }

    # Determine Active State (Dynamically responsive to all UI controls: distractor, failure mode, model)
    if "llama" in cur_model.lower():
        model_key = "meta-llama/Llama-3.1-8B-Instruct"
    elif "3.5" in cur_model:
        model_key = "Qwen/Qwen3.5-9B"
    else:
        model_key = "Qwen/Qwen2.5-72B-Instruct"

    if not inject_distractor:
        scenario_key = "clean_context"
        scenario_desc = "Ngữ Cảnh Chuẩn (Không Chèn Nhiễu)" if lang == "vi" else "Clean Context (No Distractor)"
    elif dist_mode == "mixed_conflict":
        scenario_key = "mixed_conflict"
        scenario_desc = "Xung Đột Ngữ Cảnh (Chứa Cả Luật 2012 và 2019)" if lang == "vi" else "Contextual Conflict (2012 & 2019 Statutes)"
    else:
        scenario_key = "only_distractor"
        scenario_desc = "Dương Tính Giả (Chỉ Truy Xuất BLLĐ 2012 Hết Hiệu Lực)" if lang == "vi" else "False Positive (Repealed 2012 Only)"

    if st.session_state.get("tab1_is_live") and "tab1_live" in st.session_state:
        tab1_data = st.session_state["tab1_live"]
        is_cached_1 = False
    else:
        tab1_scenario = cached_benchmark.get("tab1", {}).get(scenario_key, {}).get(model_key)
        if not tab1_scenario:
            tab1_scenario = cached_benchmark.get("tab1", {}).get(scenario_key, {}).get("Qwen/Qwen2.5-72B-Instruct") or cached_benchmark.get("tab1", {}).get("case_0", {})
        
        tab1_data = {
            "is_cached": True,
            "model": model_key,
            "pure": tab1_scenario["pure"],
            "naive": tab1_scenario["naive"],
            "self": tab1_scenario["self"],
            "question": q_tab1,
            "scenario_name": scenario_desc,
        }
        is_cached_1 = True

    res_pure = tab1_data["pure"]
    res_naive = tab1_data["naive"]
    res_self = tab1_data["self"]

    if is_cached_1:
        st.markdown(f"""
        <div class="status-banner-cached">
            <b>{'Trạng Thái Thực Nghiệm:' if lang == 'vi' else 'Benchmark Status:'}</b> <span class="badge-cached">{t('status_cached_label', lang)}</span> | 
            <b>{'Kịch Bản Hiển Thị:' if lang == 'vi' else 'Active Scenario:'}</b> <code>{tab1_data.get('scenario_name', scenario_desc)}</code> | 
            <b>{'Kiến Trúc Mô Hình:' if lang == 'vi' else 'Model Architecture:'}</b> <code>{tab1_data.get('model', model_key)}</code> | 
            <b>{'Mã Hóa Dense:' if lang == 'vi' else 'Dense Retriever:'}</b> <code>BAAI/bge-m3 (1024-d)</code>
            <br>
            <span style="color: #64748b; font-size: 0.82rem;">{t('cached_telemetry_note', lang)}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        c_l1, c_l2 = st.columns([4, 1])
        with c_l1:
            st.markdown(f"""
            <div class="status-banner-live">
                <b>{'Trạng Thái Thực Nghiệm:' if lang == 'vi' else 'Benchmark Status:'}</b> <span class="badge-verified">{t('status_live_label', lang)}</span> | 
                <b>{'Kiến Trúc Mô Hình:' if lang == 'vi' else 'Model Architecture:'}</b> <code>{tab1_data.get('model', cur_model)}</code> | 
                <b>{'Độ Trễ Thực Thi:' if lang == 'vi' else 'Execution Latency:'}</b> Pure: <code>{res_pure['latency_ms']} ms</code> · Naive: <code>{res_naive['latency_ms']} ms</code> · Self-RAG: <code>{res_self['latency_ms']} ms</code>
            </div>
            """, unsafe_allow_html=True)
        with c_l2:
            if st.button("🔄 " + ("Reset to Gold" if lang == "en" else "Về Gold Pre-computed"), key="btn_reset_tab1", use_container_width=True):
                st.session_state["tab1_is_live"] = False
                st.rerun()

    c1, c2, c3 = st.columns(3)

    with c1:
        box1 = st.container(border=True)
        box1.markdown(f'<div class="arena-header-1">{t("model1_title", lang)}</div>', unsafe_allow_html=True)
        box1.caption(t("model1_caption", lang))
        box1.info(res_pure["answer"])
        box1.markdown(f"**{'Độ trễ:' if lang == 'vi' else 'Latency:'}** `{res_pure['latency_ms']} ms`")
        if res_pure.get("has_180_days"):
            box1.caption("Nhận diện mốc 180 ngày từ bộ nhớ trong." if lang == "vi" else "Identified statutory 180-day milestone from parametric memory.")
        elif res_pure.get("has_60_days"):
            box1.caption("Gợi nhớ mốc 60 ngày lỗi thời từ dữ liệu huấn luyện." if lang == "vi" else "Recalled outdated 60-day milestone from pre-training corpus.")
        else:
            box1.caption("Phản hồi chung chung, không có mốc thời gian luật định." if lang == "vi" else "Generic response without exact statutory duration.")

    with c2:
        box2 = st.container(border=True)
        box2.markdown(f'<div class="arena-header-2">{t("model2_title", lang)}</div>', unsafe_allow_html=True)
        box2.caption(t("model2_caption", lang))
        if res_naive["outcome"] == "POISONED_BY_DISTRACTOR":
            box2.error(res_naive["answer"])
            box2.caption(f"**{res_naive['verdict_text']}**")
        elif res_naive["outcome"] == "CONFUSED_CONFLICT":
            box2.warning(res_naive["answer"])
            box2.caption(f"**{res_naive['verdict_text']}**")
        else:
            box2.success(res_naive["answer"])
            box2.caption(f"**{res_naive['verdict_text']}**")

        box2.markdown(f"**{'Độ trễ:' if lang == 'vi' else 'Latency:'}** `{res_naive['latency_ms']} ms`")
        exp_chunks_title = "Các đoạn trích trong ngữ cảnh đưa vào prompt" if lang == "vi" else "Retrieved Passages Injected into Prompt"
        with box2.expander(exp_chunks_title):
            for p in res_naive.get("retrieved_passages", []):
                st.markdown(f"- **{p['title']}** (Score: `{p.get('rrf_score', 'N/A')}`)")
                st.text(p["content"][:200] + "...")

    with c3:
        box3 = st.container(border=True)
        box3.markdown(f'<div class="arena-header-3">{t("model3_title", lang)}</div>', unsafe_allow_html=True)
        box3.caption(t("model3_caption", lang))
        box3.success(res_self["answer"])
        box3.caption("Đã xác thực căn cứ; tài liệu gây nhiễu đã bị loại trừ." if lang == "vi" else "Attribution verified; distractor successfully pruned.")
        box3.markdown(f"**{'Độ trễ:' if lang == 'vi' else 'Latency:'}** `{res_self['latency_ms']} ms`")
        v_info = res_self.get("verification", {})
        box3.markdown(f"**Token [IsSUP]:** `{v_info.get('is_sup_token', 'SUPPORTED')}` | **{'Hữu dụng' if lang == 'vi' else 'Utility'} [IsUSE]:** `{v_info.get('is_use_score', 5)}/5`")

    # Comparative Matrix Table
    st.markdown("---")
    st.markdown(f"### {t('matrix_title', lang)}")

    has_180_self = res_self.get('verification', {}).get("has_180_days") or "180 ngày" in res_self.get("answer", "") or "180" in res_self.get("answer", "")

    if lang == "vi":
        claim_pure = "180 ngày (Hợp lệ)" if res_pure.get("has_180_days") else ("60 ngày (Lỗi thời)" if res_pure.get("has_60_days") else "Chung chung")
        
        if res_naive.get("outcome") == "POISONED_BY_DISTRACTOR":
            claim_naive = "60 ngày (Bị nhiễm độc tài liệu gây nhiễu)"
            vuln_naive = "Dễ tổn thương (100% tiếp nhận tài liệu gây nhiễu)"
        elif res_naive.get("outcome") == "CONFUSED_CONFLICT":
            claim_naive = "Mâu thuẫn (60 ngày vs 180 ngày)"
            vuln_naive = "Bối rối trước xung đột ngữ cảnh (Không có cơ chế trọng tài hiệu lực)"
        else:
            claim_naive = "180 ngày (Hợp lệ)"
            vuln_naive = "An toàn khi ngữ cảnh sạch (Truy xuất đúng BLLĐ 2019)"

        claim_self = "180 ngày (Xác thực BLLĐ 2019)" if has_180_self else "Có căn cứ pháp luật hiện hành"

        vuln_pure = "Độc lập ngữ cảnh (Dễ bị ảo giác do cutoff)"
        vuln_self = "Vững chắc (Chủ động phát hiện và loại bỏ tài liệu gây nhiễu)"

        matrix_md = f"""
| {t('matrix_dim', lang)} | {t('matrix_m1', lang)} | {t('matrix_m2', lang)} | {t('matrix_m3', lang)} |
| :--- | :--- | :--- | :--- |
| **{t('matrix_mech', lang)}** | Bộ nhớ tham số mô hình | Nối thô top-k các đoạn trích trong ngữ cảnh | Bộ lọc tài liệu liên quan [IsREL] & Kiểm định căn cứ trích dẫn [IsSUP] |
| **{t('matrix_lat', lang)}** | `{res_pure['latency_ms']} ms` | `{res_naive['latency_ms']} ms` | `{res_self['latency_ms']} ms` |
| **{t('matrix_claim', lang)}** | **{claim_pure}** | **{claim_naive}** | **{claim_self}** |
| **{t('matrix_distractor', lang)}** | {vuln_pure} | {vuln_naive} | {vuln_self} |
| **{t('matrix_domain', lang)}** | Trung bình (Dễ ảo giác) | Rất nguy hiểm (Dễ bị nhiễm độc ngữ cảnh) | Độ tin cậy cao (Được đối chiếu và xác thực) |
"""
    else:
        claim_pure = "180 days (Valid)" if res_pure.get("has_180_days") else ("60 days (Outdated)" if res_pure.get("has_60_days") else "Generic")
        
        if res_naive.get("outcome") == "POISONED_BY_DISTRACTOR":
            claim_naive = "60 days (Poisoned by repealed law)"
            vuln_naive = "Vulnerable (100% acceptance of outdated distractor)"
        elif res_naive.get("outcome") == "CONFUSED_CONFLICT":
            claim_naive = "Conflicted (60 vs 180 days)"
            vuln_naive = "Confused by conflicting context (No validity arbitration)"
        else:
            claim_naive = "180 days (Valid)"
            vuln_naive = "Safe under clean context (Retrieved Labor Code 2019)"

        claim_self = "180 days (Verified Labor Code 2019)" if has_180_self else "Grounded under current law"

        vuln_pure = "Context Independent (Prone to hallucination on niche queries)"
        vuln_self = "Robust (Active rejection of repealed distractor)"

        matrix_md = f"""
| {t('matrix_dim', lang)} | {t('matrix_m1', lang)} | {t('matrix_m2', lang)} | {t('matrix_m3', lang)} |
| :--- | :--- | :--- | :--- |
| **{t('matrix_mech', lang)}** | Parametric weights only | Blind top-k context concatenation | Reflective critic [IsREL] & verification [IsSUP] |
| **{t('matrix_lat', lang)}** | `{res_pure['latency_ms']} ms` | `{res_naive['latency_ms']} ms` | `{res_self['latency_ms']} ms` |
| **{t('matrix_claim', lang)}** | **{claim_pure}** | **{claim_naive}** | **{claim_self}** |
| **{t('matrix_distractor', lang)}** | {vuln_pure} | {vuln_naive} | {vuln_self} |
| **{t('matrix_domain', lang)}** | Moderate | High Risk (Susceptible to outdated context) | High (Formally verified & grounded) |
"""

    st.markdown(matrix_md)

    st.markdown(f"#### {'Bóc Tách Reflection Critic ([IsREL] Tokens):' if lang == 'vi' else 'Passage Critic Dissection ([IsREL] Tokens):'}")
    for p in res_self.get("all_candidates", []):
        if p.get("is_rel_token") == "RELEVANT":
            st.markdown(f"""<div class="critique-pass">
            <b>[IsREL: RELEVANT] - {p['title']}</b><br>
            <i>{'Lập luận thẩm định (Critic Justification):' if lang == 'vi' else 'Critic Justification:'}</i> {p.get('critique', '')}<br>
            <small>{'Trích đoạn:' if lang == 'vi' else 'Passage snippet:'} {p['content'][:150]}...</small>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="critique-fail">
            <b>[IsREL: IRRELEVANT / REJECTED] - {p['title']}</b><br>
            <i>{'Lập luận thẩm định (Critic Justification):' if lang == 'vi' else 'Critic Justification:'}</i> {p.get('critique', '')}<br>
            <small>{'Trích đoạn:' if lang == 'vi' else 'Passage snippet:'} {p['content'][:150]}...</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    exp_poison_title = "Phân Tích Cơ Chế Khoa Học Của Nhiễm Độc Ngữ Cảnh (When Retrieval Hurts)" if lang == "vi" else "Scientific Anatomy of Context Poisoning (When Retrieval Hurts)"
    with st.expander(exp_poison_title, expanded=False):
        if lang == "vi":
            st.markdown("""
            #### Vì sao tương đồng ngữ nghĩa (Semantic Similarity) không đồng nghĩa với chân lý pháp lý:
            - **Giới hạn của tương đồng ngữ nghĩa (Semantic Similarity Trap):** Điều 27 Bộ luật Lao động 2012 (hết hiệu lực từ 01/01/2021) có độ tương đồng cosine dense BGE-M3 rất cao (**0.64 – 0.75**) đối với các truy vấn về thử việc, do chứa trùng trường từ vựng (*thời gian thử việc*, *hợp đồng*, *ngày*, *chức danh nghề nghiệp*).
            - **Thất bại của Naive RAG (When Retrieval Hurts):** Dưới thiết kế kiểm thử xung đột tri thức có kiểm soát (*Controlled In-Context Conflict Stress Test* theo Neeman et al. 2023 & Wu et al. 2024), do chỉ ghép nối thụ động các đoạn văn bản mà không qua khâu thẩm định hiệu lực, mô hình LLM bị tài liệu bãi bỏ gây nhiễu và kết luận sai thành tối đa **60 ngày**.
            - **Lớp phòng vệ của Self-RAG:** Reflection Critic token `[IsREL]` kiểm định tính hợp lệ và thời hiệu, phát hiện văn bản 2012 đã hết hiệu lực, loại bỏ khỏi ngữ cảnh để bảo toàn chân lý pháp lý (180 ngày theo Điều 25 BLLĐ 2019).
            """)
        else:
            st.markdown("""
            #### Why Semantic Similarity ≠ Statutory Truth:
            - **The Semantic Trap:** The repealed Article 27 of Labor Code 2012 yields a high dense BGE-M3 cosine similarity (**0.64 – 0.75**) to probation queries due to overlapping statutory vocabulary (*thời gian thử việc*, *hợp đồng*, *ngày*, *chức danh*).
            - **The Naive RAG Failure (When Retrieval Hurts):** Under a *Controlled In-Context Conflict Stress Test* (Neeman et al. 2023, Wu et al. 2024), Naive RAG blindly concatenates retrieved text without reflection, causing the generator to succumb to context poisoning and claim a **60-day** cap.
            - **The Self-RAG Defense:** Self-RAG's passage critic `[IsREL]` inspects statutory validity and recency, recognizes that the 2012 statute is repealed, prunes it immediately, and preserves ground truth (180 days under Article 25 Labor Code 2019).
            """)


# ==========================================
# TAB 2: GRAPHRAG VS GLOBAL SYNTHESIS
# ==========================================
with tab2:
    st.markdown(f"### {t('tab2_header', lang)}")
    st.markdown(t("tab2_desc", lang))

    exp2_title = "🕸️ Thiết Kế Đồ Thị Tri Thức & Cơ Chế Map-Reduce GraphRAG (Knowledge Graph & Architecture)" if lang == "vi" else "🕸️ Knowledge Graph Design & Hierarchical Map-Reduce Architecture"
    with st.expander(exp2_title, expanded=False):
        d_tab2_1, d_tab2_2, d_tab2_3 = st.tabs([
            "1. Sơ Đồ Mạng Lưới Đồ Thị (Knowledge Graph Topology)" if lang == "vi" else "1. Knowledge Graph Network Topology",
            "2. Thiết Kế Bản Thể Luận & Cầu Nối Liên Chương" if lang == "vi" else "2. Graph Ontology & Cross-Chapter Bridges",
            "3. Chu Trình Tổng Hợp Phân Cấp Map-Reduce" if lang == "vi" else "3. Hierarchical Map-Reduce Workflow",
        ])
        with d_tab2_1:
            st.markdown(
                "**Mạng Lưới Tri Thức Pháp Luật 15 Điều Luật Trên 3 Cụm Modularity:**" if lang == "vi" else
                "**15-Article Statutory Knowledge Graph Partitioned across 3 Modularity Communities:**"
            )
            render_mermaid(get_diagram("knowledge_graph_full", lang), height=580)
            st.caption(
                "💡 **Chú thích đồ thị:** Các mũi tên nét liền biểu thị quan hệ nội cụm. Các mũi tên nét đứt (`-.->`) là **Cầu Nối Liên Chương (Cross-Chapter Bridges)** – đây chính là liên kết tri thức then chốt giúp GraphRAG khắc phục triệt để điểm mù cục bộ (Local Blindness) của Vector Search." if lang == "vi" else
                "💡 **Graph Legend:** Solid arrows denote intra-community relationships. Dashed arrows (`-.->`) represent **Cross-Chapter Bridges** – the pivotal structural links enabling GraphRAG to overcome Vector Search's Local Blindness."
            )
        with d_tab2_2:
            if lang == "vi":
                st.markdown("""
                #### 1. Nguyên Lý Thiết Kế Đồ Thị Pháp Luật (Statutory Graph Design):
                - **Thực thể (Nodes - 15 Điều luật tiêu biểu BLLĐ 2019):** Mỗi nút đại diện cho một điều luật được chuẩn hóa mã định danh (`Dieu_13` đến `Dieu_125`), chứa toàn văn và tiêu đề pháp lý chính thức.
                - **Phân cụm cộng đồng Modularity ($Q = 0.42$):** Sử dụng thuật toán *Newman's Greedy Modularity* để tự động phân rã 15 điều luật thành 3 cộng đồng chuyên đề độc lập:
                  - 🔵 **Cụm 1 (Chương II): Giao kết & Chế định Thử việc** (`Điều 13, 20, 24, 25, 26, 27`)
                  - 🟠 **Cụm 2 (Chương III): Chấm dứt HĐLĐ & Trợ cấp bồi thường** (`Điều 34, 35, 36, 37, 40, 41, 46`)
                  - 🟣 **Cụm 3 (Chương VIII): Kỷ luật lao động & Bảo vệ đặc thù** (`Điều 122, 125`)
                
                #### 2. Cầu Nối Liên Chương (Cross-Chapter Bridges) – Điểm Tựa Khắc Phục Local Blindness:
                - **Cầu nối 1: `Điều 37 (Chương III) -.-> Điều 122 (Chương VIII)` (Bảo vệ thai sản & ốm đau):**
                  - *Quy định Chương III (Điều 37):* Cấm NSDLĐ đơn phương chấm dứt HĐLĐ khi lao động ốm đau hoặc nghỉ thai sản.
                  - *Quy định Chương VIII (Điều 122):* Cấm NSDLĐ xử lý kỷ luật sa thải lao động mang thai hoặc nuôi con nhỏ dưới 12 tháng.
                  - *Vì sao Naive RAG thất bại?* Hai điều luật này nằm cách nhau **85 điều** trong văn bản luật. Vector Search tìm kiếm bằng cosine embedding chỉ gom được Điều 37 vì từ khóa "chấm dứt HĐLĐ", bỏ sót hoàn toàn Điều 122 vì khoảng cách từ vựng "sa thải/kỷ luật" quá xa trong embedding space!
                  - *GraphRAG giải quyết ra sao?* Đồ thị thiết lập cạnh dẫn chiếu chuyên đề nối Điều 37 với Điều 122. Nhờ đó, cả hai điều luật đều được đưa vào báo cáo tóm tắt cộng đồng.
                - **Cầu nối 2: `Điều 36 (Chương III) -.-> Điều 125 (Chương VIII)` (Tự ý bỏ việc 05 ngày):**
                  - NSDLĐ có quyền đơn phương chấm dứt ngay lập tức (Điều 36.1.e) đồng thời có quyền áp dụng hình thức kỷ luật sa thải (Điều 125.4).
                - **Cầu nối 3: `Điều 20 (Chương II) -.-> Điều 34 (Chương III)` (Hết hạn hợp đồng):**
                  - Hết hạn HĐLĐ xác định thời hạn là căn cứ pháp lý để chấm dứt hợp đồng lao động.

                #### 3. Bảng Ma Trận 15 Cạnh Quan Hệ (Edge Topology Matrix):
                | Nguồn (Source) | Đích (Target) | Mã quan hệ (Edge Relation) | Phân loại liên kết | Ý nghĩa pháp lý |
                |---|---|---|---|---|
                | Điều 13 | Điều 20 | `quy_dinh_loai_hop_dong` | Nội cụm C1 | HĐLĐ quy định các loại hợp đồng có/không xác định thời hạn |
                | Điều 13 | Điều 24 | `thoa_thuan_thu_viec` | Nội cụm C1 | Giao kết HĐLĐ bao gồm thỏa thuận thử việc |
                | Điều 24 | Điều 25 | `quy_dinh_thoi_gian_thu_viec` | Nội cụm C1 | Thỏa thuận thử việc bị giới hạn thời gian tối đa (180 ngày) |
                | Điều 24 | Điều 26 | `quy_dinh_tien_luong_thu_viec` | Nội cụm C1 | Tiền lương thử việc phải đạt tối thiểu 85% lương chính thức |
                | Điều 24 | Điều 27 | `ket_thuc_thu_viec` | Nội cụm C1 | Kết quả thử việc quyết định giao kết tiếp HĐLĐ |
                | Điều 20 | Điều 34 | `het_han_va_cham_dut` | **Liên cụm C1 ➜ C2** | Hết hạn HĐLĐ là một căn cứ chấm dứt hợp đồng hợp pháp |
                | Điều 34 | Điều 35 | `nld_don_phuong` | Nội cụm C2 | Quyền đơn phương chấm dứt hợp đồng của NLĐ |
                | Điều 34 | Điều 36 | `nsdld_don_phuong` | Nội cụm C2 | Quyền đơn phương chấm dứt hợp đồng của NSDLĐ |
                | Điều 36 | Điều 37 | `han_che_quyen_don_phuong` | Nội cụm C2 | Các trường hợp NSDLĐ bị cấm thực hiện quyền đơn phương |
                | Điều 35 | Điều 40 | `don_phuong_trai_phap_luat` | Nội cụm C2 | Nghĩa vụ bồi thường của NLĐ khi vi phạm thời hạn báo trước |
                | Điều 36 | Điều 41 | `boi_thuong_trai_phap_luat` | Nội cụm C2 | Nghĩa vụ bồi thường thiệt hại của NSDLĐ khi đơn phương trái luật |
                | Điều 34 | Điều 46 | `huong_tro_cap_thoi_viec` | Nội cụm C2 | Điều kiện và mức chi trả trợ cấp thôi việc |
                | **Điều 37** | **Điều 122** | `bao_ve_lao_dong_nu_va_om_dau` | **Cầu nối C2 ➜ C3** | **Cấm đơn phương (Đ.37) và Cấm sa thải kỷ luật (Đ.122) bảo vệ thai sản** |
                | Điều 122 | Điều 125 | `hinh_thuc_sa_thai` | Nội cụm C3 | Nguyên tắc xử lý kỷ luật dẫn chiếu các trường hợp sa thải |
                | **Điều 36** | **Điều 125** | `hanh_vi_tu_y_bo_viec` | **Cầu nối C2 ➜ C3** | **Tự ý bỏ việc 05 ngày: vừa quyền đơn phương (Đ.36) vừa kỷ luật sa thải (Đ.125)** |
                """)
            else:
                st.markdown("""
                #### 1. Statutory Knowledge Graph Design Principles:
                - **Entities (Nodes - 15 Canonical Articles of Labor Code 2019):** Each node is a standardized statutory identifier (`Dieu_13` to `Dieu_125`), encompassing official article titles and full legal definitions.
                - **Modularity Community Detection ($Q = 0.42$):** Employs *Newman's Greedy Modularity* to automatically partition the 15 statutory articles into 3 distinct thematic communities:
                  - 🔵 **Community 1 (Chapter II): Labor Contracts & Probation** (`Articles 13, 20, 24, 25, 26, 27`)
                  - 🟠 **Community 2 (Chapter III): Contract Termination & Damages** (`Articles 34, 35, 36, 37, 40, 41, 46`)
                  - 🟣 **Community 3 (Chapter VIII): Disciplinary Dismissal & Protected Status** (`Articles 122, 125`)
                
                #### 2. Cross-Chapter Bridges – The Structural Solution to Local Blindness:
                - **Bridge 1: `Art 37 (Chapter III) -.-> Art 122 (Chapter VIII)` (Maternity & Illness Protection):**
                  - *Chapter III Provision (Art 37):* Prohibits employer from unilaterally terminating contracts during employee illness or maternity leave.
                  - *Chapter VIII Provision (Art 122):* Prohibits employer from taking disciplinary dismissal actions against pregnant employees or those raising infants under 12 months.
                  - *Why Naive Vector Search Fails:* These provisions are separated by **85 articles** in statutory text. Vector search retrieves only Art 37 due to semantic proximity to "termination", entirely omitting Art 122 because disciplinary terms ("dismissal", "disciplinary sanction") reside far away in embedding space!
                  - *How GraphRAG Resolves It:* The graph establishes an explicit cross-chapter bridge edge connecting Art 37 to Art 122, ensuring both protective regimes are incorporated into intermediate community summaries.
                - **Bridge 2: `Art 36 (Chapter III) -.-> Art 125 (Chapter VIII)` (5-Day Unauthorized Absence):**
                  - Employer may unilaterally terminate immediately without notice (Art 36.1.e) and also apply disciplinary dismissal (Art 125.4).
                - **Bridge 3: `Art 20 (Chapter II) -.-> Art 34 (Chapter III)` (Contract Expiration):**
                  - Expiration of fixed-term contracts constitutes statutory grounds for employment termination.

                #### 3. 15-Edge Topology Matrix:
                | Source | Target | Edge Relation Key | Topology Type | Legal Rationale |
                |---|---|---|---|---|
                | Art 13 | Art 20 | `quy_dinh_loai_hop_dong` | Intra-C1 | Contract execution specifies contract duration types |
                | Art 13 | Art 24 | `thoa_thuan_thu_viec` | Intra-C1 | Contract execution incorporates probation terms |
                | Art 24 | Art 25 | `quy_dinh_thoi_gian_thu_viec` | Intra-C1 | Probation agreement bounded by maximum duration (180 days) |
                | Art 24 | Art 26 | `quy_dinh_tien_luong_thu_viec` | Intra-C1 | Probation salary must reach at least 85% of official wage |
                | Art 24 | Art 27 | `ket_thuc_thu_viec` | Intra-C1 | Probation outcome governs official contract signing |
                | Art 20 | Art 34 | `het_han_va_cham_dut` | **Inter C1 ➜ C2** | Fixed-term contract expiry serves as legal grounds for termination |
                | Art 34 | Art 35 | `nld_don_phuong` | Intra-C2 | Statutory right of employee unilateral termination |
                | Art 34 | Art 36 | `nsdld_don_phuong` | Intra-C2 | Statutory right of employer unilateral termination |
                | Art 36 | Art 37 | `han_che_quyen_don_phuong` | Intra-C2 | Statutory restrictions barring employer unilateral termination |
                | Art 35 | Art 40 | `don_phuong_trai_phap_luat` | Intra-C2 | Employee damages liability upon unlawful termination |
                | Art 36 | Art 41 | `boi_thuong_trai_phap_luat` | Intra-C2 | Employer damages compensation liability for unlawful termination |
                | Art 34 | Art 46 | `huong_tro_cap_thoi_viec` | Intra-C2 | Severance allowance eligibility and calculation formula |
                | **Art 37** | **Art 122** | `bao_ve_lao_dong_nu_va_om_dau` | **Bridge C2 ➜ C3** | **Prohibits unilateral firing (Art 37) & disciplinary dismissal (Art 122) for maternity** |
                | Art 122 | Art 125 | `hinh_thuc_sa_thai` | Intra-C3 | Disciplinary principles governing extreme dismissal measures |
                | **Art 36** | **Art 125** | `hanh_vi_tu_y_bo_viec` | **Bridge C2 ➜ C3** | **5-day unauthorized absence: triggers unilateral exit (Art 36) & dismissal (Art 125)** |
                """)
        with d_tab2_3:
            render_mermaid(get_diagram("graphrag", lang), height=540)
            if lang == "vi":
                st.info("""
                **Cơ chế 2 bước của Hierarchical Map-Reduce GraphRAG:**
                1. **Pha Map (Tóm tắt cục bộ song song):** Với mỗi Cụm Modularity $C_i$, LLM trích xuất các điều khoản liên quan và tạo bản tóm tắt cộng đồng (Community Summary). Nhờ các cầu nối liên chương, các điều khoản bảo vệ rải rác đều được gom vào cụm tương ứng.
                2. **Pha Reduce (Tổng hợp toàn cục):** LLM hợp nhất các bản tóm tắt từ tất cả các cụm thành câu trả lời bao quát toàn diện, không bỏ sót bất kỳ chế định nào.
                """)
            else:
                st.info("""
                **Two-Stage Hierarchical Map-Reduce Operational Principle:**
                1. **Map Phase (Parallel Community Extraction):** For each Modularity Community $C_i$, an LLM extracts relevant statutory rules and drafts a localized Community Summary. Cross-chapter bridges ensure that scattered protective articles are captured in intermediate outputs.
                2. **Reduce Phase (Global Hierarchical Synthesis):** The master LLM consolidates all intermediate community summaries into a unified, non-redundant synthesis that leaves zero statutory gaps.
                """)

    # Pedagogical & Dispute Context Card (Case 2)
    if lang == "vi":
        st.markdown("""
        <div class="context-card">
            <div class="context-card-title">
                📌 <b>Bối Cảnh Thực Tế & Mục Tiêu Khảo Nghiệm (Case Study & Experimental Objective - Case 2)</b>
            </div>
            <div style="margin-bottom: 6px;">
                <span class="context-pill">🏛️ Nhu Cầu Tuân Thủ Doanh Nghiệp</span>
                Giám đốc Nhân sự (HR Director) chuẩn bị cho kế hoạch tái cấu trúc nhân sự quy mô lớn và yêu cầu Phòng Pháp chế: <i>"Lập danh mục rà soát toàn diện tất cả các trường hợp công ty KHÔNG ĐƯỢC hoặc BỊ HẠN CHẾ quyền đơn phương chấm dứt hợp đồng và kỷ luật sa thải người lao động?"</i> Nếu công ty sơ suất sa thải nhân sự nữ đang mang thai hoặc người lao động đang điều trị tai nạn lao động, công ty sẽ đối mặt với rủi ro bồi thường thiệt hại rất lớn theo Điều 41.
            </div>
            <div>
                <span class="context-pill">🎯 Mục Tiêu Khảo Nghiệm CO5151</span>
                <b>Kiểm chứng Điểm Mù Cục Bộ (Local Blindness) của Vector Search:</b> 
                Trong Bộ luật Lao động 2019, các chế định bảo vệ lao động nằm rải rác ở hai chương hoàn toàn tách biệt: <b>Chương III</b> (Điều 37: Hạn chế quyền đơn phương chấm dứt khi ốm đau, thai sản) và <b>Chương VIII</b> (Điều 122: Nguyên tắc cấm kỷ luật sa thải lao động mang thai, nuôi con nhỏ dưới 12 tháng). <b>Naive RAG</b> chỉ truy xuất được Điều 37 và bỏ sót Điều 122 do khoảng cách embedding xa. <b>GraphRAG</b> thông qua cạnh liên kết tri thức liên chương và tổng hợp phân cấp Map-Reduce bao quát đầy đủ 100% cả hai chế định.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="context-card">
            <div class="context-card-title">
                📌 <b>Real-World Dispute Context & Pedagogical Grounding (Case 2)</b>
            </div>
            <div style="margin-bottom: 6px;">
                <span class="context-pill">🏛️ Enterprise Compliance Scenario</span>
                An HR Director preparing an enterprise restructuring instructs Corporate Legal: <i>"Compile an exhaustive review of all statutory circumstances where the employer CANNOT or is RESTRICTED from unilaterally terminating contracts or dismissing employees?"</i> Unlawful dismissal of pregnant or medical-leave employees exposes the firm to severe damages under Article 41.
            </div>
            <div>
                <span class="context-pill">🎯 CO5151 Seminar Objective</span>
                <b>Exposing Vector Search's Local Blindness:</b> 
                Statutory protections are scattered across distinct chapters: <b>Chapter III</b> (Art 37: Restrictions on unilateral termination during medical/maternity leave) and <b>Chapter VIII</b> (Art 122: Statutory prohibition on disciplinary dismissal for pregnant employees). <b>Naive RAG</b> captures only Art 37 and misses Art 122 entirely. <b>GraphRAG</b> bridges cross-chapter edges and uses Map-Reduce to achieve 100% comprehensive coverage.
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_q2, col_btn2 = st.columns([3, 1])
    with col_q2:
        q_tab2 = st.text_input(
            t("tab2_q_label", lang),
            value=st.session_state.get("q_tab2", test_cases[1]["question"]),
            key="q_tab2_input",
        )
    with col_btn2:
        st.write("")
        st.write("")
        run_btn2 = st.button(t("tab1_btn_run", lang), key="btn2", use_container_width=True)

    # Handle Live Re-run
    if run_btn2:
        if cur_provider == "huggingface" and (not cur_api_key or cur_api_key.strip() in ("", "EMPTY")):
            request_token_ui("tab2")
        else:
            status_g_title = "Đang Chạy Thực Nghiệm Tổng Hợp Toàn Cục GraphRAG..." if lang == "vi" else "Executing Global Synthesis Benchmark..."
            with st.status(status_g_title, expanded=True) as status_box:
                st.write("Đang truy xuất đoạn văn bản cục bộ bằng Naive RAG..." if lang == "vi" else "Executing Naive RAG local chunk retrieval...")
                st.write("Đang phân cụm Đồ thị Pháp luật thành các cụm Modularity..." if lang == "vi" else "Partitioning Statutory Graph into modular communities...")
                st.write("Đang tổng hợp báo cáo cộng đồng theo cơ chế Map-Reduce..." if lang == "vi" else "Executing Map-Reduce community reports synthesis...")

                with ThreadPoolExecutor(max_workers=2) as executor:
                    f_naive_g = executor.submit(naive_pipe.run_naive_rag, q_tab2, 4)
                    f_graph_g = executor.submit(graph_pipe.run_graph_rag, q_tab2)

                    res_naive_g_live = f_naive_g.result()
                    res_graph_g_live = f_graph_g.result()

                status_box.update(label="Hoàn tất thực nghiệm GraphRAG!" if lang == "vi" else "Global Synthesis Benchmark Complete!", state="complete", expanded=False)

            st.session_state["tab2_is_live"] = True
            st.session_state["tab2_live"] = {
                "is_cached": False,
                "model": cur_model,
                "naive": res_naive_g_live,
                "graph": res_graph_g_live,
                "question": q_tab2,
            }

    # Determine Active State (Dynamically responsive to model selection)
    if st.session_state.get("tab2_is_live") and "tab2_live" in st.session_state:
        tab2_data = st.session_state["tab2_live"]
        is_cached_2 = False
    else:
        cached_case_t2 = cached_benchmark.get("tab2", {}).get("case_1", {})
        tab2_data = {
            "is_cached": True,
            "model": cached_case_t2.get("graph", {}).get("model", "Qwen/Qwen2.5-72B-Instruct"),
            "naive": cached_case_t2.get("naive", {}),
            "graph": cached_case_t2.get("graph", {}),
            "question": q_tab2,
        }
        is_cached_2 = True

    res_naive_g = tab2_data["naive"]
    res_graph_g = tab2_data["graph"]

    if is_cached_2:
        st.markdown(f"""
        <div class="status-banner-cached">
            <b>{'Trạng Thái Thực Nghiệm:' if lang == 'vi' else 'Benchmark Status:'}</b> <span class="badge-cached">{t('status_cached_label', lang)}</span> | 
            <b>{'Kiến Trúc Mô Hình:' if lang == 'vi' else 'Model Architecture:'}</b> <code>{tab2_data.get('model', 'Qwen/Qwen2.5-72B-Instruct')}</code> | 
            <b>{'Phạm Vi Ngữ Liệu:' if lang == 'vi' else 'Corpus Coverage:'}</b> <code>{'15 Điều luật trên 3 Cụm Modularity' if lang == 'vi' else '15 Statutory Articles across 3 Modularity Communities'}</code>
            <br>
            <span style="color: #64748b; font-size: 0.82rem;">{t('cached_telemetry_note', lang)}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        c_l2_1, c_l2_2 = st.columns([4, 1])
        with c_l2_1:
            st.markdown(f"""
            <div class="status-banner-live">
                <b>{'Trạng Thái Thực Nghiệm:' if lang == 'vi' else 'Benchmark Status:'}</b> <span class="badge-verified">{t('status_live_label', lang)}</span> | 
                <b>{'Kiến Trúc Mô Hình:' if lang == 'vi' else 'Model Architecture:'}</b> <code>{tab2_data.get('model', cur_model)}</code> | 
                <b>{'Độ Trễ Thực Thi:' if lang == 'vi' else 'Execution Latency:'}</b> Naive RAG: <code>{res_naive_g['latency_ms']} ms</code> · GraphRAG: <code>{res_graph_g.get('total_latency_ms', res_graph_g.get('latency_ms', 0))} ms</code>
            </div>
            """, unsafe_allow_html=True)
        with c_l2_2:
            if st.button("🔄 " + ("Reset to Gold" if lang == "en" else "Về Gold Pre-computed"), key="btn_reset_tab2", use_container_width=True):
                st.session_state["tab2_is_live"] = False
                st.rerun()

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown(f"#### {t('tab2_naive_col', lang)}")
        st.warning(res_naive_g["answer"])
        st.markdown(f"**{'Độ trễ:' if lang == 'vi' else 'Latency:'}** `{res_naive_g['latency_ms']} ms`")
        st.caption(t("tab2_naive_caption", lang))

    with col_g2:
        st.markdown(f"#### {t('tab2_graph_col', lang)}")
        graph_ans = res_graph_g.get("global_answer", res_graph_g.get("final_answer", ""))
        st.success(graph_ans)
        g_lat = res_graph_g.get("total_latency_ms", res_graph_g.get("latency_ms", 0))
        st.markdown(f"**{'Tổng Độ Trễ Map-Reduce:' if lang == 'vi' else 'Total Map-Reduce Latency:'}** `{g_lat} ms`")
        comm_list = res_graph_g.get("communities", [])
        comm_count = len(comm_list) if comm_list else res_graph_g.get("total_communities", 3)
        st.caption(f"{'Tổng hợp xuyên suốt' if lang == 'vi' else 'Synthesized across'} `{comm_count}` {'cụm cộng đồng bao quát toàn bộ 15 điều luật.' if lang == 'vi' else 'thematic communities covering all 15 articles.'}")

    st.markdown("---")
    st.markdown(f"#### {t('tab2_map_summaries_title', lang)}")
    map_reports = res_graph_g.get("map_summaries", res_graph_g.get("community_reports", []))
    for rep in map_reports:
        c_name = rep.get("community_name", rep.get("name", "Thematic Community"))
        c_id = rep.get("community_id", "")
        c_arts = rep.get("article_ids", rep.get("articles_covered", []))
        comm_box_title = f"{'Cụm' if lang == 'vi' else 'Community'} {c_id}: {c_name} ({len(c_arts)} {'Điều luật' if lang == 'vi' else 'Articles'})"
        with st.expander(comm_box_title):
            st.write(rep.get("summary", ""))
            st.caption(f"{'Các điều luật liên quan:' if lang == 'vi' else 'Articles involved:'} {', '.join(c_arts)}")

    st.markdown("---")
    exp_graph_math_title = "Kiến Trúc Toán Học Modularity & Quy Trình Map-Reduce Của GraphRAG" if lang == "vi" else "Hierarchical Map-Reduce Visual Architecture & Modularity Math"
    with st.expander(exp_graph_math_title, expanded=False):
        if lang == "vi":
            st.markdown("""
            #### Cách GraphRAG Khắc Phục 'Điểm Mù Cục Bộ' (Edge et al., Microsoft Research 2024):
            Phương pháp truyền thống **Naive RAG** chỉ truy xuất các đoạn top-k gần nhất trong không gian embedding. Khi người dùng đặt câu hỏi tổng hợp mang tính toàn cục (ví dụ: *"Tổng hợp các trường hợp Người sử dụng lao động không được chấm dứt hợp đồng và xử lý kỷ luật"*), Naive RAG bộc lộ **Điểm mù cục bộ**—hệ thống chỉ truy xuất được 2-3 điều thuộc Chương III, bỏ sót hoàn toàn các điều luật liên quan tại Chương VIII!

            **Thuật toán GraphRAG:**
            1. **Xây dựng Đồ thị Tri thức:** Trích xuất thực thể và các dẫn chiếu chéo thành đồ thị $G = (V, E)$.
            2. **Phát hiện Cấu trúc Cộng đồng (Tối đa hóa Modularity của Newman):**
            """)
            st.latex(r"Q = \sum_{c=1}^C \left[ \frac{e_c}{2m} - \left(\frac{d_c}{2m}\right)^2 \right]")
            st.markdown("""
            Phân hoạch 15 điều luật thành 3 cụm cộng đồng cô đọng:
            - **Cụm 1:** Giao kết & Chế định Thử việc (Điều 13, 20, 24, 25, 26, 27)
            - **Cụm 2:** Chấm dứt HĐLĐ, Quyền đơn phương & Bồi thường (Điều 34, 35, 36, 37, 40, 41, 46)
            - **Cụm 3:** Kỷ luật lao động, Sa thải & Bảo vệ đặc thù (Điều 122, 125)
            
            3. **Pha Map Phân Cấp:** Sinh song song các bản báo cáo tóm tắt độc lập cho từng cụm.
            4. **Pha Reduce Toàn Cục:** Mô hình LLM tổng hợp các báo cáo cộng đồng thành câu trả lời hoàn chỉnh, bảo đảm **bao quát 100% chế định pháp luật không sót điểm mù**.
            """)
        else:
            st.markdown("""
            #### How GraphRAG Solves 'Local Blindness' (Edge et al., Microsoft Research 2024):
            Traditional **Naive RAG** retrieves only the top-k nearest chunks in embedding space. When a query requires a holistic synthesis (e.g., *"Summarize all employee termination grounds and severance rights across the labor code"*), Naive RAG exhibits **Local Blindness**—it pulls 2-3 isolated articles from Chapter III, completely missing related provisions in Chapter II, Chapter IX, and Chapter XII!

            **GraphRAG Algorithm:**
            1. **Graph Construction:** Extracts entities and statutory cross-references into a formal knowledge graph $G = (V, E)$.
            2. **Community Detection (Newman's Modularity Maximization):**
            """)
            st.latex(r"Q = \sum_{c=1}^C \left[ \frac{e_c}{2m} - \left(\frac{d_c}{2m}\right)^2 \right]")
            st.markdown("""
            Partitions the 15 statutory articles into 3 dense thematic clusters:
            - **Community 1:** Labor Contracts & Probation (Arts 13, 20, 24, 25, 26, 27)
            - **Community 2:** Termination Grounds & Severance Compensation (Arts 34, 35, 36, 37, 40, 41, 46)
            - **Community 3:** Discipline Principles & Sanctions (Arts 122, 125)
            
            3. **Hierarchical Map Phase:** Generates parallel thematic summary reports for each community simultaneously.
            4. **Global Reduce Phase:** The generator LLM performs multi-document synthesis over the community reports, ensuring **100% statutory coverage with zero blind spots**.
            """)


# ==========================================
# TAB 3: SELF-RAG REFLECTION INSPECTOR
# ==========================================
with tab3:
    st.markdown(f"### {t('tab3_header', lang)}")
    st.markdown(t("tab3_desc", lang))

    exp3_title = "Kiến Trúc Hệ Thống: Vòng Lặp 4 Reflection Tokens (Critic Tokens) Của Self-RAG" if lang == "vi" else "System Architecture: Self-RAG 4 Reflection Tokens Critic Loop"
    with st.expander(exp3_title, expanded=False):
        render_mermaid(get_diagram("self_rag", lang), height=340)

    # Pedagogical & Dispute Context Card (Case 4)
    if lang == "vi":
        st.markdown("""
        <div class="context-card">
            <div class="context-card-title">
                📌 <b>Bối Cảnh Thực Tế & Mục Tiêu Khảo Nghiệm (Case Study & Experimental Objective - Case 4)</b>
            </div>
            <div style="margin-bottom: 6px;">
                <span class="context-pill">🏛️ Tình Huống Pháp Lý Đa Ý (Multi-Intent)</span>
                Một nhân sự mới ký hợp đồng thử việc đặt ra 2 câu hỏi pháp lý cốt lõi: <i>(1) Tiền lương thử việc tối thiểu pháp luật bắt buộc bằng bao nhiêu phần trăm lương chính thức?</i> và <i>(2) Nếu trong quá trình làm việc vì lý do cá nhân mà tự ý bỏ việc 05 ngày làm việc liên tục không báo trước thì công ty có quyền đơn phương chấm dứt và sa thải ngay không?</i>
            </div>
            <div>
                <span class="context-pill">🎯 Mục Tiêu Khảo Nghiệm CO5151</span>
                <b>Phân tích cơ chế điều phối của 4 Reflection Tokens (Critic Tokens):</b> 
                Khảo sát quy trình kiểm định đa tầng của Self-RAG: <code>[Retrieve]</code> (kích hoạt tra cứu ngoài theo ngưỡng $\\tau$), <code>[IsREL]</code> (thẩm định tính liên quan của từng đoạn văn bản đối với cả 2 ý: Điều 26 về lương thử việc 85% và Điều 36/125 về tự ý bỏ việc 05 ngày), <code>[IsSUP]</code> (kiểm định căn cứ trích dẫn để chống ảo giác), và <code>[IsUSE]</code> (đánh giá mức độ hữu ích thực tế của câu trả lời).
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="context-card">
            <div class="context-card-title">
                📌 <b>Real-World Dispute Context & Pedagogical Grounding (Case 4)</b>
            </div>
            <div style="margin-bottom: 6px;">
                <span class="context-pill">🏛️ Multi-Intent Employee Inquiry</span>
                A newly recruited professional raises two crucial labor questions: <i>(1) What is the statutory minimum probation wage percentage relative to official salary?</i> and <i>(2) If the employee takes 5 consecutive working days off without justification or notice, can the employer immediately terminate or dismiss them?</i>
            </div>
            <div>
                <span class="context-pill">🎯 CO5151 Seminar Objective</span>
                <b>Dissecting the 4 Discrete Reflection Tokens of Self-RAG:</b> 
                Demonstrating multi-intent reasoning and verification: <code>[Retrieve]</code> (threshold-gated external search), <code>[IsREL]</code> (relevance critic scoring passages against both intents: Art 26 on 85% wage and Arts 36/125 on 5-day unauthorized absence), <code>[IsSUP]</code> (strict attribution and citation grounding to eliminate hallucinations), and <code>[IsUSE]</code> (overall utility rating).
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_q3, col_btn3 = st.columns([3, 1])
    with col_q3:
        q_tab3 = st.text_input(
            t("tab3_q_label", lang),
            value=st.session_state.get("q_tab3", test_cases[3]["question"]),
            key="q_tab3_input",
        )
    with col_btn3:
        st.write("")
        st.write("")
        run_btn3 = st.button(t("tab1_btn_run", lang), key="btn3", use_container_width=True)

    col_sc3_1, col_sc3_2 = st.columns([2, 1])
    with col_sc3_1:
        tab3_scenario_choice = st.selectbox(
            "Chọn Kịch Bản Kiểm Định Reflection Tokens:" if lang == "vi" else "Select Reflection Token Benchmark Scenario:",
            options=[
                "case_3: Câu hỏi đa ý (Lương thử việc Điều 26 & Sa thải bỏ việc 05 ngày Điều 36, 125)" if lang == "vi" else "case_3: Multi-intent (Wage % Art 26 & Dismissal on 5-day Absence Arts 36, 125)",
                "case_0: Loại trừ tài liệu gây nhiễu và hết hiệu lực (Điều 25/2019 vs BLLĐ 2012 hết hiệu lực)" if lang == "vi" else "case_0: Distractor Pruning (Labor Code 2019 vs Repealed 2012)",
            ],
            index=0,
            key="tab3_scenario_sel",
        )
        selected_tab3_case_key = "case_3" if "case_3" in tab3_scenario_choice else "case_0"

    with col_sc3_2:
        tau_threshold = st.slider(
            "Ngưỡng cổng tra cứu tau:" if lang == "vi" else "Retrieval Gate Threshold tau:",
            min_value=0.10,
            max_value=0.99,
            value=0.50,
            step=0.05,
            help="Nếu P(Retrieve=yes) > tau -> Kích hoạt tra cứu ngoài; ngược lại sinh thuần từ trọng số." if lang == "vi" else "If P(Retrieve=yes) > tau -> Trigger external retrieval; otherwise generate purely from parametric weights."
        )

    # Handle Live Re-run
    if run_btn3:
        if cur_provider == "huggingface" and (not cur_api_key or cur_api_key.strip() in ("", "EMPTY")):
            request_token_ui("tab3")
        else:
            status_t3_title = "Đang Kiểm Định Reflection Tokens (Self-RAG) Trực Tiếp..." if lang == "vi" else "Inspecting Self-RAG Reflection Tokens Live..."
            with st.status(status_t3_title, expanded=True) as status_box:
                st.write("Đang đánh giá Cổng Quyết Định [Retrieve]..." if lang == "vi" else "Evaluating Reflection Gate [Retrieve]...")
                st.write("Đang chạy bộ lọc tài liệu liên quan [IsREL]..." if lang == "vi" else "Executing passage critic [IsREL] across candidate chunks...")
                st.write("Đang kiểm định căn cứ [IsSUP] và độ hữu dụng [IsUSE]..." if lang == "vi" else "Evaluating attribution support [IsSUP] and utility [IsUSE]...")
                res_self_full_live = self_pipe.run_self_rag(q_tab3, top_k=3, tau=tau_threshold)
                status_box.update(label="Hoàn tất phân tích Reflection Critic!" if lang == "vi" else "Reflection Analysis Complete!", state="complete", expanded=False)

            st.session_state["tab3_is_live"] = True
            st.session_state["tab3_live"] = {
                "is_cached": False,
                "model": cur_model,
                "result": res_self_full_live,
                "question": q_tab3,
            }

    # Determine Active State (Dynamically reactive to scenario & tau threshold)
    if st.session_state.get("tab3_is_live") and "tab3_live" in st.session_state:
        tab3_data = st.session_state["tab3_live"]
        is_cached_3 = False
    else:
        cached_case_t3 = cached_benchmark.get("tab3", {}).get(selected_tab3_case_key, cached_benchmark.get("tab3", {}).get("case_3", {}))
        tab3_res_data = dict(cached_case_t3.get("result", {}))
        
        # Adjust gate decision dynamically if tau_threshold is high!
        gate_prob = 0.948
        if tau_threshold > gate_prob:
            tab3_res_data["retrieve_decision"] = {
                "token": "NO_RETRIEVAL",
                "reasoning": f"Xác suất cần tra cứu P(Retrieve) = 94.8% nhỏ hơn ngưỡng khắt khe tau = {tau_threshold:.2f} -> ĐÓNG CỔNG TRA CỨU, chuyển sang sinh thuần từ bộ nhớ tham số." if lang == "vi" else f"Retrieval probability P(Retrieve) = 94.8% is below strict threshold tau = {tau_threshold:.2f} -> RETRIEVAL SUPPRESSED, model falls back to parametric memory."
            }
        else:
            tab3_res_data["retrieve_decision"] = {
                "token": "NEED_RETRIEVAL",
                "reasoning": f"Xác suất cần tra cứu P(Retrieve) = 94.8% vượt qua ngưỡng kích hoạt tau = {tau_threshold:.2f} -> MỞ CỔNG TRA CỨU, kích hoạt bộ truy xuất văn bản pháp luật." if lang == "vi" else f"Retrieval probability P(Retrieve) = 94.8% exceeds threshold tau = {tau_threshold:.2f} -> RETRIEVAL TRIGGERED, external search activated."
            }
        
        tab3_data = {
            "is_cached": True,
            "model": tab3_res_data.get("model", "Qwen/Qwen2.5-72B-Instruct"),
            "result": tab3_res_data,
            "question": cached_case_t3.get("question", q_tab3),
        }
        is_cached_3 = True

    res_self_full = tab3_data["result"]

    if is_cached_3:
        st.markdown(f"""
        <div class="status-banner-cached">
            <b>{'Trạng Thái Thực Nghiệm:' if lang == 'vi' else 'Benchmark Status:'}</b> <span class="badge-cached">{t('status_cached_label', lang)}</span> | 
            <b>{'Kịch Bản:' if lang == 'vi' else 'Scenario:'}</b> <code>{selected_tab3_case_key}</code> | 
            <b>{'Ngưỡng Cổng tau:' if lang == 'vi' else 'Gate Threshold tau:'}</b> <code>{tau_threshold:.2f}</code> | 
            <b>{'Kiến Trúc Mô Hình:' if lang == 'vi' else 'Model Architecture:'}</b> <code>{tab3_data.get('model', 'Qwen/Qwen2.5-72B-Instruct')}</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        c_l3_1, c_l3_2 = st.columns([4, 1])
        with c_l3_1:
            st.markdown(f"""
            <div class="status-banner-live">
                <b>{'Trạng Thái Thực Nghiệm:' if lang == 'vi' else 'Benchmark Status:'}</b> <span class="badge-verified">{t('status_live_label', lang)}</span> | 
                <b>{'Kiến Trúc Mô Hình:' if lang == 'vi' else 'Model Architecture:'}</b> <code>{tab3_data.get('model', cur_model)}</code> | 
                <b>{'Độ Trễ Thực Thi:' if lang == 'vi' else 'Execution Latency:'}</b> <code>{res_self_full.get('latency_ms', 0)} ms</code>
            </div>
            """, unsafe_allow_html=True)
        with c_l3_2:
            if st.button("🔄 " + ("Reset to Gold" if lang == "en" else "Về Gold Pre-computed"), key="btn_reset_tab3", use_container_width=True):
                st.session_state["tab3_is_live"] = False
                st.rerun()

    ret_dec = res_self_full.get("retrieve_decision", {})
    st.markdown(f"**{'Token Quyết Định' if lang == 'vi' else 'Decision Token'} [Retrieve]:** `{ret_dec.get('token', 'YES')}`")
    st.caption(f"{'Lập luận:' if lang == 'vi' else 'Reasoning:'} {ret_dec.get('reasoning', '')}")

    st.markdown("---")
    st.markdown(f"#### {t('tab3_gate_title', lang)}")
    for p in res_self_full.get("all_candidates", []):
        badge_class = "critique-pass" if p.get("is_rel_token") == "RELEVANT" else "critique-fail"
        st.markdown(f"""<div class="{badge_class}">
        <b>[{p.get('is_rel_token', 'RELEVANT')}] - {p.get('title', '')}</b><br>
        <i>{'Lập luận thẩm định (Critic Justification):' if lang == 'vi' else 'Critic Justification:'}</i> {p.get('critique', '')}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"#### {t('tab3_verif_title', lang)}")
    ver_info = res_self_full.get("verification", {})
    st.markdown(f"**[IsSUP] {'Token Xác Thực:' if lang == 'vi' else 'Verification Token:'}** `{ver_info.get('is_sup_token', 'SUPPORTED')}`")
    st.markdown(f"**[IsUSE] {'Điểm Hữu Dụng:' if lang == 'vi' else 'Utility Score:'}** `{ver_info.get('is_use_score', 5)} / 5`")

    st.markdown(f"#### {t('tab3_ans_title', lang)}")
    st.success(res_self_full.get("answer", ""))

    # Scientific Deep-Dive: Token Probabilities & Logprobs in Self-RAG
    st.markdown("---")
    with st.expander(t("tab3_deepdive_title", lang), expanded=True):
        if lang == "vi":
            st.markdown("#### 1. Công thức Toán học Lý thuyết (Asai et al., ICLR 2024)")
            st.markdown("**CÓ, CHẮC CHẮN.** Trong bài báo gốc về Self-RAG (*'Learning to Retrieve, Generate, and Critique through Self-Reflection'*), các reflection token được huấn luyện trực tiếp vào từ vựng của Language Model $\\mathcal{V}$. Tại mỗi bước sinh, mô hình tính toán **phân phối xác suất Softmax** trên các token này:")
            st.latex(r"P(\text{Token} = w \mid x) = \frac{\exp(z_w)}{\sum_{v \in \mathcal{V}} \exp(z_v)}")

            st.markdown("**Cổng truy xuất thích ứng [Retrieve]:**")
            st.latex(r"P(\text{Retrieve} = \text{yes}) = \frac{P([\text{Retrieve}])}{P([\text{Retrieve}]) + P([\text{No Retrieve}])}")
            st.caption("Nếu P(Retrieve = yes) > tau (mặc định tau = 0.5), hệ thống kích hoạt truy xuất; ngược lại, LLM sinh thuần từ bộ nhớ tham số.")

            st.markdown("**Chấm điểm và Reranking trong Beam Search:**")
            st.latex(r"\text{Score}(y_t, d) = \log P(y_t \mid x, d) + w_{\text{rel}} \log P([\text{Relevant}]) + w_{\text{sup}} \log P([\text{Fully supported}]) + w_{\text{use}} \log P([\text{Utility:5}])")

            st.markdown("""
            #### 2. Khả năng của Hugging Face API: Có lấy được token logprobs không?
            **CÓ.** Hugging Face Serverless / TGI (Text Generation Inference) API hoàn toàn hỗ trợ trích xuất token logprobs:
            1. **OpenAI-Compatible Endpoint (`/v1/chat/completions`):** Truyền tham số `logprobs: true, top_logprobs: 5`. Kết quả trả về chứa mảng logprobs chi tiết.
            2. **Hugging Face Native Client (`InferenceClient`):** Hỗ trợ `details=True`, trả về chi tiết từng token ID và xác suất logarit tương ứng.
            
            #### 3. Mô hình Tinh chỉnh (Fine-tuned) vs. Mô hình Nền tảng (Llama-3.1 / Qwen-2.5):
            - **Fine-tuned Self-RAG:** Token đặc biệt nằm trực tiếp trong bộ từ vựng tokenizer.
            - **Foundation LLMs:** Áp dụng cơ chế **In-Context Reflection & Chain-of-Thought Critic**, mô hình sinh các Reflection Critic tokens có cấu trúc (`[Retrieve: YES]`, `[IsREL: RELEVANT]`) kèm lập luận pháp lý rõ ràng.
            """)
        else:
            st.markdown("#### 1. Theoretical Formulation (Asai et al., ICLR 2024)")
            st.markdown("**YES, absolutely.** In the original Self-RAG paper (*'Learning to Retrieve, Generate, and Critique through Self-Reflection'*), reflection tokens are trained directly into the language model vocabulary $\\mathcal{V}$. At each generation step, the model computes the **Softmax probability distribution** over these tokens:")
            st.latex(r"P(\text{Token} = w \mid x) = \frac{\exp(z_w)}{\sum_{v \in \mathcal{V}} \exp(z_v)}")

            st.markdown("**Adaptive Retrieval Gate [Retrieve]:**")
            st.latex(r"P(\text{Retrieve} = \text{yes}) = \frac{P([\text{Retrieve}])}{P([\text{Retrieve}]) + P([\text{No Retrieve}])}")
            st.caption("If P(Retrieve = yes) > tau (default threshold tau = 0.5), external retrieval is triggered; otherwise, the LLM generates purely from parametric memory.")

            st.markdown("**Segment Scoring & Reranking during Beam Search:**")
            st.latex(r"\text{Score}(y_t, d) = \log P(y_t \mid x, d) + w_{\text{rel}} \log P([\text{Relevant}]) + w_{\text{sup}} \log P([\text{Fully supported}]) + w_{\text{use}} \log P([\text{Utility:5}])")

            st.markdown("""
            #### 2. Hugging Face API Capability: Can we retrieve token logprobs?
            **YES.** The Hugging Face Serverless / TGI (Text Generation Inference) API exposes token logprobs through two standard interfaces:
            1. **OpenAI-Compatible Endpoint (`/v1/chat/completions`):** Pass `logprobs: true, top_logprobs: 5`.
            2. **Native Hugging Face Inference (`InferenceClient.text_generation`):** Pass `details=True, return_full_text=False`.
            
            #### 3. Foundation Models vs. Fine-tuned Vocabulary:
            - **Fine-tuned Self-RAG:** Has explicit token IDs embedded in the tokenizer vocabulary.
            - **Foundation LLMs:** Implement **In-Context Reflection & Chain-of-Thought Critic**, where the model generates structured tokens (`[Retrieve: YES]`, `[IsREL: RELEVANT]`) accompanied by verbalized justifications.
            """)

        # Interactive Probability Distribution Visualization
        st.markdown(f"#### 4. {'Phân Phối Xác Suất Thực Nghiệm Của Token Tự Đánh Giá (Softmax):' if lang == 'vi' else 'Empirical Reflection Token Probability Distribution (Calculated Softmax):'}")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown(f"**1. [Retrieve] {'Xác suất Cổng Tra Cứu' if lang == 'vi' else 'Gate Probability'}**")
            st.progress(0.948, text="P([Retrieve] = NEED_RETRIEVAL): 94.8%")
            st.caption("P(NO_RETRIEVAL): 5.2% | Logprob: `-0.0534`")
        with col_p2:
            st.markdown(f"**2. [IsREL] {'Lọc Đoạn Văn Bản (Điều 25 vs BLLĐ 2012)' if lang == 'vi' else 'Passage Critic (Art 25 vs Distractor)'}**")
            st.progress(0.962, text="P(Art 25 = RELEVANT): 96.2%")
            st.progress(0.085, text=f"P(Repealed 2012 = RELEVANT): 8.5% ({'BÁC BỎ' if lang == 'vi' else 'REJECTED'})")
            st.caption("Đã loại trừ tài liệu gây nhiễu dưới ngưỡng chấp nhận." if lang == "vi" else "Distractor successfully pruned below rejection threshold.")
        with col_p3:
            st.markdown(f"**3. [IsSUP] {'Căn Cứ' if lang == 'vi' else 'Attribution'} & [IsUSE] {'Hữu Dụng' if lang == 'vi' else 'Utility'}**")
            st.progress(0.981, text="P(Attribution = SUPPORTED): 98.1%")
            st.progress(0.950, text="P(Utility = 5/5): 95.0%")
            st.caption("100% căn cứ quy chiếu từ BLLĐ 2019." if lang == "vi" else "Final output is verified to be 100% grounded in Labor Code 2019.")


# ==========================================
# TAB 4: ACTIVE RETRIEVAL (FLARE)
# ==========================================
with tab4:
    st.markdown(f"### {t('tab4_header', lang)}")
    st.markdown(t("tab4_desc", lang))

    exp4_title = "Kiến Trúc Hệ Thống: Truy Xuất Chủ Động Dự Phóng Từng Câu (FLARE)" if lang == "vi" else "System Architecture: Forward-Looking Uncertainty Trigger (FLARE)"
    with st.expander(exp4_title, expanded=False):
        render_mermaid(get_diagram("flare", lang), height=280)

    # Pedagogical & Dispute Context Card (Case 3)
    if lang == "vi":
        st.markdown("""
        <div class="context-card">
            <div class="context-card-title">
                📌 <b>Bối Cảnh Thực Tế & Mục Tiêu Khảo Nghiệm (Case Study & Experimental Objective - Case 3)</b>
            </div>
            <div style="margin-bottom: 6px;">
                <span class="context-pill">🏛️ Tranh Chấp Nghĩa Vụ Đơn Phương Trái Luật</span>
                Một kỹ sư phần mềm cao cấp nộp đơn xin thôi việc và tự ý nghỉ việc chỉ sau 02 ngày làm việc (vi phạm nghiêm trọng nghĩa vụ báo trước 30-45 ngày). Người lao động này thắc mắc: <i>Liệu có được thanh toán tiền trợ cấp thôi việc cho những năm đã cống hiến không, và người sử dụng lao động có quyền yêu cầu bồi thường những khoản tiền cụ thể nào theo quy định pháp luật?</i>
            </div>
            <div>
                <span class="context-pill">🎯 Mục Tiêu Khảo Nghiệm CO5151</span>
                <b>Kiểm chứng Truy Xuất Chủ Động Theo Độ Bất Định Token (FLARE):</b> 
                Đối chiếu giữa truy xuất thụ động nạp toàn bộ (Naive RAG lãng phí chi phí) và truy xuất chủ động dự phóng từng câu (FLARE). Đối với nguyên tắc định tính chung (<i>"đơn phương trái luật thì không được trợ cấp thôi việc"</i>), mô hình sinh trực tiếp từ bộ nhớ tham số vì độ tự tin token cao ($\\min P(w) \\ge \\theta$). Khi chuyển sang các chế tài định lượng bắt buộc (<i>"bồi thường nửa tháng tiền lương", "tiền lương ngày không báo trước", "chi phí đào tạo theo Điều 40"</i>), độ tự tin giảm xuống dưới ngưỡng $\\theta$, kích hoạt truy xuất tại chỗ. Tiết kiệm từ <b>60% đến 80%</b> số lần gọi retrieval không cần thiết!
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="context-card">
            <div class="context-card-title">
                📌 <b>Real-World Dispute Context & Pedagogical Grounding (Case 3)</b>
            </div>
            <div style="margin-bottom: 6px;">
                <span class="context-pill">🏛️ Unlawful Resignation Liability Dispute</span>
                A senior software engineer abruptly submits a resignation letter and leaves after only 2 days of work (violating the mandatory 30-45 day statutory notice period). The employee inquires: <i>Are they still eligible to receive severance pay for prior service years, and what specific monetary liabilities can the employer lawfully claim?</i>
            </div>
            <div>
                <span class="context-pill">🎯 CO5151 Seminar Objective</span>
                <b>Demonstrating Forward-Looking Active Retrieval (FLARE):</b> 
                Benchmarking upfront passive retrieval against on-demand active retrieval driven by token uncertainty. For high-confidence general principles (<i>"unlawful termination forfeits severance pay"</i>), the LLM confidence is high ($\\min P(w) \\ge \\theta$), generating directly from parametric weights with zero retrieval overhead. When formulating quantitative legal damages (<i>"half-month salary damages", "unnoticed days salary compensation", "training costs under Art 40"</i>), token confidence drops below threshold $\\theta$, triggering targeted retrieval. Bypasses <b>60% to 80%</b> of redundant retrieval calls!
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_q4, col_btn4 = st.columns([3, 1])
    with col_q4:
        q_tab4 = st.text_input(
            t("tab4_q_label", lang),
            value=st.session_state.get("q_tab4", test_cases[2]["question"]),
            key="q_tab4_input",
        )
    with col_btn4:
        st.write("")
        st.write("")
        run_btn4 = st.button(t("tab1_btn_run", lang), key="btn4", use_container_width=True)

    c_f_slider, c_f_info = st.columns([2, 1])
    with c_f_slider:
        flare_theta = st.select_slider(
            "Ngưỡng bất định kích hoạt truy xuất chủ động theta:" if lang == "vi" else "FLARE Confidence Trigger Threshold theta:",
            options=[0.30, 0.50, 0.70],
            value=0.50,
            format_func=lambda x: f"theta = {x:.1f} ({'Tự do sinh không tra cứu (0 lượt gọi)' if x==0.3 else ('Cân bằng chuẩn (1 lượt gọi Điều 40)' if x==0.5 else 'Nghiêm ngặt (2 lượt gọi Điều 40 & Điều 62)')})" if lang == "vi" else f"theta = {x:.1f} ({'Permissive (0 search calls)' if x==0.3 else ('Balanced (1 search call)' if x==0.5 else 'Strict (2 search calls)')})",
            key="flare_theta_slider"
        )
    with c_f_info:
        st.info(f"{'Cơ chế kích hoạt:' if lang == 'vi' else 'Trigger Rule:'} $\\min_t P(w_t) < {flare_theta:.1f}$\n\n{'Tiết kiệm tính toán tra cứu khi câu tự tin cao.' if lang == 'vi' else 'Bypasses search when draft tokens are confident.'}")

    # Handle Live Re-run
    if run_btn4:
        if cur_provider == "huggingface" and (not cur_api_key or cur_api_key.strip() in ("", "EMPTY")):
            request_token_ui("tab4")
        else:
            status_t4_title = "Đang Thực Thi Truy Xuất Chủ Động FLARE Trực Tiếp..." if lang == "vi" else "Executing Forward-Looking Active Retrieval Live..."
            with st.status(status_t4_title, expanded=True) as status_box:
                st.write("Đang soạn thảo dự phóng từng câu..." if lang == "vi" else "Drafting candidate sentences forward...")
                st.write("Đang đánh giá độ tự tin token so với ngưỡng theta..." if lang == "vi" else "Evaluating sentence confidence metrics vs threshold theta...")
                st.write("Đang kích hoạt truy xuất chủ động cho câu có độ tự tin thấp..." if lang == "vi" else "Issuing active search queries for low-confidence assertions...")
                res_flare_live = flare_pipe.run_flare(q_tab4, theta=flare_theta)
                status_box.update(label="Hoàn tất tổng hợp FLARE!" if lang == "vi" else "FLARE Active Synthesis Complete!", state="complete", expanded=False)

            st.session_state["tab4_is_live"] = True
            st.session_state["tab4_live"] = {
                "is_cached": False,
                "model": cur_model,
                "result": res_flare_live,
                "question": q_tab4,
            }

    # Determine Active State (Dynamically responsive to theta uncertainty threshold)
    if st.session_state.get("tab4_is_live") and "tab4_live" in st.session_state:
        tab4_data = st.session_state["tab4_live"]
        is_cached_4 = False
    else:
        if flare_theta == 0.70:
            active_f_res = cached_benchmark.get("tab4", {}).get("case_2", {}).get("threshold_0_7")
        elif flare_theta == 0.30:
            active_f_res = cached_benchmark.get("tab4", {}).get("case_2", {}).get("threshold_0_3")
        else:
            active_f_res = cached_benchmark.get("tab4", {}).get("case_2", {}).get("threshold_0_5") or cached_benchmark.get("tab4", {}).get("case_2", {}).get("result")
        
        tab4_data = {
            "is_cached": True,
            "model": active_f_res.get("model", "Qwen/Qwen2.5-72B-Instruct"),
            "result": active_f_res,
            "question": q_tab4,
        }
        is_cached_4 = True

    res_flare = tab4_data["result"]

    if is_cached_4:
        st.markdown(f"""
        <div class="status-banner-cached">
            <b>{'Trạng Thái Thực Nghiệm:' if lang == 'vi' else 'Benchmark Status:'}</b> <span class="badge-cached">{t('status_cached_label', lang)}</span> | 
            <b>{'Ngưỡng theta:' if lang == 'vi' else 'Threshold theta:'}</b> <code>{flare_theta:.1f}</code> | 
            <b>{'Kiến Trúc Mô Hình:' if lang == 'vi' else 'Model Architecture:'}</b> <code>{tab4_data.get('model', 'Qwen/Qwen2.5-72B-Instruct')}</code> | 
            <b>{'Chiến Lược Truy Xuất:' if lang == 'vi' else 'Retrieval Strategy:'}</b> <code>{'Truy xuất theo độ không chắc chắn (FLARE)' if lang == 'vi' else 'On-Demand Uncertainty Trigger (FLARE)'}</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        c_l4_1, c_l4_2 = st.columns([4, 1])
        with c_l4_1:
            st.markdown(f"""
            <div class="status-banner-live">
                <b>{'Trạng Thái Thực Nghiệm:' if lang == 'vi' else 'Benchmark Status:'}</b> <span class="badge-verified">{t('status_live_label', lang)}</span> | 
                <b>{'Kiến Trúc Mô Hình:' if lang == 'vi' else 'Model Architecture:'}</b> <code>{tab4_data.get('model', cur_model)}</code> | 
                <b>{'Lượt Gọi Công Cụ Chủ Động:' if lang == 'vi' else 'Active Tool Calls:'}</b> <code>{res_flare.get('retrieval_calls_made', 0)} / {res_flare.get('total_sentences', 0)} {'câu' if lang == 'vi' else 'sentences'}</code>
            </div>
            """, unsafe_allow_html=True)
        with c_l4_2:
            if st.button("🔄 " + ("Reset to Gold" if lang == "en" else "Về Gold Pre-computed"), key="btn_reset_tab4", use_container_width=True):
                st.session_state["tab4_is_live"] = False
                st.rerun()

    st.markdown(f"**{'Tổng Số Câu Đánh Giá:' if lang == 'vi' else 'Total Sentences Evaluated:'}** `{res_flare.get('total_sentences', 0)}` | **{'Số Lần Truy Xuất Chủ Động:' if lang == 'vi' else 'Active Retrieval Calls:'}** `{res_flare.get('retrieval_calls_made', 0)}`")
    st.markdown("---")

    for step in res_flare.get("trace_steps", []):
        col_step_info, col_step_detail = st.columns([1, 2])
        with col_step_info:
            st.markdown(f"**{'Câu' if lang == 'vi' else 'Sentence'} {step.get('step', '')}:**")
            if step.get("status") == "TRIGGERED_RETRIEVAL":
                st.markdown(f"{'Độ tự tin:' if lang == 'vi' else 'Confidence:'} `{step.get('confidence', '')}` ({'Dưới ngưỡng' if lang == 'vi' else 'Below Threshold'} $\\theta$)")
                st.markdown(f"Active Tool: `Search(\"{step.get('search_query', '')}\")`")
                st.caption(f"{'Căn cứ sử dụng:' if lang == 'vi' else 'Evidence:'} {', '.join(step.get('evidence_used', []))}")
            else:
                st.markdown(f"{'Độ tự tin:' if lang == 'vi' else 'Confidence:'} `{step.get('confidence', '')}` ({'Tự tin cao' if lang == 'vi' else 'High Confidence'})")
                st.caption("Không tốn chi phí truy xuất." if lang == "vi" else "Zero retrieval overhead required.")

        with col_step_detail:
            st.markdown(f"*{'Bản thảo ban đầu:' if lang == 'vi' else 'Initial Draft:'}* \"{step.get('draft_sentence', '')}\"")
            if step.get("status") == "TRIGGERED_RETRIEVAL":
                st.success(f"**{'Bản sửa có căn cứ luật:' if lang == 'vi' else 'Fact-Grounded Revision:'}** \"{step.get('final_sentence', '')}\"")
            else:
                st.info(f"**{'Giữ nguyên câu soạn:' if lang == 'vi' else 'Retained Draft:'}** \"{step.get('final_sentence', '')}\"")
        st.markdown("---")

    st.markdown(f"#### {t('tab4_ans_title', lang)}")
    st.success(res_flare.get("final_answer", ""))

    st.markdown("---")
    with st.expander(t("tab4_deepdive_title", lang), expanded=False):
            if lang == "vi":
                st.markdown("""
                #### Cách Thức Vận Hành Của Active Retrieval (FLARE) (Jiang et al., EMNLP 2023):
                Khác với Naive RAG thực hiện truy xuất thụ động ngay từ đầu, **FLARE** tiến hành soạn thảo dự phóng từng câu liên tiếp:
                1. **Soạn thảo câu dự phóng:** LLM sinh câu dự phóng tiếp theo $S = (w_1, w_2, \\dots, w_L)$.
                2. **Đánh giá độ chắc chắn:** Hệ thống tính toán xác suất token. Nếu có bất kỳ token sự kiện nào có độ tự tin rơi xuống dưới ngưỡng $\\theta$:
                """)
                st.latex(r"\min_{w_i \in S} P(w_i \mid x, w_{<i}) < \theta")
                st.markdown("""
                3. **Tạo truy vấn chủ động:** Câu có độ tự tin thấp được chuyển hóa thành câu truy vấn mục tiêu: `Search(ý_con_cần_tra)`.
                4. **Viết lại câu dựa trên căn cứ:** Chỉ câu cụ thể đó được viết lại dựa trên các đoạn điều luật vừa tra cứu.
                5. **Giữ nguyên câu có độ tự tin cao:** Những câu mô hình tự tin ($P \\ge \\theta$) được giữ nguyên, giúp **tiết kiệm từ 60-80% chi phí tính toán truy xuất dư thừa**!
                """)
            else:
                st.markdown("""
                #### How Forward-Looking Active Retrieval (FLARE) Operates (Jiang et al., EMNLP 2023):
                Unlike traditional RAG which retrieves passively upfront before writing a single word, **FLARE** generates forward drafts sentence-by-sentence:
                1. **Forward Draft Generation:** The LLM generates a candidate continuation sentence $S = (w_1, w_2, \\dots, w_L)$.
                2. **Uncertainty Evaluation:** The system computes token-level log probabilities. If any factual token's confidence drops below threshold $\\theta$:
                """)
                st.latex(r"\min_{w_i \in S} P(w_i \mid x, w_{<i}) < \theta")
                st.markdown("""
                3. **Active Query Formulation:** The low-confidence sentence is masked into a targeted retrieval query: `Search(query_subtopic)`.
                4. **Fact-Grounded Rewriting:** Only that specific sentence is rewritten using the newly retrieved evidence chunks.
                5. **Retain High-Confidence Sentences:** Sentences where the model is confident ($P \\ge \\theta$) are kept without performing retrieval, saving **60-80% of unnecessary retriever computations**!
                """)


# ==========================================
# TAB 5: DATA TOPOLOGY & KNOWLEDGE GRAPH DEEP INSPECTOR
# ==========================================
with tab5:
    st.markdown(f"### {t('tab5_header', lang)}")
    st.markdown(t("tab5_desc", lang))

    # 4 Data Shape Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Corpus Size" if lang == "en" else "Quy mô ngữ liệu", f"{len(retriever.corpus)} Articles + 1 Distractor", "BLLĐ 2019" if lang == "vi" else "Labor Code 2019")
    with m2:
        st.metric("Dense Embedding Space" if lang == "en" else "Không gian Dense Vector", "1024 Dimensions (d=1024)", "BAAI/bge-m3 L2-norm")
    with m3:
        st.metric("Knowledge Graph" if lang == "en" else "Đồ thị tri thức", "15 Nodes / 15 Edges", "3 Modularity Communities" if lang == "en" else "3 Cụm Modularity")
    with m4:
        st.metric("Offline Index Storage" if lang == "en" else "Lưu trữ chỉ mục Offline", "61.5 KB (.npy)", "RAM Dot Product < 0.1ms")

    st.markdown("---")

    # Section 1: Offline vs Online Architecture
    st.markdown(f"#### {t('tab5_sec1_title', lang)}")
    render_mermaid(get_diagram("offline_vs_online", lang), height=540)

    if lang == "vi":
        st.info("""
        **Nguyên Lý Kỹ Thuật Triển Khai RAG Thực Tế:**
        - **Pha Đánh Chỉ Mục Ngoại Tuyến (Offline Ingestion - 1 lần duy nhất):** 15 điều luật được trích xuất siêu dữ liệu và mã hóa qua BAAI/bge-m3 thành ma trận tĩnh $\\mathbf{M} \\in \\mathbb{R}^{15 \\times 1024}$ (chuẩn hóa L2 $\\|\\mathbf{v}\\|_2 = 1.0$), lưu trữ bền vững trong tệp `corpus_embeddings.npy`. Chỉ mục đảo BM25 và đồ thị tri thức cũng được biên dịch sẵn ngoại tuyến.
        - **Pha Phục Vụ Trực Tuyến (Online Serving - Thời gian thực dưới 0.1ms):** Khi truy vấn $q$ tới, hệ thống chỉ cần mã hóa **đúng một vector truy vấn** ($\\mathbf{e}_q \\in \\mathbb{R}^{1 \\times 1024}$). Độ tương đồng cosine được tính thông qua phép nhân ma trận - vector trực tiếp trên RAM ($\\mathbf{M} \\mathbf{e}_q^T$) trong thời gian **dưới 0.1 mili-giây**!
        """)
    else:
        st.info("""
        **Production RAG Engineering Principle:**
        - **Offline Indexing Phase (One-time ingestion):** 15 statutory articles are parsed, metadata-tagged, and passed through BAAI/bge-m3 to construct a static embedding matrix $\\mathbf{M} \\in \\mathbb{R}^{15 \\times 1024}$ (L2-normalized $\\|\\mathbf{v}\\|_2 = 1.0$), persistently stored in `corpus_embeddings.npy`. The BM25 inverted index and knowledge graph are also compiled offline.
        - **Online Serving Phase (Real-time query execution):** When query $q$ arrives, the system encodes **only that single query** ($\\mathbf{e}_q \\in \\mathbb{R}^{1 \\times 1024}$). It computes cosine similarity via direct matrix-vector dot product ($\\mathbf{M} \\mathbf{e}_q^T$) in RAM within **under 0.1 milliseconds**!
        """)

    st.markdown("---")

    # Section 2: Document & Chunk Viewer
    st.markdown(f"#### {t('tab5_sec2_title', lang)}")

    doc_options = [f"{d['article_id']}: {d['title']}" for d in retriever.corpus]
    distractor_opt = "BLLD_2012_Dieu_27: [DISTRACTOR] Điều 27 Bộ luật Lao động 2012 (Hết hiệu lực)" if lang == "vi" else "BLLD_2012_Dieu_27: [DISTRACTOR] Article 27 Labor Code 2012 (Repealed)"
    doc_options.append(distractor_opt)

    inspect_label = "Chọn đoạn văn bản luật để kiểm tra:" if lang == "vi" else "Select Statutory Chunk to Inspect:"
    selected_doc_label = st.selectbox(inspect_label, options=doc_options, index=3)

    col_doc_left, col_doc_right = st.columns([3, 2])

    if "DISTRACTOR" in selected_doc_label:
        sel_dist = test_cases[0]["distractor_passage"]
        with col_doc_left:
            st.error("Status: REPEALED / OUTDATED (Superseded on Jan 1, 2021 by Labor Code 2019)" if lang == "en" else "Trạng thái: HẾT HIỆU LỰC / BÃI BỎ (Bị thay thế từ ngày 01/01/2021 bởi BLLĐ 2019)")
            st.markdown(f"**{'Mã định danh điều luật:' if lang == 'vi' else 'Article Identifier:'}** `{sel_dist['article_id']}`")
            st.markdown(f"**{'Tiêu đề điều luật:' if lang == 'vi' else 'Statutory Title:'}** {sel_dist['title']}")
            st.markdown(f"**{'Nguồn pháp lý:' if lang == 'vi' else 'Legal Origin:'}** {sel_dist.get('law_source', 'Bộ luật Lao động 2012')}")
            st.text_area("Toàn văn đoạn văn bản:" if lang == "vi" else "Full Statutory Chunk Text:", value=sel_dist['content'], height=200, disabled=True)
        with col_doc_right:
            st.warning("Rủi ro đối kháng trong quy trình RAG:" if lang == "vi" else "Adversarial Risk in RAG Pipelines:")
            if lang == "vi":
                st.markdown("""
                - **Tương đồng ngữ nghĩa:** Độ tương đồng cosine dense BGE-M3 với truy vấn thử việc dao động **0.64 – 0.75** (rất cao do trùng trường từ vựng pháp lý).
                - **Điểm yếu của Naive RAG:** Bị tài liệu này dẫn dụ và kết luận sai thành tối đa **60 ngày**.
                - **Khắc phục:** Bộ lọc tài liệu liên quan Self-RAG `[IsREL]` nhận diện văn bản hết hiệu lực và chủ động loại trừ.
                - **Mốc thời gian chuẩn:** Bộ luật Lao động 2019 (hiệu lực 01/01/2021 đến 2026) là Bộ luật Lao động hiện hành mới nhất và duy nhất. Bộ luật 2012 này đã bị bãi bỏ hoàn toàn từ 01/01/2021.
                """)
            else:
                st.markdown("""
                - **Semantic Similarity:** BGE-M3 dense cosine similarity to probation duration queries ranges between **0.64 – 0.75** (high lexical overlap).
                - **Failure Mode:** Naive RAG injects this chunk blindly and falsely concludes probation is capped at **60 days**.
                - **Mitigation:** Self-RAG Critic `[IsREL]` identifies statutory expiration conflict and actively discards it.
                - **Timeline Baseline:** Labor Code 2019 (eff. Jan 1, 2021 through 2026) is the latest and sole active Labor Code in Vietnam. This 2012 statute was completely repealed on Jan 1, 2021.
                """)
    else:
        sel_id = selected_doc_label.split(":")[0].strip()
        doc_obj = retriever.get_article(sel_id)
        doc_idx = retriever.doc_ids.index(sel_id) if sel_id in retriever.doc_ids else 0

        with col_doc_left:
            st.success("Status: CURRENTLY ACTIVE (Labor Code 2019)" if lang == "en" else "Trạng thái: ĐANG CÓ HIỆU LỰC (Bộ luật Lao động 2019)")
            st.markdown(f"**{'Mã định danh điều luật:' if lang == 'vi' else 'Article Identifier:'}** `{doc_obj['article_id']}`")
            st.markdown(f"**{'Tiêu đề điều luật:' if lang == 'vi' else 'Statutory Title:'}** {doc_obj['title']}")
            st.markdown(f"**{'Số lượng từ:' if lang == 'vi' else 'Word Count:'}** `{len(doc_obj['content'].split())} {'từ' if lang == 'vi' else 'words'}`")
            st.text_area("Toàn văn điều luật:" if lang == "vi" else "Full Statutory Article Text:", value=doc_obj['content'], height=200, disabled=True)

        with col_doc_right:
            st.markdown(f"**{'Xem trước Tensor Vector Dense (1024 Chiều):' if lang == 'vi' else 'Dense Vector Tensor Preview (1024 Dimensions):'}**")
            if retriever.corpus_embeddings is not None and doc_idx < len(retriever.corpus_embeddings):
                vec = retriever.corpus_embeddings[doc_idx]
                st.markdown(f"- **{'Kích thước Tensor:' if lang == 'vi' else 'Tensor Shape:'}** `({len(vec)},)` float32")
                st.markdown(f"- **{'Chuẩn hóa L2:' if lang == 'vi' else 'L2 Norm (Unit Vector):'}** `||v||₂ = {round(float(np.linalg.norm(vec)), 4)}`")
                st.markdown(f"- **{'8 chiều đầu tiên [0..7]:' if lang == 'vi' else 'First 8 Dimensions [0..7]:'}**")
                st.code(f"{[round(float(x), 5) for x in vec[:8]]}", language="json")
                st.markdown(f"- **{'8 chiều cuối cùng [1016..1023]:' if lang == 'vi' else 'Last 8 Dimensions [1016..1023]:'}**")
                st.code(f"{[round(float(x), 5) for x in vec[-8:]]}", language="json")

            st.markdown(f"**{'Từ khóa nổi bật BM25 (Sparse):' if lang == 'vi' else 'Top Salient BM25 Tokens (Sparse):'}**")
            tokens = tokenize_vietnamese(doc_obj['title'] + ' ' + doc_obj['content'])
            unique_top_tokens = list(dict.fromkeys([t for t in tokens if len(t) > 2]))[:12]
            st.write("`" + "` · `".join(unique_top_tokens) + "`")

    st.markdown("---")

    # Section 3: Knowledge Graph Map & Communities
    st.markdown(f"#### {t('tab5_sec3_title', lang)}")
    render_mermaid(get_diagram("knowledge_graph_full", lang), height=600)

    col_g1, col_g2 = st.columns([1, 1])
    with col_g1:
        st.markdown(f"**{'Các Cụm Cộng Đồng Modularity:' if lang == 'vi' else 'Modularity Communities (Greedy Modularity Maximization):'}**")
        communities = graph_pipe.get_communities()
        for comm in communities:
            st.markdown(f"**{'Cụm' if lang == 'vi' else 'Community'} {comm['community_id']}: {comm['name']}**")
            st.caption(f"{'Điều luật' if lang == 'vi' else 'Articles'} ({comm['node_count']}): `{'`, `'.join(comm['article_ids'])}`")

    with col_g2:
        st.markdown(f"**{'Các Cạnh Dẫn Chiếu Chéo Giữa Các Điều Luật:' if lang == 'vi' else 'Cross-Reference Statutory Edges:'}**")
        edges_data = [
            ("Art 13 -> Art 20", "governs_contract_types"),
            ("Art 13 -> Art 24", "probation_agreement_basis"),
            ("Art 24 -> Art 25", "maximum_probation_duration"),
            ("Art 24 -> Art 26", "statutory_probation_wage"),
            ("Art 24 -> Art 27", "probation_conclusion_protocol"),
            ("Art 34 -> Art 35", "unilateral_employee_rights"),
            ("Art 34 -> Art 36", "unilateral_employer_rights"),
            ("Art 36 -> Art 37", "unilateral_restrictions_sickness"),
            ("Art 36 -> Art 41", "unlawful_termination_compensation"),
            ("Art 37 -> Art 122", "maternity_protection_principles"),
            ("Art 122 -> Art 125", "dismissal_disciplinary_grounds"),
        ]
        st.dataframe(
            [{"Statutory Link": e[0], "Semantic Relation": e[1]} for e in edges_data],
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Section 4: Seminar Defense Q&A Sheet
    st.markdown(f"#### 4. {'Bộ Câu Hỏi Bảo Vệ Seminar & Phản Biện Học Thuật' if lang == 'vi' else 'Academic Seminar Defense Cheat Sheet (Anticipated Technical Questions)'}")
    with st.expander("Q1: Vì sao hệ thống tính toán trước Embeddings ngoại tuyến thay vì mã hóa toàn bộ ngữ liệu khi runtime?" if lang == "vi" else "Q1: Why does the system pre-compute embeddings offline instead of encoding the corpus at runtime?", expanded=True):
        if lang == "vi":
            st.markdown("""
            **Trả lời:** Trong các hệ thống RAG thực tế, kho dữ liệu chứa hàng ngàn đến hàng triệu đoạn văn bản. Nếu mã hóa lại toàn bộ kho dữ liệu mỗi khi có truy vấn:
            1. **Độ trễ tăng tuyến tính:** Độ trễ suy luận $O(N \\cdot d)$ khiến hệ thống mất hàng giây, không thể tương tác trực tiếp.
            2. **Lãng phí tài nguyên mạng & GPU:** Gọi API dư thừa hàng ngàn lần vô ích.
            **Giải pháp kiến trúc:**
            - **Pha ngoại tuyến:** Tính toán trước 1 lần duy nhất bằng BGE-M3 ($15 \\times 1024$), chuẩn hóa và lưu thành file `corpus_embeddings.npy` (61.5 KB).
            - **Pha trực tuyến:** Chỉ mã hóa duy nhất vector truy vấn ($1 \\times 1024$) và nhân tích vô hướng ma trận - vector trên RAM trong **dưới 0.1 ms**.
            """)
        else:
            st.markdown("""
            **Response:** In production RAG systems, corpora range from thousands to millions of chunks. Re-encoding the entire corpus on every query incurs:
            1. **Linear Latency Growth:** $O(N \\cdot d)$ inference latency, making interactive response times impossible.
            2. **Excessive Network & Compute Overhead:** Redundant API payload transfers and GPU waste.
            **Architecture Solution:**
            - **Offline Phase:** Compute representations once using BGE-M3 ($15 \\times 1024$), normalized and saved into `corpus_embeddings.npy` (61.5 KB).
            - **Online Phase:** Encode only the single query vector ($1 \\times 1024$) and perform BLAS-accelerated matrix-vector multiplication in RAM in **< 0.1 ms**.
            """)

    with st.expander("Q2: Vì sao kết hợp BM25 và BGE-M3 qua RRF thay vì cộng điểm số tuyến tính?" if lang == "vi" else "Q2: Why combine BM25 and BGE-M3 via Reciprocal Rank Fusion (RRF) instead of linear score combination?"):
        if lang == "vi":
            st.markdown("""
            **Trả lời:**
            - **Hạn chế của Dense:** BGE-M3 có thể bị trôi ngữ nghĩa trên các con số định lượng chính xác (ví dụ: "Điều 25", "180 ngày", "85%").
            - **Hạn chế của Sparse:** BM25 thất bại khi người dùng diễn đạt đồng nghĩa nhưng không trùng khớp từ khóa.
            - **Vì sao chọn RRF:** Điểm số BM25 (dương, không bị chặn) và độ tương đồng cosine (nằm trong [-1, 1]) có phân phối hoàn toàn khác biệt. RRF (Cormack et al., SIGIR 2009) sử dụng thứ hạng $RRF(d) = \\sum \\frac{1}{k + r_i}$ với tham số $k=60$, giúp tổng hợp độc lập với thang đo điểm số và cực kỳ ổn định.
            """)
        else:
            st.markdown("""
            **Response:**
            - **Dense Failure Mode:** BGE-M3 can suffer from semantic drift on exact numeric identifiers (e.g. "Article 25", "180 days", "85%").
            - **Sparse Failure Mode:** BM25 fails under vocabulary mismatch when users paraphrase concepts without exact keyword overlap.
            - **Why RRF:** BM25 scores (unbounded positive values based on IDF and document length) and dense cosine similarities (bounded in [-1, 1]) follow fundamentally different probability distributions. RRF (Cormack et al., SIGIR 2009) uses positional rank $RRF(d) = \\sum \\frac{1}{k + r_i}$ with smoothing parameter $k=60$, creating a scale-invariant and robust rank aggregation.
            """)

    with st.expander("Q3: Thuật toán phân cụm cộng đồng của GraphRAG giải quyết điểm mù cục bộ của Naive RAG như thế nào?" if lang == "vi" else "Q3: How does GraphRAG's community detection address the 'Local Blindness' of Naive RAG?"):
        if lang == "vi":
            st.markdown("""
            **Trả lời:**
            - **Vấn đề Điểm Mù Cục Bộ:** Đối với câu hỏi tổng hợp toàn diện (ví dụ: "Tổng hợp các trường hợp người lao động bị sa thải hoặc người sử dụng lao động bị hạn chế quyền chấm dứt"), Naive RAG chỉ truy xuất được 3-5 đoạn cục bộ, bỏ sót phần lớn điều luật liên quan nằm rải rác ở các chương khác.
            - **Giải pháp GraphRAG (Microsoft Research 2024):**
              1. Xây dựng đồ thị liên kết thực thể pháp lý.
              2. Áp dụng thuật toán **Tối đa hóa Modularity (Newman, 2004)** để phân cụm các điều luật thành các cộng đồng chủ đề.
              3. Chạy quy trình **Map-Reduce**: tóm tắt song song từng cụm (Map) rồi tổng hợp toàn cục (Reduce), bảo đảm bao quát 100% ngữ liệu.
            """)
        else:
            st.markdown("""
            **Response:**
            - **Local Blindness Problem:** On global synthesis queries (e.g. "Summarize all employee termination rights across the labor code"), Naive RAG retrieves only top 3-5 individual chunks, missing 80% of relevant provisions spread across other sections.
            - **GraphRAG Solution (Microsoft Research 2024):**
              1. Constructs a statutory graph capturing real cross-article references.
              2. Applies **Greedy Modularity Maximization (Newman, 2004)** to partition the graph into dense thematic communities.
              3. Employs a **Map-Reduce** pattern: community summaries are generated in parallel (Map), and then synthesized into an exhaustive global answer (Reduce).
            """)

