# 🔬 Kết Quả Xác Minh Báo Cáo Agent Sol

**Ngày xác minh**: 2026-09-29 | **Xác minh bởi**: Antigravity (Claude Opus 4.6)
**Phương pháp**: Đọc trực tiếp toàn bộ mã nguồn + git history + cached data + 2 subagent đối chiếu độc lập

---

## Tóm Tắt

> **Report của Agent Sol về cơ bản ĐÚNG trên ~90% các claim.** Đây là một audit chất lượng cao, trung thực, và phần lớn các phát hiện đã được xác minh bằng chứng cứ cụ thể trong mã nguồn. Dưới đây là phán quyết từng mục.

---

## A. BẢO MẬT — Phán Quyết Từng Mục

### A1. Server `HF_TOKEN` được điền sẵn vào ô input → ✅ **ĐÚNG (Đã vá)**

Mã nguồn gốc tại `app.py:508-511`:
```python
default_hf_token = st.session_state.get("user_hf_token", HF_TOKEN)
user_hf_token = st.text_input(..., value=default_hf_token, type="password")
```
Token server nằm trong DOM websocket. **Đã vá tại commit `d93d7b6`** — ô input hiện để trống mặc định.

**Khuyến nghị**: ✅ Đã xử lý. Nhưng **revoke token cũ** nếu Space từng chạy bản chưa vá.

---

### A2. API key `sk-x8qNKU7OZAz70PL7Urcnmg` hardcode trong git history → ✅ **ĐÚNG**

Xác minh: commit `8723724` và `d5cbbeb` chứa:
```python
cur_api_key = st.text_input("API Key:", value="sk-x8qNKU7OZAz70PL7Urcnmg", type="password")
```
Đã xóa khỏi HEAD tại commit `03572dd`, nhưng **vẫn nằm nguyên trong git history**. Repo đã push lên GitHub public.

**Khuyến nghị**: 🔴 **KHẨN CẤP** — Revoke key `sk-x8qNKU7OZAz70PL7Urcnmg` trên gateway `ai-gateway01.qualgo.ai` ngay lập tức. Cân nhắc `git filter-repo` để scrub history.

---

### A3. SSRF: Server gửi key tới URL do visitor kiểm soát → ✅ **ĐÚNG**

Xác minh tại `app.py:539-575`:
1. `cur_base_url` = `st.text_input(...)` — visitor chọn URL tùy ý.
2. Nếu visitor để trống API Key, fallback `cur_api_key = env_openai_key` (line 553).
3. `UnifiedLLM(base_url=cur_base_url, api_key=cur_api_key)` → Server gửi `Authorization: Bearer <server_key>` tới URL của attacker.

Ollama: `cur_api_key = "EMPTY"` nên không leak key, nhưng vẫn có SSRF vector qua `urllib.request.urlopen`.

**Khuyến nghị**: 🔴 **NGHIÊM TRỌNG** — Whitelist/validate `base_url`, hoặc không gửi server key khi user thay đổi URL.

---

### A4. Token cross-contamination qua singleton retriever → ✅ **ĐÚNG**

Xác minh:
- `app.py:435`: `@st.cache_resource` → singleton toàn cục.
- `app.py:521`, `app.py:722`: `retriever.update_api_key(token)` → ghi đè `self.api_key` và `self.hf_client` trên singleton.
- `src/retriever.py:77-81`: `update_api_key()` thay đổi state trực tiếp.

User A nhập token → User B dùng token đó cho embedding query.

**Khuyến nghị**: 🔴 Tạo retriever client riêng per-session, hoặc truyền token trực tiếp vào mỗi lệnh gọi thay vì mutate singleton.

---

### A5. CORS/XSRF tắt → ✅ **ĐÚNG**

Xác minh: `.streamlit/config.toml:4-5` và `Dockerfile:29` đều set:
```
enableCORS = false
enableXsrfProtection = false
```

**Khuyến nghị**: 🟡 Cần thiết cho HF Space reverse proxy. Chấp nhận được cho demo, nhưng nên bật lại XSRF cho production.

---

### A6. Token trong git push URL → ✅ **ĐÚNG**

Xác minh: `.github/workflows/sync_to_hf.yml:23`:
```bash
git push --force https://tuandunghcmut:$HF_TOKEN@huggingface.co/...
```
GitHub Actions mask secrets trong logs, nhưng nên dùng credential helper hoặc `actions/huggingface/push` thay thế.

**Khuyến nghị**: 🟡 Rủi ro thấp do GitHub masking, nhưng best practice nên dùng phương thức an toàn hơn.

---

### A7. `unsafe_allow_html=True` → ✅ **ĐÚNG, rủi ro thấp**

Xác minh: 26 chỗ. LLM output đi qua `st.success()` / `st.markdown()` không inject HTML trực tiếp từ user input. Rủi ro XSS thấp nhưng cần giám sát.

---

## B. TÍNH TRUNG THỰC — Phán Quyết Từng Mục

### B1. Cached Gold Results là viết tay → ⚠️ **MỘT PHẦN ĐÚNG**

Xác minh từ `data/cached_benchmark_results.json`:
- `tab1.case_0.pure.latency_ms: 1420` — con số tròn đáng ngờ.
- `tab1.case_0.self.latency_ms: 3650` — tròn.
- `metadata.audit_status: "CONTROLLED BENCHMARK RUN"` — thay đổi từ "100% VERIFIED GOLD RUN".
- Commit `cb54625`: *"sanitize cached gold results to 100% legal accuracy"*.

**Tuy nhiên**: Nội dung answer trong cached không hoàn toàn "viết tay" — chúng có ngữ pháp tự nhiên của LLM output, và `tab4.case_2.threshold_0_5` chứa trace steps hợp lý. Có khả năng đây là output LLM được **chỉnh sửa tay** (hand-curated) chứ không phải hoàn toàn bịa. Latency thì gần như chắc chắn là **ước lượng gán tay**.

**Phán quyết**: Không phải "raw machine output" như UI gợi ý. Nên ghi rõ là "curated illustration" hoặc "edited from actual run".

---

### B2. Slider mô phỏng, không chạy thật → ✅ **ĐÚNG**

**Tab 3 `tau`** — Xác minh tại `app.py:1470-1480`:
```python
gate_prob = 0.948  # HARDCODED
if tau_threshold > gate_prob:
    tab3_res_data["retrieve_decision"] = {"token": "NO_RETRIEVAL", ...}
else:
    tab3_res_data["retrieve_decision"] = {"token": "NEED_RETRIEVAL", ...}
```
Chỉ thay đổi chuỗi `retrieve_decision`. Câu trả lời (đã có retrieval) giữ nguyên. Nếu tau > 0.948 → UI ghi "ĐÓNG CỔNG" nhưng đáp án vẫn chứa trích dẫn điều luật.

**Tab 4 `theta`** — Xác minh từ cached data:
| Theta | Step 3 confidence | Status |
|-------|:--:|---|
| 0.3 | 0.91 | CONFIDENT_NO_RETRIEVAL |
| 0.5 | 0.91 | CONFIDENT_NO_RETRIEVAL |
| 0.7 | **0.68** | TRIGGERED_RETRIEVAL |

Cùng 1 câu + cùng 1 LLM + temperature 0 → confidence **không thể thay đổi** từ 0.91 sang 0.68. Đây là 3 kịch bản viết tay để minh họa "0/1/2 lượt tra cứu".

**Phán quyết**: ✅ Đúng 100%. Slider cached mode là dàn dựng.

---

### B3. So sánh Naive vs Self-RAG không công bằng → ✅ **ĐÚNG**

Xác minh trực tiếp tại `src/naive_rag.py:53-57`:
```python
if distractor_mode == "only_distractor":
    passages_to_use.append(distractor_item)
    unrelated = [p for p in raw_retrieved if p["article_id"] != "Dieu_25"]  # ← LOẠI ĐIỀU 25
    passages_to_use.extend(unrelated[: top_k - 1])
```

Và `src/self_rag.py:164-176`:
```python
raw_retrieved = self.retriever.search_hybrid_rrf(query, top_k=top_k)
if distractor:
    candidates.append({...distractor...})
    candidates.extend(raw_retrieved[: top_k - 1])  # ← KHÔNG LOẠI ĐIỀU 25
```

**Naive RAG bị cắt Điều 25 thủ công; Self-RAG vẫn nhìn thấy Điều 25.** Thắng thua do thiết kế input, không phải do cơ chế reflection.

Ngoài ra: distractor được chèn cứng ở rank 1 với `rrf_score: 0.999` (line 51). Retriever không hề truy xuất nó — nó được chèn bởi code.

**Phán quyết**: ✅ Đây là vấn đề nghiêm trọng nhất cho tính trung thực của demo. Thông điệp "Naive RAG bị context poisoning" chỉ đúng vì code chủ ý loại Điều 25 khỏi Naive RAG.

---

### B4. [IsSUP] và [IsUSE] là hardcoded heuristic → ✅ **ĐÚNG**

Xác minh tại `src/self_rag.py:207-212`:
```python
is_supported = any(
    p.get("article_id", "") in gen_res["text"] or any(word in gen_res["text"]
    for word in ["180 ngày", "Điều 25", "85%", "thử việc", "sa thải"])
    for p in valid_passages
)
sup_token = "FULLY_SUPPORTED" if is_supported else "PARTIALLY_SUPPORTED"
use_score = 5 if is_supported else 3
```

- `"thử việc"` xuất hiện trong gần mọi câu trả lời pháp luật lao động → kết quả gần như luôn `FULLY_SUPPORTED` score=5.
- `has_180_days`: dòng 202 có `or "180" in gen_res["text"]` → khớp cả "1800", "1801", etc.
- Fallback `IsREL` khi JSON parse lỗi: mặc định `"RELEVANT"` (`self_rag.py:119`).

**Phán quyết**: ✅ Đúng. Đây là string matching, không phải LLM verification thật. UI trình bày nó như một quá trình reflection nghiêm ngặt, nhưng code chỉ kiểm tra keyword.

---

### B5. GraphRAG không phải GraphRAG thật → ⚠️ **MỘT PHẦN ĐÚNG**

Xác minh tại `src/graph_rag.py`:
- Đồ thị: Nodes = 15 điều luật từ corpus (line 25-30). Edges = 15 cạnh **viết tay** (line 33-49). Đúng là hand-crafted.
- `get_community_name()` (line 59-65): Gán tên cụm bằng if/else theo ID. Đúng.
- `run_graph_rag()`: Map-reduce trên **toàn bộ communities** (tức toàn bộ corpus), không phụ thuộc query. Đúng.
- Community detection: `nx.community.greedy_modularity_communities()` — đúng không phải Leiden nhưng vẫn là thuật toán community detection hợp lệ.

**Tuy nhiên**: Agent Sol nói "không phải GraphRAG" hơi quá tuyệt đối. Kiến trúc map-reduce trên graph communities **đúng là pattern của GraphRAG paper** (Edge et al., 2024). Vấn đề là:
1. Knowledge graph được xây dựng thủ công (không có entity extraction tự động).
2. Corpus quá nhỏ (15 docs) nên "toàn cục" = chỉ 15 điều, nằm gọn trong 1 context window.
3. Thiếu baseline "stuff toàn bộ corpus vào 1 prompt" để so sánh công bằng.

**Phán quyết**: Nên gọi đây là "GraphRAG-style demo on hand-crafted knowledge graph" thay vì "GraphRAG".

---

### B6. FLARE không phải FLARE thật → ✅ **ĐÚNG**

Xác minh tại `src/flare_rag.py:39-63`:
```python
check_prompt = """Đánh giá xem câu văn sau... Trả về JSON:
{
  "confidence_score": số từ 0.0 đến 1.0 (trên 0.8 là rất chắc chắn, dưới 0.8 là cần tra cứu),
  ...
}"""
...
if req_retrieval and confidence < theta:  # ← theta, nhưng prompt nói 0.8
```

- FLARE gốc (Jiang et al., EMNLP 2023) dùng **token logprobs** để đo uncertainty. Code dùng **LLM tự khai confidence qua JSON** ("verbalized confidence").
- Prompt nói ngưỡng 0.8, nhưng code dùng theta (mặc định 0.5) — hai ngưỡng mâu thuẫn.
- UI hiển thị công thức $\min_t P(w_t) < \theta$ (log probability) nhưng code không truy cập logprobs.

**Phán quyết**: ✅ Nên đổi tên thành "FLARE-style verbalized confidence" và bỏ công thức logprob khỏi UI.

---

### B7. Self-RAG là in-context prompt, không phải fine-tuned model → ✅ **ĐÚNG**

`src/self_rag.py` dùng LLM generate JSON cho từng reflection token. README có ghi "surrogate". Công thức `Score = LLM + w_rel·logP(IsREL)…` trong UI lấy từ paper gốc (Asai et al., ICLR 2024) nhưng code không implement scoring function đó.

**Phán quyết**: ✅ Nên nói rõ "Self-RAG emulated via in-context prompting (surrogate, not fine-tuned critic)".

---

### B8. Kiểm định đáp án chỉ bằng string matching → ✅ **ĐÚNG**

Xác minh tại `src/naive_rag.py:85-99`:
```python
has_60_days = "60 ngày" in answer_text or "không quá 60" in answer_text.lower()
has_180_days = "180 ngày" in answer_text or "không quá 180" in answer_text.lower()
```

Câu "không phải 60 ngày mà thực tế là 180 ngày" sẽ khớp cả hai → `CONFUSED_CONFLICT`, dù đáp án đúng.

**Phán quyết**: ✅ Đúng. Chỉ hợp lý cho Case 1.

---

## C. DỮ LIỆU PHÁP LÝ

### C1. Distractor 60 ngày cho CEO có đúng không?
Agent Sol nói đúng: Nghị định 44/2013 (hướng dẫn BLLĐ 2012) cho phép tối đa 180 ngày cho người quản lý doanh nghiệp ngay cả dưới luật cũ. Test case giả định "BLLĐ 2012 chỉ cho 60 ngày cho CEO" là **oversimplified**. Nên nói rõ đây là kịch bản dựng (synthetic scenario).

### C2. Corpus rút gọn → ✅ **ĐÚNG**
Agent Sol đúng rằng corpus được diễn đạt lại, không phải nguyên văn. Ví dụ mâu thuẫn nội bộ (Điều 34 vs Điều 46) cần đối chiếu lại.

### C3. "Không có bộ luật mới hơn… đến 2026"
Agent Sol đúng là đây là khẳng định thời sự khó xác minh. Nên phát biểu thận trọng hơn.

---

## D. LỖI KỸ THUẬT — Tóm Tắt

| Mục | Agent Sol đúng? | Ghi chú |
|-----|:--:|---|
| Dense fallback im lặng về BM25 | ✅ | UI vẫn ghi "BGE-M3 + BM25 (RRF)" |
| Không assert embeddings khớp corpus | ✅ | Sửa corpus sẽ lệch chỉ số |
| Router Option B payload sai | ✅ | `source_sentence/sentences` cho bge-m3 |
| `DEFAULT_BASE_URL` rối cấu hình | ✅ | Nhỏ |
| Session state giữ kết quả cũ khi đổi model | ✅ | |
| README port 8501 vs Dockerfile 7860 | ✅ | |

---

## E. ĐÁNH GIÁ TỔNG THỂ VỀ REPORT CỦA AGENT SOL

| Tiêu chí | Điểm |
|---|:--:|
| **Độ chính xác claim** | 9/10 |
| **Độ sâu phân tích** | 9/10 |
| **Công bằng / không thiên vị** | 8/10 |
| **Khả thi của đề xuất** | 8/10 |

### Những chỗ Agent Sol hơi quá tay:
1. **B5**: Nói "không phải GraphRAG" quá tuyệt đối. Pattern map-reduce trên graph communities đúng là theo paper Edge et al. Vấn đề là graph hand-crafted + corpus quá nhỏ.
2. **B1**: Nói "viết tay" (hand-written) — có thể là "hand-curated from actual LLM runs" chính xác hơn. Latency thì gần chắc chắn là ước lượng.

### Những chỗ Agent Sol phát hiện xuất sắc:
1. **B3** (unfair comparison) — đây là finding quan trọng nhất. Hard-exclude `Dieu_25` khỏi Naive RAG làm cho so sánh vô nghĩa.
2. **A3** (SSRF/credential forwarding) — lỗ hổng thực sự nguy hiểm.
3. **A2** (sk- key in git history) — cần hành động ngay.
4. **B2** (FLARE theta cached fakery) — confidence 0.91→0.68 cho cùng 1 câu là bằng chứng không thể chối cãi.

---

## F. ƯU TIÊN HÀNH ĐỘNG (Thứ Tự Khẩn Cấp)

### 🔴 Khẩn cấp (Ngay bây giờ)
1. **Revoke `sk-x8qNKU7OZAz70PL7Urcnmg`** trên `ai-gateway01.qualgo.ai`
2. **Revoke và rotate HF Token cũ** tại https://huggingface.co/settings/tokens
3. Cân nhắc `git filter-repo` để scrub secrets khỏi history, hoặc tạo repo mới

### 🟠 Trước seminar (Tính trung thực)
4. **Sửa B3**: Cả Naive RAG và Self-RAG phải nhận **cùng danh sách candidate**. Bỏ `if p["article_id"] != "Dieu_25"`.
5. **Ghi rõ nhãn**: Cached results = "Curated Illustration (Hand-edited from LLM output)" thay vì "Verified Cluster Evaluation".
6. **Sửa slider cached**: Ghi rõ slider ở chế độ cached chỉ minh họa, không phải kết quả chạy thật.
7. **Đổi tên**: "GraphRAG-style Map-Reduce Demo", "FLARE-style Verbalized Confidence", "Self-RAG Surrogate (In-Context Prompting)".
8. Bỏ công thức $\min_t P(w_t) < \theta$ khỏi phần giải thích FLARE.
9. Bỏ/sửa claim "tiết kiệm 60-80%" (chỉ dựa trên 1 câu hỏi, 3 câu văn).

### 🟡 Nên sửa (Chất lượng kỹ thuật)
10. **Sửa A3**: Validate/whitelist `base_url`
11. **Sửa A4**: Không mutate singleton retriever
12. **Sửa B4**: Thay keyword matching bằng LLM-judge cho `[IsSUP]`/`[IsUSE]`
13. Đối chiếu corpus với nguyên văn BLLĐ 2019
