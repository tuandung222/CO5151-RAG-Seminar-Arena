# 🎨 UI/UX Audit Report — Demo Seminar App

**App**: Streamlit, 2037 dòng, 5 tabs  
**Góc nhìn**: Người dùng lần đầu mở (viewer/giám khảo/thầy), KHÔNG phải developer

---

## 🔴 Vấn đề #1: Text Wall Trước Action — "Cuộn mãi không thấy nút bấm"

**Nghiêm trọng nhất.** Khi mở Tab 1, user phải cuộn qua:
1. Header + subtitle
2. Master Guide expander (chứa 2 bảng so sánh + 2 st.info + 1 st.caption)
3. Architecture expander (chứa 2 sub-tabs với Mermaid diagrams)
4. Context Card (16 dòng HTML dày đặc)
5. Radio buttons chọn distractor mode
6. **Rồi mới thấy query box + nút Run**

> **Quy tắc UX**: Action button phải nằm **above the fold** (trong viewport đầu tiên). User đến demo để **xem chạy**, không để đọc bài giảng.

### 💡 Đề xuất:
- Di chuyển `render_query_console()` lên **ngay sau header**, trước tất cả expanders
- Đặt Context Card + Architecture vào expanders **dưới** kết quả
- Pattern: **Query → Result → Giải thích** (thay vì Giải thích → Giải thích → Query → Result)

---

## 🔴 Vấn đề #2: Master Guide Expander — Bảng 5 cột quá dày

Bảng so sánh 5 paradigms trong `master_guide_expander`:
```
| Cơ Chế RAG | Cơ Chế Vận Hành | Công Thức | Điểm Yếu | Khi Nào Triển Khai |
```

- **5 cột × 5 hàng** = 25 ô, mỗi ô chứa 2-3 dòng text
- Trên mobile / HF Space (hẹp): **bảng bị tràn ngang**, phải cuộn phải
- Cột "Công Thức Toán" có LaTeX phức tạp — render rối trên Streamlit markdown
- **Nằm TRÊN tab area** → mọi tab đều phải cuộn qua cái bảng này

### 💡 Đề xuất:
- Thu gọn thành 3 cột: `Paradigm | Core Mechanism | When to Use`
- Hoặc chuyển thành cards/pills nằm ngang thay vì bảng
- Di chuyển vào Tab 5 (Data Explorer) — không cần ở header

---

## 🟡 Vấn đề #3: Callout Chồng Callout

Trong master_guide_expander (EN version):
```
st.info(...)           ← Statutory Timeline
st.markdown(table)     ← 5-column comparison matrix  
st.caption(...)        ← Author note
st.info(...)           ← Agentic retrieval table
```

4 blocks liên tiếp, **2 st.info liền nhau** → visual noise. User không biết cái nào quan trọng.

### 💡 Đề xuất:
- Giữ **1 callout** duy nhất cho thông tin quan trọng nhất
- Merge statutory timeline + author note thành 1 caption nhỏ

---

## 🟡 Vấn đề #4: Context Card HTML Quá Dày Đặc

Context Card (Tab 1, line 793-810) chứa:
- Tình huống doanh nghiệp (4 dòng)
- Mục tiêu thực nghiệm (5 dòng)  
- Mốc thời gian pháp lý (3 dòng)
- Tất cả trong **1 khối HTML duy nhất** → text wall

### 💡 Đề xuất:
- Tách thành 2 cột: `col1: Tình huống` | `col2: Mục tiêu`
- Mốc thời gian → caption nhỏ bên dưới
- Hoặc dùng `st.expander("📌 Bối cảnh thực tế", expanded=False)`

---

## 🟡 Vấn đề #5: Tab 3 — 41 Callouts, "Kinh hoàng thông tin"

Tab 3 (Self-RAG Inspector) có **41 st.markdown/info/caption calls** — mật độ cao nhất. Bao gồm:
- Mermaid diagram
- 16-dòng HTML context card
- Radio buttons
- Query console
- Result display
- Deep-dive expander (expanded=True!)
- Mathematical formulas
- Bar chart
- LaTeX disclaimers

> Khi `expanded=True` cho deep-dive, user bị **tường text** tấn công ngay lập tức.

### 💡 Đề xuất:
- Đổi deep-dive `expanded=True` → `expanded=False`
- Kết quả demo đủ rồi, ai muốn đọc sâu thì tự mở

---

## 🟡 Vấn đề #6: 26× `unsafe_allow_html=True`

Mỗi lần dùng `unsafe_allow_html`, Streamlit mất khả năng responsive tự động. CSS custom override khá nhiều (250 dòng), tạo ra:
- **Inconsistency** giữa native Streamlit widgets và custom HTML cards
- **Font weight 800 cho button** → quá bold, trông "gào" vào mặt user
- **min-height: 52px cho input** → hơi cao, chiếm nhiều không gian dọc

### 💡 Đề xuất:
- Giảm button font-weight: 800 → 600
- Giảm input min-height: 52px → 44px
- Dùng native `st.container()` + `border=True` thay custom HTML cards nếu có thể

---

## 🟢 Vấn đề #7: Mermaid Diagram Rendering — OK nhưng hơi cao

`render_mermaid()` dùng iframe với fixed height (220-600px). Trên HF Space narrow viewport:
- Một số diagram bị crop hoặc quá nhỏ
- Zoom button (+/-) hữu ích nhưng icon nhỏ

### Tốt rồi: Pop-out viewer button ✅, auto-resize ✅

---

## 🟢 Vấn đề #8: Sidebar — Quá Nhiều Provider Options

Sidebar có:
- Language toggle
- 5 provider radio buttons (HF Inference API, OpenAI, Ollama, vLLM, HF Inference Endpoint)
- Mỗi provider có 2-3 text_input (URL, API key, model)
- Token input
- 4 "Quick Test" buttons cho mỗi tab

Đối với seminar 30 phút: **chỉ cần 1 provider** (HF hoặc đã chọn sẵn). Phần còn lại gây rối.

### 💡 Đề xuất:
- Chỉ hiện provider đang active, ẩn phần còn lại
- Quick Test buttons → di chuyển vào tabs tương ứng

---

## 🟢 Vấn đề #9: EN/VI Mix Trong English Mode

Line 721 (EN table):
```
| **5. FLARE** | Sinh nháp dự phóng + Kích hoạt truy xuất theo verbalized confidence | ...
```
→ **Tiếng Việt lọt vào English mode!** Cần sửa thành English.

---

## 🟢 Vấn đề #10: Tab Names Dài Quá

Tab names hiện tại (VI):
```
["⚔️ Khi Truy Xuất Gây Hại (Self-RAG vs Naive)", "🌐 Tổng Hợp Toàn Cục (GraphRAG)", 
 "🔬 Bóc Tách Reflection Tokens (Self-RAG)", "⚡ Truy Xuất Chủ Động (FLARE)", 
 "📊 Hạ Tầng Dữ Liệu & Vector Space"]
```
→ Quá dài, bị cắt xén trên mobile, tab bar cuộn ngang

### 💡 Đề xuất:
Rút ngắn:
```
["⚔️ Self-RAG vs Naive", "🌐 GraphRAG", "🔬 Reflection Tokens", "⚡ FLARE", "📊 Data Explorer"]
```

---

## Tổng Kết Ưu Tiên Sửa

| Ưu tiên | Vấn đề | Ảnh hưởng | Effort |
|:---:|:---|:---:|:---:|
| 🔴 | Query box bị chìm dưới text wall | Cao | Trung bình |
| 🔴 | Bảng 5 cột tràn ngang | Cao | Thấp |
| 🟡 | Callout chồng callout | Trung bình | Thấp |
| 🟡 | Context card quá dày | Trung bình | Thấp |
| 🟡 | Tab 3 deep-dive expanded=True | Trung bình | 1 dòng |
| 🟡 | CSS quá aggressive | Thấp | Thấp |
| 🟢 | VI lọt vào EN table | Thấp | 1 dòng |
| 🟢 | Tab names quá dài | Thấp | Thấp |
