---
title: CO5151 RAG Paradigm Inspector
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# The RAG Paradigm Inspector · CO5151 Seminar Arena

Hệ thống thực nghiệm đối chứng các cơ chế RAG khoa học phục vụ môn học **CO5151 (Advanced Agentic AI)** - Giảng viên: TS. Lê Xuân Bách (ĐHBK ĐHQG-HCM).

## 4 Trục Thực Nghiệm Khoa Học
1. **🛡️ When Retrieval Hurts:** Đo lường sự suy giảm độ chính xác của Naive RAG khi gặp văn bản hết hiệu lực / nhiễu (Lewis et al., 2020) so với cơ chế phản biện Self-RAG.
2. **🕸️ GraphRAG vs. Global Synthesis:** Khắc phục tính mù cục bộ (Local Blindness) của Vector Search bằng đồ thị tri thức và tóm tắt cộng đồng Map-Reduce (Edge et al., Microsoft 2024).
3. **🔍 Self-RAG Reflection Inspector:** Giải phẫu 4 token phản biện `[Retrieve]`, `[IsREL]`, `[IsSUP]`, `[IsUSE]` (Asai et al., ICLR 2024).
4. **⚡ Active Retrieval (FLARE):** Truy xuất chủ động theo độ bất định của Token trong quá trình sinh nháp (Jiang et al., EMNLP 2023).

---

## Khởi Chạy Nhanh

### 1. Chạy trực tiếp (Local Python)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Chạy qua Docker Compose
```bash
docker compose up --build -d
```
Truy cập ứng dụng tại: `http://localhost:8501`.

### 3. Deploy lên Hugging Face Spaces
Tạo một Space mới trên Hugging Face (chọn SDK: Streamlit), sau đó push toàn bộ thư mục này lên repository của Space:
```bash
git remote add space https://huggingface.co/spaces/<username>/<space-name>
git push space main
```
Hệ thống sẽ tự động build và chạy trực tiếp trên cloud miễn phí!
