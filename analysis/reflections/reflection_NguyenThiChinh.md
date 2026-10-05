# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Thị Chinh  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 05/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong lecture vào code bạn vừa viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng mô hình embedding tính Cosine Similarity giữa các câu liền kề với ngưỡng threshold (mặc định 0.85). Khi similarity giảm đột ngột dưới ngưỡng, hệ thống tự động tách chunk mới. So với basic chunking (cắt cố định theo số ký tự/token dễ làm đứt đoạn câu), semantic chunking gom 3-5 câu cùng một ý hoàn chỉnh, giúp context không bị vỡ vụn và bảo toàn ngữ nghĩa khi sinh vector embedding. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF kết hợp điểm xếp hạng lexical (BM25 với bộ tách từ tiếng Việt underthesea, bắt chính xác mã số, từ khóa chuyên ngành như 'PVI', '12 ngày', '120 ngày', 'MFA') và semantic (Dense Search vector BAAI/bge-m3 trong Qdrant). Công thức $RRF(d) = \sum \frac{1}{k + rank_i(d)}$ với $k=60$ giúp trung hòa hai thang điểm hoàn toàn khác nhau mà không cần chuẩn hóa scale min-max, mang lại kết quả retrieval toàn diện cả về từ khóa lẫn ngữ nghĩa. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Áp dụng mô hình `BAAI/bge-reranker-v2-m3` với cơ chế full self-attention giữa cặp `(query, document)`. Reranker đánh giá lại top 20 candidate từ bước Hybrid search và lọc lấy top 3 (RERANK_TOP_K=3) liên quan nhất. Latency chỉ tăng thêm khoảng 45ms nhưng loại bỏ triệt để các chunk "nhiễm từ khóa" (lexical overlap nhưng sai ngữ cảnh), giúp cải thiện đáng kể Context Precision cho LLM generation. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Đánh giá tự động toàn diện hệ thống RAG qua 4 chỉ số cốt lõi: **Faithfulness** (tỷ lệ câu trả lời có bằng chứng từ context, chống hallucination), **Answer Relevancy** (mức độ câu trả lời đi thẳng vào trọng tâm câu hỏi người dùng), **Context Precision** (tỷ lệ chunk liên quan nằm ở các vị trí xếp hạng đầu), và **Context Recall** (mức độ thông tin Ground Truth được context tìm thấy bao phủ). Kết hợp hàm `failure_analysis()` phân loại lỗi theo Diagnostic Tree. |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Triển khai kỹ thuật Anthropic Contextual Retrieval. Để tránh chunk bị cô lập mất ngữ cảnh cha (document/chương/mục), chunk được bổ sung một câu định vị vị trí và tóm lược chủ đề. Đặc biệt phương thức `_enrich_single_call()` gom cả 4 tác vụ (Summary, HyQA 3 câu hỏi giả định, Contextual Prepend, Metadata trích xuất) vào 1 LLM API call duy nhất với JSON mode, giảm tới 75% chi phí API latency/token và tăng đáng kể khả năng tìm kiếm từ khóa nhờ các câu hỏi HyQA. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật 1: FlagReranker không tương thích với transformers >= 5.0**
  - **Exact error message:**
    ```text
    TypeError: XLMRobertaTokenizer.__init__() got an unexpected keyword argument 'extra_ids'
    (hoặc Crash khi khởi tạo FlagEmbedding.FlagReranker trên môi trường transformers mới)
    ```
  - **Nguyên nhân gốc rễ & Cách debug:** Thư viện `FlagEmbedding` gọi lớp tokenizer cũ không tương thích với phiên bản transformers mới cài đặt trong môi trường ảo `.venv`. 
  - **Cách khắc phục:** Đổi sang dùng `sentence_transformers.CrossEncoder("BAAI/bge-reranker-v2-m3")` như ghi chú trong bài lab. Thư viện `sentence-transformers` tương thích chuẩn với PyTorch và Hugging Face Hub, load model mượt mà và chạy dự đoán điểm `predict(pairs)` ổn định.

- **Lỗi kỹ thuật 2: Deadlock / Timeout khi chạy RAGAS evaluation trên Python 3.13**
  - **Exact error message:**
    ```text
    pytest error: Command '['.../python', '-m', 'pytest', 'tests/', ...]' timed out after 120 seconds
    (Quá trình gọi ragas.evaluate() bị treo vô hạn trong select_kqueue / asyncio loop)
    ```
  - **Nguyên nhân gốc rễ & Cách debug:** Thư viện Ragas hàm `evaluate()` mặc định sử dụng cơ chế multiprocessing (loky/concurrent executors) kết hợp lồng ghép event loop asyncio. Trên macOS ARM64 với Python 3.13, việc gọi `asyncio.run()` bên trong tiến trình con gây deadlock semaphore POSIX (`/loky-...`).
  - **Cách khắc phục:** Tối ưu hóa trực tiếp trong [src/m4_eval.py](file:///Users/hihi/Documents/aitc/K4-Track3B-DAY18-NguyenThiChinh-2A202602876-ProductionRAG/src/m4_eval.py): Khởi tạo `ChatOpenAI(model="gpt-4o-mini", request_timeout=30)` và `OpenAIEmbeddings`, bọc qua `LangchainLLMWrapper`, sau đó khởi tạo trực tiếp từng metric (`faithfulness`, `context_precision`, `context_recall`, `answer_relevancy`). Sử dụng `asyncio.Semaphore(4)` điều phối đánh giá bất đồng bộ theo từng câu hỏi, xử lý ngoại lệ graceful fallback `_clean()` khi timeout. Kết quả: toàn bộ 37/37 unit tests pass 100% trong thời gian tối ưu.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - *Kiến thức:* Cách cân bằng giữa retrieval latency và retrieval recall trong Production RAG khi số lượng document tăng lên hàng nghìn trang.
  - *Cách khắc phục:* Đọc kỹ tài liệu Anthropic Contextual Retrieval và LangChain Production Guides; áp dụng phân tầng: Hierarchical Chunking (Small-to-Big Retrieval) chỉ tìm kiếm trên child chunks (256 chars) sau đó mở rộng trả về parent chunk (2048 chars) cho LLM, kết hợp Cross-Encoder reranking top-k vừa phải (top 20 → top 3).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý AI Hỏi Đáp Quy định & Hợp đồng Pháp lý Doanh nghiệp (Enterprise Legal & Policy Q&A Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Basic RAG đơn giản với LangChain RecursiveCharacterTextSplitter (chunk_size=1000, overlap=100) và Dense Vector Search duy nhất bằng ChromaDB.
- **Vấn đề / Bottlenecks đang gặp:**
  - *Retrieval precision thấp:* Các tài liệu điều khoản pháp lý thường có các điều kiện ngoại lệ nằm rải rác; chunking cố định làm đứt gãy quan hệ logic giữa điều khoản chính và điều khoản loại trừ.
  - *Vấn đề phiên bản văn bản:* Không phân biệt được quy chế cũ (hết hiệu lực) và quy chế mới (sửa đổi, bổ sung) dẫn đến trả lời sai thông tin hiện hành.
  - *Thiếu bộ đánh giá tự động:* Không có benchmark định lượng, chỉ kiểm thử thủ công qua vài câu hỏi ngẫu nhiên.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Áp dụng **Hierarchical Chunking (Small-to-Big)** kết hợp Structure-aware. Child chunk (256-512 chars) cho phép biểu diễn vector sắc nét, khi retrieve match sẽ lấy parent chunk (Điều/Khoản hoàn chỉnh 2048 chars) đưa vào context của LLM.
2. **Search retrieval:** Triển khai **Hybrid Search (BM25 + Dense BAAI/bge-m3 qua Qdrant)** kết hợp **Reciprocal Rank Fusion (RRF)**. BM25 với tokenizer tiếng Việt giúp bắt chính xác các số hiệu văn bản (Nghị định 13/2023, Điều 12, Thông tư...), Dense search nắm bắt ngữ nghĩa câu hỏi tự nhiên.
3. **Reranking:** Sử dụng `BAAI/bge-reranker-v2-m3` rerank từ top-20 xuống top-3. Đây là bước then chốt loại bỏ các điều khoản tương đồng về câu chữ nhưng khác phạm vi áp dụng.
4. **Enrichment:** Sử dụng kỹ thuật **Contextual Prepend** và **Auto Metadata Extraction (hiệu lực, ngày ban hành, phiên bản v2023/v2024)**. Ở bước truy vấn, áp dụng metadata filter để chỉ lấy các văn bản còn hiệu lực hiện hành.
5. **Evaluation:** Thiết lập bộ test set 50 câu hỏi nghiệp vụ chuẩn hóa, chạy đánh giá định kỳ bằng **RAGAS 4 metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall)** trong CI/CD pipeline.

#### 3. Timeline triển khai (2 tuần)
- **Tuần 1:**
  - Ngày 1-2: Refactor module Document Ingestion: Xây dựng bộ parser và triển khai Hierarchical Chunking theo cấu trúc Điều/Khoản.
  - Ngày 3-4: Thiết lập Qdrant Vector DB, tích hợp BM25 + Dense Search và hàm Reciprocal Rank Fusion (RRF).
  - Ngày 5: Tích hợp Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`), đo đạc latency và tối ưu hóa batch inference.
- **Tuần 2:**
  - Ngày 6-7: Triển khai Enrichment pipeline: Trích xuất metadata phiên bản, tự động gắn Contextual Prepend vào từng chunk trước khi index.
  - Ngày 8-9: Xây dựng bộ Test Set gồm 50 câu hỏi & Ground Truth về chính sách; tích hợp pipeline RAGAS đánh giá 4 metrics.
  - Ngày 10: Chạy benchmark so sánh Naive Baseline vs Production RAG; viết báo cáo đánh giá hiệu năng và triển khai dịch vụ API FastAPI.

