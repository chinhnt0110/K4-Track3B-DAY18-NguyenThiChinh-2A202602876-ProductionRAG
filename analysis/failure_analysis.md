# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Thị Chinh  
**Khóa:** K4 - Track 3B  
**Ngày thực hiện:** 05/10/2026  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ | Đánh giá |
|--------|:-------------:|:----------:|:---:|:---|
| **Faithfulness** | 0.8139 | 0.7967 | -0.0172 | Điểm cao (vượt ngưỡng 0.70). Giảm nhẹ ở Production do các câu hỏi suy luận logic phức tạp (multi-hop, tính toán số học) bị LLM suy diễn thêm ngoài context nguyên văn. |
| **Answer Relevancy** | 0.4561 | 0.4293 | -0.0268 | Điểm thấp và giảm nhẹ. |
| **Context Precision** | 0.9333 | 0.9625 | +0.0292 | Đạt mức rất cao (~0.96). Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`) đưa đúng các chunk liên quan nhất lên đầu danh sách top 3. |
| **Context Recall** | 0.9250 | 0.8667 | -0.0583 | Đạt mức xuất sắc (~0.87). Hybrid Search (BM25 + Dense) kết hợp RRF bao phủ hầu hết các ý quan trọng trong Ground Truth. |

> **Tổng kết:** Cả **Context Precision (0.9625)**, **Context Recall (0.8667)** và **Faithfulness (0.7967)** đều vượt mốc **0.70**.

---

## Bottom-5 Failures

### #1
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Không tìm thấy thông tin hoặc trả lời thẩm quyền không chính xác / thiếu căn cứ.
- **Worst metric:** Faithfulness (0.00) / Context Precision (0.00) / Context Recall (0.00)
- **Error Tree:** Output sai → Context sai (thiếu chunk thẩm quyền mua sắm trên 50 triệu) → Query mismatch ("55 triệu" vs "50.000.000 VNĐ") → Retrieval Failure.
- **Root cause:** **Vocabulary Gap & Numerical Reasoning Gap**: Người dùng hỏi bằng ngôn ngữ tự nhiên ("55 triệu"), trong khi tài liệu quy định mốc "Trên 50.000.000 VNĐ". BM25 không khớp từ khóa số, mô hình Dense embedding chưa thực hiện được phép so sánh số học ($55.000.000 > 50.000.000$).
- **Suggested fix:** Áp dụng **Query Rewriting / Expansion** trước khi search để chuẩn hóa các cụm từ số tiền (ví dụ: "55 triệu" → "55.000.000 VNĐ, mua sắm trên 50 triệu"); hoặc bổ sung metadata khoảng giá trị (`min_value`, `max_value`) cho các chunk quy trình mua sắm.

---

### #2
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** Chỉ trả lời Giám đốc điều hành (CEO) phê duyệt, bỏ qua thông tin cảnh báo về bảo hiểm xã hội.
- **Worst metric:** Faithfulness (0.00, do câu trả lời thiếu vắng bằng chứng so với Ground Truth yêu cầu)
- **Error Tree:** Output thiếu ý → Context đúng (chunk trích từ `nghi_phep_khong_luong.md` có đầy đủ cả quy trình duyệt CEO và điều khoản bảo hiểm) → Query OK → Generation Prompt chưa nhắc nhở liệt kê các lưu ý ràng buộc pháp lý/phúc lợi.
- **Root cause:** System prompt của LLM Generation yêu cầu trả lời súc tích khiến LLM chỉ tập trung trả lời chủ thể phê duyệt mà bỏ qua các điều kiện ràng buộc quyền lợi đi kèm trong cùng điều khoản.
- **Suggested fix:** Cải tiến Prompt Generation: *"Trả lời đầy đủ cấp phê duyệt và nêu rõ các điều kiện ràng buộc / lưu ý phúc lợi bắt buộc (nếu có) được đề cập trong context."*

---

### #3
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Trả lời nhầm lẫn giữa chu kỳ 90 ngày (v1.0) và 120 ngày (v2.0) hoặc đưa ra cả hai mà không chỉ rõ bản hiện hành.
- **Worst metric:** Faithfulness (0.00) / Answer Relevancy (0.00)
- **Error Tree:** Output mâu thuẫn → Context retrieve cả 2 chunk xung đột (`mat_khau_v1.md` và `mat_khau_v2.md`) → Reranker không nhận biết được hiệu lực văn bản → Conflict Generation.
- **Root cause:** **Temporal & Version Conflict**: Trong cơ sở tri thức có tài liệu cũ (v1.0, 90 ngày) và mới (v2.0, 120 ngày). Do thiếu Metadata filtering về phiên bản hiệu lực, cả hai chunk đều có điểm tương đồng ngữ nghĩa cao với câu hỏi.
- **Suggested fix:** 
  1. Thêm Metadata Filter: `status: active` hoặc `version: latest` khi tìm kiếm.
  2. Contextual Prepend: LLM enrichment ghi rõ *"Tài liệu chính sách mật khẩu v2.0 thay thế v1.0"* và thêm chỉ dẫn vào prompt: *"Nếu có nhiều phiên bản tài liệu mâu thuẫn, ưu tiên phiên bản có ngày hiệu lực mới nhất."*

---

### #4
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Trả lời tính phạt 2% nguyên tháng (300.000 VNĐ) hoặc không tính được số ngày quá hạn (20 - 15 = 5 ngày).
- **Worst metric:** Faithfulness (0.1667)
- **Error Tree:** Output sai số tiền phạt → Context đúng (`tam_ung.md` nêu rõ hạn 15 ngày, phạt 2%/tháng trên số tiền chưa hoàn ứng) → Query OK → Generation tính toán số học phức tạp bị sai.
- **Root cause:** **Arithmetic & Reasoning Limitation**: LLM `gpt-4o-mini` khi sinh trực tiếp không có cơ chế Chain-of-Thought (CoT) sẽ dễ nhầm lẫn giữa mức phí cả tháng và mức phạt theo ngày thực tế (pro-rata).
- **Suggested fix:** Cấu hình Chain-of-Thought trong generation prompt: *"Đối với câu hỏi có tính toán số liệu: Bước 1: Liệt kê các mốc thời gian và hạn mức; Bước 2: Tính số ngày/tháng vi phạm; Bước 3: Áp dụng công thức và tính ra con số cuối cùng."*

---

### #5
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Trả lời 20.000.000 VNĐ hoặc chỉ nói được tỷ lệ 85% mà không có con số 17.000.000 VNĐ.
- **Worst metric:** Faithfulness (0.00)
- **Error Tree:** Output thiếu thông tin liên kết → Context bị phân mảnh (quy định 85% nằm ở `thu_viec.md`, bảng lương Junior nằm ở `bang_luong_2024.md`) → Single retrieval không gom đủ 2 tài liệu.
- **Root cause:** **Multi-hop Retrieval Failure**: Câu hỏi yêu cầu tổng hợp thông tin từ 2 nguồn văn bản độc lập (chính sách thử việc và thang bảng lương). RAG tìm kiếm đơn luồng (single-turn) chỉ kéo được 1 chunk nổi trội hơn về từ khóa.
- **Suggested fix:** Sử dụng kỹ thuật **Multi-hop Query Decomposition** (hoặc Sub-question Query Engine): tách câu hỏi thành 2 sub-queries: (1) *"Thang lương vị trí Junior mức cao nhất là bao nhiêu?"* và (2) *"Quy định tỷ lệ phần trăm lương thử việc là bao nhiêu?"*, sau đó hợp nhất context trước khi tổng hợp câu trả lời.

---

## Case Study (cho presentation)

### **Question chọn phân tích:**
> **"Bao lâu phải đổi mật khẩu một lần?"** (Case Study về Temporal & Version Conflict)

### **Error Tree walkthrough:**
1. **Output đúng?** → **KHÔNG**. Hệ thống có nguy cơ trả lời 90 ngày (theo v1.0) hoặc trả lời cả 90 và 120 ngày gây bối rối cho người dùng nội bộ.
2. **Context đúng?** → **MỘT NỬA**. Retrieval kéo về cả 2 chunk: một chunk từ `mat_khau_v1.md` (chu kỳ 90 ngày) và một chunk từ `mat_khau_v2.md` (chu kỳ 120 ngày).
3. **Query rewrite OK?** → **OK**. Câu hỏi người dùng rất rõ ràng, nhưng vì câu hỏi không nêu rõ "phiên bản 2024" nên cả 2 chunk đều match điểm BM25 và Dense rất cao.
4. **Fix ở bước:** 
   - **Bước Indexing / Ingestion:** Đưa trường metadata `effective_date`, `version`, `is_active` vào Qdrant payload.
   - **Bước Search:** Bổ sung metadata pre-filter `is_active == True` để loại trừ hoàn toàn các tài liệu đã hết hiệu lực.
   - **Bước Generation:** Bổ sung Prompting Directive: *"Luôn kiểm tra thuộc tính ngày hiệu lực của context để đảm bảo trả lời theo chính sách mới nhất."*

---

## Nếu có thêm 1 giờ, sẽ optimize:
1. **Multi-Query / Query Expansion:** Tích hợp module tự động sinh 3 biến thể câu hỏi (synonyms, số tiền quy đổi "55 triệu" ↔ "50.000.000 VNĐ") để giải quyết triệt để lỗi Vocabulary Gap (như trong Question #1).
2. **Metadata Versioning Filter:** Thiết lập bộ lọc trạng thái hiệu lực văn bản (`active_only=True`) trước khi chạy Hybrid Search để loại bỏ 100% rủi ro xung đột chính sách cũ/mới (như trong Question #3 và #4 về phép năm v2023/v2024).
3. **Chain-of-Thought (CoT) Prompting cho tác vụ tính toán:** Bổ sung scratchpad suy luận từng bước cho các câu hỏi yêu cầu tính toán tiền tệ, ngày phép pro-rata, hoàn trả chi phí đào tạo để nâng Faithfulness lên trên 0.85.

