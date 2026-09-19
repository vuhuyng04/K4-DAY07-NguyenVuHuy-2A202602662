# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Vũ Huy
**Nhóm:** Softmax (Nguyễn Vũ Huy — R1 Data, Đào Ngọc Bình Thiên — R2 Benchmark, Nguyễn Nguyên Phong — R3 Strategy)
**Ngày:** 2026-09-19

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector embedding "chỉ cùng một hướng" trong không gian ngữ nghĩa, tức là mô hình cho rằng hai đoạn văn bản nói về cùng một ý, bất kể chúng dài ngắn hay dùng từ ngữ khác nhau. Giá trị 1.0 là trùng hướng hoàn toàn, 0 là không liên quan, âm là trái ngược.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên phải đăng ký học phần trước ngày 15/9."
- Câu B: "Hạn chót đăng ký môn học là 15 tháng 9."
- Tại sao tương đồng: cùng diễn đạt một quy định (hạn đăng ký học phần) với từ đồng nghĩa (học phần/môn học, trước ngày/hạn chót) — embedding học được rằng các cụm này xuất hiện trong ngữ cảnh giống nhau.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Học phí được đóng theo từng học kỳ."
- Câu B: "Con mèo đang ngủ trên ghế sofa."
- Tại sao khác: hai câu thuộc hai chủ đề không liên quan (tài chính học vụ vs. sinh hoạt thú cưng), không chia sẻ từ khóa hay ngữ cảnh nào.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine chỉ đo *góc* giữa hai vector, bỏ qua *độ dài*. Với text, độ dài vector thường phản ánh độ dài/độ "đậm" của văn bản chứ không phải ý nghĩa, nên một câu ngắn và một đoạn dài cùng chủ đề vẫn có cosine cao dù khoảng cách Euclid rất xa. Ngoài ra, hầu hết embedder đã chuẩn hoá vector về độ dài 1, khi đó cosine = dot product, tính rất rẻ.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Trình bày phép tính: `ceil((10000 − 50) / (500 − 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> Đáp án: **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> `ceil((10000 − 100) / (500 − 100)) = ceil(9900 / 400) = ceil(24.75) = 25` chunks — nhiều hơn 2 chunk vì bước nhảy (step) giảm từ 450 xuống 400. Tăng overlap để một câu/ý nằm ở ranh giới không bị cắt đôi mất ngữ cảnh: phần đuôi của chunk trước lặp lại ở đầu chunk sau nên câu hỏi về ý đó vẫn tìm được ít nhất một chunk chứa trọn nó. Đổi lại là tốn thêm bộ nhớ và embedding call.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng regex lookbehind `(?<=[.!?])\s+` để tách *sau* dấu kết câu mà vẫn giữ dấu câu trong chunk (nếu split bằng `[.!?]\s+` thì dấu bị nuốt mất). Sau khi strip và bỏ câu rỗng, gom mỗi `max_sentences_per_chunk` câu liên tiếp thành một chunk nối bằng khoảng trắng. Edge case đã xử lý: text rỗng/toàn khoảng trắng → `[]`. Edge case **chưa** xử lý: chữ viết tắt (`TS.`, `v.v.`) và số thập phân (`3.5`) sẽ bị cắt sai vì regex không phân biệt được với dấu chấm hết câu.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> `_split` thử separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Base case: (1) text đã ≤ `chunk_size` → trả nguyên; (2) hết separator hoặc gặp `""` → cắt cứng theo `chunk_size` (đây là nhánh mà test `separators=[]` cần). Nếu separator không có trong text thì chuyển sang separator tiếp theo. Sau khi split, tôi gắn lại separator vào cuối mỗi mảnh để không mất ký tự, rồi làm **hai chiều**: mảnh nào vẫn > `chunk_size` thì đệ quy xuống với separator mịn hơn; các mảnh nhỏ liền kề được **gom lại** vào buffer cho tới sát `chunk_size` — thiếu bước gom này thì file nhiều dòng ngắn sẽ sinh hàng trăm chunk vụn.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Bỏ nhánh ChromaDB, chỉ dùng in-memory list. `_make_record` chuẩn hoá mỗi `Document` thành dict `{id, index, content, metadata, embedding}`; metadata được **copy** để không sửa dict của caller và luôn có `doc_id` (mặc định = `doc.id`, nhưng khi caller chunk sẵn với id kiểu `file#3` thì `doc_id` phải trỏ về file gốc). `search` embed query rồi gọi `_search_records` trên toàn bộ store: tính dot product (embedding đã chuẩn hoá nên = cosine), sort giảm dần, cắt `top_k`, và bỏ trường `embedding` khỏi kết quả để output gọn.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Lọc **trước** rồi mới search: lọc metadata (mọi cặp key/value trong `metadata_filter` phải khớp bằng `==`) ra tập ứng viên, rồi đưa tập đó qua cùng hàm `_search_records`. Nếu lọc *sau* top-k thì k slot có thể bị tài liệu sai chiếm hết và trả về 0 kết quả dù store còn tài liệu hợp lệ. `search` và `search_with_filter` đi chung một đường code nên `filter=None` cho kết quả y hệt `search`. `delete_document` rebuild list bỏ mọi record có `metadata['doc_id']` khớp, trả `True` nếu size giảm.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Ba bước: `store.search(question, top_k)` → `build_prompt` → `llm_fn(prompt)`. Nếu không retrieve được gì (store rỗng) thì trả thẳng câu "không tìm thấy" mà không gọi LLM. Prompt có **hai chế độ**: câu hỏi về nội dung tài liệu → chỉ dùng ngữ cảnh, không bịa, trích dẫn số hiệu đoạn, nói rõ khi không có thông tin; câu chào hỏi/câu chung không liên quan tài liệu → trả lời ngắn bằng hiểu biết chung, gắn nhãn `[GENERAL]` và không trích dẫn (để giao diện phân biệt được câu trả lời có căn cứ với câu trả lời chung mà không phải đoán qua văn xuôi). Khối NGỮ CẢNH đưa từng chunk **đánh số `[1] [2] [3]` kèm `doc_id`** để câu trả lời truy vết về đúng file, rồi CÂU HỎI. Tách `build_prompt` thành method riêng để dễ in ra kiểm tra và tái dùng khi benchmark.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
============================= test session starts =============================
TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
TestProjectStructure::test_src_package_exists PASSED [  4%]
TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
TestSentenceChunker::test_returns_list PASSED    [ 33%]
TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
TestRecursiveChunker::test_returns_list PASSED   [ 45%]
TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]
============================= 42 passed in 0.04s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

Môi trường: Python 3.13.13 (máy không có 3.11), `pytest 9.1.1`, `EMBEDDING_PROVIDER=mock`.

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Embedder: OpenAI `text-embedding-3-small` (1536 chiều). Dự đoán được ghi **trước** khi chạy. Cột mock (`_mock_embed`, băm MD5) đưa vào để đối chiếu.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? | (mock) |
|------|-----------|-----------|---------|--------------|-------|--------|
| 1 | Sinh viên phải đăng ký học phần trước ngày 15/9. | Hạn chót đăng ký môn học là 15 tháng 9. | cao | **0.725** | ✅ | −0.228 |
| 2 | Thư viện mở cửa từ 8h đến 22h các ngày trong tuần. | Giờ hoạt động của thư viện là 8:00–22:00. | cao | **0.732** | ✅ | +0.186 |
| 3 | Học phí được đóng theo từng học kỳ. | Con mèo đang ngủ trên ghế sofa. | thấp | **0.201** | ✅ | −0.039 |
| 4 | Sinh viên được mượn tối đa 5 cuốn sách. | Giảng viên được mượn tối đa 20 cuốn sách. | cao (cùng cấu trúc, khác đối tượng/số) | **0.863** | ✅ (cao hơn dự kiến) | +0.051 |
| 5 | Đơn phúc khảo nộp trong vòng 7 ngày sau khi có điểm. | Python là ngôn ngữ lập trình bậc cao. | thấp | **0.209** | ✅ | +0.009 |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là cặp 4: hai câu nói về **hai đối tượng khác nhau với hai con số khác nhau** (sinh viên/5 vs giảng viên/20) lại có điểm cao nhất bảng (0.863), vượt cả hai cặp đồng nghĩa thật sự (cặp 1–2 chỉ ~0.73). Embedding biểu diễn *chủ đề và cấu trúc* ("hạn mức mượn sách") rất tốt nhưng gần như mù với *thực thể và số liệu* — đúng thứ mà một câu hỏi quy định cần phân biệt. Với corpus quy định đại học, nếu chỉ dựa vào similarity thì câu hỏi "sinh viên được mượn bao nhiêu sách?" có thể kéo về chunk của giảng viên; đây là lý do phải gắn `audience` vào metadata và dùng `search_with_filter` thay vì tin vào điểm cosine. Đối chiếu với cột mock (cặp 1 ra âm) cũng thấy rõ: phép cosine tự nó không tạo ra ngữ nghĩa, ngữ nghĩa nằm ở embedder.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

**Cấu hình:** `bench.py` với `CHUNKER = RecursiveChunker(chunk_size=500)` (chiến lược của tôi), corpus `data/thu-vien-vinuni/` (8 trang × 2 ngôn ngữ = 16 file → 152 chunks), embedder OpenAI `text-embedding-3-small`, LLM `gpt-4o-mini`, `top_k=3`. Output đầy đủ: `ket_qua_benchmark.txt`. *(Lần chạy đầu trên corpus chỉ tiếng Anh, 81 chunks: 4/10 — Q3 và Q4 0đ.)*

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Tôi được mượn tối đa bao nhiêu cuốn sách và trong bao lâu? *(filter `audience=student`)* | `borrowing-undergraduate-staff-vi` — đoạn *Yêu cầu và giữ chỗ* | 0.607 | Đúng file, sai section; chunk có đáp án ("3 tài liệu, hai tuần") ở **hạng 2** (0.606) | "Tối đa 3 cuốn, hai tuần mỗi cuốn [2]" — **đúng** |
| 2 | Mức phạt trả sách muộn là bao nhiêu tiền một ngày? | `library-faq-vi` — mục 7 "Thư viện thu tiền phạt quá hạn… 20.000 VND/ngày" | 0.580 | **Có**, top-1 chứa đáp án | "20.000 VND/ngày quá hạn/tài liệu" — **đúng** |
| 3 | Thiết bị mượn quá hạn bao nhiêu ngày thì bị coi là mất? | `equipment-loans-vi` — "Thiết bị quá hạn hơn 05 ngày sẽ bị coi là mất" | 0.710 | **Có**, top-1 chứa đáp án | "Quá hạn hơn 05 ngày → coi là mất, bồi thường [1], [2]" — **đúng** |
| 4 | Một nhóm được đặt phòng học nhóm tối đa bao nhiêu giờ mỗi buổi…? | `borrowing-graduate-faculty-vi` — bullet "Phòng chỉ dùng để học nhóm. Phải có ít nhất 2 người…" | 0.695 | Đúng chủ đề, **sai chunk**: bullet "2 giờ mỗi buổi… 4 buổi mỗi tuần" ở **hạng 3** (0.628) | "2 giờ mỗi buổi và 4 buổi mỗi tuần [3]" — **đúng** |
| 5 | Giờ mở cửa thư viện từ tháng 9 là khi nào? | `hours-and-access-vi` — đoạn mở đầu, giờ **tháng 7–8** | 0.688 | Đúng file, chunk có đáp án (FAQ-vi mục 1, "8h45 – 21h00") ở **hạng 2** (0.664) | "Thứ Hai–Sáu 8h45–21h00, Thứ Bảy–CN 9h–17h" — **đúng** |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5. Điểm theo thang 2đ/câu của `docs/SCORING.md`: **7 / 10** (Q1: 1, Q2: 2, Q3: 2, Q4: 1, Q5: 1). Agent trả lời đúng cả 5 câu.

**Ngôn ngữ chi phối retrieval:** 15/15 chunk top-3 đều là bản VI, không chunk EN nào lọt dù nội dung y hệt — score cùng ngôn ngữ 0.6–0.7, ép `language=en` chỉ còn 0.23–0.25. Đây là lý do Q3 từ 0đ (corpus EN) lên 2đ.

**A/B metadata filter (Q1):** có filter → chunk đáp án ở hạng 2 (1đ); **không filter → 0đ**, top-3 là `borrowing-graduate-faculty-vi` (0.62–0.66: *"Học viên sau đại học được mượn tối đa 5 tài liệu"*) — cùng câu chữ, sai đối tượng; agent sẽ trả lời 5 tài liệu / 1 tháng. Filter là thứ duy nhất tách được hai trang.

**Hai mức chấm khác nhau ra sao:** nếu chỉ kiểm `doc_id` gold có trong top-3, tôi được 10/10; kiểm thêm chunk có chứa đáp án thì còn 7/10. Chênh lệch nằm ở Q1, Q4, Q5 — top-3 toàn đúng file nhưng chunk chứa số liệu ở hạng 2–3, vì chunk có *từ vựng* gần câu hỏi hơn thắng chunk có *số liệu*.

**Failure case (Bài 3.5) — Q4:** RecursiveChunker cắt ở `\n\n` rồi `\n` trước, nên danh sách bullet bị tách từng dòng rồi mới gom lại đến 500 ký tự; bullet *"Phòng chỉ dùng để học nhóm. Phải có ít nhất 2 người…"* có từ vựng (nhóm, buổi) gần câu hỏi hơn nên lên top-1, bullet *"2 giờ mỗi buổi, 2 buổi mỗi ngày, 4 buổi mỗi tuần"* rớt hạng 3. Không có overlap nên mỗi thông tin chỉ có đúng một cơ hội. HeadingChunker (Phong) được 2/2 ở câu này vì giữ trọn khối bullet dưới `## Phòng học nhóm`. **Đề xuất:** thêm overlap cho RecursiveChunker, hoặc không dùng `\n` đơn làm separator với văn bản dạng bullet.

**Failure case thứ hai — Q2 (grounding, so sánh trong nhóm):** cùng câu hỏi, Fixed (Thiên) đưa chunk trang faculty lên top-1 và agent trả lời **10.000 VND/ngày làm việc**, còn Recursive của tôi giữ trọn mục FAQ nên agent trả lời **20.000 VND/ngày**. Hai trang chính thức của cùng thư viện mâu thuẫn; corpus không phân xử được vì `document_version` cả hai đều `not-stated`. Bài học: chunker ảnh hưởng cả *con số agent nói ra*, không chỉ có tìm thấy hay không; và `document_version` cần có thật.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Từ Thiên: FixedSize "ngu" nhưng có overlap 50 thắng Recursive "thông minh" không overlap ở đúng hai câu số liệu (Q4, Q5) — overlap không phải chi tiết phụ, nó là bảo hiểm cho thông tin nằm sát ranh giới. Từ Phong: gắn lại heading vào từng mảnh con khi cắt section dài làm chunk tự mô tả, agent trích dẫn dễ hơn và người xem demo đọc hiểu chunk ngay. Nếu làm lại tôi sẽ thêm overlap cho RecursiveChunker và bỏ separator `
` đơn với văn bản dạng bullet.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
