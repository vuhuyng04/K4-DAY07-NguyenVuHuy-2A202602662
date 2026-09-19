# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** Softmax
**Thành viên:** Nguyễn Vũ Huy (2A202602662) — R1 Data · Đào Ngọc Bình Thiên (2A202602814) — R2 Benchmark · Nguyễn Nguyên Phong (2A202602691) — R3 Strategy
**Ngày:** 2026-09-19

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Dịch vụ thư viện VinUni — quy định mượn/trả, phạt, giờ mở cửa, đặt phòng học nhóm (nguồn: `library.vinuni.edu.vn`).

**Tại sao nhóm chọn chủ đề này?**
> Thư viện là mảng duy nhất trong danh sách gợi ý của K4-L3A mà trang công khai của VinUni (1) cho phép crawl — `robots.txt` của `library.vinuni.edu.vn` không cấm gì, (2) có nhiều **con số và mốc thời gian** kiểm chứng được (3 items / 2 weeks, 20.000 VND/ngày, 8:45–21:00, 2 hours per session…), và (3) có **hai trang riêng cho hai đối tượng** — undergraduate/staff và graduate/faculty — cùng cấu trúc, cùng từ vựng nhưng khác đáp án. Điểm (3) là điều kiện để câu hỏi cần `metadata_filter={"audience": "student"}` thực sự *cần* filter chứ không phải để cho có.

### Danh sách tài liệu (Data Inventory)

Thư mục `data/thu-vien-vinuni/`, 8 file `.md` + `sources.csv`. Tất cả crawl ngày 2026-09-19 bằng `scripts/fetch_public_pages.py`, sau đó **làm sạch tay** (bỏ menu/footer ~2–3 KB mỗi trang, chuyển bảng HTML thành bảng Markdown, giữ nguyên điều khoản và số liệu). Corpus tiếng Anh vì trang gốc chỉ có tiếng Anh; query đặt bằng tiếng Việt để kiểm cross-lingual retrieval.

| # | Tên tài liệu (`doc_id`) | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | `borrowing-undergraduate-staff` | library.vinuni.edu.vn/services/borrow-and-request/undergraduate-and-staff/ | 2026-09-19 / not-stated | 5 077 | audience=**student**, category=borrowing |
| 2 | `borrowing-graduate-faculty` | library.vinuni.edu.vn/services/borrow-and-request/graduate-faculty-and-instructors/ | 2026-09-19 / not-stated | 5 554 | audience=**faculty**, category=borrowing |
| 3 | `borrowing-privilege` | library.vinuni.edu.vn/borrowing-priviledge/ | 2026-09-19 / not-stated | 2 687 | audience=all, category=borrowing |
| 4 | `equipment-loans` | library.vinuni.edu.vn/equipment-loans/ | 2026-09-19 / not-stated | 808 | audience=all, category=borrowing |
| 5 | `fines-and-charges` | library.vinuni.edu.vn/fine-and-other-charges/ | 2026-09-19 / not-stated | 1 699 | audience=all, category=fees |
| 6 | `hours-and-access` | library.vinuni.edu.vn/about-us/hours-and-access/ | 2026-09-19 / not-stated | 1 088 | audience=all, category=access |
| 7 | `room-booking` | library.vinuni.edu.vn/room-booking/ | 2026-09-19 / not-stated | 3 652 | audience=all, category=spaces |
| 8 | `library-faq` | library.vinuni.edu.vn/faq/ | 2026-09-19 / 2024 | 8 470 | audience=all, category=faq |

Đã loại 2 trang sau khi crawl: `how-to-borrow-return-renew` (chỉ có tiêu đề video, không có nội dung text) và `borrowing-vingroup-community` (không có số liệu, lặp nội dung trang 1–2).

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ. (Đã xoá địa chỉ email cá nhân/phòng ban khỏi FAQ khi làm sạch.)
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata. `document_version` ghi `not-stated` khi trang không nêu — không bịa số hiệu.

Script kiểm tra CP2 (lab doc mục 3): 8/8 file `OK`, `sources.csv` khớp 1-1, `audience` = {student: 1, faculty: 1, all: 6}.

*Ghi chú kỹ thuật:* `scripts/fetch_public_pages.py` gốc đọc `robots.txt` bằng User-Agent mặc định `Python-urllib`, bị WAF của VinUni trả 403 → stdlib coi là "cấm toàn bộ" dù robots.txt thật cho phép. Nhóm sửa script để tải `robots.txt` bằng đúng UA khai báo (vẫn tôn trọng 401/403 thật).

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | str | `borrowing-undergraduate-staff` | Trỏ chunk về file gốc; `delete_document` và chấm benchmark dựa vào nó |
| `audience` | enum `student/faculty/staff/all` | `student` | Trường lọc chính: hai trang mượn sách cùng từ vựng nhưng khác hạn mức theo đối tượng |
| `category` | enum `borrowing/fees/access/spaces/faq` | `fees` | Lọc theo loại câu hỏi (phạt tiền vs giờ mở cửa) khi corpus lớn hơn |
| `department` | str | `library` | Chuẩn bị mở rộng sang phòng ban khác (học vụ, KTX) mà không lẫn |
| `language` | str | `en` | Corpus tiếng Anh, query tiếng Việt — ghi rõ để giải thích score thấp |
| `source_url`, `retrieved_at`, `document_version` | str | `…/faq/`, `2026-09-19`, `2024` | Truy vết và phân xử khi hai trang mâu thuẫn (xem Q2 mục 3) |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

> Mỗi thành viên thử **một chiến lược khác nhau** trên cùng bộ tài liệu; nhóm tổng hợp và so sánh ở đây.

### Phân tích đường cơ sở (Baseline Analysis)

`ChunkingStrategyComparator().compare(body, chunk_size=500)` trên 3 tài liệu (đã bỏ front matter):

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| `borrowing-privilege` (2 687) | FixedSizeChunker (`fixed_size`) | 6 | 490 | Cắt giữa hàng của bảng Markdown; overlap 50 cứu được một phần |
| | SentenceChunker (`by_sentences`) | 6 | 447 (min 68, max 1 242) | Bảng không có dấu chấm → 1 chunk 1 242 ký tự chứa cả bảng; các ghi chú thì vụn |
| | RecursiveChunker (`recursive`) | 10 | 269 (min 10, max 492) | Giữ trọn từng dòng bảng/bullet nhưng sinh chunk rất ngắn (10 ký tự) |
| `library-faq` (8 470) | FixedSizeChunker | 19 | 493 | Cắt giữa câu hỏi–câu trả lời FAQ |
| | SentenceChunker | 39 | 215 | Mỗi FAQ thành 1–3 chunk, hỏi/đáp hay bị tách |
| | RecursiveChunker | 24 | 353 | Tách theo `\n\n` nên mỗi mục FAQ thường trọn 1 chunk |
| `room-booking` (3 652) | FixedSizeChunker | 9 | 450 | Cắt giữa bảng phòng |
| | SentenceChunker | 6 | 607 (max 2 472) | Bảng 25 phòng không có dấu chấm → 1 chunk khổng lồ |
| | RecursiveChunker | 10 | 365 | Bullet quy định bị tách khỏi nhau (xem failure case Q4) |

### Chiến lược của từng thành viên

**Thành viên 1 — Nguyễn Vũ Huy (R1)**
- **Loại chiến lược:** `RecursiveChunker(chunk_size=500)`, separators mặc định `["\n\n", "\n", ". ", " ", ""]`
- **Mô tả & lý do chọn cho chủ đề này:** Trang thư viện đã được viết theo đoạn và bullet; cắt ở ranh giới "to" (`\n\n`) trước rồi mới hạ xuống ranh giới nhỏ hơn sẽ giữ trọn từng mục FAQ, từng ghi chú — kỳ vọng tốt hơn FixedSize vốn cắt mù giữa câu. Không có overlap để xem thuần tuý ranh giới ngữ nghĩa có đủ hay không.
- **Kết quả:** 81 chunks, **4/10** (Q1: 1, Q2: 2, Q3: 0, Q4: 0, Q5: 1). Thua ở Q4/Q5 vì separator `\n` tách rời từng bullet/từng dòng giờ mở cửa; không có overlap nên chunk chứa số liệu rớt khỏi top-1.

**Thành viên 2 — Đào Ngọc Bình Thiên (R2)**
- **Loại chiến lược:** `FixedSizeChunker(chunk_size=500, overlap=50)`
- **Mô tả & lý do chọn:** Chọn làm đối chứng "không thông minh": cắt mù 500 ký tự, không quan tâm câu hay đoạn, nhưng có overlap 50 để mỗi ranh giới xuất hiện ở hai chunk. Với corpus quy định nhiều bullet ngắn và bảng, giả thuyết là overlap quan trọng hơn ranh giới ngữ nghĩa — số liệu nằm sát ranh giới vẫn có một chunk chứa trọn nó. `chunk_size=500` chọn bằng Recursive của Huy để so sánh công bằng.
- **Code snippet:** không custom, dùng `FixedSizeChunker` có sẵn trong `src/chunking.py`.
- **Kết quả chạy thử trên cùng cấu hình:** 68 chunks, **7/10** (1, 2, 0, 2, 2)

**Thành viên 3 — Nguyễn Nguyên Phong (R3)**
- **Loại chiến lược:** `HeadingChunker(chunk_size=800)` — custom, chunk theo tiêu đề `#`/`##`; section dài hơn 800 ký tự hạ xuống `RecursiveChunker` và **gắn lại heading vào từng mảnh con**
- **Mô tả & lý do chọn:** Trang quy định thư viện đã được người soạn chia sẵn theo mục (`## Borrowing privileges`, `## 7. What is the penalty…`), mỗi mục là một đơn vị ngữ nghĩa trọn vẹn — cắt đúng ranh giới đó thì chunk vừa giữ đủ điều kiện + con số, vừa tự mô tả "đây là mục gì". `chunk_size=800` lớn hơn hai chiến lược kia vì một mục quy định thường dài hơn 500 ký tự; section vượt ngưỡng (FAQ 22 mục, bảng 25 phòng) hạ xuống Recursive nhưng **gắn lại heading vào từng mảnh** để mảnh thứ hai không mất ngữ cảnh. Đây là chiến lược theo heading bắt buộc của K4-L3A.
- **Code snippet:** `src/chunking.py::HeadingChunker`
```python
class HeadingChunker:
    _HEADING = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)

    def chunk(self, text):
        sections = self._split_sections(text)          # [(heading, body)]
        if not sections:
            return self._fallback.chunk(text)          # không có heading → recursive
        chunks = []
        for heading, body in sections:
            full = f"{heading}\n{body}".strip()
            if len(full) <= self.chunk_size:
                chunks.append(full)
            else:                                      # section dài → cắt nhỏ, giữ heading
                for piece in self._fallback.chunk(body):
                    chunks.append(f"{heading}\n{piece}")
        return chunks
```
- **Kết quả chạy thử trên cùng cấu hình:** 74 chunks, **7/10** (1, 2, 0, 2, 2)

### So Sánh Giữa Các Thành Viên

Cùng corpus (8 file), cùng 5 query, cùng embedder `text-embedding-3-small`, cùng `top_k=3`, cùng cách chấm (chunk chứa đáp án ở top-1 = 2đ, top-2/3 = 1đ):

| Thành viên | Chiến lược (Strategy) | Chunks | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|---|----------------------|-----------|----------|
| Huy | Recursive(500) | 81 | **4** | Giữ trọn từng mục FAQ (Q2 top-1) | Bullet list bị tách từng dòng; không overlap → Q4 0đ, Q5 chỉ 1đ |
| Thiên | FixedSize(500, 50) | 68 | **7** | Overlap 50 cho mỗi ý 2 cơ hội lọt top-k (Q4, Q5 đều 2đ) | Cắt mù giữa câu/giữa hàng bảng; chunk khó đọc khi demo |
| Phong | Heading(800) | 74 | **7** | Section trọn vẹn, heading lặp lại làm chunk tự mô tả (Q4, Q5 top-1) | Section dài (FAQ 22 mục, bảng 25 phòng) vẫn phải hạ xuống recursive |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> Với corpus quy định thư viện, **HeadingChunker** và **FixedSize có overlap** cùng đạt 7/10, còn Recursive không overlap chỉ 4/10. Điểm khác biệt không nằm ở "thông minh" hơn mà ở việc **giữ được khối thông tin liền nhau**: quy định thư viện viết dạng bullet/bảng, mỗi bullet là một mệnh đề ngắn có từ vựng gần nhau ("session", "group", "room"); Recursive tách chúng rời rạc nên chunk *có từ khoá* thắng chunk *có số liệu*. Heading giữ cả khối bullet dưới một tiêu đề, FixedSize dùng overlap để mỗi bullet xuất hiện ở hai chunk. Heading nhỉnh hơn về chất lượng ngữ cảnh khi demo (chunk tự giải thích "đây là mục gì"), nên nhóm chọn Heading là chiến lược khuyến nghị cho văn bản quy định; FixedSize+overlap là baseline rất khó thua nếu chỉ đo top-k. Cả 3 đều 0đ ở Q3 → lỗi nằm ở query/corpus, không ở chunker (xem mục 3).

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

> **Đúng 5 câu hỏi**, đa dạng, có thể kiểm chứng; **ít nhất 1 câu** cần lọc metadata mới trả lời tốt. Đây là bộ câu hỏi chung cho mọi thành viên chạy. R2 (Thiên) chốt sau khi đối chiếu từng gold answer với file nguồn; Q3 giữ nguyên dù cả 3 chiến lược đều 0đ vì đây là failure case có giá trị phân tích (mục 4).

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Tôi được mượn tối đa bao nhiêu cuốn sách và trong bao lâu? **(cần `metadata_filter={"audience": "student"}`)** | "Undergraduate students may borrow up to **3 items during two weeks** per item. Books may be renewed once for one week…" | `borrowing-undergraduate-staff` — mục *Borrowing privileges* |
| 2 | Mức phạt trả sách muộn là bao nhiêu tiền một ngày? | "Normal material: **20,000 VND/day** overdue/document." | `library-faq` — mục 7 |
| 3 | Thiết bị mượn quá hạn bao nhiêu ngày thì bị coi là mất? | "Equipment overdue for more than **05 days** will be considered lost, and the borrower will be charged for a replacement." | `equipment-loans` (và lặp trong `borrowing-undergraduate-staff` — mục *Equipment loans*) |
| 4 | Một nhóm được đặt phòng học nhóm tối đa bao nhiêu giờ mỗi buổi và bao nhiêu buổi mỗi tuần? | "**2 hours per session, 2 sessions per day, 4 sessions per week**, all rooms combined." | `room-booking` (và lặp trong 2 trang borrowing — mục *Study rooms*) |
| 5 | Giờ mở cửa thư viện từ tháng 9 là khi nào? | "Opening hours from September: Monday to Friday: **8:45 am – 9:00 pm**; Saturday and Sunday: 9:00 am – 5:00 pm." | `hours-and-access` — mục *Hours* |

Dạng hỏi: Q1 điều kiện theo đối tượng, Q2 tra số liệu, Q3 ngưỡng/điều kiện, Q4 giới hạn (nhiều con số), Q5 thời gian. Q3/Q4 chấp nhận nhiều `gold_doc` vì cùng quy định xuất hiện ở nhiều trang chính thức.

### Tổng hợp chất lượng truy xuất của nhóm

> Cách chấm (theo `docs/SCORING.md`): **2 điểm/câu** — top-3 chứa chunk liên quan + agent trả lời đúng (2), có liên quan nhưng thiếu/không ở top-1 (1), không có trong top-3 (0). Nhóm chấm ở **mức chunk**: chunk phải vừa đúng `doc_id` vừa chứa chuỗi đặc trưng của gold answer (`must_contain` trong `bench.py`).

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | Mượn tối đa bao nhiêu / bao lâu (filter student) | Heading (hạng 2) ≥ Recursive, Fixed (hạng 3) | Có (cả 3, khi có filter) | **Không filter → 0/2 ở cả 3 chiến lược**; agent trả lời đúng "3 cuốn / 2 tuần [3]" |
| 2 | Phạt trả muộn | Cả 3 (top-1) | Có | Agent (gpt-4o-mini) trả lời **10.000 VND** từ chunk [3] (trang faculty) thay vì 20.000 ở chunk [1] — hai trang chính thức mâu thuẫn |
| 3 | Thiết bị quá hạn bao nhiêu ngày | Không chiến lược nào | **Không** (0/2 cả 3) | Chunk "fined for returning items late… lost" của trang faculty thắng chunk "overdue for more than 05 days"; cross-lingual + "05 days" |
| 4 | Đặt phòng tối đa | Fixed, Heading (top-1) | Có với Fixed/Heading; **không** với Recursive | Recursive tách bullet "2 hours per session" khỏi bullet "at least 2 people" |
| 5 | Giờ mở cửa tháng 9 | Fixed, Heading (top-1) | Có | Recursive đưa đoạn giờ **tháng 7–8** lên top-1 (cùng từ vựng), đoạn tháng 9 hạng 2 |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> Giúp quyết định ở **Q1**: không filter, top-3 của cả ba chiến lược là chunk phạt tiền của FAQ và trang faculty (score 0.31–0.32), trang undergraduate không xuất hiện (0.23) → 0/2; có `audience=student`, toàn bộ top-3 về đúng trang và agent trả lời đúng. Similarity đo "cùng chủ đề mượn sách" chứ không đo "đúng đối tượng" — đúng như dự đoán cặp câu số 4 trong report cá nhân (0.863 cho hai câu khác đối tượng). Mặt trái: filter `audience=student` cứng sẽ **loại luôn** `borrowing-privilege` và `library-faq` (audience=all) — hai trang có bảng hạn mức đầy đủ nhất; nếu trang undergraduate thiếu thông tin thì filter làm mất recall. Với corpus này filter không hại ở câu nào khác vì chỉ Q1 dùng nó.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
> 1. **Chấm hai mức lật kết quả**: chỉ kiểm `doc_id` gold trong top-3 thì Recursive được 8/10, kiểm chunk có chứa đáp án thì còn 4/10. Chunk đúng chủ đề nhưng không có số liệu thắng chunk có đáp án là lỗi phổ biến nhất của cả ba chiến lược.
> 2. **A/B filter tại Q1**: 0/2 → 2/2 chỉ bằng một dòng `metadata_filter`; embedding không phân biệt được sinh viên và giảng viên khi hai trang dùng cùng câu chữ.
> 3. **Nguồn chính thức tự mâu thuẫn**: FAQ nói 20.000 VND/ngày, trang faculty nói 10.000 VND/business day; agent chọn chunk [3]. `document_version` không phải trường hình thức — nhóm không có gì để phân xử vì cả hai `not-stated`.

**Công cụ demo:** `streamlit run demo_app.py` — tab *Kịch bản demo* có 6 tình huống chọn sẵn (A/B metadata filter, so sánh chunking ở Q4, nguồn mâu thuẫn Q2, failure case Q3, chấm hai mức, bảng tổng hợp), tab truy vấn tự do và tab xem chunk; nhập API key ngay trên sidebar. Cache embedding dùng chung với `bench.py` nên demo live không gọi API embedding.

**Kịch bản demo 6–8 phút:**
> 1' Huy — chủ đề, 8 file, vì sao thư viện VinUni (robots cho phép, 2 trang student/faculty). · 2' mỗi người 40" chiến lược + điểm. · 3' Huy chạy live `python bench.py` (cache → không tốn API), chỉ vào Q1 A/B và Q4 failure; Phong giải thích vì sao Heading thắng Q4/Q5; Thiên giải thích overlap. · 1' Thiên: mâu thuẫn 10k/20k VND và bài học `document_version`. · Q&A. Terminal mở sẵn, `ket_qua_benchmark.txt` của 3 người đã chạy trước.

**Bài học rút ra khi so sánh trong nhóm:**
> Cùng 8 file, cùng 5 câu, chỉ đổi một dòng chunker mà điểm dao động 4–7/10. Thứ quyết định không phải chunk "thông minh" hay không mà là **khối thông tin có bị tách khỏi ngữ cảnh gần nó không**: overlap (Fixed) hoặc ranh giới do người soạn định sẵn (Heading) đều giữ được, còn Recursive không overlap cắt bullet list thành từng dòng rồi thua ở chính những câu hỏi số liệu. Q3 thua ở cả ba chiến lược cho thấy giới hạn nằm ở query/corpus (tiếng Việt hỏi corpus tiếng Anh, "05 days") chứ chunker không cứu được.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> (1) Tách `library-faq` và `borrowing-privilege` thành các file nhỏ theo đối tượng thay vì `audience=all`, để filter không phải đánh đổi recall. (2) Ghi `document_version` bằng ngày cập nhật trang (lấy từ sitemap `lastmod`) thay vì `not-stated`, để agent có căn cứ ưu tiên nguồn mới hơn khi hai trang mâu thuẫn. (3) Bỏ nội dung lặp giữa các trang (phòng học, thiết bị xuất hiện ở 3 file) — lặp làm top-3 bị chiếm bởi cùng một đoạn từ nhiều file, không thêm thông tin. (4) Thêm overlap cho RecursiveChunker hoặc bỏ separator `\n` đơn với văn bản dạng bullet.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | 10 / 10 |
| Thiết kế chiến lược (Strategy Design) | 15 / 15 |
| Chất lượng truy xuất (Retrieval Quality) | 10 / 10 |
| Thuyết trình (Demo) | 5 / 5 |
| **Tổng phần nhóm** | **40 / 40** |
