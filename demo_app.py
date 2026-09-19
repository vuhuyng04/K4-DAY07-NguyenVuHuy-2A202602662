"""
demo_app.py — giao diện demo trực quan cho Lab 07 (Streamlit).

    pip install -r requirements-demo.txt
    streamlit run demo_app.py

- Nhập OpenAI API key ngay trên sidebar (hoặc để trống → dùng .env; không có gì → mock).
- Tab "Kịch bản demo": chọn sẵn tình huống (metadata filter, so sánh chunking,
  nguồn mâu thuẫn, failure case, chấm hai mức, bảng tổng hợp) — bấm 1 nút là chạy.
- Tái dùng bench.py (corpus, cache embedding, cách chấm) nên demo không tốn API
  cho những gì đã chạy.
"""

from __future__ import annotations

import hashlib
import html as _html
import os
import re
import sys

import pandas as pd
import streamlit as st

import bench
from src import (
    EmbeddingStore,
    FixedSizeChunker,
    HeadingChunker,
    KnowledgeBaseAgent,
    RecursiveChunker,
    SentenceChunker,
)

sys.argv = ["bench.py"]

STRATEGIES = {
    "Recursive(500) — Huy": lambda: RecursiveChunker(chunk_size=500),
    "FixedSize(500, overlap=50) — Thiên": lambda: FixedSizeChunker(chunk_size=500, overlap=50),
    "Heading(800) — Phong": lambda: HeadingChunker(chunk_size=800),
    "Sentence(3 câu)": lambda: SentenceChunker(max_sentences_per_chunk=3),
}
MAIN3 = list(STRATEGIES)[:3]
AUDIENCES = {"(không lọc)": None, "student": {"audience": "student"}, "faculty": {"audience": "faculty"}, "all": {"audience": "all"}}
Q = {f"Q{i}": q for i, q in enumerate(bench.QUERIES, 1)}

st.set_page_config(page_title="Lab 07 — Softmax RAG demo", page_icon="📚", layout="wide")


# ============================================================================
# Backend (embedder + LLM) — cache theo fingerprint của key để đổi key là rebuild
# ============================================================================
def _fingerprint() -> str:
    raw = f"{os.getenv('EMBEDDING_PROVIDER','')}|{os.getenv('OPENAI_API_KEY','')}|{os.getenv('OPENAI_CHAT_MODEL','')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


@st.cache_resource(show_spinner="Khởi tạo embedder + LLM…")
def get_backends(fp: str):
    embedder = bench.CachedEmbedder(bench.pick_embedder())
    llm = bench.make_llm_fn()
    return embedder, llm


@st.cache_resource(show_spinner="Chunk + embed corpus…")
def build_store(strategy_name: str, fp: str) -> tuple[EmbeddingStore, int, dict[str, int]]:
    embedder, _ = get_backends(fp)
    bench.CHUNKER = STRATEGIES[strategy_name]()
    docs = bench.load_corpus(bench.CORPUS_DIR)
    store = EmbeddingStore(collection_name=strategy_name, embedding_fn=embedder)
    store.add_documents(docs)
    embedder.save()
    per_doc: dict[str, int] = {}
    for d in docs:
        per_doc[d.metadata["doc_id"]] = per_doc.get(d.metadata["doc_id"], 0) + 1
    return store, len(docs), per_doc


# ============================================================================
# Helpers
# ============================================================================
def grade_doc_level(results: list[dict], gold_doc) -> int:
    """Cách chấm 'ngây thơ': chỉ kiểm doc_id gold có trong top-k."""
    golds = {gold_doc} if isinstance(gold_doc, str) else set(gold_doc)
    ranks = [i for i, r in enumerate(results, 1) if r["metadata"].get("doc_id") in golds]
    if not ranks:
        return 0
    return 2 if ranks[0] == 1 else 1


def _contains(content: str, must_contain) -> bool:
    """must_contain: str hoặc list[str] (EN + VI) — khớp bất kỳ."""
    needles = [must_contain] if isinstance(must_contain, str) else list(must_contain or [])
    c = content.lower()
    return any(n and n.lower() in c for n in needles)


def render_results(results: list[dict], must_contain="", gold=None, chars: int = 600) -> None:
    if not results:
        st.info("Không có kết quả (filter loại hết ứng viên?)")
        return
    golds = set() if gold is None else ({gold} if isinstance(gold, str) else set(gold))
    top = max(r["score"] for r in results) or 1.0
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        hit = _contains(r["content"], must_contain)
        is_gold = meta.get("doc_id") in golds
        tags = []
        if is_gold:
            tags.append("📄 đúng file")
        if hit:
            tags.append("✅ chứa đáp án")
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**#{i} · `{meta.get('doc_id')}`** · audience=`{meta.get('audience')}` · chunk {meta.get('chunk_index')}  {' · '.join(tags)}")
            c2.metric("score", f"{r['score']:.3f}")
            st.progress(min(1.0, max(0.0, r["score"] / top)))
            st.text(r["content"][:chars] + ("…" if len(r["content"]) > chars else ""))


FILTER_SKIP = {"chunk_index", "source", "retrieved_at", "title", "source_url"}


def metadata_options(store: EmbeddingStore) -> dict[str, list[str]]:
    """Mọi trường metadata có trong store (trừ trường kỹ thuật) → danh sách giá trị."""
    opts: dict[str, set] = {}
    for r in store._store:
        for k, v in r["metadata"].items():
            if k in FILTER_SKIP or v is None:
                continue
            opts.setdefault(k, set()).add(str(v))
    order = ["audience", "category", "department", "language", "document_version", "doc_id"]
    keys = [k for k in order if k in opts] + sorted(k for k in opts if k not in order)
    return {k: sorted(opts[k]) for k in keys}


def count_candidates(store: EmbeddingStore, flt: dict | None) -> int:
    if not flt:
        return len(store._store)
    return sum(1 for r in store._store if all(str(r["metadata"].get(k)) == str(v) for k, v in flt.items()))


def filter_builder(store: EmbeddingStore, key: str, default: dict | None = None) -> dict | None:
    """Widget chọn nhiều trường metadata → dict filter (hoặc None)."""
    opts = metadata_options(store)
    default = default or {}
    fields = st.multiselect("Lọc theo trường", list(opts), default=[k for k in default if k in opts], key=f"{key}_fields")
    flt: dict = {}
    if fields:
        cols = st.columns(len(fields))
        for col, f in zip(cols, fields):
            vals = opts[f]
            idx = vals.index(str(default[f])) if f in default and str(default[f]) in vals else 0
            flt[f] = col.selectbox(f, vals, index=idx, key=f"{key}_{f}")
    n = count_candidates(store, flt or None)
    st.caption(f"metadata_filter = `{flt or None}` → **{n}/{len(store._store)}** chunk còn lại làm ứng viên")
    return flt or None


def run_query(store: EmbeddingStore, q: dict, top_k: int, use_filter: bool = True) -> tuple[list[dict], int, str]:
    flt = q["filter"] if use_filter else None
    res = store.search_with_filter(q["q"], top_k=top_k, metadata_filter=flt)
    sc, why = bench.grade(res, q["gold_doc"], q["must_contain"])
    return res, sc, why


def agent_answer(store: EmbeddingStore, llm, question: str, results: list[dict]) -> str:
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
    if not results:
        return agent.NO_CONTEXT_ANSWER
    with st.spinner("Gọi LLM…"):
        return llm(agent.build_prompt(question, results))


_CITE = re.compile(r"\[(\d+)\]")


def show_answer(answer: str, results: list[dict], must_contain="", compact: bool = False) -> None:
    """Hiện câu trả lời + bảng 'Nguồn trích dẫn' nối [n] → chunk → doc_id → source_url."""
    st.success(answer)
    cited = sorted({int(n) for n in _CITE.findall(answer) if 1 <= int(n) <= len(results)})
    if not results:
        return
    if not cited:
        if "không tìm thấy" not in answer.lower():
            st.warning("⚠️ Agent không trích dẫn [n] — không truy vết được câu trả lời lấy từ đâu.")
        return
    with st.expander(f"📎 Nguồn trích dẫn: {', '.join(f'[{n}]' for n in cited)}", expanded=not compact):
        for n in cited:
            r = results[n - 1]
            meta = r["metadata"]
            hit = _contains(r["content"], must_contain)
            url = meta.get("source_url")
            link = f"[{meta.get('doc_id')}]({url})" if url else f"`{meta.get('doc_id')}`"
            st.markdown(
                f"**[{n}]** {link} · audience=`{meta.get('audience')}` · chunk {meta.get('chunk_index')} · score {r['score']:.3f}"
                + ("  ✅ chunk này chứa gold answer" if hit else ("  ⚠️ chunk này KHÔNG chứa gold answer" if must_contain else ""))
            )
            st.caption(r["content"][:300].replace("\n", " ") + ("…" if len(r["content"]) > 300 else ""))


def score_table(strategies: list[str], top_k: int, fp: str) -> pd.DataFrame:
    rows = []
    for name in strategies:
        s_store, s_n, _ = build_store(name, fp)
        row = {"Chiến lược": name, "Chunks": s_n}
        total = 0
        for key, q in Q.items():
            _, sc, _ = run_query(s_store, q, top_k)
            row[key] = sc
            total += sc
        row["Tổng /10"] = total
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Sidebar — key, chiến lược, top_k
# ============================================================================
st.sidebar.title("📚 Softmax — Lab 07")

with st.sidebar.expander("🔑 API key", expanded=not os.getenv("OPENAI_API_KEY")):
    key_in = st.text_input("OpenAI API key", type="password", placeholder="sk-… (để trống = dùng .env)")
    chat_model = st.text_input("Chat model", value=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    if key_in.strip():
        os.environ["OPENAI_API_KEY"] = key_in.strip()
        os.environ["EMBEDDING_PROVIDER"] = "openai"
    if chat_model.strip():
        os.environ["OPENAI_CHAT_MODEL"] = chat_model.strip()
    st.caption("Key chỉ giữ trong phiên chạy này, không ghi ra file.")

fp = _fingerprint()
embedder, llm = get_backends(fp)
ok_embed = "mock" not in embedder.name.lower()
st.sidebar.markdown(
    f"{'🟢' if ok_embed else '🔴'} Embedder: `{embedder.name}`  \n"
    f"{'🟢' if llm.name != 'demo' else '🔴'} LLM: `{llm.name}`  \n"
    f"📁 Corpus: `{bench.CORPUS_DIR.name}`"
)
if not ok_embed:
    st.sidebar.warning("Đang dùng mock embedder — kết quả không có ngữ nghĩa. Nhập key ở trên.")

strategy = st.sidebar.selectbox("Chiến lược chunking (tab Truy vấn / Xem chunk)", list(STRATEGIES))
top_k = st.sidebar.slider("top_k", 1, 5, 3)
store, n_chunks, per_doc = build_store(strategy, fp)
st.sidebar.metric("Số chunk trong store", n_chunks)
with st.sidebar.expander("Chunk theo tài liệu"):
    for k, v in sorted(per_doc.items()):
        st.write(f"`{k}`: {v}")

tab_demo, tab_chat, tab_query, tab_chunks, tab_notes = st.tabs(["🎬 Kịch bản demo", "💬 Chatbot", "🔍 Truy vấn tự do", "🧩 Xem chunk", "📖 Ghi chú kỹ thuật"])


# ============================================================================
# Tab 1 — Kịch bản demo chọn sẵn
# ============================================================================
SCENARIOS = {
    "1 · Metadata filter — 5 phần: 3 đối tượng · trường bất kỳ · lọc trước/sau · mất recall · lọc sai trường": "filter",
    "2 · So sánh chiến lược chunking — Q4 (bullet bị tách, ai giữ được khối?)": "chunking",
    "3 · Nguồn chính thức mâu thuẫn — Q2 (FAQ 20.000 vs faculty 10.000 VND)": "conflict",
    "4 · Cross-lingual trước/sau — Q3 chỉ corpus EN (0đ) vs có bản VI (2đ)": "failure",
    "5 · Chấm hai mức — doc_id vs chunk chứa đáp án (kết quả bị thổi phồng)": "twolevel",
    "6 · Bảng tổng hợp 3 chiến lược × 5 query": "summary",
    "7 · Ngôn ngữ — cùng câu hỏi VI/EN, cùng ngôn ngữ thắng tuyệt đối": "language",
}

with tab_demo:
    pick = st.selectbox("Chọn kịch bản", list(SCENARIOS))
    kind = SCENARIOS[pick]
    run = st.button("▶ Chạy kịch bản", type="primary")

    # ---------------------------------------------------------------- 1
    if kind == "filter":
        q = Q["Q1"]
        strat = st.selectbox("Chiến lược", MAIN3, key="s1")
        part = st.radio(
            "Phần",
            [
                "A · Cùng câu hỏi, 3 đối tượng (không lọc / student / faculty)",
                "B · Lọc theo trường bất kỳ (category, department, doc_id, …)",
                "C · Lọc TRƯỚC vs lọc SAU top-k",
                "D · Filter làm hại — mất recall",
                "E · Lọc sai trường — đáp án nằm ngoài tập lọc",
            ],
            horizontal=False,
            key="s1_part",
        )
        s_store, _, _ = build_store(strat, fp)

        if part.startswith("A"):
            st.markdown(f"**Câu hỏi:** {q['q']}  \n**Gold:** `{q['gold_doc']}` · must_contain=`{q['must_contain']}`")
            if run:
                cols = st.columns(3)
                for col, (label, flt) in zip(cols, [("❌ Không lọc", None), ("🎓 audience=student", {"audience": "student"}), ("👩‍🏫 audience=faculty", {"audience": "faculty"})]):
                    with col:
                        st.subheader(label)
                        res = s_store.search_with_filter(q["q"], top_k=top_k, metadata_filter=flt)
                        sc, why = bench.grade(res, q["gold_doc"], q["must_contain"])
                        st.caption(f"ứng viên: **{count_candidates(s_store, flt)}** chunk")
                        st.markdown(f"### {sc}/2")
                        st.caption(why + " *(gold = trang undergraduate)*")
                        render_results(res, q["must_contain"], q["gold_doc"], chars=260)
                        st.markdown("**🤖 Agent:**")
                        show_answer(agent_answer(s_store, llm, q["q"], res), res, q["must_contain"], compact=True)
                st.info(
                    "**Điểm nhấn:** Cùng một câu hỏi, đổi `audience` là đổi câu trả lời — *3 cuốn / 2 tuần* (student) hay "
                    "*5 cuốn / 1 tháng* (faculty). Không lọc thì top-3 là chunk phạt tiền của FAQ và trang faculty: similarity đo "
                    "*chủ đề mượn sách*, không đo *đúng đối tượng*. Metadata là thứ duy nhất trả lời được 'ai đang hỏi'."
                )

        elif part.startswith("B"):
            st.markdown("Dựng filter từ **bất kỳ trường metadata nào** trong front matter — kết hợp nhiều trường (AND).")
            question = st.text_input("Câu hỏi", value=q["q"], key="s1b_q")
            flt = filter_builder(s_store, key="s1b")
            if run and question.strip():
                res = s_store.search_with_filter(question, top_k=top_k, metadata_filter=flt)
                render_results(res, chars=300)
                st.markdown("**🤖 Agent:**")
                show_answer(agent_answer(s_store, llm, question, res), res, "", compact=True)
                st.info(
                    "**Điểm nhấn:** `search_with_filter` so khớp `==` trên mọi cặp key/value → lọc được theo `category` "
                    "(fees / borrowing / access / spaces / faq), `doc_id` (một file), `language`, `document_version`… "
                    "Filter càng chặt, ứng viên càng ít — xem số chunk còn lại ở trên."
                )

        elif part.startswith("C"):
            st.markdown(f"**Câu hỏi:** {q['q']} · filter `audience=student`")
            if run:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("✅ Lọc TRƯỚC rồi search (cách đúng)")
                    res = s_store.search_with_filter(q["q"], top_k=top_k, metadata_filter=q["filter"])
                    st.caption(f"{count_candidates(s_store, q['filter'])} ứng viên → top-{top_k}")
                    render_results(res, q["must_contain"], q["gold_doc"], chars=260)
                with c2:
                    st.subheader("❌ Search top-k rồi mới lọc (lỗi hay gặp)")
                    plain = s_store.search(q["q"], top_k=top_k)
                    post = [r for r in plain if r["metadata"].get("audience") == "student"]
                    st.caption(f"{len(s_store._store)} ứng viên → top-{top_k} → lọc còn **{len(post)}**")
                    st.markdown("Top-k trước khi lọc:")
                    render_results(plain, q["must_contain"], q["gold_doc"], chars=120)
                    st.markdown(f"Sau khi lọc: **{len(post)} kết quả**")
                    render_results(post, q["must_contain"], q["gold_doc"], chars=260)
                st.info(
                    "**Điểm nhấn:** Lọc sau top-k thì k slot đã bị chunk sai chiếm hết → **0 kết quả** dù store còn 13 chunk hợp lệ. "
                    "Đây là lỗi `docs/EVALUATION.md` và lab doc nhắc thẳng: *search_with_filter lọc SAU khi search thay vì trước*. "
                    "Code của nhóm lọc trước, rồi cho cả `search()` và `search_with_filter()` đi chung `_search_records()`."
                )

        elif part.startswith("D"):
            qh = "Giảng viên được mượn sách tối đa trong bao lâu?"
            st.markdown(f"**Câu hỏi:** {qh}  \n**Đáp án đúng:** *up to 6 months* (faculty mượn giáo trình) — nằm trong `borrowing-privilege` và `library-faq`, cả hai `audience=all`.")
            if run:
                c1, c2 = st.columns(2)
                for col, (label, flt) in zip((c1, c2), [("❌ Không lọc", None), ("👩‍🏫 audience=faculty", {"audience": "faculty"})]):
                    with col:
                        st.subheader(label)
                        res = s_store.search_with_filter(qh, top_k=top_k, metadata_filter=flt)
                        st.caption(f"ứng viên: **{count_candidates(s_store, flt)}** chunk")
                        render_results(res, "6 months", ["borrowing-privilege", "library-faq"], chars=260)
                        st.markdown("**🤖 Agent:**")
                        show_answer(agent_answer(s_store, llm, qh, res), res, "6 months", compact=True)
                st.info(
                    "**Điểm nhấn:** Filter `audience=faculty` **loại luôn** hai trang `audience=all` — là nơi duy nhất ghi *6 months* — "
                    "nên agent chỉ còn *one month* của graduate. Precision đổi bằng recall. Cách sửa dữ liệu: gán `audience` ở mức "
                    "section (tách bảng hạn mức thành nhiều file), hoặc cho phép filter `audience in {faculty, all}`."
                )

        elif part.startswith("E"):
            q2 = Q["Q2"]
            st.markdown(f"**Câu hỏi:** {q2['q']}  \n**Đáp án:** *20,000 VND/day* — nằm trong `library-faq` (`category=faq`), **không** nằm trong `fines-and-charges` (`category=fees`).")
            if run:
                c1, c2 = st.columns(2)
                for col, (label, flt) in zip((c1, c2), [("❌ Không lọc", None), ("💸 category=fees (nghe hợp lý!)", {"category": "fees"})]):
                    with col:
                        st.subheader(label)
                        res = s_store.search_with_filter(q2["q"], top_k=top_k, metadata_filter=flt)
                        sc, why = bench.grade(res, q2["gold_doc"], q2["must_contain"])
                        st.caption(f"ứng viên: **{count_candidates(s_store, flt)}** chunk")
                        st.markdown(f"### {sc}/2")
                        st.caption(why)
                        render_results(res, q2["must_contain"], q2["gold_doc"], chars=260)
                        st.markdown("**🤖 Agent:**")
                        show_answer(agent_answer(s_store, llm, q2["q"], res), res, q2["must_contain"], compact=True)
                st.info(
                    "**Điểm nhấn:** `category=fees` nghe rất đúng cho câu hỏi về tiền phạt, nhưng trang *Fines and other charges* "
                    "chỉ nói về phí hư hỏng — con số 20.000 VND/ngày lại ở FAQ. Metadata chỉ tốt khi **schema khớp với câu hỏi thật**; "
                    "gán nhãn theo tiêu đề trang mà không đọc nội dung là bẫy."
                )

    # ---------------------------------------------------------------- 2
    elif kind == "chunking":
        q = Q["Q4"]
        st.markdown(f"**Câu hỏi:** {q['q']}  \n**Gold:** `{q['gold_doc']}` · must_contain=`{q['must_contain']}`")
        if run:
            cols = st.columns(3)
            for col, name in zip(cols, MAIN3):
                with col:
                    s_store, s_n, _ = build_store(name, fp)
                    res, sc, why = run_query(s_store, q, top_k)
                    st.subheader(name.split(" — ")[0])
                    st.caption(f"{s_n} chunks · {name.split(' — ')[1]}")
                    st.markdown(f"### {sc}/2")
                    st.caption(why)
                    render_results(res, q["must_contain"], q["gold_doc"], chars=350)
            st.info(
                "**Điểm nhấn:** Đáp án là **một bullet** trong danh sách quy định phòng học. Recursive cắt ở `\\n\\n` rồi `\\n` nên bullet "
                "bị tách từng dòng; FixedSize cắt mù 500 ký tự — cả hai đều để chunk *'Phải có ít nhất 2 người…'* (từ vựng gần câu hỏi: nhóm, buổi) "
                "lên top-1, còn bullet *'2 giờ mỗi buổi, 4 buổi mỗi tuần'* rớt xuống hạng 3. Heading giữ **cả khối bullet** dưới `## Phòng học nhóm` "
                "nên top-1 chứa đáp án → 2/2. Thứ quyết định không phải chunker 'thông minh' mà là *khối thông tin có bị tách khỏi ngữ cảnh gần nó không*."
            )

    # ---------------------------------------------------------------- 3
    elif kind == "conflict":
        q = Q["Q2"]
        st.markdown(f"**Câu hỏi:** {q['q']}  \n**Gold:** `{q['gold_doc']}` · must_contain=`{q['must_contain']}`")
        strat = st.selectbox("Chiến lược", MAIN3, key="s3")
        if run:
            s_store, _, _ = build_store(strat, fp)
            res, sc, why = run_query(s_store, q, top_k)
            st.markdown(f"### Điểm retrieval: {sc}/2 — {why}")
            render_results(res, q["must_contain"], q["gold_doc"])
            st.subheader("🤖 Agent answer")
            ans = agent_answer(s_store, llm, q["q"], res)
            show_answer(ans, res, q["must_contain"])
            has10 = "10,000" in ans or "10.000" in ans
            has20 = "20,000" in ans or "20.000" in ans
            verdict = "agent lấy **10.000 VND** từ trang faculty (chunk có 'per business day')" if has10 and not has20 else \
                      "agent lấy **20.000 VND** từ FAQ" if has20 and not has10 else "agent nêu cả hai / không rõ"
            st.markdown(f"**Nhận xét:** {verdict}.")
            st.info(
                "**Điểm nhấn:** Hai trang chính thức của cùng thư viện nói hai con số khác nhau (FAQ: 20.000 VND/ngày; "
                "trang graduate/faculty: 10.000 VND/business day). Retrieval vẫn được 2/2 vì top-1 chứa đáp án gold, "
                "nhưng agent có thể chọn chunk khác. Corpus không phân xử được vì `document_version` cả hai đều `not-stated`. "
                "Bài học: top-3 đúng chưa đủ, phải đọc agent answer; `document_version` không phải trường hình thức."
            )

    # ---------------------------------------------------------------- 4
    elif kind == "failure":
        q = Q["Q3"]
        st.markdown(
            f"**Câu hỏi:** {q['q']}  \n**Đáp án:** *overdue for more than 05 days* / *quá hạn hơn 05 ngày* — có trong `equipment-loans` (EN) và `equipment-loans-vi` (VI)."
        )
        st.markdown("Trước khi dịch, corpus chỉ có tiếng Anh và Q3 **0đ ở cả 3 chiến lược**. Tái hiện bằng filter `language=en`, rồi so với corpus song ngữ.")
        strat = st.selectbox("Chiến lược", MAIN3, key="s4")
        if run:
            s_store, _, _ = build_store(strat, fp)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🇬🇧 Chỉ corpus EN (`language=en`) — như trước khi dịch")
                res = s_store.search_with_filter(q["q"], top_k=top_k, metadata_filter={"language": "en"})
                sc, why = bench.grade(res, q["gold_doc"], q["must_contain"])
                st.markdown(f"### {sc}/2")
                st.caption(why)
                render_results(res, q["must_contain"], q["gold_doc"], chars=280)
                st.markdown("**🤖 Agent:**")
                show_answer(agent_answer(s_store, llm, q["q"], res), res, q["must_contain"], compact=True)
            with c2:
                st.subheader("🇻🇳🇬🇧 Corpus song ngữ (không filter)")
                res2 = s_store.search_with_filter(q["q"], top_k=top_k, metadata_filter=None)
                sc2, why2 = bench.grade(res2, q["gold_doc"], q["must_contain"])
                st.markdown(f"### {sc2}/2")
                st.caption(why2)
                render_results(res2, q["must_contain"], q["gold_doc"], chars=280)
                st.markdown("**🤖 Agent:**")
                show_answer(agent_answer(s_store, llm, q["q"], res2), res2, q["must_contain"], compact=True)
            st.info(
                "**Điểm nhấn:** Câu hỏi tiếng Việt *'quá hạn bao nhiêu ngày thì bị coi là mất'* trên corpus tiếng Anh: chunk *'fined for returning "
                "items late… damaged or lost'* của trang faculty gần nghĩa hơn chunk *'overdue for more than 05 days'* — score chỉ ~0.3, "
                "chunk có đáp án không lọt top-3. Thêm bản dịch VI: score nhảy lên ~0.7, top-1 chứa đáp án ngay. "
                "Lỗi này **không nằm ở chunker** (cả 3 chiến lược cùng 0đ trước đó) mà ở khoảng cách ngôn ngữ giữa query và corpus."
            )

    # ---------------------------------------------------------------- 5
    elif kind == "twolevel":
        strat = st.selectbox("Chiến lược", MAIN3, key="s5")
        if run:
            s_store, _, _ = build_store(strat, fp)
            rows = []
            for key, q in Q.items():
                res, sc_chunk, why = run_query(s_store, q, top_k)
                sc_doc = grade_doc_level(res, q["gold_doc"])
                rows.append({"Query": key, "Chấm theo doc_id": sc_doc, "Chấm theo chunk có đáp án": sc_chunk, "Ghi chú": why})
            df = pd.DataFrame(rows)
            tot_doc, tot_chunk = int(df["Chấm theo doc_id"].sum()), int(df["Chấm theo chunk có đáp án"].sum())
            c1, c2 = st.columns(2)
            c1.metric("Tổng chấm theo doc_id", f"{tot_doc}/10")
            c2.metric("Tổng chấm theo chunk", f"{tot_chunk}/10", delta=tot_chunk - tot_doc)
            st.dataframe(df, width="stretch", hide_index=True)
            st.info(
                "**Điểm nhấn:** Chỉ kiểm `doc_id` gold có trong top-3 sẽ thổi phồng kết quả — một chiến lược có thể lấy trọn 3 slot "
                "từ đúng file mà không chunk nào chứa câu trả lời (Q1, Q4, Q5: top-3 toàn đúng file, chunk có số liệu ở hạng 2–3). "
                "`docs/SCORING.md` yêu cầu *top-3 có chunk liên quan **và** agent trả lời đúng*, nên phải chấm ở mức nội dung."
            )

    # ---------------------------------------------------------------- 6
    elif kind == "summary":
        chosen = st.multiselect("Chiến lược", list(STRATEGIES), default=MAIN3, key="s6")
        if run and chosen:
            df = score_table(chosen, top_k, fp)
            st.dataframe(df, width="stretch", hide_index=True)
            st.bar_chart(df.set_index("Chiến lược")["Tổng /10"])
            with st.expander("5 benchmark query"):
                for key, q in Q.items():
                    st.markdown(f"**{key}.** {q['q']}  \n gold=`{q['gold_doc']}` · must_contain=`{q['must_contain']}` · filter=`{q['filter']}`")
            st.info(
                "**Điểm nhấn:** Cùng 16 file (8 EN + 8 VI), cùng 5 câu, chỉ đổi một dòng chunker mà điểm dao động 7–9/10. "
                "Heading thắng vì giữ trọn khối bullet/mục quy định dưới tiêu đề (Q1, Q4 top-1). Trước khi có bản VI, điểm là 4/7/7 và Q3 "
                "0đ ở cả ba — thêm dữ liệu cùng ngôn ngữ với query nâng mọi chiến lược lên, nhiều hơn bất kỳ thay đổi chunker nào."
            )


    # ---------------------------------------------------------------- 7
    elif kind == "language":
        st.markdown(
            "Corpus có **8 trang × 2 ngôn ngữ** (bản EN gốc + bản VI dịch, cùng `source_url`, `translated_from` trỏ về nhau). "
            "Hỏi cùng một câu bằng hai thứ tiếng, có/không ép `language` để thấy embedding ưu tiên ngôn ngữ đến mức nào."
        )
        strat = st.selectbox("Chiến lược", MAIN3, key="s7")
        q_vi = st.text_input("Câu hỏi tiếng Việt", value=Q["Q1"]["q"], key="s7_vi")
        q_en = st.text_input("Câu hỏi tiếng Anh", value="How many books can I borrow and for how long?", key="s7_en")
        base = {"audience": "student"}
        if run:
            s_store, _, _ = build_store(strat, fp)
            cells = [
                ("🇻🇳 hỏi VI · không ép ngôn ngữ", q_vi, dict(base)),
                ("🇻🇳 hỏi VI · ép `language=en`", q_vi, {**base, "language": "en"}),
                ("🇬🇧 hỏi EN · không ép ngôn ngữ", q_en, dict(base)),
                ("🇬🇧 hỏi EN · ép `language=vi`", q_en, {**base, "language": "vi"}),
            ]
            rows = []
            r1, r2 = st.columns(2), st.columns(2)
            for col, (label, qq, flt) in zip(list(r1) + list(r2), cells):
                with col:
                    st.subheader(label)
                    res = s_store.search_with_filter(qq, top_k=top_k, metadata_filter=flt)
                    langs = [r["metadata"].get("language") for r in res]
                    top = res[0]["score"] if res else 0.0
                    st.caption(f"ứng viên: {count_candidates(s_store, flt)} · top-1 score **{top:.3f}** · ngôn ngữ top-{top_k}: `{langs}`")
                    render_results(res, "", None, chars=200)
                    rows.append({"Trường hợp": label, "ứng viên": count_candidates(s_store, flt), "top-1 score": round(top, 3), "ngôn ngữ top-k": ",".join(langs)})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.info(
                "**Điểm nhấn:** Không ép ngôn ngữ, top-3 **luôn cùng ngôn ngữ với câu hỏi** (VI → toàn chunk VI, EN → toàn chunk EN) với score ~0.6–0.7; "
                "ép sang ngôn ngữ kia score rớt còn ~0.25–0.4 nhưng vẫn tìm đúng trang → cross-lingual *có* hoạt động, chỉ yếu hơn nhiều. "
                "Hệ quả: trong corpus song ngữ, bản dịch là thứ quyết định chất lượng retrieval cho người dùng tiếng Việt; "
                "`language` là trường lọc thật (không phải để cho có) khi muốn agent trích dẫn đúng bản gốc."
            )

# ============================================================================
# Tab 2 — Truy vấn tự do
# ============================================================================
with tab_query:
    presets = {f"{k}: {q['q']}": q for k, q in Q.items()}
    choice = st.selectbox("Chọn benchmark query hoặc tự gõ", ["(tự gõ)"] + list(presets))
    preset = presets.get(choice)
    question = st.text_input("Câu hỏi", value=preset["q"] if preset else "")
    flt_free = filter_builder(store, key=f"free_{choice}", default=preset["filter"] if preset else None)
    must = preset["must_contain"] if preset else ""
    gold = preset["gold_doc"] if preset else None
    if preset:
        st.caption(f"gold_doc = `{gold}` · must_contain = `{must}`")

    if st.button("Chạy truy vấn", type="primary", disabled=not question.strip()):
        flt = flt_free
        filtered = store.search_with_filter(question, top_k=top_k, metadata_filter=flt)
        if flt:
            c_a, c_b = st.columns(2)
            with c_a:
                st.subheader(f"Có filter `{flt}`")
                if preset:
                    sc, why = bench.grade(filtered, gold, must)
                    st.markdown(f"**Điểm: {sc}/2** — {why}")
                render_results(filtered, must, gold)
            with c_b:
                st.subheader("Không filter (A/B)")
                plain = store.search(question, top_k=top_k)
                if preset:
                    sc, why = bench.grade(plain, gold, must)
                    st.markdown(f"**Điểm: {sc}/2** — {why}")
                render_results(plain, must, gold)
        else:
            if preset:
                sc, why = bench.grade(filtered, gold, must)
                st.markdown(f"**Điểm: {sc}/2** — {why}")
            render_results(filtered, must, gold)

        st.subheader("🤖 Agent answer")
        ans = agent_answer(store, llm, question, filtered)
        show_answer(ans, filtered, must)
        if filtered:
            with st.expander("Prompt đã gửi"):
                st.code(KnowledgeBaseAgent(store=store, llm_fn=llm).build_prompt(question, filtered))


# ============================================================================
# Tab 3 — Xem chunk
# ============================================================================
def _overlap_len(prev: str, cur: str) -> int:
    """Số ký tự đầu của `cur` trùng với đuôi của `prev` (overlap thật giữa 2 chunk liền kề)."""
    m = min(len(prev), len(cur))
    for k in range(m, 9, -1):  # bỏ qua trùng dưới 10 ký tự (khoảng trắng, dấu câu)
        if prev.endswith(cur[:k]):
            return k
    return 0


def _locate(body: str, chunk: str) -> tuple[int, int]:
    """Vị trí chunk trong văn bản gốc; Heading gắn lại tiêu đề nên thử phần sau dòng đầu."""
    i = body.find(chunk)
    if i >= 0:
        return i, i + len(chunk)
    if "\n" in chunk:
        rest = chunk.split("\n", 1)[1].strip()
        i = body.find(rest)
        if i >= 0:
            return i, i + len(rest)
    return -1, -1


def _render_chunk_html(text: str, overlap_n: int, heading_repeat: str | None, hl: str) -> str:
    """Tô vàng phần overlap với chunk trước, xanh heading gắn lại, đỏ chuỗi tìm kiếm."""
    def esc(t: str) -> str:
        return _html.escape(t).replace("\n", "<br>")

    parts = []
    pos = 0
    if heading_repeat and text.startswith(heading_repeat):
        parts.append(f'<span style="background:#cde8ff;border-radius:3px">{esc(heading_repeat)}</span>')
        pos = len(heading_repeat)
    if overlap_n > pos:
        parts.append(f'<span style="background:#ffe08a;border-radius:3px">{esc(text[pos:overlap_n])}</span>')
        pos = overlap_n
    parts.append(esc(text[pos:]))
    out = "".join(parts)
    if hl:
        out = re.sub(re.escape(_html.escape(hl)), lambda m: f'<mark style="background:#ffb3b3">{m.group(0)}</mark>', out, flags=re.IGNORECASE)
    return f'<div style="font-family:monospace;font-size:0.85em;white-space:pre-wrap;line-height:1.45">{out}</div>'


with tab_chunks:
    doc_pick = st.selectbox("Tài liệu", sorted(per_doc))
    chunks = [r for r in store._store if r["metadata"]["doc_id"] == doc_pick]
    body_path = bench.CORPUS_DIR / f"{doc_pick}.md"
    _, body = bench.parse_front_matter(body_path.read_text(encoding="utf-8")) if body_path.exists() else ({}, "")
    n = len(chunks)
    avg = sum(len(c["content"]) for c in chunks) / max(1, n)

    # --- overlap giữa các chunk liền kề + heading lặp lại
    overlaps = [0]
    head_rep: list[str | None] = [None]
    for i in range(1, n):
        prev, cur = chunks[i - 1]["content"], chunks[i]["content"]
        overlaps.append(_overlap_len(prev, cur))
        h_prev = prev.split("\n", 1)[0]
        head_rep.append(h_prev if (h_prev.startswith("#") and cur.startswith(h_prev)) else None)
    total_ov = sum(overlaps)
    n_head = sum(1 for h in head_rep if h)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số chunk", n)
    m2.metric("Độ dài TB", f"{avg:.0f} ký tự")
    m3.metric("Ký tự overlap (tổng)", total_ov, help="Phần đầu chunk i trùng với phần đuôi chunk i−1")
    m4.metric("Heading gắn lại", n_head, help="Chunk bắt đầu bằng đúng tiêu đề của chunk trước (HeadingChunker)")
    st.caption(f"Chiến lược: **{strategy}** · văn bản gốc {len(body)} ký tự · tổng ký tự trong chunk {sum(len(c['content']) for c in chunks)} "
               f"(= gốc + overlap + heading lặp)")

    # --- bản đồ vị trí chunk trên văn bản gốc
    if body:
        st.markdown("**Bản đồ vị trí chunk trên văn bản gốc** — mỗi thanh là một chunk; chỗ hai thanh chồng lên nhau là overlap (đậm hơn).")
        L = max(1, len(body))
        bars = []
        colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
        for i, r in enumerate(chunks):
            a, b = _locate(body, r["content"])
            if a < 0:
                continue
            left, width = 100 * a / L, max(0.3, 100 * (b - a) / L)
            top = 0 if i % 2 == 0 else 14
            bars.append(
                f'<div title="chunk {i}: {a}–{b} ({b-a} ký tự)" style="position:absolute;left:{left:.2f}%;width:{width:.2f}%;top:{top}px;height:12px;'
                f'background:{colors[i % len(colors)]};opacity:0.75;border-radius:2px"></div>'
            )
        st.markdown(
            f'<div style="position:relative;height:30px;background:#f0f0f0;border-radius:4px;margin:4px 0 12px 0">{"".join(bars)}</div>',
            unsafe_allow_html=True,
        )

    hl = st.text_input("Tô đỏ chuỗi (ví dụ: 2 giờ mỗi buổi / 2 hours per session)", value="")
    st.markdown(
        '<span style="background:#ffe08a;padding:0 4px;border-radius:3px">vàng = overlap với chunk trước</span> &nbsp; '
        '<span style="background:#cde8ff;padding:0 4px;border-radius:3px">xanh = heading gắn lại</span> &nbsp; '
        '<span style="background:#ffb3b3;padding:0 4px;border-radius:3px">đỏ = chuỗi tìm</span>',
        unsafe_allow_html=True,
    )
    for i, r in enumerate(chunks):
        found = bool(hl) and hl.lower() in r["content"].lower()
        tag = " ✅" if found else ""
        ov = f" · overlap {overlaps[i]}" if overlaps[i] else ""
        hd = " · heading lặp" if head_rep[i] else ""
        with st.expander(f"chunk {r['metadata']['chunk_index']} · {len(r['content'])} ký tự{ov}{hd} · {r['content'][:55].replace(chr(10), ' ')}…{tag}", expanded=found):
            st.markdown(_render_chunk_html(r["content"], overlaps[i], head_rep[i], hl), unsafe_allow_html=True)


# ============================================================================
# Tab — Chatbot: hỏi đáp tự do trên corpus, mỗi lượt = 1 vòng RAG đầy đủ
# ============================================================================
SUGGESTED = [
    "Sinh viên được mượn bao nhiêu sách và trong bao lâu?",
    "Giảng viên mượn giáo trình được tối đa bao lâu?",
    "Trả sách muộn bị phạt bao nhiêu?",
    "Thư viện mở cửa mấy giờ vào cuối tuần?",
    "Mượn laptop của thư viện được bao lâu?",
    "Đặt phòng học nhóm bằng cách nào?",
    "Làm mất sách thì phải làm gì?",
    "Có được mang tài liệu tham khảo về nhà không?",
]


def _pick_filter_from_question(question: str) -> dict | None:
    """Gợi ý filter đơn giản từ từ khoá trong câu hỏi (chỉ để demo, người dùng có thể tắt)."""
    q = question.lower()
    if any(w in q for w in ["giảng viên", "faculty", "cao học", "sau đại học", "graduate"]):
        return {"audience": "faculty"}
    if any(w in q for w in ["sinh viên", "student", "undergraduate"]):
        return {"audience": "student"}
    return None


with tab_chat:
    st.markdown(
        "Hỏi bất kỳ điều gì về thư viện VinUni. Mỗi lượt là một vòng **RAG đầy đủ**: embed câu hỏi → lọc metadata (nếu có) → "
        "top-k chunk → prompt có trích dẫn → LLM. Mở *Nguồn trích dẫn* dưới mỗi câu trả lời để truy vết."
    )
    cc1, cc2, cc3 = st.columns([2, 2, 1])
    auto_filter = cc1.toggle("Tự gợi ý filter `audience` từ câu hỏi (bật để thấy filter làm mất recall)", value=False, key="chat_auto")
    chat_filter = None
    with cc2:
        manual = st.checkbox("Đặt filter thủ công", value=False, key="chat_manual")
    if cc3.button("🗑️ Xoá hội thoại", key="chat_clear"):
        st.session_state["chat_history"] = []
        st.rerun()
    if manual:
        chat_filter = filter_builder(store, key="chat")

    with st.expander("💡 Câu hỏi gợi ý"):
        cols = st.columns(2)
        for i, sq in enumerate(SUGGESTED):
            if cols[i % 2].button(sq, key=f"sug_{i}", width="stretch"):
                st.session_state["chat_pending"] = sq

    history: list[dict] = st.session_state.setdefault("chat_history", [])
    for turn in history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            st.caption(f"chiến lược `{turn['strategy']}` · filter `{turn['filter']}` · {turn['n_cand']} ứng viên · top-{turn['top_k']}")
            show_answer(turn["answer"], turn["results"], "", compact=True)

    pending = st.session_state.pop("chat_pending", None)
    typed = st.chat_input("Nhập câu hỏi…", key="chat_input")
    question = typed or pending
    if question:
        flt = chat_filter if manual else (_pick_filter_from_question(question) if auto_filter else None)
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Đang truy xuất + hỏi LLM…"):
                results = store.search_with_filter(question, top_k=top_k, metadata_filter=flt)
                answer = agent_answer(store, llm, question, results)
            st.caption(f"chiến lược `{strategy}` · filter `{flt}` · {count_candidates(store, flt)} ứng viên · top-{top_k}")
            show_answer(answer, results, "", compact=True)
            with st.expander(f"🔎 Top-{top_k} chunk đã dùng"):
                render_results(results, "", None, chars=300)
        history.append({
            "q": question, "answer": answer, "results": results, "filter": flt,
            "strategy": strategy, "top_k": top_k, "n_cand": count_candidates(store, flt),
        })


# ============================================================================
# Tab 4 — Ghi chú: data, pipeline, chunking, chấm điểm
# ============================================================================
with tab_notes:
    st.markdown("## Đường đi của một câu hỏi")
    st.code(
        "File .md ──► Chunker.chunk() ──► Document(id='file#i', content, metadata)\n"
        "                                        │  (metadata của file sao vào MỌI chunk)\n"
        "                                        ▼\n"
        "                    EmbeddingStore.add_documents()  → embed từng chunk (1536 chiều)\n"
        "                                        │\n"
        "Câu hỏi ──► embed ──► search_with_filter(metadata_filter) → lọc TRƯỚC → dot product → top-k\n"
        "                                        │\n"
        "                    KnowledgeBaseAgent.build_prompt() → [1][2][3] + doc_id → gpt-4o-mini\n",
        language="text",
    )

    n1, n2, n3 = st.tabs(["📁 Data", "🔪 Chunking", "🎯 Embedding · Search · Chấm"])

    with n1:
        st.markdown(
            f"""
**Corpus:** `{bench.CORPUS_DIR.name}` — {len(per_doc)} file = 8 trang công khai của `library.vinuni.edu.vn` × 2 ngôn ngữ (EN gốc + VI dịch, `translated_from` trỏ về nhau), crawl 2026-09-19 bằng
`scripts/fetch_public_pages.py` (kiểm `robots.txt`, chờ ≥1 s/request), sau đó **làm sạch tay**: bỏ menu/footer (~70 % output thô),
chuyển bảng HTML → Markdown, giữ nguyên điều khoản + con số + mốc thời gian.

**Vì sao chọn thư viện:** (1) robots.txt cho phép, (2) nhiều số liệu kiểm chứng được, (3) có **2 trang cùng câu chữ, khác đối tượng,
khác đáp án** — `borrowing-undergraduate-staff` (3 items / 2 weeks) và `borrowing-graduate-faculty` (5 items / 1 month).
Đây là bẫy cố ý để chứng minh khi nào *bắt buộc* phải có metadata filter.
"""
        )
        rows = []
        for path in sorted(bench.CORPUS_DIR.glob("*.md")):
            meta, body = bench.parse_front_matter(path.read_text(encoding="utf-8"))
            rows.append({
                "doc_id": meta.get("doc_id"), "audience": meta.get("audience"), "category": meta.get("category"),
                "ký tự": len(body), f"chunks ({strategy.split(' — ')[0]})": per_doc.get(path.stem, 0),
                "document_version": meta.get("document_version"), "source_url": meta.get("source_url"),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown(
            """
**Front matter** (đầu mỗi file) = metadata, được `parse_front_matter()` tách ra và **sao vào mọi chunk** của file đó:
```yaml
---
doc_id: borrowing-undergraduate-staff   # trỏ chunk về file gốc; delete_document + chấm điểm dựa vào nó
audience: student                       # student | faculty | staff | all  ← trường lọc chính
category: borrowing                     # borrowing | fees | access | spaces | faq
source_url: https://library.vinuni.edu.vn/...   # truy vết
retrieved_at: 2026-09-19
document_version: not-stated            # không bịa số hiệu khi trang không nêu
---
```
Đã loại 2 trang sau khi crawl: `how-to-borrow-return-renew` (chỉ tiêu đề video) và `borrowing-vingroup-community` (không số liệu, trùng nội dung).
Trang gốc chỉ có tiếng Anh; nhóm dịch tay 8 bản VI (giữ nguyên mọi con số) để (1) query tiếng Việt có chunk cùng ngôn ngữ, (2) `language` thành trường lọc thật. Trước khi dịch, Q3 0đ ở cả 3 chiến lược.
"""
        )

    with n2:
        st.markdown(
            """
Embedding không hiểu cả file 8 KB → phải cắt thành mẩu 300–800 ký tự. **Cắt ở đâu** quyết định retrieval.
Mỗi thành viên một chiến lược, chỉ đổi **một dòng** `CHUNKER = ...` trong `bench.py`:

| Chiến lược | Cách cắt | Điểm mạnh | Điểm yếu |
|---|---|---|---|
| **FixedSize(500, overlap=50)** — Thiên | Đếm 500 ký tự thì cắt, bất kể giữa câu; chunk sau lấy lại 50 ký tự cuối chunk trước | Overlap = mỗi ranh giới xuất hiện 2 lần → thông tin sát ranh giới vẫn có chunk chứa trọn | Cắt mù giữa câu / giữa hàng bảng; chunk khó đọc |
| **Recursive(500)** — Huy | Thử `\\n\\n` → `\\n` → `. ` → ` ` → cắt cứng; mảnh nhỏ liền kề gom lại đến sát 500. **Không overlap** | Giữ trọn từng mục FAQ, từng đoạn | Bullet list bị tách từng dòng; ranh giới gom có thể rơi giữa 2 bullet → mỗi thông tin chỉ có 1 cơ hội |
| **Heading(800)** — Phong | Cắt tại mỗi dòng `#`/`##`; section > 800 thì hạ xuống Recursive nhưng **gắn lại heading** lên đầu mỗi mảnh | Section = đơn vị ngữ nghĩa do người soạn chia sẵn; chunk tự mô tả "đây là mục gì" | Chỉ tốt với văn bản có cấu trúc mục; FAQ 22 mục / bảng 25 phòng vẫn phải cắt tiếp |
| Sentence(3 câu) | Regex `(?<=[.!?])\\s+`, gom 3 câu | Chunk ngắn, sạch | Bảng/bullet không có dấu chấm → 1 chunk 2 472 ký tự |

**Ví dụ Q4** — đáp án *"2 hours per session, 2 sessions per day, 4 sessions per week"* nằm trong danh sách bullet của `room-booking`:
"""
        )
        st.code(
            "Recursive:  [chunk 5: ...Reservation... - Each group can book up to 2 hours per session...]  ← có đáp án, KHÔNG lọt top-3\n"
            "            [chunk 6: - Study rooms are for group study only. At least 2 people...]          ← lọt top-1, không có đáp án\n"
            "FixedSize:  [chunk 3: ...2 hours per session, 2 sessions per day...Reserv]\n"
            "            [chunk 4: ...Reservations may be made... At least 2 people...]                  ← overlap kéo bullet sang\n"
            "Heading:    [chunk: ## Book a library Study Room ... 2 hours per session ... At least 2 people ...]  ← trọn khối",
            language="text",
        )
        st.markdown("→ Mở tab **🧩 Xem chunk**, chọn `room-booking`, gõ `2 hours` vào ô tô sáng để thấy trực tiếp với chiến lược đang chọn.")

    with n3:
        st.markdown(
            f"""
**Embedding.** Mỗi chunk và mỗi câu hỏi → OpenAI `text-embedding-3-small` → vector 1536 chiều, đã chuẩn hoá (‖v‖ = 1) nên
**cosine = dot product**. Cùng nghĩa → cùng hướng → score gần 1. Cache `.embed_cache.json` theo SHA-256 nội dung → chạy lại 0 API call.

Điều embedding **làm được**: "mượn tối đa bao nhiêu cuốn" (VI) ≈ "may borrow up to 3 items" (EN).
Điều embedding **không làm được**: phân biệt *"Sinh viên được mượn 5 cuốn"* với *"Giảng viên được mượn 20 cuốn"* — similarity **0.863**,
cao nhất trong 5 cặp thử. Embedding thấy *chủ đề*, không thấy *đối tượng / số liệu* → cần metadata.

**Search.** `search()` = embed câu hỏi → dot product với mọi chunk → sort → top-k.
`search_with_filter(metadata_filter={{"audience": "student"}})` = **lọc trước** (chỉ giữ chunk `audience == student`) rồi mới search.
Lọc *sau* top-k thì 3 slot có thể đã bị chunk sai chiếm hết → 0 kết quả dù store còn tài liệu hợp lệ.

**Agent (RAG).** top-3 chunk → prompt: quy tắc (*chỉ dùng ngữ cảnh, không bịa, trích dẫn số hiệu*) + chunk đánh số `[1][2][3]` kèm `doc_id`
+ câu hỏi → `{llm.name}`. Số `[3]` trong câu trả lời trỏ về chunk 3 → **truy vết được** câu trả lời lấy từ file nào.

**Chấm điểm — hai mức.** Mỗi query khai báo `gold_doc` (file chứa đáp án) và `must_contain` (chuỗi đặc trưng, vd `"2 hours per session"`).

| Cách chấm | Điều kiện 2đ | Recursive (Huy) |
|---|---|---|
| Theo `doc_id` (ngây thơ) | gold_doc ở top-1 | **10/10** |
| Theo chunk (thật, dùng trong bench) | chunk vừa đúng file *vừa chứa* `must_contain` ở top-1; hạng 2–3 = 1đ | **7/10** |

Chênh 3 điểm = những lần lấy **đúng file nhưng sai mẩu**. `docs/SCORING.md` yêu cầu *top-3 có chunk liên quan **và** agent trả lời đúng* → phải chấm ở mức nội dung.

**Ba con số cần nhớ:** `7 / 7 / 9` (Recursive / Fixed / Heading, corpus song ngữ; trước khi dịch là `4 / 7 / 7`) · `0 → 2` (Q1 không / có filter) · `0.25 → 0.6` (score khác / cùng ngôn ngữ).
"""
        )
