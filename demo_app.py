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


def render_results(results: list[dict], must_contain: str = "", gold=None, chars: int = 600) -> None:
    if not results:
        st.info("Không có kết quả (filter loại hết ứng viên?)")
        return
    golds = set() if gold is None else ({gold} if isinstance(gold, str) else set(gold))
    top = max(r["score"] for r in results) or 1.0
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        hit = bool(must_contain) and must_contain.lower() in r["content"].lower()
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


def show_answer(answer: str, results: list[dict], must_contain: str = "", compact: bool = False) -> None:
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
            hit = bool(must_contain) and must_contain.lower() in r["content"].lower()
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

tab_demo, tab_query, tab_chunks, tab_notes = st.tabs(["🎬 Kịch bản demo", "🔍 Truy vấn tự do", "🧩 Xem chunk", "📖 Ghi chú kỹ thuật"])


# ============================================================================
# Tab 1 — Kịch bản demo chọn sẵn
# ============================================================================
SCENARIOS = {
    "1 · Metadata filter — 5 phần: 3 đối tượng · trường bất kỳ · lọc trước/sau · mất recall · lọc sai trường": "filter",
    "2 · So sánh chiến lược chunking — Q4 (bullet bị tách, ai giữ được khối?)": "chunking",
    "3 · Nguồn chính thức mâu thuẫn — Q2 (FAQ 20.000 vs faculty 10.000 VND)": "conflict",
    "4 · Failure case cross-lingual — Q3 (cả 3 chiến lược 0đ)": "failure",
    "5 · Chấm hai mức — doc_id vs chunk chứa đáp án (kết quả bị thổi phồng)": "twolevel",
    "6 · Bảng tổng hợp 3 chiến lược × 5 query": "summary",
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
                "**Điểm nhấn:** Recursive cắt ở `\\n\\n` rồi `\\n` nên danh sách bullet bị tách từng dòng; ranh giới chunk rơi giữa "
                "bullet *2 hours per session…* và bullet *Study rooms are for group study only…*. Chunk sau có từ vựng gần câu hỏi hơn "
                "(group, session) nên lọt top-1, chunk có số liệu rớt. Không overlap = mỗi thông tin chỉ có một cơ hội. "
                "FixedSize dùng overlap 50 để bullet xuất hiện ở 2 chunk; Heading giữ cả khối bullet dưới `## Study rooms`."
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
        st.markdown(f"**Câu hỏi:** {q['q']}  \n**Gold:** `{q['gold_doc']}` · must_contain=`{q['must_contain']}`")
        if run:
            cols = st.columns(3)
            for col, name in zip(cols, MAIN3):
                with col:
                    s_store, _, _ = build_store(name, fp)
                    res, sc, why = run_query(s_store, q, top_k)
                    st.subheader(name.split(" — ")[0])
                    st.markdown(f"### {sc}/2")
                    st.caption(why)
                    render_results(res, q["must_contain"], q["gold_doc"], chars=300)
            st.markdown("**Chunk lẽ ra phải được lấy** (trong `equipment-loans`):")
            s_store, _, _ = build_store(MAIN3[0], fp)
            target = [r for r in s_store._store if r["metadata"]["doc_id"] == "equipment-loans" and "05 days" in r["content"]]
            if target:
                st.code(target[0]["content"][:500])
            st.info(
                "**Điểm nhấn:** Cả ba chiến lược đều 0đ → lỗi không nằm ở chunker. Query tiếng Việt *'quá hạn bao nhiêu ngày thì bị coi là mất'* "
                "so với corpus tiếng Anh *'overdue for more than 05 days will be considered lost'* — chunk *'fined for returning items late… damaged or lost'* "
                "của trang faculty gần nghĩa hơn về chủ đề. Cách sửa: viết lại query sát từ vựng nguồn, hoặc tách `equipment-loans` "
                "thành chunk nhỏ hơn có câu chứa số liệu đứng đầu."
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
                "từ đúng file mà không chunk nào chứa câu trả lời (Q1, Q5: cả 3 chunk đúng file, chunk có số liệu ở hạng 2–3). "
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
                "**Điểm nhấn:** Cùng 8 file, cùng 5 câu, chỉ đổi một dòng chunker mà điểm dao động 4–7/10. Thứ quyết định là "
                "**khối thông tin có bị tách khỏi ngữ cảnh gần nó không**: overlap (Fixed) hoặc ranh giới người soạn (Heading) giữ được, "
                "Recursive không overlap thì không. Q3 thua ở cả ba → giới hạn ở query/corpus."
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
with tab_chunks:
    doc_pick = st.selectbox("Tài liệu", sorted(per_doc))
    chunks = [r for r in store._store if r["metadata"]["doc_id"] == doc_pick]
    st.caption(f"{strategy} · {len(chunks)} chunk · độ dài TB {sum(len(c['content']) for c in chunks) / max(1, len(chunks)):.0f} ký tự")
    hl = st.text_input("Tô sáng chunk chứa chuỗi", value="")
    for r in chunks:
        mark = " ✅" if hl and hl.lower() in r["content"].lower() else ""
        with st.expander(f"chunk {r['metadata']['chunk_index']} · {len(r['content'])} ký tự · {r['content'][:60].replace(chr(10), ' ')}…{mark}"):
            st.text(r["content"])


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
**Corpus:** `{bench.CORPUS_DIR.name}` — {len(per_doc)} trang công khai của `library.vinuni.edu.vn`, crawl 2026-09-19 bằng
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
Corpus tiếng Anh (trang gốc chỉ có tiếng Anh); query tiếng Việt → kiểm cross-lingual retrieval.
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
| Theo `doc_id` (ngây thơ) | gold_doc ở top-1 | **8/10** |
| Theo chunk (thật, dùng trong bench) | chunk vừa đúng file *vừa chứa* `must_contain` ở top-1; hạng 2–3 = 1đ | **4/10** |

Chênh 4 điểm = những lần lấy **đúng file nhưng sai mẩu**. `docs/SCORING.md` yêu cầu *top-3 có chunk liên quan **và** agent trả lời đúng* → phải chấm ở mức nội dung.

**Ba con số cần nhớ:** `4 / 7 / 7` (Recursive / Fixed / Heading) · `0 → 2` (Q1 không / có filter) · `8 → 4` (chấm doc vs chấm chunk).
"""
        )
