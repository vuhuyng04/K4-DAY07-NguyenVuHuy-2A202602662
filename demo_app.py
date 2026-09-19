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

tab_demo, tab_query, tab_chunks = st.tabs(["🎬 Kịch bản demo", "🔍 Truy vấn tự do", "🧩 Xem chunk"])


# ============================================================================
# Tab 1 — Kịch bản demo chọn sẵn
# ============================================================================
SCENARIOS = {
    "1 · Metadata filter A/B — Q1 (0đ → 2đ chỉ bằng một dòng filter)": "filter",
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
        st.markdown(f"**Câu hỏi:** {q['q']}  \n**Gold:** `{q['gold_doc']}` · must_contain=`{q['must_contain']}` · filter=`{q['filter']}`")
        strat = st.selectbox("Chiến lược", MAIN3, key="s1")
        if run:
            s_store, _, _ = build_store(strat, fp)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("✅ Có filter `audience=student`")
                res_f, sc_f, why_f = run_query(s_store, q, top_k, True)
                st.markdown(f"### Điểm: {sc_f}/2 — {why_f}")
                render_results(res_f, q["must_contain"], q["gold_doc"])
            with c2:
                st.subheader("❌ Không filter")
                res_n, sc_n, why_n = run_query(s_store, q, top_k, False)
                st.markdown(f"### Điểm: {sc_n}/2 — {why_n}")
                render_results(res_n, q["must_contain"], q["gold_doc"])
            st.subheader("🤖 Agent answer (có filter)")
            st.success(agent_answer(s_store, llm, q["q"], res_f))
            st.info(
                "**Điểm nhấn:** Câu hỏi không nói người hỏi là ai. Corpus có trang undergraduate (3 items / 2 weeks) "
                "và trang graduate/faculty (5 items / 1 month) cùng câu chữ. Không filter, top-3 là chunk phạt tiền của FAQ "
                "và trang faculty — similarity đo *cùng chủ đề mượn sách*, không đo *đúng đối tượng*. "
                "Mặt trái: filter `student` loại luôn `borrowing-privilege` và `library-faq` (audience=all) — hai trang có bảng đầy đủ nhất."
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
            st.success(ans)
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
    col_q, col_f = st.columns([3, 1])
    question = col_q.text_input("Câu hỏi", value=preset["q"] if preset else "")
    default_aud = preset["filter"]["audience"] if (preset and preset["filter"]) else "(không lọc)"
    aud = col_f.selectbox("metadata_filter audience", list(AUDIENCES), index=list(AUDIENCES).index(default_aud))
    must = preset["must_contain"] if preset else ""
    gold = preset["gold_doc"] if preset else None
    if preset:
        st.caption(f"gold_doc = `{gold}` · must_contain = `{must}`")

    if st.button("Chạy truy vấn", type="primary", disabled=not question.strip()):
        flt = AUDIENCES[aud]
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
        st.success(ans)
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
