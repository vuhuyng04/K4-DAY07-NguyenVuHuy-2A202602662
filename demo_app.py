"""
demo_app.py — giao diện demo trực quan cho Lab 07 (Streamlit).

    pip install streamlit
    streamlit run demo_app.py

Tái dùng toàn bộ bench.py (corpus, embedder có cache, LLM, cách chấm) nên
demo không tốn thêm API call cho những chunk/query đã chạy.
"""

from __future__ import annotations

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

sys.argv = ["bench.py"]  # bench.main() đọc argv; không dùng ở đây

STRATEGIES = {
    "Recursive(500) — Huy": lambda: RecursiveChunker(chunk_size=500),
    "FixedSize(500, overlap=50) — Thiên": lambda: FixedSizeChunker(chunk_size=500, overlap=50),
    "Heading(800) — Phong": lambda: HeadingChunker(chunk_size=800),
    "Sentence(3 câu)": lambda: SentenceChunker(max_sentences_per_chunk=3),
}
AUDIENCES = {"(không lọc)": None, "student": {"audience": "student"}, "faculty": {"audience": "faculty"}, "all": {"audience": "all"}}

st.set_page_config(page_title="Lab 07 — Softmax RAG demo", page_icon="📚", layout="wide")


# ----------------------------------------------------------------------------
# Tài nguyên dùng chung (cache theo phiên Streamlit)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Khởi tạo embedder + LLM…")
def get_backends():
    embedder = bench.CachedEmbedder(bench.pick_embedder())
    llm = bench.make_llm_fn()
    return embedder, llm


@st.cache_resource(show_spinner="Chunk + embed corpus…")
def build_store(strategy_name: str) -> tuple[EmbeddingStore, int, dict[str, int]]:
    embedder, _ = get_backends()
    bench.CHUNKER = STRATEGIES[strategy_name]()
    docs = bench.load_corpus(bench.CORPUS_DIR)
    store = EmbeddingStore(collection_name=strategy_name, embedding_fn=embedder)
    store.add_documents(docs)
    embedder.save()
    per_doc: dict[str, int] = {}
    for d in docs:
        per_doc[d.metadata["doc_id"]] = per_doc.get(d.metadata["doc_id"], 0) + 1
    return store, len(docs), per_doc


def render_results(results: list[dict], must_contain: str = "") -> None:
    if not results:
        st.info("Không có kết quả (filter loại hết ứng viên?)")
        return
    top = max(r["score"] for r in results) or 1.0
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        hit = bool(must_contain) and must_contain.lower() in r["content"].lower()
        badge = " ✅ chứa đáp án" if hit else ""
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**#{i} · `{meta.get('doc_id')}`** · audience=`{meta.get('audience')}` · chunk {meta.get('chunk_index')}{badge}")
            c2.metric("score", f"{r['score']:.3f}")
            st.progress(min(1.0, max(0.0, r["score"] / top)))
            st.text(r["content"][:700] + ("…" if len(r["content"]) > 700 else ""))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("📚 Softmax — Lab 07")
strategy = st.sidebar.selectbox("Chiến lược chunking", list(STRATEGIES))
top_k = st.sidebar.slider("top_k", 1, 5, 3)
embedder, llm = get_backends()
st.sidebar.caption(f"Embedder: `{embedder.name}`  \nLLM: `{llm.name}`  \nCorpus: `{bench.CORPUS_DIR.name}`")

store, n_chunks, per_doc = build_store(strategy)
st.sidebar.metric("Số chunk trong store", n_chunks)
with st.sidebar.expander("Chunk theo tài liệu"):
    for k, v in sorted(per_doc.items()):
        st.write(f"`{k}`: {v}")

tab_query, tab_compare, tab_chunks = st.tabs(["🔍 Truy vấn", "📊 So sánh 3 chiến lược", "🧩 Xem chunk"])

# ----------------------------------------------------------------------------
# Tab 1 — truy vấn trực tiếp + A/B filter + agent
# ----------------------------------------------------------------------------
with tab_query:
    presets = {f"Q{i+1}: {q['q']}": q for i, q in enumerate(bench.QUERIES)}
    choice = st.selectbox("Chọn benchmark query hoặc tự gõ", ["(tự gõ)"] + list(presets))
    preset = presets.get(choice)

    col_q, col_f = st.columns([3, 1])
    question = col_q.text_input("Câu hỏi", value=preset["q"] if preset else "")
    default_aud = "(không lọc)"
    if preset and preset["filter"]:
        default_aud = preset["filter"]["audience"]
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
                render_results(filtered, must)
            with c_b:
                st.subheader("Không filter (A/B)")
                plain = store.search(question, top_k=top_k)
                if preset:
                    sc, why = bench.grade(plain, gold, must)
                    st.markdown(f"**Điểm: {sc}/2** — {why}")
                render_results(plain, must)
        else:
            if preset:
                sc, why = bench.grade(filtered, gold, must)
                st.markdown(f"**Điểm: {sc}/2** — {why}")
            render_results(filtered, must)

        st.subheader("🤖 Agent answer")
        agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
        if filtered:
            prompt = agent.build_prompt(question, filtered)
            with st.spinner("Gọi LLM…"):
                answer = llm(prompt)
            st.success(answer)
            with st.expander("Prompt đã gửi"):
                st.code(prompt)
        else:
            st.warning(agent.NO_CONTEXT_ANSWER)

# ----------------------------------------------------------------------------
# Tab 2 — bảng so sánh 3 chiến lược trên 5 query
# ----------------------------------------------------------------------------
with tab_compare:
    st.markdown("Cùng corpus, cùng 5 query, cùng embedder, cùng `top_k` — chỉ đổi chunker. Chấm theo chunk chứa đáp án (top-1 = 2đ, top-2/3 = 1đ).")
    chosen = st.multiselect("Chiến lược", list(STRATEGIES), default=list(STRATEGIES)[:3])
    if st.button("Chạy so sánh", type="primary", disabled=not chosen):
        rows = []
        for name in chosen:
            s_store, s_n, _ = build_store(name)
            row = {"Chiến lược": name, "Chunks": s_n}
            total = 0
            for i, q in enumerate(bench.QUERIES, 1):
                res = s_store.search_with_filter(q["q"], top_k=top_k, metadata_filter=q["filter"])
                sc, _ = bench.grade(res, q["gold_doc"], q["must_contain"])
                row[f"Q{i}"] = sc
                total += sc
            row["Tổng /10"] = total
            rows.append(row)
        st.dataframe(rows, width="stretch", hide_index=True)
        st.bar_chart(pd.DataFrame({"Tổng /10": [r["Tổng /10"] for r in rows]}, index=[r["Chiến lược"] for r in rows]))
        with st.expander("5 benchmark query"):
            for i, q in enumerate(bench.QUERIES, 1):
                st.markdown(f"**Q{i}.** {q['q']}  \n gold=`{q['gold_doc']}` · must_contain=`{q['must_contain']}` · filter=`{q['filter']}`")

# ----------------------------------------------------------------------------
# Tab 3 — xem chunk của một tài liệu với chiến lược đang chọn
# ----------------------------------------------------------------------------
with tab_chunks:
    doc_pick = st.selectbox("Tài liệu", sorted(per_doc))
    chunks = [r for r in store._store if r["metadata"]["doc_id"] == doc_pick]
    st.caption(f"{len(chunks)} chunk · độ dài TB {sum(len(c['content']) for c in chunks) / max(1, len(chunks)):.0f} ký tự")
    for r in chunks:
        with st.expander(f"chunk {r['metadata']['chunk_index']} · {len(r['content'])} ký tự · {r['content'][:60].replace(chr(10), ' ')}…"):
            st.text(r["content"])
