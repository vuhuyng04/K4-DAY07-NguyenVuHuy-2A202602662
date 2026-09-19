"""
demo_app.py — giao diện demo cho Lab 07 (Streamlit).

    pip install -r requirements-demo.txt
    streamlit run demo_app.py

- API key nhập trên sidebar (để trống → dùng .env; không có gì → mock embedder).
- Tab "Kịch bản": 7 tình huống chọn sẵn (metadata filter, chunking, nguồn mâu thuẫn,
  cross-lingual, chấm hai mức, tổng hợp, ngôn ngữ) — chọn rồi bấm Chạy.
- Tab "Chatbot": hỏi đáp tự do, mỗi lượt là một vòng RAG đầy đủ có trích dẫn.
- Tái dùng bench.py (corpus, cache embedding, cách chấm) nên số trên demo = số trong báo cáo.
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
Q = {f"Q{i}": q for i, q in enumerate(bench.QUERIES, 1)}

st.set_page_config(page_title="Softmax · Lab 07 — RAG demo", page_icon="📚", layout="wide")

# ============================================================================
# Kiểu trình bày dùng chung
# ============================================================================
st.markdown(
    """
<style>
.block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1320px}
.sx-title{font-size:1.5rem;font-weight:700;letter-spacing:-.01em;margin:0}
.sx-sub{color:#6b7280;margin:.15rem 0 .7rem 0;font-size:.95rem}
.sx-chips{display:flex;flex-wrap:wrap;gap:.4rem;margin:.1rem 0 .9rem 0}
.sx-chip{display:inline-flex;align-items:center;gap:.4rem;padding:.18rem .65rem;border:1px solid #e5e7eb;border-radius:999px;font-size:.78rem;color:#374151;background:#fff}
.sx-chip b{font-weight:600;color:#111827}
.sx-dot{width:.5rem;height:.5rem;border-radius:50%;display:inline-block}
.sx-card{border:1px solid #e5e7eb;border-radius:10px;padding:.6rem .85rem;margin:.45rem 0;background:#fff}
.sx-card.hit{border-left:3px solid #16a34a}
.sx-head{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;font-size:.8rem;color:#374151;margin-bottom:.3rem}
.sx-rank{font-weight:700;color:#111827}
.sx-doc{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.76rem;background:#f3f4f6;padding:.05rem .4rem;border-radius:4px;color:#111827}
.sx-meta{color:#6b7280}
.sx-pill{display:inline-block;padding:.05rem .55rem;border-radius:999px;font-size:.73rem;font-weight:600;line-height:1.5;white-space:nowrap}
.sx-pill.ok{background:#dcfce7;color:#166534}
.sx-pill.mid{background:#fef3c7;color:#92400e}
.sx-pill.bad{background:#fee2e2;color:#991b1b}
.sx-pill.neutral{background:#eef2ff;color:#3730a3}
.sx-pill.score{background:#f3f4f6;color:#111827;margin-left:auto;font-variant-numeric:tabular-nums}
.sx-bar{height:4px;background:#f3f4f6;border-radius:2px;margin:.3rem 0 .45rem 0;overflow:hidden}
.sx-bar>div{height:100%;background:#93c5fd}
.sx-body{font-size:.86rem;line-height:1.5;color:#1f2937;white-space:pre-wrap}
.sx-score{display:flex;align-items:center;gap:.6rem;margin:.25rem 0 .5rem 0;font-size:.86rem;color:#374151}
.sx-score .big{font-size:1rem;font-weight:700;padding:.12rem .65rem;border-radius:8px}
.sx-answer{border:1px solid #bbf7d0;background:#f0fdf4;border-radius:10px;padding:.7rem .95rem;margin:.4rem 0;font-size:.92rem;line-height:1.55;color:#14532d}
.sx-answer .lbl{font-size:.7rem;letter-spacing:.07em;text-transform:uppercase;color:#047857;font-weight:700;margin-bottom:.25rem}
.sx-q{border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0;padding:.55rem .85rem;margin:.2rem 0 .7rem 0;font-size:.9rem;line-height:1.5}
.sx-q .lbl{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.07em;font-weight:700;margin-right:.4rem}
.sx-q .row{margin:.1rem 0}
.sx-colh{font-weight:700;font-size:.95rem;margin:.15rem 0 .05rem 0;color:#111827}
.sx-note{font-size:.8rem;color:#6b7280;margin:.1rem 0 .4rem 0}
.sx-legend span{display:inline-block;padding:0 .45rem;border-radius:4px;font-size:.78rem;margin-right:.6rem}
.sx-cite{font-size:.83rem;color:#374151;margin:.25rem 0}
.sx-cite a{color:#1d4ed8;text-decoration:none;font-family:ui-monospace,Menlo,monospace;font-size:.78rem}
.sx-cite .txt{color:#6b7280;font-size:.8rem;margin:.15rem 0 .5rem 1.2rem}
.sx-docview{font-size:.86rem;line-height:1.75;color:#1f2937;border:1px solid #e5e7eb;border-radius:10px;padding:.8rem 1rem;background:#fff;max-height:600px;overflow:auto}
.sx-docview .c0{background:#eff6ff}.sx-docview .c1{background:#ecfdf5}.sx-docview .ov{background:#fde68a}.sx-docview .gap{color:#9ca3af}
.sx-docview mark{background:#fecaca;padding:0;border-radius:2px}
.sx-cmark{display:inline-block;font-size:.66rem;font-weight:700;color:#fff;background:#374151;border-radius:4px;padding:0 .38rem;margin:0 .3rem 0 .1rem;line-height:1.5;vertical-align:baseline;user-select:none}
.sx-cmark.hd{background:#2563eb}
.sx-map{position:relative;height:32px;background:#f3f4f6;border-radius:6px;overflow:hidden}
.sx-map .bar{position:absolute;height:12px;border-radius:2px;overflow:hidden;white-space:nowrap;font-size:9px;line-height:12px;color:#fff;padding-left:3px;box-sizing:border-box}
.sx-map .ovl{position:absolute;top:0;height:32px;background:#f59e0b;opacity:.45}
.sx-maprow{display:grid;grid-template-columns:230px 1fr;gap:.8rem;align-items:center;margin:.35rem 0}
.sx-maprow .lbl{font-size:.82rem;color:#111827;line-height:1.35}
.sx-maprow .lbl small{display:block;color:#6b7280;font-size:.75rem}
.sx-legend span{display:inline-block;padding:0 .45rem;border-radius:4px;font-size:.78rem;margin-right:.6rem}
div[data-testid="stMetricValue"]{font-size:1.35rem}
</style>
""",
    unsafe_allow_html=True,
)


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def esc(t) -> str:
    return _html.escape(str(t))


def pill(text, kind: str = "neutral") -> str:
    return f'<span class="sx-pill {kind}">{esc(text)}</span>'


def _kind(sc: int) -> str:
    return "ok" if sc >= 2 else ("mid" if sc == 1 else "bad")


def score_block(sc: int, why: str, label: str = "") -> None:
    lbl = f'<span class="sx-meta">{esc(label)}</span>' if label else ""
    html(f'<div class="sx-score"><span class="big sx-pill {_kind(sc)}">{sc}/2</span>{lbl}<span>{esc(why)}</span></div>')


def question_block(q: str, gold=None, must=None, flt=None, note: str | None = None) -> None:
    rows = [f'<div class="row"><span class="lbl">Câu hỏi</span>{esc(q)}</div>']
    if gold is not None:
        g = gold if isinstance(gold, str) else ", ".join(gold)
        rows.append(f'<div class="row"><span class="lbl">Gold</span><span class="sx-doc">{esc(g)}</span></div>')
    if must:
        m = must if isinstance(must, str) else " / ".join(must)
        rows.append(f'<div class="row"><span class="lbl">Đáp án phải chứa</span>{esc(m)}</div>')
    if flt is not None:
        rows.append(f'<div class="row"><span class="lbl">Filter</span><span class="sx-doc">{esc(flt)}</span></div>')
    if note:
        rows.append(f'<div class="row sx-meta">{esc(note)}</div>')
    html('<div class="sx-q">' + "".join(rows) + "</div>")


def col_title(text: str, note: str | None = None) -> None:
    html(f'<div class="sx-colh">{esc(text)}</div>' + (f'<div class="sx-note">{esc(note)}</div>' if note else ""))


def callout(text_md: str, label: str = "Điểm nhấn") -> None:
    with st.container(border=True):
        st.caption(label.upper())
        st.markdown(text_md)


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
# Helpers: chấm, hiển thị kết quả, filter
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


def render_results(results: list[dict], must_contain="", gold=None, chars: int = 420) -> None:
    if not results:
        st.info("Không có kết quả — filter đã loại hết ứng viên.")
        return
    golds = set() if gold is None else ({gold} if isinstance(gold, str) else set(gold))
    top = max(r["score"] for r in results) or 1.0
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        hit = _contains(r["content"], must_contain)
        is_gold = meta.get("doc_id") in golds
        tags = (pill("đúng file", "neutral") if is_gold else "") + (pill("chứa đáp án", "ok") if hit else "")
        body = r["content"][:chars] + ("…" if len(r["content"]) > chars else "")
        pct = max(0.0, min(1.0, r["score"] / top)) * 100
        html(
            f'<div class="sx-card{" hit" if hit else ""}">'
            f'<div class="sx-head"><span class="sx-rank">#{i}</span><span class="sx-doc">{esc(meta.get("doc_id"))}</span>'
            f'<span class="sx-meta">audience {esc(meta.get("audience"))} · lang {esc(meta.get("language", "–"))} · chunk {esc(meta.get("chunk_index"))}</span>'
            f'{tags}<span class="sx-pill score">{r["score"]:.3f}</span></div>'
            f'<div class="sx-bar"><div style="width:{pct:.0f}%"></div></div>'
            f'<div class="sx-body">{esc(body)}</div></div>'
        )


FILTER_SKIP = {"chunk_index", "source", "retrieved_at", "title", "source_url", "translated_from"}


def metadata_options(store: EmbeddingStore) -> dict[str, list[str]]:
    """Mọi trường metadata có trong store (trừ trường kỹ thuật) → danh sách giá trị."""
    opts: dict[str, set] = {}
    for r in store._store:
        for k, v in r["metadata"].items():
            if k in FILTER_SKIP or v is None:
                continue
            opts.setdefault(k, set()).add(str(v))
    order = ["audience", "language", "category", "department", "document_version", "doc_id"]
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
    html(f'<div class="sx-note">metadata_filter = <span class="sx-doc">{esc(flt or None)}</span> → <b>{n}/{len(store._store)}</b> chunk làm ứng viên</div>')
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
    with st.spinner("Đang hỏi LLM…"):
        return llm(agent.build_prompt(question, results))


_CITE = re.compile(r"\[(\d+)\]")


def show_answer(answer: str, results: list[dict], must_contain="", compact: bool = False) -> None:
    """Câu trả lời + panel 'Nguồn trích dẫn' nối [n] → chunk → doc_id → source_url."""
    html(f'<div class="sx-answer"><div class="lbl">Trả lời</div>{esc(answer).replace(chr(10), "<br>")}</div>')
    cited = sorted({int(n) for n in _CITE.findall(answer) if 1 <= int(n) <= len(results)})
    if not results:
        return
    if not cited:
        if "không tìm thấy" not in answer.lower():
            st.warning("Agent không trích dẫn [n] — không truy vết được câu trả lời lấy từ đâu.")
        return
    with st.expander("Nguồn trích dẫn " + ", ".join(f"[{n}]" for n in cited), expanded=not compact):
        for n in cited:
            r = results[n - 1]
            meta = r["metadata"]
            hit = _contains(r["content"], must_contain)
            url = meta.get("source_url")
            doc = esc(meta.get("doc_id"))
            link = f'<a href="{esc(url)}" target="_blank">{doc}</a>' if url else f'<span class="sx-doc">{doc}</span>'
            tag = pill("chứa gold answer", "ok") if hit else (pill("không chứa gold answer", "bad") if must_contain else "")
            snippet = r["content"][:300].replace("\n", " ") + ("…" if len(r["content"]) > 300 else "")
            html(
                f'<div class="sx-cite"><b>[{n}]</b> {link} <span class="sx-meta">· audience {esc(meta.get("audience"))} · '
                f'chunk {esc(meta.get("chunk_index"))} · {r["score"]:.3f}</span> {tag}<div class="txt">{esc(snippet)}</div></div>'
            )


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
# Sidebar — kết nối + cấu hình
# ============================================================================
st.sidebar.markdown("### Kết nối")
with st.sidebar.expander("OpenAI API key", expanded=not os.getenv("OPENAI_API_KEY")):
    key_in = st.text_input("API key", type="password", placeholder="sk-… (để trống = dùng .env)")
    chat_model = st.text_input("Chat model", value=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    if key_in.strip():
        os.environ["OPENAI_API_KEY"] = key_in.strip()
        os.environ["EMBEDDING_PROVIDER"] = "openai"
    if chat_model.strip():
        os.environ["OPENAI_CHAT_MODEL"] = chat_model.strip()
    st.caption("Key chỉ giữ trong phiên chạy, không ghi ra file.")

fp = _fingerprint()
embedder, llm = get_backends(fp)
ok_embed = "mock" not in embedder.name.lower()
ok_llm = llm.name != "demo"
if not ok_embed:
    st.sidebar.warning("Đang dùng mock embedder — kết quả không có ngữ nghĩa. Nhập API key ở trên.")

st.sidebar.markdown("### Cấu hình")
strategy = st.sidebar.selectbox("Chiến lược chunking", list(STRATEGIES), help="Áp dụng cho Chatbot, Truy vấn, Chunk. Các kịch bản có ô chọn riêng.")
top_k = st.sidebar.slider("Top-k", 1, 5, 3)
store, n_chunks, per_doc = build_store(strategy, fp)
with st.sidebar.expander(f"Chunk theo tài liệu ({n_chunks})"):
    st.dataframe(
        pd.DataFrame({"doc_id": list(sorted(per_doc)), "chunks": [per_doc[k] for k in sorted(per_doc)]}),
        hide_index=True, width="stretch", height=260,
    )

# ============================================================================
# Header
# ============================================================================
def _chip(label: str, value: str, ok: bool | None = None) -> str:
    dot = "" if ok is None else f'<span class="sx-dot" style="background:{"#16a34a" if ok else "#dc2626"}"></span>'
    return f'<span class="sx-chip">{dot}{esc(label)} <b>{esc(value)}</b></span>'


html('<div class="sx-title">Softmax · Lab 07 — Retrieval & RAG demo</div>'
     '<div class="sx-sub">Thư viện VinUni · 8 trang × 2 ngôn ngữ · so sánh chiến lược chunking, metadata filter và grounding của agent</div>')
html('<div class="sx-chips">'
     + _chip("Embedder", embedder.name, ok_embed)
     + _chip("LLM", llm.name, ok_llm)
     + _chip("Corpus", f"{bench.CORPUS_DIR.name} · {len(per_doc)} file")
     + _chip("Chunker", strategy.split(" — ")[0])
     + _chip("Chunks", str(n_chunks))
     + _chip("Top-k", str(top_k))
     + "</div>")

tab_demo, tab_chat, tab_query, tab_chunks, tab_notes = st.tabs(["Kịch bản", "Chatbot", "Truy vấn", "Chunk", "Ghi chú kỹ thuật"])


# ============================================================================
# Tab 1 — Kịch bản demo chọn sẵn
# ============================================================================
SCENARIOS = {
    "1 · Metadata filter": ("filter", "Cùng một câu hỏi, đổi filter là đổi câu trả lời — và khi nào filter làm hại."),
    "2 · So sánh chunking (Q4)": ("chunking", "Ba chiến lược trên cùng câu hỏi: chiến lược nào giữ được khối bullet chứa số liệu?"),
    "3 · Nguồn mâu thuẫn (Q2)": ("conflict", "Hai trang chính thức nói hai con số; retrieval đúng nhưng agent có thể trích sai nguồn."),
    "4 · Cross-lingual trước / sau (Q3)": ("failure", "Chỉ corpus tiếng Anh: 0/2 ở mọi chiến lược. Thêm bản tiếng Việt: 2/2."),
    "5 · Chấm hai mức": ("twolevel", "Chấm theo doc_id thổi phồng kết quả so với chấm theo chunk chứa đáp án."),
    "6 · Tổng hợp 3 chiến lược × 5 query": ("summary", "Bảng điểm và biểu đồ cho ba chiến lược trên năm benchmark query."),
    "7 · Ngôn ngữ": ("language", "Cùng câu hỏi bằng tiếng Việt / tiếng Anh, có hoặc không ép language."),
}

with tab_demo:
    c_sel, c_btn = st.columns([4, 1])
    pick = c_sel.selectbox("Kịch bản", list(SCENARIOS))
    kind, desc = SCENARIOS[pick]
    c_btn.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    run = c_btn.button("Chạy kịch bản", type="primary", width="stretch")
    st.caption(desc)

    # ---------------------------------------------------------------- 1
    if kind == "filter":
        q = Q["Q1"]
        cs, cp = st.columns([1, 2])
        strat = cs.selectbox("Chiến lược", MAIN3, key="s1")
        part = cp.radio(
            "Phần",
            ["A · Ba đối tượng", "B · Lọc theo trường bất kỳ", "C · Lọc trước vs lọc sau", "D · Filter làm mất recall", "E · Lọc sai trường"],
            horizontal=True, key="s1_part",
        )
        s_store, _, _ = build_store(strat, fp)

        if part.startswith("A"):
            question_block(q["q"], q["gold_doc"], q["must_contain"], note="Gold = trang undergraduate. Câu hỏi không nói người hỏi là ai.")
            if run:
                cols = st.columns(3)
                for col, (label, flt) in zip(cols, [("Không lọc", None), ("audience = student", {"audience": "student"}), ("audience = faculty", {"audience": "faculty"})]):
                    with col:
                        res = s_store.search_with_filter(q["q"], top_k=top_k, metadata_filter=flt)
                        sc, why = bench.grade(res, q["gold_doc"], q["must_contain"])
                        col_title(label, f"{count_candidates(s_store, flt)} ứng viên")
                        score_block(sc, why)
                        render_results(res, q["must_contain"], q["gold_doc"], chars=240)
                        show_answer(agent_answer(s_store, llm, q["q"], res), res, q["must_contain"], compact=True)
                callout(
                    "Cùng một câu hỏi, đổi `audience` là đổi câu trả lời — *3 cuốn / 2 tuần* (student) hay *5 cuốn / 1 tháng* (faculty). "
                    "Không lọc, top-3 là chunk của trang graduate/faculty: similarity đo *chủ đề mượn sách*, không đo *đúng đối tượng*. "
                    "Metadata là thứ duy nhất trả lời được câu hỏi \"ai đang hỏi\"."
                )

        elif part.startswith("B"):
            st.caption("Dựng filter từ bất kỳ trường metadata nào trong front matter; nhiều trường kết hợp bằng AND.")
            question = st.text_input("Câu hỏi", value=q["q"], key="s1b_q")
            flt = filter_builder(s_store, key="s1b")
            if run and question.strip():
                res = s_store.search_with_filter(question, top_k=top_k, metadata_filter=flt)
                render_results(res, chars=300)
                show_answer(agent_answer(s_store, llm, question, res), res, "", compact=True)
                callout(
                    "`search_with_filter` so khớp `==` trên mọi cặp key/value → lọc được theo `category` (fees / borrowing / access / spaces / faq), "
                    "`language`, `doc_id` (một file), `document_version`… Filter càng chặt, ứng viên càng ít — xem số chunk còn lại ở trên."
                )

        elif part.startswith("C"):
            question_block(q["q"], flt=q["filter"])
            if run:
                c1, c2 = st.columns(2)
                with c1:
                    res = s_store.search_with_filter(q["q"], top_k=top_k, metadata_filter=q["filter"])
                    col_title("Lọc trước rồi search — cách đúng", f"{count_candidates(s_store, q['filter'])} ứng viên → top-{top_k}")
                    render_results(res, q["must_contain"], q["gold_doc"], chars=240)
                with c2:
                    plain = s_store.search(q["q"], top_k=top_k)
                    post = [r for r in plain if r["metadata"].get("audience") == "student"]
                    col_title("Search top-k rồi mới lọc — lỗi hay gặp", f"{len(s_store._store)} ứng viên → top-{top_k} → lọc còn {len(post)}")
                    st.caption("Top-k trước khi lọc")
                    render_results(plain, q["must_contain"], q["gold_doc"], chars=110)
                    st.caption(f"Sau khi lọc: {len(post)} kết quả")
                    render_results(post, q["must_contain"], q["gold_doc"], chars=240)
                callout(
                    "Lọc sau top-k thì k slot đã bị chunk sai chiếm hết → **0 kết quả** dù store còn hàng chục chunk hợp lệ. "
                    "Đây là lỗi lab doc nhắc thẳng: *search_with_filter lọc SAU khi search thay vì trước*. "
                    "Code của nhóm lọc trước, rồi cho cả `search()` và `search_with_filter()` đi chung `_search_records()`."
                )

        elif part.startswith("D"):
            qh = "Giảng viên được mượn sách tối đa trong bao lâu?"
            question_block(qh, ["borrowing-privilege", "library-faq"], "6 months", note="Đáp án 'up to 6 months' chỉ nằm trong hai trang audience = all.")
            if run:
                c1, c2 = st.columns(2)
                for col, (label, flt) in zip((c1, c2), [("Không lọc", None), ("audience = faculty", {"audience": "faculty"})]):
                    with col:
                        res = s_store.search_with_filter(qh, top_k=top_k, metadata_filter=flt)
                        col_title(label, f"{count_candidates(s_store, flt)} ứng viên")
                        render_results(res, "6 months", ["borrowing-privilege", "library-faq"], chars=240)
                        show_answer(agent_answer(s_store, llm, qh, res), res, "6 months", compact=True)
                callout(
                    "Filter `audience=faculty` **loại luôn** hai trang `audience=all` — nơi duy nhất ghi *6 months* — nên agent chỉ còn *one month* của graduate. "
                    "Precision đổi bằng recall. Cách sửa dữ liệu: gán `audience` ở mức section (tách bảng hạn mức thành nhiều file), "
                    "hoặc cho phép filter `audience in {faculty, all}`."
                )

        elif part.startswith("E"):
            q2 = Q["Q2"]
            question_block(q2["q"], q2["gold_doc"], q2["must_contain"], note="20,000 VND/day nằm ở FAQ (category = faq), không ở trang Fines and other charges (category = fees).")
            if run:
                c1, c2 = st.columns(2)
                for col, (label, flt) in zip((c1, c2), [("Không lọc", None), ("category = fees — nghe hợp lý", {"category": "fees"})]):
                    with col:
                        res = s_store.search_with_filter(q2["q"], top_k=top_k, metadata_filter=flt)
                        sc, why = bench.grade(res, q2["gold_doc"], q2["must_contain"])
                        col_title(label, f"{count_candidates(s_store, flt)} ứng viên")
                        score_block(sc, why)
                        render_results(res, q2["must_contain"], q2["gold_doc"], chars=240)
                        show_answer(agent_answer(s_store, llm, q2["q"], res), res, q2["must_contain"], compact=True)
                callout(
                    "`category=fees` nghe rất đúng cho câu hỏi về tiền phạt, nhưng trang *Fines and other charges* chỉ nói về phí hư hỏng — "
                    "con số 20.000 VND/ngày lại ở FAQ. Metadata chỉ tốt khi **schema khớp với câu hỏi thật**; gán nhãn theo tiêu đề trang mà không đọc nội dung là bẫy."
                )

    # ---------------------------------------------------------------- 2
    elif kind == "chunking":
        q = Q["Q4"]
        question_block(q["q"], q["gold_doc"], q["must_contain"])
        if run:
            cols = st.columns(3)
            for col, name in zip(cols, MAIN3):
                with col:
                    s_store, s_n, _ = build_store(name, fp)
                    res, sc, why = run_query(s_store, q, top_k)
                    col_title(name.split(" — ")[0], f"{s_n} chunks · {name.split(' — ')[1]}")
                    score_block(sc, why)
                    render_results(res, q["must_contain"], q["gold_doc"], chars=320)
            callout(
                "Đáp án là **một bullet** trong danh sách quy định phòng học. Recursive cắt ở `\\n\\n` rồi `\\n` nên bullet bị tách từng dòng; "
                "FixedSize cắt mù 500 ký tự — cả hai đều để chunk *'Phải có ít nhất 2 người…'* (từ vựng gần câu hỏi: nhóm, buổi) lên top-1, "
                "còn bullet *'2 giờ mỗi buổi, 4 buổi mỗi tuần'* rớt xuống hạng 3. Heading giữ **cả khối bullet** dưới `## Phòng học nhóm` nên top-1 chứa đáp án. "
                "Thứ quyết định không phải chunker 'thông minh' mà là *khối thông tin có bị tách khỏi ngữ cảnh gần nó không*."
            )

    # ---------------------------------------------------------------- 3
    elif kind == "conflict":
        q = Q["Q2"]
        strat = st.selectbox("Chiến lược", MAIN3, key="s3")
        question_block(q["q"], q["gold_doc"], q["must_contain"], note="FAQ: 20.000 VND/ngày. Trang graduate/faculty: 10.000 VND/business day.")
        if run:
            s_store, _, _ = build_store(strat, fp)
            res, sc, why = run_query(s_store, q, top_k)
            score_block(sc, why, "Retrieval")
            render_results(res, q["must_contain"], q["gold_doc"])
            ans = agent_answer(s_store, llm, q["q"], res)
            show_answer(ans, res, q["must_contain"])
            has10 = "10,000" in ans or "10.000" in ans
            has20 = "20,000" in ans or "20.000" in ans
            verdict = ("agent lấy **10.000 VND** từ trang faculty (chunk 'per business day')" if has10 and not has20
                       else "agent lấy **20.000 VND** từ FAQ" if has20 and not has10
                       else "agent nêu cả hai con số / không rõ nguồn")
            st.markdown(f"**Nhận xét:** {verdict}.")
            callout(
                "Hai trang chính thức của cùng thư viện nói hai con số khác nhau. Retrieval vẫn được 2/2 vì top-1 chứa đáp án gold, "
                "nhưng agent có thể chọn chunk khác — đổi chiến lược ở trên để thấy ba chunker cho ba câu trả lời. "
                "Corpus không phân xử được vì `document_version` cả hai đều `not-stated`. Bài học: top-3 đúng chưa đủ, phải đọc agent answer."
            )

    # ---------------------------------------------------------------- 4
    elif kind == "failure":
        q = Q["Q3"]
        strat = st.selectbox("Chiến lược", MAIN3, key="s4")
        question_block(q["q"], q["gold_doc"], q["must_contain"], note="Trước khi dịch, corpus chỉ có tiếng Anh và Q3 0đ ở cả 3 chiến lược. Tái hiện bằng filter language = en.")
        if run:
            s_store, _, _ = build_store(strat, fp)
            c1, c2 = st.columns(2)
            with c1:
                res = s_store.search_with_filter(q["q"], top_k=top_k, metadata_filter={"language": "en"})
                sc, why = bench.grade(res, q["gold_doc"], q["must_contain"])
                col_title("Chỉ corpus EN (language = en)", "như trước khi dịch")
                score_block(sc, why)
                render_results(res, q["must_contain"], q["gold_doc"], chars=260)
                show_answer(agent_answer(s_store, llm, q["q"], res), res, q["must_contain"], compact=True)
            with c2:
                res2 = s_store.search_with_filter(q["q"], top_k=top_k, metadata_filter=None)
                sc2, why2 = bench.grade(res2, q["gold_doc"], q["must_contain"])
                col_title("Corpus song ngữ (không filter)", "sau khi dịch")
                score_block(sc2, why2)
                render_results(res2, q["must_contain"], q["gold_doc"], chars=260)
                show_answer(agent_answer(s_store, llm, q["q"], res2), res2, q["must_contain"], compact=True)
            callout(
                "Câu hỏi tiếng Việt *'quá hạn bao nhiêu ngày thì bị coi là mất'* trên corpus tiếng Anh: chunk *'fined for returning items late… lost'* "
                "của trang faculty gần nghĩa hơn chunk *'overdue for more than 05 days'* — score chỉ ~0.3, chunk có đáp án không lọt top-3. "
                "Thêm bản dịch: score ~0.7, top-1 chứa đáp án. Lỗi **không nằm ở chunker** (cả 3 cùng 0đ trước đó) mà ở khoảng cách ngôn ngữ giữa query và corpus."
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
            c1, c2, _ = st.columns([1, 1, 2])
            c1.metric("Chấm theo doc_id", f"{tot_doc}/10")
            c2.metric("Chấm theo chunk", f"{tot_chunk}/10", delta=tot_chunk - tot_doc)
            st.dataframe(df, width="stretch", hide_index=True)
            callout(
                "Chỉ kiểm `doc_id` gold có trong top-3 sẽ thổi phồng kết quả — một chiến lược có thể lấy trọn 3 slot từ đúng file mà không chunk nào chứa câu trả lời "
                "(Q1, Q4, Q5: top-3 toàn đúng file, chunk có số liệu ở hạng 2–3). `docs/SCORING.md` yêu cầu *top-3 có chunk liên quan **và** agent trả lời đúng*, "
                "nên phải chấm ở mức nội dung."
            )

    # ---------------------------------------------------------------- 6
    elif kind == "summary":
        chosen = st.multiselect("Chiến lược", list(STRATEGIES), default=MAIN3, key="s6")
        if run and chosen:
            df = score_table(chosen, top_k, fp)
            c1, c2 = st.columns([3, 2])
            c1.dataframe(df, width="stretch", hide_index=True)
            c2.bar_chart(df.set_index("Chiến lược")["Tổng /10"], height=220)
            with st.expander("5 benchmark query"):
                for key, q in Q.items():
                    question_block(f"{key}. {q['q']}", q["gold_doc"], q["must_contain"], q["filter"])
            callout(
                "Cùng 16 file (8 EN + 8 VI), cùng 5 câu, chỉ đổi một dòng chunker mà điểm dao động 7–9/10. Heading thắng vì giữ trọn khối bullet/mục quy định dưới tiêu đề. "
                "Trước khi có bản VI, điểm là 4/7/7 và Q3 0đ ở cả ba — thêm dữ liệu cùng ngôn ngữ với query nâng mọi chiến lược lên, nhiều hơn bất kỳ thay đổi chunker nào."
            )

    # ---------------------------------------------------------------- 7
    elif kind == "language":
        st.caption("Corpus có 8 trang × 2 ngôn ngữ (bản EN gốc + bản VI dịch, cùng source_url). Hỏi cùng một câu bằng hai thứ tiếng, có/không ép language.")
        strat = st.selectbox("Chiến lược", MAIN3, key="s7")
        c1, c2 = st.columns(2)
        q_vi = c1.text_input("Câu hỏi tiếng Việt", value=Q["Q1"]["q"], key="s7_vi")
        q_en = c2.text_input("Câu hỏi tiếng Anh", value="How many books can I borrow and for how long?", key="s7_en")
        base = {"audience": "student"}
        if run:
            s_store, _, _ = build_store(strat, fp)
            cells = [
                ("Hỏi VI · không ép ngôn ngữ", q_vi, dict(base)),
                ("Hỏi VI · language = en", q_vi, {**base, "language": "en"}),
                ("Hỏi EN · không ép ngôn ngữ", q_en, dict(base)),
                ("Hỏi EN · language = vi", q_en, {**base, "language": "vi"}),
            ]
            rows = []
            r1, r2 = st.columns(2), st.columns(2)
            for col, (label, qq, flt) in zip(list(r1) + list(r2), cells):
                with col:
                    res = s_store.search_with_filter(qq, top_k=top_k, metadata_filter=flt)
                    langs = [r["metadata"].get("language") for r in res]
                    top = res[0]["score"] if res else 0.0
                    col_title(label, f"{count_candidates(s_store, flt)} ứng viên · top-1 {top:.3f} · ngôn ngữ top-{top_k}: {', '.join(map(str, langs))}")
                    render_results(res, "", None, chars=180)
                    rows.append({"Trường hợp": label, "Ứng viên": count_candidates(s_store, flt), "Top-1 score": round(top, 3), "Ngôn ngữ top-k": ", ".join(map(str, langs))})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            callout(
                "Không ép ngôn ngữ, top-3 **luôn cùng ngôn ngữ với câu hỏi** với score ~0.6–0.7; ép sang ngôn ngữ kia score rớt còn ~0.25–0.4 nhưng vẫn tìm đúng trang "
                "→ cross-lingual *có* hoạt động, chỉ yếu hơn nhiều. Hệ quả: trong corpus song ngữ, bản dịch quyết định chất lượng retrieval cho người dùng tiếng Việt; "
                "`language` là trường lọc thật khi muốn agent trích dẫn đúng bản gốc."
            )


# ============================================================================
# Tab 2 — Chatbot: hỏi đáp tự do, mỗi lượt = 1 vòng RAG đầy đủ
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
    """Gợi ý filter đơn giản từ từ khoá trong câu hỏi (chỉ để demo)."""
    q = question.lower()
    if any(w in q for w in ["giảng viên", "faculty", "cao học", "sau đại học", "graduate"]):
        return {"audience": "faculty"}
    if any(w in q for w in ["sinh viên", "student", "undergraduate"]):
        return {"audience": "student"}
    return None


def _turn_meta(strategy_name: str, flt, n_cand: int, k: int) -> None:
    html('<div class="sx-chips" style="margin:.1rem 0 .4rem 0">'
         + _chip("Chunker", strategy_name.split(" — ")[0]) + _chip("Filter", str(flt)) + _chip("Ứng viên", str(n_cand)) + _chip("Top-k", str(k))
         + "</div>")


with tab_chat:
    st.caption("Mỗi lượt là một vòng RAG đầy đủ: embed câu hỏi → lọc metadata (nếu có) → top-k chunk → prompt có trích dẫn → LLM. "
               "Mở Nguồn trích dẫn dưới mỗi câu trả lời để truy vết.")
    cc1, cc2, cc3 = st.columns([2, 2, 1])
    auto_filter = cc1.toggle("Tự gợi ý filter audience từ câu hỏi", value=False, key="chat_auto",
                             help="Bật để thấy filter làm mất recall với câu hỏi về giảng viên.")
    manual = cc2.toggle("Đặt filter thủ công", value=False, key="chat_manual")
    if cc3.button("Xoá hội thoại", key="chat_clear", width="stretch"):
        st.session_state["chat_history"] = []
        st.rerun()
    chat_filter = filter_builder(store, key="chat") if manual else None

    with st.expander("Câu hỏi gợi ý"):
        cols = st.columns(2)
        for i, sq in enumerate(SUGGESTED):
            if cols[i % 2].button(sq, key=f"sug_{i}", width="stretch"):
                st.session_state["chat_pending"] = sq

    history: list[dict] = st.session_state.setdefault("chat_history", [])
    for turn in history:
        with st.chat_message("user"):
            st.write(turn["q"])
        with st.chat_message("assistant"):
            _turn_meta(turn["strategy"], turn["filter"], turn["n_cand"], turn["top_k"])
            show_answer(turn["answer"], turn["results"], "", compact=True)

    pending = st.session_state.pop("chat_pending", None)
    typed = st.chat_input("Nhập câu hỏi về thư viện VinUni…", key="chat_input")
    question = typed or pending
    if question:
        flt = chat_filter if manual else (_pick_filter_from_question(question) if auto_filter else None)
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Đang truy xuất và hỏi LLM…"):
                results = store.search_with_filter(question, top_k=top_k, metadata_filter=flt)
                answer = agent_answer(store, llm, question, results)
            _turn_meta(strategy, flt, count_candidates(store, flt), top_k)
            show_answer(answer, results, "", compact=True)
            with st.expander(f"Top-{top_k} chunk đã dùng"):
                render_results(results, "", None, chars=300)
        history.append({
            "q": question, "answer": answer, "results": results, "filter": flt,
            "strategy": strategy, "top_k": top_k, "n_cand": count_candidates(store, flt),
        })


# ============================================================================
# Tab 3 — Truy vấn tự do
# ============================================================================
with tab_query:
    presets = {f"{k}: {q['q']}": q for k, q in Q.items()}
    choice = st.selectbox("Benchmark query hoặc tự gõ", ["(tự gõ)"] + list(presets))
    preset = presets.get(choice)
    question = st.text_input("Câu hỏi", value=preset["q"] if preset else "")
    flt_free = filter_builder(store, key=f"free_{choice}", default=preset["filter"] if preset else None)
    must = preset["must_contain"] if preset else ""
    gold = preset["gold_doc"] if preset else None
    if preset:
        question_block(preset["q"], gold, must, preset["filter"])

    if st.button("Chạy truy vấn", type="primary", disabled=not question.strip()):
        flt = flt_free
        filtered = store.search_with_filter(question, top_k=top_k, metadata_filter=flt)
        if flt:
            c_a, c_b = st.columns(2)
            with c_a:
                col_title(f"Có filter {flt}", f"{count_candidates(store, flt)} ứng viên")
                if preset:
                    score_block(*bench.grade(filtered, gold, must))
                render_results(filtered, must, gold)
            with c_b:
                plain = store.search(question, top_k=top_k)
                col_title("Không filter (A/B)", f"{len(store._store)} ứng viên")
                if preset:
                    score_block(*bench.grade(plain, gold, must))
                render_results(plain, must, gold)
        else:
            if preset:
                score_block(*bench.grade(filtered, gold, must))
            render_results(filtered, must, gold)

        ans = agent_answer(store, llm, question, filtered)
        show_answer(ans, filtered, must)
        if filtered:
            with st.expander("Prompt đã gửi"):
                st.code(KnowledgeBaseAgent(store=store, llm_fn=llm).build_prompt(question, filtered))


# ============================================================================
# Tab 4 — Chunk: văn bản gốc tô màu theo chunk, bản đồ 3 chiến lược, overlap, heading gắn lại
# ============================================================================
_WS = re.compile(r"\s+")


def _norm_index(text: str) -> tuple[str, list[int]]:
    """Gộp mọi khoảng trắng thành 1 dấu cách + bảng ánh xạ vị trí chuẩn hoá → vị trí gốc."""
    out: list[str] = []
    idx: list[int] = []
    prev_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_ws:
                continue
            out.append(" ")
            idx.append(i)
            prev_ws = True
        else:
            out.append(ch)
            idx.append(i)
            prev_ws = False
    return "".join(out), idx


def _locate_all(body: str, chunks: list[str]) -> list[tuple[int, int]]:
    """(start, end) của từng chunk trong văn bản gốc, (-1, -1) nếu không định vị được.

    So khớp trên bản gộp khoảng trắng để chịu được chunker strip/join (Sentence, Heading);
    tìm tuần tự từ vị trí chunk trước nên overlap (chunk sau bắt đầu trước khi chunk trước kết thúc) vẫn đúng.
    """
    nb, idx = _norm_index(body)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for c in chunks:
        cands = [c]
        if "\n" in c:
            cands.append(c.split("\n", 1)[1])  # Heading gắn lại tiêu đề → định vị phần thân
        found = (-1, -1)
        for cc in cands:
            nc = _WS.sub(" ", cc).strip()
            if not nc:
                continue
            j = nb.find(nc, cursor)
            if j < 0:
                j = nb.find(nc)
            if j >= 0:
                found = (idx[j], idx[j + len(nc) - 1] + 1)
                cursor = j
                break
        spans.append(found)
    return spans


def _overlap_len(prev: str, cur: str) -> int:
    """Số ký tự đầu của `cur` trùng với đuôi của `prev` (đo bằng so khớp chuỗi)."""
    m = min(len(prev), len(cur))
    for k in range(m, 9, -1):
        if prev.endswith(cur[:k]):
            return k
    return 0


def _chunk_stats(chunks: list[str], body: str) -> dict:
    spans = _locate_all(body, chunks) if body else [(-1, -1)] * len(chunks)
    overlaps = [0]
    head_rep: list[str | None] = [None]
    for i in range(1, len(chunks)):
        prev, cur = chunks[i - 1], chunks[i]
        overlaps.append(_overlap_len(prev, cur))
        h_prev = prev.split("\n", 1)[0]
        head_rep.append(h_prev if (h_prev.startswith("#") and cur.startswith(h_prev)) else None)
    return {
        "spans": spans, "overlaps": overlaps, "head_rep": head_rep,
        "n": len(chunks), "avg": sum(map(len, chunks)) / max(1, len(chunks)),
        "total_ov": sum(overlaps), "n_head": sum(1 for h in head_rep if h),
        "located": sum(1 for a, _ in spans if a >= 0),
    }


_BAR_COLORS = ("#3b82f6", "#10b981")


def _map_html(spans: list[tuple[int, int]], L: int) -> str:
    L = max(1, L)
    parts = []
    for i, (a, b) in enumerate(spans):
        if a < 0:
            continue
        left, width = 100 * a / L, max(0.3, 100 * (b - a) / L)
        top = 2 if i % 2 == 0 else 18
        label = str(i) if width > 2.2 else ""
        parts.append(
            f'<div class="bar" title="chunk {i}: {a}–{b} ({b - a} ký tự)" '
            f'style="left:{left:.2f}%;width:{width:.2f}%;top:{top}px;background:{_BAR_COLORS[i % 2]}">{label}</div>'
        )
    for i in range(1, len(spans)):
        (a0, b0), (a1, b1) = spans[i - 1], spans[i]
        if a0 >= 0 and a1 >= 0 and a1 < b0:
            parts.append(f'<div class="ovl" title="overlap chunk {i-1}–{i}: {b0 - a1} ký tự" style="left:{100 * a1 / L:.2f}%;width:{max(0.3, 100 * (b0 - a1) / L):.2f}%"></div>')
    return f'<div class="sx-map">{"".join(parts)}</div>'


def _document_html(body: str, spans: list[tuple[int, int]], matches: list[tuple[int, int]], head_rep: list[str | None]) -> str:
    """Văn bản gốc, mỗi ký tự tô theo chunk phủ nó: tint xen kẽ theo chunk, vàng nếu ≥2 chunk phủ, đỏ = chuỗi tìm."""
    pts = {0, len(body)}
    for a, b in spans:
        if a >= 0:
            pts.update((a, b))
    for a, b in matches:
        pts.update((a, b))
    starts: dict[int, int] = {}
    for i, (a, _) in enumerate(spans):
        if a >= 0:
            starts.setdefault(a, i)
    out = []
    pts_sorted = sorted(pts)
    for a, b in zip(pts_sorted, pts_sorted[1:]):
        if a in starts:
            i = starts[a]
            out.append(f'<span class="sx-cmark" title="chunk {i} bắt đầu">{i}</span>')
            if head_rep[i]:
                out.append('<span class="sx-cmark hd" title="chunk này được gắn lại tiêu đề của chunk trước">↻ heading</span>')
        cov = [i for i, (s0, e0) in enumerate(spans) if s0 >= 0 and s0 <= a and b <= e0]
        cls = "gap" if not cov else ("ov" if len(cov) >= 2 else ("c0" if cov[0] % 2 == 0 else "c1"))
        txt = _html.escape(body[a:b]).replace("\n", "<br>")
        if any(s0 <= a and b <= e0 for s0, e0 in matches):
            txt = f"<mark>{txt}</mark>"
        out.append(f'<span class="{cls}">{txt}</span>')
    return f'<div class="sx-docview">{"".join(out)}</div>'


def _chunk_html(text: str, overlap_n: int, heading_repeat: str | None, hl: str) -> str:
    def e(t: str) -> str:
        return _html.escape(t).replace("\n", "<br>")

    parts, pos = [], 0
    if heading_repeat and text.startswith(heading_repeat):
        parts.append(f'<span style="background:#dbeafe;border-radius:3px">{e(heading_repeat)}</span>')
        pos = len(heading_repeat)
    if overlap_n > pos:
        parts.append(f'<span style="background:#fde68a;border-radius:3px">{e(text[pos:overlap_n])}</span>')
        pos = overlap_n
    parts.append(e(text[pos:]))
    out = "".join(parts)
    if hl:
        out = re.sub(re.escape(_html.escape(hl)), lambda m: f'<mark style="background:#fecaca">{m.group(0)}</mark>', out, flags=re.IGNORECASE)
    return f'<div class="sx-body">{out}</div>'


with tab_chunks:
    c1, c2, c3 = st.columns([2, 2, 3])
    doc_pick = c1.selectbox("Tài liệu", sorted(per_doc))
    view_strat = c2.selectbox("Chiến lược", list(STRATEGIES), index=list(STRATEGIES).index(strategy), key="chunk_strat")
    hl = c3.text_input("Tô đỏ chuỗi", value="", placeholder="ví dụ: 2 giờ mỗi buổi / 2 hours per session")

    body_path = bench.CORPUS_DIR / f"{doc_pick}.md"
    _, body = bench.parse_front_matter(body_path.read_text(encoding="utf-8")) if body_path.exists() else ({}, "")
    v_store, _, _ = build_store(view_strat, fp)
    chunks = [r["content"] for r in v_store._store if r["metadata"]["doc_id"] == doc_pick]
    stt = _chunk_stats(chunks, body)
    matches = [(m.start(), m.end()) for m in re.finditer(re.escape(hl), body, re.IGNORECASE)] if (hl and body) else []
    n_hit = sum(1 for c in chunks if hl and hl.lower() in c.lower())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số chunk", stt["n"])
    m2.metric("Độ dài trung bình", f"{stt['avg']:.0f} ký tự")
    m3.metric("Ký tự overlap", stt["total_ov"], help="Phần đầu chunk i trùng với phần đuôi chunk i−1, đo bằng so khớp chuỗi")
    m4.metric("Heading gắn lại", stt["n_head"], help="Chunk bắt đầu bằng đúng tiêu đề của chunk trước (HeadingChunker)")
    note = f"{view_strat} · văn bản gốc {len(body)} ký tự · tổng ký tự trong chunk {sum(map(len, chunks))} (= gốc + overlap + heading lặp)"
    if hl:
        note += f" · chuỗi tìm xuất hiện {len(matches)} lần trong văn bản, nằm trong {n_hit}/{stt['n']} chunk"
    if body and stt["located"] < stt["n"]:
        note += f" · định vị được {stt['located']}/{stt['n']} chunk trên văn bản gốc"
    st.caption(note)

    if body:
        st.markdown("**Bản đồ vị trí — ba chiến lược trên cùng tài liệu**")
        st.caption("Mỗi thanh là một chunk (xếp so le, số là chỉ số chunk); dải vàng phủ hai hàng là vùng hai chunk chồng nhau.")
        rows = []
        for name in MAIN3:
            s_store, _, _ = build_store(name, fp)
            cs = [r["content"] for r in s_store._store if r["metadata"]["doc_id"] == doc_pick]
            st_ = _chunk_stats(cs, body)
            rows.append(
                f'<div class="sx-maprow"><div class="lbl">{esc(name)}<small>{st_["n"]} chunk · TB {st_["avg"]:.0f} ký tự · overlap {st_["total_ov"]} · heading lặp {st_["n_head"]}</small></div>'
                f'{_map_html(st_["spans"], len(body))}</div>'
            )
        html("".join(rows))

        st.markdown(f"**Văn bản gốc tô màu theo chunk — {esc(view_strat)}**")
        html('<div class="sx-legend"><span style="background:#eff6ff">chunk chẵn</span><span style="background:#ecfdf5">chunk lẻ</span>'
             '<span style="background:#fde68a">hai chunk chồng nhau (overlap)</span><span style="background:#fecaca">chuỗi tìm</span>'
             '<span class="sx-cmark" style="margin:0 .6rem 0 0">n</span>bắt đầu chunk n'
             '<span class="sx-cmark hd" style="margin:0 .6rem 0 .6rem">↻ heading</span>chunk được gắn lại tiêu đề</div>')
        html(_document_html(body, stt["spans"], matches, stt["head_rep"]))
    else:
        st.info("Không tìm thấy file nguồn của tài liệu này — chỉ hiển thị danh sách chunk.")

    st.markdown("**Danh sách chunk**")
    for i, c in enumerate(chunks):
        found = bool(hl) and hl.lower() in c.lower()
        bits = [f"Chunk {i}", f"{len(c)} ký tự"]
        if stt["overlaps"][i]:
            bits.append(f"overlap {stt['overlaps'][i]}")
        if stt["head_rep"][i]:
            bits.append("heading lặp")
        if found:
            bits.append("chứa chuỗi tìm")
        a, b = stt["spans"][i]
        bits.append(f"vị trí {a}–{b}" if a >= 0 else "không định vị được")
        with st.expander(" · ".join(bits), expanded=found):
            html(_chunk_html(c, stt["overlaps"][i], stt["head_rep"][i], hl))


# ============================================================================
# Tab 5 — Ghi chú kỹ thuật: data, pipeline, chunking, chấm điểm
# ============================================================================
with tab_notes:
    st.markdown("#### Đường đi của một câu hỏi")
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

    n1, n2, n3 = st.tabs(["Dữ liệu", "Chunking", "Embedding · Search · Chấm điểm"])

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
