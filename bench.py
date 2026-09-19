"""
bench.py — công cụ chạy benchmark retrieval cho Lab 07 (K4-L3A).

Cách dùng:
    python bench.py                       # dùng CORPUS_DIR + QUERIES bên dưới
    python bench.py data/thu-vien         # chỉ định thư mục corpus khác

Mỗi thành viên chỉ đổi ĐÚNG MỘT DÒNG `CHUNKER = ...` sang chiến lược của mình.
Mọi thứ khác (embedder, top_k, query, cách chấm) giữ nguyên để so sánh công bằng.

Output in ra màn hình và ghi vào ket_qua_benchmark.txt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from src import (
    EMBEDDING_PROVIDER_ENV,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    GeminiEmbedder,
    HeadingChunker,
    KnowledgeBaseAgent,
    LocalEmbedder,
    OpenAIEmbedder,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env", override=False)

# ============================================================================
# CẤU HÌNH — mỗi thành viên chỉ đổi dòng CHUNKER
# ============================================================================

CORPUS_DIR = ROOT / "data" / "thu-vien-vinuni"
TOP_K = 3
OUTPUT_FILE = ROOT / "ket_qua_benchmark.txt"

# --- Chiến lược của TÔI (chỉ đổi dòng này) ---------------------------------
CHUNKER = RecursiveChunker(chunk_size=500)
# CHUNKER = FixedSizeChunker(chunk_size=500, overlap=50)
# CHUNKER = HeadingChunker(chunk_size=800)
# CHUNKER = SentenceChunker(max_sentences_per_chunk=3)
# ----------------------------------------------------------------------------

# 5 benchmark query của NHÓM (phải trùng với REPORT_NHOM.md).
#   q            : câu hỏi
#   gold_doc     : doc_id (tên file không .md) chứa gold answer — hoặc list nếu nhiều file cùng chứa
#   must_contain : chuỗi (hoặc list chuỗi, vd EN + VI) đặc trưng phải xuất hiện trong ngữ cảnh truy xuất được
#                  (dùng để chấm mức 2 — top-3 đúng file chưa đủ, phải có đáp án)
#   filter       : metadata_filter hoặc None; ít nhất 1 câu cần {"audience": "student"}
QUERIES: list[dict] = [
    # Corpus song ngữ: mỗi trang có bản EN (doc_id) và bản VI (doc_id-vi), cùng nội dung.
    # gold_doc / must_contain nhận cả hai bản để chấm công bằng bất kể ngôn ngữ chunk được lấy.
    # Q1 — CẦN filter audience: không nêu người hỏi là ai; trang undergraduate (3 items/2 weeks)
    #      và trang graduate/faculty (5 items/1 month) cùng từ vựng, khác đáp án.
    {
        "q": "Tôi được mượn tối đa bao nhiêu cuốn sách và trong bao lâu?",
        "gold_doc": ["borrowing-undergraduate-staff", "borrowing-undergraduate-staff-vi"],
        "must_contain": ["3 items", "3 tài liệu"],
        "filter": {"audience": "student"},
    },
    # Q2 — tra số liệu
    {
        "q": "Mức phạt trả sách muộn là bao nhiêu tiền một ngày?",
        "gold_doc": ["library-faq", "library-faq-vi"],
        "must_contain": ["20,000 VND", "20.000 VND"],
        "filter": None,
    },
    # Q3 — hỏi điều kiện (trước đây cross-lingual fail ở cả 3 chiến lược)
    {
        "q": "Thiết bị mượn quá hạn bao nhiêu ngày thì bị coi là mất?",
        "gold_doc": ["equipment-loans", "equipment-loans-vi", "borrowing-undergraduate-staff", "borrowing-undergraduate-staff-vi"],
        "must_contain": ["05 days", "05 ngày"],
        "filter": None,
    },
    # Q4 — hỏi quy trình / giới hạn
    {
        "q": "Một nhóm được đặt phòng học nhóm tối đa bao nhiêu giờ mỗi buổi và bao nhiêu buổi mỗi tuần?",
        "gold_doc": ["room-booking", "room-booking-vi", "borrowing-undergraduate-staff", "borrowing-undergraduate-staff-vi",
                     "borrowing-graduate-faculty", "borrowing-graduate-faculty-vi"],
        "must_contain": ["2 hours per session", "2 giờ mỗi buổi"],
        "filter": None,
    },
    # Q5 — liệt kê / thời gian
    {
        "q": "Giờ mở cửa thư viện từ tháng 9 là khi nào?",
        "gold_doc": ["hours-and-access", "hours-and-access-vi", "library-faq", "library-faq-vi"],
        "must_contain": ["8:45 am – 9:00 pm", "8h45 – 21h00"],
        "filter": None,
    },
]


# ============================================================================
# Nạp corpus
# ============================================================================

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Tách YAML front matter đơn giản (key: value, không lồng) khỏi phần thân."""
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0].rstrip()  # bỏ comment cuối dòng
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, text[m.end() :]


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Đọc mọi .md/.txt, chunk phần thân, mỗi chunk → một Document."""
    docs: list[Document] = []
    files = sorted(p for p in corpus_dir.iterdir() if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise SystemExit(f"Không tìm thấy .md/.txt trong {corpus_dir}")

    for path in files:
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        chunks = CHUNKER.chunk(body)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    id=f"{path.stem}#{i}",
                    content=chunk,
                    metadata={**meta, "doc_id": path.stem, "chunk_index": i, "source": str(path)},
                )
            )
        print(f"  {path.name:45} {len(chunks):3d} chunks  ({len(body):5d} ký tự)")
    return docs


# ============================================================================
# Embedder + cache (chạy lại không tốn tiền API)
# ============================================================================

CACHE_FILE = ROOT / ".embed_cache.json"


def pick_embedder():
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    try:
        if provider == "openai":
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        if provider == "gemini":
            return GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))
        if provider == "local":
            return LocalEmbedder()
    except Exception as exc:  # thiếu key / thiếu thư viện → mock
        print(f"  [warn] {provider} embedder không khả dụng ({exc}); dùng mock")
    return _mock_embed


class CachedEmbedder:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = getattr(inner, "_backend_name", inner.__class__.__name__)
        self._cache: dict[str, list[float]] = {}
        self.hits = self.misses = 0
        if CACHE_FILE.exists():
            try:
                self._cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def __call__(self, text: str) -> list[float]:
        key = hashlib.sha256(f"{self.name}\n{text}".encode("utf-8")).hexdigest()
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        vec = self.inner(text)
        self._cache[key] = vec
        return vec

    def save(self) -> None:
        CACHE_FILE.write_text(json.dumps(self._cache), encoding="utf-8")


# ============================================================================
# LLM cho agent answer
# ============================================================================


def make_llm_fn():
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI()
            model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

            def llm(prompt: str) -> str:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return resp.choices[0].message.content.strip()

            llm.name = model  # type: ignore[attr-defined]
            return llm
        except Exception as exc:
            print(f"  [warn] OpenAI chat không khả dụng ({exc}); dùng demo LLM")

    def demo(prompt: str) -> str:
        return "[DEMO LLM] " + prompt[:200].replace("\n", " ") + "..."

    demo.name = "demo"  # type: ignore[attr-defined]
    return demo


# ============================================================================
# Chấm
# ============================================================================


def grade(results: list[dict], gold_doc, must_contain) -> tuple[int, str]:
    """Chấm 2 mức theo hạng của CHUNK vừa đúng doc vừa chứa đáp án.

    2đ: chunk đó ở top-1; 1đ: ở top-2/3; 0đ: gold_doc vắng hoặc không chunk nào chứa đáp án.
    gold_doc và must_contain có thể là str hoặc list[str] (nhiều file / nhiều ngôn ngữ cùng chứa đáp án).
    """
    golds = {gold_doc} if isinstance(gold_doc, str) else set(gold_doc)
    doc_ranks = [i for i, r in enumerate(results, 1) if r["metadata"].get("doc_id") in golds]
    if not doc_ranks:
        return 0, "gold_doc KHÔNG có trong top-k"
    needles = [n.lower() for n in ([must_contain] if isinstance(must_contain, str) else must_contain) if n]
    hit_ranks = [
        i for i, r in enumerate(results, 1)
        if r["metadata"].get("doc_id") in golds and (not needles or any(n in r["content"].lower() for n in needles))
    ]
    if not hit_ranks:
        return 0, f"gold_doc ở hạng {doc_ranks[0]} nhưng KHÔNG chunk nào chứa '{must_contain}'"
    if hit_ranks[0] == 1:
        return 2, "chunk chứa đáp án ở top-1"
    return 1, f"chunk chứa đáp án ở hạng {hit_ranks[0]}"


def fmt_results(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        preview = r["content"].replace("\n", " ")[:110]
        lines.append(f"    {i}. score={r['score']:.3f}  doc_id={r['metadata'].get('doc_id')!s:30} aud={r['metadata'].get('audience')!s:8} | {preview}...")
    return "\n".join(lines) if lines else "    (không có kết quả)"


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    corpus_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else CORPUS_DIR
    out: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        out.append(s)

    log("=" * 78)
    params = {k: v for k, v in vars(CHUNKER).items() if not k.startswith("_")}
    log(f"BENCHMARK  | chunker={CHUNKER.__class__.__name__} {params}")
    log(f"           | corpus={corpus_dir}  top_k={TOP_K}")
    log("=" * 78)

    log("\n[1] Nạp corpus")
    docs = load_corpus(corpus_dir)
    log(f"  → {len(docs)} chunks từ {len({d.metadata['doc_id'] for d in docs})} tài liệu")

    embedder = CachedEmbedder(pick_embedder())
    llm_fn = make_llm_fn()
    log(f"\n[2] Embedder: {embedder.name}   |   LLM: {llm_fn.name}")

    store = EmbeddingStore(collection_name="bench", embedding_fn=embedder)
    store.add_documents(docs)
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    log(f"  → store size = {store.get_collection_size()}  (cache hit={embedder.hits} miss={embedder.misses})")

    log("\n[3] Chạy benchmark")
    total = 0
    relevant_in_top3 = 0
    for n, q in enumerate(QUERIES, 1):
        log("\n" + "-" * 78)
        log(f"Q{n}: {q['q']}")
        log(f"    gold_doc={q['gold_doc']}  must_contain='{q['must_contain']}'  filter={q['filter']}")

        results = store.search_with_filter(q["q"], top_k=TOP_K, metadata_filter=q["filter"])
        score, why = grade(results, q["gold_doc"], q["must_contain"])
        total += score
        if score > 0:
            relevant_in_top3 += 1
        log(f"  Top-{TOP_K} (filter={'có' if q['filter'] else 'không'}):")
        log(fmt_results(results))
        log(f"  → Điểm: {score}/2  ({why})")

        # A/B: nếu câu này có filter, chạy thêm bản KHÔNG filter để so sánh
        if q["filter"]:
            plain = store.search(q["q"], top_k=TOP_K)
            p_score, p_why = grade(plain, q["gold_doc"], q["must_contain"])
            log(f"  [A/B] Top-{TOP_K} KHÔNG filter:")
            log(fmt_results(plain))
            same = [r["id"] for r in plain] == [r["id"] for r in results]
            log(f"  → Điểm không filter: {p_score}/2  ({p_why})" + ("   ⚠ giống hệt có filter — câu này chưa thực sự cần filter" if same else ""))

        # Agent answer (dùng đúng tập kết quả đã filter)
        if results:
            prompt = agent.build_prompt(q["q"], results)
            answer = llm_fn(prompt)
        else:
            answer = agent.NO_CONTEXT_ANSWER
        log("  Agent: " + answer.replace("\n", "\n         "))

    log("\n" + "=" * 78)
    log(f"TỔNG: {total}/{2 * len(QUERIES)} điểm   |   có chunk liên quan trong top-{TOP_K}: {relevant_in_top3}/{len(QUERIES)}")
    log(f"embedding cache: hit={embedder.hits} miss={embedder.misses}")
    log("=" * 78)

    embedder.save()
    OUTPUT_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n→ Đã ghi {OUTPUT_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
