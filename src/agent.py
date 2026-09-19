from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    NO_CONTEXT_ANSWER = "Không tìm thấy tài liệu liên quan trong cơ sở tri thức để trả lời câu hỏi này."

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            # Empty store / nothing retrieved: don't waste an LLM call.
            return self.NO_CONTEXT_ANSWER

        prompt = self.build_prompt(question, results)
        return self.llm_fn(prompt)

    def build_prompt(self, question: str, results: list[dict]) -> str:
        """Number each chunk with its source so the answer is traceable."""
        context_blocks = []
        for i, r in enumerate(results, start=1):
            meta = r.get("metadata", {})
            source = meta.get("doc_id") or meta.get("source") or r.get("id", "unknown")
            context_blocks.append(f"[{i}] (nguồn: {source})\n{r['content'].strip()}")
        context = "\n\n".join(context_blocks)

        return (
            "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu được cung cấp.\n"
            "Quy tắc:\n"
            "- CHỈ dùng thông tin trong phần NGỮ CẢNH bên dưới, không tự suy đoán hay bịa thêm.\n"
            "- Khi dùng thông tin từ đoạn nào, trích dẫn số hiệu đoạn đó, ví dụ [1], [2].\n"
            "- Nếu ngữ cảnh không chứa câu trả lời, nói rõ là không tìm thấy trong tài liệu.\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {question}\n\n"
            "TRẢ LỜI:"
        )
