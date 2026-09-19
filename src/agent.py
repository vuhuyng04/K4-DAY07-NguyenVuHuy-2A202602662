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
    # Prefix the LLM must put on answers that come from general knowledge (greetings,
    # small talk, off-topic questions) rather than from the retrieved context, so the
    # caller can tell a grounded answer from a general one without parsing prose.
    GENERAL_TAG = "[GENERAL]"

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
        """Number each chunk with its source so the answer is traceable.

        Two modes: grounded answers (default) must only use the context and cite [n];
        greetings / clearly off-topic questions get a short general answer prefixed
        with GENERAL_TAG and no citations.
        """
        context_blocks = []
        for i, r in enumerate(results, start=1):
            meta = r.get("metadata", {})
            source = meta.get("doc_id") or meta.get("source") or r.get("id", "unknown")
            context_blocks.append(f"[{i}] (nguồn: {source})\n{r['content'].strip()}")
        context = "\n\n".join(context_blocks)

        return (
            "Bạn là trợ lý hỏi đáp trên một cơ sở tri thức (các đoạn NGỮ CẢNH bên dưới).\n\n"
            "QUY TẮC MẶC ĐỊNH — câu hỏi về nội dung tài liệu (quy định, số liệu, thủ tục, thời hạn, điều kiện, dịch vụ...):\n"
            "- CHỈ dùng thông tin trong NGỮ CẢNH, không suy đoán hay bổ sung từ kiến thức chung.\n"
            "- Trích dẫn số hiệu đoạn đã dùng, ví dụ [1], [2].\n"
            "- Nếu NGỮ CẢNH không chứa câu trả lời, nói rõ là không tìm thấy trong tài liệu — không đoán.\n"
            "- Hễ NGỮ CẢNH có thông tin trả lời được câu hỏi thì LUÔN áp dụng quy tắc này.\n\n"
            "NGOẠI LỆ — chỉ khi câu hỏi là lời chào, xã giao, hoặc hoàn toàn không liên quan đến chủ đề của tài liệu "
            "(ví dụ: \"hello\", \"bạn là ai\", \"thủ đô của Pháp là gì\"): trả lời ngắn gọn, thân thiện bằng hiểu biết chung, "
            f"bắt đầu câu trả lời bằng nhãn {self.GENERAL_TAG}, KHÔNG trích dẫn số hiệu đoạn, và gợi ý người dùng hỏi về nội dung tài liệu. "
            f"Không bao giờ dùng nhãn {self.GENERAL_TAG} cho câu trả lời lấy từ NGỮ CẢNH.\n\n"
            "Trả lời bằng ngôn ngữ của câu hỏi, ngắn gọn và đúng trọng tâm.\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {question}\n\n"
            "TRẢ LỜI:"
        )
