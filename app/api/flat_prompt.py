import tiktoken

class FlatPromptBuilder:
    def __init__(self, max_chunk_tokens: int = 300, model: str = "gpt-4o"):
        self.max_chunk_tokens = max_chunk_tokens
        self.encoder = tiktoken.encoding_for_model(model)

    def build(self, user_query: str, retrieved_chunk: str = "") -> str:
        chunk = self._truncate(retrieved_chunk, self.max_chunk_tokens)
        return f"{chunk}\nQ: {user_query}\nA:" if chunk else f"Q: {user_query}\nA:"

    def _truncate(self, text: str, max_tokens: int) -> str:
        tokens = self.encoder.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return self.encoder.decode(tokens)