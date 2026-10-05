class TokenBatcher:
    def __init__(self, k: int = 4, flush_on_sentence: bool = True):
        self.k = k
        self.flush_on_sentence = flush_on_sentence
        self.buffer = []

    def add(self, token: str) -> list[str] | None:
        self.buffer.append(token)
        if len(self.buffer) >= self.k:
            return self._flush()
        if self.flush_on_sentence and token.rstrip().endswith(('.','!','?')):
            return self._flush()
        return None

    def flush(self) -> list[str] | None:
        if self.buffer:
            return self._flush()
        return None

    def _flush(self) -> list[str]:
        batch = self.buffer[:]
        self.buffer.clear()
        return batch
    