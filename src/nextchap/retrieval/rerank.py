"""Optional Qwen3-Reranker cross-encoder (yes/no relevance probability)."""

from __future__ import annotations

from typing import Any

from nextchap.errors import ConfigError

_INSTRUCTION = "Given a request for an upcoming scene, judge whether this passage from the novel is relevant context"
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class QwenReranker:
    def __init__(self, model_name: str, max_length: int = 4096) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ConfigError(
                "Reranking needs the 'rerank' extra.",
                hint="uv sync --extra rerank   (or set retrieval.rerank = false)",
            ) from exc
        self._torch: Any = torch
        self._tok: Any = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self._model: Any = AutoModelForCausalLM.from_pretrained(model_name).eval()
        self._yes = self._tok.convert_tokens_to_ids("yes")
        self._no = self._tok.convert_tokens_to_ids("no")
        self._max = max_length

    def scores(self, query: str, docs: list[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(docs), 8):
            batch = [
                f"{_PREFIX}<Instruct>: {_INSTRUCTION}\n<Query>: {query}\n<Document>: {d}{_SUFFIX}"
                for d in docs[i : i + 8]
            ]
            enc = self._tok(
                batch, padding=True, truncation=True, max_length=self._max, return_tensors="pt"
            )
            with self._torch.no_grad():
                logits = self._model(**enc).logits[:, -1, :]
            pair = self._torch.stack([logits[:, self._no], logits[:, self._yes]], dim=1)
            probs = self._torch.nn.functional.log_softmax(pair, dim=1)[:, 1].exp()
            out.extend(float(p) for p in probs)
        return out
