"""Wrapper around KRLabsOrg's LettuceDetect for our (query, context, output) test format.

Predictions come back from `HallucinationDetector.predict(..., output_format='spans')` as a list
of dicts with token-level start/end offsets *into the answer*. We use those directly as char
spans into `output`.
"""
from __future__ import annotations

from typing import Any, Iterable


DEFAULT_MODEL = "KRLabsOrg/lettucedect-large-modernbert-en-v1"


class LettuceDetectRunner:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        from lettucedetect.models.inference import HallucinationDetector  # noqa: F401

        self.detector = HallucinationDetector(method="transformer", model_path=model_path)
        self.model_path = model_path

    def predict_one(self, query: str, context: str, output: str) -> list[dict[str, Any]]:
        # LettuceDetect expects context as list-of-strings (for multi-chunk RAG); we pass our
        # tool output as a single chunk.
        raw = self.detector.predict(
            context=[context],
            question=query,
            answer=output,
            output_format="spans",
        )
        result: list[dict[str, Any]] = []
        for span in raw:
            result.append({
                "start": int(span["start"]),
                "end": int(span["end"]),
                "text": span.get("text", output[int(span["start"]) : int(span["end"])]),
                "confidence": float(span.get("confidence", 0.0)),
            })
        return result

    def predict_many(self, samples: Iterable[dict[str, Any]],
                     progress: bool = True) -> list[list[dict[str, Any]]]:
        try:
            from tqdm import tqdm  # noqa: F401
            iterable = tqdm(list(samples)) if progress else samples
        except ImportError:
            iterable = samples
        out: list[list[dict[str, Any]]] = []
        for s in iterable:
            preds = self.predict_one(s["query"], s["context"], s["output"])
            out.append(preds)
        return out
