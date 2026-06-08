"""Sentence embedding wrapper with in-memory cache."""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SentenceEncoder:
    """sentence-transformers 래퍼 (캐싱 포함).

    동일 문자열에 대한 중복 인코딩을 줄이기 위해 dict 캐시를 사용.
    `encode(texts)`는 항상 L2-normalized vector(=cosine 유사도용)를 반환.
    """

    _DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 64,
        cache_size: int = 100_000,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[encoder] loading {model_name} on {device}")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_size = cache_size

    @property
    def dim(self) -> int:
        return self._dim

    def _key(self, text: str) -> str:
        return text if len(text) < 256 else hashlib.md5(text.encode("utf-8")).hexdigest()

    def _zero(self) -> np.ndarray:
        return np.zeros(self._dim, dtype=np.float32)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """텍스트 리스트 → (N, D) L2-normalized embedding."""
        texts = list(texts)
        out: List[Optional[np.ndarray]] = [None] * len(texts)

        # 캐시 조회 + 빈 문자열 zero-vec 처리
        miss_idx: List[int] = []
        miss_text: List[str] = []
        for i, t in enumerate(texts):
            t = (t or "").strip()
            if not t:
                out[i] = self._zero()
                continue
            key = self._key(t)
            if key in self._cache:
                out[i] = self._cache[key]
            else:
                miss_idx.append(i)
                miss_text.append(t)

        if miss_text:
            embs = self._model.encode(
                miss_text,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for i, t, e in zip(miss_idx, miss_text, embs):
                e = e.astype(np.float32, copy=False)
                if len(self._cache) < self._cache_size:
                    self._cache[self._key(t)] = e
                out[i] = e

        return np.stack(out, axis=0)  # type: ignore[arg-type]

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def cosine(self, a: str, b: str) -> float:
        """두 텍스트 간 코사인 유사도 (정규화된 벡터의 dot product)."""
        if not a or not b:
            return 0.0
        v = self.encode([a, b])
        return float(np.dot(v[0], v[1]))


_default_encoder: Optional[SentenceEncoder] = None


def get_default_encoder(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> SentenceEncoder:
    """프로세스 단위 싱글턴 인코더."""
    global _default_encoder
    if _default_encoder is None:
        _default_encoder = SentenceEncoder(
            model_name=model_name or os.environ.get("TASI_ENCODER", SentenceEncoder._DEFAULT_MODEL),
            device=device,
        )
    return _default_encoder
