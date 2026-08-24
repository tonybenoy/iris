"""SigLIP 2 image/text embeddings, in-process.

Deliberately not routed through LM Studio: its /v1/embeddings endpoint is
text-only and cannot embed an image, so semantic photo search has to run the
vision tower locally regardless of where the captioner lives.
"""
from __future__ import annotations

import io
import threading

import numpy as np
from PIL import Image, ImageOps

from pa.providers.base import ImageEmbedder, TextEmbedder
from pa.providers.registry import register


def _from_cache_first(loader, model_id: str, **kwargs):
    """Load from the local cache without contacting the Hub, downloading only
    if it is genuinely not there yet.

    transformers otherwise revalidates against huggingface.co on every single
    load. The weights are not re-downloaded -- they are cached and reused -- but
    each start pays a network round-trip, prints an unauthenticated-requests
    warning, and fails outright with no internet. For a tool whose whole promise
    is that nothing leaves the machine, reaching for the network to load a model
    it already has is the wrong default.
    """
    try:
        return loader.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception:
        # Not cached yet (or the cache is incomplete): fetch it, once.
        return loader.from_pretrained(model_id, **kwargs)


@register("image_embed", "siglip")
@register("text_embed", "siglip")
class SiglipEmbedder(ImageEmbedder, TextEmbedder):
    def __init__(self, cfg):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._torch = None

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel, AutoProcessor

            self._torch = torch
            device = self.cfg.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self.device = device
            # fp16 halves VRAM and is faster; the retrieval quality difference is
            # not measurable, and this GPU is shared with the captioner.
            dtype = torch.float16 if device == "cuda" else torch.float32
            self._model = _from_cache_first(
                AutoModel, self.cfg.model, dtype=dtype).to(device).eval()
            self._processor = _from_cache_first(AutoProcessor, self.cfg.model)

    @property
    def model_version(self) -> str:
        return self.cfg.model

    @property
    def dim(self) -> int:
        return self.cfg.dim

    @staticmethod
    def _features(out):
        """Unwrap whatever get_*_features returned.

        SigLIP 2 returns a BaseModelOutputWithPooling here where CLIP and SigLIP 1
        return a plain tensor, so accepting only a tensor breaks on the model this
        project actually defaults to.
        """
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state.mean(dim=1)
        return out

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        """Unit-normalise so cosine similarity is a plain dot product, which is
        what the vector index expects."""
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    def embed_images(self, images: list[bytes]) -> np.ndarray:
        self._load()
        torch = self._torch
        pil = []
        for raw in images:
            with Image.open(io.BytesIO(raw)) as img:
                pil.append(ImageOps.exif_transpose(img).convert("RGB"))
        out = []
        for i in range(0, len(pil), self.cfg.batch_size):
            batch = pil[i:i + self.cfg.batch_size]
            inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                feats = self._features(self._model.get_image_features(**inputs))
            out.append(feats.float().cpu().numpy())
        return self._normalize(np.vstack(out))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self._load()
        torch = self._torch
        # SigLIP was trained with padding="max_length"; using the tokenizer's
        # default dynamic padding measurably degrades retrieval quality.
        inputs = self._processor(
            text=texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            feats = self._features(self._model.get_text_features(**inputs))
        return self._normalize(feats.float().cpu().numpy())
