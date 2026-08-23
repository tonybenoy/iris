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
            self._model = AutoModel.from_pretrained(
                self.cfg.model, dtype=dtype).to(device).eval()
            self._processor = AutoProcessor.from_pretrained(self.cfg.model)

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
