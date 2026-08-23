"""InsightFace detection + ArcFace embeddings."""
from __future__ import annotations

import io
import threading

import numpy as np
from PIL import Image, ImageOps

from pa.providers.base import DetectedFace, FaceAnalyzer
from pa.providers.registry import register


@register("face", "insightface")
class InsightFaceAnalyzer(FaceAnalyzer):
    def __init__(self, cfg):
        self.cfg = cfg
        self._app = None
        self._lock = threading.Lock()

    def _load(self):
        if self._app is not None:
            return
        with self._lock:
            if self._app is not None:
                return
            from insightface.app import FaceAnalysis

            from pa.providers import _cuda

            if self.cfg.device == "cuda":
                _cuda.preload()

            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if self.cfg.device == "cuda" else ["CPUExecutionProvider"])
            app = FaceAnalysis(name=self.cfg.model, providers=providers,
                               allowed_modules=["detection", "recognition"])
            app.prepare(ctx_id=0 if self.cfg.device == "cuda" else -1,
                        det_size=(self.cfg.det_size, self.cfg.det_size))
            self._app = app
            self.active_providers = app.models["detection"].session.get_providers()

    @property
    def model_version(self) -> str:
        return f"insightface/{self.cfg.model}"

    def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        self._load()
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
        # InsightFace is OpenCV-based and expects BGR channel order; feeding it
        # RGB silently degrades both detection and embedding quality.
        arr = np.asarray(rgb)[:, :, ::-1]

        out: list[DetectedFace] = []
        for face in self._app.get(arr):
            if face.det_score < self.cfg.min_det_score:
                continue
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            w, h = x2 - x1, y2 - y1
            # Tiny faces in the background produce unreliable embeddings that
            # poison clustering far more than they help recall.
            if min(w, h) < self.cfg.min_face_px:
                continue
            emb = np.asarray(face.normed_embedding, dtype=np.float32)
            out.append(DetectedFace(bbox=(max(x1, 0), max(y1, 0), w, h),
                                    det_score=float(face.det_score), embedding=emb,
                                    src_size=(rgb.width, rgb.height)))
        return out
