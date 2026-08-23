"""Provider interfaces.

Nothing in the pipeline knows what LM Studio is. A backend is one file
implementing one of these, plus a line in the registry -- which is what makes it
possible to swap the captioner for Ollama, vLLM or a hosted API later without
touching the indexer.

Image and text embedders are separate interfaces but must share a vector space,
so they are versioned together via `model_version`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Annotation:
    caption: str = ""
    tags: list[str] = field(default_factory=list)
    scene: str = ""
    setting: str = "unknown"
    people_count: int = 0
    ocr_text: str = ""
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"caption": self.caption, "tags": self.tags, "scene": self.scene,
                "setting": self.setting, "people_count": self.people_count,
                "ocr_text": self.ocr_text}


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]
    det_score: float
    embedding: np.ndarray
    blur_score: float | None = None
    # Size of the image the box was measured against. Boxes are meaningless
    # without it -- they cannot be rescaled to the original or normalised for
    # an XMP sidecar.
    src_size: tuple[int, int] = (0, 0)


class CaptionProvider(ABC):
    @property
    @abstractmethod
    def model_version(self) -> str: ...

    @abstractmethod
    def annotate(self, image_bytes: bytes) -> Annotation: ...

    def health(self) -> tuple[bool, str]:
        return True, "ok"


class ImageEmbedder(ABC):
    @property
    @abstractmethod
    def model_version(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed_images(self, images: list[bytes]) -> np.ndarray: ...


class TextEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


class FaceAnalyzer(ABC):
    @property
    @abstractmethod
    def model_version(self) -> str: ...

    @abstractmethod
    def detect(self, image_bytes: bytes) -> list[DetectedFace]: ...
