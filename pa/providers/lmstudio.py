"""LM Studio caption provider (OpenAI-compatible chat completions with images)."""
from __future__ import annotations

import base64
import io
import json
import re

import httpx
from PIL import Image, ImageOps

from pa.providers.base import Annotation, CaptionProvider
from pa.providers.registry import register

# NOTE: no maxLength / maxItems anywhere in this schema. LM Studio compiles it to
# a GBNF grammar for constrained decoding, and llama.cpp cannot express length
# bounds -- including them makes the model return an empty completion. Brevity is
# enforced by the prompt instead.
ANNOTATION_SCHEMA = {
    "type": "object",
    # Field order is deliberate: the schema is generated in order, so if the model
    # runs out of tokens mid-object it loses the LAST fields. ocr_text is last
    # because a truncated text transcription still leaves caption and tags intact.
    "properties": {
        "caption": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "scene": {"type": "string"},
        "setting": {"type": "string", "enum": ["indoor", "outdoor", "unknown"]},
        "people_count": {"type": "integer"},
        "ocr_text": {"type": "string"},
    },
    "required": ["caption", "tags", "scene", "setting", "people_count", "ocr_text"],
}

PROMPT = """Annotate this photo for a personal photo-search index. JSON only, be terse.
caption: ONE short factual sentence, max 20 words, describing what is visible.
tags: 6-10 lowercase keywords someone would actually search for - subjects, objects,
  place type, activity, event, notable colours.
scene: 2-3 word scene label, e.g. "mountain landscape", "birthday party", "screenshot".
setting: indoor, outdoor or unknown.
people_count: number of visible people.
ocr_text: legible text transcribed verbatim, else empty string. Transcribe only, never describe."""


def _loads_lenient(content: str) -> dict:
    """Parse the model's JSON, repairing the two ways it realistically breaks.

    Text-heavy photos (screenshots, documents) can exhaust the token budget
    partway through ocr_text, and small models occasionally emit a malformed
    \\uXXXX escape. Both leave caption, tags and scene already complete, so
    salvaging beats discarding the whole annotation and retrying forever.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    repaired = re.sub(r"\\u(?![0-9a-fA-F]{4})", "", content)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Truncated mid-string: close the string, then close any open containers.
    candidate = repaired.rstrip()
    if candidate.endswith("\\"):
        candidate = candidate[:-1]
    quotes = len(re.findall(r'(?<!\\)"', candidate))
    if quotes % 2:
        candidate += '"'
    depth_obj = candidate.count("{") - candidate.count("}")
    depth_arr = candidate.count("[") - candidate.count("]")
    candidate += "]" * max(depth_arr, 0) + "}" * max(depth_obj, 0)
    return json.loads(candidate)


@register("caption", "lmstudio", "openai_compat", "ollama", "vllm", "llamacpp")
class LMStudioCaptioner(CaptionProvider):
    def __init__(self, cfg):
        self.cfg = cfg
        self._client = httpx.Client(base_url=cfg.base_url, timeout=cfg.timeout_s)

    @property
    def model_version(self) -> str:
        return f"{self.cfg.model}@{self.cfg.long_edge}"

    def health(self) -> tuple[bool, str]:
        try:
            r = self._client.get("/v1/models", timeout=10)
            ids = [m["id"] for m in r.json().get("data", [])]
        except Exception as exc:
            return False, f"cannot reach {self.cfg.base_url}: {exc}"
        if self.cfg.model not in ids:
            return False, f"model {self.cfg.model!r} not available; have: {', '.join(ids[:6])}"
        return True, "ok"

    def _encode(self, image_bytes: bytes) -> str:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((self.cfg.long_edge, self.cfg.long_edge), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()

    def annotate(self, image_bytes: bytes) -> Annotation:
        b64 = self._encode(image_bytes)
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "annotation", "strict": True, "schema": ANNOTATION_SCHEMA}},
        }
        if self.cfg.reasoning_effort:
            body["reasoning_effort"] = self.cfg.reasoning_effort

        r = self._client.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        payload = r.json()
        choice = payload["choices"][0]
        content = choice["message"].get("content") or ""
        if not content.strip():
            reason = choice.get("finish_reason")
            detail = (payload.get("usage") or {}).get("completion_tokens_details") or {}
            if reason == "length" and detail.get("reasoning_tokens"):
                raise RuntimeError(
                    f"model spent all {self.cfg.max_tokens} tokens on hidden reasoning; "
                    f"set caption.reasoning_effort='none' or raise max_tokens")
            raise RuntimeError(f"empty completion (finish_reason={reason})")

        data = _loads_lenient(content)
        return Annotation(
            caption=(data.get("caption") or "").strip(),
            tags=[t.strip().lower() for t in data.get("tags", []) if t and t.strip()],
            scene=(data.get("scene") or "").strip(),
            setting=(data.get("setting") or "unknown").strip(),
            people_count=int(data.get("people_count") or 0),
            ocr_text=(data.get("ocr_text") or "").strip(),
            raw=data,
        )

    def close(self) -> None:
        self._client.close()
