"""Benchmark a VLM provider on real photos: quality, latency, tokens/sec.

Usage:
    uv run scripts/bench_vlm.py --model google/gemma-4-12b-qat --long-edge 896 IMG1.jpg IMG2.jpg
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
import urllib.request

from PIL import Image

ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "scene": {"type": "string"},
        "setting": {"type": "string", "enum": ["indoor", "outdoor", "unknown"]},
        # NOTE: no maxLength/maxItems here - LM Studio compiles the schema to a GBNF
        # grammar and llama.cpp cannot express length bounds; adding them yields empty
        # completions. Brevity is enforced by the prompt instead.
        "people_count": {"type": "integer"},
        "ocr_text": {"type": "string"},
    },
    "required": ["caption", "tags", "scene", "setting", "people_count", "ocr_text"],
}

PROMPT = """Annotate this photo for a photo-search index. JSON only, be terse.
caption: ONE short factual sentence, max 20 words.
tags: exactly 8 lowercase keywords, one or two words each.
scene: 2-3 word scene label.
setting: indoor, outdoor or unknown.
people_count: integer count of visible people.
ocr_text: legible text transcribed verbatim, else empty string. Transcribe only, never describe."""


def encode(path: str, long_edge: int) -> str:
    im = Image.open(path)
    im.draft("RGB", (long_edge * 2, long_edge * 2))  # JPEG DCT downscale: decodes ~4x faster
    im = im.convert("RGB")
    im.thumbnail((long_edge, long_edge), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def annotate(host: str, model: str, b64: str, max_tokens: int) -> tuple[dict, dict, float]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "annotation", "strict": True, "schema": ANNOTATION_SCHEMA}},
    }
    req = urllib.request.Request(
        f"{host}/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    return json.loads(data["choices"][0]["message"]["content"]), data.get("usage", {}), dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--host", default="http://127.0.0.1:1234")
    ap.add_argument("--model", default="google/gemma-4-12b-qat")
    ap.add_argument("--long-edge", type=int, default=896)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--quiet", action="store_true", help="timings only, no annotations")
    args = ap.parse_args()

    lat: list[float] = []
    out_tok: list[int] = []
    for path in args.images:
        b64 = encode(path, args.long_edge)
        try:
            ann, usage, dt = annotate(args.host, args.model, b64, args.max_tokens)
        except Exception as exc:
            print(f"FAIL {path}: {exc}")
            continue
        lat.append(dt)
        out_tok.append(usage.get("completion_tokens", 0))
        name = path.rsplit("/", 1)[-1]
        print(f"\n=== {name}  {dt:.2f}s  in={usage.get('prompt_tokens')} "
              f"out={usage.get('completion_tokens')}")
        if not args.quiet:
            print(f"  caption: {ann['caption']}")
            print(f"  tags:    {', '.join(ann['tags'])}")
            print(f"  scene:   {ann['scene']} | {ann['setting']} | people={ann['people_count']}"
                  + (f" | ocr={ann['ocr_text'][:60]!r}" if ann["ocr_text"] else ""))

    if lat:
        med = statistics.median(lat)
        tps = sum(out_tok) / sum(lat)
        print(f"\n--- {args.model} @ {args.long_edge}px ---")
        print(f"median {med:.2f}s/photo | mean {statistics.mean(lat):.2f}s | "
              f"{tps:.0f} tok/s | median out {statistics.median(out_tok):.0f} tok")
        print(f"projected 500k photos: {med * 500_000 / 86400:.1f} days single-stream")


if __name__ == "__main__":
    main()
