"""Is the configured stack actually usable right now?

`pa config check` and the Settings screen both need this, and they need it to
agree -- a setup that the terminal calls broken and the browser calls fine is
worse than no check at all. So the answers are computed here, as data, and each
caller decides how to draw them.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def wsl_host_ip() -> str | None:
    """The Windows host's address as seen from inside WSL, or None if not WSL."""
    try:
        if "microsoft" not in Path("/proc/version").read_text().lower():
            return None
    except OSError:
        return None
    try:
        # The default route's gateway is the Windows host on WSL2 NAT networking.
        out = subprocess.run(["ip", "route", "show", "default"],
                             capture_output=True, text=True, timeout=5).stdout
        parts = out.split()
        return parts[parts.index("via") + 1] if "via" in parts else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _caption_check(cfg) -> dict[str, Any]:
    from pa.providers.registry import get_captioner
    try:
        ok, msg = get_captioner(cfg).health()
    except Exception as exc:
        ok, msg = False, str(exc)
    out: dict[str, Any] = {
        "id": "caption", "label": "Caption model", "ok": ok,
        "detail": f"{cfg.caption.model} @ {cfg.caption.base_url}",
        "message": None if ok else msg, "hint": None,
    }
    if not ok and ("127.0.0.1" in cfg.caption.base_url or "localhost" in cfg.caption.base_url):
        # Under WSL, localhost is the Linux side. A model server running on the
        # Windows host is reachable at the host's IP, not 127.0.0.1 -- the single
        # most likely reason a fresh install cannot connect.
        host_ip = wsl_host_ip()
        if host_ip:
            out["hint"] = (f"Running under WSL: localhost is not the Windows host. "
                           f"If the model server runs on Windows, try "
                           f"http://{host_ip}:1234")
    return out


def _gpu_check(cfg) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"id": "gpu", "label": "GPU (embeddings)", "ok": False, "level": "warn",
                "detail": "torch not installed",
                "message": "install with: uv sync --extra ml", "hint": None}
    cuda = torch.cuda.is_available()
    return {"id": "gpu", "label": "GPU (embeddings)", "ok": True,
            "level": "ok" if cuda else "warn",
            "detail": torch.cuda.get_device_name(0) if cuda
            else "no CUDA - embeddings will use the CPU, roughly 10x slower",
            "message": None, "hint": None}


def _face_check(cfg) -> dict[str, Any]:
    try:
        import onnxruntime
    except ImportError:
        return {"id": "faces", "label": "GPU (faces)", "ok": False, "level": "warn",
                "detail": "onnxruntime not installed",
                "message": "install with: uv sync --extra ml", "hint": None}
    gpu = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    return {"id": "faces", "label": "GPU (faces)", "ok": True,
            "level": "ok" if gpu else "warn",
            "detail": "GPU" if gpu else "CPU only - face detection ~10x slower",
            "message": None, "hint": None}


def report(cfg) -> dict[str, Any]:
    """Every check, as data. `ok` is the overall pass/fail for an exit code."""
    checks = [_caption_check(cfg), _gpu_check(cfg), _face_check(cfg)]
    for c in checks:
        c.setdefault("level", "ok" if c["ok"] else "fail")
    return {"ok": all(c["level"] != "fail" for c in checks), "checks": checks}
