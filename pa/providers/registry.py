"""Provider registry.

A provider is anything that can caption a photo, embed it, or find faces in it.
The pipeline never imports a concrete one; it asks here by the name in config.

Three ways to add one, in increasing order of effort:

1. **In this package** -- write the class, decorate it with `@register`, and add
   the module to `_BUILTIN` below so it gets imported.

2. **From your own file, no packaging** -- point config at it:

       [caption]
       provider = "my_captioner"
       plugins  = ["~/my_captioner.py"]

   The file is imported and any `@register`-decorated classes in it become
   available under the names they declare.

3. **From an installed package** -- expose an entry point and it is discovered
   automatically, with no config at all:

       [project.entry-points."iris.providers"]
       my_captioner = "my_package.providers"

Registration is by (kind, name). Registering an existing name replaces it, which
is what lets you override a built-in without editing this file.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pa.providers.base import (
    CaptionProvider,
    FaceAnalyzer,
    ImageEmbedder,
    TextEmbedder,
)

ENTRY_POINT_GROUP = "iris.providers"

KINDS: dict[str, type] = {
    "caption": CaptionProvider,
    "image_embed": ImageEmbedder,
    "text_embed": TextEmbedder,
    "face": FaceAnalyzer,
}

# kind -> name -> factory taking the relevant config section
_REGISTRY: dict[str, dict[str, Callable[[Any], Any]]] = {k: {} for k in KINDS}

# Built-ins are imported lazily: importing the SigLIP provider pulls in torch,
# which costs seconds and hundreds of MB even when you only wanted a caption.
_BUILTIN = {
    "caption": {
        # These all speak the same OpenAI-shaped API; only base_url and model
        # differ, and both already live in config.
        "lmstudio": "pa.providers.lmstudio",
        "openai_compat": "pa.providers.lmstudio",
        "ollama": "pa.providers.lmstudio",
        "vllm": "pa.providers.lmstudio",
        "llamacpp": "pa.providers.lmstudio",
    },
    "image_embed": {"siglip": "pa.providers.siglip"},
    "text_embed": {"siglip": "pa.providers.siglip"},
    "face": {"insightface": "pa.providers.insightface_provider"},
}

_loaded_plugins: set[str] = set()
_discovered = False


def register(kind: str, *names: str) -> Callable[[type], type]:
    """Class decorator: make this class available as `kind`/`name` in config."""
    if kind not in KINDS:
        raise ValueError(f"unknown provider kind {kind!r}; expected one of {list(KINDS)}")

    def wrap(cls: type) -> type:
        expected = KINDS[kind]
        if not issubclass(cls, expected):
            raise TypeError(f"{cls.__name__} must subclass {expected.__name__} "
                            f"to register as a {kind!r} provider")
        for name in names:
            _REGISTRY[kind][name] = cls
        return cls

    return wrap


def _load_entry_points() -> None:
    """Import any installed package advertising itself as a provider."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    try:
        from importlib.metadata import entry_points
        found = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:
        return
    for ep in found:
        try:
            ep.load()
        except Exception as exc:  # a broken plugin must not take the app down
            print(f"[iris] provider plugin {ep.name!r} failed to load: {exc}",
                  file=sys.stderr)


def _load_plugin_file(path_str: str) -> None:
    """Import a loose .py file so its @register decorators run."""
    if path_str in _loaded_plugins:
        return
    _loaded_plugins.add(path_str)
    path = Path(path_str).expanduser()
    if not path.exists():
        print(f"[iris] plugin file not found: {path}", file=sys.stderr)
        return
    spec = importlib.util.spec_from_file_location(f"iris_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"[iris] plugin {path} failed to load: {exc}", file=sys.stderr)


def _resolve(kind: str, name: str, plugins: list[str] | None = None) -> Callable[[Any], Any]:
    _load_entry_points()
    for path in plugins or []:
        _load_plugin_file(path)

    if name not in _REGISTRY[kind] and name in _BUILTIN.get(kind, {}):
        importlib.import_module(_BUILTIN[kind][name])  # its @register runs on import

    factory = _REGISTRY[kind].get(name)
    if factory is None:
        raise ValueError(
            f"unknown {kind} provider {name!r}. Available: "
            f"{', '.join(sorted(available(kind))) or 'none'}")
    return factory


def available(kind: str) -> set[str]:
    """Every name usable for this kind, built-in or plugged in."""
    _load_entry_points()
    return set(_REGISTRY[kind]) | set(_BUILTIN.get(kind, {}))


def _plugins(cfg, section) -> list[str]:
    return list(getattr(section, "plugins", None) or getattr(cfg, "plugins", []) or [])


def get_captioner(cfg) -> CaptionProvider:
    return _resolve("caption", cfg.caption.provider, _plugins(cfg, cfg.caption))(cfg.caption)


def get_image_embedder(cfg) -> ImageEmbedder:
    return _resolve("image_embed", cfg.embed.provider, _plugins(cfg, cfg.embed))(cfg.embed)


def get_text_embedder(cfg) -> TextEmbedder:
    """Must share a vector space with the image embedder, so it defaults to the
    same provider unless config deliberately names a different one."""
    name = getattr(cfg.embed, "text_provider", None) or cfg.embed.provider
    return _resolve("text_embed", name, _plugins(cfg, cfg.embed))(cfg.embed)


def get_face_analyzer(cfg) -> FaceAnalyzer:
    return _resolve("face", cfg.face.provider, _plugins(cfg, cfg.face))(cfg.face)
