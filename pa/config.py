"""Configuration: TOML file + env overrides, resolved once at startup."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir
from pydantic import BaseModel, Field, field_validator

from pa.db.repo import STAGES

APP = "photo_anotater"
CONFIG_PATH = Path(user_data_dir(APP)) / "config.toml"


class CaptionConfig(BaseModel):
    provider: str = "lmstudio"
    base_url: str = "http://127.0.0.1:1234"
    model: str = "google/gemma-4-12b-qat"
    long_edge: int = 896
    max_tokens: int = 1400  # text-heavy screenshots need room for a full OCR pass
    temperature: float = 0.2
    timeout_s: int = 300
    # Gemma 4 is a reasoning model: it spends hundreds of hidden tokens thinking
    # before emitting any JSON, which costs ~4x wall clock for no gain on a
    # captioning task. "none" skips that phase. Raise to "low"/"medium" if
    # captions ever look careless.
    reasoning_effort: str | None = "none"
    # Extra .py files to import so their @register providers become available.
    plugins: list[str] = Field(default_factory=list)


class EmbedConfig(BaseModel):
    provider: str = "siglip"
    model: str = "google/siglip2-so400m-patch14-384"
    device: str = "cuda"
    batch_size: int = 16
    dim: int = 1152
    # Vector-hit cutoffs, calibrated against SigLIP 2 on a real library: genuine
    # matches score ~0.12-0.13, the best *non*-match tops out around 0.055.
    # A hit must clear both the absolute floor and the relative one.
    min_score: float = 0.07
    rel_score: float = 0.6


class FaceConfig(BaseModel):
    provider: str = "insightface"
    model: str = "buffalo_l"
    device: str = "cuda"
    det_size: int = 640
    min_det_score: float = 0.55
    min_face_px: int = 40
    # Cosine distance below which two ArcFace embeddings are the same person.
    cluster_eps: float = 0.42
    min_cluster_size: int = 3
    # Similarity at which an unnamed group is worth *offering* as somebody you
    # already named. Deliberately below the automatic threshold (1 - cluster_eps):
    # anything above that was already attached without asking, so a suggestion
    # only ever concerns the band that is too uncertain to decide for you.
    suggest_min_similarity: float = 0.35


class SidecarConfig(BaseModel):
    # "app"    -> under the app's data directory, mirroring the folder tree, so
    #             your photo folders stay exactly as you left them.
    # "beside" -> photo.jpg.xmp next to the original, which is what Lightroom and
    #             digiKam read automatically.
    location: str = "app"


class ThumbConfig(BaseModel):
    grid_px: int = 320
    view_px: int = 1600
    quality: int = 82
    format: str = "WEBP"
    # Photos thumbnailed at once. Pillow releases the GIL around decode, resize
    # and encode, so threads genuinely use more than one core here: measured on
    # 12MP JPEGs, 8 threads is about 6x one. 0 picks a number from the machine.
    workers: int = 0


class Paths(BaseModel):
    data_dir: Path = Field(default_factory=lambda: Path(user_data_dir(APP)))
    cache_dir: Path = Field(default_factory=lambda: Path(user_cache_dir(APP)))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "library.db"

    @property
    def vectors_dir(self) -> Path:
        return self.data_dir / "vectors"

    @property
    def thumbs_dir(self) -> Path:
        return self.cache_dir / "thumbs"

    @property
    def models_dir(self) -> Path:
        return self.cache_dir / "models"

    @property
    def sidecars_dir(self) -> Path:
        return self.data_dir / "sidecars"


class Config(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    caption: CaptionConfig = Field(default_factory=CaptionConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    face: FaceConfig = Field(default_factory=FaceConfig)
    thumbs: ThumbConfig = Field(default_factory=ThumbConfig)
    sidecar: SidecarConfig = Field(default_factory=SidecarConfig)
    host: str = "127.0.0.1"
    port: int = 8420
    scan_workers: int = 8
    # Stages to run automatically when a scan finishes. A scan only enqueues
    # work; without this, nothing consumes the queue until someone asks it to,
    # and a freshly added folder shows a grid of grey placeholders. Thumbnails
    # alone are the safe default: they need no GPU and no model server, and
    # they are the only stage whose absence is visible on every single tile.
    # Empty list turns it off entirely.
    auto_process: list[str] = Field(default_factory=lambda: ["thumbs"])

    @field_validator("auto_process")
    @classmethod
    def _known_stages(cls, v: list[str]) -> list[str]:
        bad = [s for s in v if s not in STAGES]
        if bad:
            raise ValueError(f"unknown stage(s) {bad}; expected any of {list(STAGES)}")
        return v

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        path = path or CONFIG_PATH
        data = tomllib.loads(path.read_text()) if path.exists() else {}
        # [server] reads better in a config file than three loose top-level keys.
        server = data.pop("server", None)
        if isinstance(server, dict):
            data.update(server)
        cfg = cls.model_validate(data)
        for d in (cfg.paths.data_dir, cfg.paths.cache_dir,
                  cfg.paths.thumbs_dir, cfg.paths.models_dir):
            d.mkdir(parents=True, exist_ok=True)
        return cfg


DEFAULT_TOML = """\
# Iris configuration.
# Every value here is optional -- delete a line to fall back to the default.
# Edit with:  pa config edit        Show what is active:  pa config show

[caption]
# Which vision model describes your photos, and where it runs.
# provider: lmstudio | ollama | vllm | llamacpp | openai_compat
#   (all speak the same OpenAI-shaped API; only base_url and model differ)
provider = "{caption_provider}"
base_url = "{caption_base_url}"
model = "{caption_model}"

# Longest edge the image is scaled to before being sent. Bigger reads small text
# better and costs more time.
long_edge = {caption_long_edge}
max_tokens = {caption_max_tokens}
temperature = {caption_temperature}
timeout_s = {caption_timeout_s}

# Some models think before answering. On a captioning task that is pure cost:
# "none" skips it and is roughly 4x faster. Raise to "low" or "medium" if
# captions start looking careless.
reasoning_effort = "{caption_reasoning_effort}"

# Extra .py files to import so their @register providers become available.
plugins = {caption_plugins}

[embed]
# Turns photos and your search text into vectors, so "mountains" finds a
# mountain nobody tagged. Runs in this process -- LM Studio cannot embed images.
provider = "{embed_provider}"
model = "{embed_model}"
device = "{embed_device}"          # cuda | cpu
batch_size = {embed_batch_size}
dim = {embed_dim}

# How close a photo must be to count as a match. A real match scores about 0.12,
# unrelated content below 0.06. Lower these to widen results, raise them if
# searches feel noisy. A hit must clear both.
min_score = {embed_min_score}
rel_score = {embed_rel_score}      # also: at least this fraction of the best hit

[face]
provider = "{face_provider}"
model = "{face_model}"             # buffalo_l (accurate) | buffalo_s (faster)
device = "{face_device}"
det_size = {face_det_size}
min_det_score = {face_min_det_score}
min_face_px = {face_min_face_px}   # ignore faces smaller than this, they cluster badly

# Cosine distance below which two faces are treated as the same person.
# Lower = stricter, splits people into more groups. Higher = merges strangers.
cluster_eps = {face_cluster_eps}
min_cluster_size = {face_min_cluster_size}

# How alike an unnamed group must be to someone already named before the People
# screen offers them as a guess. Lower to get more guesses and more wrong ones.
suggest_min_similarity = {face_suggest_min_similarity}

[thumbs]
grid_px = {thumbs_grid_px}
view_px = {thumbs_view_px}
quality = {thumbs_quality}
format = "{thumbs_format}"

# How many photos to thumbnail at once. 0 picks a number from the machine.
# Thumbnailing is the one stage that is pure CPU, and it scales almost linearly
# until the cores run out.
workers = {thumbs_workers}

[sidecar]
# "app"    -> sidecars live under the app's data directory, mirroring your
#             folder tree, leaving your photo folders untouched.
# "beside" -> photo.jpg.xmp next to the original, which Lightroom and digiKam
#             read automatically.
location = "{sidecar_location}"

[server]
host = "{host}"                    # 127.0.0.1 keeps it to this machine only
port = {port}
scan_workers = {scan_workers}

# Stages run automatically when a scan finishes, so a folder you just added
# does not sit there as a grid of grey placeholders waiting to be told to
# index. Any of: thumbs, embed, faces, caption. [] turns it off.
# thumbs is the safe default -- no GPU, no model server, and it is the one
# stage whose absence you can see on every tile.
auto_process = {auto_process}
"""


def _toml_value(value):
    """Render one setting as TOML. Lists go through JSON, which produces a valid
    TOML array and -- the part that matters on Windows -- escapes the
    backslashes in a plugin path instead of emitting a broken string."""
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return value


def render_default_toml(cfg: Config) -> str:
    flat = {
        "host": cfg.host, "port": cfg.port, "scan_workers": cfg.scan_workers,
        "auto_process": _toml_value(cfg.auto_process),
        "sidecar_location": cfg.sidecar.location,
    }
    for section in ("caption", "embed", "face", "thumbs"):
        for key, value in getattr(cfg, section).model_dump().items():
            flat[f"{section}_{key}"] = _toml_value(value)
    return DEFAULT_TOML.format(**flat)


_cfg: Config | None = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config.load()
    return _cfg


# Settings whose new value cannot take effect in a running server, and why.
# The UI shows these next to the field so a save that appears to do nothing is
# explained rather than mysterious.
RESTART_REQUIRED = {
    "server.host": "the server is already listening on the old address",
    "server.port": "the server is already listening on the old port",
}
# Settings that change what a stored vector or face *means*. Saving them is
# harmless, but everything already indexed was computed with the old value and
# has to be recomputed before it agrees with the new one.
REINDEX_REQUIRED = {
    "embed.provider": "embed",
    "embed.model": "embed",
    "embed.dim": "embed",
    "face.provider": "faces",
    "face.model": "faces",
    "face.det_size": "faces",
    "face.min_det_score": "faces",
    "face.min_face_px": "faces",
    # Thumbnails are skipped when the file already exists, so these need the
    # cache thrown away as well -- re-queueing alone would change nothing.
    "thumbs.grid_px": "thumbs",
    "thumbs.view_px": "thumbs",
    "thumbs.quality": "thumbs",
    "thumbs.format": "thumbs",
}


def _normalise(data: dict) -> dict:
    """Fold the [server] table up to the top level, where Config expects it."""
    data = dict(data)
    server = data.pop("server", None)
    if isinstance(server, dict):
        data.update(server)
    return data


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def save_config(patch: dict, path: Path | None = None) -> Config:
    """Merge a partial settings dict into the config file and reload it.

    Partial on purpose: the UI sends only the sections it edited, so a setting
    this version of the UI has never heard of survives being saved over.

    Validation happens before the write, so a rejected value leaves the file
    exactly as it was rather than half-applied.
    """
    path = path or CONFIG_PATH
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    merged = _deep_merge(raw, patch)
    cfg = Config.model_validate(_normalise(merged))  # raises before anything is written

    text = render_default_toml(cfg)
    # DEFAULT_TOML has no [paths] section, so a hand-set data_dir or cache_dir
    # would be quietly dropped the first time anyone pressed Save in the UI.
    paths = merged.get("paths")
    if isinstance(paths, dict) and paths:
        text += "\n[paths]\n" + "".join(
            f"{k} = {json.dumps(str(v))}\n" for k, v in paths.items())

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)  # atomic: an interrupted save never leaves a truncated config
    return reload_config(path)


def reload_config(path: Path | None = None) -> Config:
    """Re-read the file into the process-wide config object, *in place*.

    In place matters. create_app() captures the Config in a closure and every
    request handler reads through that one reference, so rebinding the module
    global would leave the running server on the old settings until restart.
    Mutating the object everyone already holds is what makes Save take effect.
    """
    global _cfg
    fresh = Config.load(path)
    if _cfg is None:
        _cfg = fresh
    else:
        for name in Config.model_fields:
            setattr(_cfg, name, getattr(fresh, name))
    return _cfg
