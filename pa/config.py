"""Configuration: TOML file + env overrides, resolved once at startup."""
from __future__ import annotations

import tomllib
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir
from pydantic import BaseModel, Field

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

[thumbs]
grid_px = {thumbs_grid_px}
view_px = {thumbs_view_px}
quality = {thumbs_quality}
format = "{thumbs_format}"

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
"""


def render_default_toml(cfg: Config) -> str:
    flat = {
        "host": cfg.host, "port": cfg.port, "scan_workers": cfg.scan_workers,
        "sidecar_location": cfg.sidecar.location,
    }
    for section in ("caption", "embed", "face", "thumbs"):
        for key, value in getattr(cfg, section).model_dump().items():
            flat[f"{section}_{key}"] = value
    return DEFAULT_TOML.format(**flat)


_cfg: Config | None = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config.load()
    return _cfg
