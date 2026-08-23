# Iris

*Every photo, in view.*

Local AI photo index and search. Points at folders on any drive, describes every
photo with a local vision model, groups faces into people you name once, and
makes the lot searchable — by meaning, by words, by person, by date.

Nothing leaves the machine. Originals are never modified.

Install first — the same one command on Linux, macOS and Windows.
[`uv`](https://docs.astral.sh/uv/) installs Python 3.12 itself, so it is the only
prerequisite:

```bash
uv sync --extra ml
```

That puts the `pa` command inside the project's `.venv`, not on your PATH. Either
prefix commands with `uv run` (`uv run pa config init`), or activate the
environment once per shell:

| | |
|---|---|
| **Linux / macOS** | `source .venv/bin/activate` |
| **Windows (PowerShell)** | `.\.venv\Scripts\Activate.ps1` |

Then:

```bash
pa config init                      # write a commented settings file
pa config check                     # confirm the model server and GPU are reachable
pa root add /mnt/d/Photos/2024      # register a folder (any drive)
pa process                          # thumbnails, embeddings, faces, captions
pa people cluster                   # group faces
pa serve                            # http://127.0.0.1:8420
pa search "birds on a lake"         # or search from the terminal
```

## What it does

- **Finds photos by meaning.** "mountains", "golden hour", "crowded street" work
  without anyone having tagged them, via SigLIP 2 image embeddings.
- **Finds photos by person.** Faces are detected, grouped, and named once; new
  photos of that person are matched automatically on the next run.
- **Reads text in photos.** Screenshots, signs and documents are OCR'd into the
  search index.
- **Accepts your own labels** alongside the AI's, and never overwrites them.
- **Survives unplugged drives.** Photos on a disconnected external HDD still
  appear in search with thumbnails, marked as offline.
- **Never stores a photo twice.** Identity is a BLAKE3 hash of the file's
  contents, so the same image on three drives is one entry with three locations,
  annotated once. Near-identical shots (resized exports, re-compressed copies)
  are found separately with a perceptual hash.
- **Exports to XMP sidecars** so Lightroom, digiKam and exiftool can read your
  captions, keywords and named faces. Originals are never touched.
- **Maps photos that recorded a location.**

## Formats

| | |
|---|---|
| **Photos** | jpg, jpeg, png, webp, gif, bmp, tif, tiff, avif |
| **Phone photos** | heic, heif (via `pillow-heif`) |
| **Camera RAW** | cr2, cr3, nef, arw, dng, orf, rw2, raf, pef (via `rawpy`/LibRaw) |
| **Video** | not supported |

RAW files are decoded from their embedded full-size JPEG preview where one
exists — an order of magnitude faster than demosaicing, and the same image every
other photo manager shows you. Files without a preview are demosaiced properly.

## Deleting photos

Iris never deletes or modifies your photos. It only reacts to what you do:

- Delete a file and rescan → that copy is marked `missing` and stops appearing
  in search. If the photo exists in another indexed folder, it stays findable
  through that copy.
- Unplug a drive → its files go `offline`, **not** missing. They keep appearing
  in search with cached thumbnails, marked as needing that drive reconnected.
  This distinction matters: one is gone for good, the other is coming back.
- `pa prune` permanently drops photos with no readable copy left anywhere.
  `pa prune --keep-missing` only clears photos from folders you removed.

## Requirements

- Python 3.12 (`uv` installs it)
- An NVIDIA GPU for embeddings and face detection (CPU works, ~10x slower).
  The caption model needs roughly 7GB of VRAM at 4-bit; embeddings and face
  detection add about 3GB.
- [LM Studio](https://lmstudio.ai) with a vision model loaded, for captions

Everything else is installed by `uv sync --extra ml`. Without `--extra ml` you
get a working CLI that cannot embed, detect faces or caption.

## Models

| Job | Model | Runs |
|---|---|---|
| Caption, tags, scene, OCR | `google/gemma-4-12b-qat` | LM Studio |
| Image + text embeddings | SigLIP 2 `so400m-patch14-384` | in-process, CUDA |
| Face detect + recognise | InsightFace `buffalo_l` | in-process, CUDA |

Measured on a 24GB consumer GPU: captions ~2.5s/photo, embeddings ~60ms, faces
~26ms. The fast stages (thumbnails, embeddings, faces) make a folder searchable
in about a tenth of a second per photo; captions backfill afterwards. Run
`scripts/bench_vlm.py` to measure your own hardware before committing to a
model.

Image embeddings deliberately do **not** go through LM Studio: its
`/v1/embeddings` endpoint is text-only and cannot embed an image. See
[Extending](#extending) for swapping any of these out.

## Platforms

| | |
|---|---|
| **Linux** | fully supported and what this was developed on |
| **WSL** | supported; Windows drives are reached through `/mnt/*` |
| **Windows (native)** | supported; drive serials via `GetVolumeInformationW`, DLL preloading for the GPU |
| **macOS** | untested. No CUDA, so embeddings and faces fall back to CPU |

A library is portable between WSL and native Windows: both read the same NTFS
volume serial, so the same `library.db` resolves either way.

## Configuration

```bash
pa config init      # create it, fully commented, with every current value
pa config show      # what is actually in effect right now
pa config edit      # open it in $EDITOR
pa config check     # is the model server up? is the GPU being used?
pa config path      # where the file lives
pa init             # create the database and print every path it uses
```

The file lives at `~/.local/share/photo_anotater/config.toml`
(`%LOCALAPPDATA%\photo_anotater\config.toml` on Windows). Every value is
optional — delete a line to fall back to the default.

```toml
[caption]
provider = "lmstudio"                    # or ollama / vllm / llamacpp / openai_compat
base_url = "http://127.0.0.1:1234"
model = "google/gemma-4-12b-qat"
reasoning_effort = "none"                # models that think first cost ~4x here

[embed]
model = "google/siglip2-so400m-patch14-384"
min_score = 0.07                         # a real match scores ~0.12, noise below 0.06
rel_score = 0.6                          # and at least this fraction of the best hit

[face]
cluster_eps = 0.42                       # cosine distance for "same person"
min_cluster_size = 3

[sidecar]
location = "app"                         # or "beside"

[server]
host = "127.0.0.1"                       # this machine only
port = 8420
```

**Running under WSL?** `127.0.0.1` is the Linux side, not Windows. If LM Studio
runs on the Windows host, point `base_url` at the host's IP —
`pa config check` detects this case and tells you which address to use.

## Searching

The query box takes plain language and structured filters together:

```
mountains
person:Sarah last summer
photos of Sarah in the mountains
tag:sunset favorites
screenshots from May 2026
camera:"OnePlus 13" 2024
```

Three engines run and their rankings are fused with Reciprocal Rank Fusion:
literal words (SQLite FTS5), meaning (SigLIP vectors), and structured filters.
Every result tile shows which of them found it — amber for a word match, blue for
a meaning match, grey for a filter. When nothing genuinely matches, you get no
results rather than the whole library.

## Editing what it worked out

The models produce a starting point, not the last word.

- **Captions and transcribed text** are editable in place in the lightbox. Click,
  type, and it saves when you click away (Escape abandons the edit). Your version
  is stored as its own annotation and **always wins over model output**, however
  many times you re-caption the library. "Use the model's version" puts it back.
- **Tags** are stored per source. Adding or removing your own in the lightbox
  never disturbs what the captioner produced, and re-running the AI stage never
  deletes a tag you typed.
- **Faces** split across several groups whenever lighting or angle varies — tick
  "same person as…" on each and merge them in one action. `×` on a face in the
  lightbox means "that isn't them" and returns it to the queue.
- **People** can be renamed, or un-named to send their faces back for regrouping.

## Sidecars

```bash
pa sidecar export                  # into the app directory (default)
pa sidecar export --beside         # next to each photo instead
pa sidecar import                  # read keywords and ratings back in
```

By default sidecars go to `~/.local/share/photo_anotater/sidecars/`, mirroring
your folder tree, so **your photo folders are left exactly as they were**. The
mirrored tree can be copied over your photo folders later if you ever do want
them beside the originals — which is what Lightroom and digiKam read
automatically. `--beside` writes them there directly.

Export writes `dc:description`, `dc:subject` and `mwg-rs:Regions` (named face
rectangles, normalised and centre-based per the MWG spec). Import only adds
keywords and ratings as *manual* tags — it will not overwrite captions or faces,
since those are the pipeline's output.

```toml
[sidecar]
location = "app"    # or "beside"
```

## Duplicates

```bash
pa duplicates              # identical files, and what they waste
pa duplicates --near       # also visually similar, via perceptual hash
```

Because a photo is identified by content hash, the same image on three drives is
already one entry with three locations. `--near` catches the other case: resized
exports, re-compressed copies and lightly edited versions, compared with a
64-bit pHash within a Hamming distance you choose.

## Managing folders

```bash
pa root add /mnt/d/Photos --label "Backup HDD" -x "*/Screenshots/*"
pa root list                  # what is indexed, and which drives are attached
pa root remove 2              # stop indexing a folder (nothing on disk changes)
pa prune                      # drop photos with no remaining file anywhere
pa status                     # counts and pipeline progress
```

Removing a root deletes only its file records. A photo that also exists in
another indexed folder keeps its other locations; one that does not is dropped.

## Re-indexing

Every stage is idempotent and keyed on `(photo hash, stage, model version)`, so
re-running skips finished work and an interrupted run resumes. Kill it mid-run
and start again — nothing is lost or duplicated.

```bash
pa scan                       # find new, changed and deleted files
pa scan --root 2              # one registered folder
pa scan --force               # re-hash everything
pa process                    # all stages
pa process --stage caption    # one stage, e.g. after a model change
pa process --limit 500        # bounded batch
```

Stages run in order `thumbs → embed → faces → caption`. The first three are fast
(~0.1s/photo together) and make a folder searchable almost immediately; captions
are the slow one and can run whenever.

## People

```bash
pa people cluster             # group faces, and match new ones to known people
pa people list                # named people, and clusters awaiting a name
pa people name 3 "Sarah"      # name a cluster from the terminal
```

Naming is much faster in the web UI, where you can see the faces.

## How it is put together

```
pa/
  db/          schema.sql, migrations/, connection, repo   SQLite is the truth
  ingest/      volumes, scanner, hashing, exif, thumbs
  providers/   base (ABCs), lmstudio, siglip, insightface, _cuda, registry
  jobs/        worker            resumable per-stage queue
  faces/       cluster           anchor to known people, then propose new ones
  search/      parser, query, vectors
  sidecar/     xmp               export/import beside the originals
  api/         FastAPI app
web/           the UI - no build step
scripts/       bench_vlm.py      measure a captioner before committing to it
```

**Photos are identified by content hash, not path.** Moving or renaming a file
changes nothing; copying it to another drive adds a location rather than a photo.

**Folders are registered against a drive UUID, not a mount point.** An external
drive that comes back as `G:` instead of `F:` still resolves, because identity
comes from the disk itself:

| Drive type | Identified by |
|---|---|
| Windows / NTFS (via WSL) | NTFS volume serial, read with `vol X:` |
| Linux filesystems | filesystem UUID from `/dev/disk/by-uuid` |
| Anything else | a small marker file written at the mount root |

Nothing in the database stores a drive letter or an absolute path — a `file` row
holds only a path relative to its root, and the mount point is re-derived on
every scan. A *different* disk appearing at the old letter is correctly treated
as a different disk, not as the original, so it can never overwrite that library
or mark its photos missing. See `tests/test_portability.py`.

**`library.db` is the only source of truth.** The vector index under `vectors/`
and the thumbnails under the cache directory are derived and can be deleted and
rebuilt. Back up `library.db` and you have backed up the library.

**Schema changes go in `pa/db/migrations/`** as numbered `.sql` files, applied in
order at startup inside their own transaction. `schema.sql` is the shape of a
*new* database; migrations carry an existing one forward.

## Extending

Four provider kinds, defined as small ABCs in `pa/providers/base.py`:
`caption`, `image_embed`, `text_embed`, `face`. Nothing in the pipeline imports a
concrete provider — it asks the registry for the name in your config.

Adding one needs no changes to this project. Write a class, decorate it, and
point config at the file:

```python
# ~/my_captioner.py
from pa.providers.base import Annotation, CaptionProvider
from pa.providers.registry import register

@register("caption", "my_captioner")
class MyCaptioner(CaptionProvider):
    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def model_version(self):
        return "mine@1"

    def annotate(self, image_bytes: bytes) -> Annotation:
        return Annotation(caption="...", tags=["..."])
```

```toml
[caption]
provider = "my_captioner"
plugins  = ["~/my_captioner.py"]
```

Installed packages are discovered automatically with no config at all, via the
`iris.providers` entry-point group. Registering a name that already exists
replaces it, so a built-in can be overridden without editing this project.
Providers that do not implement their interface are rejected at registration
rather than failing deep inside the indexer. See `tests/test_plugins.py`.

## Tests

```bash
uv run pytest -q
uv run ruff check pa/ scripts/ tests/
```

Before trusting a new captioning model, measure it on your own photos — quality
and speed both vary far more than benchmarks suggest:

```bash
uv run scripts/bench_vlm.py --model google/gemma-4-12b-qat ~/Pictures/*.jpg
```

## Known limits

- Near-duplicate scanning is brute-force pairwise. Vectorised, but O(n²); expect
  it to slow noticeably past ~100k photos.
- The Places map loads Leaflet from a CDN, so it is the one screen that needs an
  internet connection. It says so rather than showing a broken pane.
- The vector index is a memory-mapped brute-force scan: ~150ms at 500k photos.
  Past roughly 1M it should be swapped for a real ANN index, behind the same
  three methods in `pa/search/vectors.py`.
- There is no authentication. The server binds to `127.0.0.1`, and the folder
  browser would expose your directory structure to anything that can reach the
  port, so do not bind it to `0.0.0.0` on a shared network.
