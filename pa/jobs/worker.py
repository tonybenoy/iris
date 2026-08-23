"""Stage workers. Each pulls its own jobs from the queue and is safe to kill."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from pa.db import repo
from pa.db.connection import transaction
from pa.ingest import scanner, thumbs


class StageResult:
    def __init__(self) -> None:
        self.done = 0
        self.failed = 0
        self.skipped = 0
        self.errors: list[str] = []


def _photo_path(conn: sqlite3.Connection, photo_id: int) -> Path | None:
    return scanner.resolve_file_path(conn, photo_id)


def run_thumbs(conn: sqlite3.Connection, cfg, limit: int = 500,
               on_progress=None) -> StageResult:
    res = StageResult()
    while True:
        jobs = repo.claim_jobs(conn, "thumbs", min(50, limit - res.done - res.failed))
        if not jobs:
            break
        for job in jobs:
            path = _photo_path(conn, job["photo_id"])
            if path is None:
                # Drive unplugged: put it back rather than burning an attempt.
                repo.finish_job(conn, job["id"], "pending", "source offline")
                res.skipped += 1
                continue
            digest = conn.execute("SELECT blake3 FROM photo WHERE id=?",
                                  (job["photo_id"],)).fetchone()["blake3"]
            try:
                thumbs.generate(path, digest, cfg.paths.thumbs_dir, cfg.thumbs)
                repo.finish_job(conn, job["id"], "done")
                res.done += 1
            except Exception as exc:
                repo.finish_job(conn, job["id"], "failed", str(exc)[:500])
                res.failed += 1
                res.errors.append(f"{path.name}: {exc}")
            if on_progress:
                on_progress(res)
        if res.done + res.failed >= limit:
            break
    return res


def run_caption(conn: sqlite3.Connection, cfg, provider, limit: int = 500,
                on_progress=None) -> StageResult:
    res = StageResult()
    version = provider.model_version
    while res.done + res.failed < limit:
        jobs = repo.claim_jobs(conn, "caption", 1)  # one at a time: LM Studio serialises anyway
        if not jobs:
            break
        job = jobs[0]
        photo_id = job["photo_id"]
        digest_row = conn.execute("SELECT blake3 FROM photo WHERE id=?", (photo_id,)).fetchone()

        # Prefer the cached view thumbnail: it is already downscaled, sits on
        # fast local disk, and means captioning does not need the source drive.
        source = thumbs.thumb_path(cfg.paths.thumbs_dir, digest_row["blake3"], "view",
                                   cfg.thumbs.format.lower())
        if not source.exists():
            source = _photo_path(conn, photo_id)
        if source is None:
            repo.finish_job(conn, job["id"], "pending", "source offline")
            res.skipped += 1
            continue

        try:
            annotation = provider.annotate(source.read_bytes())
            with transaction(conn):
                repo.save_annotation(conn, photo_id, annotation.as_dict(),
                                     cfg.caption.model, version)
                repo.set_tags(conn, photo_id, annotation.tags, source="ai")
                repo.reindex_fts(conn, photo_id)
                repo.finish_job(conn, job["id"], "done", model_version=version)
            res.done += 1
        except Exception as exc:
            repo.finish_job(conn, job["id"], "failed", str(exc)[:500])
            res.failed += 1
            res.errors.append(f"photo {photo_id}: {str(exc)[:160]}")
        if on_progress:
            on_progress(res)
    return res


def _stage_source(conn: sqlite3.Connection, cfg, photo_id: int) -> Path | None:
    """Bytes to feed a model: the cached view thumbnail when present, else the
    original. Using the thumbnail keeps GPU stages running when the source drive
    is unplugged, and avoids decoding a 45MP RAW to make a 384px embedding."""
    digest = conn.execute("SELECT blake3 FROM photo WHERE id=?", (photo_id,)).fetchone()
    if digest is None:
        return None
    cached = thumbs.thumb_path(cfg.paths.thumbs_dir, digest["blake3"], "view",
                              cfg.thumbs.format.lower())
    return cached if cached.exists() else _photo_path(conn, photo_id)


def run_embed(conn: sqlite3.Connection, cfg, embedder, limit: int = 2000,
              on_progress=None) -> StageResult:
    """Batched: the GPU is far better used on 16 images at once than on one."""
    res = StageResult()
    version = embedder.model_version
    batch_size = cfg.embed.batch_size

    while res.done + res.failed < limit:
        jobs = repo.claim_jobs(conn, "embed", min(batch_size, limit - res.done - res.failed))
        if not jobs:
            break
        payload, keep = [], []
        for job in jobs:
            src = _stage_source(conn, cfg, job["photo_id"])
            if src is None:
                repo.finish_job(conn, job["id"], "pending", "source offline")
                res.skipped += 1
                continue
            try:
                payload.append(src.read_bytes())
                keep.append(job)
            except OSError as exc:
                repo.finish_job(conn, job["id"], "failed", str(exc)[:500])
                res.failed += 1
        if not payload:
            continue

        try:
            vectors = embedder.embed_images(payload)
        except Exception as exc:
            # A whole batch failing is usually OOM or a bad model, not bad data:
            # return the jobs to the queue rather than burning all their attempts.
            for job in keep:
                repo.finish_job(conn, job["id"], "pending", str(exc)[:500])
            res.errors.append(f"batch of {len(keep)}: {str(exc)[:160]}")
            break

        with transaction(conn):
            for job, vec in zip(keep, vectors, strict=True):
                conn.execute(
                    """INSERT INTO photo_embedding (photo_id, embedding, model, dim, created_at)
                       VALUES (?,?,?,?,?)
                       ON CONFLICT(photo_id) DO UPDATE SET
                         embedding=excluded.embedding, model=excluded.model,
                         dim=excluded.dim, created_at=excluded.created_at""",
                    (job["photo_id"], vec.astype(np.float32).tobytes(), version,
                     int(vec.shape[0]), repo.now()))
                repo.finish_job(conn, job["id"], "done", model_version=version)
                res.done += 1
        if on_progress:
            on_progress(res)
    return res


def run_faces(conn: sqlite3.Connection, cfg, analyzer, limit: int = 2000,
              on_progress=None) -> StageResult:
    res = StageResult()
    version = analyzer.model_version

    while res.done + res.failed < limit:
        jobs = repo.claim_jobs(conn, "faces", min(32, limit - res.done - res.failed))
        if not jobs:
            break
        for job in jobs:
            photo_id = job["photo_id"]
            src = _stage_source(conn, cfg, photo_id)
            if src is None:
                repo.finish_job(conn, job["id"], "pending", "source offline")
                res.skipped += 1
                continue
            try:
                faces = analyzer.detect(src.read_bytes())
                with transaction(conn):
                    # Drop only this model's unconfirmed detections; a face the
                    # user has named survives a re-run of the stage.
                    conn.execute(
                        "DELETE FROM face WHERE photo_id=? AND confirmed=0 AND model=?",
                        (photo_id, version))
                    for face in faces:
                        x, y, w, h = face.bbox
                        conn.execute(
                            """INSERT INTO face (photo_id, bbox_x, bbox_y, bbox_w, bbox_h,
                                                 det_score, embedding, model,
                                                 src_w, src_h, created_at)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                            (photo_id, x, y, w, h, face.det_score,
                             face.embedding.astype(np.float32).tobytes(), version,
                             face.src_size[0], face.src_size[1], repo.now()))
                    repo.finish_job(conn, job["id"], "done", model_version=version)
                res.done += 1
            except Exception as exc:
                repo.finish_job(conn, job["id"], "failed", str(exc)[:500])
                res.failed += 1
                res.errors.append(f"photo {photo_id}: {str(exc)[:160]}")
            if on_progress:
                on_progress(res)
    return res
