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


def _claim_batches(conn: sqlite3.Connection, stage: str, batch: int, limit: int,
                   res: StageResult):
    """Yield batches of claimed jobs until the queue is empty or the limit is hit.

    A job whose source is offline is put back as 'pending' rather than failed,
    so that an unplugged drive costs nothing and the work resumes when it comes
    back. That makes it immediately claimable again -- and since `skipped`
    advances neither `done` nor `failed`, a stage with nothing but offline
    photos left would claim the same job, put it back, and claim it again
    forever, pinning a core and counting attempts into the millions.

    Remembering what this run has already been handed ends it: once a batch
    contains nothing new, there is no work left that this run can do.
    """
    seen: set[int] = set()
    while res.done + res.failed < limit:
        jobs = repo.claim_jobs(conn, stage, min(batch, limit - res.done - res.failed))
        if not jobs:
            return
        fresh = [job for job in jobs if job["id"] not in seen]
        # Claiming marks a job 'running'. Anything handed back a second time is
        # not going to be processed, so put it back now -- left claimed, it
        # would sit there looking like work in progress until a restart.
        repeats = [job["id"] for job in jobs if job["id"] in seen]
        if repeats:
            repo.unclaim_jobs(conn, repeats)
        if not fresh:
            return
        seen.update(job["id"] for job in fresh)
        yield fresh


def run_thumbs(conn: sqlite3.Connection, cfg, limit: int = 500,
               on_progress=None) -> StageResult:
    res = StageResult()
    for jobs in _claim_batches(conn, "thumbs", 50, limit, res):
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
    return res


def run_caption(conn: sqlite3.Connection, cfg, provider, limit: int = 500,
                on_progress=None) -> StageResult:
    res = StageResult()
    version = provider.model_version
    # One at a time: LM Studio serialises anyway.
    for jobs in _claim_batches(conn, "caption", 1, limit, res):
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


def _photo_is_the_problem(exc: BaseException) -> bool:
    """Whether a failure belongs to this photo or to the machine.

    PIL raises OSError for a file that stops halfway and UnidentifiedImageError
    for something that is not an image at all. Those are permanent facts about
    that photo. A CUDA out-of-memory or a missing model is the opposite: the
    photo is innocent and the same work will succeed later.
    """
    from PIL import UnidentifiedImageError
    return isinstance(exc, (UnidentifiedImageError, OSError, ValueError))


def _embed_one_at_a_time(conn: sqlite3.Connection, embedder, jobs: list, payload: list,
                         res: StageResult):
    """Re-run a failed batch photo by photo, to find out whose fault it was.

    Returning a whole batch to the queue is right when the GPU is out of memory
    and wrong when one file is truncated -- and the two are indistinguishable
    from the outside. Being wrong about it is not a lost photo but a stalled
    library: the next run claims the same sixteen, fails on the same one, and
    puts them all back, forever. One corrupt file left 1171 photos unindexed
    and every run reported "0 done" in under a second.

    So the batch is retried singly. A photo that fails on its own is failed on
    its own; anything that is not about the photo puts the untried remainder
    back and ends the run, which is what the old code did for every case.

    Returns None if the failure was systemic, else (vectors, jobs) for the
    photos that did work -- possibly none of them.
    """
    ok_jobs, vectors = [], []
    for job, blob in zip(jobs, payload, strict=True):
        try:
            vectors.append(embedder.embed_images([blob])[0])
            ok_jobs.append(job)
        except Exception as exc:
            if not _photo_is_the_problem(exc):
                done = {j["id"] for j in ok_jobs}
                for untried in jobs:
                    if untried["id"] not in done:
                        repo.finish_job(conn, untried["id"], "pending", str(exc)[:500])
                res.errors.append(f"batch of {len(jobs)}: {str(exc)[:160]}")
                return None
            repo.finish_job(conn, job["id"], "failed", str(exc)[:500])
            res.failed += 1
            res.errors.append(f"photo {job['photo_id']}: {str(exc)[:160]}")
    return (np.stack(vectors) if ok_jobs else None), ok_jobs


def run_embed(conn: sqlite3.Connection, cfg, embedder, limit: int = 2000,
              on_progress=None) -> StageResult:
    """Batched: the GPU is far better used on 16 images at once than on one."""
    res = StageResult()
    version = embedder.model_version
    batch_size = cfg.embed.batch_size

    for jobs in _claim_batches(conn, "embed", batch_size, limit, res):
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
        except Exception:
            # Could be the GPU, could be one unreadable file among sixteen.
            # Retrying singly is how to tell, and it costs a pass only over a
            # batch that has already failed.
            outcome = _embed_one_at_a_time(conn, embedder, keep, payload, res)
            if outcome is None:
                break                    # systemic: the jobs are back in the queue
            vectors, keep = outcome
            if not keep:
                if on_progress:
                    on_progress(res)
                continue                 # every photo in this batch was broken

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

    for jobs in _claim_batches(conn, "faces", 32, limit, res):
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
                    # Drop only this model's undecided detections. A face the
                    # user has named survives a re-run of the stage, and so does
                    # one they ignored -- otherwise re-running faces would
                    # resurrect every stranger they had already dismissed.
                    conn.execute(
                        "DELETE FROM face WHERE photo_id=? AND confirmed=0 "
                        "AND rejected=0 AND model=?",
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
