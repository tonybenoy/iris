"""Runs pipeline stages inside the web server, on a background thread.

`pa process` drives the same stage functions from the CLI. The difference is
who is watching: the CLI can block a terminal and print a spinner, while the
server has to start the work, return immediately, and let the browser poll.

Everything here is about that gap -- one run at a time, progress readable from
another thread, and a cancel that never loses committed work.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Any

from pa.db import repo
from pa.db.connection import init_db
from pa.jobs import worker

# The four queue-backed stages, plus clustering. Clustering is not a per-photo
# job -- it reads every face at once and regroups them -- but it belongs at the
# end of a run, because faces that were just detected mean nothing until they
# are grouped into people.
STAGES: tuple[str, ...] = (*repo.STAGES, "cluster")
DEFAULT_STAGES: tuple[str, ...] = STAGES


class Cancelled(Exception):
    """Raised inside a progress callback to unwind a running stage.

    The stage functions call `on_progress` after committing each unit of work
    and outside their own try/except, so raising there stops the run at a
    boundary where the database is consistent and nothing is half-written.
    """


class PipelineRunner:
    """One run at a time, with progress another thread can read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._providers: dict[str, Any] = {}
        self._state: dict[str, Any] = {
            "running": False, "stage": None, "stages": [], "queued": [],
            "progress": {"done": 0, "failed": 0, "skipped": 0},
            "results": {}, "started_at": None, "finished_at": None,
            "error": None, "cancelled": False, "trigger": None,
            # What the current stage is doing. Loading a model is minutes of
            # nothing on the first run, and a progress bar at zero with no
            # explanation is indistinguishable from a hang.
            "phase": None, "note": None,
        }

    # ---------------------------------------------------------------- public
    @property
    def running(self) -> bool:
        return self._state["running"]

    def status(self) -> dict[str, Any]:
        # A shallow copy is enough for a status read: the worker thread replaces
        # these sub-dicts wholesale rather than mutating them in place.
        return dict(self._state)

    def start(self, cfg, stages: list[str] | None = None, limit: int = 100_000,
              trigger: str = "manual") -> dict[str, Any]:
        """Begin a run. Raises RuntimeError if one is already going."""
        wanted = list(stages or DEFAULT_STAGES)
        unknown = [s for s in wanted if s not in STAGES]
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; expected any of {list(STAGES)}")
        if not wanted:
            raise ValueError("no stages given")

        with self._lock:
            if self._state["running"]:
                raise RuntimeError("indexing is already running")
            self._cancel.clear()
            self._state = {
                "running": True, "stage": None, "stages": wanted, "queued": wanted,
                "progress": {"done": 0, "failed": 0, "skipped": 0},
                "results": {}, "started_at": time.time(), "finished_at": None,
                "error": None, "cancelled": False, "trigger": trigger,
                "phase": "starting", "note": None,
            }
            self._thread = threading.Thread(
                target=self._run, args=(cfg, wanted, limit), daemon=True)
            self._thread.start()
        return self.status()

    def cancel(self) -> bool:
        """Ask the run to stop at the next safe point. Returns False if idle."""
        if not self._state["running"]:
            return False
        self._cancel.set()
        return True

    def forget_providers(self) -> None:
        """Drop cached models so the next run picks up new settings.

        Called after a config save. The models are loaded lazily on the next
        run, which is also when the GPU memory the old ones held is released.
        """
        self._providers.clear()

    # --------------------------------------------------------------- internals
    def provider(self, kind: str, cfg):
        """The one loaded instance of a model, keyed on the settings that define it.

        Loading SigLIP is seconds of weight shuffling and over a gigabyte of
        VRAM. Doing that per button press would make the UI feel broken, so a
        run reuses what the last one loaded -- unless the settings that decide
        *which* model it is have changed underneath it.

        Search shares this cache too, rather than keeping its own embedder. They
        are the same weights for the same purpose, and holding two copies meant
        a second "Loading weights" bar and double the VRAM the moment anyone
        pressed Index after a search had warmed up.
        """
        from pa.providers.registry import (
            get_captioner,
            get_face_analyzer,
            get_image_embedder,
        )
        section = {"embed": cfg.embed, "faces": cfg.face, "caption": cfg.caption}[kind]
        key = (section.provider, section.model, getattr(section, "device", None))
        cached = self._providers.get(kind)
        if cached and cached[0] == key:
            return cached[1]
        # Say so, in the terminal and in the UI. The first load of an 800M
        # parameter model is a minute of complete silence, and silence is
        # indistinguishable from a hang -- the more so because transformers
        # prints its own progress bar and then stops, which reads as the point
        # where it got stuck rather than the point where it finished.
        device = getattr(section, "device", "") or ""
        self._say(f"loading the {kind} model {section.model}"
                  f"{f' on {device}' if device else ''}")
        self._state["phase"] = "loading"
        self._state["note"] = f"Loading {section.model}"
        began = time.monotonic()
        try:
            built = {"embed": get_image_embedder, "faces": get_face_analyzer,
                     "caption": get_captioner}[kind](cfg)
        finally:
            self._state["phase"] = "working"
            self._state["note"] = None
        self._say(f"{kind} model ready in {time.monotonic() - began:.0f}s")
        self._providers[kind] = (key, built)
        return built

    @staticmethod
    def _say(message: str) -> None:
        print(f"[iris] {message}", file=sys.stderr, flush=True)

    def _progress_cb(self, stage: str):
        """Publish a stage's counters, and turn a cancel request into an unwind."""
        def cb(res: worker.StageResult) -> None:
            if self._cancel.is_set():
                raise Cancelled
            self._state["stage"] = stage
            self._state["phase"] = "working"
            self._state["progress"] = {
                "done": res.done, "failed": res.failed, "skipped": res.skipped}
        return cb

    def _record(self, stage: str, res: worker.StageResult) -> None:
        self._state["results"] = self._state["results"] | {stage: {
            "done": res.done, "failed": res.failed, "skipped": res.skipped,
            "errors": res.errors[:5],
        }}

    def _run(self, cfg, stages: list[str], limit: int) -> None:
        conn = None
        try:
            conn = init_db(cfg.paths.db_path)  # this thread needs its own handle
            repo.requeue_stale(conn)
            for stage in stages:
                if self._cancel.is_set():
                    raise Cancelled
                self._state["stage"] = stage
                self._state["queued"] = stages[stages.index(stage) + 1:]
                self._state["progress"] = {"done": 0, "failed": 0, "skipped": 0}
                self._state["phase"] = "working"
                self._say(f"stage {stage}: starting")
                began = time.monotonic()
                self._run_stage(conn, cfg, stage, limit)
                done = self._state["results"].get(stage, {})
                self._say(f"stage {stage}: finished in {time.monotonic() - began:.0f}s "
                          f"({done.get('done', 0)} done, {done.get('failed', 0)} failed, "
                          f"{done.get('skipped', 0)} waiting on a drive)")
        except Cancelled:
            self._state["cancelled"] = True
            if conn is not None:
                # Release whatever was claimed but never started, so the work is
                # not stuck in 'running' until the next process restart.
                repo.requeue_stale(conn)
                conn.commit()
        except Exception as exc:
            self._state["error"] = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            if conn is not None:
                conn.close()
            self._state["stage"] = None
            self._state["queued"] = []
            self._state["phase"] = None
            self._state["note"] = None
            self._state["finished_at"] = time.time()
            self._state["running"] = False

    def _run_stage(self, conn, cfg, stage: str, limit: int) -> None:
        cb = self._progress_cb(stage)

        if stage == "thumbs":
            self._record(stage, worker.run_thumbs(conn, cfg, limit, cb))
            return

        if stage == "embed":
            res = worker.run_embed(conn, cfg, self.provider("embed", cfg), limit, cb)
            self._record(stage, res)
            if res.done:
                # Search reads a prebuilt index file, not the embedding rows, so
                # without this the photos just embedded stay unfindable.
                from pa.search.vectors import VectorIndex
                VectorIndex(cfg.paths.vectors_dir, cfg.embed.dim, cfg.embed.model).build(conn)
            return

        if stage == "faces":
            self._record(stage, worker.run_faces(conn, cfg, self.provider("faces", cfg),
                                                 limit, cb))
            return

        if stage == "caption":
            provider = self.provider("caption", cfg)
            ok, msg = provider.health()
            if not ok:
                # A dead model server is a setup problem, not a per-photo failure.
                # Burning every job's attempts against it would make the queue
                # unrecoverable once LM Studio came back.
                raise RuntimeError(f"caption provider unavailable: {msg}")
            self._record(stage, worker.run_caption(conn, cfg, provider, limit, cb))
            return

        if stage == "cluster":
            from pa.faces.cluster import recluster
            stats = recluster(conn, cfg)
            self._state["results"] = self._state["results"] | {"cluster": {
                "anchored": stats.anchored, "new_clusters": stats.new_clusters,
                "clustered_faces": stats.clustered_faces,
                "unassigned": stats.unassigned, "errors": [],
            }}
            return


runner = PipelineRunner()
