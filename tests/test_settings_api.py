"""The web UI has to be able to do everything the CLI can.

These cover the two things that were previously terminal-only: editing settings,
and making the pipeline actually run. Both are the difference between a scanned
folder and a usable one, so a silent regression here looks exactly like the
"it scans but nothing happens" bug that prompted them.
"""
import tomllib

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pa.api.app import create_app
from pa.config import Config, render_default_toml, save_config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    import pa.config as cfgmod

    cfg = Config()
    cfg.paths.data_dir = tmp_path / "data"
    cfg.paths.cache_dir = tmp_path / "cache"
    for d in (cfg.paths.data_dir, cfg.paths.cache_dir, cfg.paths.thumbs_dir):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfgmod, "_cfg", cfg)
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", tmp_path / "config.toml")
    return cfg


@pytest.fixture
def client(cfg):
    return TestClient(create_app())


# --------------------------------------------------------------------- config
def test_template_covers_every_setting():
    """A field the template forgets is a field the UI silently deletes on Save.

    render_default_toml() is what writes the file back, so anything it does not
    emit is gone the first time anyone presses Save. Round-tripping a default
    Config through it is the cheapest way to keep that honest as fields are added.
    """
    original = Config()
    data = tomllib.loads(render_default_toml(original))
    server = data.pop("server")
    assert Config.model_validate({**data, **server}).model_dump() == original.model_dump()


def test_get_config_reports_current_values(client):
    d = client.get("/api/config").json()
    assert d["config"]["embed"]["min_score"] == Config().embed.min_score
    assert d["config"]["server"]["auto_process"] == ["thumbs"]
    assert "caption" in d["providers"] and d["providers"]["caption"]
    assert "server.port" in d["restart_required"]
    assert d["reindex_required"]["embed.model"] == "embed"


def test_save_config_writes_and_applies_live(client, cfg):
    r = client.put("/api/config", json={"embed": {"min_score": 0.2}})
    assert r.status_code == 200, r.text
    assert r.json()["config"]["embed"]["min_score"] == 0.2
    # Applied to the object every request handler already holds, not just to disk.
    assert cfg.embed.min_score == 0.2


def test_save_config_is_partial(client, cfg):
    client.put("/api/config", json={"thumbs": {"grid_px": 256}})
    client.put("/api/config", json={"face": {"cluster_eps": 0.5}})
    d = client.get("/api/config").json()["config"]
    assert d["thumbs"]["grid_px"] == 256   # not clobbered by the second save
    assert d["face"]["cluster_eps"] == 0.5


def test_save_config_rejects_bad_values_without_writing(client, cfg):
    before = cfg.thumbs.grid_px
    r = client.put("/api/config", json={"thumbs": {"grid_px": "enormous"}})
    assert r.status_code == 400
    assert "grid_px" in r.json()["detail"]
    assert cfg.thumbs.grid_px == before


def test_save_config_rejects_unknown_stage(client):
    r = client.put("/api/config", json={"server": {"auto_process": ["telepathy"]}})
    assert r.status_code == 400
    assert "telepathy" in r.json()["detail"]


def test_unknown_settings_survive_a_save(cfg, tmp_path):
    """A key this version of the UI has never heard of must not be deleted."""
    import json

    import pa.config as cfgmod
    elsewhere = tmp_path / "elsewhere"
    cfgmod.CONFIG_PATH.write_text(f"[paths]\ndata_dir = {json.dumps(str(elsewhere))}\n")
    save_config({"thumbs": {"quality": 70}})
    written = tomllib.loads(cfgmod.CONFIG_PATH.read_text())
    assert written["paths"]["data_dir"] == str(elsewhere)
    assert written["thumbs"]["quality"] == 70


def test_plugins_round_trip_through_a_save(cfg, monkeypatch):
    """Windows plugin paths contain backslashes, which a naive TOML write breaks."""
    import pa.config as cfgmod
    save_config({"caption": {"plugins": [r"C:\Users\tony\my_captioner.py"]}})
    written = tomllib.loads(cfgmod.CONFIG_PATH.read_text())
    assert written["caption"]["plugins"] == [r"C:\Users\tony\my_captioner.py"]


# ------------------------------------------------------------------- pipeline
def test_process_status_lists_every_stage(client):
    d = client.get("/api/process").json()
    assert d["running"] is False
    assert d["stages_all"] == ["thumbs", "embed", "faces", "caption", "cluster"]


def test_thumbs_stage_runs_from_the_api(client, cfg, tmp_path, monkeypatch):
    """The whole point: a queued photo becomes a servable thumbnail with no CLI.

    This is the exact path that was missing -- a scan enqueued the job and then
    nothing in the server could ever consume it.
    """
    from pa.db import repo
    from pa.db.connection import init_db
    from pa.ingest import hashing, volumes

    photos = tmp_path / "photos"
    photos.mkdir()
    src = photos / "red.jpg"
    Image.new("RGB", (600, 400), (200, 40, 40)).save(src)
    # Paths are re-derived from the drive's identity on every read, so the test
    # drive just has to answer "where am I mounted right now".
    monkeypatch.setattr(volumes, "current_mountpoint",
                        lambda uuid: tmp_path if uuid == "test-vol" else None)

    conn = init_db(cfg.paths.db_path)
    digest = hashing.content_hash(src)
    volume_id = repo.upsert_volume(conn, "test-vol", "test", str(tmp_path))
    root_id = repo.add_root(conn, volume_id, "photos", "test", [])
    photo_id = conn.execute(
        "INSERT INTO photo (blake3, created_at) VALUES (?,0)", (digest,)).lastrowid
    conn.execute("INSERT INTO file (photo_id, root_id, rel_path, filename, mtime, size, "
                 "state, seen_at) VALUES (?,?,'red.jpg','red.jpg',0,?,'present',0)",
                 (photo_id, root_id, src.stat().st_size))
    repo.enqueue(conn, photo_id, stages=("thumbs",))
    conn.commit()

    assert client.post("/api/process", json={"stages": ["thumbs"]}).status_code == 200
    _wait_for_idle(client)

    status = client.get("/api/process").json()
    assert status["results"]["thumbs"]["done"] == 1, status
    assert client.get(f"/api/thumb/{digest}/grid").status_code == 200


def test_second_run_is_refused_while_one_is_going(client, cfg, monkeypatch):
    import pa.jobs.runner as runnermod

    started = __import__("threading").Event()
    release = __import__("threading").Event()

    def slow(conn, cfg, limit=500, on_progress=None):
        started.set()
        release.wait(5)
        return runnermod.worker.StageResult()

    monkeypatch.setattr(runnermod.worker, "run_thumbs", slow)
    assert client.post("/api/process", json={"stages": ["thumbs"]}).status_code == 200
    started.wait(5)
    try:
        r = client.post("/api/process", json={"stages": ["thumbs"]})
        assert r.status_code == 409
    finally:
        release.set()
    _wait_for_idle(client)


def test_unknown_stage_is_rejected(client):
    r = client.post("/api/process", json={"stages": ["astrology"]})
    assert r.status_code == 400


def test_retry_failed_requeues(client, cfg):
    from pa.db.connection import init_db
    conn = init_db(cfg.paths.db_path)
    conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (1,'h',0)")
    conn.execute("INSERT INTO job (photo_id, stage, state, priority, created_at) "
                 "VALUES (1,'thumbs','failed',100,0)")
    conn.commit()
    assert client.post("/api/process/retry-failed").json()["requeued"] == 1
    assert client.get("/api/process").json()["queue"]["thumbs"]["pending"] == 1


def test_health_check_reports_structured_results(client):
    d = client.get("/api/config/check").json()
    ids = {c["id"] for c in d["checks"]}
    assert ids == {"caption", "gpu", "faces"}
    assert all(c["level"] in ("ok", "warn", "fail") for c in d["checks"])


def _wait_for_idle(client, tries=100):
    import time
    for _ in range(tries):
        if not client.get("/api/process").json()["running"]:
            return
        time.sleep(0.05)
    raise AssertionError("pipeline never went idle")


# ------------------------------------------------------- redoing the library
# Stages are keyed on what they have already done, which is what makes indexing
# resumable -- and what makes changing a model change nothing at all, because
# every photo is already marked done under the old one.


@pytest.fixture
def library(cfg):
    """Three photos; only the first has ever been through a stage."""
    from pa.db.connection import init_db
    conn = init_db(cfg.paths.db_path)
    for photo_id in (1, 2, 3):
        conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (?,?,0)",
                     (photo_id, f"hash{photo_id}"))
    conn.execute("INSERT INTO job (photo_id, stage, state, attempts, priority, created_at) "
                 "VALUES (1,'thumbs','done',2,100,0)")
    conn.commit()
    return conn


def _redo(client, stage, **kw):
    return client.post("/api/process/reset",
                       json={"stage": stage, "rebuild": True, **kw}).json()


def test_redo_covers_photos_that_never_had_a_job_row(client, library):
    """A stage that was switched off when a photo was scanned leaves no job row
    to reset, and those are exactly the photos someone switching it on wants."""
    r = _redo(client, "thumbs")
    assert r["requeued"] == 3
    assert client.get("/api/process").json()["queue"]["thumbs"]["pending"] == 3
    assert library.execute(
        "SELECT attempts FROM job WHERE photo_id=1").fetchone()[0] == 0


def test_redo_thumbs_throws_the_cache_away(client, cfg, library):
    """generate() skips any thumbnail whose file is already there, so without
    this a redo reports 'done' for every photo and changes nothing."""
    from pa.ingest import thumbs
    path = thumbs.thumb_path(cfg.paths.thumbs_dir, "hash1", "grid", "webp")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10)).save(path)

    assert _redo(client, "thumbs")["discarded"] == {"thumbnails": 1}
    assert not path.exists()


def test_redo_faces_keeps_the_people_you_named(client, library):
    """Detections are output and go. Names and dismissals are decisions and stay
    -- a redo that lost them would cost hours of naming to change a model."""
    import numpy as np
    vec = np.zeros(512, dtype=np.float32).tobytes()
    for face_id, confirmed, rejected in ((1, 1, 0), (2, 0, 1), (3, 0, 0)):
        library.execute(
            """INSERT INTO face (id, photo_id, bbox_x, bbox_y, bbox_w, bbox_h,
                                 det_score, embedding, model, confirmed, rejected, created_at)
               VALUES (?,1,0,0,10,10,0.9,?,'old-model',?,?,0)""",
            (face_id, vec, confirmed, rejected))
    library.commit()

    assert _redo(client, "faces")["discarded"] == {"faces": 1}
    assert {r[0] for r in library.execute("SELECT id FROM face")} == {1, 2}


def test_redo_embed_forgets_the_vectors_and_the_index(client, cfg, library):
    """Vectors from another model are a different space; keeping them would mix
    two coordinate systems in one index."""
    library.execute(
        "INSERT INTO photo_embedding (photo_id, embedding, model, dim, created_at) "
        "VALUES (1,?, 'old-model', 4, 0)", (b"\x00" * 16,))
    library.commit()
    cfg.paths.vectors_dir.mkdir(parents=True, exist_ok=True)
    stale = cfg.paths.vectors_dir / "meta.json"
    stale.write_text('{"model": "old-model", "dim": 4, "count": 1}')

    assert _redo(client, "embed")["discarded"] == {"vectors": 1}
    assert library.execute("SELECT COUNT(*) FROM photo_embedding").fetchone()[0] == 0
    assert not stale.exists()


def test_a_plain_requeue_discards_nothing(client, cfg, library):
    from pa.ingest import thumbs
    path = thumbs.thumb_path(cfg.paths.thumbs_dir, "hash1", "grid", "webp")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10)).save(path)

    r = client.post("/api/process/reset",
                    json={"stage": "thumbs", "rebuild": False}).json()
    assert r["requeued"] == 3 and r["discarded"] == {}
    assert path.exists()


def test_redo_rejects_a_stage_that_is_not_one(client, library):
    r = client.post("/api/process/reset", json={"stage": "cluster", "rebuild": True})
    assert r.status_code == 400, "clustering is not a per-photo queue"
    assert "thumbs" in r.json()["detail"]


def test_a_stage_with_only_offline_photos_still_finishes(cfg):
    """A photo whose drive is unplugged is put back in the queue rather than
    failed, which makes it immediately claimable again. Nothing else advanced
    the loop, so a run with no reachable photos left spun forever -- claim, put
    back, claim -- pinning a core and counting attempts into the thousands.
    """
    import threading

    from pa.db.connection import init_db
    from pa.jobs import worker

    conn = init_db(cfg.paths.db_path)
    conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (1,'gone',0)")
    conn.execute("INSERT INTO job (photo_id, stage, state, priority, created_at) "
                 "VALUES (1,'thumbs','pending',100,0)")
    conn.commit()

    res = {}
    # In a thread with a deadline: the bug's symptom is not finishing, and a
    # test that reproduces it by hanging the suite is no use to anyone.
    run = threading.Thread(
        target=lambda: res.update(out=worker.run_thumbs(conn, cfg, limit=50)),
        daemon=True)
    run.start()
    run.join(10)
    assert not run.is_alive(), "run_thumbs never returned: it is spinning on an offline photo"
    assert res["out"].done == 0 and res["out"].skipped >= 1

    row = conn.execute("SELECT state, attempts FROM job WHERE photo_id=1").fetchone()
    # Back in the queue, not left mid-claim: a job still marked 'running' when
    # nothing is running reads as work in progress on the Library tab, and only
    # a restart clears it.
    assert row["state"] == "pending"
    assert row["attempts"] <= 2


def test_the_status_says_a_model_is_loading(cfg, monkeypatch):
    """Loading an 800M parameter model is a minute in which transformers prints
    a progress bar, finishes it, and then goes quiet. Every report of "it seems
    stuck" has been that minute, so the run has to be able to say what it is
    doing while nothing is countable."""
    import threading

    import pa.providers.registry as registry
    from pa.jobs.runner import PipelineRunner

    loading, release = threading.Event(), threading.Event()

    def slow_load(_cfg):
        loading.set()
        release.wait(5)
        return object()

    monkeypatch.setattr(registry, "get_image_embedder", slow_load)
    runner = PipelineRunner()
    runner._state["running"] = True   # provider() reports into a live run

    got = threading.Thread(target=lambda: runner.provider("embed", cfg), daemon=True)
    got.start()
    assert loading.wait(5)
    state = runner.status()
    assert state["phase"] == "loading"
    assert cfg.embed.model in state["note"]
    release.set()
    got.join(5)
    assert runner.status()["phase"] == "working"


def test_two_callers_do_not_each_load_their_own_model(cfg, monkeypatch):
    """Opening the app runs a search, which loads the embedder in a request
    thread. Pressing Index a moment later found the cache still empty and
    started a second copy: double the VRAM and two CUDA initialisations at
    once, which on a card with no room for the second looks like a hang."""
    import threading

    import pa.providers.registry as registry
    from pa.jobs.runner import PipelineRunner

    loads = []
    started, release = threading.Event(), threading.Event()

    def slow_load(_cfg):
        loads.append(1)
        started.set()
        release.wait(5)
        return object()

    monkeypatch.setattr(registry, "get_image_embedder", slow_load)
    runner = PipelineRunner()

    got = []
    threads = [threading.Thread(target=lambda: got.append(runner.provider("embed", cfg)),
                                daemon=True) for _ in range(2)]
    threads[0].start()
    assert started.wait(5)      # the first is inside the load
    threads[1].start()          # the second arrives while it is still going
    release.set()
    for t in threads:
        t.join(5)

    assert len(loads) == 1, "the model was loaded twice"
    assert got[0] is got[1], "and the two callers got different instances"


def test_detecting_faces_always_regroups_them(cfg, monkeypatch):
    """A face with no group belongs to nobody and cannot be named, so a run that
    detects faces without clustering leaves the People screen empty with no sign
    of why. A full run always ended with cluster; a run of one stage did not,
    which made redoing faces alone look like it had lost every face."""
    from pa.jobs.runner import PipelineRunner

    ran = []
    monkeypatch.setattr(PipelineRunner, "_run_stage",
                        lambda self, conn, cfg, stage, limit: ran.append(stage))
    runner = PipelineRunner()
    state = runner.start(cfg, ["faces"])
    assert state["stages"] == ["faces", "cluster"]

    runner._thread.join(10)
    assert ran == ["faces", "cluster"]


def test_asking_for_clustering_twice_does_not_run_it_twice(cfg, monkeypatch):
    from pa.jobs.runner import PipelineRunner

    ran = []
    monkeypatch.setattr(PipelineRunner, "_run_stage",
                        lambda self, conn, cfg, stage, limit: ran.append(stage))
    runner = PipelineRunner()
    runner.start(cfg, ["faces", "cluster"])
    runner._thread.join(10)
    assert ran == ["faces", "cluster"]


# --------------------------------------------------- one bad file in a batch
@pytest.fixture
def embed_queue(cfg):
    """Four photos waiting to be embedded, with a readable thumbnail each."""
    from pa.db.connection import init_db
    from pa.ingest import thumbs as thumbmod

    conn = init_db(cfg.paths.db_path)
    for photo_id in (1, 2, 3, 4):
        digest = f"hash{photo_id}"
        conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (?,?,0)",
                     (photo_id, digest))
        conn.execute("INSERT INTO job (photo_id, stage, state, priority, created_at) "
                     "VALUES (?,'embed','pending',100,0)", (photo_id,))
        path = thumbmod.thumb_path(cfg.paths.thumbs_dir, digest, "view",
                                   cfg.thumbs.format.lower())
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path, cfg.thumbs.format)
    conn.commit()
    return conn


class _Embedder:
    """Fails on one photo, the way PIL does for a file that stops halfway."""
    model_version = "test"

    def __init__(self, bad_index=None, error=None):
        self.bad, self.error = bad_index, error or OSError(
            "image file is truncated (20 bytes not processed)")
        self.calls = []

    def embed_images(self, images):
        import numpy as np
        self.calls.append(len(images))
        for image in images:
            if self.bad is not None and bytes(image).find(self.bad) >= 0:
                raise self.error
        return np.ones((len(images), 4), dtype=np.float32)


def _mark_bad(cfg, conn, photo_id, marker=b"BROKEN"):
    """Make one photo's source identifiable to the fake embedder."""
    from pa.ingest import thumbs as thumbmod
    digest = conn.execute("SELECT blake3 FROM photo WHERE id=?", (photo_id,)).fetchone()[0]
    path = thumbmod.thumb_path(cfg.paths.thumbs_dir, digest, "view",
                               cfg.thumbs.format.lower())
    path.write_bytes(marker)
    return marker


def test_one_truncated_file_does_not_block_the_whole_queue(cfg, embed_queue):
    """The batch was returned to the queue whole, so the next run claimed the
    same photos, failed on the same one, and put them back again. One corrupt
    file left 1171 photos unindexed, every run reporting "0 done" in a second."""
    from pa.jobs import worker

    marker = _mark_bad(cfg, embed_queue, 3)
    res = worker.run_embed(embed_queue, cfg, _Embedder(bad_index=marker))

    assert res.done == 3, "the readable photos must still be embedded"
    assert res.failed == 1
    states = dict(embed_queue.execute(
        "SELECT photo_id, state FROM job WHERE stage='embed'").fetchall())
    assert states == {1: "done", 2: "done", 3: "failed", 4: "done"}


def test_the_bad_photo_is_named_in_the_errors(cfg, embed_queue):
    from pa.jobs import worker

    marker = _mark_bad(cfg, embed_queue, 2)
    res = worker.run_embed(embed_queue, cfg, _Embedder(bad_index=marker))
    assert any("photo 2" in e and "truncated" in e for e in res.errors), res.errors


def test_a_gpu_failure_still_puts_the_batch_back(cfg, embed_queue):
    """A photo is not at fault when the GPU runs out of memory, and marking four
    thousand of them failed for it would be unrecoverable without a retry pass."""
    from pa.jobs import worker

    class OutOfMemory(_Embedder):
        def embed_images(self, images):
            raise RuntimeError("CUDA out of memory")

    res = worker.run_embed(embed_queue, cfg, OutOfMemory())
    assert res.done == 0 and res.failed == 0
    states = {r[0] for r in embed_queue.execute(
        "SELECT DISTINCT state FROM job WHERE stage='embed'")}
    assert states == {"pending"}, "an out-of-memory batch must stay in the queue"
