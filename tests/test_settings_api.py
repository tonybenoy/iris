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
