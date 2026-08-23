"""Regression: every write endpoint must accept a JSON body.

app.py uses `from __future__ import annotations`, so annotations are strings that
FastAPI resolves against module globals. Pydantic models declared inside
create_app() are invisible there, and FastAPI silently reclassifies them as
query parameters -- so tag saving, cluster naming and merging all returned 422
while looking correct in the source.
"""
import pytest
from fastapi.testclient import TestClient

from pa.api.app import create_app
from pa.config import Config


@pytest.fixture
def client(tmp_path, monkeypatch):
    import pa.config as cfgmod

    cfg = Config()
    cfg.paths.data_dir = tmp_path
    cfg.paths.cache_dir = tmp_path / "cache"
    for d in (cfg.paths.data_dir, cfg.paths.cache_dir, cfg.paths.thumbs_dir):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfgmod, "_cfg", cfg)

    from pa.db.connection import init_db
    conn = init_db(cfg.paths.db_path)
    conn.execute("INSERT INTO photo (id, blake3, created_at) VALUES (1,'h',0)")
    conn.execute("INSERT INTO person (id, name, created_at) VALUES (1,'Tony',0)")
    conn.commit()
    return TestClient(create_app())


def test_put_tags_accepts_body(client):
    r = client.put("/api/photos/1/tags", json={"tags": ["holiday", "kerala"]})
    assert r.status_code == 200, r.text
    got = client.get("/api/photos/1").json()
    assert {t["name"] for t in got["tags"]} == {"holiday", "kerala"}


def test_rename_person_accepts_body(client):
    r = client.patch("/api/people/1", json={"name": "Tony Benoy"})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "Tony Benoy"


def test_merge_requires_two_clusters(client):
    r = client.post("/api/clusters/merge", json={"cluster_ids": [1]})
    assert r.status_code == 400


def test_empty_name_rejected(client):
    assert client.patch("/api/people/1", json={"name": "  "}).status_code == 400


def test_stats_and_duplicates_serve(client):
    assert client.get("/api/stats").status_code == 200
    assert client.get("/api/duplicates").status_code == 200
    assert client.get("/api/map").json() == {"points": []}
