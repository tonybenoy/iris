"""FastAPI backend. Serves the SPA and a small JSON API over the library."""
from __future__ import annotations

import contextlib
import sqlite3
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pa.config import get_config
from pa.db import repo
from pa.db.connection import init_db
from pa.db.repo import ANNOTATION_ORDER
from pa.ingest import scanner, thumbs, volumes
from pa.jobs.runner import STAGES as runner_stages
from pa.jobs.runner import runner
from pa.search.query import search as run_search

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


# These MUST live at module scope. This file uses `from __future__ import
# annotations`, so every annotation is a string that FastAPI resolves against
# the module's globals. A model defined inside create_app() is invisible there,
# so FastAPI cannot tell it is a body model and silently demands it as a query
# parameter instead -- every write endpoint then fails with a 422.
class TagsIn(BaseModel):
    tags: list[str]


class NameIn(BaseModel):
    name: str


class MergeIn(BaseModel):
    cluster_ids: list[int]
    name: str | None = None


class AnnotationIn(BaseModel):
    caption: str | None = None
    ocr_text: str | None = None
    scene: str | None = None


class RootIn(BaseModel):
    path: str
    label: str | None = None
    exclude: list[str] = []
    scan: bool = True


class ProcessIn(BaseModel):
    stages: list[str] | None = None   # None = every stage, in pipeline order
    limit: int = 100_000


class SidecarIn(BaseModel):
    root_id: int | None = None
    overwrite: bool = False
    beside: bool | None = None        # None = whatever [sidecar] location says


class PruneIn(BaseModel):
    keep_missing: bool = False


class ResetIn(BaseModel):
    stage: str
    rebuild: bool = False   # also discard what the stage already produced


class ConfigIn(BaseModel):
    """A partial settings patch, shaped like the config file.

    Partial matters: the UI sends only the sections a person actually touched,
    so a setting it has never heard of is not wiped by a save.
    """
    model_config = {"extra": "allow"}


def create_app() -> FastAPI:
    cfg = get_config()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """Load the embedding model in the background at boot.

        It is ~10s of weight loading, and doing it lazily inside the first search
        makes the very first thing a user types feel broken. Warming on a thread
        keeps startup instant and the API responsive while it loads.
        """
        def go() -> None:
            try:
                fn = _vector_search()
                if fn is not None:
                    fn("warmup", 1)
            except Exception:
                pass  # search degrades to FTS; not worth failing startup over

        threading.Thread(target=go, daemon=True).start()
        yield

    app = FastAPI(title="Iris", docs_url="/api/docs", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.index = None
    app.state.scanning = {}

    def db() -> sqlite3.Connection:
        return init_db(cfg.paths.db_path)

    def _vector_search():
        """Lazily bring up the embedding model and vector index. Returns None if
        embeddings are not built yet, so search degrades to FTS rather than
        failing or paying a multi-second model load on every request."""
        if app.state.index is False:
            return None
        if app.state.index is None:
            from pa.search.vectors import VectorIndex
            index = VectorIndex(cfg.paths.vectors_dir, cfg.embed.dim, cfg.embed.model)
            if index.count() == 0:
                app.state.index = False
                return None
            from pa.providers.registry import get_image_embedder
            app.state.index = (index, get_image_embedder(cfg))
        index, embedder = app.state.index

        def go(text: str, limit: int) -> list[int]:
            return index.search(embedder.embed_texts([text])[0], limit,
                                min_score=cfg.embed.min_score,
                                rel_score=cfg.embed.rel_score)
        return go

    # ------------------------------------------------------------------ search
    @app.get("/api/search")
    def api_search(q: str = "", limit: int = Query(80, le=500), offset: int = 0,
                   sort: str = "newest") -> Any:
        conn = db()
        if not q.strip():
            # Whitelist the ordering: this is interpolated into SQL, and taking
            # it from the query string unchecked would be an injection.
            order = {"newest": "p.taken_at DESC", "oldest": "p.taken_at ASC",
                     "added": "p.created_at DESC",
                     "name": "(SELECT filename FROM file f WHERE f.photo_id=p.id LIMIT 1) ASC"
                     }.get(sort, "p.taken_at DESC")
            rows = conn.execute(
                f"""SELECT p.id, p.blake3, p.taken_at, p.width, p.height,
                          (SELECT caption FROM annotation a WHERE a.photo_id=p.id
                           ORDER BY (a.model='manual') DESC, a.created_at DESC LIMIT 1) caption,
                          (SELECT filename FROM file f WHERE f.photo_id=p.id LIMIT 1) filename,
                          (SELECT rel_path FROM file f WHERE f.photo_id=p.id LIMIT 1) rel_path
                   FROM photo p WHERE p.hidden=0
                   ORDER BY {order} LIMIT ? OFFSET ?""", (limit, offset)).fetchall()
            return {"query": q, "total": None,
                    "results": [dict(r) | {"score": None} for r in rows]}

        # Rank well past the page so `total` is a real count, not just the page
        # size wearing a number. Capped so a two-word query on a 500k library
        # does not fuse a million rankings to print a headline.
        COUNT_CAP = 2000
        hits = run_search(conn, q, limit=max(COUNT_CAP, limit + offset),
                          vector_search=_vector_search())
        page = hits[offset:offset + limit]
        ids = [h.photo_id for h in page]
        dims = {}
        if ids:
            ph = ",".join("?" * len(ids))
            dims = {r["id"]: r for r in conn.execute(
                f"""SELECT p.id, p.width, p.height,
                           (SELECT rel_path FROM file f WHERE f.photo_id=p.id LIMIT 1) rel_path
                    FROM photo p WHERE p.id IN ({ph})""", ids)}
        out = []
        for h in page:
            row = dims.get(h.photo_id)
            out.append({
                "id": h.photo_id, "blake3": h.blake3, "caption": h.caption,
                "filename": h.filename, "taken_at": h.taken_at,
                "score": round(h.score, 5), "sources": list(h.sources),
                "width": row["width"] if row else None,
                "height": row["height"] if row else None,
                "rel_path": row["rel_path"] if row else None,
            })
        return {"query": q, "total": len(hits), "capped": len(hits) >= COUNT_CAP,
                "results": out}

    # ------------------------------------------------------------------ photos
    @app.get("/api/photos/{photo_id}")
    def api_photo(photo_id: int) -> Any:
        conn = db()
        photo = conn.execute("SELECT * FROM photo WHERE id=?", (photo_id,)).fetchone()
        if photo is None:
            raise HTTPException(404, "no such photo")
        ann = conn.execute(
            f"SELECT * FROM annotation a WHERE photo_id=? "
            f"ORDER BY {ANNOTATION_ORDER} LIMIT 1", (photo_id,)).fetchone()
        tags = conn.execute(
            """SELECT t.name, pt.source FROM photo_tag pt JOIN tag t ON t.id=pt.tag_id
               WHERE pt.photo_id=? ORDER BY pt.source DESC, t.name""", (photo_id,)).fetchall()
        faces = conn.execute(
            """SELECT f.id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, f.det_score,
                      f.person_id, f.cluster_id, f.confirmed, pe.name AS person_name
               FROM face f LEFT JOIN person pe ON pe.id=f.person_id
               WHERE f.photo_id=? AND f.rejected=0""", (photo_id,)).fetchall()
        files = conn.execute(
            """SELECT f.rel_path, f.filename, f.state, v.label AS drive, v.online
               FROM file f JOIN root r ON r.id=f.root_id JOIN volume v ON v.id=r.volume_id
               WHERE f.photo_id=?""", (photo_id,)).fetchall()
        return {"photo": dict(photo), "annotation": dict(ann) if ann else None,
                "tags": [dict(t) for t in tags], "faces": [dict(f) for f in faces],
                "files": [dict(f) for f in files]}

    @app.put("/api/photos/{photo_id}/tags")
    def api_set_tags(photo_id: int, body: TagsIn) -> Any:
        conn = db()
        repo.set_tags(conn, photo_id, body.tags, source="manual")
        repo.reindex_fts(conn, photo_id)
        conn.commit()
        return {"ok": True, "tags": body.tags}

    @app.put("/api/photos/{photo_id}/annotation")
    def api_edit_annotation(photo_id: int, body: AnnotationIn) -> Any:
        """Edit the caption, scene or transcribed text by hand.

        Saved as a separate annotation row with model='manual'. Because the
        photo's newest annotation wins everywhere, an edit takes effect at once
        and re-running the AI stage cannot silently overwrite it -- the model
        writes its own row keyed by (photo, model, version).
        """
        conn = db()
        if conn.execute("SELECT 1 FROM photo WHERE id=?", (photo_id,)).fetchone() is None:
            raise HTTPException(404, "no such photo")
        current = conn.execute(
            f"""SELECT caption, scene, setting, people_count, ocr_text FROM annotation a
                WHERE photo_id=? ORDER BY {ANNOTATION_ORDER} LIMIT 1""",
            (photo_id,)).fetchone()
        merged = {
            "caption": body.caption if body.caption is not None
            else (current["caption"] if current else ""),
            "scene": body.scene if body.scene is not None
            else (current["scene"] if current else ""),
            "ocr_text": body.ocr_text if body.ocr_text is not None
            else (current["ocr_text"] if current else ""),
            "setting": current["setting"] if current else "unknown",
            "people_count": current["people_count"] if current else 0,
        }
        repo.save_annotation(conn, photo_id, merged, "manual", "edited")
        repo.reindex_fts(conn, photo_id)
        conn.commit()
        return {"ok": True, **merged}

    @app.delete("/api/photos/{photo_id}/annotation")
    def api_revert_annotation(photo_id: int) -> Any:
        """Throw away hand edits and fall back to what the model said."""
        conn = db()
        n = conn.execute("DELETE FROM annotation WHERE photo_id=? AND model='manual'",
                         (photo_id,)).rowcount
        repo.reindex_fts(conn, photo_id)
        conn.commit()
        return {"ok": True, "reverted": n}

    @app.post("/api/photos/{photo_id}/favorite")
    def api_favorite(photo_id: int, on: bool = True) -> Any:
        conn = db()
        conn.execute("UPDATE photo SET favorite=? WHERE id=?", (1 if on else 0, photo_id))
        conn.commit()
        return {"ok": True, "favorite": on}

    # -------------------------------------------------------------------- media
    @app.get("/api/thumb/{digest}/{kind}")
    def api_thumb(digest: str, kind: str = "grid") -> Any:
        if kind not in ("grid", "view"):
            raise HTTPException(400, "kind must be grid or view")
        path = thumbs.thumb_path(cfg.paths.thumbs_dir, digest, kind,
                                 cfg.thumbs.format.lower())
        if not path.exists():
            raise HTTPException(404, "no thumbnail yet")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=31536000"})

    @app.get("/api/face/{face_id}")
    def api_face(face_id: int, size: int = 160) -> Any:
        """A cropped face, which is the only useful thing on the People screen --
        a naming queue showing whole photos makes you squint at every card.

        Bounding boxes are in the coordinate space of the image the detector
        actually ran on: the cached 'view' thumbnail (see _stage_source), not the
        original. Cropping from that same thumbnail keeps the two in step.
        """
        import io

        from fastapi.responses import Response
        from PIL import Image

        conn = db()
        face = conn.execute(
            """SELECT f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, p.blake3
               FROM face f JOIN photo p ON p.id=f.photo_id WHERE f.id=?""",
            (face_id,)).fetchone()
        if face is None:
            raise HTTPException(404, "no such face")
        src = thumbs.thumb_path(cfg.paths.thumbs_dir, face["blake3"], "view",
                                cfg.thumbs.format.lower())
        if not src.exists():
            raise HTTPException(404, "no thumbnail for that photo yet")

        with Image.open(src) as img:
            img = img.convert("RGB")
            x, y, w, h = face["bbox_x"], face["bbox_y"], face["bbox_w"], face["bbox_h"]
            # 35% padding: a tight ArcFace box cuts the hair and chin, which are
            # most of what a person uses to recognise someone at thumbnail size.
            pad = int(max(w, h) * 0.35)
            box = (max(x - pad, 0), max(y - pad, 0),
                   min(x + w + pad, img.width), min(y + h + pad, img.height))
            crop = img.crop(box)
            crop.thumbnail((size, size), Image.LANCZOS)
            buf = io.BytesIO()
            crop.save(buf, "WEBP", quality=85)
        return Response(buf.getvalue(), media_type="image/webp",
                        headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/original/{photo_id}")
    def api_original(photo_id: int) -> Any:
        conn = db()
        path = scanner.resolve_file_path(conn, photo_id)
        if path is None:
            raise HTTPException(
                410, "source drive is not connected")
        return FileResponse(path)

    # ------------------------------------------------------------------- people
    @app.get("/api/people")
    def api_people() -> Any:
        conn = db()
        rows = conn.execute(
            """SELECT pe.id, pe.name, pe.cover_face_id, COUNT(f.id) n,
                      (SELECT f2.id FROM face f2
                       WHERE f2.person_id=pe.id ORDER BY f2.det_score DESC LIMIT 1) cover_face
               FROM person pe LEFT JOIN face f ON f.person_id=pe.id AND f.rejected=0
               GROUP BY pe.id ORDER BY n DESC""").fetchall()
        return {"people": [dict(r) for r in rows]}

    @app.get("/api/clusters")
    def api_clusters(limit: int = 60) -> Any:
        """Unnamed face clusters, biggest first - the naming queue."""
        conn = db()
        rows = conn.execute(
            """SELECT cluster_id, COUNT(*) n FROM face
               WHERE cluster_id IS NOT NULL AND person_id IS NULL AND rejected=0
               GROUP BY cluster_id ORDER BY n DESC LIMIT ?""", (limit,)).fetchall()
        out = []
        for r in rows:
            faces = conn.execute(
                """SELECT f.id, f.photo_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                          f.det_score, p.blake3
                   FROM face f JOIN photo p ON p.id=f.photo_id
                   WHERE f.cluster_id=? ORDER BY f.det_score DESC LIMIT 12""",
                (r["cluster_id"],)).fetchall()
            out.append({"cluster_id": r["cluster_id"], "count": r["n"],
                        "faces": [dict(f) for f in faces]})
        return {"clusters": out}

    @app.post("/api/clusters/{cluster_id}/name")
    def api_name_cluster(cluster_id: int, body: NameIn) -> Any:
        from pa.faces.cluster import name_cluster
        conn = db()
        n = name_cluster(conn, cluster_id, body.name.strip())
        return {"ok": True, "named": n, "name": body.name.strip()}

    @app.post("/api/clusters/merge")
    def api_merge_clusters(body: MergeIn) -> Any:
        """Fold several proposed clusters into one. Clustering splits a person
        across several groups whenever lighting, age or angle vary enough, so
        merging is the single most common correction."""
        if len(body.cluster_ids) < 2:
            raise HTTPException(400, "need at least two clusters to merge")
        conn = db()
        target = min(body.cluster_ids)
        marks = ",".join("?" * len(body.cluster_ids))
        conn.execute(f"UPDATE face SET cluster_id=? WHERE cluster_id IN ({marks})",
                     [target, *body.cluster_ids])
        conn.commit()
        if body.name and body.name.strip():
            from pa.faces.cluster import name_cluster
            n = name_cluster(conn, target, body.name.strip())
            return {"ok": True, "cluster_id": target, "named": n}
        n = conn.execute("SELECT COUNT(*) FROM face WHERE cluster_id=?", (target,)).fetchone()[0]
        return {"ok": True, "cluster_id": target, "faces": n}

    @app.post("/api/faces/{face_id}/detach")
    def api_detach_face(face_id: int) -> Any:
        """Remove one face from its cluster or person - the 'that isn't them'
        action. It becomes unassigned rather than deleted, so the next
        clustering run can place it somewhere better."""
        conn = db()
        row = conn.execute("SELECT photo_id FROM face WHERE id=?", (face_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "no such face")
        conn.execute(
            "UPDATE face SET person_id=NULL, cluster_id=NULL, confirmed=0 WHERE id=?",
            (face_id,))
        repo.reindex_fts(conn, row["photo_id"])
        conn.commit()
        return {"ok": True}

    @app.post("/api/faces/{face_id}/assign")
    def api_assign_face(face_id: int, person_id: int) -> Any:
        conn = db()
        row = conn.execute("SELECT photo_id FROM face WHERE id=?", (face_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "no such face")
        conn.execute(
            "UPDATE face SET person_id=?, cluster_id=NULL, confirmed=1 WHERE id=?",
            (person_id, face_id))
        repo.reindex_fts(conn, row["photo_id"])
        conn.commit()
        return {"ok": True}

    @app.patch("/api/people/{person_id}")
    def api_rename_person(person_id: int, body: NameIn) -> Any:
        conn = db()
        name = body.name.strip()
        if not name:
            raise HTTPException(400, "name cannot be empty")
        try:
            conn.execute("UPDATE person SET name=? WHERE id=?", (name, person_id))
        except sqlite3.IntegrityError:
            raise HTTPException(409, f"someone is already called {name!r}") from None
        for r in conn.execute("SELECT DISTINCT photo_id FROM face WHERE person_id=?",
                              (person_id,)):
            repo.reindex_fts(conn, r["photo_id"])
        conn.commit()
        return {"ok": True, "name": name}

    @app.delete("/api/people/{person_id}")
    def api_delete_person(person_id: int) -> Any:
        """Un-name a person. Their faces go back to the naming queue."""
        conn = db()
        photos = [r["photo_id"] for r in conn.execute(
            "SELECT DISTINCT photo_id FROM face WHERE person_id=?", (person_id,))]
        conn.execute("UPDATE face SET person_id=NULL, confirmed=0 WHERE person_id=?",
                     (person_id,))
        conn.execute("DELETE FROM person WHERE id=?", (person_id,))
        for pid in photos:
            repo.reindex_fts(conn, pid)
        conn.commit()
        return {"ok": True}

    @app.get("/api/duplicates")
    def api_duplicates(near: bool = False, distance: int = 6) -> Any:
        from pa.cli import _near_duplicates
        conn = db()
        exact = [dict(r) for r in conn.execute(
            """SELECT p.id, p.blake3, p.bytes, COUNT(f.id) n,
                      group_concat(v.label || ': ' || f.rel_path, char(10)) locations
               FROM photo p JOIN file f ON f.photo_id=p.id
               JOIN root r ON r.id=f.root_id JOIN volume v ON v.id=r.volume_id
               GROUP BY p.id HAVING n > 1 ORDER BY p.bytes * (n - 1) DESC LIMIT 200""")]
        wasted = sum((r["bytes"] or 0) * (r["n"] - 1) for r in exact)
        out: dict[str, Any] = {"exact": exact, "wasted_bytes": wasted, "near": []}
        if near:
            groups = _near_duplicates(conn, distance)[:100]
            for g in groups:
                marks = ",".join("?" * len(g["ids"]))
                g["photos"] = [dict(r) for r in conn.execute(
                    f"""SELECT p.id, p.blake3, p.bytes,
                               (SELECT filename FROM file f WHERE f.photo_id=p.id LIMIT 1) filename
                        FROM photo p WHERE p.id IN ({marks})""", g["ids"])]
            out["near"] = groups
        return out

    @app.get("/api/map")
    def api_map(limit: int = 5000) -> Any:
        conn = db()
        rows = conn.execute(
            """SELECT p.id, p.blake3, p.gps_lat, p.gps_lon, p.taken_at, p.place_name,
                      (SELECT caption FROM annotation a WHERE a.photo_id=p.id
                       ORDER BY a.created_at DESC LIMIT 1) caption
               FROM photo p WHERE p.gps_lat IS NOT NULL AND p.hidden=0
               ORDER BY p.taken_at DESC LIMIT ?""", (limit,)).fetchall()
        return {"points": [dict(r) for r in rows]}

    # -------------------------------------------------------------------- admin
    # ------------------------------------------------------------------- roots
    @app.get("/api/browse")
    def api_browse(path: str | None = None) -> Any:
        """List folders on the machine running the server, so the UI can offer a
        picker. A browser's own file dialog can only see the client's disks."""
        from dataclasses import asdict

        from pa.api import roots as rootlib
        try:
            return asdict(rootlib.browse(path))
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from None
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise HTTPException(404, str(exc)) from None

    @app.get("/api/roots")
    def api_roots() -> Any:
        conn = db()
        out = []
        for r in repo.list_roots(conn):
            mount = volumes.current_mountpoint(r["volume_uuid"])
            counts = conn.execute(
                """SELECT COUNT(DISTINCT f.photo_id) photos,
                          SUM(f.state='missing') missing
                   FROM file f WHERE f.root_id=?""", (r["id"],)).fetchone()
            out.append(dict(r) | {
                "online": mount is not None,
                "full_path": str(mount / r["rel_path"]) if mount else None,
                "photos": counts["photos"] or 0,
                "missing": counts["missing"] or 0,
                "scanning": r["id"] in app.state.scanning,
            })
        return {"roots": out}

    @app.post("/api/roots")
    def api_add_root(body: RootIn) -> Any:
        path = Path(body.path).expanduser()
        if not path.is_dir():
            raise HTTPException(400, f"not a folder: {path}")
        conn = db()
        try:
            vol_uuid, vol_label, mount = volumes.identify(path)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(400, f"could not identify the drive: {exc}") from None
        rel = volumes.relative_to_mount(path, mount)
        existing = conn.execute(
            """SELECT r.id FROM root r JOIN volume v ON v.id=r.volume_id
               WHERE v.uuid=? AND r.rel_path=?""", (vol_uuid, rel)).fetchone()
        if existing:
            raise HTTPException(409, "that folder is already in the library")

        volume_id = repo.upsert_volume(conn, vol_uuid, vol_label, str(mount.mountpoint))
        root_id = repo.add_root(conn, volume_id, rel, body.label or path.name, body.exclude)
        conn.commit()
        if body.scan:
            _start_scan(root_id)
        return {"ok": True, "root_id": root_id, "drive": vol_label, "scanning": body.scan}

    @app.delete("/api/roots/{root_id}")
    def api_remove_root(root_id: int) -> Any:
        conn = db()
        if conn.execute("SELECT 1 FROM root WHERE id=?", (root_id,)).fetchone() is None:
            raise HTTPException(404, "no such folder")
        files, orphans = repo.remove_root(conn, root_id)
        conn.commit()
        return {"ok": True, "files_removed": files, "photos_dropped": orphans}

    @app.post("/api/roots/{root_id}/scan")
    def api_scan_root(root_id: int) -> Any:
        conn = db()
        if conn.execute("SELECT 1 FROM root WHERE id=?", (root_id,)).fetchone() is None:
            raise HTTPException(404, "no such folder")
        if root_id in app.state.scanning:
            raise HTTPException(409, "that folder is already being scanned")
        _start_scan(root_id)
        return {"ok": True}

    @app.get("/api/scans")
    def api_scans() -> Any:
        """Progress of running scans, for the UI to poll."""
        return {"scanning": {str(k): v for k, v in app.state.scanning.items()}}

    def _start_scan(root_id: int) -> None:
        """Scan on a worker thread with its own connection.

        A scan walks a whole drive and can take minutes; doing it inside the
        request would hold the HTTP worker and time out the browser. Each thread
        opens its own SQLite connection (connect() is thread-local), so this does
        not share a handle across threads.
        """
        def go() -> None:
            app.state.scanning[root_id] = {"seen": 0, "new": 0, "state": "scanning"}
            try:
                worker_conn = init_db(cfg.paths.db_path)
                row = next((r for r in repo.list_roots(worker_conn)
                            if r["id"] == root_id), None)
                if row is None:
                    return

                def progress(st) -> None:
                    app.state.scanning[root_id] = {
                        "seen": st.seen, "new": st.new_photos, "state": "scanning"}

                st = scanner.scan_root(worker_conn, row, cfg, on_progress=progress)
                app.state.scanning[root_id] = {
                    "seen": st.seen, "new": st.new_photos, "state": "done",
                    "errors": st.errors[:3]}
                # A scan leaves a queue behind, not results. Draining it here is
                # what makes "add a folder" produce visible photos rather than a
                # grid of placeholders waiting for someone to know about the CLI.
                # Empty auto_process, or a run already going, leaves it alone.
                if cfg.auto_process and not runner.running:
                    # A manual run getting there first is fine: its work covers
                    # the same queue, so there is nothing to recover from.
                    with contextlib.suppress(RuntimeError, ValueError):
                        runner.start(cfg, list(cfg.auto_process), trigger="scan")
            except Exception as exc:
                app.state.scanning[root_id] = {"state": "failed", "error": str(exc)[:300]}
            finally:
                threading.Timer(20, lambda: app.state.scanning.pop(root_id, None)).start()

        threading.Thread(target=go, daemon=True).start()

    @app.get("/api/stats")
    def api_stats() -> Any:
        conn = db()
        one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
        roots = [dict(r) | {"online": bool(r["online"])} for r in repo.list_roots(conn)]
        return {
            "photos": one("SELECT COUNT(*) FROM photo"),
            "files": one("SELECT COUNT(*) FROM file"),
            "offline": one("SELECT COUNT(*) FROM file WHERE state='offline'"),
            "captioned": one("SELECT COUNT(DISTINCT photo_id) FROM annotation"),
            "faces": one("SELECT COUNT(*) FROM face"),
            "people": one("SELECT COUNT(*) FROM person"),
            "tags": one("SELECT COUNT(*) FROM tag"),
            "jobs": repo.job_stats(conn),
            "roots": roots,
        }

    @app.get("/api/tags")
    def api_tags(limit: int = 200) -> Any:
        conn = db()
        rows = conn.execute(
            """SELECT t.name, COUNT(pt.photo_id) n FROM tag t
               JOIN photo_tag pt ON pt.tag_id=t.id
               GROUP BY t.id ORDER BY n DESC LIMIT ?""", (limit,)).fetchall()
        return {"tags": [dict(r) for r in rows]}

    # --------------------------------------------------------------- pipeline
    # A scan only *enqueues* work. Something has to drain that queue, and until
    # these existed the only something was `pa process` in a terminal -- which
    # is why a freshly scanned folder sat there as a grid of grey placeholders.
    @app.get("/api/process")
    def api_process_status() -> Any:
        conn = db()
        # "stages" belongs to the running job (what this run was asked to do);
        # "stages_all" is every stage that exists, which is what the UI draws
        # cards for. Naming them apart keeps the run's own list from being
        # clobbered by the catalogue.
        return {**runner.status(), "queue": repo.job_stats(conn),
                "stages_all": list(runner_stages)}

    @app.post("/api/process")
    def api_process_start(body: ProcessIn) -> Any:
        try:
            return runner.start(cfg, body.stages, body.limit, trigger="manual")
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post("/api/process/cancel")
    def api_process_cancel() -> Any:
        return {"ok": True, "cancelling": runner.cancel()}

    @app.post("/api/process/retry-failed")
    def api_retry_failed(stage: str | None = None) -> Any:
        """Put failed jobs back in the queue.

        Failures are usually environmental -- an unplugged drive, a model server
        that was down -- so they should be retryable without re-scanning the
        whole folder, which would re-hash every file to learn nothing.
        """
        conn = db()
        sql = "UPDATE job SET state='pending', attempts=0, error=NULL WHERE state='failed'"
        args: list = []
        if stage:
            if stage not in repo.STAGES:
                raise HTTPException(400, f"unknown stage {stage!r}")
            sql += " AND stage=?"
            args.append(stage)
        n = conn.execute(sql, args).rowcount
        conn.commit()
        return {"ok": True, "requeued": n}

    @app.post("/api/process/reset")
    def api_reset_stage(body: ResetIn) -> Any:
        """Re-run a stage over the whole library, e.g. after changing its model.

        `rebuild` additionally throws away what the stage already produced. It
        matters for thumbnails: generate() skips any thumbnail whose file is
        already there, so re-queueing alone after changing the size or format
        would report 'done' for every photo and change nothing at all.
        """
        if body.stage not in repo.STAGES:
            raise HTTPException(400, f"unknown stage {body.stage!r}")
        removed = 0
        if body.rebuild and body.stage == "thumbs":
            # Derived data by definition -- the README calls the thumbnail cache
            # rebuildable. The cost is that photos on a disconnected drive lose
            # their previews until that drive is back, so the UI confirms first.
            for path in cfg.paths.thumbs_dir.rglob("*.*"):
                with contextlib.suppress(OSError):
                    path.unlink()
                    removed += 1
        conn = db()
        n = conn.execute(
            "UPDATE job SET state='pending', attempts=0, error=NULL WHERE stage=?",
            (body.stage,)).rowcount
        conn.commit()
        return {"ok": True, "stage": body.stage, "requeued": n, "removed": removed}

    @app.post("/api/people/cluster")
    def api_cluster() -> Any:
        """Regroup faces into people. Runs through the same runner as indexing so
        the two cannot collide over the database."""
        try:
            return runner.start(cfg, ["cluster"], trigger="manual")
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None

    # -------------------------------------------------------------- maintenance
    @app.post("/api/sidecar/export")
    def api_sidecar_export(body: SidecarIn) -> Any:
        from pa.sidecar import bulk
        conn = db()
        use = cfg
        if body.beside is not None:
            use = cfg.model_copy(deep=True)
            use.sidecar.location = "beside" if body.beside else "app"
        res = bulk.export(conn, use, body.root_id, overwrite=body.overwrite)
        return {"ok": True, "written": res.written, "skipped": res.skipped,
                "offline": res.offline, "total": res.total, "errors": res.errors[:5],
                "location": "beside your photos" if use.sidecar.location == "beside"
                else str(use.paths.sidecars_dir)}

    @app.post("/api/sidecar/import")
    def api_sidecar_import(body: SidecarIn) -> Any:
        from pa.sidecar import bulk
        conn = db()
        res = bulk.read_back(conn, cfg, body.root_id)
        return {"ok": True, "found": res.found, "tags": res.tags}

    @app.post("/api/prune")
    def api_prune(body: PruneIn) -> Any:
        conn = db()
        n = repo.prune_orphans(conn, drop_missing=not body.keep_missing)
        conn.commit()
        return {"ok": True, "dropped": n}

    # ------------------------------------------------------------------ config
    @app.get("/api/config")
    def api_config() -> Any:
        """Current settings, plus everything the UI needs to render them well:
        the defaults to offer as a reset, the provider names actually installed,
        and which fields cannot take effect without a restart or a re-index."""
        from pa.config import (
            CONFIG_PATH,
            REINDEX_REQUIRED,
            RESTART_REQUIRED,
            Config,
        )
        from pa.providers.registry import available

        def shape(c) -> dict[str, Any]:
            return {
                "caption": c.caption.model_dump(), "embed": c.embed.model_dump(),
                "face": c.face.model_dump(), "thumbs": c.thumbs.model_dump(),
                "sidecar": c.sidecar.model_dump(),
                "server": {"host": c.host, "port": c.port,
                           "scan_workers": c.scan_workers,
                           "auto_process": list(c.auto_process)},
            }

        return {
            "path": str(CONFIG_PATH), "exists": CONFIG_PATH.exists(),
            "config": shape(cfg), "defaults": shape(Config()),
            "providers": {kind: sorted(available(kind))
                          for kind in ("caption", "image_embed", "face")},
            "restart_required": RESTART_REQUIRED,
            "reindex_required": REINDEX_REQUIRED,
            "stages": list(repo.STAGES),
            "paths": {"data": str(cfg.paths.data_dir), "cache": str(cfg.paths.cache_dir),
                      "database": str(cfg.paths.db_path),
                      "thumbnails": str(cfg.paths.thumbs_dir),
                      "vectors": str(cfg.paths.vectors_dir),
                      "sidecars": str(cfg.paths.sidecars_dir)},
        }

    @app.put("/api/config")
    def api_save_config(body: ConfigIn) -> Any:
        from pydantic import ValidationError

        from pa.config import save_config
        try:
            save_config(body.model_dump())
        except ValidationError as exc:
            # Surface the field and the reason, not a wall of pydantic JSON.
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"])
            raise HTTPException(400, f"{where}: {first['msg']}") from None
        except OSError as exc:
            raise HTTPException(500, f"could not write the config file: {exc}") from None

        # save_config() mutates the Config object in place, so the `cfg` captured
        # by every handler in this closure is already current. What it cannot fix
        # is anything built *from* those settings, so drop those here.
        app.state.index = None
        runner.forget_providers()
        return api_config()

    @app.get("/api/config/check")
    def api_config_check() -> Any:
        from pa.providers import health
        return health.report(cfg)

    @app.exception_handler(sqlite3.OperationalError)
    def _sqlite_error(request, exc):  # pragma: no cover
        return JSONResponse({"error": str(exc)}, status_code=500)

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
