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


def _model_cache_dir() -> str:
    """Where the downloaded model weights actually sit.

    Not under this app's cache directory: transformers and insightface keep
    their own, shared with every other tool on the machine, and pointing them
    somewhere private would mean downloading several gigabytes again for no
    gain. Worth reporting, since "why is it downloading" is otherwise
    unanswerable from inside the app.
    """
    try:
        from huggingface_hub import constants
        return str(constants.HF_HUB_CACHE)
    except Exception:
        return "(transformers not installed)"


def _scaled_box(face, width: int, height: int) -> tuple[float, float, float, float]:
    """A face box in the pixel space of the image being cropped right now.

    Boxes are stored in the coordinates of the thumbnail the detector ran on,
    and that thumbnail is not permanent: changing thumbs.view_px and rebuilding
    remakes it at another size. Cropping with the stale numbers then asks for a
    region outside the image, which PIL rejects with "Coordinate 'right' is less
    than 'left'". src_w/src_h are recorded for exactly this, so rescale instead
    of trusting that the thumbnail never changed.
    """
    x, y, w, h = (float(face["bbox_x"]), float(face["bbox_y"]),
                  float(face["bbox_w"]), float(face["bbox_h"]))
    src_w, src_h = face["src_w"], face["src_h"]
    if src_w and src_h and (src_w != width or src_h != height):
        sx, sy = width / src_w, height / src_h
        x, y, w, h = x * sx, y * sy, w * sx, h * sy
    return x, y, w, h


def _clamped(left: float, top: float, right: float, bottom: float,
             width: int, height: int) -> tuple[int, int, int, int]:
    """An integer crop box guaranteed to be inside the image and at least 1px.

    Faces detected at the frame edge have negative coordinates, and a box from
    a photo whose thumbnail changed shape can miss entirely. Neither is worth a
    500: a slightly wrong crop still lets you recognise and name the person.
    """
    left = min(max(left, 0), width - 1)
    top = min(max(top, 0), height - 1)
    right = max(min(right, width), left + 1)
    bottom = max(min(bottom, height), top + 1)
    return int(left), int(top), int(right), int(bottom)


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
    start: bool = False     # and begin the run, rather than only filling the queue


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
            # Shared with the indexer rather than loaded separately: same model,
            # same settings, and two copies is a wasted gigabyte of VRAM plus a
            # second wait for weights that were already in memory.
            app.state.index = (index, runner.provider("embed", cfg))
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
        original. That thumbnail can have been rebuilt at another size since, so
        the box is rescaled to the file on disk rather than assumed to match it.
        """
        import io

        from fastapi.responses import Response
        from PIL import Image

        conn = db()
        face = conn.execute(
            """SELECT f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, f.src_w, f.src_h, p.blake3
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
            x, y, w, h = _scaled_box(face, img.width, img.height)
            # 35% padding: a tight ArcFace box cuts the hair and chin, which are
            # most of what a person uses to recognise someone at thumbnail size.
            pad = max(w, h) * 0.35
            crop = img.crop(_clamped(x - pad, y - pad, x + w + pad, y + h + pad,
                                     img.width, img.height))
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

        # Recognising a name you are shown beats recalling one from a face crop,
        # and it is what keeps one person from being entered twice under two
        # spellings.
        from pa.faces.cluster import suggest_people
        guesses = suggest_people(conn, cfg, [c["cluster_id"] for c in out])
        for c in out:
            c["suggestions"] = guesses.get(c["cluster_id"], [])
        return {"clusters": out}

    @app.get("/api/clusters/{cluster_id}/faces")
    def api_cluster_faces(cluster_id: int, limit: int = 200) -> Any:
        """Every face in one group, with the photo each came from.

        The naming queue shows five crops, which is enough to say "that is
        Sarah" and not enough for anything else. Detection is imperfect: a group
        can be a pattern on a shirt, a face on a poster, or two people the
        clusterer ran together -- and none of that is visible until you can see
        the whole photo a crop was cut out of.
        """
        conn = db()
        rows = conn.execute(
            f"""SELECT f.id, f.photo_id, f.det_score, p.blake3, p.taken_at,
                      (SELECT a.caption FROM annotation a WHERE a.photo_id=p.id
                       ORDER BY {ANNOTATION_ORDER} LIMIT 1) AS caption,
                      (SELECT fi.filename FROM file fi WHERE fi.photo_id=p.id LIMIT 1)
                          AS filename
               FROM face f JOIN photo p ON p.id=f.photo_id
               WHERE f.cluster_id=? ORDER BY f.det_score DESC LIMIT ?""",
            (cluster_id, limit)).fetchall()
        if not rows:
            raise HTTPException(404, "no such group, or it is already gone")
        return {"cluster_id": cluster_id, "faces": [dict(r) for r in rows]}

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

    # Group photos are full of strangers. Without a way to say "not a person I
    # care about", they come back to the naming queue after every clustering
    # run and there is no way to ever finish naming. `face.rejected` already
    # excluded them everywhere -- clustering, the queue, people counts and
    # sidecar export -- but nothing could set it.
    @app.post("/api/clusters/{cluster_id}/ignore")
    def api_ignore_cluster(cluster_id: int) -> Any:
        conn = db()
        photos = [r["photo_id"] for r in conn.execute(
            "SELECT DISTINCT photo_id FROM face WHERE cluster_id=?", (cluster_id,))]
        n = conn.execute(
            "UPDATE face SET rejected=1, ignored_as=?, cluster_id=NULL WHERE cluster_id=?",
            (cluster_id, cluster_id)).rowcount
        if not n:
            raise HTTPException(404, "no such group, or it is already gone")
        for pid in photos:
            repo.reindex_fts(conn, pid)
        conn.commit()
        return {"ok": True, "ignored": n}

    @app.post("/api/faces/{face_id}/ignore")
    def api_ignore_face(face_id: int) -> Any:
        conn = db()
        row = conn.execute("SELECT photo_id FROM face WHERE id=?", (face_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "no such face")
        # A face ignored one at a time never belonged to a group, so ignored_as
        # stays NULL and it shows in the "loose" count rather than as a group.
        conn.execute("UPDATE face SET rejected=1, person_id=NULL, cluster_id=NULL, "
                     "confirmed=0 WHERE id=?", (face_id,))
        repo.reindex_fts(conn, row["photo_id"])
        conn.commit()
        return {"ok": True}

    @app.get("/api/faces/ignored")
    def api_ignored_faces(limit: int = 60) -> Any:
        """Ignoring has to be undoable, so it needs somewhere to be seen.

        Grouped by the cluster they were in when ignored, so undoing puts back
        the same group that was dismissed rather than a heap of loose faces.
        """
        conn = db()
        rows = conn.execute(
            """SELECT ignored_as AS cluster_id, COUNT(*) n FROM face
               WHERE rejected=1 AND ignored_as IS NOT NULL
               GROUP BY ignored_as ORDER BY n DESC LIMIT ?""", (limit,)).fetchall()
        out = []
        for r in rows:
            faces = conn.execute(
                """SELECT id, det_score FROM face WHERE ignored_as=? AND rejected=1
                   ORDER BY det_score DESC LIMIT 6""", (r["cluster_id"],)).fetchall()
            out.append({"cluster_id": r["cluster_id"], "count": r["n"],
                        "faces": [dict(f) for f in faces]})
        loose = conn.execute(
            "SELECT COUNT(*) FROM face WHERE rejected=1 AND ignored_as IS NULL").fetchone()[0]
        return {"groups": out, "loose": loose}

    @app.post("/api/faces/ignored/{cluster_id}/restore")
    def api_restore_ignored(cluster_id: int) -> Any:
        conn = db()
        photos = [r["photo_id"] for r in conn.execute(
            "SELECT DISTINCT photo_id FROM face WHERE ignored_as=? AND rejected=1",
            (cluster_id,))]
        n = conn.execute(
            "UPDATE face SET rejected=0, cluster_id=ignored_as, ignored_as=NULL "
            "WHERE ignored_as=? AND rejected=1", (cluster_id,)).rowcount
        if not n:
            raise HTTPException(404, "nothing ignored under that group")
        for pid in photos:
            repo.reindex_fts(conn, pid)
        conn.commit()
        return {"ok": True, "restored": n}

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

    @app.post("/api/people/{person_id}/merge")
    def api_merge_people(person_id: int, into: int) -> Any:
        """Fold one named person into another.

        Clustering splits a person across groups, and naming those groups
        separately -- "Sarah" and "Sarah B" -- is easy to do before you realise
        they are the same. Without this the only way back was to un-name one and
        re-name every face by hand.
        """
        if person_id == into:
            raise HTTPException(400, "that is the same person")
        conn = db()
        rows = conn.execute("SELECT id, name FROM person WHERE id IN (?,?)",
                            (person_id, into)).fetchall()
        if len(rows) != 2:
            raise HTTPException(404, "no such person")
        photos = [r["photo_id"] for r in conn.execute(
            "SELECT DISTINCT photo_id FROM face WHERE person_id=?", (person_id,))]
        moved = conn.execute("UPDATE face SET person_id=? WHERE person_id=?",
                             (into, person_id)).rowcount
        conn.execute("DELETE FROM person WHERE id=?", (person_id,))
        for pid in photos:
            repo.reindex_fts(conn, pid)
        conn.commit()
        kept = next(r["name"] for r in rows if r["id"] == into)
        return {"ok": True, "moved": moved, "name": kept}

    @app.patch("/api/people/{person_id}")
    def api_rename_person(person_id: int, body: NameIn) -> Any:
        conn = db()
        name = body.name.strip()
        if not name:
            raise HTTPException(400, "name cannot be empty")
        # Renaming onto a name that already exists almost always means "these
        # two are the same person". Say so, and hand back the id to merge into,
        # rather than refusing with a bare conflict and no way forward.
        other = conn.execute("SELECT id FROM person WHERE name=? AND id<>?",
                             (name, person_id)).fetchone()
        if other:
            raise HTTPException(409, {
                "message": f"{name} already exists. Merge them together?",
                "merge_into": other["id"], "name": name})
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
    def api_duplicates(near: bool = False, distance: int = 6,
                       raw_pairs: bool = False) -> Any:
        from pa.duplicates import near_duplicates
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
            groups, pairs = near_duplicates(conn, distance,
                                            include_format_pairs=raw_pairs)
            out["format_pairs_skipped"] = pairs
            groups = groups[:100]
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

    def _discard(conn: sqlite3.Connection, stage: str) -> dict[str, int]:
        """Throw away what a stage produced, keeping every decision you made.

        Changing a model is the reason this exists, and each stage needs a
        different amount of help to actually start over:

        thumbs  -- generate() skips any thumbnail whose file already exists, so
                   without deleting them a re-run reports 'done' for every photo
                   and changes nothing at all.
        embed   -- vectors from the old model are a different space entirely.
                   The index file is keyed on model and dim and refuses to load
                   when they disagree, so search degrades to words until the
                   run finishes and rebuilds it.
        faces   -- run_faces only clears detections from the *same* model, which
                   is what lets a re-run keep your names. Change the model and
                   the old model's faces would otherwise stay forever, and every
                   face would appear twice in the naming queue.
        caption -- nothing to discard. Re-captioning replaces each photo's entry
                   as it goes and rewrites that photo's search row with it, so
                   the library is never inconsistent for longer than one photo.
        """
        if stage == "thumbs":
            removed = 0
            for path in cfg.paths.thumbs_dir.rglob("*.*"):
                with contextlib.suppress(OSError):
                    path.unlink()
                    removed += 1
            return {"thumbnails": removed}
        if stage == "embed":
            n = conn.execute("DELETE FROM photo_embedding").rowcount
            # Drop the app's index first: it holds a memory map of these files,
            # and on Windows a mapped file will not be deleted.
            app.state.index = None
            for path in cfg.paths.vectors_dir.glob("*"):
                with contextlib.suppress(OSError):
                    path.unlink()
            return {"vectors": n}
        if stage == "faces":
            # Named (confirmed) and ignored (rejected) faces are decisions, not
            # output, and survive. Auto-assigned ones do not: they are a guess
            # this very stage is about to make again.
            n = conn.execute(
                "DELETE FROM face WHERE confirmed=0 AND rejected=0").rowcount
            return {"faces": n}
        return {}

    @app.post("/api/process/reset")
    def api_reset_stage(body: ResetIn) -> Any:
        """Re-run a stage over the entire library, e.g. after changing its model.

        Stages are idempotent and keyed on what they have already done, which is
        what makes indexing resumable -- and what makes a new model change
        nothing until something says "do it all again". This is that something.
        """
        if body.stage not in repo.STAGES:
            raise HTTPException(
                400, f"unknown stage {body.stage!r}; expected any of {list(repo.STAGES)}")
        conn = db()
        discarded = _discard(conn, body.stage) if body.rebuild else {}
        # Every photo, not just the ones with a job row: a stage that was off
        # when a photo was scanned has no row to reset, and those are exactly
        # the photos someone turning a stage on wants covered.
        n = conn.execute(
            """INSERT INTO job (photo_id, stage, state, priority, created_at)
               SELECT id, ?, 'pending', 100, ? FROM photo WHERE 1
               ON CONFLICT(photo_id, stage) DO UPDATE SET
                 state='pending', attempts=0, error=NULL,
                 started_at=NULL, finished_at=NULL""",
            (body.stage, repo.now())).rowcount
        conn.commit()

        started = False
        if body.start and n:
            try:
                runner.start(cfg, [body.stage], trigger="redo")
                started = True
            except (RuntimeError, ValueError):
                started = False   # already running; the queue is filled either way
        return {"ok": True, "stage": body.stage, "requeued": n,
                "discarded": discarded, "started": started}

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
                      "models": _model_cache_dir(),
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
