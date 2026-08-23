"""XMP sidecar export and import.

Writes `photo.jpg.xmp` next to the original, never touching the original itself.
The format is what Lightroom, digiKam, Immich and exiftool already read:

  dc:description   the caption
  dc:subject       tags (keywords)
  mwg-rs:Regions   named face rectangles, normalised 0-1

This is the escape hatch from the app: everything the pipeline worked out about
a photo travels with the photo, so the library is never a lock-in.
"""
from __future__ import annotations

import re
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from pa.db.repo import ANNOTATION_ORDER

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "mwg-rs": "http://www.metadataworkinggroup.com/schemas/regions/",
    "stArea": "http://ns.adobe.com/xmp/sType/Area#",
    "stDim": "http://ns.adobe.com/xap/1.0/sType/Dimensions#",
    "pa": "https://github.com/photo-anotater/ns/1.0/",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

SIDECAR_SUFFIX = ".xmp"


def sidecar_path(photo_path: Path) -> Path:
    """`IMG_1234.jpg` -> `IMG_1234.jpg.xmp`, beside the original.

    Keeping the full original name (rather than replacing the extension) is what
    digiKam and exiftool default to, and it keeps RAW+JPEG pairs distinct.
    """
    return photo_path.with_name(photo_path.name + SIDECAR_SUFFIX)


def app_sidecar_path(base: Path, drive: str, rel_path: str) -> Path:
    """Where a sidecar lives when it is kept inside the app's own directory.

    The photo's folder tree is mirrored under `base` rather than flattening to a
    hash, so the result stays human-readable and the whole tree can simply be
    copied over the photo folders later if you ever do want it beside them.
    """
    safe_drive = "".join(c if c.isalnum() or c in "-_ " else "_" for c in drive).strip() or "drive"
    return base / safe_drive / (rel_path + SIDECAR_SUFFIX)


def resolve_path(cfg, conn: sqlite3.Connection, photo_id: int,
                 photo_path: Path) -> Path:
    """The sidecar location for this photo, honouring the configured mode."""
    if cfg.sidecar.location == "beside":
        return sidecar_path(photo_path)
    row = conn.execute(
        """SELECT f.rel_path, v.label FROM file f
           JOIN root r ON r.id = f.root_id JOIN volume v ON v.id = r.volume_id
           WHERE f.photo_id = ? LIMIT 1""", (photo_id,)).fetchone()
    if row is None:
        return app_sidecar_path(cfg.paths.sidecars_dir, "unknown", photo_path.name)
    return app_sidecar_path(cfg.paths.sidecars_dir, row["label"], row["rel_path"])


@dataclass
class Sidecar:
    caption: str = ""
    tags: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    rating: int | None = None
    regions: list[dict] = field(default_factory=list)


def _q(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def _bag(parent: ET.Element, prefix: str, tag: str, values: list[str]) -> None:
    node = ET.SubElement(parent, _q(prefix, tag))
    bag = ET.SubElement(node, _q("rdf", "Bag"))
    for v in values:
        ET.SubElement(bag, _q("rdf", "li")).text = v


def build(sc: Sidecar) -> bytes:
    root = ET.Element(_q("x", "xmpmeta"))
    rdf = ET.SubElement(root, _q("rdf", "RDF"))
    desc = ET.SubElement(rdf, _q("rdf", "Description"))
    desc.set(_q("rdf", "about"), "")

    if sc.caption:
        node = ET.SubElement(desc, _q("dc", "description"))
        alt = ET.SubElement(node, _q("rdf", "Alt"))
        li = ET.SubElement(alt, _q("rdf", "li"))
        li.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
        li.text = sc.caption
    if sc.tags:
        _bag(desc, "dc", "subject", sc.tags)
    if sc.rating is not None:
        ET.SubElement(desc, _q("xmp", "Rating")).text = str(sc.rating)

    if sc.regions:
        regions = ET.SubElement(desc, _q("mwg-rs", "Regions"))
        regions.set(_q("rdf", "parseType"), "Resource")
        dims = ET.SubElement(regions, _q("mwg-rs", "AppliedToDimensions"))
        dims.set(_q("rdf", "parseType"), "Resource")
        ET.SubElement(dims, _q("stDim", "w")).text = "1"
        ET.SubElement(dims, _q("stDim", "h")).text = "1"
        ET.SubElement(dims, _q("stDim", "unit")).text = "normalized"

        rlist = ET.SubElement(regions, _q("mwg-rs", "RegionList"))
        bag = ET.SubElement(rlist, _q("rdf", "Bag"))
        for r in sc.regions:
            li = ET.SubElement(bag, _q("rdf", "li"))
            li.set(_q("rdf", "parseType"), "Resource")
            ET.SubElement(li, _q("mwg-rs", "Name")).text = r["name"]
            ET.SubElement(li, _q("mwg-rs", "Type")).text = "Face"
            area = ET.SubElement(li, _q("mwg-rs", "Area"))
            area.set(_q("rdf", "parseType"), "Resource")
            # MWG areas are centre-based, not top-left based. Getting this wrong
            # puts every face box down and to the right in other applications.
            for key in ("x", "y", "w", "h"):
                ET.SubElement(area, _q("stArea", key)).text = f"{r[key]:.6f}"
            ET.SubElement(area, _q("stArea", "unit")).text = "normalized"

    ET.indent(root, space="  ")
    return (b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            + ET.tostring(root, encoding="utf-8")
            + b'\n<?xpacket end="w"?>\n')


def parse(data: bytes) -> Sidecar:
    sc = Sidecar()
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return sc

    for node in root.iter(_q("dc", "description")):
        for li in node.iter(_q("rdf", "li")):
            if li.text:
                sc.caption = li.text.strip()
                break
    for node in root.iter(_q("dc", "subject")):
        for li in node.iter(_q("rdf", "li")):
            if li.text and li.text.strip():
                sc.tags.append(li.text.strip())
    for node in root.iter(_q("xmp", "Rating")):
        if node.text and node.text.strip().isdigit():
            sc.rating = int(node.text.strip())
    for node in root.iter(_q("mwg-rs", "Name")):
        if node.text and node.text.strip():
            sc.people.append(node.text.strip())
    return sc


# ------------------------------------------------------------------- database
def collect(conn: sqlite3.Connection, photo_id: int) -> Sidecar:
    """Everything known about a photo, in sidecar shape."""
    ann = conn.execute(
        "SELECT caption FROM annotation a WHERE photo_id=? "
        f"ORDER BY {ANNOTATION_ORDER} LIMIT 1", (photo_id,)).fetchone()
    tags = [r["name"] for r in conn.execute(
        """SELECT t.name FROM tag t JOIN photo_tag pt ON pt.tag_id=t.id
           WHERE pt.photo_id=? ORDER BY pt.source DESC, t.name""", (photo_id,))]
    photo = conn.execute("SELECT rating FROM photo WHERE id=?", (photo_id,)).fetchone()

    regions, people = [], []
    for f in conn.execute(
            """SELECT f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, f.src_w, f.src_h, pe.name
               FROM face f JOIN person pe ON pe.id=f.person_id
               WHERE f.photo_id=? AND f.rejected=0 AND pe.name IS NOT NULL""",
            (photo_id,)):
        if not f["src_w"] or not f["src_h"]:
            continue  # box has no known coordinate space; cannot be normalised
        people.append(f["name"])
        regions.append({
            "name": f["name"],
            "x": (f["bbox_x"] + f["bbox_w"] / 2) / f["src_w"],
            "y": (f["bbox_y"] + f["bbox_h"] / 2) / f["src_h"],
            "w": f["bbox_w"] / f["src_w"],
            "h": f["bbox_h"] / f["src_h"],
        })

    return Sidecar(caption=(ann["caption"] if ann else "") or "", tags=tags,
                   people=sorted(set(people)),
                   rating=photo["rating"] if photo and photo["rating"] else None,
                   regions=regions)


def write(conn: sqlite3.Connection, photo_id: int, photo_path: Path,
          cfg=None) -> Path | None:
    sc = collect(conn, photo_id)
    if not (sc.caption or sc.tags or sc.regions):
        return None
    target = (resolve_path(cfg, conn, photo_id, photo_path) if cfg is not None
              else sidecar_path(photo_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(build(sc))
    tmp.replace(target)  # atomic: never leave a half-written sidecar
    return target


_TAG_SPLIT = re.compile(r"[,;|]")


def read_into(conn: sqlite3.Connection, photo_id: int, photo_path: Path,
              cfg=None) -> Sidecar | None:
    """Import a sidecar's keywords as manual tags. Deliberately does not touch
    captions or faces: those are the pipeline's output, and an import should add
    the user's own vocabulary, not overwrite what the models produced."""
    from pa.db import repo

    candidates = [sidecar_path(photo_path)]
    if cfg is not None:
        # Look in the app directory too, so a library that exported there can be
        # re-imported without moving anything.
        candidates.insert(0, resolve_path(cfg, conn, photo_id, photo_path))
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        return None
    sc = parse(path.read_bytes())
    tags = []
    for raw in sc.tags:
        tags.extend(t.strip() for t in _TAG_SPLIT.split(raw) if t.strip())
    if tags:
        existing = {r["name"] for r in conn.execute(
            """SELECT t.name FROM tag t JOIN photo_tag pt ON pt.tag_id=t.id
               WHERE pt.photo_id=? AND pt.source='manual'""", (photo_id,))}
        repo.set_tags(conn, photo_id, sorted(existing | {t.lower() for t in tags}),
                      source="manual")
        repo.reindex_fts(conn, photo_id)
    if sc.rating is not None:
        conn.execute("UPDATE photo SET rating=? WHERE id=?", (sc.rating, photo_id))
    return sc
