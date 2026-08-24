"""Iris - local AI photo index and search."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pa.config import get_config
from pa.db import repo
from pa.db.connection import init_db
from pa.ingest import scanner, volumes

OFFLINE_SQL = "SELECT COUNT(*) FROM file WHERE state='offline'"

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Iris - find any photo you own, by what is in it.")

root_app = typer.Typer(no_args_is_help=True, help="Manage indexed folders.")
app.add_typer(root_app, name="root")
console = Console()


def _db():
    cfg = get_config()
    return init_db(cfg.paths.db_path), cfg


@app.command()
def init() -> None:
    """Create the database and show where everything lives."""
    conn, cfg = _db()
    console.print(f"[green]database[/]  {cfg.paths.db_path}")
    console.print(f"[green]thumbs[/]    {cfg.paths.thumbs_dir}")
    console.print(f"[green]vectors[/]   {cfg.paths.vectors_dir}")
    console.print(f"[green]caption[/]   {cfg.caption.model} @ {cfg.caption.base_url}")
    conn.close()


@root_app.command("add")
def root_add(
    path: Path = typer.Argument(..., help="Folder to index (any drive)."),
    label: str = typer.Option(None, "--label", "-l"),
    exclude: list[str] = typer.Option([], "--exclude", "-x", help="Glob to skip."),
    scan: bool = typer.Option(True, help="Scan immediately after adding."),
) -> None:
    """Register a folder. Stored against the drive's UUID, so it survives remounts."""
    path = path.expanduser().resolve()
    if not path.is_dir():
        console.print(f"[red]not a directory:[/] {path}")
        raise typer.Exit(1)
    conn, cfg = _db()
    vol_uuid, vol_label, mount = volumes.identify(path)
    rel = volumes.relative_to_mount(path, mount)
    volume_id = repo.upsert_volume(conn, vol_uuid, vol_label, str(mount.mountpoint))
    root_id = repo.add_root(conn, volume_id, rel, label or path.name, list(exclude))
    conn.commit()
    console.print(f"[green]added root[/] {path}")
    console.print(f"  drive [cyan]{vol_label}[/] ({vol_uuid})  rel=[dim]{rel or '/'}[/]")
    if scan:
        _scan_rows(conn, cfg, [r for r in repo.list_roots(conn) if r["id"] == root_id])
    conn.close()


@root_app.command("remove")
def root_remove(
    root_id: int = typer.Argument(..., help="Root id, from `pa root list`."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Stop indexing a folder. Your photos on disk are not touched."""
    conn, _ = _db()
    row = conn.execute(
        """SELECT r.*, v.label AS volume_label,
                  (SELECT COUNT(*) FROM file f WHERE f.root_id=r.id) n
           FROM root r JOIN volume v ON v.id=r.volume_id WHERE r.id=?""",
        (root_id,)).fetchone()
    if row is None:
        console.print(f"[red]no root with id {root_id}[/]")
        raise typer.Exit(1)
    console.print(f"[bold]{row['label'] or row['rel_path']}[/] on {row['volume_label']} "
                  f"- {row['n']} indexed files")
    if not yes and not typer.confirm("Remove this folder from the library?"):
        raise typer.Exit()
    files, orphans = repo.remove_root(conn, root_id)
    conn.commit()
    console.print(f"removed [green]{files}[/] file records; "
                  f"[yellow]{orphans}[/] photos had no other location and were dropped")
    console.print("[dim]nothing on disk was changed[/]")
    conn.close()


@app.command()
def prune(
    keep_missing: bool = typer.Option(
        False, "--keep-missing",
        help="Keep photos whose files were deleted (only drop removed folders)."),
) -> None:
    """Drop photos that no longer exist in any indexed folder."""
    conn, _ = _db()
    n = repo.prune_orphans(conn, drop_missing=not keep_missing)
    conn.commit()
    console.print(f"dropped [green]{n}[/] photos with no remaining file"
                  if n else "[green]nothing to prune[/]")
    conn.close()


@root_app.command("list")
def root_list() -> None:
    """Show every indexed folder and whether its drive is attached."""
    conn, _ = _db()
    rows = repo.list_roots(conn)
    if not rows:
        console.print("[yellow]no roots yet[/] - add one with: pa root add <folder>")
        raise typer.Exit()
    table = Table("id", "label", "drive", "path", "photos", "status")
    for r in rows:
        online = volumes.current_mountpoint(r["volume_uuid"]) is not None
        n = conn.execute("SELECT COUNT(DISTINCT photo_id) n FROM file WHERE root_id=?",
                         (r["id"],)).fetchone()["n"]
        table.add_row(str(r["id"]), r["label"] or "-", r["volume_label"],
                      r["rel_path"] or "/", str(n),
                      "[green]online[/]" if online else "[yellow]offline[/]")
    console.print(table)
    conn.close()


def _vector_search(cfg):
    """Bring up the vector index if one has been built, else None.

    Without this the CLI silently runs in degraded (FTS-only) mode, where a query
    nothing matches falls back to the filters and returns the whole library --
    the API and the CLI would then answer the same question differently.
    """
    from pa.search.vectors import VectorIndex

    index = VectorIndex(cfg.paths.vectors_dir, cfg.embed.dim, cfg.embed.model)
    if index.count() == 0:
        return None
    from pa.providers.registry import get_image_embedder

    embedder = get_image_embedder(cfg)

    def go(text: str, n: int) -> list[int]:
        return index.search(embedder.embed_texts([text])[0], n,
                            min_score=cfg.embed.min_score,
                            rel_score=cfg.embed.rel_score)
    return go


def _scan_rows(conn, cfg, rows, recursive=True, subdir=None, force=False) -> None:
    for r in rows:
        console.print(f"[bold]scanning[/] {r['label'] or r['rel_path']} "
                      f"on {r['volume_label']}")
        with console.status("walking...") as status:
            def progress(s, status=status):
                status.update(f"seen {s.seen}  new {s.new_photos}  "
                              f"dup {s.duplicates}  skip {s.skipped}")
            st = scanner.scan_root(conn, r, cfg, recursive=recursive,
                                   subdir=subdir, force=force, on_progress=progress)
        console.print(f"  seen [cyan]{st.seen}[/]  new photos [green]{st.new_photos}[/]  "
                      f"duplicates [magenta]{st.duplicates}[/]  "
                      f"unchanged [dim]{st.skipped}[/]  missing [yellow]{st.missing}[/]")
        for err in st.errors[:5]:
            console.print(f"  [red]error[/] {err}")
        if len(st.errors) > 5:
            console.print(f"  [red]...and {len(st.errors) - 5} more errors[/]")


@app.command()
def scan(
    root_id: int = typer.Option(None, "--root", help="Only this root id."),
    force: bool = typer.Option(False, "--force", help="Re-hash unchanged files."),
) -> None:
    """Scan registered folders for new, changed and deleted photos."""
    conn, cfg = _db()
    rows = [r for r in repo.list_roots(conn) if root_id is None or r["id"] == root_id]
    if not rows:
        console.print("[yellow]no matching roots[/]")
        raise typer.Exit(1)
    _scan_rows(conn, cfg, rows, force=force)
    conn.close()


@app.command()
def process(
    stage: str = typer.Option("all", "--stage", "-s",
                              help="thumbs | caption | all"),
    limit: int = typer.Option(1000, "--limit", "-n"),
) -> None:
    """Run pending pipeline work."""
    from pa.jobs import worker
    conn, cfg = _db()
    requeued = repo.requeue_stale(conn)
    if requeued:
        console.print(f"[dim]requeued {requeued} jobs left running by a previous run[/]")

    stages = ["thumbs", "embed", "faces", "caption"] if stage == "all" else [stage]

    if "thumbs" in stages:
        with console.status("thumbnails...") as st:
            res = worker.run_thumbs(conn, cfg, limit,
                              lambda r: st.update(f"thumbnails: {r.done} done, {r.failed} failed"))
        console.print(f"[bold]thumbs[/]   done [green]{res.done}[/] "
                      f"failed [red]{res.failed}[/] deferred [yellow]{res.skipped}[/]")
        for e in res.errors[:3]:
            console.print(f"  [red]{e}[/]")

    if "embed" in stages:
        from pa.providers.registry import get_image_embedder
        embedder = get_image_embedder(cfg)
        with console.status("embedding...") as st:
            res = worker.run_embed(conn, cfg, embedder, limit,
                       lambda r: st.update(f"embedding: {r.done} done, {r.failed} failed"))
        console.print(f"[bold]embed[/]    done [green]{res.done}[/] "
                      f"failed [red]{res.failed}[/] deferred [yellow]{res.skipped}[/]")
        for e in res.errors[:3]:
            console.print(f"  [red]{e}[/]")
        if res.done:
            from pa.search.vectors import VectorIndex
            n = VectorIndex(cfg.paths.vectors_dir, cfg.embed.dim, cfg.embed.model).build(conn)
            console.print(f"[dim]vector index rebuilt: {n} vectors[/]")

    if "faces" in stages:
        from pa.providers.registry import get_face_analyzer
        analyzer = get_face_analyzer(cfg)
        with console.status("detecting faces...") as st:
            res = worker.run_faces(conn, cfg, analyzer, limit,
                       lambda r: st.update(f"faces: {r.done} done, {r.failed} failed"))
        console.print(f"[bold]faces[/]    done [green]{res.done}[/] "
                      f"failed [red]{res.failed}[/] deferred [yellow]{res.skipped}[/]")
        for e in res.errors[:3]:
            console.print(f"  [red]{e}[/]")

    if "caption" in stages:
        from pa.providers.registry import get_captioner
        provider = get_captioner(cfg)
        ok, msg = provider.health()
        if not ok:
            console.print(f"[red]caption provider unavailable:[/] {msg}")
            raise typer.Exit(1)
        with console.status("captioning...") as st:
            res = worker.run_caption(conn, cfg, provider, limit,
                     lambda r: st.update(f"captioning: {r.done} done, {r.failed} failed"))
        console.print(f"[bold]caption[/]  done [green]{res.done}[/] "
                      f"failed [red]{res.failed}[/] deferred [yellow]{res.skipped}[/]")
        for e in res.errors[:3]:
            console.print(f"  [red]{e}[/]")
    conn.close()


@app.command("search")
def search_cmd(
    query: list[str] = typer.Argument(..., help="Natural language query."),
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """Search the library."""
    from datetime import datetime

    from pa.search.query import search
    conn, cfg = _db()
    raw = " ".join(query)
    hits = search(conn, raw, limit=limit, vector_search=_vector_search(cfg))
    if not hits:
        console.print(f"[yellow]no matches for[/] {raw!r}")
        raise typer.Exit()
    table = Table("#", "date", "file", "caption", "via")
    for i, h in enumerate(hits, 1):
        when = datetime.fromtimestamp(h.taken_at).strftime("%Y-%m-%d") if h.taken_at else "-"
        table.add_row(str(i), when, (h.filename or "-")[:30],
                      (h.caption or "[dim]no caption yet[/]")[:70], ",".join(h.sources))
    console.print(table)
    conn.close()


people_app = typer.Typer(no_args_is_help=True, help="Faces and people.")
app.add_typer(people_app, name="people")


@people_app.command("cluster")
def people_cluster() -> None:
    """Group detected faces into people, and match new faces to known people."""
    from pa.faces.cluster import recluster
    conn, cfg = _db()
    st = recluster(conn, cfg)
    console.print(f"matched to known people [green]{st.anchored}[/]")
    console.print(f"new clusters [cyan]{st.new_clusters}[/] "
                  f"covering [cyan]{st.clustered_faces}[/] faces")
    console.print(f"unassigned (too few similar faces) [dim]{st.unassigned}[/]")
    conn.close()


@people_app.command("list")
def people_list() -> None:
    """Named people and unnamed clusters awaiting a name."""
    conn, _ = _db()
    named = conn.execute(
        """SELECT pe.name, COUNT(f.id) n FROM person pe
           LEFT JOIN face f ON f.person_id=pe.id GROUP BY pe.id ORDER BY n DESC""").fetchall()
    clusters = conn.execute(
        """SELECT cluster_id, COUNT(*) n FROM face
           WHERE cluster_id IS NOT NULL AND person_id IS NULL
           GROUP BY cluster_id ORDER BY n DESC LIMIT 20""").fetchall()
    if named:
        t = Table("person", "photos")
        for r in named:
            t.add_row(r["name"] or "-", str(r["n"]))
        console.print(t)
    if clusters:
        t = Table("cluster", "faces", title="unnamed - name these in the web UI")
        for r in clusters:
            t.add_row(str(r["cluster_id"]), str(r["n"]))
        console.print(t)
    if not named and not clusters:
        console.print("[yellow]no faces clustered yet[/] - run: pa process --stage faces "
                      "then pa people cluster")
    conn.close()


@people_app.command("name")
def people_name(cluster_id: int, name: str) -> None:
    """Give a cluster a name, turning it into a person."""
    from pa.faces.cluster import name_cluster
    conn, _ = _db()
    n = name_cluster(conn, cluster_id, name)
    console.print(f"named [green]{n}[/] faces as [bold]{name}[/]")
    conn.close()


sidecar_app = typer.Typer(no_args_is_help=True, help="XMP sidecars next to your photos.")
app.add_typer(sidecar_app, name="sidecar")


@sidecar_app.command("export")
def sidecar_export(
    root_id: int = typer.Option(None, "--root", help="Only this root id."),
    limit: int = typer.Option(0, "--limit", "-n", help="0 = no limit."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Rewrite existing sidecars."),
    beside: bool = typer.Option(
        False, "--beside",
        help="Write next to each photo instead of inside the app directory."),
) -> None:
    """Write caption, tags and named faces to .xmp files.

    By default these go under the app's own directory, mirroring your folder
    tree, so your photo folders are left untouched. Use --beside to put them
    next to the originals, which is what Lightroom and digiKam read directly.
    """
    from pa.sidecar import bulk

    conn, cfg = _db()
    if beside:
        cfg = cfg.model_copy(deep=True)
        cfg.sidecar.location = "beside"
    where = "beside your photos" if beside else str(cfg.paths.sidecars_dir)
    with console.status("writing sidecars...") as st:
        res = bulk.export(conn, cfg, root_id, limit, overwrite,
                          on_progress=lambda r, i: st.update(
                              f"{i}/{r.total} - wrote {r.written}"))
    for err in res.errors[:5]:
        console.print(f"  [red]{err}[/]")
    console.print(f"wrote [green]{res.written}[/] sidecars, skipped [dim]{res.skipped}[/] "
                  f"existing, [yellow]{res.offline}[/] on disconnected drives")
    console.print(f"[dim]location: {where}[/]")
    conn.close()


@sidecar_app.command("import")
def sidecar_import(
    root_id: int = typer.Option(None, "--root", help="Only this root id."),
) -> None:
    """Read keywords and ratings from existing .xmp files into the library."""
    from pa.sidecar import bulk

    conn, cfg = _db()
    res = bulk.read_back(conn, cfg, root_id)
    console.print(f"read [green]{res.found}[/] sidecars, "
                  f"[cyan]{res.tags}[/] keywords imported")
    conn.close()


@app.command()
def duplicates(
    near: bool = typer.Option(False, "--near", help="Also find visually similar photos."),
    distance: int = typer.Option(6, "--distance", help="pHash bits for --near (0-64)."),
) -> None:
    """Photos stored more than once, and optionally near-identical shots."""
    conn, _ = _db()
    exact = conn.execute(
        """SELECT p.id, p.blake3, p.bytes, COUNT(f.id) n,
                  group_concat(v.label || ':' || f.rel_path, char(10)) locations
           FROM photo p JOIN file f ON f.photo_id=p.id
           JOIN root r ON r.id=f.root_id JOIN volume v ON v.id=r.volume_id
           GROUP BY p.id HAVING n > 1 ORDER BY p.bytes * (n - 1) DESC""").fetchall()
    if exact:
        wasted = sum(r["bytes"] * (r["n"] - 1) for r in exact if r["bytes"])
        t = Table("copies", "size", "locations", title="Identical files")
        for r in exact[:25]:
            locations = (r["locations"] or "").replace(chr(10), " | ")
            t.add_row(str(r["n"]), _human(r["bytes"]), locations)
        console.print(t)
        console.print(f"[bold]{len(exact)}[/] photos stored more than once, "
                      f"wasting [bold]{_human(wasted)}[/]")
    else:
        console.print("[green]no identical duplicates[/]")

    if near:
        groups = _near_duplicates(conn, distance)
        if groups:
            t = Table("photos", "example", title=f"Visually similar (within {distance} bits)")
            for g in groups[:25]:
                t.add_row(str(len(g["ids"])), g["example"])
            console.print(t)
            console.print(f"[bold]{len(groups)}[/] groups of near-identical photos")
        else:
            console.print("[green]no near-duplicates[/]")
    conn.close()


def _human(n: int | None) -> str:
    if not n:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def _near_duplicates(conn, max_distance: int) -> list[dict]:
    """Group photos whose perceptual hashes are within `max_distance` bits.

    Brute-force pairwise, but only over photos that HAVE a pHash and using
    numpy popcount over a packed array, so 500k photos is one vectorised pass
    rather than 125 billion Python-level comparisons.
    """
    import numpy as np

    rows = conn.execute(
        """SELECT p.id, p.phash, (SELECT filename FROM file f WHERE f.photo_id=p.id LIMIT 1) fn
           FROM photo p WHERE p.phash IS NOT NULL""").fetchall()
    if len(rows) < 2:
        return []
    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    names = {r["id"]: r["fn"] for r in rows}
    hashes = np.array([r["phash"] & 0xFFFFFFFFFFFFFFFF for r in rows], dtype=np.uint64)
    packed = hashes.view(np.uint8).reshape(len(hashes), 8)

    seen: set[int] = set()
    groups: list[dict] = []
    for i in range(len(ids)):
        if int(ids[i]) in seen:
            continue
        dist = np.unpackbits(packed[i] ^ packed, axis=1).sum(axis=1)
        hit = np.where(dist <= max_distance)[0]
        if len(hit) < 2:
            continue
        members = [int(ids[j]) for j in hit]
        seen.update(members)
        groups.append({"ids": members, "example": names.get(members[0]) or str(members[0])})
    return groups


config_app = typer.Typer(no_args_is_help=True, help="Settings: models, devices, thresholds.")
app.add_typer(config_app, name="config")


@config_app.command("path")
def config_path_cmd() -> None:
    """Print the config file location."""
    from pa.config import CONFIG_PATH
    console.print(str(CONFIG_PATH))
    if not CONFIG_PATH.exists():
        console.print("[dim]does not exist yet - create it with: pa config init[/]")


@config_app.command("init")
def config_init(
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file."),
) -> None:
    """Write a commented config file with every setting and its current value."""
    from pa.config import CONFIG_PATH, Config, render_default_toml
    if CONFIG_PATH.exists() and not force:
        console.print(f"[yellow]already exists:[/] {CONFIG_PATH}")
        console.print("[dim]use --force to overwrite, or: pa config edit[/]")
        raise typer.Exit(1)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(render_default_toml(Config.load()))
    console.print(f"[green]wrote[/] {CONFIG_PATH}")


@config_app.command("show")
def config_show(
    section: str = typer.Argument(None, help="caption | embed | face | thumbs | sidecar"),
) -> None:
    """Show the settings actually in effect."""
    from pa.config import CONFIG_PATH, get_config
    cfg = get_config()
    console.print(f"[dim]file: {CONFIG_PATH}"
                  f"{'' if CONFIG_PATH.exists() else ' (not created - using defaults)'}[/]\n")
    sections = {"caption": cfg.caption, "embed": cfg.embed, "face": cfg.face,
                "thumbs": cfg.thumbs, "sidecar": cfg.sidecar}
    if section:
        if section not in sections:
            console.print(f"[red]unknown section {section!r}[/] - "
                          f"try one of: {', '.join(sections)}")
            raise typer.Exit(1)
        sections = {section: sections[section]}
    for name, model in sections.items():
        t = Table("setting", "value", title=f"[{name}]", title_justify="left")
        for k, v in model.model_dump().items():
            t.add_row(k, str(v))
        console.print(t)
    if not section:
        t = Table("setting", "value", title="[server]", title_justify="left")
        for k, v in (("host", cfg.host), ("port", cfg.port),
                     ("scan_workers", cfg.scan_workers)):
            t.add_row(k, str(v))
        console.print(t)
        console.print(f"[dim]database: {cfg.paths.db_path}[/]")
        console.print(f"[dim]sidecars: {cfg.paths.sidecars_dir}[/]")


@config_app.command("edit")
def config_edit() -> None:
    """Open the config file in your editor, creating it first if needed."""
    import os
    import shutil
    import subprocess

    from pa.config import CONFIG_PATH, Config, render_default_toml
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(render_default_toml(Config.load()))
        console.print(f"[green]created[/] {CONFIG_PATH}")
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        editor = next((e for e in ("nano", "vim", "vi", "notepad.exe")
                       if shutil.which(e)), None)
    if not editor:
        console.print(f"[yellow]no editor found.[/] Edit it yourself: {CONFIG_PATH}")
        raise typer.Exit(1)
    subprocess.call([editor, str(CONFIG_PATH)])


@config_app.command("check")
def config_check() -> None:
    """Test that the configured models are actually reachable."""
    from pa.config import get_config
    from pa.providers import health

    rep = health.report(get_config())
    badge = {"ok": "[green]OK  [/]", "warn": "[yellow]WARN[/]", "fail": "[red]FAIL[/]"}
    for check in rep["checks"]:
        console.print(f"{badge[check['level']]} {check['id']:<9} {check['detail']}")
        if check["message"]:
            console.print(f"       [dim]{check['message']}[/]")
        if check["hint"]:
            console.print(f"       [yellow]{check['hint']}[/]")
            console.print("       [dim]pa config edit  ->  [caption] base_url[/]")
    raise typer.Exit(0 if rep["ok"] else 1)


@app.command()
def serve(
    host: str = typer.Option(None, "--host"),
    port: int = typer.Option(None, "--port"),
) -> None:
    """Start the web app."""
    import uvicorn

    cfg = get_config()
    from pa.api.app import create_app
    console.print(f"[green]Iris[/] http://{host or cfg.host}:{port or cfg.port}")
    uvicorn.run(create_app(), host=host or cfg.host, port=port or cfg.port, log_level="warning")


@app.command()
def status() -> None:
    """Library counts and pipeline progress."""
    conn, cfg = _db()
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    console.print(f"[bold]photos[/]   {q('SELECT COUNT(*) FROM photo')}")
    console.print(f"[bold]files[/]    {q('SELECT COUNT(*) FROM file')} "
                  f"([dim]{q(OFFLINE_SQL)} offline[/])")
    console.print(f"[bold]people[/]   {q('SELECT COUNT(*) FROM person')}")
    console.print(f"[bold]faces[/]    {q('SELECT COUNT(*) FROM face')}")
    console.print(f"[bold]tags[/]     {q('SELECT COUNT(*) FROM tag')}")
    stats = repo.job_stats(conn)
    if stats:
        table = Table("stage", "pending", "running", "done", "failed")
        for stage in repo.STAGES:
            s = stats.get(stage, {})
            table.add_row(stage, str(s.get("pending", 0)), str(s.get("running", 0)),
                          str(s.get("done", 0)), str(s.get("failed", 0)))
        console.print(table)
    conn.close()


if __name__ == "__main__":
    app()
