-- photo_anotater schema. All tables STRICT: SQLite's default type affinity would
-- happily store 'banana' in an INTEGER column, and this DB is the source of truth.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL) STRICT;

-- ---------------------------------------------------------------- volumes/roots
-- A volume is a physical drive, identified by filesystem UUID rather than mount
-- point: external drives remount at different letters/paths, and keying on path
-- would turn a replugged HDD into thousands of spurious "deleted" photos.
CREATE TABLE IF NOT EXISTS volume (
    id              INTEGER PRIMARY KEY,
    uuid            TEXT NOT NULL UNIQUE,
    label           TEXT NOT NULL,
    last_mount      TEXT,
    last_seen_at    INTEGER,
    online          INTEGER NOT NULL DEFAULT 0
) STRICT;

CREATE TABLE IF NOT EXISTS root (
    id              INTEGER PRIMARY KEY,
    volume_id       INTEGER NOT NULL REFERENCES volume(id) ON DELETE CASCADE,
    rel_path        TEXT NOT NULL,          -- path relative to the volume mount point
    label           TEXT,
    exclude_globs   TEXT,                   -- JSON array
    enabled         INTEGER NOT NULL DEFAULT 1,
    added_at        INTEGER NOT NULL,
    last_scan_at    INTEGER,
    UNIQUE (volume_id, rel_path)
) STRICT;

-- ---------------------------------------------------------------------- photos
-- Identity is the content hash, not the path. One photo may exist as many files
-- (internal drive + two backups); annotating it once covers every copy, and
-- duplicate detection falls out for free.
CREATE TABLE IF NOT EXISTS photo (
    id              INTEGER PRIMARY KEY,
    blake3          TEXT NOT NULL UNIQUE,
    phash           INTEGER,                -- 64-bit perceptual hash, near-dupe detection
    width           INTEGER,
    height          INTEGER,
    bytes           INTEGER,
    mime            TEXT,
    taken_at        INTEGER,                -- unix seconds, from EXIF where available
    taken_at_source TEXT,                   -- exif | filename | mtime
    tz_offset_min   INTEGER,
    camera_make     TEXT,
    camera_model    TEXT,
    lens            TEXT,
    iso             INTEGER,
    f_number        REAL,
    exposure_s      REAL,
    focal_len       REAL,
    orientation     INTEGER,
    gps_lat         REAL,
    gps_lon         REAL,
    place_name      TEXT,
    rating          INTEGER,                -- user star rating, 0-5
    favorite        INTEGER NOT NULL DEFAULT 0,
    hidden          INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_photo_taken   ON photo(taken_at DESC);
CREATE INDEX IF NOT EXISTS idx_photo_phash   ON photo(phash) WHERE phash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_photo_fav     ON photo(favorite) WHERE favorite = 1;

CREATE TABLE IF NOT EXISTS file (
    id              INTEGER PRIMARY KEY,
    photo_id        INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    root_id         INTEGER NOT NULL REFERENCES root(id) ON DELETE CASCADE,
    rel_path        TEXT NOT NULL,          -- relative to root, so roots can move
    filename        TEXT NOT NULL,
    mtime           INTEGER NOT NULL,
    size            INTEGER NOT NULL,
    state           TEXT NOT NULL DEFAULT 'present',  -- present | missing | offline
    seen_at         INTEGER NOT NULL,
    UNIQUE (root_id, rel_path)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_file_photo ON file(photo_id);
CREATE INDEX IF NOT EXISTS idx_file_root  ON file(root_id, state);

-- ----------------------------------------------------------------- annotations
-- Keyed by (photo, model) so a model upgrade can be backfilled and compared
-- rather than silently overwriting what the previous model said.
CREATE TABLE IF NOT EXISTS annotation (
    id              INTEGER PRIMARY KEY,
    photo_id        INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    caption         TEXT,
    scene           TEXT,
    setting         TEXT,
    people_count    INTEGER,
    ocr_text        TEXT,
    raw_json        TEXT,
    model           TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    UNIQUE (photo_id, model, model_version)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_annotation_photo ON annotation(photo_id);

-- ------------------------------------------------------------------------ tags
CREATE TABLE IF NOT EXISTS tag (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE COLLATE NOCASE,
    kind            TEXT NOT NULL DEFAULT 'ai',   -- ai | manual | auto
    created_at      INTEGER NOT NULL
) STRICT;

-- source='manual' always wins over 'ai': the UI filters on it, and a re-run of
-- the AI stage deletes only its own rows.
CREATE TABLE IF NOT EXISTS photo_tag (
    photo_id        INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    tag_id          INTEGER NOT NULL REFERENCES tag(id)   ON DELETE CASCADE,
    source          TEXT NOT NULL DEFAULT 'ai',
    confidence      REAL,
    created_at      INTEGER NOT NULL,
    PRIMARY KEY (photo_id, tag_id, source)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_phototag_tag ON photo_tag(tag_id);

-- ---------------------------------------------------------------------- people
CREATE TABLE IF NOT EXISTS person (
    id              INTEGER PRIMARY KEY,
    name            TEXT UNIQUE COLLATE NOCASE,
    cover_face_id   INTEGER,
    notes           TEXT,
    created_at      INTEGER NOT NULL
) STRICT;

-- `confirmed` is what separates a machine guess from Tony's decision.
-- Re-clustering may freely reshuffle unconfirmed faces as new photos arrive,
-- but must never touch a face the user has confirmed or reassigned by hand.
CREATE TABLE IF NOT EXISTS face (
    id              INTEGER PRIMARY KEY,
    photo_id        INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    person_id       INTEGER REFERENCES person(id) ON DELETE SET NULL,
    cluster_id      INTEGER,
    bbox_x          INTEGER NOT NULL,
    bbox_y          INTEGER NOT NULL,
    bbox_w          INTEGER NOT NULL,
    bbox_h          INTEGER NOT NULL,
    det_score       REAL NOT NULL,
    blur_score      REAL,
    src_w           INTEGER,        -- size of the image the detector ran on, so
    src_h           INTEGER,        -- boxes can be rescaled or normalised later
    embedding       BLOB NOT NULL,          -- float32 x512, source of truth; LanceDB is derived
    model           TEXT NOT NULL,
    confirmed       INTEGER NOT NULL DEFAULT 0,
    rejected        INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_face_photo   ON face(photo_id);
CREATE INDEX IF NOT EXISTS idx_face_person  ON face(person_id);
CREATE INDEX IF NOT EXISTS idx_face_cluster ON face(cluster_id) WHERE person_id IS NULL;

-- ------------------------------------------------------------------ embeddings
CREATE TABLE IF NOT EXISTS photo_embedding (
    photo_id        INTEGER PRIMARY KEY REFERENCES photo(id) ON DELETE CASCADE,
    embedding       BLOB NOT NULL,          -- float32 x dim
    model           TEXT NOT NULL,
    dim             INTEGER NOT NULL,
    created_at      INTEGER NOT NULL
) STRICT;

-- ---------------------------------------------------------------------- albums
CREATE TABLE IF NOT EXISTS album (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'manual',   -- manual | smart
    query           TEXT,                             -- for kind='smart'
    cover_photo_id  INTEGER REFERENCES photo(id) ON DELETE SET NULL,
    created_at      INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS album_photo (
    album_id        INTEGER NOT NULL REFERENCES album(id) ON DELETE CASCADE,
    photo_id        INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    added_at        INTEGER NOT NULL,
    PRIMARY KEY (album_id, photo_id)
) STRICT;

-- ------------------------------------------------------------------- job queue
-- Every pipeline stage is a row here. Keyed on (photo, stage) so re-running is
-- idempotent, and state survives a kill -9 mid-index.
CREATE TABLE IF NOT EXISTS job (
    id              INTEGER PRIMARY KEY,
    photo_id        INTEGER REFERENCES photo(id) ON DELETE CASCADE,
    stage           TEXT NOT NULL,          -- thumbs | embed | faces | caption
    state           TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed|skipped
    priority        INTEGER NOT NULL DEFAULT 100,     -- lower runs first
    attempts        INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    model_version   TEXT,
    created_at      INTEGER NOT NULL,
    started_at      INTEGER,
    finished_at     INTEGER,
    UNIQUE (photo_id, stage)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_job_queue ON job(stage, state, priority, id);

-- ---------------------------------------------------------------------- search
-- A regular (not contentless) FTS5 table. Contentless would save ~100MB at 500k
-- photos, but it cannot UPDATE and needs the original column values to DELETE --
-- and this index is rewritten constantly as captions land, tags are edited and
-- faces get named. rowid is always photo.id, so re-indexing one photo is a
-- DELETE + INSERT on a known key.
CREATE VIRTUAL TABLE IF NOT EXISTS photo_fts USING fts5(
    caption, tags, ocr_text, people, filename, folder, place,
    tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS setting (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL
) STRICT;
