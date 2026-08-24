-- Ignoring a stranger in a group photo has to be undoable, and undoing it is
-- only useful if it puts back the group that was dismissed rather than a heap
-- of loose faces. `rejected` records that a face was ignored; this records
-- which proposed cluster it was ignored as, so a restore can rebuild it.
--
-- Nullable on purpose: a single face ignored from the lightbox never belonged
-- to a group, and rows rejected before this column existed have no group to
-- remember.
ALTER TABLE face ADD COLUMN ignored_as INTEGER;

CREATE INDEX IF NOT EXISTS idx_face_ignored ON face(ignored_as) WHERE rejected = 1;
