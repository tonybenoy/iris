-- Face bounding boxes are stored in the pixel coordinates of whatever image the
-- detector actually ran on -- the cached 'view' thumbnail, not the original.
-- Without recording that image's size, the boxes cannot be converted to any
-- other coordinate space: crops break if the thumbnail size ever changes, and
-- XMP sidecars (which require normalised 0-1 coordinates) are impossible.
ALTER TABLE face ADD COLUMN src_w INTEGER;
ALTER TABLE face ADD COLUMN src_h INTEGER;
