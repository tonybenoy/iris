"""Split a natural-language query into structured filters plus free text.

`photos of Sarah in the mountains last summer` becomes
    filters: person=Sarah, date=2025-06-01..2025-08-31
    free text: "mountains"
so the structured part can hit indexed columns and only the genuinely fuzzy
remainder goes to FTS and the vector index.
"""
from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import datetime, timedelta

FIELD_RE = re.compile(r"\b(person|people|tag|camera|place|folder|filename|type|in|from)"
                      r":\s*(\"[^\"]+\"|'[^']+'|\S+)", re.I)

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}

STOPWORDS = {"photos", "photo", "pictures", "picture", "pics", "pic", "images",
             "image", "of", "the", "a", "an", "with", "and", "show", "me", "find",
             "in", "at", "on", "my", "all", "from", "taken", "some", "any"}

# Northern-hemisphere month spans. People say "last summer" far more often than
# they say a date range, and leaving it in the free text poisons the FTS query.
SEASONS = {"spring": (3, 5), "summer": (6, 8), "autumn": (9, 11),
           "fall": (9, 11), "winter": (12, 2)}


@dataclass
class Query:
    text: str = ""
    people: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    camera: str | None = None
    place: str | None = None
    folder: str | None = None
    filename: str | None = None
    date_from: int | None = None
    date_to: int | None = None
    favorite: bool = False
    has_faces: bool | None = None
    raw: str = ""


def _strip_quotes(v: str) -> str:
    return v[1:-1] if len(v) > 1 and v[0] in "\"'" and v[-1] == v[0] else v


def _year_range(y: int) -> tuple[int, int]:
    return (int(datetime(y, 1, 1).timestamp()),
            int(datetime(y, 12, 31, 23, 59, 59).timestamp()))


def _month_range(y: int, m: int) -> tuple[int, int]:
    last = monthrange(y, m)[1]
    return (int(datetime(y, m, 1).timestamp()),
            int(datetime(y, m, last, 23, 59, 59).timestamp()))


def _parse_dates(text: str, now: datetime) -> tuple[str, int | None, int | None]:
    """Pull relative and absolute date expressions out of the text."""
    lo = hi = None

    def consume(pattern: str) -> str | None:
        nonlocal text
        m = re.search(pattern, text, re.I)
        if m:
            text = (text[:m.start()] + " " + text[m.end():]).strip()
            return m.group(0)
        return None

    if consume(r"\btoday\b"):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        lo, hi = int(start.timestamp()), int(now.timestamp())
    elif consume(r"\byesterday\b"):
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        lo, hi = int(start.timestamp()), int((start + timedelta(days=1)).timestamp())
    elif m := consume(r"\blast (week|month|year)\b"):
        unit = m.split()[-1].lower()
        days = {"week": 7, "month": 31, "year": 365}[unit]
        lo, hi = int((now - timedelta(days=days)).timestamp()), int(now.timestamp())
    elif m := consume(r"\bpast (\d+) (day|days|week|weeks|month|months|year|years)\b"):
        n = int(re.search(r"\d+", m).group())
        unit = m.split()[-1].rstrip("s")
        days = n * {"day": 1, "week": 7, "month": 31, "year": 365}[unit]
        lo, hi = int((now - timedelta(days=days)).timestamp()), int(now.timestamp())

    if lo is None:
        season_re = r"\b(?:(last|this)\s+)?(" + "|".join(SEASONS) + r")(?:\s+(20\d{2}))?\b"
        if m := consume(season_re):
            parts = re.match(season_re, m, re.I)
            qualifier, season, year_txt = parts.group(1), parts.group(2).lower(), parts.group(3)
            start_m, end_m = SEASONS[season]
            year = int(year_txt) if year_txt else now.year
            if not year_txt and (qualifier or "").lower() == "last":
                year -= 1
            if start_m > end_m:  # winter spans the new year
                lo = _month_range(year, start_m)[0]
                hi = _month_range(year + 1, end_m)[1]
            else:
                lo = _month_range(year, start_m)[0]
                hi = _month_range(year, end_m)[1]

    if lo is None:
        if m := consume(r"\b(" + "|".join(MONTHS) + r")\s+(20\d{2})\b"):
            parts = m.split()
            lo, hi = _month_range(int(parts[1]), MONTHS[parts[0].lower()])
        elif m := consume(r"\b(20\d{2})\b"):
            lo, hi = _year_range(int(m))
        elif m := consume(r"\b(" + "|".join(MONTHS) + r")\b"):
            month = MONTHS[m.lower()]
            year = now.year if month <= now.month else now.year - 1
            lo, hi = _month_range(year, month)
    return text, lo, hi


def parse(raw: str, now: datetime | None = None) -> Query:
    now = now or datetime.now()
    q = Query(raw=raw)
    text = raw

    for m in FIELD_RE.finditer(raw):
        key, value = m.group(1).lower(), _strip_quotes(m.group(2))
        if key in ("person", "people"):
            q.people.append(value)
        elif key == "tag":
            q.tags.append(value.lower())
        elif key == "camera":
            q.camera = value
        elif key in ("place", "in", "from"):
            q.place = value
        elif key == "folder":
            q.folder = value
        elif key == "filename":
            q.filename = value
        text = text.replace(m.group(0), " ")

    if re.search(r"\b(favou?rites?|starred)\b", text, re.I):
        q.favorite = True
        text = re.sub(r"\b(favou?rites?|starred)\b", " ", text, flags=re.I)

    text, q.date_from, q.date_to = _parse_dates(text, now)

    words = [w for w in re.split(r"\s+", text) if w and w.lower() not in STOPWORDS]
    q.text = " ".join(words).strip()
    return q
