from datetime import datetime

import pytest

from pa.ingest.exif import date_from_filename


@pytest.mark.parametrize("name,expected", [
    ("IMG20260401195853.jpg",            datetime(2026, 4, 1, 19, 58, 53)),
    ("IMG_20260412_134132.jpg",          datetime(2026, 4, 12, 13, 41, 32)),
    ("Screenshot 2026-08-21 132906.png", datetime(2026, 8, 21, 13, 29, 6)),
    ("2026-08-08 20.43.42.jpg",          datetime(2026, 8, 8, 20, 43, 42)),
    ("PXL_20240115_093000123.jpg",       datetime(2024, 1, 15, 9, 30, 0)),
    ("VID_20231225_180000.mp4",          datetime(2023, 12, 25, 18, 0, 0)),
    ("2022-07-04.jpg",                   datetime(2022, 7, 4, 0, 0, 0)),
    ("20220704.jpg",                     datetime(2022, 7, 4, 0, 0, 0)),
])
def test_parses(name, expected):
    assert date_from_filename(name) == expected


@pytest.mark.parametrize("name", [
    "holiday.jpg",
    "DSC_0042.jpg",
    "IMG_1234.jpg",
    "1999-01-01.jpg",          # before 2000: rejected as implausible
    "2026-13-45.jpg",          # impossible month/day
    "screenshot.png",
])
def test_rejects(name):
    assert date_from_filename(name) is None
