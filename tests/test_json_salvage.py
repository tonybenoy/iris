import json

import pytest

from pa.providers.lmstudio import _loads_lenient


def test_clean_json():
    assert _loads_lenient('{"caption": "a cat", "tags": ["cat"]}')["caption"] == "a cat"


def test_truncated_mid_string_keeps_earlier_fields():
    truncated = '{"caption": "a terminal", "tags": ["code"], "ocr_text": "rustc --vers'
    got = _loads_lenient(truncated)
    assert got["caption"] == "a terminal"
    assert got["tags"] == ["code"]
    assert got["ocr_text"].startswith("rustc")


def test_truncated_mid_array():
    got = _loads_lenient('{"caption": "x", "tags": ["a", "b"')
    assert got["tags"] == ["a", "b"]


def test_bad_unicode_escape():
    got = _loads_lenient(r'{"caption": "price \u20 euros", "tags": []}')
    assert "price" in got["caption"]


def test_unsalvageable_still_raises():
    with pytest.raises(json.JSONDecodeError):
        _loads_lenient("not json at all <<<")
