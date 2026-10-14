import numpy as np
import pytest

from chapter_anchor.core import InputError, parse_chapters, parse_srt, parse_time, review, to_markdown


SRT = """1
00:00:00,000 --> 00:00:03,000
Garden soil and tomato seedlings.

2
00:01:00,000 --> 00:01:03,000
Monthly budget and savings.

3
00:02:00,000 --> 00:02:03,000
Camera lens selection.
"""


class FakeEncoder:
    def encode(self, texts, **kwargs):
        def vector(text):
            lower = text.lower()
            return [float("garden" in lower), float("budget" in lower), float("camera" in lower)]
        rows = np.array([vector(text) for text in texts])
        return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1)


def test_timestamp_rejects_bad_components():
    assert parse_time("1:02:03,400") == 3723.4
    with pytest.raises(InputError):
        parse_time("1:65:03")
    with pytest.raises(InputError):
        parse_time("1:60")


def test_srt_parses_tags_and_bom_compatible_content():
    cues = parse_srt("1\n00:00:01,000 --> 00:00:02,000\n<i>Hello</i> world\n")
    assert cues[0].text == "Hello world"


def test_srt_rejects_bad_range():
    with pytest.raises(InputError, match="non-positive"):
        parse_srt("1\n00:00:02,000 --> 00:00:01,000\nOops\n")


def test_chapter_list_rejects_duplicate_time():
    with pytest.raises(InputError, match="increase"):
        parse_chapters("0:00 One\n0:00 Two\n")


def test_swapped_titles_are_reviewed_with_fake_encoder():
    cues = parse_srt(SRT)
    chapters = parse_chapters("0:00 Garden\n1:00 Camera\n2:00 Budget")
    report = review(cues, chapters, FakeEncoder())
    assert [item["status"] for item in report["chapters"]] == [
        "no_stronger_elsewhere", "review_location", "review_location"
    ]
    assert report["chapters"][1]["elsewhere_evidence"]["start"] == "00:02:00"


def test_correct_titles_have_no_stronger_elsewhere():
    report = review(parse_srt(SRT), parse_chapters("0:00 Garden\n1:00 Budget\n2:00 Camera"), FakeEncoder())
    assert all(item["status"] == "no_stronger_elsewhere" for item in report["chapters"])


def test_no_captions_in_chapter_range_is_explicit():
    report = review(parse_srt(SRT), parse_chapters("0:00 Garden\n0:30 Empty\n1:00 Budget"), FakeEncoder())
    assert report["chapters"][1]["status"] == "no_caption_in_range"


def test_markdown_contains_source_times():
    report = review(parse_srt(SRT), parse_chapters("0:00 Garden\n1:00 Camera\n2:00 Budget"), FakeEncoder())
    assert "00:02:00" in to_markdown(report)
