"""Parsing and local-model comparison for chapter/title alignment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class InputError(ValueError):
    """Human-readable input error."""


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Chapter:
    start: float
    title: str


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    text: str


class Encoder(Protocol):
    def encode(self, texts: list[str], **kwargs: object) -> object: ...


TIME_RE = re.compile(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[,.](\d{1,3}))?$")
SRT_RANGE_RE = re.compile(r"^(\S+)\s+-->\s+(\S+)(?:\s+.*)?$")
CHAPTER_RE = re.compile(r"^\[?((?:\d{1,2}:)?\d{1,2}:\d{2})\]?\s*(?:[-–—]\s*)?(.+)$")
TAG_RE = re.compile(r"<[^>]*>")


def parse_time(value: str) -> float:
    match = TIME_RE.fullmatch(value.strip())
    if not match:
        raise InputError(f"Invalid timestamp: {value!r}")
    hours_text, minutes_text, seconds_text, millis_text = match.groups()
    hours = int(hours_text or 0)
    minutes = int(minutes_text)
    seconds = int(seconds_text)
    if minutes > 59 and hours_text is not None or seconds > 59:
        raise InputError(f"Timestamp component out of range: {value!r}")
    fraction = float(f"0.{millis_text.ljust(3, '0')}") if millis_text else 0.0
    return hours * 3600 + minutes * 60 + seconds + fraction


def format_time(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def parse_srt(content: str) -> list[Cue]:
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip())
    cues: list[Cue] = []
    for index, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].isdigit():
            lines = lines[1:]
        if len(lines) < 2:
            raise InputError(f"SRT block {index}: expected timestamp range and text")
        range_match = SRT_RANGE_RE.fullmatch(lines[0])
        if not range_match:
            raise InputError(f"SRT block {index}: invalid timestamp range")
        start, end = (parse_time(part) for part in range_match.groups())
        text = " ".join(TAG_RE.sub("", line).strip() for line in lines[1:]).strip()
        if end <= start or not text:
            raise InputError(f"SRT block {index}: empty text or non-positive duration")
        if cues and start < cues[-1].start:
            raise InputError(f"SRT block {index}: cues are not in time order")
        cues.append(Cue(start, end, text))
    if not cues:
        raise InputError("SRT has no caption cues")
    return cues


def parse_chapters(content: str) -> list[Chapter]:
    chapters: list[Chapter] = []
    for line_number, raw in enumerate(content.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        match = CHAPTER_RE.fullmatch(line)
        if not match:
            raise InputError(f"Chapter line {line_number}: expected 'M:SS Title'")
        start = parse_time(match.group(1))
        title = match.group(2).strip()
        if not title:
            raise InputError(f"Chapter line {line_number}: title is empty")
        if chapters and start <= chapters[-1].start:
            raise InputError(f"Chapter line {line_number}: timestamps must increase")
        chapters.append(Chapter(start, title))
    if not chapters:
        raise InputError("Chapter list is empty")
    return chapters


def _windows_for_span(cues: list[Cue], start: float, end: float) -> list[Window]:
    selected = [cue for cue in cues if cue.start < end and cue.end > start]
    windows: list[Window] = []
    batch: list[Cue] = []
    word_count = 0
    for cue in selected:
        batch.append(cue)
        word_count += len(cue.text.split())
        if word_count >= 65 or len(batch) >= 8:
            windows.append(Window(batch[0].start, batch[-1].end, " ".join(item.text for item in batch)))
            batch = batch[-1:] if len(batch) > 1 else []
            word_count = len(batch[0].text.split()) if batch else 0
    if batch and (not windows or batch[-1].end > windows[-1].end):
        windows.append(Window(batch[0].start, batch[-1].end, " ".join(item.text for item in batch)))
    return windows


def review(cues: list[Cue], chapters: list[Chapter], encoder: Encoder,
           min_other: float = 0.40, margin: float = 0.12) -> dict[str, object]:
    if not 0 <= min_other <= 1 or not 0 <= margin <= 1:
        raise InputError("Score thresholds must be between 0 and 1")
    end_time = max(cue.end for cue in cues)
    if chapters[0].start > end_time:
        raise InputError("All chapters start after the transcript ends")
    chapter_windows = [
        _windows_for_span(cues, chapter.start,
                          chapters[index + 1].start if index + 1 < len(chapters) else end_time + 0.001)
        for index, chapter in enumerate(chapters)
    ]
    flat = [(chapter_index, window) for chapter_index, windows in enumerate(chapter_windows) for window in windows]
    if not flat:
        raise InputError("No caption text falls inside the chapter ranges")
    import numpy as np

    texts = [chapter.title for chapter in chapters] + [window.text for _, window in flat]
    vectors = np.asarray(encoder.encode(texts, normalize_embeddings=True), dtype=float)
    if vectors.shape != (len(texts), vectors.shape[-1]) or not np.isfinite(vectors).all():
        raise RuntimeError("Embedding model returned invalid vectors")
    scores = vectors[:len(chapters)] @ vectors[len(chapters):].T
    results: list[dict[str, object]] = []
    for index, chapter in enumerate(chapters):
        same = [(float(scores[index, pos]), window) for pos, (owner, window) in enumerate(flat) if owner == index]
        other = [(float(scores[index, pos]), owner, window) for pos, (owner, window) in enumerate(flat) if owner != index]
        best_same = max(same, key=lambda item: item[0]) if same else None
        best_other = max(other, key=lambda item: item[0]) if other else None
        own_score = best_same[0] if best_same else None
        other_score = best_other[0] if best_other else None
        if own_score is None:
            status = "no_caption_in_range"
        elif (best_other and other_score is not None and other_score >= min_other
              and other_score - own_score >= margin):
            status = "review_location"
        else:
            status = "no_stronger_elsewhere"
        results.append({
            "chapter": {"start": format_time(chapter.start), "title": chapter.title},
            "status": status,
            "own_best_similarity": round(own_score, 4) if own_score is not None else None,
            "own_evidence": _evidence(best_same[1]) if best_same else None,
            "elsewhere_best_similarity": round(other_score, 4) if other_score is not None else None,
            "elsewhere_chapter": best_other[1] + 1 if best_other else None,
            "elsewhere_evidence": _evidence(best_other[2]) if best_other else None,
        })
    return {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "method": "cosine similarity of chapter titles and transcript windows; not a calibrated error probability",
        "thresholds": {"min_other": min_other, "margin": margin},
        "chapters": results,
    }


def _evidence(window: Window) -> dict[str, str]:
    return {"start": format_time(window.start), "end": format_time(window.end),
            "excerpt": window.text[:240]}


def to_markdown(report: dict[str, object]) -> str:
    lines = ["# ChapterAnchor review", "", "This is a review queue, not proof of correctness or an automatic edit.", ""]
    for item in report["chapters"]:  # type: ignore[union-attr]
        chapter = item["chapter"]
        lines += [f"## {chapter['start']} {chapter['title']}", "",
                  f"Status: `{item['status']}`", "",
                  f"Best in assigned range: {item['own_best_similarity']} — {_display(item['own_evidence'])}", "",
                  f"Best elsewhere: {item['elsewhere_best_similarity']} "
                  f"(chapter {item['elsewhere_chapter']}) — {_display(item['elsewhere_evidence'])}", ""]
    lines += ["Embedding similarities are relative signals, not confidence scores. Listen to the source audio before editing chapters.", ""]
    return "\n".join(lines)


def _display(evidence: dict[str, str] | None) -> str:
    if not evidence:
        return "none"
    excerpt = evidence["excerpt"].replace("\n", " ").replace("`", "'")
    return f"{evidence['start']}–{evidence['end']} `{excerpt}`"
