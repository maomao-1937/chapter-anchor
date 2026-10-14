"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chapter_anchor.core import InputError, parse_chapters, parse_srt, review, to_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Review draft chapter titles against an SRT transcript locally")
    parser.add_argument("transcript", type=Path, help="Your SRT captions")
    parser.add_argument("chapters", type=Path, help="One 'M:SS Title' line per chapter")
    parser.add_argument("--out", type=Path, help="Write a Markdown review report")
    parser.add_argument("--json-out", type=Path, help="Write machine-readable review report")
    parser.add_argument("--min-other", type=float, default=0.40)
    parser.add_argument("--margin", type=float, default=0.12)
    args = parser.parse_args()
    try:
        cues = parse_srt(args.transcript.read_text(encoding="utf-8-sig"))
        chapters = parse_chapters(args.chapters.read_text(encoding="utf-8-sig"))
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        report = review(cues, chapters, encoder, args.min_other, args.margin)
        markdown = to_markdown(report)
        if args.out:
            args.out.write_text(markdown, encoding="utf-8")
        if args.json_out:
            args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(markdown)
        return 0
    except (InputError, OSError) as exc:
        print(f"chapter-anchor: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
