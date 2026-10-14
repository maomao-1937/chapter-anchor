# ChapterAnchor

Review draft podcast/video chapter titles against your own timestamped SRT **before** publication. ChapterAnchor uses a real local sentence-embedding model to find when a title fits a different transcript span substantially better than its assigned span. It gives timestamped excerpts for human review; it never moves a chapter automatically.

This is for creators and editors who already have an SRT transcript and a proposed chapter list. It is not a transcript fetcher, chapter generator, media editor, YouTube/Apple integration, or correctness certificate.

## Install and try

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). From this directory:

```sh
uv sync --python 3.11 --extra test
uv run chapter-anchor examples/demo.srt examples/swapped-chapters.txt --out review.local.md --json-out review.local.json
uv run chapter-anchor examples/demo.srt examples/accurate-chapters.txt
uv run pytest -q
```

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0); later runs can use the cached model without a model API key. No transcript is uploaded by this tool. To prove an offline repeat run after the model is cached, set `HF_HUB_OFFLINE=1`.

The chapter file format is one `M:SS Title` or `H:MM:SS Title` per line, in increasing time order. The example SRT and chapter lists are **fictional test data**, not a real podcast. Inputs should be your own material; do not commit private transcripts to a public repository.

## What the report means

Each chapter gets one of three statuses:

- `review_location`: another chapter's caption text resembles the title enough more than the assigned span to merit review.
- `no_stronger_elsewhere`: no other span crossed the configured relative threshold. This does **not** establish the title is accurate.
- `no_caption_in_range`: the assigned time range contains no SRT cues.

The report includes a best excerpt/time from the assigned range and a best excerpt/time elsewhere. Similarity is cosine similarity, **not** calibrated confidence, and the defaults (`--min-other 0.40`, `--margin 0.12`) are heuristics for triage, not measured error rates. Listen to the audio or inspect the video before changing a title. A missing caption, code-switching, metaphorical heading, bad transcript, long topic drift, or title spanning several topics may produce false positives or misses. English is the intended language; other languages are unvalidated. The tool accepts SRT text only and does not check platform-specific chapter rules or actual media timing.

## Why this shape

Existing tools such as Kadenzo and SocialCal primarily generate chapters or validate formatting. This small, local CLI tests a different moment: an editor has a draft already and needs a source-linked second look without sending their transcript to another API. [RESEARCH.md](RESEARCH.md) records the actual sources, competitor counterevidence, and what remains hypothetical. No demand or adoption metrics are claimed.

MIT licensed. The third-party embedding model and its Apache-2.0 license are separate from this project's code license.
