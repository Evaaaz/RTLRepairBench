#!/usr/bin/env python3
"""Check the AI for Chip Design poster-paper body limit and checklist.

The workshop poster track allows 3--4 pages of main text, excluding references
and appendices. The References heading may share a page with body text, so the
page containing the heading counts whenever non-empty text precedes it.

This implementation intentionally uses Poppler's ``pdftotext`` rather than a
third-party Python PDF package; ``pdftotext`` is already required for paper QA.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


def extract_pages(pdf: Path) -> list[list[str]]:
    executable = shutil.which("pdftotext")
    if executable is None:
        raise RuntimeError("pdftotext is required for page-limit QA")
    completed = subprocess.run(
        [executable, "-layout", str(pdf), "-"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise RuntimeError(f"pdftotext failed: {detail}")
    return [page.splitlines() for page in completed.stdout.split("\f") if page.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--minimum", type=int, default=3)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--require-checklist", action="store_true")
    args = parser.parse_args()

    if args.minimum <= 0 or args.limit < args.minimum:
        parser.error("require 0 < --minimum <= --limit")
    if not args.pdf.is_file():
        print(f"missing PDF: {args.pdf}", file=sys.stderr)
        return 2

    try:
        pages = extract_pages(args.pdf)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    ref_page = ref_line = None
    for page_index, lines in enumerate(pages):
        for line_index, line in enumerate(lines):
            # The anonymous workshop style enables line numbering, so Poppler
            # commonly extracts this as e.g. ``139   References``.
            if re.fullmatch(r"(?:\d+\s+)?References", line.strip()):
                ref_page, ref_line = page_index, line_index
                break
        if ref_page is not None:
            break
    if ref_page is None or ref_line is None:
        print(f"{args.pdf}: no standalone References heading", file=sys.stderr)
        return 1

    before = [line.strip() for line in pages[ref_page][:ref_line] if line.strip()]
    # Ignore a bare page number if References begins at the top of a fresh page.
    spill = [line for line in before if not line.isdigit()]
    main_pages = ref_page + 1 if spill else ref_page

    print(
        f"main text: {main_pages} pages (required {args.minimum}--{args.limit}); "
        f"References starts on page {ref_page + 1}"
    )
    if not args.minimum <= main_pages <= args.limit:
        print(
            f"body page count {main_pages} is outside {args.minimum}--{args.limit}",
            file=sys.stderr,
        )
        return 1

    if args.require_checklist:
        full_text = "\n".join(line for page in pages for line in page)
        if "NeurIPS Paper Checklist" not in full_text:
            print("NeurIPS Paper Checklist is missing", file=sys.stderr)
            return 1
        print("NeurIPS Paper Checklist: present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
