#!/usr/bin/env python
"""Apply page_tweaks to jlens readout pages already written to disk.

collect_readouts.py applies the same tweaks to pages it renders fresh; this
script retrofits pages produced by earlier runs (which skip-by-pid won't
regenerate). Safe to re-run: the edit is idempotent.

Usage:
  python patch_pages.py                 # patches ./readouts/pages/*.html
  python patch_pages.py --out ~/readouts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import page_tweaks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "readouts"),
                   help="readouts dir whose pages/*.html get patched")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pages = sorted((Path(args.out).expanduser() / "pages").glob("*.html"))
    if not pages:
        raise SystemExit(f"no pages under {args.out}/pages/")
    changed = 0
    for p in pages:
        html = p.read_text(encoding="utf-8")
        new = page_tweaks.apply_tweaks(html)
        if new != html:
            p.write_text(new, encoding="utf-8")
            changed += 1
    print(f"[done] patched {changed}/{len(pages)} pages (already-tweaked ones skipped)")


if __name__ == "__main__":
    main()
