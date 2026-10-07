#!/usr/bin/env python3
"""Re-verify an existing transformers manifest against the current policy.

Needed because the first harvest ran under a vocabulary matcher that used plain
substring tests. Short entries like `vit` and `lora` matched inside ordinary
words ("acti**vit**y", "exp**lora**tion"), which let unrelated high-citation
papers through -- *Array programming with NumPy* was admitted at 24k citations.
The matcher is now word-bounded, so anything admitted by the old one has to be
re-checked rather than trusted.

Downloads are never deleted: a rejected PDF is moved to
`.cache/transformers-rejected/` with its title recorded, so a policy change can
be revisited without re-fetching.

Run:  python3 scripts/tf_verify.py
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import harvest_transformers as T
import harvest_neuroevolution as H

ROOT = H.ROOT
PDF_DIR = ROOT / "papers" / "transformers"
REJECT = ROOT / ".cache" / "transformers-rejected"


def main() -> int:
    mf = PDF_DIR / "manifest.json"
    if not mf.exists():
        print("manifest.json missing; run the harvest first", file=sys.stderr)
        return 1
    rows = json.loads(mf.read_text())
    print(f"re-verifying {len(rows)} papers against the word-bounded policy")
    print(
        f"(IDs are preserved here; scripts/harvest_transformers.py renumbers "
        f"to the final citation order on its next write)"
    )

    REJECT.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    drop: list[str] = []

    def check(row: dict) -> tuple[dict, str, str]:
        path = PDF_DIR / row["fileName"]
        text = H.pdftotext(path)
        rel = T.relevance_of(
            {"title": row["title"], "abstract": row.get("abstract", "")}, text
        )
        # relevance_of needs the abstract for its own gate call; the stored row
        # has none, so fall back to the title-only path when it is missing.
        if rel == "off-topic" and not row.get("abstract"):
            rel = T.LLM_RELEVANCE(row["title"], (row["title"] + "\n" + text[:22000]).lower())
        return row, rel, text

    with ThreadPoolExecutor(max_workers=4) as ex:
        for row, rel, text in ex.map(check, rows):
            if rel == "transformers":
                continue
            drop.append(row["id"])
            src = PDF_DIR / row["fileName"]
            if src.exists():
                shutil.move(str(src), str(REJECT / row["fileName"]))
            reason = "too-thin" if rel == "too-thin" else rel
            notes.append(
                f"{row['fileName']}  [{reason}]  {row['citations']:>7}  {row['title']}"
            )

    kept = [r for r in rows if r["id"] not in set(drop)]

    # Collapse the same work filed under several OpenAlex IDs. This lives here
    # as well as in discover(), because a corpus assembled over several runs
    # can already contain duplicates that no single discovery pass saw together.
    by_title: dict[str, dict] = {}
    for r in kept:
        key = T.norm_title(r["title"])
        cur = by_title.get(key)
        if cur is None or r["citations"] > cur["citations"]:
            by_title[key] = r
    if len(by_title) != len(kept):
        dupes = [
            r
            for r in kept
            if by_title[T.norm_title(r["title"])] is not r
        ]
        for r in dupes:
            src = PDF_DIR / r["fileName"]
            if src.exists():
                shutil.move(str(src), str(REJECT / r["fileName"]))
            drop.append(r["id"])
            notes.append(
                f"{r['fileName']}  [duplicate-title]  {r['citations']:>7}  {r['title']}"
            )
        kept = [r for r in kept if r["id"] not in set(drop)]
        print(f"  removed {len(dupes)} duplicate-title records")

    # Repair OpenAlex publication_year from the arXiv id. OpenAlex takes the
    # year from whichever version it indexed last, so "Attention Is All You Need"
    # (arXiv 1706.03762, the 2017 Vaswani et al. paper) was labelled 2025 and
    # sorted as the second most cited paper *in the corpus*. The id encodes when
    # the work was actually posted.
    for r in kept:
        ay = T.arxiv_year(r.get("arxiv", ""))
        if ay and r.get("year") != ay:
            notes.append(
                f"{r['fileName']}  [year {r['year']} -> {ay} from arXiv "
                f"{r['arxiv']}]  {r['title']}"
            )
            r["year"] = ay

    # Apply the window *after* correcting years, not before. Two papers were
    # misdated upward by OpenAlex and only revealed themselves once the arXiv id
    # was applied: SegNet (arXiv 1511.00561) and a 2016 compression paper. They
    # were admitted on a year the corpus had decided was in range.
    for r in list(kept):
        if not (T.YEAR_MIN <= (r.get("year") or 0) <= T.YEAR_MAX):
            src = PDF_DIR / r["fileName"]
            if src.exists():
                shutil.move(str(src), str(REJECT / r["fileName"]))
            drop.append(r["id"])
            notes.append(
                f"{r['fileName']}  [outside {T.YEAR_MIN}-{T.YEAR_MAX} after year "
                f"repair: {r['year']}]  {r['title']}"
            )
            kept = [k for k in kept if k["id"] != r["id"]]
    if notes:
        print(f"  {len(notes)} corrections recorded")

    kept.sort(key=lambda r: (-r["citations"], -(r.get("year") or 0), r["title"]))
    for i, r in enumerate(kept, 1):
        r["id"] = f"T{i:03d}"

    mf.write_text(json.dumps(kept, indent=2) + "\n")

    print(f"kept {len(kept)}, rejected {len(drop)}")
    if notes:
        (REJECT / "REJECTED.txt").write_text(
            "\n".join(
                [
                    "Papers rejected by scripts/tf_verify.py",
                    "",
                    "Kept under .cache/transformers-rejected/ rather than deleted, so a",
                    "later policy change can re-admit them without re-downloading.",
                    "",
                    *notes,
                ]
            )
            + "\n"
        )
        for n in notes[:40]:
            print("  " + n)
        if len(notes) > 40:
            print(f"  ... and {len(notes) - 40} more, see {REJECT / 'REJECTED.txt'}")

    if kept and len(kept) < 400:
        print(
            f"\nNOTE: {len(kept)} papers is under the 400 target; "
            f"run scripts/harvest_transformers.py --extra to widen the pool"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())