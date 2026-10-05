#!/usr/bin/env python3
"""Top the corpus back up to 300 human-sensing papers.

scripts/harvest_wifi.py gates relevance on *title text only*, which let through
network-throughput, age-of-information, drone and RF-security papers that are
not human-sensing at all. scripts/wifi_index.py classifies each downloaded PDF
from its own text and labels it human-sensing / adjacent / off-topic.

This script closes the loop: it drops everything that is not human-sensing and
walks down the remaining citation-ranked candidates (reusing the cached arXiv
responses, so no new API cost) until the corpus holds 300 papers that are
actually about sensing people. Off-topic downloads are deleted rather than kept.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import harvest_wifi as H  # noqa: E402
import wifi_index as X  # noqa: E402

CORPUS = ROOT / "papers" / "wifi_sensing"
TARGET = 300


def relevance_of(row: dict) -> str:
    pdf = CORPUS / row["fileName"]
    text = X.pdf_text(pdf) if pdf.exists() else ""
    hay = (row["title"] + "\n" + X.head_text(text)).lower()
    return X.classify_relevance(row["title"], hay)


def main() -> int:
    rows = json.loads((CORPUS / "manifest.json").read_text())
    print(f"starting corpus: {len(rows)}", flush=True)

    # 1. classify what we already hold
    kept: list[dict] = []
    dropped: list[dict] = []
    for r in rows:
        rel = relevance_of(r)
        r["relevance"] = rel
        (kept if rel == "human-sensing" else dropped).append(r)
    print(f"  human-sensing: {len(kept)}   removed: {len(dropped)}", flush=True)
    for d in dropped[:20]:
        print(f"    drop {d['id']} ({d['relevance']}) {d['title'][:62]}")
    if len(dropped) > 20:
        print(f"    … and {len(dropped) - 20} more")

    need = TARGET - len(kept)
    if need <= 0:
        print(f"nothing to refill (already {len(kept)} ≥ {TARGET})")
        return 0

    # 2. walk down the remaining candidates by citation
    print(f"\nrefilling {need} slots…", flush=True)
    cands = H.arxiv_gate()
    H.enrich(cands)
    cands.sort(key=lambda c: (-c["citations"], -(c.get("year") or 0), c["title"]))
    have = {H.norm_title(r["title"]) for r in rows}
    pool = [c for c in cands if H.norm_title(c["title"]) not in have]
    print(f"  candidate pool after de-dup: {len(pool)}", flush=True)

    added: list[dict] = []
    tried = 0
    for c in pool:
        if len(kept) + len(added) >= TARGET:
            break
        tried += 1
        dest = CORPUS / f"{c['arxiv']}.pdf"
        if not dest.exists():
            time.sleep(H.ARXIV_PAUSE / H.DOWNLOAD_WORKERS)
            ok, _why = H.fetch(c["pdfUrl"], dest)
            if not ok or not H.valid_pdf(dest):
                dest.unlink(missing_ok=True)
                continue
        probe = {
            "fileName": dest.name,
            "title": c["title"],
            "modality": "other",
        }
        rel = relevance_of(probe)
        if rel != "human-sensing":
            dest.unlink(missing_ok=True)
            if tried % 25 == 0:
                print(f"  scanned {tried} candidates, admitted {len(added)}", flush=True)
            continue
        c["relevance"] = "human-sensing"
        added.append(c)
        if len(added) % 5 == 0:
            print(f"  scanned {tried}, admitted {len(added)}", flush=True)

    print(f"\nscanned {tried} candidates, admitted {len(added)}")

    final = kept + added
    final.sort(key=lambda c: (-c["citations"], -(c.get("year") or 0), c["title"]))
    out = [
        {
            "id": f"W{i:03d}",
            "arxiv": c["arxiv"],
            "doi": c.get("doi", ""),
            "title": c["title"],
            "year": c.get("year"),
            "venue": c.get("venue", ""),
            "authors": c.get("authors", []),
            "citations": c["citations"],
            "fileName": f"{c['arxiv']}.pdf",
            "pdfUrl": c["pdfUrl"],
            "relevance": "human-sensing",
        }
        for i, c in enumerate(final, 1)
    ]
    CORPUS.joinpath("manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    CORPUS.joinpath("refill_summary.json").write_text(
        json.dumps(
            {
                "keptFromOriginal": len(kept),
                "dropped": [{"id": d["id"], "title": d["title"], "relevance": d["relevance"]} for d in dropped],
                "added": [{"arxiv": a["arxiv"], "title": a["title"], "citations": a["citations"]} for a in added],
                "finalCount": len(out),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"corpus now {len(out)} human-sensing papers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())