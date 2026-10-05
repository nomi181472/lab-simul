#!/usr/bin/env python3
"""Harvest the Neuroevolution corpus via OpenAlex.

Why not the arXiv API: `export.arxiv.org/api/query` is rate-limited to the point
of hard 429 ("Rate exceeded") in this environment, while `arxiv.org/pdf/<id>`
serves PDFs fine. OpenAlex is reachable and is a better citation source anyway:

  - `cited_by_count` is the real citation count, so no fuzzy Crossref
    title-matching is needed to build the ordering;
  - `abstract_inverted_index` lets the relevance gate run on title+abstract
    *before* spending a download;
  - `locations[].pdf_url` / `best_oa_location.pdf_url` resolves an open-access
    copy directly, arXiv-hosted or not.

The gate itself is the same SIGNAL/TASK/OFFTOPIC policy used by
scripts/harvest_neuroevolution.py, and is re-verified against the real PDF text
before a paper is admitted (see relevance_of below), so a paper whose abstract
is missing cannot slip through on a title alone.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import harvest_neuroevolution as H

ROOT = H.ROOT
OUT = ROOT / "papers" / "neuroevolution"
TARGET = 300
OA = "https://api.openalex.org/works"
MAILTO = H.MAILTO
UA = H.UA

# OpenAlex full-text search is broad, so these lean on the gate rather than on
# the query wording. Several phrasings are used because `search` matches title,
# abstract and full text, and one phrasing alone misses whole subfields.
QUERIES = [
    "neuroevolution",
    "neuroevolution evolved neural network",
    "evolutionary deep learning neural architecture search",
    "genetic algorithm evolve neural network weights",
    "genetic programming symbolic regression",
    "quality diversity algorithm map elites",
    "novelty search open-ended evolution",
    "evolution strategy reinforcement learning policy",
    "CMA-ES covariance matrix adaptation",
    "NEAT neuroevolution augmenting topologies",
    "strongly typed genetic programming",
    "evolved neurocontroller robot locomotion",
    "evolving network architecture",
    "population based training hyperparameter",
    "behavioral diversity evolution robot",
    "evolutionary optimization neural network training",
    "direct encoding indirect encoding neuroevolution",
    "modular neural network evolution",
    "aging evolution neuroevolution",
    "evolved deep learning policy agent",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def abstract_of(w: dict) -> str:
    inv = w.get("abstract_inverted_index")
    if not inv:
        return ""
    n = max((max(v) for v in inv.values() if v), default=-1) + 1
    buf = [""] * n
    for word, poss in inv.items():
        for p in poss:
            if 0 <= p < n:
                buf[p] = word
    return " ".join(buf)


def pdf_url_of(w: dict) -> str:
    """Prefer an arXiv copy, then the best OA location, then anything else."""
    cands: list[str] = []
    best = w.get("best_oa_location") or {}
    if best.get("pdf_url"):
        cands.append(best["pdf_url"])
    for loc in w.get("locations") or []:
        if loc.get("pdf_url"):
            cands.append(loc["pdf_url"])
    if best.get("landing_page_url") and "arxiv.org" in best["landing_page_url"]:
        aid = best["landing_page_url"].rstrip("/").split("/abs/")[-1]
        if aid:
            cands.append(f"https://arxiv.org/pdf/{aid}")
    arxiv_first = [c for c in cands if "arxiv.org" in c]
    for c in arxiv_first + cands:
        return c
    return ""


def fetch_page(params: dict) -> dict:
    u = OA + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(u, headers=UA), timeout=60
            ) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3:
                d = 8.0 * (2**attempt)
                log(f"    ~ HTTP {e.code}, backing off {d:.0f}s")
                time.sleep(d)
                continue
            log(f"    ! HTTP {e.code}")
            return {}
        except Exception as e:
            log(f"    ! {type(e).__name__}: {str(e)[:60]}")
            return {}
    return {}


def discover(per_query: int) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    for q in QUERIES:
        params = {
            "search": q,
            "per-page": min(per_query, 200),
            "mailto": MAILTO,
            "select": "id,doi,title,display_name,publication_year,cited_by_count,"
            "abstract_inverted_index,locations,best_oa_location,authorships,primary_location",
        }
        d = fetch_page(params)
        rows = d.get("results") or []
        added = 0
        for w in rows:
            wid = w.get("id")
            if not wid or wid in seen:
                continue
            title = w.get("title") or w.get("display_name") or ""
            if not title:
                continue
            abstract = abstract_of(w)
            ok, _reason = H.gate(f"{title} {abstract}")
            if not ok:
                continue
            pdf = pdf_url_of(w)
            if not pdf:
                continue
            authors = [
                (a.get("author") or {}).get("display_name", "")
                for a in (w.get("authorships") or [])[:12]
            ]
            venue = ""
            pl = w.get("primary_location") or {}
            if pl.get("source"):
                venue = (pl["source"] or {}).get("display_name") or ""
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            seen[wid] = {
                "openalex": wid.rsplit("/", 1)[-1],
                "doi": doi,
                "title": title,
                "year": w.get("publication_year"),
                "authors": authors,
                "venue": venue,
                "citations": w.get("cited_by_count") or 0,
                "pdfUrl": pdf,
                "abstract": abstract[:4000],
            }
            added += 1
        log(f"  {added:>4}  {q}")
        time.sleep(1.0)
    log(f"OpenAlex gated+OA pool: {len(seen)}")
    return seen


def relevance_of(row: dict, text: str) -> str:
    """Second gate on the real PDF text, so a thin abstract cannot carry a paper."""
    hay = (row["title"] + "\n" + text[:22000]).lower()
    ok, _ = H.gate(f"{row['title']} {row['abstract']}")
    if not ok:
        return "off-topic"
    # OpenAlex often points at a repository wrapper (thesis deposit notice,
    # cover page, scanned front matter). Those carry a title and ~5k chars of
    # boilerplate but no paper, so they fail here and the next candidate is used.
    if len(text.strip()) < H.MIN_TEXT:
        return "too-thin"
    return H.NEURO_RELEVANCE(row["title"], hay)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--per-query", type=int, default=200)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    H.CACHE = ROOT / ".cache" / "harvest_neuro"
    H.CACHE.mkdir(parents=True, exist_ok=True)

    pool = discover(args.per_query)
    if not pool:
        log("empty pool; nothing to do")
        return 1

    ranked = sorted(
        pool.values(), key=lambda c: (-c["citations"], -(c.get("year") or 0), c["title"])
    )
    log(f"top 8 by citation:")
    for c in ranked[:8]:
        log(f"   {c['citations']:>6}  {c['title'][:66]}")

    manifest: list[dict] = []
    failed: list[list[str]] = []

    def job(c: dict) -> dict | None:
        arxiv = ""
        m = re.search(r"arxiv\.org/pdf/([^?#]+)", c["pdfUrl"])
        if m:
            arxiv = m.group(1)
        name = f"{c['openalex']}.pdf"
        dest = OUT / name
        if not H.valid_pdf(dest):
            time.sleep(H.ARXIV_PAUSE / H.DOWNLOAD_WORKERS)
            ok, why = H.fetch(c["pdfUrl"], dest)
            if not ok or not H.valid_pdf(dest):
                dest.unlink(missing_ok=True)
                failed.append([c["openalex"], why])
                return None
        # verify against the real text before admitting
        text = H.pdftotext(dest)
        rel = relevance_of(c, text)
        if rel == "too-thin":
            # Keep the PDF: an empty pdftotext under load looks identical to a
            # thin one, and deleting on that would lose the paper for good.
            log(f"    x too-thin ({len(text)} ch) {c['title'][:56]}")
            return None
        if rel != "neuroevolution":
            dest.unlink(missing_ok=True)
            return None
        c = dict(c)
        c["fileName"] = name
        c["arxiv"] = arxiv
        c["relevance"] = rel
        return c

    log(f"downloading + verifying until {args.target}")
    with ThreadPoolExecutor(max_workers=H.DOWNLOAD_WORKERS) as ex:
        for n, r in enumerate(ex.map(job, ranked), 1):
            if r:
                manifest.append(r)
            if n % 25 == 0:
                log(f"  {len(manifest)} verified / {n} attempted")
            if len(manifest) >= args.target:
                break

    manifest.sort(key=lambda c: (-c["citations"], -(c.get("year") or 0), c["title"]))
    rows = []
    for i, c in enumerate(manifest, 1):
        rows.append(
            {
                "id": f"N{i:03d}",
                "openalex": c["openalex"],
                "arxiv": c.get("arxiv", ""),
                "doi": c.get("doi", ""),
                "title": c["title"],
                "year": c.get("year"),
                "venue": c.get("venue", ""),
                "authors": c.get("authors", []),
                "citations": c["citations"],
                "fileName": c["fileName"],
                "pdfUrl": c["pdfUrl"],
                "relevance": "neuroevolution",
            }
        )
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "download_summary.json").write_text(
        json.dumps(
            {
                "source": "openalex",
                "success": [r["openalex"] for r in rows],
                "failed": sorted(failed),
            },
            indent=2,
        )
        + "\n"
    )
    log(
        f"DONE manifest={len(rows)} failed={len(failed)} "
        f"pdfs={len(list(OUT.glob('*.pdf')))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())