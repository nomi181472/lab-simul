#!/usr/bin/env python3
"""Object Detection corpus manifest.

Papers folder -> pdftotext text cache (.cache/od_text/) + metadata
(arXiv API for title/year/authors/abstract, OpenAlex for DOI + citation
count) -> papers/object_detection/manifest.json.

Idempotent: every network call and every text extraction is cached, so
re-running only fills gaps (safe for the continuous knowledge loop).

Unique papers are deduped across duplicate PDFs (e.g. yolo.pdf ==
1506.02640v5.pdf, the two IET downloads). Manifest is ordered by
(original arXiv submission) year then arXiv id; IDs are OD001..ODnnn.

Usage: python3 scripts/od_manifest.py
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "papers" / "object_detection"
OUT = PDF_DIR / "manifest.json"
CACHE = ROOT / ".cache" / "od_text"
META = ROOT / ".cache" / "od_meta"

UA = {"User-Agent": "research-lab/1.0 (mailto:research@local)"}
MAX_RETRIES = 5
ARXIV_CHUNK = 40

_log_t = [time.time()]


def log(msg: str) -> None:
    now = time.time()
    print(f"[{now - _log_t[0]:7.1f}s] {msg}", flush=True)
    _log_t[0] = now


# --------------------------------------------------------------- http cache


def cached_get(url: str, cache_key: str, headers: dict | None = None,
               timeout: int = 45, max_retries: int = MAX_RETRIES) -> bytes | None:
    path = META / f"{cache_key}.cache"
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    delay = 4.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers or UA), timeout=timeout
            ) as r:
                raw = r.read()
            META.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            return raw
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                log(f"    ~ HTTP {e.code} retry {attempt + 1} in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
                continue
            log(f"    ! HTTP {e.code} for {url[:90]}")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            log(f"    ! {type(e).__name__}: {str(e)[:70]}")
            return None
    return None


# ----------------------------------------------------------- pdf extraction


def pdf_text(pdf: pathlib.Path) -> str:
    """Full text via pdftotext (raw reading order, NOT -layout)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{pdf.stem}.txt"
    if cache.exists() and cache.stat().st_size > 200:
        return cache.read_text(errors="replace")
    try:
        r = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, timeout=180)
        text = r.stdout.decode("utf-8", "replace")
    except Exception:
        text = ""
    cache.write_text(text)
    return text


def pdf_pages(pdf: pathlib.Path) -> int:
    try:
        r = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, timeout=60)
        m = re.search(r"^Pages:\s+(\d+)", r.stdout.decode("utf-8", "replace"), re.M)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


# ------------------------------------------------------------------ helpers


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def title_match(a: str, b: str) -> bool:
    return norm_title(a) == norm_title(b)


ARXIV_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
ARXIV_IN_TEXT = re.compile(r"ar[Xx]iv:\s*(\d{4}\.\d{4,5})(v\d+)?")


def parse_ids(pdfs: list[pathlib.Path]) -> list[dict]:
    """One candidate row per PDF, resolving arXiv ids from filename or text."""
    rows = []
    for pdf in sorted(pdfs):
        stem = pdf.stem
        m = ARXIV_RE.match(stem)
        if m:
            rows.append({"pdf": pdf, "arxiv": m.group(1), "key": m.group(1)})
            continue
        # non-standard name (yolo.pdf, IET ...): peek at page 1
        try:
            r = subprocess.run(
                ["pdftotext", "-f", "1", "-l", "1", str(pdf), "-"],
                capture_output=True, timeout=60,
            )
            head = r.stdout.decode("utf-8", "replace")
        except Exception:
            head = ""
        m = ARXIV_IN_TEXT.search(head)
        if m:
            rows.append({"pdf": pdf, "arxiv": m.group(1), "key": m.group(1)})
        else:
            # derive a title-ish key from the filename (IET papers)
            guess = re.sub(r"[-_]?\d+$", "", stem)
            guess = re.sub(r"^(IET Computer Vision\s*-\s*\d{4}\s*-\s*\w+\s*-\s*)", "", guess)
            ym = re.search(r"\b((?:19|20)\d{2})\b", stem)
            rows.append({
                "pdf": pdf,
                "arxiv": None,
                "key": norm_title(guess) or stem,
                "name_title": guess.strip(),
                "name_year": int(ym.group(1)) if ym else None,
            })
    return rows


# ------------------------------------------------------------------- arxiv


def arxiv_meta(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(ids), ARXIV_CHUNK):
        chunk = ids[i : i + ARXIV_CHUNK]
        url = (
            "https://export.arxiv.org/api/query?id_list="
            + ",".join(chunk)
            + f"&max_results={len(chunk)}"
        )
        raw = cached_get(url, "arxiv_" + "_".join(sorted(chunk))[:120])
        if not raw:
            continue
        xml = raw.decode("utf-8", "replace")
        for entry in xml.split("<entry>")[1:]:
            def field(tag: str) -> str:
                m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", entry, re.S)
                return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

            def ax(tag: str) -> str:
                m = re.search(rf"<arxiv:{tag}[^>]*>(.*?)</arxiv:{tag}>", entry, re.S)
                return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

            idm = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", entry)
            if not idm:
                continue
            aid = idm.group(1).split("v")[0]
            authors = re.findall(r"<name>([^<]+)</name>", entry)
            published = field("published")[:10]
            out[aid] = {
                "title": field("title"),
                "abstract": field("summary"),
                "authors": authors,
                "published": published,
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "categories": re.findall(r'<category term="([^"]+)"', entry),
                "doi": ax("doi"),
                "pdfUrl": f"https://arxiv.org/pdf/{aid}",
            }
        log(f"    arXiv metadata {i + len(chunk)}/{len(ids)}")
        time.sleep(3.0)  # arXiv API courtesy delay
    return out


# ----------------------------------------------------------------- openalex


def oa_cache_key(title: str) -> str:
    return "oa_" + re.sub(r"[^a-z0-9]+", "_", title.lower())[:80]


def openalex_search(title: str) -> dict | None:
    q = urllib.parse.urlencode(
        {
            "filter": f'title.search:"{title}"',
            "per-page": 5,
            "select": "id,title,doi,publication_year,cited_by_count,primary_location",
            "mailto": "research@local",
        }
    )
    raw = cached_get(f"https://api.openalex.org/works?{q}", oa_cache_key(title),
                     max_retries=3)
    if not raw:
        return None
    try:
        results = json.loads(raw)["results"]
    except Exception:
        return None
    if not results:
        return None
    nt = norm_title(title)
    exact = [w for w in results if norm_title(w.get("title") or "") == nt]
    pick = (exact or results)[0]
    return {
        "doi": (pick.get("doi") or "").replace("https://doi.org/", "") or None,
        "citations": pick.get("cited_by_count") or 0,
        "openalex": pick.get("id"),
        "openalexYear": pick.get("publication_year"),
        "venue": ((pick.get("primary_location") or {}).get("source") or {}).get("display_name"),
    }


# --------------------------------------------------------------------- main


def main() -> int:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = list(PDF_DIR.glob("*.pdf"))
    log(f"{len(pdfs)} PDFs found")

    rows = parse_ids(pdfs)
    arxiv_ids = sorted({r["arxiv"] for r in rows if r["arxiv"]})
    log(f"{len(arxiv_ids)} arXiv ids (deduped)")

    log("fetching arXiv metadata ...")
    axmeta = arxiv_meta(arxiv_ids)

    # dedupe rows by arXiv id / key -> one paper, list of fileNames
    papers: dict[str, dict] = {}
    for r in rows:
        key = r["key"]
        if key not in papers:
            papers[key] = {
                "files": [],
                "arxiv": r["arxiv"],
                "key": key,
                "name_title": r.get("name_title"),
                "name_year": r.get("name_year"),
            }
        papers[key]["files"].append(r["pdf"].name)
    log(f"{len(papers)} unique papers after dedupe")

    # extract text + pages for each paper's primary file
    for p in papers.values():
        primary = PDF_DIR / p["files"][0]
        text = pdf_text(primary)
        p["textChars"] = len(text)
        p["pages"] = pdf_pages(primary)
        p["textFile"] = f".cache/od_text/{primary.stem}.txt"

    # metadata: arXiv first, then filename-derived title, then text-derived
    missing_arxiv = [p for p in papers.values() if p["arxiv"] and p["arxiv"] not in axmeta]
    if missing_arxiv:
        log(f"WARNING: {len(missing_arxiv)} arXiv ids missing metadata: "
            + ", ".join(p["arxiv"] or "?" for p in missing_arxiv))

    entries = []
    for p in papers.values():
        aid = p["arxiv"]
        meta = axmeta.get(aid, {}) if aid else {}
        title = meta.get("title") or p.get("name_title")
        if not title:
            # last resort: first substantive line of the extracted text
            text = (CACHE / (pathlib.Path(p["files"][0]).stem + ".txt")).read_text(errors="replace")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            title = next((ln for ln in lines[:12] if len(ln) > 12), p["key"])
        year = meta.get("year") or p.get("name_year")
        if not year and aid and aid[:2].isdigit():
            yy = int(aid[:2])
            year = 1900 + yy if yy >= 91 else 2000 + yy
        entries.append({
            "files": p["files"],
            "arxiv": aid,
            "title": title,
            "year": year,
            "authors": meta.get("authors", []),
            "abstract": meta.get("abstract", ""),
            "categories": meta.get("categories", []),
            "doi": meta.get("doi") or None,
            "citations": None,
            "venue": None,
            "pages": p["pages"],
            "textChars": p["textChars"],
            "textFile": p["textFile"],
            "pdfUrl": meta.get("pdfUrl")
            or f"https://arxiv.org/pdf/{aid}" if aid else None,
        })

    # OpenAlex enrichment (DOI + citations + venue)
    log("OpenAlex enrichment ...")
    for i, e in enumerate(entries):
        if not (META / f"{oa_cache_key(e['title'])}.cache").exists():
            time.sleep(1.1)  # pace OpenAlex (polite pool)
        oa = openalex_search(e["title"])
        if oa:
            if not e["doi"]:
                e["doi"] = oa["doi"]
            e["citations"] = oa["citations"]
            e["openalex"] = oa["openalex"]
            if not e["venue"]:
                e["venue"] = oa["venue"]
        if (i + 1) % 20 == 0:
            log(f"    {i + 1}/{len(entries)} enriched")

    # year-sort, assign ids
    entries.sort(key=lambda e: (e["year"] or 9999, e["arxiv"] or "zzz"))
    for i, e in enumerate(entries, 1):
        e["id"] = f"OD{i:03d}"
        e["fileName"] = e["files"][0]
        e["duplicateFiles"] = e["files"][1:]

    OUT.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + "\n")

    years = sorted({e["year"] for e in entries if e["year"]})
    log(f"wrote {OUT.relative_to(ROOT)}: {len(entries)} papers, "
        f"{years[0] if years else '?'}-{years[-1] if years else '?'}")
    log(f"  with doi: {sum(1 for e in entries if e['doi'])}, "
        f"with citations: {sum(1 for e in entries if e['citations'] is not None)}, "
        f"deduped files: {sum(len(e['duplicateFiles']) for e in entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
