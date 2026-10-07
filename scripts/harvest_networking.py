#!/usr/bin/env python3
"""Harvest top-cited networking & application-network papers (post-2010).

Sources
  arXiv     - candidate universe; many have retrievable PDFs
  Crossref  - citation counts (is-referenced-by-count), DOI, venue, metadata

Relevance gate (combined "networking + application networks")
  Signal terms cover:
    - core networking: network, protocol, routing, forwarding, packet, flow,
      congestion control, tcp, udp, ip, bgp, ospf, quic, mptcp, sdn, nfv,
      network architecture/measurement, datacenter/wan/wireless/mobile/edge
    - application networks/overlays: overlay, peer-to-peer, p2p, CDN,
      content delivery, DHT, bittorrent, ipfs, service mesh, application-layer,
      http/2/3, grpc/rpc, microservices networking, overlay routing

  Task terms: design, architecture, protocol, performance, scalability,
    latency, throughput, reliability, availability, fairness, efficiency, optimization

Date filter: published after 2010 (year >= 2011).
Ranking: by citations desc, year desc, title asc.
Target: top 300 verified PDFs.

Outputs (papers/networking/)
  manifest.json          citation-ordered metadata for verified PDFs
  download_summary.json  {success, failed}
  unavailable.json       high-citation works with no open-access copy

Re-running is safe: API responses cached; finished PDFs skipped.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers" / "networking"
CACHE = ROOT / ".cache" / "harvest_networking"

TARGET = 300
PDF_TIMEOUT = 90
ARXIV_PAUSE = 10.0
CROSSREF_PAUSE = 1.0
DOWNLOAD_WORKERS = 2
MAX_RETRIES = 5
CONTACT = "nomansoomro51@gmail.com"
MAILTO = CONTACT

UA = {"User-Agent": f"paper-harvest/1.0 (mailto:{MAILTO})"}
ARXIV_UA = {"User-Agent": f"paper-harvest/1.0 (mailto:{MAILTO})"}

# ---------------------------------------------------------------- relevance

SIGNAL = re.compile(
    r"\bnetwork\b(?! social)|\bcomputer network\b|\bprotocol\b|\brout(ing|e)\b|\bforward(ing)?\b|"
    r"\bpacket(s)?\b|\bflow(s)?\b|\bcongestion control\b|\btcp\b|\budp\b|\bip\b|"
    r"\bbgp\b|\bospf\b|\bquic\b|\bmptcp\b|\bsdn\b|\bnfv\b|"
    r"\bnetwork architecture\b|\bnetwork measurement\b|"
    r"\bdatacenter network\b|\bdata center network\b|\bwan\b|\bwide.area network\b|"
    r"\bwireless network\b|\bmobile network\b|\bedge network\b|\bedge computing\b|"
    r"\boverlay network\b|\boverlay(s)?\b|\bpeer.to.peer\b|\bp2p\b|"
    r"\bcontent delivery network\b|\bcdn\b|\bdht\b|\bbittorrent\b|\bipfs\b|"
    r"\bservice mesh\b|\bapplication.layer\b|\bapplication.layer protocol\b|"
    r"\bhttp/2\b|\bhttp/3\b|\bgrpc\b|\bg.r.p.c\b|\brpc\b|"
    r"\bmicroservices networking\b",
    re.I,
)

TASK = re.compile(
    r"\bdesign\b|\barchitecture\b|\bprotocol\b|\bperformance\b|\bscalab(il|le)\b|"
    r"\blatenc(y|ies)\b|\bthroughput\b|\breliab(le|ility)\b|\bavailab(le|ility)\b|"
    r"\bfairness\b|\befficien(cy|t)\b|\boptimi[sz](ation|e)\b",
    re.I,
)

OFFTOPIC = re.compile(
    r"\bnetwork security\b|\bcybersecurity\b|\bcryptograph(y|ic)\b|"
    r"\bphishing\b|"
    r"\bblockchain\b|\bsmart contract\b|\bcryptocurrency\b|"
    r"\bgraph theory\b|\bsocial network analysis\b(?! networking)|"
    r"\bdeep learning for network\b|\bml for network\b|\bai for network\b",
    re.I,
)

# arXiv queries (broader to catch relevant papers)
ARXIV_QUERIES = [
    'cat:cs.NI',
    'cat:cs.DC',
]

# Crossref queries for venue-published works
CROSSREF_QUERIES = [
    "software defined networking",
    "content delivery network",
    "data center networking",
    "congestion control",
]

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slug_key(title: str) -> str:
    t = re.sub(r"[^\w\s]", " ", title.lower())
    return re.sub(r"\s+", " ", t).strip()[:120]


def cached_get(
    url: str, headers: dict, cache_key: str, timeout: int = 45, tag: str = ""
) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    c = CACHE / cache_key
    if c.exists() and c.stat().st_size > 0:
        return c.read_bytes()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            c.write_bytes(data)
            return data
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == MAX_RETRIES:
                raise
            pause = 2**attempt
            log(f"retry {attempt}/{MAX_RETRIES} {tag or url[:40]}: {e}; sleep {pause}s")
            time.sleep(pause)
    return b""


def arxiv_query(q: str, start: int = 0, max_results: int = 100) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "search_query": q,
            "start": start,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"http://export.arxiv.org/api/query?{params}"
    data = cached_get(url, ARXIV_UA, f"arxiv_{abs(hash(q+str(start)))}.xml", timeout=60, tag="arxiv")
    text = data.decode("utf-8", "replace")

    entries = []
    for m in re.finditer(r"<entry>(.*?)</entry>", text, re.S):
        e = m.group(1)

        def g(tag: str) -> str:
            mm = re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S)
            return re.sub(r"\s+", " ", unescape(mm.group(1).strip())) if mm else ""

        idm = re.search(r"<id>.*?/(\d+\.\d+)</id>", e)
        arxiv = idm.group(1) if idm else g("id").split("/")[-1]
        pdf = f"https://arxiv.org/pdf/{arxiv}.pdf"

        updated = g("updated")[:10] or g("published")[:10]
        year = int(updated[:4]) if updated[:4].isdigit() else 0

        authors: list[str] = []
        for am in re.finditer(r"<author><name>(.*?)</name>", e, re.S):
            authors.append(unescape(am.group(1).strip()))

        abs_raw = g("summary")
        title = g("title")

        entries.append(
            {
                "arxiv": arxiv,
                "doi": g("arxiv:doi") or doi_from_arxiv_abs(abs_raw),
                "title": title,
                "year": year,
                "venue": "",
                "authors": authors,
                "citations": 0,
                "pdfUrl": pdf,
                "source": "arxiv",
                "abstract": abs_raw,
                "absKey": slug_key(title),
            }
        )
    return entries


def doi_from_arxiv_abs(abs_text: str) -> str:
    m = re.search(r"doi:\s*(10\.\d{4,}/[^\s<]+)", abs_text, re.I)
    if m:
        return m.group(1).rstrip(".,)")
    return ""


def unescape(s: str) -> str:
    from html import unescape as hu

    return hu(s)


def relevant(c: dict) -> bool:
    t = (c.get("title") or "") + " " + (c.get("abstract") or "")
    if not t:
        return False
    if OFFTOPIC.search(t):
        return False
    return bool(SIGNAL.search(t) and TASK.search(t))


def arxiv_gate() -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for q in ARXIV_QUERIES:
        for start in (0, 100, 200, 300, 400, 500):
            try:
                batch = arxiv_query(q, start=start, max_results=100)
            except Exception as e:
                log(f"arxiv err {q} start={start}: {e}")
                break
            if not batch:
                break
            for c in batch:
                if c["year"] and c["year"] <= 2010:
                    continue
                key = c["arxiv"] or c["absKey"]
                if key in seen:
                    continue
                seen.add(key)
                if relevant(c):
                    out.append(c)
            if len(batch) < 100:
                break
    log(f"arxiv candidates: {len(out)}")
    return out

def crossref_search(q: str, rows: int = 100, cursor: str = "*") -> dict:
    params = urllib.parse.urlencode(
        {
            "query": q,
            "rows": rows,
            "cursor": cursor,
            "select": "DOI,title,author,container-title,published-print,published-online,issued,created,is-referenced-by-count,abstract,link",
            "filter": "from-pub-date:2011-01-01",
        }
    )
    url = f"https://api.crossref.org/works?{params}"
    data = cached_get(url, UA, f"cr_{abs(hash(q+cursor))}.json", timeout=60, tag="crossref")
    return json.loads(data.decode("utf-8", "replace"))


def extract_year(w: dict) -> int:
    for k in ("issued", "published-print", "published-online", "created"):
        p = w.get(k)
        if p and p.get("date-parts"):
            try:
                y = p["date-parts"][0][0]
                if isinstance(y, int):
                    return y
            except Exception:
                pass
    return 0


def authors_from_cr(w: dict) -> list[str]:
    a = []
    for au in w.get("authors") or []:
        g = au.get("given") or ""
        f = au.get("family") or ""
        n = (g + " " + f).strip()
        if n:
            a.append(n)
    return a


def venue_from_cr(w: dict) -> str:
    ct = w.get("container-title") or []
    return ct[0] if ct else ""


def abstract_from_cr(w: dict) -> str:
    ab = w.get("abstract") or ""
    ab = re.sub(r"^<jats:title>.*?</jats:title>", "", ab, flags=re.S | re.I)
    ab = re.sub(r"<[^>]+>", " ", ab)
    return unescape(ab)


def pdf_from_links(links: list[dict]) -> str:
    for l in links or []:
        if l.get("content-type") == "application/pdf":
            return l["URL"]
        if l["URL"].lower().endswith(".pdf"):
            return l["URL"]
    return ""


def crossref_gate() -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for q in CROSSREF_QUERIES:
        cursor = "*"
        for _ in range(10):  # pages
            try:
                res = crossref_search(q, rows=50, cursor=cursor)
            except Exception as e:
                log(f"crossref err {q}: {e}")
                break
            msg = res.get("message", {})
            items = msg.get("items", [])
            for w in items:
                doi = w.get("DOI") or ""
                title = " ".join(w.get("title") or [])
                if not title:
                    continue
                y = extract_year(w)
                if y and y <= 2010:
                    continue
                key = doi or slug_key(title)
                if key in seen:
                    continue
                seen.add(key)
                c = {
                    "arxiv": "",
                    "doi": doi,
                    "title": title,
                    "year": y,
                    "venue": venue_from_cr(w),
                    "authors": authors_from_cr(w),
                    "citations": int(w.get("is-referenced-by-count") or 0),
                    "pdfUrl": pdf_from_links(w.get("link") or []),
                    "source": "crossref",
                    "abstract": abstract_from_cr(w),
                    "absKey": slug_key(title),
                }
                if relevant(c):
                    out.append(c)
            cursor = msg.get("next-cursor")
            if not cursor:
                break
            time.sleep(CROSSREF_PAUSE)
    log(f"crossref candidates: {len(out)}")
    return out

def enrich(cands: list[dict]) -> None:
    for c in cands:
        if c.get("citations") == 0 and c.get("doi"):
            try:
                url = f"https://api.crossref.org/works/{urllib.parse.quote(c['doi'], safe='')}"
                data = cached_get(url, UA, f"crw_{abs(hash(c['doi']))}.json", timeout=45, tag="crwork")
                w = json.loads(data.decode("utf-8", "replace")).get("message", {})
                c["citations"] = int(w.get("is-referenced-by-count") or c["citations"])
                if not c.get("venue"):
                    ct = w.get("container-title") or []
                    c["venue"] = ct[0] if ct else c["venue"]
            except Exception as e:
                log(f"enrich fail {c.get('doi')}: {e}")


def valid_pdf(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 5000:
            return False
        with open(path, "rb") as f:
            h = f.read(4)
        if h not in (b"%PDF", b"\x89PNG"):
            return False
        return True
    except Exception:
        return False


def fetch(url: str, dest: Path) -> tuple[bool, str]:
    if not url:
        return False, "no-pdf-url"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=PDF_TIMEOUT) as r:
            data = r.read()
        dest.write_bytes(data)
        return True, f"{len(data)}B"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:50]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--no-landscape", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    cands = arxiv_gate()
    cands.extend(crossref_gate())
    enrich(cands)

    cands.sort(key=lambda c: (-c["citations"], -(c.get("year") or 0), c["title"]))
    log(f"top 5 by citation: {[ (c['citations'], c['title'][:34]) for c in cands[:5] ]}")

    manifest: list[dict] = []
    failed: list[list[str]] = []

    def job(c: dict) -> dict | None:
        dest = OUT / f"{c['arxiv'] or c['doi'].replace('/', '_')}.pdf"
        # prefer arxiv name if exists
        if c["arxiv"]:
            dest = OUT / f"{c['arxiv']}.pdf"
        if valid_pdf(dest):
            return c
        time.sleep(ARXIV_PAUSE / DOWNLOAD_WORKERS)
        ok, why = fetch(c["pdfUrl"], dest)
        if ok and valid_pdf(dest):
            return c
        dest.unlink(missing_ok=True)
        failed.append([c.get("arxiv") or c.get("doi") or c["title"][:20], why])
        return None

    log(f"downloading until {args.target} verified PDFs")
    with cf.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        for n, r in enumerate(ex.map(job, cands), 1):
            if r:
                manifest.append(r)
            if n % 25 == 0:
                log(f"  {len(manifest)} verified / {n} attempted")
            if len(manifest) >= args.target:
                break

    manifest.sort(key=lambda c: (-c["citations"], -(c.get("year") or 0), c["title"]))

    rows = []
    for i, c in enumerate(manifest, 1):
        fname = f"{c['arxiv']}.pdf" if c.get("arxiv") else f"{c['doi'].replace('/', '_')}.pdf"
        rows.append(
            {
                "id": f"N{i:03d}",
                "arxiv": c.get("arxiv", ""),
                "doi": c.get("doi", ""),
                "title": c["title"],
                "year": c.get("year"),
                "venue": c.get("venue", ""),
                "authors": c.get("authors", []),
                "citations": c["citations"],
                "fileName": fname,
                "source": c.get("source", ""),
            }
        )

    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    (OUT / "download_summary.json").write_text(
        json.dumps({"success": len(rows), "failed": len(failed)}, indent=2)
    )
    (OUT / "unavailable.json").write_text(json.dumps(failed, indent=2))

    log(f"done: {len(rows)} PDFs -> {OUT}")
    if failed:
        log(f"failed: {len(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
