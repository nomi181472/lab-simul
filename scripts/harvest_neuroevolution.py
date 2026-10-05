#!/usr/bin/env python3
"""Harvest the Neuroevolution corpus.

Same proven pipeline as scripts/harvest_wifi.py (arXiv universe -> title/abstract
gate -> Crossref citation ranking -> verified PDF download -> manifest), with the
domain logic swapped for neuroevolution.

Scope: work in which a neural network, brain, controller or network architecture
is produced or improved by an evolutionary search. That covers neuroevolution and
evolutionary deep learning, genetic programming, quality-diversity and novelty
search, CMA-ES, NEAT, and evolutionary architecture search. It deliberately
excludes classical evolutionary optimisation of *non-neural* problems and
pure reinforcement learning with no evolutionary component, because the lab's
subject is how a network gets built by a search rather than by backpropagation.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import subprocess
from concurrent.futures import ThreadPoolExecutor as cf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers" / "neuroevolution"
CACHE = ROOT / ".cache" / "harvest_neuro"

TARGET = 300
PDF_TIMEOUT = 90
# A real paper's pdftotext output is far longer than a repository wrapper page.
MIN_TEXT = 25_000

# Concurrency is deliberately low: pdftotext under 8+ concurrent workers returns
# empty stdout on this machine, which would silently look like an empty paper.
DOWNLOAD_WORKERS = 2

ARXIV_PAUSE = 3.0
CROSSREF_PAUSE = 0.4
MAX_RETRIES = 5
CONTACT = "nomansoomro51@gmail.com"
MAILTO = CONTACT

UA = {"User-Agent": f"paper-harvest/1.0 (mailto:{MAILTO})"}
ARXIV_UA = {"User-Agent": f"paper-harvest/1.0 (mailto:{MAILTO})"}

# ---------------------------------------------------------------- relevance

# The evolutionary machinery that makes a paper neuroevolutionary.
# Literal search terms. The regex below is built from these so that
# NEURO_RELEVANCE can substring-match the same vocabulary.
# Literal search terms. The regexes below are built from these tuples so that
# NEURO_RELEVANCE can substring-match the exact same vocabulary.
SIGNAL_TERMS: tuple[str, ...] = (
    # the field's own name
    "neuroevolution", "neuro-evolution", "neuro evolutionary",
    "evolutionary neural", "evolutionary deep learning",
    "evolutionary deep neural", "evolutionary algorithm",
    "evolutionary computation", "evolutionary search",
    "evolutionary strategy", "evolution strategies",
    "evolutionary architecture search", "evolutionary optimization of neural",
    "evolutionary robotics", "evolutionary robot",
    # classic machinery
    "genetic algorithm", "genetic programming", "genetic search",
    "genetic operator", "genetic operators",
    "evolution strategy", "natural evolution strategies",
    "cma-es", "covariance matrix adaptation",
    "differential evolution", "particle swarm",
    # diversity-driven search
    "quality-diversity", "quality diversity", "map-elites", "map elites",
    "illumination search", "novelty search", "novelty archive",
    "behavioral novelty", "behavioural novelty",
    "unified novelty search", "novelty-based diversity",
    "open-ended evolution", "open ended evolution",
    "open-ended search", "aging evolution", "age of structure",
    "modular neuroevolution",
    # topology-growing / typed systems
    "neat", "hyper-neat", "neuroevolution of augmenting topologies",
    "augmenting topologies", "strongly typed genetic programming",
    "typed genetic programming",
    # what gets evolved
    "evolved controller", "evolved policy", "evolved network",
    "evolved neural network", "evolving neural network",
    "evolve neural network", "evolve the network",
    "neurocontroller", "neuro-controller",
    "population-based training", "population based training",
    "mutation operator", "crossover operator",
    "gradient-based hyperheuristic", "lineage-based encoding",
    "evolve the architecture", "evolving the architecture",
    "network weights are evolved", "weights are evolved",
)

TASK_TERMS: tuple[str, ...] = (
    "neural network", "deep network", "convolutional", "recurrent",
    "lstm", "gru", "transformer", "architecture", "controller",
    "policy", "policies", "brain", "connectome",
    "locomotion", "gait", "walking", "robot", "agent",
    "reinforcement", "control", "weight", "weights", "topology",
    "hyperparameter", "search space", "feature", "classifier",
    "ensemble", "program", "rule", "open-ended", "novelty",
    "diversity", "behaviour", "behavior", "navigation",
    "manipulation", "perception", "sensing", "task", "goal",
    "evolution", "evolve", "genetic", "mutate", "mutation",
    "crossover", "selection", "robotics", "swarm", "flock",
    "multi-agent", "self-organiz", "self-organis", "fitness",
    "objective", "reward",
)

# Evolutionary work in this repository's other domains, or in fields that merely
# borrow the vocabulary. Deliberately narrow: phrases, not bare keywords, so a
# genuine neural-evolution paper is never caught by coincidence.
OFF_TERMS: tuple[str, ...] = (
    # classical combinatorial optimisation, no neural subject
    "traveling salesman", "travelling salesman", "vehicle routing",
    "nurse scheduling", "job shop scheduling", "knapsack",
    "bin packing", "supply chain", "test suite prioriti",
    "program repair", "semantic patch",
    # evolutionary design of physical artefacts
    "antenna design", "structural optimization", "structural optimisation",
    "topology optimization of", "wind farm layout", "crashworthiness",
    # evolutionary biology of molecules
    "protein folding", "protein design", "drug discovery",
    "molecular design", "rna folding", "dna design", "molecule generation",
    # evolutionary arts
    "evolutionary poetry", "evolutionary music", "turing test",
)

SIGNAL = re.compile("|".join(re.escape(t) for t in SIGNAL_TERMS), re.I)
TASK = re.compile("|".join(re.escape(t) for t in TASK_TERMS), re.I)
OFFTOPIC = re.compile("|".join(re.escape(t) for t in OFF_TERMS), re.I)
ARXIV_QUERIES = [
    'all:neuroevolution',
    'all:"evolutionary deep learning"',
    'all:"evolutionary neural network"',
    'all:"neuroevolution" AND all:controller',
    'all:"evolutionary algorithm" AND all:"neural network"',
    'all:"genetic algorithm" AND all:"neural network"',
    'all:"quality diversity"',
    'all:"map elites"',
    'all:"novelty search"',
    'all:"open-ended" AND all:evolution',
    'all:"behavioral diversity" AND all:search',
    'all:NEAT AND all:neuroevolution',
    'all:"strongly typed genetic programming"',
    'all:"genetic programming" AND all:neural',
    'all:"evolution strategy" AND all:policy',
    'all:"natural evolution strategy"',
    'all:CMA-ES',
    'all:"covariance matrix adaptation"',
    'all:"evolutionary architecture search"',
    'all:"neural architecture search" AND all:evolution',
    'all:"evolved neural network"',
    'all:"evolved controller"',
    'all:"neurocontroller"',
    'all:"evolving" AND all:"neural network"',
    'all:"mutation operator" AND all:neuro',
    'all:"evolutionary robot"',
    'all:"evolution" AND all:"locomotion"',
    'all:"evolutionary" AND all:"reinforcement learning"',
    'all:"population based training"',
    'all:"evolving neural" AND all:policy',
    'all:"connectome" AND all:evolution',
    'all:"evolution" AND all:"neural architecture"',
    'all:"genetic" AND all:"topology" AND all:"neural network"',
    'all:"evolutionary" AND all:"hyperparameter optimization" AND all:network',
]

CROSSREF_QUERIES = [
    "neuroevolution",
    "evolutionary neural network architecture search",
    "quality diversity algorithm",
    "novelty search open-ended evolution",
    "map elites illumination algorithm",
    "genetic programming neuroevolution",
    "NEAT neuroevolution augmented topologies",
    "evolution strategy reinforcement learning policy",
    "CMA-ES neural network optimization",
    "evolutionary deep learning",
    "evolved neurocontroller robot",
    "evolutionary architecture search neural network",
    "natural evolution strategies",
    "behavioral diversity neuroevolution",
    "evolutionary computation brain evolution",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- net utils


def cached_get(
    url: str, headers: dict, cache_key: str, timeout: int = 45, tag: str = ""
) -> bytes | None:
    path = CACHE / f"{cache_key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())["body"].encode()
        except Exception:
            path.unlink(missing_ok=True)

    delay = 4.0
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=timeout
            ) as r:
                raw = r.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"body": raw.decode("utf-8", "replace")}))
            return raw
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                log(f"    ~ {e.code} retry {attempt + 1} in {delay:.0f}s {tag}")
                time.sleep(delay)
                delay *= 2
                continue
            log(f"    ! HTTP {e.code} {tag}")
            return None
        except Exception as e:
            log(f"    ! {type(e).__name__}: {str(e)[:70]} {tag}")
            return None
    return None


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def title_match(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


# ------------------------------------------------------------------- arxiv


def harvest_arxiv() -> dict[str, dict]:
    log("arXiv harvest (candidate universe)")
    works: dict[str, dict] = {}
    for q in ARXIV_QUERIES:
        try:
            u = (
                "https://export.arxiv.org/api/query?search_query="
                + urllib.parse.quote(q)
                + "&start=0&max_results=100&sortBy=relevance"
            )
            raw = cached_get(u, ARXIV_UA, f"ax_{abs(hash(q))}", 60, q[:24])
        finally:
            time.sleep(ARXIV_PAUSE)
        added = 0
        if not raw:
            log(f"     0  {q}")
            continue
        for e in re.findall(r"<entry>(.*?)</entry>", raw.decode("utf-8", "replace"), re.S):
            idm = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", e)
            tm = re.search(r"<title>(.*?)</title>", e, re.S)
            if not idm or not tm:
                continue
            aid = idm.group(1).split("v")[0]
            title = " ".join(tm.group(1).split())
            key = norm_title(title)
            if not key or key in works:
                continue
            pm = re.search(r"<published>(\d{4})", e)
            am = re.search(r"<summary>(.*?)</summary>", e, re.S)
            works[key] = {
                "title": title,
                "year": int(pm.group(1)) if pm else None,
                "arxiv": aid,
                "venue": "arXiv preprint",
                "authors": re.findall(r"<name>([^<]+)</name>", e)[:12],
                "abstract": " ".join(am.group(1).split())[:4000] if am else "",
                "pdfUrl": f"https://arxiv.org/pdf/{aid}",
            }
            added += 1
        log(f"  {added:>4}  {q}")
    log(f"arXiv raw: {len(works)}")
    return works


# ---------------------------------------------------------------- crossref


def crossref_search(q: str, pages: int = 3) -> list[dict]:
    out: list[dict] = []
    for page in range(pages):
        u = (
            "https://api.crossref.org/works?query.bibliographic="
            + urllib.parse.quote(q)
            + f"&sort=is-referenced-by-count&order=desc&rows=100&offset={page * 100}"
            "&select=title,is-referenced-by-count,DOI,container-title,issued,author"
        )
        raw = cached_get(u, UA, f"cr_{abs(hash(q))}_{page}", 45, q[:24])
        time.sleep(CROSSREF_PAUSE)
        if not raw:
            break
        try:
            items = json.loads(raw)["message"]["items"]
        except Exception:
            break
        if not items:
            break
        for it in items:
            title = (it.get("title") or [""])[0]
            if not title:
                continue
            yr = None
            parts = (it.get("issued") or {}).get("date-parts") or [[]]
            if parts and parts[0] and parts[0][0]:
                yr = parts[0][0]
            out.append(
                {
                    "title": title,
                    "citations": it.get("is-referenced-by-count", -1),
                    "doi": it.get("DOI", ""),
                    "venue": (it.get("container-title") or [""])[0],
                    "year": yr,
                    "authors": [
                        f"{a.get('given','')} {a.get('family','')}".strip()
                        for a in (it.get("author") or [])[:12]
                    ],
                }
            )
    return out


def harvest_crossref_landscape() -> list[dict]:
    """The full citation-ordered landscape, incl. works we cannot download."""
    log("Crossref harvest (landscape / unavailable catalog)")
    seen: dict[str, dict] = {}
    for q in CROSSREF_QUERIES:
        items = crossref_search(q)
        n = 0
        for it in items:
            ok, _ = gate(f"{it['title']}")
            if not ok:
                continue
            key = norm_title(it["title"])
            if key not in seen:
                seen[key] = it
                n += 1
        log(f"  {n:>4}  {q}")
    log(f"Crossref landscape: {len(seen)}")
    return list(seen.values())


def crossref_citations(title: str) -> dict | None:
    """Citation count for one title, with strict title verification.

    An unverified match is worse than no match: a loose query returns unrelated
    works whose count would silently corrupt the citation ordering.
    """
    u = (
        "https://api.crossref.org/works?query.title="
        + urllib.parse.quote(title)
        + "&rows=3&select=title,is-referenced-by-count,DOI,container-title,issued"
    )
    raw = cached_get(u, UA, f"crt_{norm_title(title)[:70]}", 30)
    if not raw:
        return None
    try:
        items = json.loads(raw)["message"]["items"]
    except Exception:
        return None
    best, best_score = None, 0.0
    for it in items:
        t = (it.get("title") or [""])[0]
        s = title_match(title, t)
        if s > best_score:
            best, best_score = it, s
    if not best or best_score < 0.88:
        return None
    yr = None
    parts = (best.get("issued") or {}).get("date-parts") or [[]]
    if parts and parts[0] and parts[0][0]:
        yr = parts[0][0]
    return {
        "citations": best.get("is-referenced-by-count", -1),
        "doi": best.get("DOI", ""),
        "venue": (best.get("container-title") or [""])[0],
        "year": yr,
    }


# ------------------------------------------------------------------- gate


def gate(text: str) -> tuple[bool, str]:
    if not SIGNAL.search(text):
        return False, "no-signal-term"
    if not TASK.search(text):
        return False, "no-task-term"
    if OFFTOPIC.search(text):
        return False, "off-topic"
    return True, "ok"


def arxiv_gate() -> list[dict]:
    log("relevance gate")
    raw = harvest_arxiv()
    kept, rej = [], {}
    for w in raw.values():
        ok, reason = gate(f"{w['title']} {w['abstract']}")
        if ok:
            kept.append(w)
        else:
            rej[reason] = rej.get(reason, 0) + 1
    log(f"gate passed: {len(kept)} / {len(raw)}   rejected: {rej}")
    return kept


def enrich(cands: list[dict]) -> None:
    log(f"Crossref citation enrichment for {len(cands)} candidates")
    found = 0
    for i, c in enumerate(cands, 1):
        m = crossref_citations(c["title"])
        if m:
            found += 1
            c["citations"] = m["citations"]
            if not c.get("doi"):
                c["doi"] = m["doi"]
            if c.get("venue") == "arXiv preprint" and m["venue"]:
                c["venue"] = m["venue"]
        else:
            c["citations"] = 0
        time.sleep(CROSSREF_PAUSE)
        if i % 50 == 0:
            log(f"  {i}/{len(cands)} enriched ({found} matched)")
    log(f"citation matches: {found}/{len(cands)}")


# -------------------------------------------------------------- download


def valid_pdf(p: Path) -> bool:
    try:
        if p.stat().st_size < 10_000:
            return False
        with p.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def pdftotext(path: Path) -> str:
    """Body text of one PDF, for second-pass verification.

    Raw pdftotext (no -layout): -layout interleaves two-column bodies and
    shreds the sentences the record generator quotes.
    """
    try:
        r = subprocess.run(
            ["pdftotext", "-q", str(path), "-"], capture_output=True, timeout=120
        )
        return r.stdout.decode("utf-8", "replace")
    except Exception:
        return ""


def NEURO_RELEVANCE(title: str, hay: str) -> str:
    """neuroevolution | adjacent | off-topic, from the paper's own text."""
    t = (title or "").lower()
    neuro = any(c in hay for c in SIGNAL_TERMS) or any(c in t for c in SIGNAL_TERMS)
    off = [c for c in OFF_TERMS if c in t]
    if off and not any(c in t for c in ("neural", "network", "controller", "policy")):
        return "off-topic"
    if neuro and not off:
        return "neuroevolution"
    return "adjacent"


def fetch(url: str, dest: Path) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={**UA, "Accept": "application/pdf,*/*"}),
            timeout=PDF_TIMEOUT,
        ) as r:
            ctype = r.headers.get("Content-Type", "")
            data = r.read()
        if not data.startswith(b"%PDF-"):
            return False, f"not a pdf (content-type={ctype})"
        if len(data) < 10_000:
            return False, f"too small ({len(data)}B)"
        dest.write_bytes(data)
        return True, f"{len(data)}B"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:50]}"


# ------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--no-landscape", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    cands = arxiv_gate()
    enrich(cands)
    cands.sort(key=lambda c: (-c["citations"], -(c.get("year") or 0), c["title"]))
    log(
        f"top 5 by citation: {[(c['citations'], c['title'][:34]) for c in cands[:5]]}"
    )

    manifest: list[dict] = []
    failed: list[list[str]] = []

    def job(c: dict) -> dict | None:
        dest = OUT / f"{c['arxiv']}.pdf"
        if valid_pdf(dest):
            return c
        time.sleep(ARXIV_PAUSE / DOWNLOAD_WORKERS)
        ok, why = fetch(c["pdfUrl"], dest)
        if ok and valid_pdf(dest):
            return c
        dest.unlink(missing_ok=True)
        failed.append([c["arxiv"], why])
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
    rows = [
        {
            "id": f"N{i:03d}",
            "arxiv": c["arxiv"],
            "doi": c.get("doi", ""),
            "title": c["title"],
            "year": c.get("year"),
            "venue": c.get("venue", ""),
            "authors": c.get("authors", []),
            "citations": c["citations"],
            "fileName": f"{c['arxiv']}.pdf",
            "pdfUrl": c["pdfUrl"],
        }
        for i, c in enumerate(manifest, 1)
    ]
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "download_summary.json").write_text(
        json.dumps(
            {"success": [r["arxiv"] for r in rows], "failed": sorted(failed)}, indent=2
        )
        + "\n"
    )

    if not args.no_landscape:
        try:
            land = harvest_crossref_landscape()
            held = {norm_title(c["title"]) for c in manifest}
            un = [
                {
                    "title": w["title"],
                    "citations": w["citations"],
                    "year": w["year"],
                    "venue": w["venue"],
                    "doi": w["doi"],
                    "reason": "no open-access copy located",
                }
                for w in land
                if norm_title(w["title"]) not in held
            ]
            un.sort(key=lambda w: -w["citations"])
            (OUT / "unavailable.json").write_text(json.dumps(un[:400], indent=2) + "\n")
            log(f"unavailable catalog: {len(un)}")
        except Exception as e:
            log(f"landscape failed (non-fatal): {type(e).__name__}: {e}")

    log(f"DONE manifest={len(rows)} failed={len(failed)} pdfs={len(list(OUT.glob('*.pdf')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())