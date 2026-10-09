#!/usr/bin/env python3
"""Harvest structured knowledge from the networking corpus (state-diagram step 2-11).

Per paper it extracts, each field grounded in the paper's own text:

  problem            problem-statement sentences (cue match, quoted)
  background         background / related-work sentences (quoted)
  solution           the proposed approach (contribution cues, quoted)
  experiments        evaluation sentences (results section only, quoted)
  limitations        limitation sentences (quoted)
  futureDirections   future-work sentences (quoted)
  technical          datasets / metrics / algorithms detected by keyword rules
  industry           deployment / industry-evidence sentences (quoted)
  citations          DOI / arXiv / venue links already in the manifest

Evidence policy (same as wifi_records.py): the script never authors a claim.
Prose fields are sentences copied out of the paper's own text (title/abstract
for metadata-only papers, full PDF body when one was retrieved), or an honest
"not stated in the retrieved text" when the cue machinery finds nothing.

Source text priority:
  1. full PDF body  (papers with a verified PDF; cached pdftotext output)
  2. OpenAlex abstract from manifest.json (252 of 300 papers carry one)

Outputs
  papers/networking/extracted.json   per-paper extraction, keyed by paper id
  (consumed by networking_kb.py / networking_records.py / networking_analysis.py)

Re-running is safe: PDF text is cached under .cache/networking_text/.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "papers" / "networking"
CACHE = ROOT / ".cache" / "networking_text"
OUT = CORPUS / "extracted.json"
NOT_STATED = "not stated in the retrieved text"

# ------------------------------------------------------------------- cues

PROBLEM_CUES = [
    "however", "challenge", "difficult", "hard to", "suffer", "drawback",
    "limitation", "cannot", "can not", "struggl", "remains", "is limited",
    "are limited", "lack", "bottleneck", "intractable", "prohibitiv",
    "fail", "unreliable", "open problem", "problematic", "does not scale",
    "does not provide", "no existing", "have not been", "has not been",
]
GAP_CUES = [
    "has not been", "have not been", "not been explored", "remains an open",
    "open problem", "little attention", "rarely", "few works", "no prior",
    "unexplored", "underexplored", "still lacking", "does not exist",
    "lack of", "absence of", "never been", "little work", "no study",
]
CONTRIB_CUES = [
    "we propose", "we present", "we introduce", "we develop", "we design",
    "we build", "we construct", "we implement", "we describe", "we formulate",
    "we devise", "we establish", "this paper proposes", "this paper presents",
    "this paper introduces", "this work proposes", "in this paper, we",
    "our contribution", "the main contribution", "we show that", "we demonstrate",
    "to this end, we",
]
RESULT_CUES = [
    "outperform", "outperforms", "improve", "improves", "reduce", "reduces",
    "achieve", "achieves", "obtained", "yield", "yields", "reach", "reaches",
    "state-of-the-art", "state of the art", "accuracy", "latency", "throughput",
    "gain", "gains", "better than", "surpass", "demonstrat", "experiment",
    "evaluation", "results show", "we find that", "significant",
]
LIMIT_CUES = [
    "limitation", "drawback", "future work", "does not", "cannot", "fails",
    "restricted to", "assumes", "only consider", "we leave", "beyond the scope",
    "not address", "shortcoming", "drawbacks", "trade-off", "tradeoff",
]
FUTURE_CUES = [
    "future work", "we plan", "we intend", "remains to", "will be explored",
    "future research", "further research", "promising direction", "next step",
    "we expect", "could be extended", "can be extended", "open question",
    "in the future", "forthcoming work",
]
BACKGROUND_CUES = [
    "prior work", "previous work", "existing work", "recently proposed",
    "has been proposed", "has been studied", "related work", "in contrast",
    "state of the art", "state-of-the-art", "unlike", "earlier work",
    "conventional", "traditional", "standard approach", "widely used",
    "commonly used", "is known as", "was proposed in", "building on",
]
INDUSTRY_CUES = [
    "deployed", "deployment", "in production", "industry", "commercial",
    "data center operators", "isp", "cloud provider", "standardiz", "ietf",
    "rfc ", "operator", "operator networks", "practical experience",
    "real-world deployment", "at scale", "google", "facebook", "microsoft",
    "amazon", "netflix", "cloudflare", "akamai", "baidu", "alibaba",
    "field trial", "testbed", "open-source", "open source", "product",
]

# Section headings for full-text extraction.
EXPERIMENT_SECTION_RE = re.compile(
    r"\n\s*(?:\d+(?:\.\d+)*\s*[.)]?\s*)?"
    r"(?:EXPERIMENTS?|EVALUATION|RESULTS?|SIMULATIONS?|PERFORMANCE EVALUATION"
    r"|EXPERIMENTAL RESULTS|EMPIRICAL (?:RESULTS|EVALUATION)"
    r"|NUMERICAL RESULTS?|CASE STUDY)\b[^\n]*\n",
    re.I,
)
END_SECTION_RE = re.compile(
    r"\n\s*(?:\d+(?:\.\d+)*\s*[.)]?\s*)?"
    r"(?:CONCLUSIONS?|REFERENCES|APPENDIX|ACKNOWLEDG)\b",
    re.I,
)
FUTURE_SECTION_RE = re.compile(
    r"\n\s*(?:\d+(?:\.\d+)*\s*[.)]?\s*)?"
    r"(?:CONCLUSIONS?|DISCUSSION AND CONCLUSIONS?|FUTURE WORK"
    r"|CONCLUSION AND FUTURE WORK|CONCLUDING REMARKS)\b[^\n]*\n",
    re.I,
)

# Front-matter junk that survives pdftotext as ordinary sentences.
JUNK_RE = re.compile(
    r"CCS CONCEPTS|\bKEYWORDS\b|Index Terms|ACM Reference|Reference Format"
    r"|\bDOI\b|arXiv:\d|This work is licensed|Permission to make"
    r"|Downloaded from|Preprint\. |under review|copyright|©|\bISBN\b"
    r"|Proceedings of the \d|https?://|@\w+\.\w+|Peer-review"
    r"|Subject descriptors|Received \d|Accepted \d|Published \d"
    r"|\bFig\.?\s*\d|\bFigure\s*\d|\bTable\s*\d|^\s*\d+\s*$",
    re.I,
)

# Sentences that describe general practice rather than this paper.
FRAMING_RE = re.compile(
    r"^(?:often|typically|usually|commonly|generally|frequently|in general|many)\b"
    r"|(?:approaches|methods|works|algorithms|systems|researchers)\s+"
    r"(?:often|typically|usually|commonly|generally|frequently)\b"
    r"|has been (?:widely |extensively |long )?(?:studied|proposed|explored)"
    r"|existing (?:approaches|methods|works)|prior (?:work|works|approaches)",
    re.I,
)
FIRST_PERSON_RE = re.compile(
    r"\b(?:we|our|ours|us)\b|\bthis (?:paper|work|article|study|section)\b", re.I
)

# ------------------------------------------------------- technical detectors

DATASET_RULES: list[tuple[str, str]] = [
    (r"\bCIFAR[- ]?(10|100)\b", "CIFAR-10/100"),
    (r"\bImageNet\b", "ImageNet"),
    (r"\bMNIST\b", "MNIST"),
    (r"\bUNSW[- ]NB15\b", "UNSW-NB15"),
    (r"\bNSL[- ]?KDD\b", "NSL-KDD"),
    (r"\bKDD[- ]?Cup\b", "KDD Cup"),
    (r"\bCICIDS\b", "CICIDS"),
    (r"\bCAIDA\b", "CAIDA traces"),
    (r"\bMAWI\b", "MAWI traces"),
    (r"\bAbilene\b", "Abilene topology"),
    (r"\bGEANT\b", "GEANT topology"),
    (r"\bInternet2\b", "Internet2"),
    (r"\bMininet\b", "Mininet"),
    (r"\bOMNeT\+\+\b", "OMNeT++"),
    (r"\bNS-?3\b", "ns-3"),
    (r"\bns-2\b", "ns-2"),
    (r"\bMATLAB\b", "MATLAB"),
    (r"\bTensorFlow\b", "TensorFlow"),
    (r"\bPyTorch\b", "PyTorch"),
    (r"\bcorenet\b|\bCoreNet\b", "CoreNet"),
    (r"\bFANET\b", "FANET traces"),
    (r"\bSWaT\b", "SWaT testbed"),
    (r"\bWADI\b", "WADI testbed"),
    (r"\bN-BaIoT\b", "N-BaIoT"),
    (r"\bBoT[- ]IoT\b", "Bot-IoT"),
    (r"\bToN_IoT\b", "ToN-IoT"),
    (r"\bCICIoT2023\b", "CICIoT2023"),
]

METRIC_RULES: list[tuple[str, str]] = [
    (r"\bthroughput\b", "throughput"),
    (r"\blatency\b|\bdelay\b", "latency / delay"),
    (r"\bp[- ]?99\b|99th percentile", "tail latency (p99)"),
    (r"\bjitter\b", "jitter"),
    (r"\bpacket loss\b", "packet loss"),
    (r"\butilization\b", "link / resource utilization"),
    (r"\bfairness\b|\bJain'?s\b", "fairness (Jain index)"),
    (r"\benergy (efficiency|consumption)\b", "energy efficiency"),
    (r"\baccuracy\b", "accuracy"),
    (r"\bprecision\b|\brecall\b|\bF1\b", "precision / recall / F1"),
    (r"\bAUC\b|ROC", "AUC / ROC"),
    (r"\bfalse positive rate\b|\bFPR\b", "false positive rate"),
    (r"\bgoodput\b", "goodput"),
    (r"\bPSNR\b|\bSSIM\b", "PSNR / SSIM"),
    (r"\bconvergence\b", "convergence"),
    (r"\bscalability\b", "scalability"),
    (r"\bdelivery ratio\b", "delivery ratio"),
    (r"\brouting overhead\b", "routing overhead"),
]

ALGORITHM_RULES: list[tuple[str, str]] = [
    (r"\bDijkstra\b", "Dijkstra shortest path"),
    (r"\bA\*|A-star\b", "A* search"),
    (r"\bQ[- ]?learning\b", "Q-learning"),
    (r"\breinforcement learning\b", "reinforcement learning"),
    (r"\bdeep (neural|reinforcement)\b", "deep learning"),
    (r"\bconvolutional neural network\b|\bCNN\b", "CNN"),
    (r"\blong short-term memory\b|\bLSTM\b", "LSTM"),
    (r"\btransformer\b", "transformer"),
    (r"\bgenetic algorithm\b", "genetic algorithm"),
    (r"\bpso\b|particle swarm", "particle swarm optimisation"),
    (r"\bannealing\b", "simulated annealing"),
    (r"\bant colony\b", "ant colony optimisation"),
    (r"\bsupport vector machine\b|\bSVM\b", "SVM"),
    (r"\brandom forest\b", "random forest"),
    (r"\bgradient boost|\bXGBoost\b", "gradient boosting"),
    (r"\bk[- ]means\b", "k-means clustering"),
    (r"\bMarkov\b", "Markov model"),
    (r"\bBayesian\b", "Bayesian inference"),
    (r"\bsimplex\b", "linear programming (simplex)"),
    (r"\blinear programming\b|\bLP relaxation\b", "linear programming"),
    (r"\bHungarian algorithm\b", "Hungarian assignment"),
    (r"\bKarmarkar\b", "interior-point method"),
    (r"\bKalman filter\b", "Kalman filter"),
    (r"\bgame theory\b|\bNash equilibrium\b", "game-theoretic model"),
    (r"\bsimulated annealing\b", "simulated annealing"),
    (r"\bdeep reinforcement learning\b|\bDRL\b", "deep RL"),
    (r"\bfederated learning\b", "federated learning"),
    (r"\badversarial\b", "adversarial training / GAN"),
]

DOMAIN_RULES: list[tuple[str, str]] = [
    (r"\bcongestion control\b|\b拥塞", "congestion-control"),
    (r"\brouting\b|\brouter\b|\bforwarding\b", "routing"),
    (r"\bsoftware.defined networking\b|\bSDN\b", "sdn"),
    (r"\bnetwork function virtual|\bNFV\b", "nfv"),
    (r"\bdata ?cent(er|re)s?\b", "datacenter-networking"),
    (r"\bedge comput|\bfog comput", "edge-computing"),
    (r"\binternet of things\b|\bIoT\b", "iot-networking"),
    (r"\b5g\b|\b6g\b|\bmmWave\b|\bmmwave\b", "5g-6g"),
    (r"\bwireless|\bWi-?Fi\b|\bWLAN\b|\b802\.11", "wireless"),
    (r"\bmobil|\bcellular\b|\bV2X\b|\bvehicular\b|\b5g", "mobile-networking"),
    (r"\bcontent delivery|\bCDN\b|\bcaching\b", "cdn-caching"),
    (r"\bpeer.to.peer\b|\bp2p\b|\bblockchain\b", "p2p-overlay"),
    (r"\bintrusion detection\b|\bcybersec|\bsecurity\b", "network-security"),
    (r"\bnetwork traffic\b|\btraffic classif|\btraffic classificat", "traffic-analysis"),
    (r"\bquantum (network|communication)", "quantum-networking"),
    (r"\bsatellite\b|\bLEO\b", "satellite-networking"),
    (r"\bsemantic commun", "semantic-communication"),
    (r"\bdigital twin", "digital-twin-network"),
    (r"\bmetaverse\b", "metaverse-networking"),
    (r"\bdomain name system\b|\bDNS\b", "dns"),
    (r"\bprogrammable data plane|\bp4\b", "programmable-data-plane"),
    (r"\bnetwork measur|\btelemetry\b|\bmonitoring\b", "measurement"),
    (r"\bzero.trust\b", "zero-trust"),
    (r"\bdistributed (ledger|ledger)|consensus protocol", "consensus-networking"),
    (r"\bbackhaul\b|\bfronthaul\b", "backhaul"),
    (r"\bmachine.to.machine|\bM2M\b", "m2m"),
    (r"\bdelay.tolerant|\bDTN\b", "delay-tolerant"),
    (r"\bsensor network", "sensor-networks"),
    (r"\bmultipath\b|\bMPTCP\b", "multipath"),
    (r"\bTCP\b|\bQUIC\b", "transport-protocols"),
]


# ------------------------------------------------------------- text helpers


def pdf_text(path: pathlib.Path) -> str:
    """Body text of a PDF, cached per file (raw pdftotext, no -layout)."""
    if not path.exists():
        return ""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / (path.stem + ".txt")
    if out.exists() and out.stat().st_size > 200:
        return out.read_text(errors="replace")
    try:
        r = subprocess.run(
            ["pdftotext", "-q", str(path), "-"], capture_output=True, timeout=120
        )
        text = r.stdout.decode("utf-8", "replace")
    except Exception:
        text = ""
    out.write_text(text)
    return text


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_REF_HEAD = re.compile(
    r"\n\s*(?:\d+(?:\.\d+)*\s*[.)]?\s*)?(?:REFERENCES|References|Bibliography)\s*\n",
    re.I,
)


def sentences(text: str) -> list[str]:
    """Sentence list with reference blocks, junk and over-short units dropped."""
    if not text:
        return []
    # Drop the reference block: its sentences are other papers' prose.
    m = _REF_HEAD.search(text)
    if m:
        text = text[: m.start()]
    out: list[str] = []
    for raw in _SENT_SPLIT.split(text.replace("\n", " ")):
        s = re.sub(r"\s+", " ", raw).strip()
        if len(s) < 45 or len(s) > 600:
            continue
        if JUNK_RE.search(s):
            continue
        out.append(s)
    return out


def is_framing(sent: str) -> bool:
    if FRAMING_RE.search(sent):
        return not FIRST_PERSON_RE.search(sent)
    return False


def pick(sents: list[str], cues: list[str], limit: int, min_len: int = 55) -> list[str]:
    """First `limit` sentences carrying a cue, framing/junk filtered out."""
    hits: list[str] = []
    low_cues = [c.lower() for c in cues]
    for s in sents:
        low = s.lower()
        if any(c in low for c in low_cues) and len(s) >= min_len and not is_framing(s):
            hits.append(s)
        if len(hits) >= limit:
            break
    return hits


def uniq(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        k = x.lower()[:80]
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def clip(s: str, n: int = 420) -> str:
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return cut + " …"


def section(text: str, start_re: re.Pattern, stop_re: re.Pattern) -> list[str]:
    m = start_re.search(text)
    if not m:
        return []
    rest = text[m.end():]
    stop = stop_re.search(rest)
    if stop:
        rest = rest[: stop.start()]
    return sentences(rest)


def detect(text: str, rules: list[tuple[str, str]], limit: int) -> list[str]:
    low = text.lower()
    hits: list[str] = []
    for pat, label in rules:
        if label in hits:
            continue
        try:
            if re.search(pat, low, re.I):
                hits.append(label)
        except re.error:
            continue
        if len(hits) >= limit:
            break
    return hits


# ------------------------------------------------------------- extraction


def build_record(row: dict, body: str, abstract: str) -> dict:
    """Extract the eleven state-diagram fields for one paper."""
    # Prefer the full body when a PDF exists; else the OpenAlex abstract.
    full = bool(body)
    source_text = body if full else abstract
    sents = sentences(source_text)

    problem = pick(sents, PROBLEM_CUES, 2)
    gap = pick(sents, GAP_CUES, 1)
    problem = uniq(problem + gap)[:3]
    background = pick(sents, BACKGROUND_CUES, 3)
    solution = pick(sents, CONTRIB_CUES, 3)
    if not solution and abstract:
        # abstract's own contribution sentence when body cues miss
        solution = pick(sentences(abstract), CONTRIB_CUES, 2)

    experiments: list[str] = []
    if full:
        exp_sents = section(body, EXPERIMENT_SECTION_RE, END_SECTION_RE)
        experiments = pick(exp_sents, RESULT_CUES, 3)
    if not experiments and abstract:
        experiments = pick(sentences(abstract), RESULT_CUES, 2)

    limitations = pick(sents, LIMIT_CUES, 2)
    future_sents = section(body, FUTURE_SECTION_RE, END_SECTION_RE) if full else sents
    future = pick(future_sents, FUTURE_CUES, 2) or pick(sents, FUTURE_CUES, 2)
    industry = pick(sents, INDUSTRY_CUES, 2)

    # Technical details are keyword detections over whichever text we hold.
    datasets = detect(source_text, DATASET_RULES, 6)
    metrics = detect(source_text, METRIC_RULES, 8)
    algorithms = detect(source_text, ALGORITHM_RULES, 8)
    domains = detect(
        (row.get("title", "") + " " + abstract + " " + body[:60000]),
        DOMAIN_RULES,
        5,
    )

    def field(xs: list[str]) -> str:
        if not xs:
            return NOT_STATED
        return " ".join(clip(x) for x in xs)

    return {
        "id": row["id"],
        "doi": row.get("doi", ""),
        "title": row["title"],
        "year": row.get("year"),
        "venue": row.get("venue", ""),
        "authors": row.get("authors", [])[:12],
        "citations": row.get("citations", 0),
        "fileName": row.get("fileName", ""),
        "hasFullText": full,
        "source": "full-text" if full else ("abstract" if abstract else "metadata-only"),
        "problem": field(problem),
        "background": field(background),
        "solution": field(solution),
        "experiments": field(experiments),
        "limitations": field(limitations),
        "futureDirections": field(future),
        "industryEvidence": field(industry),
        "datasets": datasets,
        "metrics": metrics,
        "algorithms": algorithms,
        "domains": domains,
        # Every field above is quoted from the paper, or NOT_STATED.
        "evidenceQuotes": {
            "problem": problem,
            "solution": solution,
            "experiments": experiments,
            "limitations": limitations,
            "future": future,
            "industry": industry,
        },
        "links": {
            "doi": f"https://doi.org/{row.get('doi')}" if row.get("doi") else "",
            "arxiv": "",
            "pdf": row.get("fileName", ""),
        },
    }


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    CACHE.mkdir(parents=True, exist_ok=True)

    def work(row: dict) -> dict:
        body = ""
        fname = row.get("fileName") or ""
        if fname:
            body = pdf_text(CORPUS / fname)
        return build_record(row, body, row.get("abstract") or "")

    with ThreadPoolExecutor(max_workers=6) as ex:
        records = list(ex.map(work, manifest))

    OUT.write_text(json.dumps(records, indent=1, ensure_ascii=False))
    full = sum(1 for r in records if r["hasFullText"])
    abs_ = sum(1 for r in records if r["source"] == "abstract")
    print(f"extracted {len(records)} papers: {full} full-text, {abs_} abstract-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
