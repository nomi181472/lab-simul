#!/usr/bin/env python3
"""Corpus validation + knowledge construction for the networking lab.

Implements the diagram's "CORPUS VALIDATION & KNOWLEDGE CONSTRUCTION" state:

  ValidateHarvest   every record must carry id/title/year/links; evidence quotes
                    must be non-empty when the prose field is not NOT_STATED
  ResolveLinks      DOI resolves to a doi.org URL; missing DOI recorded honestly
  Normalize         domain vocabulary (DOMAIN_RULES from networking_extract) is
                    the canonical terminology; each paper gets canonical domain ids
  Deduplicate       DOI first, then normalized title; survivors keep the highest
                    citation count and merge nothing silently — dropped ids are
                    recorded in the audit block
  Corpus            final corpus stats (by year, venue, domain, source depth)
  ProblemDictionary canonical problem clusters (domain -> problem statement set)
  SolutionDictionary canonical solution families (algorithm/approach detections)
  Relationships     paper->problem, paper->solution, citation (shared-domain) edges

Outputs (all machine-generated, every entry cites paper ids):
  papers/networking/validated.json
  src/labs/networking/data/kb.ts
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "papers" / "networking"
OUT_TS = ROOT / "src" / "labs" / "networking" / "data" / "kb.ts"
NOT_STATED = "not stated in the retrieved text"

# Canonical domain vocabulary: id -> display metadata.
DOMAINS: list[tuple[str, str, str]] = [
    # id, label, blurb
    ("congestion-control", "Congestion Control", "End-to-end and router-assisted schemes that keep offered load inside the network's carrying capacity."),
    ("routing", "Routing & Forwarding", "Path computation and packet forwarding: shortest path, traffic engineering, source routing, SDN-controlled forwarding."),
    ("sdn", "Software-Defined Networking", "Control/data-plane separation, OpenFlow-style flow tables, network operating systems."),
    ("nfv", "Network Function Virtualization", "Running middleboxes (firewall, NAT, DPI) as software on commodity servers."),
    ("datacenter-networking", "Datacenter Networks", "Clos/fat-tree topologies, equal-cost multipath, RDMA, transport inside the DC."),
    ("edge-computing", "Edge / Fog Computing", "Compute placed near the user; offloading, service placement, latency-critical workloads."),
    ("iot-networking", "IoT Networking", "Constrained-device protocols, low-power networks, massive machine-type connectivity."),
    ("5g-6g", "5G / 6G", "Cellular generation research: mmWave, network slicing, massive MIMO, 6G visions."),
    ("wireless", "Wireless & WiFi", "802.11, spectrum, interference, link adaptation, WiFi sensing."),
    ("mobile-networking", "Mobile & Vehicular", "Handover, mobility management, vehicular V2X, drones/UA V networks."),
    ("cdn-caching", "CDN & Caching", "Content delivery, edge caches, request routing, video distribution."),
    ("p2p-overlay", "P2P & Overlay Networks", "Peer-to-peer overlays, DHTs, blockchain networking, gossip."),
    ("network-security", "Network Security", "Intrusion detection, DDoS, traffic analysis attacks, zero trust."),
    ("traffic-analysis", "Traffic Measurement & Classification", "Traffic engineering, flow classification, anomaly detection from traces."),
    ("quantum-networking", "Quantum Networking", "Quantum key distribution links, entanglement distribution, quantum internet."),
    ("satellite-networking", "Satellite Networks", "LEO constellations, inter-satellite links, non-terrestrial networks."),
    ("semantic-communication", "Semantic Communication", "Task-oriented communication transmitting meaning rather than bits."),
    ("digital-twin-network", "Digital Twin Networks", "Mirror networks as digital twins for control and prediction."),
    ("metaverse-networking", "Metaverse Networking", "Rendering-aware delivery, avatar telepresence, immersive transport."),
    ("dns", "DNS & Name Resolution", "Domain name system, anycast, DNS security and performance."),
    ("programmable-data-plane", "Programmable Data Planes", "P4, protocol-independent switching, in-network computing."),
    ("measurement", "Network Measurement & Telemetry", "Active/passive measurement, path inference, observability."),
    ("zero-trust", "Zero-Trust Architecture", "Identity-centric perimeterless network access control."),
    ("sensor-networks", "Sensor Networks", "Wireless sensor networks, duty cycling, in-network aggregation."),
    ("multipath", "Multipath Transport", "MPTCP and related: striping, reordering, scheduling."),
    ("transport-protocols", "Transport Protocols", "TCP/QUIC-family: reliability, head-of-line blocking, handshake."),
    ("backhaul", "Backhaul & Fronthaul", "Radio backhaul, Xhaul, transport for RAN disaggregation."),
    ("delay-tolerant", "Delay-Tolerant Networking", "Store-carry-forward, intermittent links, challenged networks."),
    ("m2m", "Machine-to-Machine", "M2M/MTC communication patterns and random access."),
    ("consensus-networking", "Consensus Protocols", "Distributed consensus over networks (beyond the ledger itself)."),
    ("device-free", "Device-Free Sensing", "Sensing presence/motion from ambient wireless signals."),
    ("application-network", "Application-Layer Networks", "HTTP-family, gRPC/RPC, service mesh, API gateways, WebRTC."),
]
DOMAIN_ID = {d[0]: d for d in DOMAINS}

# Fallback: papers with no domain detection still enter the corpus with an
# honest "general networking" bucket rather than being dropped.
GENERIC = ("general-networking", "General Networking", "Networking work not reducible to a single canonical domain.")

# Solution families: canonical id -> (label, detection pattern over solution text)
SOLUTION_FAMILIES: list[tuple[str, str, str]] = [
    ("shortest-path-routing", "Shortest-Path Routing", r"dijkstra|shortest path|link state"),
    ("centralized-control", "Centralized / SDN Control", r"\bsdn\b|centralized controller|openflow|sdn-based"),
    ("reinforcement-learning", "Reinforcement-Learning Control", r"reinforcement learning|\bq[- ]?learning|deep rl|\bdrl\b"),
    ("deep-learning", "Deep-Learning Models", r"deep (neural|learning)|convolutional|\bcnn\b|\blstm\b|transformer"),
    ("ml-classification", "Classical ML Classification", r"support vector|\bsvm\b|random forest|naive bayes|decision tree|gradient boost|xgboost"),
    ("evolutionary-optimization", "Evolutionary / Swarm Optimisation", r"genetic algorithm|particle swarm|\bpso\b|ant colony|annealing|evolutionary"),
    ("mathematical-optimization", "Formal Optimisation", r"linear programming|integer programming|convex optim|simplex|optimization problem|formulated as"),
    ("game-theoretic", "Game-Theoretic Models", r"game theory|nash equilibrium|mechanism design|auction"),
    ("caching-placement", "Caching & Placement", r"caching|cache placement|content placement|replication"),
    ("network-slicing", "Network Slicing", r"network slicing|slice orchestration|slice"),
    ("digital-twin-control", "Digital-Twin Control", r"digital twin"),
    ("federated-learning", "Federated / Collaborative Learning", r"federated learning|collaborative learning"),
    ("gossip-epidemic", "Gossip / Epidemic Protocols", r"gossip|epidemic"),
    ("consensus", "Consensus Protocols", r"consensus|paxos|raft\b|byzantine|pbft"),
    ("compression-semantic", "Compression / Semantic Coding", r"compression|source coding|semantic|joint source"),
    ("queueing-theory", "Queueing-Theoretic Analysis", r"queueing|queuing|m/m/1|markov chain"),
    ("control-theoretic", "Control-Theoretic Design", r"control theoretic|lyapunov|pid control|feedback control"),
    ("probabilistic-modeling", "Probabilistic / Bayesian Models", r"bayesian|probabilistic model|markov|hidden markov|hmm"),
    ("clustering", "Clustering & Grouping", r"k[- ]means|clustering|cluster analysis"),
    ("blockchain-ledger", "Ledger-Based Trust", r"blockchain|smart contract|ledger"),
]


def load() -> list[dict]:
    return json.loads((CORPUS / "extracted.json").read_text())


# ---------------------------------------------------------------- validate


def validate(recs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (survivors, problems). Each problem is an audit record."""
    problems: list[dict] = []
    for r in recs:
        for key in ("id", "title", "year", "citations", "links"):
            if r.get(key) in (None, "", {}):
                problems.append({"paperId": r.get("id"), "issue": f"missing {key}"})
        for field in ("problem", "solution", "experiments", "limitations"):
            val = r.get(field, "")
            if val != NOT_STATED and not val.strip():
                problems.append({"paperId": r.get("id"), "issue": f"empty {field} with claims"})
        if r.get("hasFullText") and not r.get("evidenceQuotes", {}).get("solution"):
            # full text but no contribution sentence found — record, don't drop
            problems.append({"paperId": r.get("id"), "issue": "full text with no contribution quote"})
    return recs, problems


def resolve_links(recs: list[dict]) -> None:
    for r in recs:
        doi = r.get("doi", "")
        r["links"]["doi"] = f"https://doi.org/{doi}" if doi else ""
        if not doi and not r["links"].get("pdf"):
            r["linksStatus"] = "no-resolvable-link"
        elif doi:
            r["linksStatus"] = "doi-resolved"
        else:
            r["linksStatus"] = "pdf-only"


def normalize(recs: list[dict]) -> None:
    for r in recs:
        doms = [d for d in r.get("domains", []) if d in DOMAIN_ID]
        r["domains"] = doms[:4] if doms else [GENERIC[0]]
        # solution families detected over solution + experiments + algorithm list
        text = " ".join(
            [
                r.get("solution", ""),
                r.get("experiments", ""),
                " ".join(r.get("algorithms", [])),
            ]
        ).lower()
        fams: list[str] = []
        for fid, _label, pat in SOLUTION_FAMILIES:
            if fams and len(fams) >= 3:
                break
            try:
                if re.search(pat, text):
                    fams.append(fid)
            except re.error:
                continue
        r["solutionFamilies"] = fams


def deduplicate(recs: list[dict]) -> tuple[list[dict], list[dict]]:
    seen: dict[str, dict] = {}
    dropped: list[dict] = []
    for r in recs:
        key = r.get("doi") or re.sub(r"[^a-z0-9]", "", r["title"].lower())[:70]
        if key not in seen:
            seen[key] = r
            continue
        keep, cur = seen[key], r
        if (cur["citations"], cur["year"] or 0) > (keep["citations"], keep["year"] or 0):
            seen[key] = cur
            dropped.append({"kept": cur["id"], "dropped": keep["id"]})
        else:
            dropped.append({"kept": keep["id"], "dropped": cur["id"]})
    return list(seen.values()), dropped


# ------------------------------------------------------------ dictionaries


def build_dictionaries(recs: list[dict]) -> dict:
    # problem dictionary: domain -> aggregated problem statements + paper ids
    problems: dict[str, dict] = {}
    for r in recs:
        for d in r["domains"]:
            slot = problems.setdefault(
                d,
                {
                    "domainId": d,
                    "label": DOMAIN_ID.get(d, GENERIC)[1],
                    "blurb": DOMAIN_ID.get(d, GENERIC)[2],
                    "paperIds": [],
                    "statements": [],
                    "firstYear": 9999,
                    "lastYear": 0,
                    "totalCitations": 0,
                },
            )
            slot["paperIds"].append(r["id"])
            if r.get("year"):
                slot["firstYear"] = min(slot["firstYear"], r["year"])
                slot["lastYear"] = max(slot["lastYear"], r["year"])
            slot["totalCitations"] += r.get("citations", 0)
            if r.get("problem") and r["problem"] != NOT_STATED:
                slot["statements"].append(
                    {"paperId": r["id"], "year": r.get("year"), "text": clip(r["problem"])}
                )
    # solution dictionary: family -> papers + statements
    solutions: dict[str, dict] = {}
    for r in recs:
        for f in r.get("solutionFamilies", []):
            label = next(lbl for fid, lbl, _ in SOLUTION_FAMILIES if fid == f)
            slot = solutions.setdefault(
                f,
                {
                    "familyId": f,
                    "label": label,
                    "paperIds": [],
                    "statements": [],
                    "firstYear": 9999,
                    "lastYear": 0,
                },
            )
            slot["paperIds"].append(r["id"])
            if r.get("year"):
                slot["firstYear"] = min(slot["firstYear"], r["year"])
                slot["lastYear"] = max(slot["lastYear"], r["year"])
            if r.get("solution") and r["solution"] != NOT_STATED:
                slot["statements"].append(
                    {"paperId": r["id"], "year": r.get("year"), "text": clip(r["solution"])}
                )
    # relationships: shared-domain paper pairs (co-occurrence edges)
    by_domain: dict[str, list[str]] = defaultdict(list)
    for r in recs:
        for d in r["domains"]:
            by_domain[d].append(r["id"])
    edges: list[dict] = []
    for d, ids in by_domain.items():
        # only report adjacency counts; full pairwise of 300 nodes is noise
        if len(ids) >= 2:
            edges.append({"domainId": d, "paperCount": len(ids), "paperIds": ids})
    return {
        "domains": [
            {"id": d[0], "label": d[1], "blurb": d[2],
             "paperIds": problems.get(d[0], {}).get("paperIds", []),
             "firstYear": problems.get(d[0], {}).get("firstYear", 0) or None,
             "lastYear": problems.get(d[0], {}).get("lastYear", 0) or None,
             "totalCitations": problems.get(d[0], {}).get("totalCitations", 0),
             "statements": problems.get(d[0], {}).get("statements", [])[:12]}
            for d in DOMAINS if d[0] in problems
        ],
        "solutions": [
            {"familyId": s[0], "label": s[1],
             "paperIds": solutions[s[0]]["paperIds"],
             "firstYear": solutions[s[0]]["firstYear"] if solutions[s[0]]["firstYear"] < 9999 else 0,
             "lastYear": solutions[s[0]]["lastYear"],
             "statements": solutions[s[0]]["statements"][:10]}
            for s in SOLUTION_FAMILIES if s[0] in solutions
        ],
        "relationships": edges,
    }


def clip(s: str, n: int = 400) -> str:
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0] + " …"


def corpus_stats(recs: list[dict]) -> dict:
    years = Counter(r["year"] for r in recs if r.get("year"))
    venues = Counter(r.get("venue") or "— (venue not recorded)" for r in recs)
    sources = Counter(r["source"] for r in recs)
    links = Counter(r.get("linksStatus", "?") for r in recs)
    return {
        "total": len(recs),
        "byYear": dict(sorted(years.items())),
        "topVenues": venues.most_common(12),
        "bySource": dict(sources),
        "byLinkStatus": dict(links),
        "citationsTotal": sum(r.get("citations", 0) for r in recs),
        "fullText": sum(1 for r in recs if r["hasFullText"]),
        "withAbstract": sum(1 for r in recs if r["source"] == "abstract"),
        "metadataOnly": sum(1 for r in recs if r["source"] == "metadata-only"),
    }


# ---------------------------------------------------------------- emit TS


def ts(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False)


def main() -> int:
    recs = load()
    recs, problems = validate(recs)
    resolve_links(recs)
    normalize(recs)
    survivors, dropped = deduplicate(recs)
    kb = build_dictionaries(survivors)
    stats = corpus_stats(survivors)

    (CORPUS / "validated.json").write_text(
        json.dumps(survivors, indent=1, ensure_ascii=False)
    )

    OUT_TS.parent.mkdir(parents=True, exist_ok=True)
    OUT_TS.write_text(
        "// GENERATED by scripts/networking_kb.py — do not edit by hand.\n"
        "// Every domain/solution entry cites the paper ids that support it.\n"
        f"export const KB = {ts(kb)};\n\n"
        f"export const CORPUS_STATS = {ts(stats)};\n\n"
        f"export const AUDIT = {ts({'validationProblems': problems, 'deduped': dropped})};\n"
    )
    print(
        f"validated {len(survivors)} papers (dropped {len(dropped)} dups); "
        f"{len(kb['domains'])} domains, {len(kb['solutions'])} solution families; "
        f"{len(problems)} validation notes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
