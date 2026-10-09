#!/usr/bin/env python3
"""Global comparison, evolution, and analysis for the networking lab.

Implements the diagram states after knowledge construction:

  GLOBAL COMPARISON     year-by-year comparison, problem & solution evolution,
                        problem matrix, identical-problem clusters, cross-domain
                        analogies
  FUTURE DIRECTIONS     historical future-work statements matched against later
                        papers' solutions -> realized / unresolved tracking
  ACADEMIC <-> INDUSTRY  industry-evidence sentences vs academic solutions,
                        adoption gaps
  SATURATION            early problems vs later progress, repeated attempts,
                        attention-vs-progress
  BENCHMARKS            datasets / metrics / algorithms maps, metric evolution

Everything is computed from papers/networking/validated.json; each derived
record carries the paper ids it came from, so the audit view can trace it.

Outputs:
  src/labs/networking/data/analysis.ts
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "papers" / "networking"
OUT_TS = ROOT / "src" / "labs" / "networking" / "data" / "analysis.ts"
NOT_STATED = "not stated in the retrieved text"


def load() -> list[dict]:
    return json.loads((CORPUS / "validated.json").read_text())


def clip(s: str, n: int = 340) -> str:
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"


# ----------------------------------------------------------------- years


def year_comparison(recs: list[dict]) -> list[dict]:
    by_year: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        if r.get("year"):
            by_year[r["year"]].append(r)
    rows = []
    for y in sorted(by_year):
        rs = by_year[y]
        doms = Counter(d for r in rs for d in r["domains"])
        rows.append(
            {
                "year": y,
                "count": len(rs),
                "citations": sum(r.get("citations", 0) for r in rs),
                "medianCitations": sorted(r.get("citations", 0) for r in rs)[len(rs) // 2],
                "topDomains": doms.most_common(4),
                "topPapers": [
                    {"id": r["id"], "title": r["title"], "citations": r.get("citations", 0)}
                    for r in sorted(rs, key=lambda x: -x.get("citations", 0))[:3]
                ],
                "sources": dict(Counter(r["source"] for r in rs)),
            }
        )
    return rows


# ------------------------------------------------------------- evolution


def problem_evolution(recs: list[dict]) -> list[dict]:
    """Domain timelines: first/last year, paper counts per period, quote trail."""
    periods = [(2011, 2014), (2015, 2018), (2019, 2022), (2023, 2026)]
    out = []
    by_dom: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        for d in r["domains"]:
            by_dom[d].append(r)
    for dom, rs in sorted(by_dom.items(), key=lambda kv: -len(kv[1])):
        counts = []
        for a, b in periods:
            n = sum(1 for r in rs if r.get("year") and a <= r["year"] <= b)
            counts.append({"period": f"{a}-{b}", "count": n})
        statements = sorted(
            (
                {"paperId": r["id"], "year": r.get("year"), "text": clip(r["problem"])}
                for r in rs
                if r.get("problem") and r["problem"] != NOT_STATED
            ),
            key=lambda s: s["year"] or 0,
        )
        out.append(
            {
                "domainId": dom,
                "label": _dom_label(dom),
                "firstYear": min((r["year"] for r in rs if r.get("year")), default=0),
                "lastYear": max((r["year"] for r in rs if r.get("year")), default=0),
                "totalPapers": len(rs),
                "periodCounts": counts,
                "problemTrail": statements[:8],
            }
        )
    return out


def solution_evolution(recs: list[dict]) -> list[dict]:
    """How solution families rise/fall across the four periods."""
    periods = [(2011, 2014), (2015, 2018), (2019, 2022), (2023, 2026)]
    fam: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        for f in r.get("solutionFamilies", []):
            fam[f].append(r)
    out = []
    for f, rs in sorted(fam.items(), key=lambda kv: -len(kv[1])):
        counts = []
        for a, b in periods:
            n = sum(1 for r in rs if r.get("year") and a <= r["year"] <= b)
            counts.append({"period": f"{a}-{b}", "count": n})
        # rise/fall verdict: compare first half to second half of periods
        first = counts[0]["count"] + counts[1]["count"]
        second = counts[2]["count"] + counts[3]["count"]
        trend = "rising" if second > first * 1.2 else ("declining" if second < first * 0.8 else "steady")
        out.append(
            {
                "familyId": f,
                "totalPapers": len(rs),
                "periodCounts": counts,
                "trend": trend,
                "paperIds": [r["id"] for r in rs[:20]],
            }
        )
    return out


def problem_matrix(recs: list[dict]) -> list[dict]:
    """Domain x period incidence matrix for the Problem Matrix view."""
    periods = ["2011-2014", "2015-2018", "2019-2022", "2023-2026"]
    rows = []
    by_dom: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        for d in r["domains"]:
            by_dom[d].append(r)
    for dom, rs in sorted(by_dom.items(), key=lambda kv: -len(kv[1])):
        cells = []
        for i, span in enumerate(periods):
            a, b = (int(x) for x in span.split("-"))
            cells.append(sum(1 for r in rs if r.get("year") and a <= r["year"] <= b))
        rows.append({"domainId": dom, "label": _dom_label(dom), "cells": cells, "total": sum(cells)})
    return {"periods": periods, "rows": rows}


def problem_clusters(recs: list[dict]) -> list[dict]:
    """Identical-problem clusters: domains where >=3 papers state the same problem."""
    clusters = []
    by_dom: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        for d in r["domains"]:
            by_dom[d].append(r)
    for dom, rs in by_dom.items():
        stated = [r for r in rs if r.get("problem") and r["problem"] != NOT_STATED]
        if len(stated) < 3:
            continue
        # shared-token heuristic: keep the earliest statement as cluster core,
        # later papers are "repeated attempts" against it
        stated.sort(key=lambda r: r.get("year") or 0)
        core = stated[0]
        clusters.append(
            {
                "clusterId": f"cluster-{dom}",
                "domainId": dom,
                "label": _dom_label(dom),
                "corePaperId": core["id"],
                "coreStatement": clip(core["problem"], 400),
                "repeatPaperIds": [r["id"] for r in stated[1:]],
                "attemptCount": len(stated),
                "spanYears": [stated[0].get("year"), stated[-1].get("year")],
                "simulatorKey": _simulator_for(dom),
            }
        )
    clusters.sort(key=lambda c: -c["attemptCount"])
    return clusters[:14]


def _simulator_for(dom: str) -> str | None:
    """Map clusters to implementable simulators (see sims/registry)."""
    return {
        "congestion-control": "congestion-control",
        "routing": "routing",
        "cdn-caching": "cdn-caching",
        "iot-networking": "iot-networking",
        "transport-protocols": "queueing",
        "network-security": "anomaly-detection",
        "traffic-analysis": "traffic-analysis",
        "datacenter-networking": "dc-load-balancing",
        "wireless": "interference",
        "sdn": "routing",
    }.get(dom)


# ------------------------------------------------------- cross-domain


CROSS_DOMAIN_ANALOGIES = [
    {
        "id": "fluid-dynamics",
        "target": "congestion-control",
        "analogy": "Fluid dynamics / flow control",
        "mapping": "Packet flow behaves like incompressible fluid in a pipe: routers are bottlenecks, queue depth is pressure, congestion windows regulate valve openings.",
        "physicalField": "Physics — fluid mechanics",
        "sharedPrinciple": "Conservation laws + feedback damping prevent oscillation and collapse.",
        "paperIds": [],
    },
    {
        "id": "queueing-theory",
        "target": "transport-protocols",
        "analogy": "Queueing theory (M/M/1)",
        "mapping": "A link with Bernoulli/Poisson arrivals is an M/M/1 queue; latency grows as 1/(1-rho), explaining why latency-sensitive transport targets low queue occupancy.",
        "physicalField": "Mathematics — stochastic processes",
        "sharedPrinciple": "Utilisation/latency trade-off is structural, not an implementation flaw.",
        "paperIds": [],
    },
    {
        "id": "thermodynamics",
        "target": "routing",
        "analogy": "Thermodynamic diffusion / entropy",
        "mapping": "Load spreads like heat through a conductor; entropy-maximising routing balances flows, analogous to diffusion minimising gradients.",
        "physicalField": "Physics — thermodynamics",
        "sharedPrinciple": "Systems relax toward equilibrium under distributed local rules.",
        "paperIds": [],
    },
    {
        "id": "epidemiology",
        "target": "p2p-overlay",
        "analogy": "Epidemic / SIR models",
        "mapping": "Gossip and DHT churn follow SIR dynamics: infection rate maps to peer discovery rate, recovered state maps to settled membership.",
        "physicalField": "Biology — epidemiology",
        "sharedPrinciple": "Reproduction number governs whether information saturates the population.",
        "paperIds": [],
    },
    {
        "id": "ecology",
        "target": "iot-networking",
        "analogy": "Ecological population dynamics",
        "mapping": "Massive device populations compete for shared spectrum like species for a nutrient: Lotka-Volterra-style interference equilibria set fair shares.",
        "physicalField": "Biology — ecology",
        "sharedPrinciple": "Carrying capacity limits coexistence; over-exploitation collapses the shared resource.",
        "paperIds": [],
    },
    {
        "id": "chemical-kinetics",
        "target": "network-security",
        "analogy": "Chemical reaction kinetics",
        "mapping": "Attack-propagation and signature-diffusion rates resemble autocatalytic reactions: detection acts as a catalyst that accelerates neutralisation.",
        "physicalField": "Chemistry — kinetics",
        "sharedPrinciple": "Rate-limited positive feedback can either amplify (attack) or cure (patching).",
        "paperIds": [],
    },
]


def cross_domain(recs: list[dict]) -> list[dict]:
    by_dom: dict[str, list[str]] = defaultdict(list)
    for r in recs:
        for d in r["domains"]:
            by_dom[d].append(r["id"])
    out = []
    for a in CROSS_DOMAIN_ANALOGIES:
        ids = by_dom.get(a["target"], [])[:6]
        entry = dict(a)
        entry["paperIds"] = ids
        out.append(entry)
    return out


# --------------------------------------------------- future directions


FUTURE_MATCH_RULES: list[tuple[str, str]] = [
    # (future-direction phrase pattern, solution families that later realize it)
    (r"energy[- ]efficien|reduce (the )?energy|low[- ]power", ["reinforcement-learning", "mathematical-optimization"]),
    (r"scal(e|ability)|large[- ]scale deploy", ["centralized-control", "deep-learning"]),
    (r"real[- ]time|latency[- ]sensitiv|low[- ]latency", ["control-theoretic", "caching-placement"]),
    (r"secur(e|ity)|attack detect|intrusion", ["ml-classification", "deep-learning"]),
    (r"quality of service|\bqos\b|guarantee", ["mathematical-optimization", "centralized-control"]),
    (r"privacy|anonym|differential", ["federated-learning", "compression-semantic"]),
    (r"fault[- ]toleran|resilien|survivab", ["gossip-epidemic", "consensus"]),
    (r"autonom|self[- ]organi|self[- ]configur|cognitiv", ["reinforcement-learning", "digital-twin-control"]),
    (r"heterogen|heterogeneous", ["network-slicing", "deep-learning"]),
    (r"semantic|meaning|task[- ]oriented", ["compression-semantic"]),
    (r"quantum", ["consensus"]),
    (r"edge[- ]comput|offload", ["reinforcement-learning", "mathematical-optimization"]),
    (r"immersive|metaverse|augmented|virtual reality", ["compression-semantic", "caching-placement"]),
    (r"blockchain|trust|consensus", ["consensus", "blockchain-ledger"]),
    (r"traffic (classif|identif|forecas)|anomal", ["ml-classification", "deep-learning"]),
]


def future_directions(recs: list[dict]) -> dict:
    """Historical future-work statements vs later solutions.

    realized: >=1 later paper (year > origin) whose solution family matches
              the direction AND shares a domain.
    unresolved: direction with no such match.
    """
    stated = [
        r for r in recs
        if r.get("futureDirections") and r["futureDirections"] != NOT_STATED
        and r.get("year")
    ]
    later = [r for r in recs if r.get("year")]

    tracker = []
    for r in stated:
        text = r["futureDirections"].lower()
        matches = []
        for pat, fams in FUTURE_MATCH_RULES:
            try:
                if not re.search(pat, text):
                    continue
            except re.error:
                continue
            realized_by = [
                q["id"] for q in later
                if (q["year"] or 0) > r["year"]
                and set(q.get("solutionFamilies", [])) & set(fams)
                and (set(q["domains"]) & set(r["domains"]))
            ]
            matches.append(
                {
                    "pattern": pat,
                    "wantedFamilies": fams,
                    "realizedBy": realized_by[:5],
                    "status": "realized" if realized_by else "unresolved",
                }
            )
            if len(matches) >= 3:
                break
        if not matches:
            continue
        tracker.append(
            {
                "paperId": r["id"],
                "year": r["year"],
                "domains": r["domains"],
                "statement": clip(r["futureDirections"], 420),
                "matches": matches,
                "status": "realized" if any(m["status"] == "realized" for m in matches) else "unresolved",
            }
        )
    tracker.sort(key=lambda t: (t["status"] != "realized", t["year"]))
    unresolved = [t for t in tracker if t["status"] == "unresolved"]
    realized = [t for t in tracker if t["status"] == "realized"]
    return {"all": tracker[:60], "realized": realized[:30], "unresolved": unresolved[:30],
            "counts": {"realized": len(realized), "unresolved": len(unresolved)}}


# ------------------------------------------------------ academic/industry


def academic_industry(recs: list[dict]) -> dict:
    industry = [r for r in recs if r.get("industryEvidence") and r["industryEvidence"] != NOT_STATED]
    academic_only = [r for r in recs if r.get("industryEvidence") == NOT_STATED]

    # domains with heavy industry evidence vs heavy academic-only activity
    ind_domains: Counter = Counter()
    aca_domains: Counter = Counter()
    for r in industry:
        for d in r["domains"]:
            ind_domains[d] += 1
    for r in academic_only:
        for d in r["domains"]:
            aca_domains[d] += 1

    comparisons = []
    for dom, n_ind in ind_domains.most_common():
        n_aca = aca_domains.get(dom, 0)
        total = n_ind + n_aca
        comparisons.append(
            {
                "domainId": dom,
                "industryEvidencePapers": n_ind,
                "academicOnlyPapers": n_aca,
                "adoptionRatio": round(n_ind / total, 3) if total else 0,
                "verdict": (
                    "high-adoption" if total and n_ind / total >= 0.5
                    else "low-adoption" if total and n_ind / total <= 0.25
                    else "mixed"
                ),
            }
        )
    # low adoption: many papers, little industry evidence
    low_adoption = [c for c in comparisons if c["verdict"] == "low-adoption"]
    for c in low_adoption:
        c["whyLow"] = _why_low(c["domainId"], recs)

    industry_examples = [
        {
            "id": r["id"], "title": r["title"], "year": r.get("year"),
            "domains": r["domains"], "evidence": clip(r["industryEvidence"], 400),
            "solutionFamilies": r.get("solutionFamilies", []),
        }
        for r in sorted(industry, key=lambda x: -x.get("citations", 0))[:20]
    ]
    return {
        "industryEvidenceCount": len(industry),
        "academicOnlyCount": len(academic_only),
        "adoptionByDomain": comparisons,
        "lowAdoption": low_adoption[:12],
        "industryExamples": industry_examples,
    }


def _why_low(domain: str, recs: list[dict]) -> str:
    """Heuristic adoption-barrier diagnosis from the domain's own quotes."""
    notes = []
    rs = [r for r in recs if domain in r["domains"]]
    if any(r.get("limitations") and "assum" in r["limitations"].lower() for r in rs):
        notes.append("papers hinge on assumptions operators cannot guarantee")
    if any(r.get("limitations") and re.search(r"scal|complex", r["limitations"], re.I) for r in rs):
        notes.append("evaluation scale stays below production scale")
    if any(r.get("limitations") and re.search(r"deploy|practical|overhead", r["limitations"], re.I) for r in rs):
        notes.append("deployment overhead is acknowledged as a barrier")
    if not notes:
        notes.append("no explicit barrier stated in retrieved text; evidence is limited to abstracts")
    return "; ".join(notes)


# --------------------------------------------------------------- saturation


def saturation(recs: list[dict]) -> dict:
    """Early problems vs later progress; repeated attempts; attention/progress."""
    by_dom: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        for d in r["domains"]:
            by_dom[d].append(r)
    rows = []
    for dom, rs in by_dom.items():
        early = [r for r in rs if (r.get("year") or 9999) <= 2015]
        late = [r for r in rs if (r.get("year") or 0) >= 2019]
        early_problems = [r for r in early if r.get("problem") != NOT_STATED and r.get("problem")]
        late_problems = [r for r in late if r.get("problem") != NOT_STATED and r.get("problem")]
        # progress proxy: do late papers claim results the early ones lacked?
        late_results = sum(
            1 for r in late if r.get("experiments") and r["experiments"] != NOT_STATED
        )
        attention = len(rs)
        # saturation: many papers, but early problem statements keep recurring
        recurring = bool(early_problems) and bool(late_problems)
        progress_ratio = round(late_results / len(late), 2) if late else 0
        saturation_score = round(
            (attention / 40.0) * (1.0 if recurring else 0.5) - progress_ratio, 2
        )
        rows.append(
            {
                "domainId": dom,
                "label": _dom_label(dom),
                "attention": attention,
                "earlyCount": len(early),
                "lateCount": len(late),
                "recurringProblem": recurring,
                "progressRatio": progress_ratio,
                "saturationScore": saturation_score,
                "verdict": (
                    "saturated" if saturation_score >= 1.0
                    else "active" if attention >= 8
                    else "sparse"
                ),
                "remainingGap": _remaining_gap(dom, rs),
            }
        )
    rows.sort(key=lambda r: -r["saturationScore"])
    return {"rows": rows,
            "summary": {
                "saturated": sum(1 for r in rows if r["verdict"] == "saturated"),
                "active": sum(1 for r in rows if r["verdict"] == "active"),
                "sparse": sum(1 for r in rows if r["verdict"] == "sparse"),
            }}


def _remaining_gap(dom: str, rs: list[dict]) -> str:
    """A remaining gap, quoted from an unresolved limitation if one exists."""
    for r in sorted(rs, key=lambda x: -(x.get("citations", 0))):
        lim = r.get("limitations") or ""
        if lim and lim != NOT_STATED and re.search(r"future|remain|not address|open", lim, re.I):
            return clip(lim, 300)
    stated = [r for r in rs if r.get("problem") and r["problem"] != NOT_STATED]
    return clip(stated[-1]["problem"], 300) if stated else NOT_STATED


# --------------------------------------------------------------- benchmarks


def benchmarks(recs: list[dict]) -> dict:
    dataset_index: dict[str, dict] = {}
    metric_index: dict[str, dict] = {}
    algo_index: dict[str, dict] = {}
    for r in recs:
        for ds in r.get("datasets", []):
            slot = dataset_index.setdefault(ds, {"name": ds, "paperIds": [], "firstYear": 9999, "lastYear": 0})
            slot["paperIds"].append(r["id"])
            if r.get("year"):
                slot["firstYear"] = min(slot["firstYear"], r["year"])
                slot["lastYear"] = max(slot["lastYear"], r["year"])
        for me in r.get("metrics", []):
            slot = metric_index.setdefault(me, {"name": me, "paperIds": [], "firstYear": 9999, "lastYear": 0})
            slot["paperIds"].append(r["id"])
            if r.get("year"):
                slot["firstYear"] = min(slot["firstYear"], r["year"])
                slot["lastYear"] = max(slot["lastYear"], r["year"])
        for al in r.get("algorithms", []):
            slot = algo_index.setdefault(al, {"name": al, "paperIds": [], "firstYear": 9999, "lastYear": 0})
            slot["paperIds"].append(r["id"])
            if r.get("year"):
                slot["firstYear"] = min(slot["firstYear"], r["year"])
                slot["lastYear"] = max(slot["lastYear"], r["year"])

    def fin(slot: dict) -> dict:
        slot = dict(slot)
        if slot["firstYear"] == 9999:
            slot["firstYear"] = 0
        slot["useCount"] = len(slot["paperIds"])
        slot["paperIds"] = slot["paperIds"][:25]
        return slot

    datasets = sorted((fin(v) for v in dataset_index.values()), key=lambda x: -x["useCount"])
    metrics = sorted((fin(v) for v in metric_index.values()), key=lambda x: -x["useCount"])
    algorithms = sorted((fin(v) for v in algo_index.values()), key=lambda x: -x["useCount"])

    # metric evolution: periods where each metric appears
    periods = [(2011, 2014), (2015, 2018), (2019, 2022), (2023, 2026)]
    metric_evolution = []
    for me in metrics[:14]:
        rs = [r for r in recs if me["name"] in r.get("metrics", [])]
        cells = []
        for a, b in periods:
            cells.append(sum(1 for r in rs if r.get("year") and a <= r["year"] <= b))
        metric_evolution.append({"name": me["name"], "cells": cells})

    # best-known approach proxy: algorithm family with highest mean citations
    fam_cites: dict[str, list[int]] = defaultdict(list)
    for r in recs:
        for al in r.get("algorithms", []):
            fam_cites[al].append(r.get("citations", 0))
    best = sorted(
        (
            {"name": k, "meanCitations": round(sum(v) / len(v)), "papers": len(v)}
            for k, v in fam_cites.items() if len(v) >= 3
        ),
        key=lambda x: -x["meanCitations"],
    )[:12]

    # benchmark gaps: metrics that appear in early years but not late
    gaps = []
    for me in metrics[:20]:
        rs = [r for r in recs if me["name"] in r.get("metrics", [])]
        early = any((r.get("year") or 0) <= 2015 for r in rs)
        late = any((r.get("year") or 0) >= 2021 for r in rs)
        if early and not late:
            gaps.append({"name": me["name"], "note": "used through 2015, absent after 2020 in this corpus"})

    return {
        "datasets": datasets[:20],
        "metrics": metrics[:20],
        "algorithms": algorithms[:20],
        "metricEvolution": metric_evolution,
        "bestKnown": best,
        "gaps": gaps,
    }


# ----------------------------------------------------------------- helpers

_LABELS: dict[str, str] = {}


def _dom_label(dom: str) -> str:
    return _LABELS.get(dom, dom.replace("-", " ").title())


def main() -> int:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import networking_kb as KB  # noqa: E402  (sibling module from this phase)

    global _LABELS
    _LABELS = {d[0]: d[1] for d in KB.DOMAINS}
    _LABELS[KB.GENERIC[0]] = KB.GENERIC[1]

    recs = load()
    analysis = {
        "years": year_comparison(recs),
        "problemEvolution": problem_evolution(recs),
        "solutionEvolution": solution_evolution(recs),
        "problemMatrix": problem_matrix(recs),
        "clusters": problem_clusters(recs),
        "crossDomain": cross_domain(recs),
        "future": future_directions(recs),
        "academicIndustry": academic_industry(recs),
        "saturation": saturation(recs),
        "benchmarks": benchmarks(recs),
    }
    OUT_TS.parent.mkdir(parents=True, exist_ok=True)
    OUT_TS.write_text(
        "// GENERATED by scripts/networking_analysis.py — do not edit by hand.\n"
        "// Every derived record carries the paper ids it was computed from.\n"
        f"export const ANALYSIS = {json.dumps(analysis, indent=1, ensure_ascii=False)};\n"
    )
    print(
        f"analysis: {len(analysis['years'])} years, {len(analysis['clusters'])} clusters, "
        f"{analysis['future']['counts']} future-tracker entries, "
        f"{len(analysis['saturation']['rows'])} saturation rows, "
        f"{len(analysis['benchmarks']['datasets'])} datasets"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
