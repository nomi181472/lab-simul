#!/usr/bin/env python3
"""Validate object-detection harvest records.

Checks per papers/object_detection/harvest/<ID>.json:
  - JSON shape: required sections and field types (schema v1)
  - identity fields match papers/object_detection/manifest.json
  - EVERY quote is verbatim-present in the paper's extracted text
    (whitespace/case/ligature normalized)
  - tag slugs are kebab-case; evidence arrays non-empty where required
  - benchmark rows have dataset/metric/value

Usage:
  python3 scripts/od_harvest_check.py                 # all records
  python3 scripts/od_harvest_check.py OD001 OD002     # specific ids
Exit code 1 if any record fails.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HARVEST = ROOT / "papers" / "object_detection" / "harvest"
MANIFEST = ROOT / "papers" / "object_detection" / "manifest.json"

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
EVIDENCE_KEYS = ["problem", "background", "solution", "experiments", "limitations"]
MAX_QUOTE = 300
MAX_BYTES = 15_000

LIG = {
    "\ufb01": "fi", "\ufb02": "fl", "\u2019": "'", "\u2018": "'",
    "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
    "\u00a0": " ",
}


def norm(s: str) -> str:
    for k, v in LIG.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()


def load_manifest() -> dict[str, dict]:
    return {e["id"]: e for e in json.loads(MANIFEST.read_text())}


def load_text(entry: dict) -> str:
    p = ROOT / entry["textFile"]
    return p.read_text(errors="replace") if p.exists() else ""


def check_record(mid: str, rec: dict, man: dict, text: str) -> list[str]:
    errs: list[str] = []
    text_n = norm(text).casefold()

    # --- identity ---
    if rec.get("id") != mid:
        errs.append(f"id mismatch: file says {rec.get('id')}")
    for field in ("title", "year"):
        if field in man and rec.get(field) != man[field]:
            errs.append(f"{field} mismatch: {rec.get(field)!r} vs manifest {man[field]!r}")
    if rec.get("arxiv") != man.get("arxiv"):
        errs.append(f"arxiv mismatch: {rec.get('arxiv')} vs {man.get('arxiv')}")

    # --- required sections ---
    for sec in ["problem", "background", "solution", "experiments",
                "limitations", "futureDirections", "technical", "industry", "citations"]:
        if sec not in rec:
            errs.append(f"missing section: {sec}")
    if errs:
        return errs

    # --- shape ---
    def need(d: dict, keys: list[str], typ: type = str) -> None:
        for k in keys:
            if k not in d:
                errs.append(f"{sec}.{k} missing")
            elif typ is str and not (isinstance(d[k], str) and d[k].strip()):
                errs.append(f"{sec}.{k} empty")
            elif typ is list and not isinstance(d[k], list):
                errs.append(f"{sec}.{k} not a list")

    for sec in ["problem", "background", "solution", "experiments", "limitations"]:
        sec_obj = rec[sec]
        if not isinstance(sec_obj, dict):
            errs.append(f"{sec} not an object")
            continue
        need(sec_obj, ["summary"])
    for sec, keys in [("problem", ["statements", "problemTags", "evidence"]),
                      ("background", ["relatedWork", "evidence"]),
                      ("solution", ["components", "solutionTags", "evidence"]),
                      ("experiments", ["benchmarks", "baselines", "evidence"]),
                      ("limitations", ["items", "evidence"])]:
        sec_obj = rec[sec]
        if isinstance(sec_obj, dict):
            for k in keys:
                if k not in sec_obj or not isinstance(sec_obj[k], list):
                    errs.append(f"{sec}.{k} missing/not a list")

    # --- evidence non-empty where required ---
    for sec in ["problem", "solution", "experiments"]:
        sec_obj = rec[sec]
        if isinstance(sec_obj, dict) and not sec_obj.get("evidence"):
            errs.append(f"{sec}.evidence empty")

    # --- tags ---
    for sec, key in [("problem", "problemTags"), ("solution", "solutionTags")]:
        sec_obj = rec[sec]
        if isinstance(sec_obj, dict):
            for t in sec_obj.get(key, []):
                if not isinstance(t, str) or not SLUG_RE.match(t):
                    errs.append(f"{sec}.{key} bad slug: {t!r}")
            tags = sec_obj.get(key, [])
            if tags and not (1 <= len(tags) <= 8):
                errs.append(f"{sec}.{key} count {len(tags)} outside 1..8")

    # --- future directions ---
    fd = rec["futureDirections"]
    if not isinstance(fd, list):
        errs.append("futureDirections not a list")
    else:
        for i, f in enumerate(fd):
            if not isinstance(f, dict) or not f.get("direction") or not f.get("quote"):
                errs.append(f"futureDirections[{i}] missing direction/quote")

    # --- technical / industry / citations ---
    tech = rec["technical"]
    if not isinstance(tech, dict):
        errs.append("technical not an object")
    else:
        for k in ["datasets", "backbones", "models", "algorithms", "metrics", "equations"]:
            if not isinstance(tech.get(k), list):
                errs.append(f"technical.{k} not a list")
    ind = rec["industry"]
    if not isinstance(ind, dict):
        errs.append("industry not an object")
    else:
        for k in ["applications", "deploymentClaims", "evidence"]:
            if not isinstance(ind.get(k), list):
                errs.append(f"industry.{k} not a list")
    cit = rec["citations"]
    if not isinstance(cit, dict):
        errs.append("citations not an object")
    else:
        if man.get("doi") and cit.get("doi") and cit["doi"] != man["doi"]:
            errs.append(f"citations.doi {cit['doi']} != manifest {man['doi']}")
        if man.get("arxiv") and not cit.get("arxivUrl"):
            errs.append("citations.arxivUrl missing (arxiv paper)")

    # --- benchmarks rows ---
    if isinstance(rec["experiments"], dict):
        for i, b in enumerate(rec["experiments"].get("benchmarks", [])):
            if not all(b.get(k) for k in ("dataset", "metric", "value")):
                errs.append(f"experiments.benchmarks[{i}] missing dataset/metric/value")

    # --- QUOTE VERIFICATION ---
    def verify(container: str, quotes: list) -> None:
        for i, ev in enumerate(quotes):
            if not isinstance(ev, dict) or not ev.get("quote"):
                errs.append(f"{container}[{i}] no quote")
                continue
            q = ev["quote"]
            if len(q) > MAX_QUOTE:
                errs.append(f"{container}[{i}] quote too long ({len(q)} chars)")
            if norm(q).casefold() not in text_n:
                qn = norm(q)
                errs.append(
                    f"{container}[{i}] quote NOT FOUND: {qn[:90]!r}"
                    + ("..." if len(qn) > 90 else "")
                )
            if not ev.get("location"):
                errs.append(f"{container}[{i}] missing location")

    for sec in EVIDENCE_KEYS:
        sec_obj = rec[sec]
        if isinstance(sec_obj, dict):
            verify(f"{sec}.evidence", sec_obj.get("evidence", []))
            if sec == "background":
                for i, rw in enumerate(sec_obj.get("relatedWork", [])):
                    if rw.get("quote"):
                        verify(f"background.relatedWork[{i}]", [rw])
            if sec == "solution" and isinstance(sec_obj.get("approachEvidence"), list):
                verify("solution.approachEvidence", sec_obj["approachEvidence"])
    if isinstance(fd, list):
        for i, f in enumerate(fd):
            if isinstance(f, dict) and f.get("quote"):
                verify(f"futureDirections[{i}]", [{**f, "location": f.get("location")}])
    if isinstance(ind, dict):
        verify("industry.evidence", ind.get("evidence", []))
        for i, a in enumerate(ind.get("applications", [])):
            if isinstance(a, dict) and a.get("quote"):
                verify(f"industry.applications[{i}]", [{**a, "location": a.get("location")}])

    return errs


def main(argv: list[str]) -> int:
    manifest = load_manifest()
    ids = argv or sorted(manifest)
    bad = 0
    for mid in ids:
        path = HARVEST / f"{mid}.json"
        man = manifest.get(mid)
        if not man:
            print(f"{mid}: FAIL unknown id (not in manifest)")
            bad += 1
            continue
        if not path.exists():
            print(f"{mid}: FAIL missing harvest file")
            bad += 1
            continue
        raw = path.read_bytes()
        if len(raw) > MAX_BYTES:
            print(f"{mid}: FAIL {len(raw)} bytes > {MAX_BYTES}")
            bad += 1
            continue
        try:
            rec = json.loads(raw)
        except Exception as e:
            print(f"{mid}: FAIL invalid JSON: {e}")
            bad += 1
            continue
        errs = check_record(mid, rec, man, load_text(man))
        if errs:
            bad += 1
            print(f"{mid}: FAIL ({len(errs)})")
            for e in errs:
                print(f"   - {e}")
        else:
            n_q = sum(len(rec[s].get("evidence", [])) for s in EVIDENCE_KEYS)
            print(f"{mid}: PASS ({n_q} verified quotes, {len(raw)} bytes)")
    print(f"\n{len(ids) - bad}/{len(ids)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
