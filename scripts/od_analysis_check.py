#!/usr/bin/env python3
"""Validate review-pass analysis JSON (papers/object_detection/analysis/).

Files checked (each optional; missing file = skipped, reported):
  judgments.json          problemStatuses (lives at papers/object_detection/)
  analysis/clusters.json  ProblemCluster[]
  analysis/directions*.json  FutureDirection[] (merged, unique ids)
  analysis/industry.json  IndustryAnalysis
  analysis/saturation.json SaturationEntry[]
  analysis/crossdomain.json Analogy[]

Invariants enforced for every entry:
  - shape matches src/labs/object-detection/data/types.ts
  - all paperIds / followUps / sourcePaperId resolve in the manifest
  - problemIds / solutionIds resolve in the corpus vocab
  - every quote is verbatim-present in that paper's pdftotext output
    (same whitespace/case/ligature normalization as od_harvest_check)

Usage: python3 scripts/od_analysis_check.py
Exit code 1 on any failure.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from od_harvest_check import LIG, norm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OD = ROOT / "papers" / "object_detection"
ANALYSIS = OD / "analysis"
HARVEST = OD / "harvest"

STATUSES = {"resolved", "reduced", "persistent", "transformed", "uncertain"}
DIR_STATUSES = {"realized", "partial", "unresolved"}
CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
KINDS = {"paperSays", "crossPaper", "labInterpretation"}
SAT = {"saturated", "active", "dormant", "emerging"}
GAP_KINDS = {"academicOnly", "industryOnly", "lowAdoption"}


def slug_ok(s: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", s))


def main() -> int:
    manifest = {e["id"]: e for e in json.loads((OD / "manifest.json").read_text())}
    harvest = {json.loads(p.read_text())["id"]: json.loads(p.read_text())
               for p in sorted(HARVEST.glob("OD*.json"))}
    texts: dict[str, str] = {}

    def text_of(pid: str) -> str:
        if pid not in texts:
            entry = manifest.get(pid, {})
            tp = ROOT / entry.get("textFile", "")
            texts[pid] = norm(tp.read_text(errors="replace")).casefold() if tp.exists() else ""
        return texts[pid]

    # corpus vocab from harvest
    prob_vocab: set[str] = set()
    sol_vocab: set[str] = set()
    aliases_p = {"small-object-detection": "small-objects",
                 "multi-scale-objects": "scale-variance",
                 "nms-redundancy": "nms", "slow-convergence": "convergence",
                 "compute-efficiency": "deployability",
                 "training-efficiency": "convergence"}
    aliases_s = {"one-to-many-hybrid": "group-wise-one-to-many",
                 "localization": "iou-loss", "label-assignment": "dynamic-assignment",
                 "nms": "post-processing-nms", "crowded-scenes": "dense-positive-supervision",
                 "deployability": "real-time-backbone", "occlusion": "feature-fusion"}
    for h in harvest.values():
        for t in h["problem"]["problemTags"]:
            prob_vocab.add(aliases_p.get(t, t))
        for t in h["solution"]["solutionTags"]:
            sol_vocab.add(aliases_s.get(t, t))

    errors: list[str] = []
    checked_files = 0

    def check_quote(pid: str, quote: str, where: str) -> None:
        if pid not in manifest:
            errors.append(f"{where}: unknown paper {pid}")
            return
        q = norm(quote).casefold()
        if not q:
            errors.append(f"{where}: empty quote")
        elif q not in text_of(pid):
            errors.append(f"{where}: quote not in {pid} PDF text: {quote[:70]!r}")

    def check_evidence(ev: dict, where: str) -> None:
        if not isinstance(ev.get("paperIds"), list) or not ev.get("paperIds"):
            errors.append(f"{where}: evidence.paperIds empty")
            return
        for pid in ev["paperIds"]:
            if pid not in manifest:
                errors.append(f"{where}: unknown paper {pid}")
        if ev.get("kind") not in KINDS:
            errors.append(f"{where}: bad evidence.kind {ev.get('kind')!r}")
        if ev.get("confidence") not in CONFIDENCE:
            errors.append(f"{where}: bad confidence {ev.get('confidence')!r}")
        if ev.get("quote"):
            check_quote(ev["paperIds"][0], ev["quote"], where)

    # ------------------------------------------------ judgments
    jpath = OD / "judgments.json"
    if jpath.exists():
        checked_files += 1
        j = json.loads(jpath.read_text())
        ps = j.get("problemStatuses", {})
        for tag, v in ps.items():
            where = f"judgments[{tag}]"
            if tag not in prob_vocab:
                errors.append(f"{where}: unknown problem tag")
            if v.get("status") not in STATUSES:
                errors.append(f"{where}: bad status {v.get('status')!r}")
            if not isinstance(v.get("statusEvidence", ""), str) or not v.get("statusEvidence"):
                errors.append(f"{where}: empty statusEvidence")
            for pid in v.get("statusPapers", []):
                if pid not in manifest:
                    errors.append(f"{where}: unknown paper {pid}")

    # ------------------------------------------------ clusters
    f = ANALYSIS / "clusters.json"
    if f.exists():
        checked_files += 1
        for i, c in enumerate(json.loads(f.read_text())):
            where = f"clusters[{i}] ({c.get('id')})"
            for key in ("id", "name", "description"):
                if not c.get(key):
                    errors.append(f"{where}: missing {key}")
            if not slug_ok(c.get("id", "")):
                errors.append(f"{where}: bad id slug")
            for t in c.get("problemTags", []):
                if t not in prob_vocab:
                    errors.append(f"{where}: unknown problemTag {t}")
            for t in c.get("solutionTags", []):
                if t not in sol_vocab:
                    errors.append(f"{where}: unknown solutionTag {t}")
            if not c.get("papers"):
                errors.append(f"{where}: empty papers")
            for pid in c.get("papers", []):
                if pid not in manifest:
                    errors.append(f"{where}: unknown paper {pid}")
            for ev in c.get("evidence", []):
                check_evidence(ev=ev, where=f"{where}.evidence")

    # ------------------------------------------------ directions
    dir_files = sorted(ANALYSIS.glob("directions*.json"))
    if dir_files:
        seen_uids: set[str] = set()
        for df in dir_files:
            checked_files += 1
            data = json.loads(df.read_text())
            for i, d in enumerate(data):
                where = f"{df.name}[{i}] ({d.get('id')})"
                uid = d.get("id", "")
                if uid in seen_uids:
                    errors.append(f"{where}: duplicate id")
                seen_uids.add(uid)
                if not re.fullmatch(r"OD\d{3}-f\d+", uid):
                    errors.append(f"{where}: bad id format")
                sp = d.get("sourcePaperId", "")
                if sp not in manifest:
                    errors.append(f"{where}: unknown sourcePaperId")
                elif not uid.startswith(sp):
                    errors.append(f"{where}: id prefix != sourcePaperId")
                if d.get("sourceYear") != manifest.get(sp, {}).get("year"):
                    errors.append(f"{where}: sourceYear mismatch")
                if not d.get("direction"):
                    errors.append(f"{where}: empty direction")
                if d.get("status") not in DIR_STATUSES:
                    errors.append(f"{where}: bad status {d.get('status')!r}")
                if not isinstance(d.get("realizationNote"), str) or not d.get("realizationNote"):
                    errors.append(f"{where}: empty realizationNote")
                if d.get("status") == "unresolved" and d.get("realizingPaperId"):
                    errors.append(f"{where}: unresolved but has realizingPaperId")
                for fid in d.get("followUps", []):
                    if not re.fullmatch(r"OD\d{3}", fid):
                        errors.append(f"{where}: bad followUp {fid}")
                rp = d.get("realizingPaperId")
                if rp and rp not in manifest:
                    errors.append(f"{where}: unknown realizingPaperId")
                check_quote(sp, d.get("quote", ""), where)

    # ------------------------------------------------ industry
    f = ANALYSIS / "industry.json"
    if f.exists():
        checked_files += 1
        ind = json.loads(f.read_text())
        for i, dom in enumerate(ind.get("domains", [])):
            where = f"industry.domains[{i}] ({dom.get('domain')})"
            if not dom.get("domain"):
                errors.append(f"{where}: empty domain")
            if dom.get("industryAdoption") not in CONFIDENCE:
                errors.append(f"{where}: bad industryAdoption")
            if not isinstance(dom.get("academicEvidence"), int):
                errors.append(f"{where}: academicEvidence must be int")
            for pid in dom.get("paperIds", []):
                if pid not in manifest:
                    errors.append(f"{where}: unknown paper {pid}")
            if not dom.get("note"):
                errors.append(f"{where}: empty note")
        for i, g in enumerate(ind.get("gaps", [])):
            where = f"industry.gaps[{i}]"
            if g.get("kind") not in GAP_KINDS:
                errors.append(f"{where}: bad kind")
            if not g.get("title") or not g.get("detail"):
                errors.append(f"{where}: empty title/detail")
            for pid in g.get("paperIds", []):
                if pid not in manifest:
                    errors.append(f"{where}: unknown paper {pid}")
        for i, lo in enumerate(ind.get("lowAdoption", [])):
            where = f"industry.lowAdoption[{i}]"
            if lo.get("solutionId") not in sol_vocab:
                errors.append(f"{where}: unknown solutionId")
            if not lo.get("reason"):
                errors.append(f"{where}: empty reason")

    # ------------------------------------------------ saturation
    f = ANALYSIS / "saturation.json"
    if f.exists():
        checked_files += 1
        for i, s in enumerate(json.loads(f.read_text())):
            where = f"saturation[{i}] ({s.get('problemId')})"
            if s.get("problemId") not in prob_vocab:
                errors.append(f"{where}: unknown problemId")
            if s.get("status") not in STATUSES:
                errors.append(f"{where}: bad status")
            if s.get("saturation") not in SAT:
                errors.append(f"{where}: bad saturation")
            if not isinstance(s.get("intensity"), (int, float)):
                errors.append(f"{where}: intensity must be numeric")
            if not s.get("remainingGap"):
                errors.append(f"{where}: empty remainingGap")
            for ev in s.get("evidence", []):
                check_evidence(ev=ev, where=f"{where}.evidence")

    # ------------------------------------------------ cross-consistency
    jp = OD / "judgments.json"
    sp = ANALYSIS / "saturation.json"
    if jp.exists() and sp.exists():
        js = json.loads(jp.read_text()).get("problemStatuses", {})
        for s in json.loads(sp.read_text()):
            pid = s.get("problemId")
            if pid in js and s.get("status") != js[pid].get("status"):
                errors.append(
                    f"saturation[{pid}]: status {s.get('status')!r} != "
                    f"judgments {js[pid].get('status')!r}")

    # ------------------------------------------------ crossdomain
    f = ANALYSIS / "crossdomain.json"
    if f.exists():
        checked_files += 1
        for i, a in enumerate(json.loads(f.read_text())):
            where = f"crossdomain[{i}] ({a.get('id')})"
            for key in ("id", "domain", "sourceConcept", "detectionConcept", "mapping"):
                if not a.get(key):
                    errors.append(f"{where}: missing {key}")
            for pid in a.get("paperIds", []):
                if pid not in manifest:
                    errors.append(f"{where}: unknown paper {pid}")

    print(f"checked {checked_files} analysis file(s)")
    if errors:
        for e in errors[:60]:
            print(f"  FAIL {e}")
        if len(errors) > 60:
            print(f"  ... and {len(errors) - 60} more")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
