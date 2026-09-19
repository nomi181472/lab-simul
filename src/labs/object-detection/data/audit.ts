import { PAPERS, PAPER_BY_ID } from "./papers";
import { YEAR_STATES } from "./years";
import { YEAR_DELTAS } from "./year-deltas";
import { PROBLEM_LIFECYCLES } from "./problems";
import { RESEARCH_EPISODES } from "./loops";
import { RESEARCH_DIRECTIONS } from "./directions";
import { APPLICATIONS } from "./applications";
import { DETECTION_SCENARIOS } from "./edgecases";
import { CONCEPTS } from "./concepts";

export type AuditCheck = {
  id: string;
  label: string;
  pass: boolean;
  detail: string;
};

export function runAudit(): AuditCheck[] {
  const checks: AuditCheck[] = [];

  // 1. all papers processed
  const missingPoor = PAPERS.filter((p) => !p.summary || p.contributions.length === 0);
  checks.push({
    id: "papers-processed",
    label: "All papers processed (summary + contributions)",
    pass: missingPoor.length === 0,
    detail: `${PAPERS.length} papers; ${missingPoor.length} with missing summary/contributions.`,
  });

  // 2. duplicate papers
  const arxivSeen = new Map<string, string>();
  const dups: string[] = [];
  for (const p of PAPERS) {
    if (p.arxiv && arxivSeen.has(p.arxiv)) dups.push(`${p.arxiv}: ${arxivSeen.get(p.arxiv)} & ${p.id}`);
    if (p.arxiv) arxivSeen.set(p.arxiv, p.id);
  }
  checks.push({
    id: "duplicates",
    label: "No duplicate papers",
    pass: dups.length === 0,
    detail: dups.length ? dups.join("; ") : "No duplicate arXiv ids.",
  });

  // 3. year validity
  const years = Object.keys(YEAR_STATES).map(Number).sort((a, b) => a - b);
  const validYears =
    years.length >= years[0] - years[0] + 1 &&
    years.every((y, i) => (i === 0 ? true : y === years[i - 1] + 1));
  checks.push({
    id: "years",
    label: "Years contiguous 2015-2026",
    pass: validYears && years[0] === 2015 && years[years.length - 1] === 2026,
    detail: `Years: ${years.join(", ")}.`,
  });

  // 4. year states reference known papers
  const badRefs: string[] = [];
  for (const ys of Object.values(YEAR_STATES)) {
    for (const pid of ys.supportingPapers) {
      if (!PAPER_BY_ID[pid]) badRefs.push(`${ys.year}->${pid}`);
    }
  }
  checks.push({
    id: "year-evidence",
    label: "Year state paper refs resolve",
    pass: badRefs.length === 0,
    detail: badRefs.length ? badRefs.join("; ") : "All supportingPapers resolve.",
  });

  // 5. deltas reference known papers & years
  const badDelta: string[] = [];
  for (const d of YEAR_DELTAS) {
    if (!YEAR_STATES[d.from] || !YEAR_STATES[d.to]) badDelta.push(`${d.from}->${d.to}`);
    for (const pid of d.supportingPapers) if (!PAPER_BY_ID[pid]) badDelta.push(`p:${pid}`);
  }
  checks.push({
    id: "deltas",
    label: "Year deltas consistent",
    pass: badDelta.length === 0,
    detail: badDelta.length ? badDelta.join("; ") : "Deltas OK.",
  });

  // 6. problem lifecycles reference known papers
  const badProb: string[] = [];
  for (const p of PROBLEM_LIFECYCLES) {
    for (const o of p.occurrences) {
      for (const pid of o.papers) if (!PAPER_BY_ID[pid]) badProb.push(`${p.id}->${pid}`);
    }
  }
  checks.push({
    id: "problem-lifecycles",
    label: "Problem lifecycle refs resolve",
    pass: badProb.length === 0,
    detail: badProb.length ? badProb.join("; ") : "Lifecycles OK.",
  });

  // 7. episodes resolve
  const badEp: string[] = [];
  for (const e of RESEARCH_EPISODES) {
    for (const pid of e.paperIds) if (!PAPER_BY_ID[pid]) badEp.push(`${e.id}->${pid}`);
    for (const t of e.leadsTo) if (!RESEARCH_EPISODES.some((x) => x.id === t)) badEp.push(`${e.id}->leadsTo ${t}`);
  }
  checks.push({
    id: "episodes",
    label: "Research episodes resolve",
    pass: badEp.length === 0,
    detail: badEp.length ? badEp.join("; ") : "Episodes OK.",
  });

  // 8. directions resolve
  const badDir: string[] = [];
  for (const d of RESEARCH_DIRECTIONS) {
    for (const pid of d.representativePapers) if (!PAPER_BY_ID[pid]) badDir.push(`${d.id}->${pid}`);
  }
  checks.push({
    id: "directions",
    label: "Directions resolve",
    pass: badDir.length === 0,
    detail: badDir.length ? badDir.join("; ") : "Directions OK.",
  });

  // 9. applications resolve
  const badApp: string[] = [];
  for (const a of APPLICATIONS) for (const pid of a.paperIds) if (!PAPER_BY_ID[pid]) badApp.push(`${a.id}->${pid}`);
  checks.push({
    id: "applications",
    label: "Application claims resolve to papers",
    pass: badApp.length === 0,
    detail: badApp.length ? badApp.join("; ") : "Applications OK.",
  });

  // 10. edge cases resolve
  const badSc: string[] = [];
  for (const s of DETECTION_SCENARIOS) for (const pid of s.papers) if (!PAPER_BY_ID[pid]) badSc.push(`${s.name}->${pid}`);
  checks.push({
    id: "edge-cases",
    label: "Edge cases resolve to papers",
    pass: badSc.length === 0,
    detail: badSc.length ? badSc.join("; ") : "Edge cases OK.",
  });

  // 11. concepts resolve
  const badC: string[] = [];
  for (const c of CONCEPTS) for (const pid of c.paperIds) if (!PAPER_BY_ID[pid]) badC.push(`${c.id}->${pid}`);
  checks.push({
    id: "concepts",
    label: "Concepts resolve to papers",
    pass: badC.length === 0,
    detail: badC.length ? badC.join("; ") : "Concepts OK.",
  });

  // 12. paradigm shift chronology (directions don't claim years before evidence)
  const firstYearByPaper = new Map<string, number>();
  for (const p of PAPERS) firstYearByPaper.set(p.id, p.year);
  const badChrono: string[] = [];
  for (const d of RESEARCH_DIRECTIONS) {
    const repYears = d.representativePapers.map((p) => firstYearByPaper.get(p) ?? 0);
    const earliest = Math.min(...repYears);
    if (earliest > d.firstAppearance) badChrono.push(`${d.id}: claims ${d.firstAppearance}, earliest paper ${earliest}`);
  }
  checks.push({
    id: "chronology",
    label: "Direction first-appearance <= earliest paper",
    pass: badChrono.length === 0,
    detail: badChrono.length ? badChrono.join("; ") : "Chronology OK.",
  });

  return checks;
}

export function countPapersByYear() {
  const m: Record<number, number> = {};
  for (const p of PAPERS) m[p.year] = (m[p.year] ?? 0) + 1;
  return m;
}