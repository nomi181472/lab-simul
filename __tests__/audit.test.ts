import { describe, it, expect } from "vitest";
import { runAudit, countPapersByYear } from "../src/labs/object-detection/data/audit";
import { PAPERS } from "../src/labs/object-detection/data/papers";
import { YEAR_STATES } from "../src/labs/object-detection/data/years";
import { RESEARCH_EPISODES } from "../src/labs/object-detection/data/loops";
import { PROBLEM_LIFECYCLES } from "../src/labs/object-detection/data/problems";
import { RESEARCH_DIRECTIONS } from "../src/labs/object-detection/data/directions";
import { APPLICATIONS } from "../src/labs/object-detection/data/applications";
import { DETECTION_SCENARIOS } from "../src/labs/object-detection/data/edgecases";
import { CONCEPTS } from "../src/labs/object-detection/data/concepts";

describe("research data integrity", () => {
  it("every audit check passes (0 hallucination / reference gaps)", () => {
    const checks = runAudit();
    const failed = checks.filter((c) => !c.pass);
    expect(
      failed.map((c) => c.detail),
      `failing: ${failed.map((c) => c.id).join(", ")}`,
    ).toEqual([]);
  });

  it("corpus is 48 unique papers over a contiguous 2015-2026 range", () => {
    expect(PAPERS).toHaveLength(48);
    const byYear = countPapersByYear();
    const years = Object.keys(byYear).map(Number).sort((a, b) => a - b);
    expect(years[0]).toBe(2015);
    expect(years[years.length - 1]).toBe(2026);
    expect(PAPERS.filter((p) => p.year < 2015 || p.year > 2026)).toEqual([]);
  });

  it("every paper id referenced anywhere resolves", () => {
    const all = new Set(PAPERS.map((p) => p.id));
    const referenced = new Set<string>();
    for (const e of RESEARCH_EPISODES) for (const p of e.paperIds) referenced.add(p);
    for (const p of PROBLEM_LIFECYCLES) for (const o of p.occurrences) for (const q of o.papers) referenced.add(q);
    for (const d of RESEARCH_DIRECTIONS) for (const p of d.representativePapers) referenced.add(p);
    for (const a of APPLICATIONS) for (const p of a.paperIds) referenced.add(p);
    for (const s of DETECTION_SCENARIOS) for (const p of s.papers) referenced.add(p);
    for (const c of CONCEPTS) for (const p of c.paperIds) referenced.add(p);
    for (const y of Object.values(YEAR_STATES)) for (const p of y.supportingPapers) referenced.add(p);

    const dangling = [...referenced].filter((p) => !all.has(p));
    expect(dangling).toEqual([]);
  });

  it("year states report the year they describe and stay within range", () => {
    for (const ys of Object.values(YEAR_STATES)) {
      expect(ys.year).toBeGreaterThanOrEqual(2015);
      expect(ys.year).toBeLessThanOrEqual(2026);
      expect(ys.supportingPapers.every((p) => {
        // weakest check: the id must exist (year membership is checked in the audit)
        return PAPERS.some((pp) => pp.id === p);
      })).toBe(true);
    }
  });
});