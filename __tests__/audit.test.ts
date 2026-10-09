import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { runAudit, countPapersByYear } from "../src/labs/object-detection/data/audit";
import { PAPERS, PAPER_BY_ID } from "../src/labs/object-detection/data/papers";
import { PROBLEM_LIFECYCLES } from "../src/labs/object-detection/data/problems";
import { CLUSTERS } from "../src/labs/object-detection/data/clusters";
import { FUTURE_DIRECTIONS } from "../src/labs/object-detection/data/directions";
import { SATURATION } from "../src/labs/object-detection/data/saturation";
import { RELATIONS } from "../src/labs/object-detection/data/relations";
import { SOLUTIONS } from "../src/labs/object-detection/data/solutions";
import { INDUSTRY } from "../src/labs/object-detection/data/industry";
import { ANALOGIES } from "../src/labs/object-detection/data/crossdomain";
import { BENCHMARKS } from "../src/labs/object-detection/data/benchmarks";
import { DATASETS } from "../src/labs/object-detection/data/datasets";
import { METRICS } from "../src/labs/object-detection/data/metrics";
import { YEAR_STATES } from "../src/labs/object-detection/data/years";
import { HARVEST_RECORDS } from "../src/labs/object-detection/data/harvest-records";
import { SIM_IDS } from "../src/labs/object-detection/sims/registry";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OD = path.join(ROOT, "papers", "object_detection");

type ManifestEntry = {
  id: string;
  year: number;
  textFile: string;
};

const manifest: ManifestEntry[] = JSON.parse(
  fs.readFileSync(path.join(OD, "manifest.json"), "utf8"),
);

/* Same whitespace/case/ligature normalization as scripts/od_harvest_check.py
 * so quotes are compared exactly the way the Python verifier compares them. */
const LIG: Record<string, string> = {
  "ﬁ": "fi", "ﬂ": "fl", "’": "'", "‘": "'",
  "“": '"', "”": '"', "–": "-", "—": "-", "\u00a0": " ",
};

function norm(s: string): string {
  let out = s;
  for (const [k, v] of Object.entries(LIG)) out = out.split(k).join(v);
  return out.replace(/\s+/g, " ").trim().toLowerCase();
}

const textCache = new Map<string, string>();
function pdfText(entry: ManifestEntry): string {
  let t = textCache.get(entry.id);
  if (t === undefined) {
    t = fs.readFileSync(path.join(ROOT, entry.textFile), "utf8");
    textCache.set(entry.id, t);
  }
  return t;
}

const harvestCache = new Map<string, string>();
function harvestRaw(pid: string): string {
  let t = harvestCache.get(pid);
  if (t === undefined) {
    t = fs.readFileSync(path.join(OD, "harvest", `${pid}.json`), "utf8");
    harvestCache.set(pid, t);
  }
  return t;
}

/** Every object anywhere with a string `quote` field. */
function collectQuotes(node: unknown, out: string[] = []): string[] {
  if (Array.isArray(node)) {
    for (const v of node) collectQuotes(v, out);
  } else if (node && typeof node === "object") {
    for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
      if (k === "quote" && typeof v === "string" && v.length > 0) out.push(v);
      else collectQuotes(v, out);
    }
  }
  return out;
}

type DataQuote = { paperId: string; quote: string; where: string };

function dataQuotes(): DataQuote[] {
  const out: DataQuote[] = [];
  for (const p of PAPERS) {
    p.futureDirections.forEach((d, i) =>
      out.push({ paperId: p.id, quote: d.quote, where: `${p.id}.futureDirections[${i}]` }));
    p.industryApplications.forEach((a, i) => {
      if (a.quote)
        out.push({ paperId: p.id, quote: a.quote, where: `${p.id}.industryApplications[${i}]` });
    });
  }
  for (const pr of PROBLEM_LIFECYCLES) {
    pr.occurrences.forEach((o) => {
      o.evidence.forEach((e, i) =>
        out.push({ paperId: e.paperId, quote: e.quote, where: `${pr.id}.evidence[${i}]@${o.year}` }));
      o.attemptedSolutions.forEach((a, i) => {
        if (a.quote)
          out.push({ paperId: a.paperId, quote: a.quote, where: `${pr.id}.attempt[${i}]@${o.year}` });
      });
    });
  }
  for (const c of CLUSTERS) {
    c.evidence.forEach((e, i) => {
      if (e.quote)
        out.push({ paperId: e.paperIds[0], quote: e.quote, where: `${c.id}.evidence[${i}]` });
    });
  }
  for (const s of SATURATION) {
    s.evidence.forEach((e, i) => {
      if (e.quote)
        out.push({ paperId: e.paperIds[0], quote: e.quote, where: `${s.problemId}.saturation[${i}]` });
    });
  }
  for (const d of FUTURE_DIRECTIONS) {
    out.push({ paperId: d.sourcePaperId, quote: d.quote, where: `${d.id}.quote` });
  }
  for (const r of RELATIONS) {
    if (r.quote)
      out.push({ paperId: r.from, quote: r.quote, where: `${r.from}->${r.to}` });
  }
  for (const [pid, rec] of Object.entries(HARVEST_RECORDS)) {
    const sections = [
      ["problem", rec.problem],
      ["background", rec.background],
      ["solution", rec.solution],
      ["experiments", rec.experiments],
      ["limitations", rec.limitations],
      ["industry", rec.industry],
    ] as const;
    for (const [name, sec] of sections) {
      sec.evidence.forEach((e, i) =>
        out.push({ paperId: pid, quote: e.quote, where: `${pid}.${name}.evidence[${i}]` }));
    }
    rec.background.relatedWork.forEach((rw, i) => {
      if (rw.quote)
        out.push({ paperId: pid, quote: rw.quote, where: `${pid}.relatedWork[${i}]` });
    });
  }
  return out;
}

describe("research data integrity", () => {
  it("every deterministic audit check passes", () => {
    const checks = runAudit();
    const failed = checks.filter((c) => !c.pass);
    expect(
      failed.map((c) => `${c.id}: ${c.detail}`),
    ).toEqual([]);
  });

  it("corpus is 121 unique papers over a contiguous 2015-2026 range", () => {
    expect(PAPERS).toHaveLength(121);
    expect(new Set(PAPERS.map((p) => p.id)).size).toBe(121);
    for (const p of PAPERS) expect(p.id).toMatch(/^OD\d{3}$/);
    const byYear = countPapersByYear();
    const years = Object.keys(byYear).map(Number).sort((a, b) => a - b);
    expect(years[0]).toBe(2015);
    expect(years[years.length - 1]).toBe(2026);
    expect(years).toEqual([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]);
    expect(PAPERS.filter((p) => p.year < 2015 || p.year > 2026)).toEqual([]);
    expect(manifest.map((m) => m.id).sort()).toEqual(PAPERS.map((p) => p.id).sort());
  });

  it("every paper id referenced anywhere resolves", () => {
    const ids = new Set(PAPERS.map((p) => p.id));
    const referenced = new Set<string>();
    const add = (pid: string) => referenced.add(pid);

    // p.evidences holds corpus concept tags, validated by runAudit()'s
    // tag-vocabulary check — not paper ids.
    for (const pr of PROBLEM_LIFECYCLES)
      pr.occurrences.forEach((o) => { o.papers.forEach(add); o.evidence.forEach((e) => add(e.paperId)); o.attemptedSolutions.forEach((a) => add(a.paperId)); });
    for (const s of SOLUTIONS) s.paperIds.forEach(add);
    for (const c of CLUSTERS) { c.papers.forEach(add); c.evidence.forEach((e) => e.paperIds.forEach(add)); }
    for (const d of FUTURE_DIRECTIONS) {
      add(d.sourcePaperId);
      d.followUps.forEach(add);
      if (d.realizingPaperId) add(d.realizingPaperId);
    }
    for (const s of SATURATION) s.evidence.forEach((e) => e.paperIds.forEach(add));
    for (const dom of INDUSTRY.domains) dom.paperIds.forEach(add);
    for (const g of INDUSTRY.gaps) g.paperIds.forEach(add);
    for (const a of ANALOGIES) a.paperIds.forEach(add);
    for (const r of RELATIONS) { add(r.from); add(r.to); }
    for (const b of BENCHMARKS) add(b.paperId);
    for (const d of DATASETS) d.paperIds.forEach(add);
    for (const m of METRICS) m.papers.forEach(add);
    for (const y of Object.values(YEAR_STATES)) y.supportingPapers.forEach(add);

    const dangling = [...referenced].filter((pid) => !ids.has(pid));
    expect(dangling).toEqual([]);
  });

  it("year states describe their year and stay in range", () => {
    for (const ys of Object.values(YEAR_STATES)) {
      expect(ys.year).toBeGreaterThanOrEqual(2015);
      expect(ys.year).toBeLessThanOrEqual(2026);
      expect(ys.supportingPapers.every((p) => PAPER_BY_ID[p])).toBe(true);
      expect(ys.supportingPapers.length).toBe(
        PAPERS.filter((p) => p.year === ys.year).length,
      );
    }
  });

  it("saturation statuses match the judgments merged into problems", () => {
    const status = new Map(PROBLEM_LIFECYCLES.map((p) => [p.id, p.currentStatus]));
    for (const s of SATURATION) expect(s.status).toBe(status.get(s.problemId));
  });

  it("harvest records cover every paper with full-section evidence", () => {
    expect(Object.keys(HARVEST_RECORDS).length).toBe(121);
    for (const p of PAPERS) {
      const rec = HARVEST_RECORDS[p.id];
      expect(rec, `missing record for ${p.id}`).toBeTruthy();
      expect(rec.problem.summary.length).toBeGreaterThan(0);
      expect(rec.solution.approach.length).toBeGreaterThan(0);
      expect(rec.experiments.summary.length).toBeGreaterThan(0);
      expect(rec.background.summary.length).toBeGreaterThan(0);
      const quotes =
        rec.problem.evidence.length +
        rec.background.evidence.length +
        rec.solution.evidence.length +
        rec.experiments.evidence.length +
        rec.limitations.evidence.length +
        rec.industry.evidence.length;
      expect(quotes, `${p.id} section evidence`).toBeGreaterThan(0);
    }
  });

  it("every simulator link on solutions and clusters resolves", () => {
    const ids = new Set(SIM_IDS);
    for (const s of SOLUTIONS) if (s.simulator) expect(ids.has(s.simulator), `${s.id} -> ${s.simulator}`).toBe(true);
    for (const c of CLUSTERS) if (c.simulator) expect(ids.has(c.simulator), `${c.id} -> ${c.simulator}`).toBe(true);
  });
});

describe("chain of trust: harvest quotes verbatim in PDF text", () => {
  it("every quote in all 121 harvest records appears in its paper's pdftotext output", () => {
    const failures: string[] = [];
    for (const entry of manifest) {
      const textN = norm(pdfText(entry));
      const quotes = collectQuotes(JSON.parse(harvestRaw(entry.id)));
      expect(quotes.length, `${entry.id} harvest has quotes`).toBeGreaterThan(0);
      for (const q of quotes) {
        if (!textN.includes(norm(q))) {
          failures.push(`${entry.id}: ${JSON.stringify(q.slice(0, 80))}`);
        }
      }
    }
    expect(failures).toEqual([]);
  });
});

describe("chain of trust: data quotes verbatim in harvest records", () => {
  /* A data-file quote must appear inside one of the harvest record's own
   * quote strings (which chain A proved are verbatim in the PDF text),
   * under the same normalization. Strict overlap either way covers both
   * exact copies and deliberate sub-quote clipping. */
  function harvestQuotes(pid: string): string[] {
    return collectQuotes(JSON.parse(harvestRaw(pid))).map(norm);
  }

  it("every quote used by a data file appears in its paper's harvest quotes", () => {
    const failures: string[] = [];
    for (const q of dataQuotes()) {
      const nq = norm(q.quote);
      const hq = harvestQuotes(q.paperId);
      const ok = hq.some((h) => h.includes(nq) || nq.includes(h));
      if (!ok) failures.push(`${q.where} (${q.paperId}): ${JSON.stringify(q.quote.slice(0, 80))}`);
    }
    expect(failures).toEqual([]);
  });

  it("data quotes carry a location and stay within size/shape limits", () => {
    for (const q of dataQuotes()) {
      expect(q.quote.trim().length).toBeGreaterThan(0);
      expect(q.quote.length).toBeLessThanOrEqual(300);
      expect(PAPER_BY_ID[q.paperId]).toBeTruthy();
    }
  });
});
