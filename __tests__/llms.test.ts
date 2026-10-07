import { describe, expect, it } from "vitest";
import { MANIFEST } from "@/labs/llms/data/manifest";
import { PAPERS, PAPER_BY_ID } from "@/labs/llms/data/papers";
import { CONCEPTS } from "@/labs/llms/data/concepts";
import { DATASETS } from "@/labs/llms/data/datasets";
import { SIMULATOR_PAPERS } from "@/labs/llms/sims/papers";
import {
  ALL_ARCH_KINDS,
  ALL_CHANGE_OPERATORS,
  ARCH_COLOR,
  CHANGE_COLOR,
} from "@/labs/llms/data/vocab";
import * as K from "@/labs/llms/sims/kernels";

/**
 * Corpus invariants for the transformers / LLM / micro-LLM lab.
 *
 * These are the checks that would otherwise fail silently: a manifest that
 * claims 420 papers while the generated records hold 400, a record whose
 * "evidence" is an empty string, a simulator citing a paper that does not
 * exist.
 */

const MIN_CORPUS = 400;

describe("corpus size and shape", () => {
  it(`holds at least ${MIN_CORPUS} papers`, () => {
    expect(PAPERS.length).toBeGreaterThanOrEqual(MIN_CORPUS);
    expect(MANIFEST.total).toBe(PAPERS.length);
    expect(MANIFEST.papers.length).toBe(PAPERS.length);
  });

  it("uses unique ids", () => {
    expect(new Set(PAPERS.map((p) => p.id)).size).toBe(PAPERS.length);
  });

  it("does not repeat a title", () => {
    const norm = (t: string) => t.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    const seen = new Map<string, string[]>();
    for (const p of PAPERS) {
      const k = norm(p.title);
      seen.set(k, [...(seen.get(k) ?? []), p.id]);
    }
    const dupes = [...seen.entries()].filter(([, ids]) => ids.length > 1);
    expect(dupes.map(([t, ids]) => `${t} -> ${ids.join(",")}`)).toEqual([]);
  });

  it("gives every paper a fileName that the manifest agrees with", () => {
    for (const m of MANIFEST.papers) {
      const p = PAPER_BY_ID[m.id];
      expect(p, `manifest ${m.id} has no record`).toBeDefined();
      expect(p!.fileName).toBe(m.fileName);
    }
  });
});

describe("requested ordering and date window", () => {
  it("is sorted by descending citation count", () => {
    for (let i = 1; i < MANIFEST.papers.length; i++) {
      expect(
        MANIFEST.papers[i - 1].citations,
        `${MANIFEST.papers[i - 1].id} (${MANIFEST.papers[i - 1].citations}) must be >= ` +
          `${MANIFEST.papers[i].id} (${MANIFEST.papers[i].citations})`,
      ).toBeGreaterThanOrEqual(MANIFEST.papers[i].citations);
    }
  });

  it("keeps generated records in the same descending order", () => {
    for (let i = 1; i < PAPERS.length; i++) {
      expect(PAPERS[i - 1].citations).toBeGreaterThanOrEqual(PAPERS[i].citations);
    }
  });

  // 2017, not 2019: the floor was moved so Vaswani et al., GPT-1/2 and BERT are
  // in the corpus. Without them the lab describes efficient variants of a
  // transformer without ever showing what it descends from.
  it("holds no paper outside 2017-2026", () => {
    for (const p of PAPERS) {
      expect(p.year, `${p.id} has year ${p.year}`).not.toBeNull();
      expect(p.year!).toBeGreaterThanOrEqual(2017);
      expect(p.year!).toBeLessThanOrEqual(2026);
    }
  });

  it("carries the foundational papers the arc starts from", () => {
    const hay = MANIFEST.papers.map((p) => p.title.toLowerCase()).join("\n");
    for (const needle of [
      "attention is all you need",
      "language models are few-shot",
      "sentence-bert",
      "gpt",
    ]) {
      expect(hay, `no foundational paper matching ${needle}`).toContain(needle);
    }
  });

  it("sums its citation total correctly", () => {
    const sum = MANIFEST.papers.reduce((n, p) => n + p.citations, 0);
    expect(MANIFEST.totalCitations).toBe(sum);
  });
});

describe("evidence policy", () => {
  it("never leaves an empty quote", () => {
    const thin: string[] = [];
    for (const p of PAPERS) {
      for (const l of p.architecture.layers) {
        if (l.quote.trim().length < 20) thin.push(`${p.id}:${l.kind}`);
      }
      for (const c of p.changes) {
        if (c.quote.trim().length < 20) thin.push(`${p.id}:${c.operator}`);
      }
      for (const o of p.objectives) {
        if (o.quote.trim().length < 20) thin.push(`${p.id}:${o.kind}`);
      }
      for (const r of p.results) {
        if (r.quote.trim().length < 20) thin.push(`${p.id}:${r.dataset}`);
      }
    }
    expect(thin).toEqual([]);
  });

  it("never attaches an undeclared arch kind or operator", () => {
    const bad: string[] = [];
    for (const p of PAPERS) {
      for (const l of p.architecture.layers)
        if (!ALL_ARCH_KINDS.includes(l.kind)) bad.push(`${p.id}:${l.kind}`);
      for (const c of p.changes)
        if (!ALL_CHANGE_OPERATORS.includes(c.operator)) bad.push(`${p.id}:${c.operator}`);
    }
    expect(bad).toEqual([]);
  });

  it("always carries an architecture block, even when empty", () => {
    for (const p of PAPERS) {
      expect(p.architecture, `${p.id} has no architecture block`).toBeDefined();
      expect(Array.isArray(p.architecture.layers)).toBe(true);
      expect(typeof p.architecture.summary).toBe("string");
    }
  });

  it("extracts real text for most of the corpus", () => {
    const ok = PAPERS.filter((p) => p.textChars >= 12000).length;
    // A few repository wrappers are expected to slip through; a corpus where
    // most papers have no text means the extractor is broken, not unlucky.
    expect(ok / PAPERS.length).toBeGreaterThan(0.9);
  });

  it("finds an architecture stack on a majority of papers", () => {
    const n = PAPERS.filter((p) => p.architecture.layers.length > 0).length;
    expect(n / PAPERS.length).toBeGreaterThan(0.5);
  });

  it("finds a modification operator on a meaningful minority", () => {
    const n = PAPERS.filter((p) => p.changes.length > 0).length;
    expect(n / PAPERS.length).toBeGreaterThan(0.25);
  });
});

describe("knowledge base links", () => {
  it("only links papers that exist", () => {
    const dangling: string[] = [];
    for (const c of CONCEPTS)
      for (const id of c.papers) if (!PAPER_BY_ID[id]) dangling.push(`${c.key}->${id}`);
    for (const d of DATASETS)
      for (const id of d.papers) if (!PAPER_BY_ID[id]) dangling.push(`${d.key}->${id}`);
    expect(dangling).toEqual([]);
  });

  it("does not link a concept to an empty paper list", () => {
    expect(CONCEPTS.filter((c) => c.papers.length === 0)).toEqual([]);
    expect(DATASETS.filter((d) => d.papers.length === 0)).toEqual([]);
  });

  it("uses unique keys", () => {
    expect(new Set(CONCEPTS.map((c) => c.key)).size).toBe(CONCEPTS.length);
    expect(new Set(DATASETS.map((d) => d.key)).size).toBe(DATASETS.length);
  });
});

describe("simulators", () => {
  it("binds every simulator to papers that exist", () => {
    expect(SIMULATOR_PAPERS.length).toBeGreaterThan(0);
    for (const s of SIMULATOR_PAPERS) {
      expect(s.papers.length, `${s.id} has no papers`).toBeGreaterThan(0);
      for (const id of s.papers) expect(PAPER_BY_ID[id], `${s.id}->${id}`).toBeDefined();
    }
  });

  it("has no duplicate simulator ids", () => {
    expect(new Set(SIMULATOR_PAPERS.map((s) => s.id)).size).toBe(
      SIMULATOR_PAPERS.length,
    );
  });

  it("produces deterministic series and steps", () => {
    const kernels = [
      K.quantization,
      K.pruning,
      K.layerReduction,
      K.distillation,
      K.peft,
      K.moe,
      K.kvCache,
      K.speculative,
      K.retrieval,
      K.reasoning,
    ];
    for (const k of kernels) {
      const a = k.run({});
      const b = k.run({});
      expect(a.series, "series missing").not.toHaveLength(0);
      expect(a.steps.length).toBeGreaterThan(1);
      expect(b).toEqual(a);
    }
  });

  it("produces only finite values", () => {
    const kernels = [
      K.quantization,
      K.pruning,
      K.layerReduction,
      K.distillation,
      K.peft,
      K.moe,
      K.kvCache,
      K.speculative,
      K.retrieval,
      K.reasoning,
    ];
    const bad: string[] = [];
    for (const k of kernels) {
      for (const s of k.run({}).series) {
        for (const p of s.points) {
          if (!Number.isFinite(p.value)) bad.push(`${s.key}/${p.label}`);
        }
      }
    }
    expect(bad).toEqual([]);
  });
});

describe("vocabulary maps are total", () => {
  it("colours every declared arch kind and operator", () => {
    for (const k of ALL_ARCH_KINDS) expect(ARCH_COLOR[k]).toMatch(/^#/);
    for (const o of ALL_CHANGE_OPERATORS) expect(CHANGE_COLOR[o]).toMatch(/^#/);
  });
});

describe("relevance policy as applied to the corpus", () => {
  /*
   * The policy itself lives in scripts/harvest_transformers.py and cannot be
   * imported from TypeScript. Testing a hand-written copy of it here would
   * assert that the copy agrees with itself, so these check the outcome
   * instead: what the policy actually admitted.
   */
  it("admits only papers labelled transformers", () => {
    expect(PAPERS.filter((p) => p.relevance !== "transformers")).toEqual([]);
  });

  it("admitted no paper whose title is in an off-topic domain", () => {
    const off = [
      "power system",
      "power grid",
      "power distribution",
      "volt/var",
      "radio signal",
      "signal classification",
      "protein",
      "molecular docking",
      "drug docking",
      "radiology",
      "histopatholog",
      "eeg",
      "gene expression",
      "traffic flow",
      "recommender system",
    ];
    const leaked = PAPERS.filter((p) => {
      const t = p.title.toLowerCase();
      return off.some((o) => t.includes(o));
    }).map((p) => `${p.id}: ${p.title}`);
    expect(leaked).toEqual([]);
  });

  it("covers both ends of the arc", () => {
    const blob = PAPERS.map((p) => `${p.title} ${p.abstractNote}`).join(" ").toLowerCase();
    // the large-model end and the micro end both have to be represented, or
    // the corpus is not the arc it claims to be
    expect(blob).toMatch(/large language model|llm/);
    expect(blob).toMatch(/edge|on-device|mobile|distil|small language model/);
  });
});