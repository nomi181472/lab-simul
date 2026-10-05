import { describe, it, expect } from "vitest";
import { SIMULATOR_IDS, SIMULATOR_META } from "../src/labs/neuroevolution/sims";
import { SIM_BINDINGS } from "../src/labs/neuroevolution/sims/papers";
import { PAPERS, PAPER_BY_ID } from "../src/labs/neuroevolution/data/papers";
import { MANIFEST, MANIFEST_BY_ID } from "../src/labs/neuroevolution/data/manifest";
import { MANIFEST as WIFI_MANIFEST } from "../src/labs/wifi-sensing/data/manifest";
import { CONCEPTS } from "../src/labs/neuroevolution/data/concepts";
import { METRICS } from "../src/labs/neuroevolution/data/metrics";
import { DATASETS } from "../src/labs/neuroevolution/data/datasets";

/* The lab's central claim is that every field is grounded in a corpus PDF and
 * that absence is stated rather than filled in. These tests are what stops a
 * regeneration from quietly breaking that. */

const UNSTATED = /not stated|not enumerated|not described|not recorded/i;

describe("simulator grounding", () => {
  it("registers ten simulators, each distinct", () => {
    expect(SIMULATOR_IDS).toHaveLength(10);
    expect(new Set(SIMULATOR_IDS).size).toBe(10);
  });

  it("has a display name and binding for every simulator", () => {
    for (const id of SIMULATOR_IDS) {
      expect(SIMULATOR_META.find((m) => m.id === id), `missing name for ${id}`).toBeTruthy();
      expect(SIM_BINDINGS[id], `missing binding for ${id}`).toBeTruthy();
    }
  });

  it("binds every simulator to real corpus papers", () => {
    const unbound = SIMULATOR_IDS.filter((id) => (SIM_BINDINGS[id]?.length ?? 0) === 0);
    expect(unbound).toEqual([]);
    for (const id of SIMULATOR_IDS) {
      for (const p of SIM_BINDINGS[id]) {
        expect(PAPER_BY_ID[p.id], `${id} -> ${p.id} is not a paper`).toBeTruthy();
        expect(MANIFEST_BY_ID[p.id], `${id} -> ${p.id} is not in the manifest`).toBeTruthy();
      }
    }
  });
});

describe("corpus integrity", () => {
  it("holds three hundred unique, citation-ordered papers", () => {
    expect(PAPERS).toHaveLength(300);
    expect(new Set(PAPERS.map((p) => p.id)).size).toBe(300);
    expect(MANIFEST).toHaveLength(300);
  });

  it("gives every paper a manifest entry and a unique filename", () => {
    const files = new Set<string>();
    for (const p of PAPERS) {
      const m = MANIFEST_BY_ID[p.id];
      expect(m, `${p.id} missing from manifest`).toBeTruthy();
      expect(m.fileName).toBe(p.fileName);
      expect(files.has(p.fileName), `duplicate filename ${p.fileName}`).toBe(false);
      files.add(p.fileName);
    }
  });

  it("orders records and manifest identically, so ranks agree", () => {
    expect(PAPERS.map((p) => p.id)).toEqual(MANIFEST.map((m) => m.id));
  });
});

describe("architecture is on every paper", () => {
  it("never omits the architecture field", () => {
    for (const p of PAPERS) {
      expect(p.architecture, `${p.id} has no architecture`).toBeTruthy();
      expect(p.architecture.summary, `${p.id} architecture has no summary`).toBeTruthy();
      expect(Array.isArray(p.architecture.layers)).toBe(true);
    }
  });

  it("states the phenotype as a known kind, not free text", () => {
    const kinds = new Set([
      "mlp", "cnn", "rnn", "program", "policy", "neurocontroller",
      "architecture-search", "modular", "ensemble", "other",
    ]);
    for (const p of PAPERS) {
      expect(kinds.has(p.architecture.phenotype), `${p.id} -> ${p.architecture.phenotype}`).toBe(true);
    }
  });

  it("carries a quote with every detected layer", () => {
    for (const p of PAPERS) {
      for (const l of p.architecture.layers) {
        expect(l.quote.length, `${p.id} layer ${l.label} has no quote`).toBeGreaterThan(20);
      }
    }
  });

  it("quotes a genotype-phenotype mapping or says it is not stated", () => {
    const missing: string[] = [];
    for (const p of PAPERS) {
      const g = p.architecture.genotypeToPhenotype;
      if (!g || UNSTATED.test(g)) missing.push(p.id);
    }
    // a small minority may genuinely never state it; a large share means the
    // extractor regressed
    expect(missing.length, `${missing.length} papers state no encoding`).toBeLessThan(120);
  });
});

describe("modification is grounded", () => {
  it("records which loci the operators changed, with an operator and quote", () => {
    for (const p of PAPERS) {
      for (const c of p.method.evolution.changes) {
        expect(c.locus, `${p.id} change has no locus`).toBeTruthy();
        expect(c.operator, `${p.id}/${c.locus} has no operator`).toBeTruthy();
        expect(c.quote.length, `${p.id}/${c.locus} has no quote`).toBeGreaterThan(20);
      }
    }
  });

  it("names a real operator variant on the majority of changes", () => {
    const all = PAPERS.flatMap((p) => p.method.evolution.changes);
    const named = all.filter((c) => !/not named|Not stated/.test(c.operator));
    expect(all.length).toBeGreaterThan(0);
    expect(named.length / all.length).toBeGreaterThan(0.4);
  });

  it("never generalises about other people's work as if it were its own", () => {
    // "we commonly use lr=0.01" is this paper's operator; "learning-rate
    // schedules are often hand-crafted" is background and must not be a locus
    const generalising = /\b(typically|often|usually|commonly|generally|frequently)\b/i;
    const firstPerson = /\b(we|our|ours|us)\b|\bthis (paper|work|article|study)\b/i;
    const bad = PAPERS.flatMap((p) =>
      p.method.evolution.changes
        .filter((c) => generalising.test(c.quote) && !firstPerson.test(c.quote))
        .map((c) => `${p.id}/${c.locus}`),
    );
    expect(bad).toEqual([]);
  });

  it("never quotes publisher metadata as evidence", () => {
    const meta = /CCS CONCEPTS|Index Terms|ACM Reference Format|arXiv:\d{4}\.|This work is licensed|copyright/i;
    for (const p of PAPERS) {
      for (const c of p.method.evolution.changes) {
        expect(meta.test(c.quote), `${p.id} quotes metadata: ${c.quote.slice(0, 60)}`).toBe(false);
      }
      for (const l of p.architecture.layers) {
        expect(meta.test(l.quote), `${p.id} layer quotes metadata`).toBe(false);
      }
    }
  });
});

describe("absence is stated, not filled in", () => {
  it("leaves fitness empty rather than inventing one", () => {
    for (const p of PAPERS) {
      const f = p.method.evolution.fitness;
      if (f) expect(f.length).toBeGreaterThan(20);
    }
  });

  it("only links concepts, metrics and datasets that exist", () => {
    const cids = new Set(CONCEPTS.map((c) => c.id));
    const mids = new Set(METRICS.map((m) => m.id));
    const dids = new Set(DATASETS.map((d) => d.id));
    for (const p of PAPERS) {
      for (const c of p.concepts) expect(cids.has(c), `${p.id} -> ${c}`).toBe(true);
      for (const m of p.metrics) expect(mids.has(m), `${p.id} -> ${m}`).toBe(true);
      for (const d of p.datasets) expect(dids.has(d), `${p.id} -> ${d}`).toBe(true);
    }
  });

  it("links relations only to papers that exist", () => {
    for (const p of PAPERS) {
      for (const r of p.relations) {
        expect(PAPER_BY_ID[r.to], `${p.id} -> ${r.to} is not a paper`).toBeTruthy();
        expect(r.to).not.toBe(p.id);
      }
    }
  });

  it("gives every knowledge-base entry the arrays its type requires", () => {
    for (const c of CONCEPTS) {
      expect(Array.isArray(c.paperIds), `${c.id} paperIds`).toBe(true);
      expect(Array.isArray(c.prereqs), `${c.id} prereqs`).toBe(true);
    }
    for (const m of METRICS) expect(Array.isArray(m.paperIds), `${m.id} paperIds`).toBe(true);
    for (const d of DATASETS) {
      expect(Array.isArray(d.paperIds), `${d.id} paperIds`).toBe(true);
      expect(Array.isArray(d.metrics), `${d.id} metrics`).toBe(true);
    }
  });
});

describe("paper viewer links resolve to corpus files", () => {
  it("does not collide with the wifi corpus, since the route serves both", () => {
    // the shared /papers/[file] route resolves a name to one corpus dir, so an
    // overlap would serve the wrong lab's PDF
    const wifi = new Set(WIFI_MANIFEST.map((m) => m.fileName));
    const clash = MANIFEST.map((m) => m.fileName).filter((f) => wifi.has(f));
    expect(clash).toEqual([]);
  });
});