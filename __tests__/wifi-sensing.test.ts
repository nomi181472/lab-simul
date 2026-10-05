import { describe, it, expect } from "vitest";
import { SIM_PAPERS, SIMULATOR_INDEX, checkSim } from "../src/labs/wifi-sensing/sims/papers";
import { SIMULATOR_IDS } from "../src/labs/wifi-sensing/sims";
import { PAPER_BY_ID } from "../src/labs/wifi-sensing/data/papers";
import { PAPERS } from "../src/labs/wifi-sensing/data/papers";
import { MANIFEST_BY_ID } from "../src/labs/wifi-sensing/data/manifest";
import { CONCEPTS } from "../src/labs/wifi-sensing/data/concepts";
import { METRICS } from "../src/labs/wifi-sensing/data/metrics";
import { DATASETS } from "../src/labs/wifi-sensing/data/datasets";

describe("simulator grounding", () => {
  it("registers exactly twenty simulators", () => {
    expect(SIMULATOR_IDS).toHaveLength(20);
    expect(new Set(SIMULATOR_IDS).size).toBe(20);
  });

  it("has a display name and binding for every simulator", () => {
    for (const id of SIMULATOR_IDS) {
      expect(SIMULATOR_INDEX[id], `missing name for ${id}`).toBeTruthy();
      expect(SIM_PAPERS[id], `missing binding for ${id}`).toBeTruthy();
    }
  });

  it("binds every simulator to at least one real corpus paper", () => {
    const unbound = SIMULATOR_IDS.filter((id) => (SIM_PAPERS[id]?.papers.length ?? 0) === 0);
    expect(unbound).toEqual([]);
  });

  it("only names paper ids that exist in the corpus", () => {
    for (const id of SIMULATOR_IDS) {
      for (const pid of SIM_PAPERS[id]?.papers ?? []) {
        expect(PAPER_BY_ID[pid], `${id} -> ${pid} is not a paper`).toBeTruthy();
        expect(MANIFEST_BY_ID[pid], `${id} -> ${pid} is not in the manifest`).toBeTruthy();
      }
    }
  });

  it("states what each binding reproduces", () => {
    for (const id of SIMULATOR_IDS) {
      expect(SIM_PAPERS[id]?.note, `missing note for ${id}`).toBeTruthy();
    }
  });

  it("rejects an unknown simulator id from the url hash", () => {
    expect(checkSim("doppler")).toBe("doppler");
    expect(checkSim("not-a-sim")).toBeNull();
    expect(checkSim(undefined)).toBeNull();
  });
});

describe("corpus integrity", () => {
  it("holds three hundred unique, citation-ordered papers", () => {
    expect(PAPERS).toHaveLength(300);
    expect(new Set(PAPERS.map((p) => p.id)).size).toBe(300);
  });

  it("keeps manifest and deep records in the same order", () => {
    const manifestIds = Object.keys(MANIFEST_BY_ID);
    expect(manifestIds).toHaveLength(300);
    expect(PAPERS.map((p) => p.id)).toEqual(manifestIds);
  });

  it("never cites a paper outside the corpus", () => {
    for (const p of PAPERS) {
      for (const r of p.relations) {
        expect(PAPER_BY_ID[r.to], `${p.id} -> ${r.to}`).toBeTruthy();
      }
      for (const pid of p.datasets) {
        expect(DATASETS.some((d) => d.id === pid), `${p.id} -> ${pid}`).toBeTruthy();
      }
      for (const mid of p.metrics) {
        expect(METRICS.some((m) => m.id === mid), `${p.id} -> ${mid}`).toBeTruthy();
      }
      for (const eid of p.equationIds) {
        expect(CONCEPTS.some((c) => c.id === eid), `${p.id} -> ${eid}`).toBeTruthy();
      }
    }
  });

  it("keeps every record readable and free of extraction debris", () => {
    for (const p of PAPERS) {
      expect(p.title.length, p.id).toBeGreaterThan(10);
      expect(p.summary.length, p.id).toBeGreaterThan(80);
      // "-layout" extraction interleaved two-column body text into the cache;
      // the raw extraction must not reintroduce that as a stray citation marker
      expect(p.summary, p.id).not.toMatch(/\s,\s*,/);
    }
  });
});