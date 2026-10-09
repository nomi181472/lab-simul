import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement, type ComponentType } from "react";
import { RootProvider } from "../src/root/root-context";
import { LabProvider } from "../src/labs/object-detection/context";
import { LabShell } from "../src/labs/object-detection/shell";
import { SECTIONS } from "../src/labs/object-detection/data/types";
import { PAPERS } from "../src/labs/object-detection/data/papers";
import { SIM_REGISTRY } from "../src/labs/object-detection/sims/registry";
import { OverviewView } from "../src/labs/object-detection/views/overview";
import { YearView } from "../src/labs/object-detection/views/year";
import { EvolutionView } from "../src/labs/object-detection/views/evolution";
import { ProblemsView } from "../src/labs/object-detection/views/problems";
import { ClustersView } from "../src/labs/object-detection/views/clusters";
import { DirectionsView } from "../src/labs/object-detection/views/directions";
import { IndustryView } from "../src/labs/object-detection/views/industry";
import { SaturationView } from "../src/labs/object-detection/views/saturation";
import { BenchmarkView } from "../src/labs/object-detection/views/benchmark";
import { ExplorerView } from "../src/labs/object-detection/views/explorer";
import { PapersView } from "../src/labs/object-detection/views/papers";
import { SimulatorsView } from "../src/labs/object-detection/views/simulators";
import { AnalogyView } from "../src/labs/object-detection/views/analogy";
import { AskView } from "../src/labs/object-detection/views/ask";
import { AuditView } from "../src/labs/object-detection/views/audit";
import PaperDetail from "../src/labs/object-detection/views/paper-detail";

// One entry per section in SECTIONS, so this fails loudly if a view is renamed
// or dropped without the registry being updated.
const VIEWS: Record<string, ComponentType> = {
  overview: OverviewView,
  year: YearView,
  evolution: EvolutionView,
  problems: ProblemsView,
  clusters: ClustersView,
  directions: DirectionsView,
  industry: IndustryView,
  saturation: SaturationView,
  benchmark: BenchmarkView,
  explorer: ExplorerView,
  papers: PapersView,
  simulators: SimulatorsView,
  analogy: AnalogyView,
  ask: AskView,
  audit: AuditView,
};

function render(node: React.ReactNode) {
  return renderToStaticMarkup(
    createElement(RootProvider, null, createElement(LabProvider, null, node)),
  );
}

// The lab is client-rendered, so no other check exercises the views. Rendering
// each one catches what a type check cannot: a view reading an undefined import
// or a missing record field at runtime.
describe("object detection lab views render", () => {
  it("registers fifteen sections and a view for each", () => {
    expect(SECTIONS).toHaveLength(15);
    expect(Object.keys(VIEWS).sort()).toEqual(SECTIONS.map((s) => s.id).sort());
  });

  it("renders the lab shell", () => {
    expect(render(createElement(LabShell, null, createElement(OverviewView))).length)
      .toBeGreaterThan(500);
  });

  for (const [id, View] of Object.entries(VIEWS)) {
    it(`renders ${id}`, () => {
      const html = render(createElement(View));
      expect(html.length, `${id} rendered almost nothing`).toBeGreaterThan(300);
    });
  }
});

describe("every simulator renders", () => {
  it("registers nineteen simulators", () => {
    expect(SIM_REGISTRY).toHaveLength(19);
  });

  for (const entry of SIM_REGISTRY) {
    it(`renders sim ${entry.id}`, () => {
      const html = render(createElement(entry.comp));
      expect(html.length, `sim ${entry.id} rendered almost nothing`).toBeGreaterThan(150);
    });
  }
});

describe("paper detail renders full harvest records", () => {
  for (const id of ["OD001", "OD049", "OD121"]) {
    it(`renders ${id}`, () => {
      const paper = PAPERS.find((p) => p.id === id);
      expect(paper).toBeTruthy();
      const html = render(createElement(PaperDetail, { p: paper! }));
      expect(html.length, `${id} detail rendered almost nothing`).toBeGreaterThan(1000);
    });
  }
});
