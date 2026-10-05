import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement, type ComponentType } from "react";
import { RootProvider } from "../src/root/root-context";
import { WifiLabProvider } from "../src/labs/wifi-sensing/context";
import { WifiLabShell } from "../src/labs/wifi-sensing/shell";
import { SECTIONS } from "../src/labs/wifi-sensing/data/types";
import { SIMULATORS } from "../src/labs/wifi-sensing/sims";
import { AuditView } from "../src/labs/wifi-sensing/views/audit";
import { CompareView } from "../src/labs/wifi-sensing/views/compare";
import { DatasetsView } from "../src/labs/wifi-sensing/views/datasets";
import { FoundationsView } from "../src/labs/wifi-sensing/views/foundations";
import { GapsView } from "../src/labs/wifi-sensing/views/gaps";
import { GraphView } from "../src/labs/wifi-sensing/views/graph";
import { MathView } from "../src/labs/wifi-sensing/views/math";
import { MetricsView } from "../src/labs/wifi-sensing/views/metrics";
import { OverviewView } from "../src/labs/wifi-sensing/views/overview";
import { PapersView } from "../src/labs/wifi-sensing/views/papers";
import { PathView } from "../src/labs/wifi-sensing/views/path";
import { SignalView } from "../src/labs/wifi-sensing/views/signal";
import { TimelineView } from "../src/labs/wifi-sensing/views/timeline";

// One entry per section in SECTIONS, so this fails loudly if a view is renamed
// or dropped without the registry being updated.
const VIEWS: Record<string, ComponentType> = {
  overview: OverviewView,
  path: PathView,
  timeline: TimelineView,
  graph: GraphView,
  gaps: GapsView,
  foundations: FoundationsView,
  signal: SignalView,
  math: MathView,
  papers: PapersView,
  compare: CompareView,
  datasets: DatasetsView,
  metrics: MetricsView,
  audit: AuditView,
};

function render(node: React.ReactNode) {
  return renderToStaticMarkup(
    createElement(RootProvider, null, createElement(WifiLabProvider, null, node)),
  );
}

// The lab is client-rendered, so no other check exercises the views. Rendering
// each one catches what a type check cannot: a view reading an undefined import
// or a missing record field at runtime.
describe("wifi lab views render", () => {
  it("registers thirteen sections and a view for each", () => {
    expect(SECTIONS).toHaveLength(13);
    expect(Object.keys(VIEWS).sort()).toEqual(SECTIONS.map((s) => s.id).sort());
  });

  it("renders the lab shell", () => {
    expect(render(createElement(WifiLabShell, null, createElement(OverviewView))).length)
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
  it("registers twenty simulators", () => {
    expect(Object.keys(SIMULATORS)).toHaveLength(20);
  });

  for (const [id, Comp] of Object.entries(SIMULATORS)) {
    it(`renders simulator ${id}`, () => {
      const html = render(createElement(Comp));
      expect(html.length, `${id} rendered almost nothing`).toBeGreaterThan(150);
    });
  }
});
