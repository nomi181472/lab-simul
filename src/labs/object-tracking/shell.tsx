"use client";

import { SECTIONS, type SectionId } from "@/labs/object-tracking/data/types";
import { useTrackingLab } from "@/labs/object-tracking/context";
import { OverviewView } from "@/labs/object-tracking/views/overview";
import { PathView } from "@/labs/object-tracking/views/path";
import { TimelineView } from "@/labs/object-tracking/views/timeline";
import { GraphView } from "@/labs/object-tracking/views/graph";
import { GapsView } from "@/labs/object-tracking/views/gaps";
import { FoundationsView } from "@/labs/object-tracking/views/foundations";
import { PipelineView } from "@/labs/object-tracking/views/pipeline";
import { MathView } from "@/labs/object-tracking/views/math";
import { PapersView } from "@/labs/object-tracking/views/papers";
import { CompareView } from "@/labs/object-tracking/views/compare";
import { DatasetsView } from "@/labs/object-tracking/views/datasets";
import { MetricsView } from "@/labs/object-tracking/views/metrics";
import { AuditView } from "@/labs/object-tracking/views/audit";
import { PAPER_COUNT } from "@/labs/object-tracking/data/papers";
import { MANIFEST } from "@/labs/object-tracking/data/manifest";

export function TrackingLabShell({ children }: { children: React.ReactNode }) {
  const { section, setSection } = useTrackingLab();

  return (
    <div className="flex flex-1 flex-col">
      <header className="border-b border-zinc-800 bg-zinc-950/95 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-2 px-4 pt-3 lg:flex-row lg:items-center lg:justify-between lg:gap-4 lg:pt-4">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-sky-700/60 bg-sky-950/40 font-mono text-[13px] font-bold text-sky-300">
              ◉
            </div>
            <div className="leading-tight">
              <button
                onClick={() => setSection("overview")}
                className="text-sm font-semibold tracking-tight text-zinc-100"
              >
                OBJECT TRACKING
              </button>
              <p className="font-mono text-[10px] text-zinc-500">
                Research Lab · {PAPER_COUNT}/{MANIFEST.length} papers structured · 2014–2026 · evidence-grounded
              </p>
            </div>
          </div>
        </div>
        <nav className="mx-auto w-full max-w-7xl overflow-x-auto px-4">
          <div className="flex min-w-max gap-1 pb-2 pt-2 lg:pt-3">
            {SECTIONS.map((s) => (
              <button
                key={s.id}
                onClick={() => setSection(s.id)}
                title={s.blurb}
                className={`rounded-md px-3 py-1.5 text-[11px] font-medium transition-colors ${
                  section === s.id
                    ? "border border-sky-700/50 bg-sky-600/20 text-sky-300"
                    : "border border-transparent text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
        </nav>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">{children}</main>

      <footer className="border-t border-zinc-800 bg-zinc-950 py-4">
        <div className="mx-auto flex w-full max-w-7xl flex-wrap items-center justify-between gap-2 px-4 font-mono text-[10px] text-zinc-600">
          <span>CORPUS: {MANIFEST.length} papers · {PAPER_COUNT} structured · T001–T159</span>
          <span>EVIDENCE POLICY: no claim without a paper · lab synthesis never shown as author claim</span>
        </div>
      </footer>
    </div>
  );
}

export function ActiveTrackingSection() {
  const { section } = useTrackingLab();
  const map: Record<SectionId, React.ComponentType> = {
    overview: OverviewView,
    path: PathView,
    timeline: TimelineView,
    graph: GraphView,
    gaps: GapsView,
    foundations: FoundationsView,
    pipeline: PipelineView,
    math: MathView,
    papers: PapersView,
    compare: CompareView,
    datasets: DatasetsView,
    metrics: MetricsView,
    audit: AuditView,
  };
  const Comp = map[section];
  return <Comp />;
}
