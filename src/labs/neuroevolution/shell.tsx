"use client";

import { SECTIONS, type SectionId } from "@/labs/neuroevolution/data/types";
import { useNeuroLab } from "@/labs/neuroevolution/context";
import { OverviewView } from "@/labs/neuroevolution/views/overview";
import { ArchitectureView } from "@/labs/neuroevolution/views/architecture";
import { ModificationView } from "@/labs/neuroevolution/views/modification";
import { PathView } from "@/labs/neuroevolution/views/path";
import { TimelineView } from "@/labs/neuroevolution/views/timeline";
import { GraphView } from "@/labs/neuroevolution/views/graph";
import { FoundationsView } from "@/labs/neuroevolution/views/foundations";
import { OperatorsView } from "@/labs/neuroevolution/views/operators";
import { PapersView } from "@/labs/neuroevolution/views/papers";
import { CompareView } from "@/labs/neuroevolution/views/compare";
import { DatasetsView } from "@/labs/neuroevolution/views/datasets";
import { MetricsView } from "@/labs/neuroevolution/views/metrics";
import { AuditView } from "@/labs/neuroevolution/views/audit";
import { PAPER_COUNT } from "@/labs/neuroevolution/data/papers";
import { MANIFEST } from "@/labs/neuroevolution/data/manifest";

export function NeuroLabShell({ children }: { children: React.ReactNode }) {
  const { section, setSection } = useNeuroLab();

  return (
    <div className="flex flex-1 flex-col">
      <header className="border-b border-zinc-800 bg-zinc-950/95 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-2 px-4 pt-3 lg:flex-row lg:items-center lg:justify-between lg:gap-4 lg:pt-4">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-rose-700/60 bg-rose-950/40 font-mono text-[13px] font-bold text-rose-300">
              ⧬
            </div>
            <div className="leading-tight">
              <button
                onClick={() => setSection("overview")}
                className="text-sm font-semibold tracking-tight text-zinc-100"
              >
                NEUROEVOLUTION
              </button>
              <p className="font-mono text-[10px] text-zinc-500">
                Research Lab · {PAPER_COUNT}/{MANIFEST.length} papers structured ·
                citation-ordered · every paper carries its evolved architecture
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
                    ? "border border-rose-700/50 bg-rose-600/20 text-rose-300"
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
          <span>CORPUS: {MANIFEST.length} papers · {PAPER_COUNT} structured</span>
          <span>EVIDENCE POLICY: architecture and operators are quoted from the paper · unstated means unstated</span>
        </div>
      </footer>
    </div>
  );
}

export function ActiveNeuroSection() {
  const { section } = useNeuroLab();
  const map: Record<SectionId, React.ComponentType> = {
    overview: OverviewView,
    architecture: ArchitectureView,
    modification: ModificationView,
    path: PathView,
    timeline: TimelineView,
    graph: GraphView,
    foundations: FoundationsView,
    operators: OperatorsView,
    papers: PapersView,
    compare: CompareView,
    datasets: DatasetsView,
    metrics: MetricsView,
    audit: AuditView,
  };
  const Comp = map[section];
  return <Comp />;
}