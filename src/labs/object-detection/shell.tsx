"use client";

import { SECTIONS, type SectionId } from "@/labs/object-detection/data/types";
import { useLab } from "@/labs/object-detection/context";
import { YearTicker } from "@/labs/object-detection/year-ticker";
import { PAPER_COUNT, YEAR_MIN, YEAR_MAX, TOTAL_CITATIONS } from "@/labs/object-detection/data/papers";
import { OverviewView } from "@/labs/object-detection/views/overview";
import { YearView } from "@/labs/object-detection/views/year";
import { EvolutionView } from "@/labs/object-detection/views/evolution";
import { ProblemsView } from "@/labs/object-detection/views/problems";
import { ClustersView } from "@/labs/object-detection/views/clusters";
import { DirectionsView } from "@/labs/object-detection/views/directions";
import { IndustryView } from "@/labs/object-detection/views/industry";
import { SaturationView } from "@/labs/object-detection/views/saturation";
import { BenchmarkView } from "@/labs/object-detection/views/benchmark";
import { ExplorerView } from "@/labs/object-detection/views/explorer";
import { PapersView } from "@/labs/object-detection/views/papers";
import { SimulatorsView } from "@/labs/object-detection/views/simulators";
import { AnalogyView } from "@/labs/object-detection/views/analogy";
import { AskView } from "@/labs/object-detection/views/ask";
import { AuditView } from "@/labs/object-detection/views/audit";

export function LabShell({ children }: { children: React.ReactNode }) {
  const { section, setSection } = useLab();

  return (
    <div className="flex flex-1 flex-col">
      {/* brand + lab-local top bar */}
      <header className="border-b border-zinc-800 bg-zinc-950/95 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-2 px-4 pt-3 sm:pt-3 lg:flex-row lg:items-center lg:justify-between lg:gap-4 lg:pt-4">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-emerald-700/60 bg-emerald-950/40 font-mono text-[13px] font-bold text-emerald-300">
              ⊙
            </div>
            <div className="leading-tight">
              <button
                onClick={() => setSection("overview")}
                className="text-sm font-semibold tracking-tight text-zinc-100"
              >
                OBJECT DETECTION
              </button>
              <p className="font-mono text-[10px] text-zinc-500">
                Research Lab · {PAPER_COUNT} papers · {YEAR_MIN}–{YEAR_MAX} · evidence-grounded
              </p>
            </div>
          </div>
          <YearTicker />
        </div>
        {/* internal section nav */}
        <nav className="mx-auto w-full max-w-7xl overflow-x-auto px-4">
          <div className="flex min-w-max gap-1 pb-2 pt-2 lg:pt-3">
            {SECTIONS.map((s) => (
              <button
                key={s.id}
                onClick={() => setSection(s.id)}
                className={`rounded-md px-3 py-1.5 text-[11px] font-medium transition-colors ${
                  section === s.id
                    ? "bg-emerald-600/20 text-emerald-300 border border-emerald-700/50"
                    : "text-zinc-400 border border-transparent hover:bg-zinc-900 hover:text-zinc-200"
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
          <span>
            CORPUS: {PAPER_COUNT} papers · {YEAR_MIN}–{YEAR_MAX} · {TOTAL_CITATIONS.toLocaleString()} citations
          </span>
          <span>EVIDENCE POLICY: no claim without a paper · INSUFFICIENT_EVIDENCE shown when unsupported</span>
        </div>
      </footer>
    </div>
  );
}

export function ActiveSection() {
  const { section } = useLab();
  const map: Record<SectionId, React.ComponentType> = {
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
  const Comp = map[section];
  return <Comp />;
}
