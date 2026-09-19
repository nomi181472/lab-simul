"use client";

import { SECTIONS, type SectionId } from "@/labs/object-detection/data/types";
import { useLab } from "@/labs/object-detection/context";
import { YearTicker } from "@/labs/object-detection/year-ticker";
import { OverviewView } from "@/labs/object-detection/views/overview";
import { YearView } from "@/labs/object-detection/views/year";
import { EvolutionView } from "@/labs/object-detection/views/evolution";
import { ProblemsView } from "@/labs/object-detection/views/problems";
import { DirectionsView } from "@/labs/object-detection/views/directions";
import { ApplicationsView } from "@/labs/object-detection/views/applications";
import { BenchmarkView } from "@/labs/object-detection/views/benchmark";
import { ExplorerView } from "@/labs/object-detection/views/explorer";
import { AskView } from "@/labs/object-detection/views/ask";
import { MissingView } from "@/labs/object-detection/views/missing";
import { PapersView } from "@/labs/object-detection/views/papers";
import { MathView } from "@/labs/object-detection/views/math";
import { EdgeView } from "@/labs/object-detection/views/edge";
import { AuditView } from "@/labs/object-detection/views/audit";

export function LabShell({ children }: { children: React.ReactNode }) {
  const { section, setSection } = useLab();

  return (
    <div className="flex flex-1 flex-col">
      {/* brand + lab-local top bar */}
      <header className="border-b border-zinc-800 bg-zinc-950/95 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-emerald-700/60 bg-emerald-950/40 font-mono text-[13px] font-bold text-emerald-300">
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
                Research Lab · 48 papers · 2015–2026 · evidence-grounded
              </p>
            </div>
          </div>
          <YearTicker />
        </div>
        {/* internal section nav */}
        <nav className="mx-auto w-full max-w-7xl overflow-x-auto px-4">
          <div className="flex min-w-max gap-1 pb-2">
            {SECTIONS.map((s) => (
              <button
                key={s.id}
                onClick={() => setSection(s.id)}
                className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${
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
          <span>CORPUS: 48 papers (49 PDFs) · 2015–2026</span>
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
    directions: DirectionsView,
    applications: ApplicationsView,
    benchmark: BenchmarkView,
    explorer: ExplorerView,
    ask: AskView,
    missing: MissingView,
    papers: PapersView,
    math: MathView,
    edge: EdgeView,
    audit: AuditView,
  };
  const Comp = map[section];
  return <Comp />;
}