"use client";

import { useNetworking } from "@/labs/networking/context";
import { SECTIONS, type SectionId } from "@/labs/networking/data/sections";
import { MANIFEST } from "@/labs/networking/data/manifest";
import { CORPUS_STATS } from "@/labs/networking/data/kb";
import { OverviewView } from "@/labs/networking/views/overview";
import { CorpusView } from "@/labs/networking/views/corpus";
import { YearsView } from "@/labs/networking/views/years";
import { EvolutionView } from "@/labs/networking/views/evolution";
import { MatrixView } from "@/labs/networking/views/matrix";
import { ClustersView } from "@/labs/networking/views/clusters";
import { IndustryView } from "@/labs/networking/views/industry";
import { FutureView } from "@/labs/networking/views/future";
import { BenchmarksView } from "@/labs/networking/views/benchmarks";
import { SaturationView } from "@/labs/networking/views/saturation";
import { GapsView } from "@/labs/networking/views/gaps";
import { AdoptionView } from "@/labs/networking/views/adoption";
import { SimulatorsView } from "@/labs/networking/views/simulators";
import { AnalogiesView } from "@/labs/networking/views/analogies";
import { AuditView } from "@/labs/networking/views/audit";

export function NetworkingShell({ children }: { children: React.ReactNode }) {
  const { section, setSection } = useNetworking();

  return (
    <div className="flex flex-1 flex-col">
      <header className="border-b border-zinc-800 bg-zinc-950/95 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-2 px-4 pt-3 lg:flex-row lg:items-center lg:justify-between lg:gap-4 lg:pt-4">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-sky-700/60 bg-sky-950/40 font-mono text-[13px] font-bold text-sky-300">
              ⛓
            </div>
            <div className="leading-tight">
              <button
                onClick={() => setSection("overview")}
                className="text-sm font-semibold tracking-tight text-zinc-100"
              >
                NETWORKING &amp; APPLICATION NETWORKS
              </button>
              <p className="font-mono text-[10px] text-zinc-500">
                {MANIFEST.total} papers · {MANIFEST.window} · citation-ordered ·{" "}
                {CORPUS_STATS.fullText} full-text / {CORPUS_STATS.withAbstract} abstract
                extracts · every quote sourced
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
          <span>
            CORPUS: {MANIFEST.total} papers · {MANIFEST.totalCitations.toLocaleString()}{" "}
            citations · {CORPUS_STATS.fullText} full-text extracts
          </span>
          <span>
            EVIDENCE POLICY: every prose field is quoted from the paper, or marked not
            stated
          </span>
        </div>
      </footer>
    </div>
  );
}

export function ActiveNetworkingSection() {
  const { section } = useNetworking();
  const map: Record<SectionId, React.ComponentType> = {
    overview: OverviewView,
    corpus: CorpusView,
    years: YearsView,
    evolution: EvolutionView,
    matrix: MatrixView,
    clusters: ClustersView,
    industry: IndustryView,
    future: FutureView,
    benchmarks: BenchmarksView,
    saturation: SaturationView,
    gaps: GapsView,
    adoption: AdoptionView,
    simulators: SimulatorsView,
    analogies: AnalogiesView,
    audit: AuditView,
  };
  const Comp = map[section];
  return <Comp />;
}
