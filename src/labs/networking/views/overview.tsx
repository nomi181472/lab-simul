"use client";

import { Card, SectionTitle, Badge, CiteBadge, BarRow, truncate } from "@/labs/networking/ui";
import { useNetworking } from "@/labs/networking/context";
import { CORPUS_STATS, KB } from "@/labs/networking/data/kb";
import { MANIFEST } from "@/labs/networking/data/manifest";
import { ANALYSIS } from "@/labs/networking/data/analysis";

export function OverviewView() {
  const { setSection } = useNetworking();

  const topDomains = [...KB.domains]
    .sort((a, b) => b.paperIds.length - a.paperIds.length)
    .slice(0, 8);
  const maxDom = topDomains[0]?.paperIds.length ?? 1;
  const topPapers = MANIFEST.papers.slice(0, 5);
  const sat = ANALYSIS.saturation.summary;

  return (
    <div className="space-y-6">
      <SectionTitle>
        Networking research intelligence · {CORPUS_STATS.total} papers
      </SectionTitle>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card tone="accent">
          <p className="font-mono text-[10px] uppercase tracking-wide text-sky-400">corpus</p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{CORPUS_STATS.total}</p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            {MANIFEST.window} · citation-ordered
          </p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">citations</p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {CORPUS_STATS.citationsTotal.toLocaleString()}
          </p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">OpenAlex cited_by_count</p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">evidence depth</p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {CORPUS_STATS.fullText}
            <span className="text-sm text-zinc-500"> / {CORPUS_STATS.total}</span>
          </p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            full-text · {CORPUS_STATS.withAbstract} abstract-only
          </p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">saturation</p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {sat.saturated}
            <span className="text-sm text-zinc-500"> saturated</span>
          </p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            {sat.active} active · {sat.sparse} sparse domains
          </p>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            problem domains by paper count
          </p>
          <div className="space-y-1">
            {topDomains.map((d) => (
              <BarRow
                key={d.id}
                label={d.label}
                value={d.paperIds.length}
                max={maxDom}
                onClick={() => setSection("matrix")}
              />
            ))}
          </div>
          <button
            onClick={() => setSection("evolution")}
            className="mt-3 font-mono text-[10px] text-sky-400 hover:text-sky-300"
          >
            → problem &amp; solution evolution
          </button>
        </Card>

        <Card>
          <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            most cited in the corpus
          </p>
          <div className="space-y-2">
            {topPapers.map((p) => (
              <div key={p.id} className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate text-[12px] text-zinc-200">{p.title}</p>
                  <p className="font-mono text-[10px] text-zinc-500">
                    {p.id} · {p.year} · {truncate(p.venue || "venue not recorded", 42)}
                  </p>
                </div>
                <CiteBadge n={p.citations} />
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            solution families
          </p>
          <div className="flex flex-wrap gap-1">
            {KB.solutions.slice(0, 10).map((s) => (
              <button key={s.familyId} onClick={() => setSection("evolution")}>
                <Badge tone="violet">{s.label}</Badge>
              </button>
            ))}
          </div>
        </Card>
        <Card>
          <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            future directions
          </p>
          <p className="text-2xl font-semibold text-zinc-100">
            {ANALYSIS.future.counts.realized}
            <span className="text-sm text-zinc-500"> realized</span> ·{" "}
            {ANALYSIS.future.counts.unresolved}
            <span className="text-sm text-zinc-500"> unresolved</span>
          </p>
          <button
            onClick={() => setSection("future")}
            className="mt-2 font-mono text-[10px] text-sky-400 hover:text-sky-300"
          >
            → future direction tracker
          </button>
        </Card>
        <Card>
          <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            industry evidence
          </p>
          <p className="text-2xl font-semibold text-zinc-100">
            {ANALYSIS.academicIndustry.industryEvidenceCount}
            <span className="text-sm text-zinc-500"> papers</span>
          </p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            {ANALYSIS.academicIndustry.academicOnlyCount} academic-only
          </p>
          <button
            onClick={() => setSection("adoption")}
            className="mt-2 font-mono text-[10px] text-sky-400 hover:text-sky-300"
          >
            → adoption explorer
          </button>
        </Card>
      </div>

      <Card tone="warn">
        <p className="text-[12px] leading-6 text-zinc-400">
          <span className="font-semibold text-amber-300">Evidence policy:</span> problem,
          solution, experiment, limitation and future-work fields are sentences quoted from
          the paper&apos;s own text (PDF body where a file was retrieved, else the OpenAlex
          abstract — each record badges which). Fields with no matching sentence read{" "}
          <span className="font-mono text-zinc-500">not stated in the retrieved text</span>{" "}
          rather than being guessed.
        </p>
      </Card>
    </div>
  );
}
