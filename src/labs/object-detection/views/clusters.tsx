"use client";

import { useMemo, useState } from "react";
import { CLUSTERS } from "@/labs/object-detection/data/clusters";
import { Badge, Card, EvidenceLine, PaperLink, checkSim } from "@/labs/object-detection/ui";
import { Segmented } from "@/labs/object-detection/sims/shared";
import { useLab } from "@/labs/object-detection/context";

type SortKey = "paperCount" | "year" | "name";

const SORTS: { value: SortKey; label: string }[] = [
  { value: "paperCount", label: "papers" },
  { value: "year", label: "year" },
  { value: "name", label: "name" },
];

const PAPER_N = 10;

export function ClustersView() {
  const { setSection, setSim } = useLab();
  const [sort, setSort] = useState<SortKey>("paperCount");
  const [openPapers, setOpenPapers] = useState<Set<string>>(new Set());

  const clusters = useMemo(() => {
    return [...CLUSTERS].sort((a, b) => {
      if (sort === "paperCount") return b.papers.length - a.papers.length || a.id.localeCompare(b.id);
      if (sort === "year") return a.firstYear - b.firstYear || a.id.localeCompare(b.id);
      return a.name.localeCompare(b.name);
    });
  }, [sort]);

  const togglePapers = (id: string) =>
    setOpenPapers((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Problem clusters</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          {CLUSTERS.length} clusters grouping papers that attack the identical problem —
          each card ties the shared problem and solution tags to the papers and quotes that
          evidence the grouping.
        </p>
      </div>

      <Card>
        <div className="flex flex-wrap items-end gap-4">
          <Segmented label="sort by" value={sort} options={SORTS} onChange={setSort} />
          <div className="ml-auto font-mono text-[10px] text-zinc-500">{clusters.length} clusters</div>
        </div>
      </Card>

      <div className="space-y-4">
        {clusters.map((c) => {
          const open = openPapers.has(c.id);
          const shown = open ? c.papers : c.papers.slice(0, PAPER_N);
          const sim = checkSim(c.simulator);
          return (
            <Card key={c.id}>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="text-sm font-semibold tracking-tight text-zinc-100">{c.name}</h2>
                    <Badge tone="zinc">
                      {c.firstYear}–{c.lastYear}
                    </Badge>
                    <Badge tone="emerald">{c.papers.length} papers</Badge>
                  </div>
                  <p className="mt-1.5 text-[12px] leading-5 text-zinc-400">{c.description}</p>
                </div>
                {sim && (
                  <button
                    onClick={() => {
                      setSim(sim);
                      setSection("simulators");
                    }}
                    className="shrink-0 rounded-md border border-emerald-700/60 bg-emerald-950/40 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wide text-emerald-300 hover:bg-emerald-900/40"
                  >
                    open simulator
                  </button>
                )}
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-1.5">
                <span className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                  problems
                </span>
                {c.problemTags.map((t) => (
                  <button key={t} onClick={() => setSection("problems")}>
                    <Badge tone="amber">{t}</Badge>
                  </button>
                ))}
                <span className="ml-3 font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                  solutions
                </span>
                {c.solutionTags.map((t) => (
                  <button key={t} onClick={() => setSection("explorer")}>
                    <Badge tone="sky">{t}</Badge>
                  </button>
                ))}
              </div>

              <div className="mt-3">
                <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                  papers ({c.papers.length})
                </div>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {shown.map((pid) => (
                    <PaperLink key={pid} id={pid} />
                  ))}
                  {c.papers.length > PAPER_N && (
                    <button
                      onClick={() => togglePapers(c.id)}
                      className="rounded-md border border-zinc-800 px-2 py-0.5 font-mono text-[10px] text-zinc-400 hover:text-zinc-200"
                    >
                      {open ? "show fewer" : `+${c.papers.length - PAPER_N} more`}
                    </button>
                  )}
                </div>
              </div>

              {c.evidence.length > 0 && (
                <div className="mt-3 space-y-2">
                  <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                    evidence
                  </div>
                  {c.evidence.map((e, i) => (
                    <EvidenceLine key={i} evidence={e} />
                  ))}
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}
