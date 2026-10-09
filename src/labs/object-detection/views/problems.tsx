"use client";

import { useMemo, useState } from "react";
import type { Evidence, ProblemEvidence, ProblemLifecycle } from "@/labs/object-detection/data/types";
import { PROBLEM_LIFECYCLES } from "@/labs/object-detection/data/problems";
import { Badge, Card, EvidenceLine, PaperLink, StatusBadge } from "@/labs/object-detection/ui";
import { Segmented } from "@/labs/object-detection/sims/shared";

type SortKey = "paperCount" | "firstObservedYear" | "id";

const SORTS: { value: SortKey; label: string }[] = [
  { value: "paperCount", label: "papers" },
  { value: "firstObservedYear", label: "first year" },
  { value: "id", label: "id" },
];

const TOP_N = 15;
const STATEMENT_N = 6;

function clip(s: string, n: number) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function occurrenceEvidence(e: ProblemEvidence): Evidence {
  return { paperIds: [e.paperId], kind: "paperSays", confidence: "MEDIUM", quote: e.quote, location: e.location };
}

function LifecycleDetail({ p }: { p: ProblemLifecycle }) {
  const [allStatements, setAllStatements] = useState(false);
  const shown = allStatements ? p.statements : p.statements.slice(0, STATEMENT_N);
  const hidden = p.statements.length - shown.length;

  return (
    <div className="space-y-4 border-t border-zinc-800 px-3 py-4 sm:px-4">
      <div>
        <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
          consolidated problem
        </div>
        <p className="mt-1 text-[13px] leading-5 text-zinc-200">{p.problem}</p>
      </div>

      <div>
        <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
          statements from the corpus ({p.statements.length})
        </div>
        <ul className="mt-1.5 space-y-1">
          {shown.map((s, i) => (
            <li key={i} className="flex gap-2 text-[12px] leading-5 text-zinc-400">
              <span className="text-emerald-700">·</span>
              <span>{s}</span>
            </li>
          ))}
        </ul>
        {p.statements.length > STATEMENT_N && (
          <button
            onClick={() => setAllStatements((v) => !v)}
            className="mt-1.5 rounded-md border border-zinc-800 px-2 py-0.5 font-mono text-[10px] text-zinc-400 hover:text-zinc-200"
          >
            {allStatements ? "show less" : `+${hidden} more`}
          </button>
        )}
      </div>

      <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="emerald">status evidence</Badge>
          <StatusBadge status={p.currentStatus} />
        </div>
        <p className="mt-2 text-[12px] leading-5 text-zinc-400">{p.statusEvidence}</p>
        {p.statusPapers.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {p.statusPapers.map((pid) => (
              <PaperLink key={pid} id={pid} />
            ))}
          </div>
        )}
      </div>

      <div>
        <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
          occurrences ({p.occurrences.length})
        </div>
        <div className="mt-2 space-y-3">
          {p.occurrences
            .slice()
            .sort((a, b) => a.year - b.year)
            .map((o) => (
              <div key={o.year} className="border-l-2 border-zinc-800 pl-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="emerald">{o.year}</Badge>
                  {o.papers.map((pid) => (
                    <PaperLink key={pid} id={pid} />
                  ))}
                </div>

                {o.attemptedSolutions.length > 0 && (
                  <div className="mt-2 space-y-2">
                    <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                      attempted solutions
                    </div>
                    {o.attemptedSolutions.map((s, i) => (
                      <div
                        key={`${s.paperId}-${i}`}
                        className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5"
                      >
                        <div className="flex flex-wrap items-center gap-2">
                          <PaperLink id={s.paperId} />
                          <Badge tone="zinc">{s.year}</Badge>
                          {s.solutionTags.map((t) => (
                            <Badge key={t} tone="sky">
                              {t}
                            </Badge>
                          ))}
                        </div>
                        <p className="mt-1.5 text-[12px] leading-5 text-zinc-400">
                          {clip(s.approach, 220)}
                        </p>
                        {s.quote && (
                          <blockquote className="mt-1.5 border-l-2 border-zinc-700 pl-2 text-[11px] italic leading-4 text-zinc-500">
                            “{clip(s.quote, 220)}”
                          </blockquote>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {o.evidence.length > 0 && (
                  <div className="mt-2 space-y-2">
                    <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                      evidence
                    </div>
                    {o.evidence.map((e, i) => (
                      <EvidenceLine key={i} evidence={occurrenceEvidence(e)} />
                    ))}
                  </div>
                )}
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

export function ProblemsView() {
  const [sort, setSort] = useState<SortKey>("paperCount");
  const [showAll, setShowAll] = useState(false);
  const [q, setQ] = useState("");
  const [openId, setOpenId] = useState<string | null>(null);

  const maxCount = useMemo(
    () => Math.max(...PROBLEM_LIFECYCLES.map((p) => p.paperCount)),
    [],
  );

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const list = PROBLEM_LIFECYCLES.filter((p) => {
      if (!needle) return true;
      const hay = `${p.id} ${p.shortLabel} ${p.problem} ${p.statements.join(" ")}`.toLowerCase();
      return needle.split(/\s+/).filter(Boolean).every((t) => hay.includes(t));
    });
    return list.sort((a, b) => {
      if (sort === "paperCount") return b.paperCount - a.paperCount || a.id.localeCompare(b.id);
      if (sort === "firstObservedYear")
        return a.firstObservedYear - b.firstObservedYear || a.id.localeCompare(b.id);
      return a.id.localeCompare(b.id);
    });
  }, [sort, q]);

  const visible = showAll ? rows : rows.slice(0, TOP_N);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Problem matrix</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          {PROBLEM_LIFECYCLES.length} problem lifecycles traced across the corpus. Each row
          shows how many papers evidence the problem and the years it was observed. Click a
          row to open the consolidated statement, status judgment and occurrence timeline.
        </p>
      </div>

      {/* ---------- controls ---------- */}

      <Card>
        <div className="flex flex-wrap items-end gap-4">
          <Segmented label="sort by" value={sort} options={SORTS} onChange={setSort} />
          <label className="flex min-w-52 flex-col gap-1">
            <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
              filter
            </span>
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="e.g. small objects, NMS, end-to-end…"
              className="rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-[12px] text-zinc-200 placeholder:text-zinc-600 focus:border-emerald-600 focus:outline-none"
            />
          </label>
          <button
            onClick={() => setShowAll((v) => !v)}
            className="rounded-md border border-zinc-800 px-2 py-1.5 font-mono text-[10px] uppercase tracking-wide text-zinc-400 hover:text-zinc-200"
          >
            {showAll ? `top ${TOP_N}` : "show all"}
          </button>
          <div className="ml-auto font-mono text-[10px] text-zinc-500">
            {visible.length} / {rows.length} problems
          </div>
        </div>

        {/* ---------- matrix ---------- */}

        <div className="mt-4 overflow-x-auto">
          <div className="min-w-[560px]">
            <div className="grid grid-cols-[minmax(0,1fr)_7rem_3rem_7rem] items-center gap-3 border-b border-zinc-800 pb-1.5 font-mono text-[9px] uppercase tracking-wide text-zinc-600">
              <span>problem</span>
              <span>papers</span>
              <span className="text-right">count</span>
              <span className="text-right">years</span>
            </div>

            <div className="divide-y divide-zinc-900">
              {visible.map((p) => {
                const open = openId === p.id;
                return (
                  <div key={p.id} className={open ? "bg-emerald-950/20" : ""}>
                    <button
                      onClick={() => setOpenId(open ? null : p.id)}
                      className="grid w-full grid-cols-[minmax(0,1fr)_7rem_3rem_7rem] items-center gap-3 px-1 py-2 text-left transition-colors hover:bg-zinc-900/60"
                    >
                      <span className="flex min-w-0 items-center gap-2">
                        <span className="truncate text-[12px] text-zinc-200">{p.shortLabel}</span>
                        <StatusBadge status={p.currentStatus} />
                      </span>
                      <span className="h-2.5 w-full overflow-hidden rounded-sm bg-zinc-800">
                        <span
                          className="block h-full bg-emerald-500"
                          style={{ width: `${(p.paperCount / maxCount) * 100}%` }}
                        />
                      </span>
                      <span className="text-right font-mono text-[10px] text-zinc-400">
                        {p.paperCount}
                      </span>
                      <span className="text-right font-mono text-[10px] text-zinc-500">
                        {p.firstObservedYear}–{p.lastObservedYear}
                      </span>
                    </button>
                    {open && <LifecycleDetail p={p} />}
                  </div>
                );
              })}
              {visible.length === 0 && (
                <div className="rounded-lg border border-rose-900/60 bg-rose-950/20 p-3 text-[12px] text-zinc-400">
                  No problems match the current filter.
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
