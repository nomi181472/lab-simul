"use client";

import { useState } from "react";
import { YEAR_DELTAS } from "@/labs/object-detection/data/year-deltas";
import { useLab } from "@/labs/object-detection/context";
import { Card, Badge, PaperLink, type BadgeTone } from "@/labs/object-detection/ui";

function DeltaCol({
  label,
  items,
  tone,
}: {
  label: string;
  items: string[];
  tone: BadgeTone;
}) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="flex items-center gap-2">
        <Badge tone={tone}>{label}</Badge>
        <span className="font-mono text-[10px] text-zinc-600">{items.length}</span>
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {items.length === 0 ? (
          <span className="font-mono text-[11px] text-zinc-600">—</span>
        ) : (
          items.map((it, i) => (
            <Badge key={`${it}-${i}`} tone={tone}>
              {it}
            </Badge>
          ))
        )}
      </div>
    </div>
  );
}

export function EvolutionView() {
  const { setYear, setSection } = useLab();
  const [open, setOpen] = useState<Record<number, boolean>>({});

  const openYear = (y: number) => {
    setYear(y);
    setSection("year");
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Evolution</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          {YEAR_DELTAS.length} year-over-year deltas over the corpus: which approaches entered,
          which fell away, which persisted, and which problems were newly named. Click a delta
          header to open the destination year&apos;s field state.
        </p>
      </div>

      {/* timeline */}
      <div className="relative space-y-5 border-l border-zinc-800 pl-6">
        {YEAR_DELTAS.map((d) => (
          <div key={d.from} className="relative">
            <span className="absolute -left-[31px] top-5 h-2.5 w-2.5 rounded-full border border-emerald-700 bg-emerald-950" />
            <Card>
              <button
                onClick={() => openYear(d.to)}
                className="group flex w-full flex-wrap items-center justify-between gap-2 text-left"
              >
                <span className="flex items-center gap-2 font-mono text-sm text-zinc-100 group-hover:text-emerald-200">
                  {d.from}
                  <span className="text-emerald-500">→</span>
                  {d.to}
                  <Badge tone="emerald">open {d.to}</Badge>
                </span>
                <span className="font-mono text-[10px] text-zinc-500">
                  {d.supportingPapers.length} supporting papers
                </span>
              </button>

              <p className="mt-2 text-xs leading-5 text-zinc-400">{d.summary}</p>

              <div className="mt-3 grid gap-3 sm:grid-cols-3">
                <DeltaCol label="added" items={d.added} tone="emerald" />
                <DeltaCol label="removed" items={d.removed} tone="rose" />
                <DeltaCol label="persisted" items={d.persisted} tone="zinc" />
              </div>

              <div className="mt-3">
                <DeltaCol label="new problems" items={d.newProblems} tone="violet" />
              </div>

              <div className="mt-3 border-t border-zinc-800 pt-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                    evidence · {d.supportingPapers.length}
                  </span>
                  {(open[d.from] ?? false) && (
                    <button
                      onClick={() => setOpen((o) => ({ ...o, [d.from]: false }))}
                      className="rounded-md border border-zinc-700 px-1.5 py-0.5 font-mono text-[10px] uppercase text-zinc-400 hover:bg-zinc-800"
                    >
                      show less
                    </button>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {(open[d.from] ?? false)
                    ? d.supportingPapers.map((id) => <PaperLink key={id} id={id} />)
                    : d.supportingPapers.slice(0, 8).map((id) => (
                        <PaperLink key={id} id={id} />
                      ))}
                </div>
                {d.supportingPapers.length > 8 && !(open[d.from] ?? false) && (
                  <button
                    onClick={() => setOpen((o) => ({ ...o, [d.from]: true }))}
                    className="mt-2 rounded-md border border-zinc-700 px-1.5 py-0.5 font-mono text-[10px] uppercase text-zinc-400 hover:bg-zinc-800"
                  >
                    +{d.supportingPapers.length - 8} more
                  </button>
                )}
              </div>
            </Card>
          </div>
        ))}
      </div>

      <Card>
        <div className="flex flex-wrap items-start gap-2">
          <Badge tone="emerald">policy</Badge>
          <p className="text-[11px] leading-5 text-zinc-400">
            Each delta is computed from the corpus&apos;s own year states — no transition is
            asserted unless papers in both years evidence it. Missing fields render a muted “—”.
          </p>
        </div>
      </Card>
    </div>
  );
}
