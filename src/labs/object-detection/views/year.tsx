"use client";

import { useMemo } from "react";
import { yearState, YEAR_LIST } from "@/labs/object-detection/data/years";
import { DELTAS_BY_FROM } from "@/labs/object-detection/data/year-deltas";
import { PAPER_BY_ID } from "@/labs/object-detection/data/papers";
import { useLab } from "@/labs/object-detection/context";
import { Card, Badge, PaperLink, SectionTitle, ConfidenceBadge } from "@/labs/object-detection/ui";
import { runAudit } from "@/labs/object-detection/data/audit";

function FieldList({
  title,
  items,
  tone = "zinc",
}: {
  title: string;
  items: string[];
  tone?: "zinc" | "emerald" | "amber" | "sky" | "rose" | "violet";
}) {
  if (items.length === 0) return null;
  return (
    <div>
      <div className="flex items-center gap-2">
        <Badge tone={tone}>{title}</Badge>
      </div>
      <ul className="mt-2 space-y-1">
        {items.map((it, i) => (
          <li key={i} className="flex gap-2 text-[12px] leading-5 text-zinc-300">
            <span className="text-zinc-600">›</span>
            <span>{it}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function YearView() {
  const { year, setYear } = useLab();
  const state = yearState(year);
  const delta = DELTAS_BY_FROM[year];
  const audit = useMemo(() => runAudit(), []);
  const yearErr = audit.find((c) => c.id === "years");

  if (!state) {
    return (
      <div className="space-y-4">
        <p className="text-sm text-zinc-400">No field-state record for {year}.</p>
      </div>
    );
  }

  const papersOfYear = (state.supportingPapers ?? []).map((id) => PAPER_BY_ID[id]).filter(Boolean);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">
            Field state · {state.year}
          </h1>
          <p className="mt-1 font-mono text-[11px] text-zinc-500">
            {papersOfYear.length} corpus paper{papersOfYear.length === 1 ? "" : "s"} anchor this
            year{yearErr && !yearErr.pass ? " · ⚠ year audit failing" : ""}
          </p>
        </div>
      </div>

      {/* navigation */}
      <div className="flex items-center justify-between">
        {year > YEAR_LIST[0] ? (
          <button
            onClick={() => setYear(year - 1)}
            className="rounded-lg border border-zinc-800 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-900"
          >
            ← {year - 1}
          </button>
        ) : <span />}
        {year < YEAR_LIST[YEAR_LIST.length - 1] ? (
          <button
            onClick={() => setYear(year + 1)}
            className="rounded-lg border border-zinc-800 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-900"
          >
            {year + 1} →
          </button>
        ) : <span />}
      </div>

      {/* delta headline if available */}
      {delta && (
        <Card tone="accent">
          <SectionTitle>What changed since {delta.from}</SectionTitle>
          <p className="mt-2 text-xs leading-5 text-zinc-400">{delta.summary}</p>
          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <FieldList title="added" items={delta.added} tone="emerald" />
            <FieldList title="removed" items={delta.removed} tone="rose" />
            <FieldList title="persisted" items={delta.persisted} tone="zinc" />
            <FieldList title="new problems" items={delta.newProblems} tone="amber" />
            <FieldList title="new architectures" items={delta.newArchitectures} tone="sky" />
            <FieldList title="new losses" items={delta.newLosses} tone="violet" />
          </div>
          {delta.supportingPapers.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-1.5">
              {delta.supportingPapers.map((id) => (
                <PaperLink key={id} id={id} />
              ))}
            </div>
          )}
        </Card>
      )}

      {/* field snapshot grid */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <FieldList title="dominant approaches" items={state.dominantApproaches} tone="emerald" />
        </Card>
        <Card>
          <FieldList title="emerging ideas" items={state.emergingIdeas} tone="sky" />
          <div className="mt-4">
            <FieldList title="declining" items={state.decliningIdeas} tone="rose" />
          </div>
        </Card>
        <Card>
          <FieldList title="problems entering" items={state.newProblems} tone="amber" />
          <div className="mt-4">
            <FieldList title="problems improving" items={state.solvedOrReducedProblems} tone="emerald" />
          </div>
        </Card>
        <Card>
          <FieldList title="persistent problems" items={state.persistentProblems} tone="zinc" />
          <div className="mt-4">
            <FieldList title="unsolved" items={state.unsolvedProblems} tone="rose" />
          </div>
        </Card>
      </div>

      {/* developments */}
      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <FieldList title="mathematics" items={state.mathematicalDevelopments} tone="violet" />
        </Card>
        <Card>
          <FieldList title="architecture" items={state.architecturalDevelopments} tone="sky" />
        </Card>
        <Card>
          <FieldList title="compute & training" items={state.computationalDevelopments} tone="emerald" />
        </Card>
      </div>

      {/* applications + edge cases */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <FieldList title="applications" items={state.applications} tone="zinc" />
        </Card>
        <Card>
          <FieldList title="edge cases surfacing" items={state.edgeCases} tone="amber" />
        </Card>
      </div>

      {/* papers of the year */}
      <Card>
        <SectionTitle>Corpus papers anchoring {state.year}</SectionTitle>
        <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {papersOfYear.map((p) => (
            <div key={p.id} className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3">
              <div className="font-mono text-[10px] text-emerald-400">{p.id}</div>
              <div className="mt-1 text-[12px] font-medium leading-5 text-zinc-200">
                {p.shortTitle}
              </div>
              <div className="mt-1 font-mono text-[10px] text-zinc-500">
                {p.authors.slice(0, 3).join(", ")}
                {p.authors.length > 3 ? " et al." : ""}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* research directions of the year */}
      {state.researchDirections.length > 0 && (
        <Card tone="accent">
          <FieldList title="research directions" items={state.researchDirections} tone="emerald" />
        </Card>
      )}

      <div className="flex items-center gap-2 font-mono text-[10px] text-zinc-600">
        <ConfidenceBadge confidence="HIGH" /> field-state claims above are traced to the year&apos;s
        corpus papers
      </div>
    </div>
  );
}