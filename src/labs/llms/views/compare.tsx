"use client";

import { useMemo, useState } from "react";
import {
  Card,
  SectionTitle,
  Badge,
  Quote,
  Unstated,
  CiteBadge,
  TfPaperLink,
  ArchChip,
  ChangeChip,
} from "@/labs/llms/ui";
import { PAPERS } from "@/labs/llms/data/papers";
import { ARCH_LABEL } from "@/labs/llms/data/vocab";

/** Union of the two papers' structural elements, so gaps line up column-wise. */
function compareColumns(a: (typeof PAPERS)[number], b: (typeof PAPERS)[number]) {
  const kinds = new Set<string>();
  for (const l of [...a.architecture.layers, ...b.architecture.layers]) kinds.add(l.kind);
  return [...kinds].sort();
}

export function CompareView() {
  const [leftId, setLeftId] = useState<string>(PAPERS[0]?.id ?? "");
  const [rightId, setRightId] = useState<string>(PAPERS[1]?.id ?? "");

  const left = PAPERS.find((p) => p.id === leftId) ?? PAPERS[0];
  const right = PAPERS.find((p) => p.id === rightId) ?? PAPERS[1];

  const kinds = useMemo(
    () => (left && right ? compareColumns(left, right) : []),
    [left, right],
  );

  if (!left || !right) {
    return <p className="font-mono text-[11px] text-zinc-600">corpus too small to compare</p>;
  }

  const selector = (
    value: string,
    onChange: (v: string) => void,
    exclude: string,
  ) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-md border border-zinc-800 bg-black/40 px-2 py-1.5 font-mono text-[11px] text-zinc-200"
    >
      {PAPERS.map((p) => (
        <option key={p.id} value={p.id} disabled={p.id === exclude}>
          {p.id} · {p.shortTitle}
        </option>
      ))}
    </select>
  );

  return (
    <div className="space-y-6">
      <SectionTitle>Compare</SectionTitle>

      <Card>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <p className="font-mono text-[10px] uppercase tracking-wide text-cyan-400">left</p>
            {selector(leftId, setLeftId, rightId)}
          </div>
          <div className="space-y-1.5">
            <p className="font-mono text-[10px] uppercase tracking-wide text-cyan-400">right</p>
            {selector(rightId, setRightId, leftId)}
          </div>
        </div>
      </Card>

      <Card>
        <div className="grid gap-3 sm:grid-cols-3">
          {[
            { label: "citations", l: left.citations, r: right.citations },
            { label: "year", l: left.year ?? 0, r: right.year ?? 0 },
            { label: "pages", l: left.pages, r: right.pages },
            { label: "structural elements", l: left.architecture.layers.length, r: right.architecture.layers.length },
            { label: "transformations", l: left.changes.length, r: right.changes.length },
            { label: "benchmarks named", l: left.results.length, r: right.results.length },
          ].map((row) => (
            <div key={row.label} className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-2.5">
              <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-600">
                {row.label}
              </p>
              <div className="mt-1 flex items-baseline justify-between gap-2">
                <span className="font-mono text-[15px] text-zinc-100">
                  {typeof row.l === "number" && row.label === "citations"
                    ? row.l.toLocaleString()
                    : row.l}
                </span>
                <span className="font-mono text-[15px] text-zinc-100">
                  {typeof row.r === "number" && row.label === "citations"
                    ? row.r.toLocaleString()
                    : row.r}
                </span>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <SectionTitle>Structural elements, side by side</SectionTitle>
        <div className="mt-3 space-y-1.5">
          {kinds.map((k) => {
            const ll = left.architecture.layers.find((l) => l.kind === k);
            const rl = right.architecture.layers.find((l) => l.kind === k);
            return (
              <div key={k} className="grid gap-2 sm:grid-cols-2">
                <div
                  className={`rounded-lg border p-2 ${
                    ll ? "border-zinc-700 bg-zinc-900/60" : "border-dashed border-zinc-800/70"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-[10px] text-zinc-400">
                      {ARCH_LABEL[k as keyof typeof ARCH_LABEL] ?? k}
                    </span>
                    {ll && <ArchChip kind={ll.kind} />}
                  </div>
                  {ll ? (
                    <div className="mt-1">
                      <Quote>{ll.quote}</Quote>
                    </div>
                  ) : (
                    <div className="mt-1">
                      <Unstated what={k} />
                    </div>
                  )}
                </div>
                <div
                  className={`rounded-lg border p-2 ${
                    rl ? "border-zinc-700 bg-zinc-900/60" : "border-dashed border-zinc-800/70"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-[10px] text-zinc-400">
                      {ARCH_LABEL[k as keyof typeof ARCH_LABEL] ?? k}
                    </span>
                    {rl && <ArchChip kind={rl.kind} />}
                  </div>
                  {rl ? (
                    <div className="mt-1">
                      <Quote>{rl.quote}</Quote>
                    </div>
                  ) : (
                    <div className="mt-1">
                      <Unstated what={k} />
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      <div className="grid gap-3 lg:grid-cols-2">
        {[left, right].map((p) => (
          <Card key={p.id}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge tone="cyan">{p.id}</Badge>
                <CiteBadge n={p.citations} />
              </div>
              <TfPaperLink id={p.id} link short />
            </div>
            <p className="mt-1.5 text-[13px] leading-5 text-zinc-100">{p.title}</p>
            <div className="mt-2 flex flex-wrap gap-1">
              {p.changes.length === 0 ? (
                <Unstated what="modification" />
              ) : (
                p.changes.map((c) => <ChangeChip key={c.operator} operator={c.operator} />)
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}