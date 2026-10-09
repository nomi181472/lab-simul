"use client";

import { useState } from "react";
import { INDUSTRY } from "@/labs/object-detection/data/industry";
import { SOLUTION_BY_ID } from "@/labs/object-detection/data/solutions";
import type { DomainAdoption, Gap } from "@/labs/object-detection/data/types";
import { Card, SectionTitle, Badge, ConfidenceBadge, PaperLink } from "@/labs/object-detection/ui";

const GAP_GROUPS: { kind: Gap["kind"]; label: string; tone: "rose" | "sky" | "amber" }[] = [
  { kind: "academicOnly", label: "academic only", tone: "rose" },
  { kind: "industryOnly", label: "industry only", tone: "sky" },
  { kind: "lowAdoption", label: "low adoption", tone: "amber" },
];

function PaperToggle({ ids, short }: { ids: string[]; short?: boolean }) {
  const [open, setOpen] = useState(false);
  const shown = open ? ids : ids.slice(0, 8);
  const rest = ids.length - shown.length;
  return (
    <div className="mt-3 border-t border-zinc-800 pt-2">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          evidence papers
        </span>
        <button
          onClick={() => setOpen((v) => !v)}
          className="rounded-md border border-zinc-800 px-2 py-0.5 font-mono text-[10px] text-zinc-400 transition-colors hover:border-emerald-700/60 hover:text-emerald-300"
        >
          {open ? "collapse" : `show all ${ids.length}`}
        </button>
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {shown.map((id) => (
          <PaperLink key={id} id={id} short={short} />
        ))}
        {rest > 0 && <Badge tone="zinc">+{rest}</Badge>}
      </div>
    </div>
  );
}

function DomainCard({ d, max }: { d: DomainAdoption; max: number }) {
  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-[13px] font-medium text-zinc-100">{d.domain}</h3>
          <span className="font-mono text-[10px] text-zinc-500">first seen {d.firstYear}</span>
        </div>
        <ConfidenceBadge confidence={d.industryAdoption} />
      </div>
      <div className="mt-3 flex items-center gap-2">
        <span className="w-28 shrink-0 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          academic evidence
        </span>
        <div className="h-2.5 flex-1 overflow-hidden rounded-sm bg-zinc-800">
          <div
            className="h-full bg-emerald-500"
            style={{ width: `${(d.academicEvidence / max) * 100}%` }}
          />
        </div>
        <span className="w-8 shrink-0 text-right font-mono text-[11px] text-zinc-300">
          {d.academicEvidence}
        </span>
      </div>
      <p className="mt-3 text-[12px] leading-5 text-zinc-400">{d.note}</p>
      <PaperToggle ids={d.paperIds} short />
    </Card>
  );
}

export function IndustryView() {
  const domains = [...INDUSTRY.domains].sort((a, b) => b.academicEvidence - a.academicEvidence);
  const maxEvidence = Math.max(1, ...domains.map((d) => d.academicEvidence));

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">
          Academic ↔ industry
        </h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Where the academic literature and real deployment agree — and where they diverge.
        </p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          <Badge tone="zinc">{domains.length} domains</Badge>
          <Badge tone="zinc">{INDUSTRY.gaps.length} gaps</Badge>
          <Badge tone="zinc">{INDUSTRY.lowAdoption.length} low-adoption families</Badge>
        </div>
      </div>

      {/* domains */}
      <section>
        <SectionTitle>Application domains</SectionTitle>
        <p className="mt-1 text-[12px] leading-5 text-zinc-500">
          Sorted by academic evidence. Industry adoption is a review-pass judgment, not a
          count — the note carries the evidence.
        </p>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          {domains.map((d) => (
            <DomainCard key={d.domain} d={d} max={maxEvidence} />
          ))}
        </div>
      </section>

      {/* gaps */}
      <section>
        <SectionTitle>Where the two sides diverge</SectionTitle>
        <div className="mt-3 space-y-5">
          {GAP_GROUPS.map((g) => {
            const gaps = INDUSTRY.gaps.filter((x) => x.kind === g.kind);
            if (gaps.length === 0) return null;
            return (
              <div key={g.kind}>
                <div className="flex items-center gap-2">
                  <Badge tone={g.tone}>{g.label}</Badge>
                  <span className="font-mono text-[10px] text-zinc-600">
                    {gaps.length} gap{gaps.length > 1 ? "s" : ""}
                  </span>
                </div>
                <div className="mt-2 grid gap-3 md:grid-cols-2">
                  {gaps.map((gap) => (
                    <Card key={gap.id} tone={g.kind === "industryOnly" ? "ok" : g.kind === "lowAdoption" ? "warn" : "default"}>
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-[10px] text-zinc-600">{gap.id}</span>
                      </div>
                      <h3 className="mt-1 text-[13px] font-medium text-zinc-100">{gap.title}</h3>
                      <p className="mt-2 text-[12px] leading-5 text-zinc-400">{gap.detail}</p>
                      <div className="mt-3 flex flex-wrap gap-1.5 border-t border-zinc-800 pt-2">
                        {gap.paperIds.map((id) => (
                          <PaperLink key={id} id={id} short />
                        ))}
                      </div>
                    </Card>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* low adoption */}
      <section>
        <SectionTitle>Published but rarely shipped</SectionTitle>
        <p className="mt-1 text-[12px] leading-5 text-zinc-500">
          Solution families with many corpus papers and little deployment evidence.
        </p>
        <div className="mt-3 space-y-2">
          {INDUSTRY.lowAdoption.map((row) => {
            const solution = SOLUTION_BY_ID[row.solutionId];
            return (
              <Card key={row.solutionId}>
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-[13px] font-medium text-zinc-100">
                    {solution?.name ?? row.solutionId}
                  </h3>
                  {!solution && (
                    <span className="font-mono text-[10px] text-rose-400">?{row.solutionId}</span>
                  )}
                  <Badge tone="amber">{row.paperCount} papers</Badge>
                  {solution && <Badge tone="zinc">{solution.category}</Badge>}
                </div>
                <p className="mt-2 text-[12px] leading-5 text-zinc-400">{row.reason}</p>
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
}
