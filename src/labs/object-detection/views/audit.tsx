"use client";

import { useMemo } from "react";
import { runAudit, countPapersByYear } from "@/labs/object-detection/data/audit";
import { PAPERS } from "@/labs/object-detection/data/papers";
import { Card, Badge } from "@/labs/object-detection/ui";

export function AuditView() {
  const checks = useMemo(() => runAudit(), []);
  const byYear = useMemo(() => countPapersByYear(), []);
  const passed = checks.filter((c) => c.pass).length;

  const totalPapers = PAPERS.length;
  const evidenceCarrying = PAPERS.filter(
    (p) => (p.evidences ?? []).length > 0,
  ).length;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Research audit</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Machine checks over the entire lab: every paper processed, no duplicates,
          contiguous years, every reference resolves, and chronology is consistent.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card tone={passed === checks.length ? "ok" : "warn"}>
          <div className="font-mono text-3xl text-emerald-300">
            {passed}/{checks.length}
          </div>
          <div className="mt-1 text-xs text-zinc-500">audit checks passing</div>
        </Card>
        <Card>
          <div className="font-mono text-3xl text-emerald-300">{totalPapers}</div>
          <div className="mt-1 text-xs text-zinc-500">papers in manifest</div>
        </Card>
        <Card>
          <div className="font-mono text-3xl text-emerald-300">{evidenceCarrying}</div>
          <div className="mt-1 text-xs text-zinc-500">papers with evidence keys</div>
        </Card>
        <Card>
          <div className="font-mono text-3xl text-emerald-300">
            {Object.keys(byYear).length}
          </div>
          <div className="mt-1 text-xs text-zinc-500">years with at least one paper</div>
        </Card>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        {checks.map((c) => (
          <div
            key={c.id}
            className={`rounded-xl border p-4 ${
              c.pass ? "border-emerald-800/50 bg-emerald-950/10" : "border-rose-800/60 bg-rose-950/15"
            }`}
          >
            <div className="flex items-center gap-2">
              <span className={`font-mono text-sm ${c.pass ? "text-emerald-400" : "text-rose-400"}`}>
                {c.pass ? "✓" : "✗"}
              </span>
              <span className="text-sm font-medium text-zinc-200">{c.label}</span>
            </div>
            <p className="mt-1.5 font-mono text-[10px] leading-4 text-zinc-500">{c.detail}</p>
          </div>
        ))}
      </div>

      <Card>
        <div className="flex items-center gap-2">
          <Badge tone="emerald">policy</Badge>
          <p className="text-[11px] leading-5 text-zinc-400">
            Every assertion in this lab is either traced to a corpus paper, flagged as
            cross-paper or lab synthesis, or withheld as{" "}
            <span className="font-mono text-rose-300">INSUFFICIENT EVIDENCE</span>. Novel
            equations, years, and claims are never invented to fill gaps.
          </p>
        </div>
      </Card>
    </div>
  );
}