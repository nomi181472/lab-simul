"use client";
import { useState } from "react";
import { Card, SectionTitle, Badge, Quote, PaperLink } from "@/labs/networking/ui";
import { ANALYSIS } from "@/labs/networking/data/analysis";

export function FutureView() {
  const [tab, setTab] = useState<"all" | "realized" | "unresolved">("all");
  const f = ANALYSIS.future;
  const rows = tab === "all" ? f.all : tab === "realized" ? f.realized : f.unresolved;

  return (
    <div className="space-y-6">
      <SectionTitle>Future direction tracker</SectionTitle>

      <div className="grid gap-3 sm:grid-cols-3">
        <Card tone="accent">
          <p className="font-mono text-[10px] uppercase tracking-wide text-sky-400">
            directions stated
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {f.counts.realized + f.counts.unresolved}
          </p>
        </Card>
        <Card tone="ok">
          <p className="font-mono text-[10px] uppercase tracking-wide text-emerald-400">
            realized by later papers
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{f.counts.realized}</p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            later paper shares domain + uses the matching solution family
          </p>
        </Card>
        <Card tone="warn">
          <p className="font-mono text-[10px] uppercase tracking-wide text-amber-400">
            still unresolved
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{f.counts.unresolved}</p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            no matching realization found in corpus
          </p>
        </Card>
      </div>

      <Card>
        <div className="flex gap-1">
          {(["all", "realized", "unresolved"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${
                tab === t
                  ? "border-sky-700/50 bg-sky-600/20 text-sky-300"
                  : "border-zinc-700 bg-zinc-900 text-zinc-400"
              }`}
            >
              {t} ({t === "all" ? f.all.length : t === "realized" ? f.realized.length : f.unresolved.length})
            </button>
          ))}
        </div>
        <p className="mt-2 text-[12px] leading-6 text-zinc-500">
          Matching rule: the origin paper&apos;s future-work sentence is pattern-matched to
          solution families (e.g. “energy-efficient” → RL / optimisation); a direction is{" "}
          <span className="text-emerald-400">realized</span> when a later paper in the same
          domain uses such a family. This is a corpus-internal check, not a claim about the
          whole literature.
        </p>
      </Card>

      <div className="space-y-3">
        {rows.map((t) => (
          <Card key={t.paperId}>
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div className="min-w-0">
                <PaperLink id={t.paperId} />
                <p className="mt-0.5 font-mono text-[10px] text-zinc-500">
                  stated {t.year} · {t.domains.join(", ")}
                </p>
              </div>
              <Badge tone={t.status === "realized" ? "emerald" : "amber"}>{t.status}</Badge>
            </div>
            <div className="mt-2">
              <Quote>{t.statement}</Quote>
            </div>
            <div className="mt-2 space-y-1">
              {t.matches.map((m, i) => (
                <div key={i} className="flex flex-wrap items-center gap-2">
                  <Badge tone={m.status === "realized" ? "emerald" : "zinc"}>
                    {m.status}
                  </Badge>
                  <span className="font-mono text-[10px] text-zinc-500">
                    → {m.wantedFamilies.join(" / ")}
                  </span>
                  {m.realizedBy.length > 0 && (
                    <span className="flex flex-wrap gap-1">
                      {m.realizedBy.map((id) => (
                        <PaperLink key={id} id={id} short />
                      ))}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </Card>
        ))}
        {rows.length === 0 && (
          <Card>
            <p className="text-[12px] text-zinc-500">no entries in this bucket</p>
          </Card>
        )}
      </div>
    </div>
  );
}
