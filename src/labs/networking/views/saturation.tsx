"use client";
import { Card, SectionTitle, Badge, Quote, BarRow } from "@/labs/networking/ui";
import { ANALYSIS } from "@/labs/networking/data/analysis";

export function SaturationView() {
  const s = ANALYSIS.saturation;
  const maxAtt = Math.max(1, ...s.rows.map((r) => r.attention));

  return (
    <div className="space-y-6">
      <SectionTitle>Research saturation &amp; unsolved problems</SectionTitle>

      <div className="grid gap-3 sm:grid-cols-3">
        <Card tone="warn">
          <p className="font-mono text-[10px] uppercase tracking-wide text-amber-400">
            saturated
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {s.summary.saturated}
          </p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            high attention, recurring early problems, low late progress
          </p>
        </Card>
        <Card tone="ok">
          <p className="font-mono text-[10px] uppercase tracking-wide text-emerald-400">
            active
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{s.summary.active}</p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">≥8 papers, progressing</p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            sparse
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{s.summary.sparse}</p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            too few papers to judge either way
          </p>
        </Card>
      </div>

      <Card>
        <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          attention (papers) by domain
        </p>
        <div className="space-y-1">
          {s.rows.map((r) => (
            <BarRow
              key={r.domainId}
              label={r.label}
              value={r.attention}
              max={maxAtt}
              tone={
                r.verdict === "saturated" ? "rose" : r.verdict === "active" ? "emerald" : "cyan"
              }
            />
          ))}
        </div>
      </Card>

      <div className="space-y-3">
        {s.rows.map((r) => (
          <Card key={r.domainId}>
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="text-[13px] font-medium text-zinc-100">{r.label}</span>
              <div className="flex items-center gap-2">
                <Badge
                  tone={
                    r.verdict === "saturated"
                      ? "rose"
                      : r.verdict === "active"
                        ? "emerald"
                        : "zinc"
                  }
                >
                  {r.verdict}
                </Badge>
                <span className="font-mono text-[10px] text-zinc-500">
                  early {r.earlyCount} → late {r.lateCount} · progress{" "}
                  {(r.progressRatio * 100).toFixed(0)}% · score {r.saturationScore}
                </span>
              </div>
            </div>
            {r.recurringProblem && (
              <p className="mt-1 font-mono text-[10px] text-amber-400/80">
                early problem statements recur in 2019+ papers — repeated attempts, not
                resolution
              </p>
            )}
            <div className="mt-2">
              <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                remaining gap (quoted)
              </p>
              <Quote>{r.remainingGap}</Quote>
            </div>
          </Card>
        ))}
      </div>

      <Card tone="accent">
        <p className="text-[12px] leading-6 text-zinc-400">
          <span className="font-semibold text-sky-300">Method:</span> attention = papers in
          domain; progress ratio = share of 2019+ papers with a quoted experiments
          section; recurring = an early (≤2015) problem quote and a late (≥2019) one both
          exist. saturationScore = attention/40 × recurrence − progress. These are corpus
          signals, deliberately crude, and the quote under each row is the evidence to
          check them against.
        </p>
      </Card>
    </div>
  );
}
