"use client";

import { METRICS } from "@/labs/object-tracking/data/metrics";
import { Card, Badge, Formula } from "@/labs/object-tracking/ui";
import { SimShell } from "@/labs/object-tracking/sims";
import { useTrackingLab } from "@/labs/object-tracking/context";

const FAMILY_TONE: Record<string, "sky" | "emerald" | "amber" | "violet" | "zinc"> = {
  detection: "sky",
  association: "amber",
  localization: "emerald",
  robustness: "violet",
  efficiency: "zinc",
};

export function MetricsView() {
  const { setMathSim } = useTrackingLab();
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Metrics</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Each metric: formula → meaning → intuition → worked example → limitations.
          MOTA rewards detectors; IDF1 rewards identity; HOTA forces both.
        </p>
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        {METRICS.map((m) => (
          <Card key={m.id}>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-semibold text-zinc-100">{m.name}</span>
              <Badge tone={FAMILY_TONE[m.family] ?? "zinc"}>{m.family}</Badge>
              {m.simulator && (
                <button onClick={() => m.simulator && setMathSim(m.simulator)} className="rounded border border-sky-800 px-1.5 py-0.5 font-mono text-[10px] text-sky-300 hover:bg-sky-950/40">simulate →</button>
              )}
            </div>
            <div className="mt-2"><Formula>{m.formula}</Formula></div>
            {m.variables && (
              <div className="mt-1.5 rounded bg-black/30 p-2 font-mono text-[10px] leading-4 text-zinc-500">
                {m.variables.map((v) => <div key={v.symbol}><span className="text-amber-300">{v.symbol}</span> — {v.meaning}</div>)}
              </div>
            )}
            <p className="mt-1.5 text-[12px] leading-5 text-zinc-300">{m.meaning}</p>
            <p className="text-[12px] italic leading-5 text-zinc-400">{m.intuition}</p>
            <p className="mt-1 font-mono text-[11px] text-sky-200/80">e.g. {m.example}</p>
            <ul className="mt-1.5 list-disc space-y-0.5 pl-4 text-[11px] text-zinc-500">
              {m.limitations.map((l, i) => <li key={i}>{l}</li>)}
            </ul>
          </Card>
        ))}
      </div>
      <Card tone="accent">
        <div className="font-mono text-[10px] uppercase tracking-widest text-sky-300">try the error budget</div>
        <div className="mt-2"><SimShell id="metrics" /></div>
      </Card>
    </div>
  );
}
