"use client";

import { SIMULATOR_IDS, SIMULATOR_META, SIMULATOR_GROUP, SimShell } from "@/labs/wifi-sensing/sims";
import { checkSim } from "@/labs/wifi-sensing/sims/papers";
import { SimErrorBoundary } from "@/labs/wifi-sensing/sims/error-boundary";
import { useWifiLab } from "@/labs/wifi-sensing/context";
import { Badge, Card, SectionTitle } from "@/labs/wifi-sensing/ui";

export function MathView() {
  const { mathSim, setMathSim } = useWifiLab();
  const sim = checkSim(mathSim) ?? SIMULATOR_IDS[0];
  const meta = SIMULATOR_META.find((m) => m.id === sim);
  const groups = Object.entries(
    SIMULATOR_META.reduce<Record<string, typeof SIMULATOR_META>>((acc, m) => {
      const g = SIMULATOR_GROUP[m.id];
      (acc[g] ??= []).push(m);
      return acc;
    }, {}),
  );

  return (
    <div className="space-y-5">
      <SectionTitle>Math Lab — {SIMULATOR_IDS.length} simulators</SectionTitle>
      <p className="max-w-3xl text-[12px] leading-6 text-zinc-500">
        Every simulator is a deterministic kernel plus a set of range inputs — no simulation is
        pre-recorded. Each one names the research problem it represents and links to the papers
        that instantiate it. Move a slider until the phenomenon breaks, because the breakage is
        the point: that is the constraint a real paper is negotiating.
      </p>

      <div className="space-y-3">
        {groups.map(([group, items]) => (
          <div key={group}>
            <div className="mb-1.5 font-mono text-[9px] uppercase tracking-wide text-zinc-500">
              {group}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {items.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setMathSim(m.id)}
                  title={m.blurb}
                  className={`rounded-lg border px-2.5 py-1.5 text-left transition-colors ${
                    sim === m.id
                      ? "border-cyan-600 bg-cyan-600/20"
                      : "border-zinc-800 bg-zinc-900/40 hover:border-zinc-700"
                  }`}
                >
                  <div
                    className={`text-[11px] font-medium ${
                      sim === m.id ? "text-cyan-200" : "text-zinc-300"
                    }`}
                  >
                    {m.label}
                  </div>
                  <div className="mt-0.5 max-w-52 font-mono text-[9px] leading-3 text-zinc-600">
                    {m.blurb}
                  </div>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {meta ? (
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="cyan">{meta.label}</Badge>
          <Badge tone="zinc">{SIMULATOR_GROUP[meta.id]}</Badge>
        </div>
      ) : null}

      <SimErrorBoundary key={sim}>
        <SimShell id={sim} />
      </SimErrorBoundary>

      <Card tone="accent">
        <div className="font-mono text-[10px] uppercase tracking-wide text-cyan-400">
          reading a simulator critically
        </div>
        <p className="mt-1 text-[11px] leading-5 text-zinc-400">
          The formulas here are the standard textbook versions, and the parameter ranges are chosen
          to make each phenomenon visible rather than to match any one paper&apos;s setup. Treat the
          &ldquo;real research problem&rdquo; line as the claim under discussion and the linked
          papers as the evidence that the problem is real. Where a simulator oversimplifies, the
          oversimplification is usually the thing worth arguing about.
        </p>
      </Card>
    </div>
  );
}