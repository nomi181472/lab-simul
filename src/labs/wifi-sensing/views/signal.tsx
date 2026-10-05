"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/wifi-sensing/data/papers";
import { MANIFEST } from "@/labs/wifi-sensing/data/manifest";
import { useWifiLab } from "@/labs/wifi-sensing/context";
import { Card, Formula, SectionTitle, WifiPaperLink } from "@/labs/wifi-sensing/ui";

/* The signal chain is the spine of every paper in this field. This view reads
 * the chains straight out of the generated records, so what you see is what the
 * detector found in the papers — not a curated description of the pipeline. */

const STAGES = [
  {
    key: "hardware",
    label: "1 · radio",
    title: "Hardware & link",
    why: "What actually emits and receives. The NIC determines your bandwidth, your subcarrier count, and whether you get phase at all.",
    field: "hardware" as const,
  },
  {
    key: "captured",
    label: "2 · capture",
    title: "What is captured",
    why: "One scalar per packet (RSSI) or a complex value per subcarrier (CSI). Everything downstream is bounded by this choice.",
    field: "captured" as const,
  },
  {
    key: "preprocessing",
    label: "3 · preprocess",
    title: "Preprocessing",
    why: "Filtering, detrending, unwrapping, jitter handling, CFO correction. Where most silently-fatal bugs live.",
    field: "preprocessing" as const,
  },
  {
    key: "features",
    label: "4 · features",
    title: "Features",
    why: "What the model actually consumes. If you cannot name these, you cannot say why a result happened.",
    field: "features" as const,
  },
  {
    key: "model",
    label: "5 · model",
    title: "Model",
    why: "Where the accuracy number comes from — and, per this corpus, often less of it than the feature set suggests.",
    field: "model" as const,
  },
  {
    key: "deployment",
    label: "6 · deployment",
    title: "Deployment conditions",
    why: "Distance, room, obstruction, bandwidth. The conditions a result was obtained under are part of the result.",
    field: "deployment" as const,
  },
];

const NOT_DESCRIBED = "not described in the retrieved text";

function tally(values: string[], split: boolean) {
  const m = new Map<string, number>();
  for (const v of values) {
    const parts = split ? v.split(/,\s*/) : [v];
    for (const p of parts) {
      const k = p.trim();
      if (!k) continue;
      m.set(k, (m.get(k) ?? 0) + 1);
    }
  }
  return [...m.entries()].sort((a, b) => b[1] - a[1]);
}

export function SignalView() {
  const { setSection } = useWifiLab();
  const [task, setTask] = useState("all");

  const papers = useMemo(
    () => (task === "all" ? PAPERS : PAPERS.filter((p) => p.task === task)),
    [task],
  );
  const tasks = useMemo(() => [...new Set(MANIFEST.map((m) => m.task))].sort(), []);

  return (
    <div className="space-y-5">
      <SectionTitle>Signal chain</SectionTitle>
      <p className="max-w-3xl text-[12px] leading-6 text-zinc-500">
        Every WiFi sensing paper runs the same six stages. The tallies below are counted across
        {" "}
        {papers.length} generated records, so they show what the field actually does rather than
        what a textbook says it should. The right-hand panel follows one paper end to end.
      </p>

      <div className="flex flex-wrap gap-1.5">
        <button
          onClick={() => setTask("all")}
          className={`rounded-md border px-2.5 py-1 text-[11px] ${
            task === "all"
              ? "border-cyan-600 bg-cyan-600/20 text-cyan-200"
              : "border-zinc-800 text-zinc-500"
          }`}
        >
          all tasks
        </button>
        {tasks.map((t) => (
          <button
            key={t}
            onClick={() => setTask(t)}
            className={`rounded-md border px-2.5 py-1 text-[11px] ${
              task === t
                ? "border-cyan-600 bg-cyan-600/20 text-cyan-200"
                : "border-zinc-800 text-zinc-500"
            }`}
          >
            {t.replace(/-/g, " ")}
          </button>
        ))}
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        {STAGES.map((s) => {
          const isList = s.key === "preprocessing" || s.key === "features";
          const values = papers.map((p) => {
            const v = p.chain[s.field];
            return Array.isArray(v) ? v.join(", ") : v;
          });
          const t = tally(values, isList).slice(0, 8);
          const max = t[0]?.[1] ?? 1;
          const missing = values.filter((v) => v === NOT_DESCRIBED).length;
          return (
            <Card key={s.key}>
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] text-cyan-500">{s.label}</span>
                <span className="text-[12px] font-semibold text-zinc-100">{s.title}</span>
                {missing ? (
                  <span className="ml-auto font-mono text-[10px] text-zinc-600">
                    {missing} unstated
                  </span>
                ) : null}
              </div>
              <p className="mt-1 text-[11px] leading-5 text-zinc-500">{s.why}</p>
              <div className="mt-2 space-y-1">
                {t.map(([k, n]) => (
                  <div key={k} className="flex items-center gap-2">
                    <span className="w-44 shrink-0 truncate text-[10px] text-zinc-400" title={k}>
                      {k}
                    </span>
                    <div className="h-2 flex-1 overflow-hidden rounded bg-zinc-800">
                      <div
                        className="h-full bg-cyan-700/60"
                        style={{ width: `${(n / max) * 100}%` }}
                      />
                    </div>
                    <span className="w-7 text-right font-mono text-[10px] text-zinc-500">{n}</span>
                  </div>
                ))}
              </div>
            </Card>
          );
        })}
      </div>

      <Card tone="accent">
        <div className="font-mono text-[10px] uppercase tracking-wide text-cyan-400">
          the two equations every stage depends on
        </div>
        <div className="mt-2 grid gap-2 lg:grid-cols-2">
          <div>
            <Formula>CSI(fₖ) = Σᵢ αᵢ · e^(−j2πfₖτᵢ)</Formula>
            <p className="mt-1 text-[11px] leading-5 text-zinc-500">
              Stage 1–2. Every measured subcarrier is a coherent sum of delayed, attenuated
              echoes. Multipath is not noise here — it is the signal, and a person is one more
              term in the sum whose delay and amplitude move.
            </p>
          </div>
          <div>
            <Formula>Δφ = 2π·f_c·Δd/c  ⇒  f_d = v·cosθ·f_c / c</Formula>
            <p className="mt-1 text-[11px] leading-5 text-zinc-500">
              Stage 2–5. Motion changes path length, so it changes phase; across consecutive
              packets that change becomes a rate, and the rate is a velocity. This is the entire
              chain from a person walking to a Doppler bin.
            </p>
          </div>
        </div>
        <button
          onClick={() => setSection("math")}
          className="mt-3 rounded-md border border-cyan-800 px-2.5 py-1 font-mono text-[10px] text-cyan-300 hover:bg-cyan-950"
        >
          operate these in the Math Lab →
        </button>
      </Card>

      <Card>
        <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          the same chain, one paper at a time
        </div>
        <div className="mt-2 space-y-2">
          {[...papers]
            .sort((a, b) => b.citations - a.citations)
            .slice(0, 12)
            .map((p) => (
              <div key={p.id} className="rounded-lg border border-zinc-800 bg-black/20 p-2.5">
                <div className="flex items-center gap-2">
                  <WifiPaperLink id={p.id} link />
                  <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-400">{p.title}</span>
                  <span className="shrink-0 font-mono text-[10px] text-zinc-600">{p.year}</span>
                </div>
                <div className="mt-1.5 flex flex-wrap items-center gap-1 text-[10px]">
                  {STAGES.map((s, i) => {
                    const v = p.chain[s.field];
                    const text = Array.isArray(v) ? v.join(", ") : v;
                    const short = text === NOT_DESCRIBED ? "—" : text.slice(0, 42);
                    return (
                      <span key={s.key} className="flex items-center gap-1">
                        {i > 0 ? <span className="text-zinc-700">→</span> : null}
                        <span
                          title={text}
                          className={`rounded px-1.5 py-0.5 font-mono ${
                            text === NOT_DESCRIBED
                              ? "bg-zinc-900 text-zinc-600"
                              : "bg-zinc-800/80 text-zinc-300"
                          }`}
                        >
                          {short}
                        </span>
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
        </div>
      </Card>
    </div>
  );
}