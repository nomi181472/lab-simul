"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/wifi-sensing/data/papers";
import { Card, CiteBadge, SectionTitle, TaskBadge, WifiPaperLink } from "@/labs/wifi-sensing/ui";

type Mode = "year" | "cites" | "lineage";

export function TimelineView() {
  const [mode, setMode] = useState<Mode>("year");

  const years = useMemo(() => {
    const m = new Map<number, typeof PAPERS>();
    for (const p of PAPERS) {
      if (!p.year) continue;
      const arr = m.get(p.year) ?? [];
      arr.push(p);
      m.set(p.year, arr);
    }
    return [...m.entries()].sort((a, b) => a[0] - b[0]);
  }, []);

  const maxYearCount = Math.max(...years.map(([, ps]) => ps.length), 1);

  const eras: { from: number; to: number; title: string; blurb: string }[] = [
    {
      from: 2009,
      to: 2014,
      title: "Passive detection with commodity hardware",
      blurb:
        "The first demonstrations: someone is present because a link's signal changed. Inference is close to a threshold, because the available measurement is one number per packet.",
    },
    {
      from: 2015,
      to: 2018,
      title: "Per-subcarrier phase arrives",
      blurb:
        "802.11n CSI toolkits make complex per-subcarrier response available. Fine-grained sensing becomes possible and the field's real problems — unwrapping, bandwidth, phase noise — become its real subject.",
    },
    {
      from: 2019,
      to: 2021,
      title: "Deep learning and hand-crafted feature baselines",
      blurb:
        "Convolutional and recurrent models take over the accuracy tables. The counter-movement is equally important: training-free baselines on a handful of statistics stay competitive, which keeps asking whether the features or the model were doing the work.",
    },
    {
      from: 2022,
      to: 2026,
      title: "Scale, cross-domain deployment, and non-WiFi modalities",
      blurb:
        "Benchmarks and toolkits mature, cross-device and cross-room transfer becomes the headline problem, and FMCW radar plus hybrid sensing enter the corpus as alternatives to channel feedback.",
    },
  ];

  if (mode === "lineage") {
    return (
      <div className="space-y-4">
        <SectionTitle>Lineage — most-cited papers in the corpus</SectionTitle>
        <ModeSwitch mode={mode} setMode={setMode} />
        <Card>
          <p className="text-[12px] leading-6 text-zinc-400">
            Papers ordered by citation count within the corpus. A high count here means the ideas
            were reused, not that the method is currently the state of the art.
          </p>
          <div className="mt-3 space-y-1">
            {topByCitations(25).map((p, i) => (
              <div key={p.id} className="flex items-center gap-2">
                <span className="w-5 text-right font-mono text-[10px] text-zinc-600">{i + 1}</span>
                <WifiPaperLink id={p.id} link />
                <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-400">{p.title}</span>
                <span className="shrink-0 font-mono text-[10px] text-zinc-500">{p.year}</span>
                <CiteBadge n={p.citations} />
              </div>
            ))}
          </div>
        </Card>
      </div>
    );
  }

  if (mode === "cites") {
    const sorted = [...PAPERS].sort((a, b) => b.citations - a.citations);
    const show = sorted.slice(0, 30);
    const max = show[0]?.citations || 1;
    return (
      <div className="space-y-4">
        <SectionTitle>Citation distribution</SectionTitle>
        <ModeSwitch mode={mode} setMode={setMode} />
        <Card>
          <p className="text-[12px] leading-6 text-zinc-400">
            Crossref <span className="font-mono">is-referenced-by-count</span> at harvest time.
            The distribution is heavily long-tailed: a handful of survey and landmark systems carry
            most of the field&apos;s references, so &ldquo;citation rank&rdquo; and &ldquo;current
            best method&rdquo; are not the same ordering.
          </p>
          <div className="mt-3 space-y-1">
            {show.map((p) => (
              <div key={p.id} className="flex items-center gap-2">
                <span className="w-12 shrink-0 font-mono text-[10px] text-cyan-400">{p.id}</span>
                <div className="h-2 flex-1 overflow-hidden rounded bg-zinc-800">
                  <div
                    className="h-full bg-cyan-600/70"
                    style={{ width: `${(p.citations / max) * 100}%` }}
                  />
                </div>
                <span className="w-10 text-right font-mono text-[10px] text-zinc-400">
                  {p.citations}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <SectionTitle>Timeline</SectionTitle>
      <ModeSwitch mode={mode} setMode={setMode} />

      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
        {eras.map((e) => (
          <Card key={e.from} tone={e.from === 2009 ? "accent" : "default"}>
            <div className="font-mono text-[10px] text-cyan-400">
              {e.from}–{e.to}
            </div>
            <div className="mt-1 text-[12px] font-semibold leading-5 text-zinc-100">{e.title}</div>
            <p className="mt-1.5 text-[11px] leading-5 text-zinc-500">{e.blurb}</p>
            <div className="mt-2 font-mono text-[10px] text-zinc-600">
              {PAPERS.filter((p) => p.year >= e.from && p.year <= e.to).length} corpus papers
            </div>
          </Card>
        ))}
      </div>

      <Card>
        <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          papers per year
        </div>
        <div className="mt-3 space-y-1">
          {years.map(([y, ps]) => (
            <div key={y} className="flex items-center gap-2">
              <span className="w-9 shrink-0 text-right font-mono text-[10px] text-zinc-500">{y}</span>
              <div className="h-3 flex-1 overflow-hidden rounded bg-zinc-800/60">
                <div
                  className="h-full bg-cyan-700/60"
                  style={{ width: `${(ps.length / maxYearCount) * 100}%` }}
                />
              </div>
              <span className="w-6 text-right font-mono text-[10px] text-zinc-500">
                {ps.length}
              </span>
              <span className="hidden w-40 truncate text-[10px] text-zinc-600 sm:inline">
                {[...new Set(ps.map((p) => p.task))]
                  .slice(0, 2)
                  .join(", ")}
              </span>
            </div>
          ))}
        </div>
      </Card>

      <div className="space-y-2">
        {years
          .slice()
          .reverse()
          .map(([y, ps]) => (
            <Card key={y}>
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-cyan-400">{y}</span>
                <span className="font-mono text-[10px] text-zinc-600">
                  {ps.length} paper{ps.length === 1 ? "" : "s"}
                </span>
              </div>
              <div className="mt-1.5 space-y-1">
                {ps
                  .slice()
                  .sort((a, b) => b.citations - a.citations)
                  .slice(0, 5)
                  .map((p) => (
                    <div key={p.id} className="flex items-center gap-2">
                      <WifiPaperLink id={p.id} short link />
                      <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-400">
                        {p.title}
                      </span>
                      <TaskBadge t={p.task.replace(/-/g, " ")} />
                      <span className="shrink-0 font-mono text-[10px] text-zinc-600">
                        {p.citations}c
                      </span>
                    </div>
                  ))}
              </div>
            </Card>
          ))}
      </div>
    </div>
  );
}

function topByCitations(n: number) {
  return [...PAPERS].sort((a, b) => b.citations - a.citations).slice(0, n);
}

function ModeSwitch({ mode, setMode }: { mode: Mode; setMode: (m: Mode) => void }) {
  const opts: { value: Mode; label: string }[] = [
    { value: "year", label: "by year" },
    { value: "cites", label: "by citations" },
    { value: "lineage", label: "most cited" },
  ];
  return (
    <div className="inline-flex gap-1">
      {opts.map((o) => (
        <button
          key={o.value}
          onClick={() => setMode(o.value)}
          className={`rounded-md border px-2.5 py-1 text-[11px] transition-colors ${
            mode === o.value
              ? "border-cyan-600 bg-cyan-600/20 text-cyan-200"
              : "border-zinc-800 text-zinc-500 hover:bg-zinc-900"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
