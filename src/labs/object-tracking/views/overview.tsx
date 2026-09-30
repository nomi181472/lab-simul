"use client";

import { PAPERS } from "@/labs/object-tracking/data/papers";
import { MANIFEST } from "@/labs/object-tracking/data/manifest";
import { SECTIONS } from "@/labs/object-tracking/data/types";
import { CONCEPTS } from "@/labs/object-tracking/data/concepts";
import { useTrackingLab } from "@/labs/object-tracking/context";
import { Card, Badge } from "@/labs/object-tracking/ui";

export function OverviewView() {
  const { setSection } = useTrackingLab();
  const covered = PAPERS.length;
  const total = MANIFEST.length;
  const years = [...new Set(PAPERS.map((p) => p.year))].sort();
  const byYear: Record<number, number> = {};
  for (const p of PAPERS) byYear[p.year] = (byYear[p.year] ?? 0) + 1;

  return (
    <div className="space-y-8">
      <section className="rounded-2xl border border-sky-800/40 bg-gradient-to-b from-sky-950/30 to-zinc-950 p-8">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="sky">evidence-grounded</Badge>
          <Badge tone="zinc">{covered}/{total} papers structured</Badge>
          <Badge tone="zinc">2014–2026</Badge>
          <Badge tone="emerald">detection-linked</Badge>
        </div>
        <h1 className="mt-4 max-w-2xl text-3xl font-semibold leading-tight tracking-tight text-zinc-50">
          Object Tracking Research Lab
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-zinc-400">
          Detection answers <span className="text-zinc-200">“what objects are present?”</span> —
          tracking additionally answers{" "}
          <span className="text-sky-300">“which object is the same object across frames?”</span>{" "}
          Every record (T001…T159) is grounded in its corpus PDF: no invented results,
          no invented relations, no invented equations. Papers appear here as a lineage —
          each one exists because of previous ideas and leaves a gap the next one attacks.
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          <button
            onClick={() => setSection("path")}
            className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-zinc-950 hover:bg-sky-500"
          >
            Start the learning path
          </button>
          <button
            onClick={() => setSection("papers")}
            className="rounded-lg border border-zinc-700 px-4 py-2 text-sm font-medium text-zinc-200 hover:bg-zinc-900"
          >
            Browse {covered} papers
          </button>
          <button
            onClick={() => setSection("timeline")}
            className="rounded-lg border border-zinc-700 px-4 py-2 text-sm font-medium text-zinc-200 hover:bg-zinc-900"
          >
            Research timeline
          </button>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <div className="font-mono text-3xl text-sky-300">{covered}</div>
          <div className="mt-1 text-xs text-zinc-500">papers structured of {total} in corpus</div>
        </Card>
        <Card>
          <div className="font-mono text-3xl text-sky-300">{CONCEPTS.length}</div>
          <div className="mt-1 text-xs text-zinc-500">concepts in the prerequisite DAG</div>
        </Card>
        <Card>
          <div className="font-mono text-3xl text-sky-300">11</div>
          <div className="mt-1 text-xs text-zinc-500">executable simulators in Math Lab</div>
        </Card>
        <Card>
          <div className="font-mono text-3xl text-sky-300">{years.length > 0 ? `${years[0]}–${years[years.length - 1]}` : "—"}</div>
          <div className="mt-1 text-xs text-zinc-500">years covered so far</div>
        </Card>
      </section>

      <section>
        <h2 className="mb-3 font-mono text-[11px] uppercase tracking-widest text-zinc-500">
          Detection → tracking bridge
        </h2>
        <Card tone="accent">
          <div className="flex flex-col items-stretch gap-2 font-mono text-[11px] md:flex-row md:items-center">
            {["detector per frame", "predict (Kalman / learned)", "cost matrix", "associate (Hungarian)", "birth / death", "track IDs"].map((s, i, a) => (
              <div key={s} className="flex flex-1 flex-col items-center gap-2">
                <div className="w-full rounded-lg border border-sky-800/50 bg-sky-950/30 px-3 py-2 text-center text-sky-200">{s}</div>
                {i < a.length - 1 && <span className="text-sky-600">→</span>}
              </div>
            ))}
          </div>
          <p className="mt-3 text-[12px] leading-5 text-zinc-400">
            Tracking-by-detection reuses the Object Detection lab&apos;s boxes as input. Open{" "}
            <a href="#object-detection" className="text-emerald-300 underline underline-offset-2">Object Detection</a>{" "}
            for detectors (Faster R-CNN, YOLO, DETR); this lab studies what happens{" "}
            <span className="text-zinc-200">between</span> frames.
          </p>
        </Card>
      </section>

      <section>
        <h2 className="mb-3 font-mono text-[11px] uppercase tracking-widest text-zinc-500">
          Corpus by year (structured so far)
        </h2>
        <div className="flex h-28 items-end gap-1 overflow-x-auto rounded-xl border border-zinc-900 bg-zinc-900/40 p-3">
          {Object.entries(byYear).sort((a, b) => Number(a[0]) - Number(b[0])).map(([y, n]) => (
            <div key={y} className="flex min-w-[28px] flex-1 flex-col items-center justify-end gap-1">
              <span className="font-mono text-[9px] text-zinc-500">{n}</span>
              <div className="w-full rounded-t bg-sky-800/70" style={{ height: `${Math.max(4, (n / Math.max(...Object.values(byYear))) * 80)}px` }} />
              <span className="font-mono text-[9px] text-zinc-600">{String(y).slice(2)}</span>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-3 font-mono text-[11px] uppercase tracking-widest text-zinc-500">Inside the lab</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {SECTIONS.filter((s) => s.id !== "overview").map((s) => (
            <button
              key={s.id}
              onClick={() => setSection(s.id)}
              className="group rounded-xl border border-zinc-800 bg-zinc-900/40 p-4 text-left transition-colors hover:border-sky-700/60 hover:bg-sky-950/20"
            >
              <div className="font-mono text-[10px] uppercase tracking-wide text-sky-400">{s.id}</div>
              <div className="mt-1 text-sm font-semibold text-zinc-100 group-hover:text-sky-200">{s.label}</div>
              <div className="mt-1 text-xs text-zinc-500">{s.blurb}</div>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
