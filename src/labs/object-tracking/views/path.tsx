"use client";

import { CONCEPTS, CONCEPT_BY_ID } from "@/labs/object-tracking/data/concepts";
import { Card, Badge } from "@/labs/object-tracking/ui";
import { useTrackingLab } from "@/labs/object-tracking/context";

/* Beginner → advanced: topological walk over the concept DAG. */

const STAGES: { title: string; blurb: string; concepts: string[] }[] = [
  {
    title: "1 · See the problem",
    blurb: "What a box is, what overlap means, and why video adds identity.",
    concepts: ["bounding-box", "iou", "detection", "sot", "mot", "online-vs-offline"],
  },
  {
    title: "2 · The tracking loop",
    blurb: "Predict where boxes go, match them, manage their lives.",
    concepts: ["tracking-by-detection", "motion-model", "state-space", "kalman", "data-association", "cost-matrix", "hungarian", "track-management", "occlusion"],
  },
  {
    title: "3 · Appearance & identity",
    blurb: "What survives when motion fails: looks, memory, Re-ID.",
    concepts: ["appearance-features", "reid", "learned-association", "memory-network", "deep-detection"],
  },
  {
    title: "4 · Architectures through time",
    blurb: "CF → Siamese → discriminative → transformer → diffusion.",
    concepts: ["correlation-filter", "siamese", "rpn-head", "anchor-free-head", "attention", "end-to-end-mot", "diffusion-tracking", "dual-threshold", "greedy-matching"],
  },
  {
    title: "5 · Measure & redirect",
    blurb: "Metrics and benchmarks decide what the field optimizes next.",
    concepts: ["mota", "idf1", "hota", "success-plot", "benchmark-design"],
  },
  {
    title: "6 · Frontiers",
    blurb: "Where current papers live: modalities, 3D, unification.",
    concepts: ["multimodal", "3d-tracking"],
  },
];

export function PathView() {
  const { setSection, setMathSim } = useTrackingLab();
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Learning path</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Follow the stages in order. Each concept lists its prerequisites; anything with a
          simulator can be <span className="text-sky-300">operated, not just read</span>.
        </p>
      </div>
      {STAGES.map((st, i) => (
        <section key={st.title} className="relative">
          <div className="mb-2 flex items-baseline gap-3">
            <span className="font-mono text-[11px] text-sky-400">{st.title}</span>
            <span className="text-xs text-zinc-500">{st.blurb}</span>
          </div>
          <div className="grid gap-2 md:grid-cols-2">
            {st.concepts.map((cid) => {
              const c = CONCEPT_BY_ID[cid];
              if (!c) return null;
              return (
                <Card key={cid} className="!py-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold text-zinc-100">{c.name}</span>
                    <Badge tone="zinc">{c.category}</Badge>
                    {c.simulator && <Badge tone="sky">sim</Badge>}
                  </div>
                  <p className="mt-1 text-[12px] leading-5 text-zinc-400">{c.intuition}</p>
                  {c.prereqs.length > 0 && (
                    <p className="mt-1 font-mono text-[10px] text-zinc-600">
                      needs: {c.prereqs.join(" · ")}
                    </p>
                  )}
                  <div className="mt-2 flex gap-2">
                    {c.simulator && (
                      <button
                        onClick={() => setMathSim(c.simulator!)}
                        className="rounded-md border border-sky-800 px-2 py-1 font-mono text-[10px] text-sky-300 hover:bg-sky-950/40"
                      >
                        open simulator →
                      </button>
                    )}
                  </div>
                </Card>
              );
            })}
          </div>
          {i < STAGES.length - 1 && (
            <div className="py-1 text-center font-mono text-sky-700">↓</div>
          )}
        </section>
      ))}
      <Card tone="accent">
        <p className="text-[12px] text-zinc-400">
          Done with the path? Continue in <button className="text-sky-300 underline" onClick={() => setSection("timeline")}>Research Timeline</button> to
          see the same ideas appear as dated papers, or jump to{" "}
          <button className="text-sky-300 underline" onClick={() => setSection("papers")}>Paper Explorer</button> — {CONCEPTS.length} concepts
          indexed across the corpus.
        </p>
      </Card>
    </div>
  );
}
