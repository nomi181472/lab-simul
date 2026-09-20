"use client";

import { useMemo } from "react";
import type { Paper } from "@/labs/object-detection/data/types";
import { PAPERS } from "@/labs/object-detection/data/papers";
import { Card, SectionTitle, PaperLink, Badge } from "@/labs/object-detection/ui";

/* Built from the corpus's own research summaries and tags. No AP/latency numbers
 * are invented: numeric claims are withheld unless the corpus states a figure.
 * Each paper is positioned by its design point (tags) × release year (data). */

type Family = "two-stage" | "one-stage" | "transformer" | "other";

const FAMILY_ORDER: Family[] = ["two-stage", "one-stage", "transformer", "other"];
const FAMILY_COLOR: Record<Family, string> = {
  "two-stage": "#a78bfa",
  "one-stage": "#34d399",
  transformer: "#60a5fa",
  other: "#52525b",
};

function familyOf(p: Paper): Family {
  const t = p.tags.map((x) => x.toLowerCase()).join(" ");
  if (/(transformer|detr|vit|attention)/.test(t)) return "transformer";
  if (t.includes("two-stage")) return "two-stage";
  if (t.includes("one-stage")) return "one-stage";
  return "other";
}

const YEAR_MIN = 2015;
const YEAR_MAX = 2026;

export function BenchmarkView() {
  const { points, byFamily } = useMemo(() => {
    const points = PAPERS.map((p, i) => ({
      p,
      f: familyOf(p),
      x: FAMILY_ORDER.indexOf(familyOf(p)) + 0.12 * ((i * 37) % 10) + 0.2,
      y: p.year,
    }));
    const byFamily: Record<Family, number> = { "two-stage": 0, "one-stage": 0, transformer: 0, other: 0 };
    for (const pt of points) byFamily[pt.f]++;
    return { points, byFamily };
  }, []);

  const realtimeClaims = useMemo(
    () =>
      PAPERS.filter((p) =>
        p.tags.some((t) => /real-time|speed/.test(t.toLowerCase())),
      ).sort((a, b) => b.year - a.year),
    [],
  );

  const xMax = FAMILY_ORDER.length;
  const yRange = YEAR_MAX - YEAR_MIN;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Benchmark map</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Where the 48 papers landed: design point on the x-axis, release year on the
          y-axis. Positions come from the papers’ own tags; figures (mAP, ms, FPS) are
          only shown when the corpus states them. Otherwise numbers are withheld.
        </p>
      </div>

      <Card>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <SectionTitle>Design space, 2015–{YEAR_MAX}</SectionTitle>
          <div className="flex flex-wrap gap-2">
            {FAMILY_ORDER.map((f) => (
              <span key={f} className="inline-flex items-center gap-1 font-mono text-[10px] text-zinc-400">
                <span className="h-2 w-2 rounded-full" style={{ background: FAMILY_COLOR[f] }} />
                {f} · {byFamily[f]}
              </span>
            ))}
          </div>
        </div>

        <div className="mt-4 overflow-x-auto">
          <svg
            viewBox={`0 0 900 320`}
            className="w-full min-w-[560px] rounded-lg border border-zinc-800 bg-black/30"
          >
            {/* y gridlines (years) */}
            {Array.from({ length: yRange + 1 }, (_, i) => YEAR_MIN + i).map((y) => (
              <g key={y}>
                <line
                  x1={40}
                  x2={870}
                  y1={300 - ((y - YEAR_MIN) / yRange) * 260 - 10}
                  y2={300 - ((y - YEAR_MIN) / yRange) * 260 - 10}
                  stroke="#3f3f46"
                  strokeWidth={0.5}
                />
                <text
                  x={34}
                  y={300 - ((y - YEAR_MIN) / yRange) * 260 - 5}
                  fontSize={10}
                  fill="#71717a"
                  textAnchor="end"
                  fontFamily="monospace"
                >
                  {y}
                </text>
              </g>
            ))}
            {FAMILY_ORDER.map((f, xi) => (
              <text
                key={f}
                x={40 + ((xi + 0.5) / xMax) * 830}
                y={316}
                fontSize={12}
                fill={FAMILY_COLOR[f]}
                textAnchor="middle"
                fontFamily="monospace"
              >
                {f}
              </text>
            ))}
            {points.map((pt) => (
              <g key={pt.p.id}>
                <circle
                  cx={40 + (pt.x / xMax) * 830}
                  cy={300 - ((pt.y - YEAR_MIN) / yRange) * 260 - 10}
                  r={7}
                  fill={FAMILY_COLOR[pt.f]}
                  fillOpacity={0.75}
                  stroke="#18181b"
                  strokeWidth={1}
                >
                  <title>
                    {pt.p.id} · {pt.p.shortTitle} ({pt.p.year}) · {pt.p.authors[0]} et al.
                  </title>
                </circle>
              </g>
            ))}
          </svg>
        </div>
        <p className="mt-3 text-[11px] leading-5 text-zinc-500">
          The transformer column (2019→) contains the corpus’s DETR-family papers; the
          one-stage column spans YOLO-style and anchor-free real-time detectors. Hover a
          dot for the paper identity. “Other” = papers whose tags name neither family.
        </p>
      </Card>

      <Card>
        <SectionTitle>Speed / latency claims the corpus actually states</SectionTitle>
        <p className="mt-1 text-[12px] leading-5 text-zinc-500">
          Direct quotes or paraphrases from paper summaries — the closest our evidence
          policy allows to a benchmark table.
        </p>
        <div className="mt-3 grid gap-2 md:grid-cols-2">
          {realtimeClaims.map((p) => (
            <div key={p.id} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <PaperLink id={p.id} />
                <span className="font-mono text-[10px] text-zinc-500">{p.year}</span>
                <span className="ml-auto flex gap-1">
                  {p.tags.slice(0, 3).map((t) => (
                    <Badge key={t} tone="zinc">{t}</Badge>
                  ))}
                </span>
              </div>
              <p className="mt-2 text-[12px] leading-5 text-zinc-400">{p.summary}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}