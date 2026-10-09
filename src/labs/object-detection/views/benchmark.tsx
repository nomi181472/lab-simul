"use client";

import { useMemo, useState } from "react";
import { BENCHMARKS, BENCHMARK_BEST, BENCHMARK_DATASETS } from "@/labs/object-detection/data/benchmarks";
import { PAPER_BY_ID } from "@/labs/object-detection/data/papers";
import type { BenchmarkObservation } from "@/labs/object-detection/data/types";
import {
  Card,
  SectionTitle,
  Badge,
  PaperLink,
  InsufficientEvidence,
} from "@/labs/object-detection/ui";

const PAGE_SIZE = 15;
const CAP = 40;
const DATASET_VISIBLE = 12;
const METRIC_VISIBLE = 14;

const W = 900;
const H = 300;
const L = 48;
const R = 12;
const T = 18;
const B = 38;
const PLOT_W = W - L - R;
const PLOT_H = H - T - B;

function chip(active: boolean) {
  return `rounded-md border px-2 py-1 font-mono text-[10px] transition-colors ${
    active
      ? "border-emerald-600 bg-emerald-600/20 text-emerald-200"
      : "border-zinc-800 text-zinc-500 hover:text-zinc-300"
  }`;
}

function countBy(rows: BenchmarkObservation[], key: (o: BenchmarkObservation) => string) {
  const m = new Map<string, number>();
  for (const o of rows) m.set(key(o), (m.get(key(o)) ?? 0) + 1);
  return m;
}

function topMetricFor(dataset: string): string {
  const m = countBy(
    BENCHMARKS.filter((o) => o.dataset === dataset),
    (o) => o.metric,
  );
  let best = "";
  let bestN = -1;
  for (const [k, n] of m) {
    if (n > bestN || (n === bestN && k < best)) {
      best = k;
      bestN = n;
    }
  }
  return best;
}

function ChipRow({
  label,
  items,
  active,
  onPick,
  visible,
}: {
  label: string;
  items: { id: string; count: number }[];
  active: string;
  onPick: (id: string) => void;
  visible: number;
}) {
  const [open, setOpen] = useState(false);
  const shown = open ? items : items.slice(0, visible);
  const rest = items.length - shown.length;
  return (
    <div>
      <div className="flex items-center gap-2">
        <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          {label}
        </span>
        <span className="font-mono text-[10px] text-zinc-600">{items.length}</span>
        {rest > 0 && (
          <button
            onClick={() => setOpen(true)}
            className="rounded-md border border-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500 hover:text-zinc-300"
          >
            +{rest} more
          </button>
        )}
        {open && items.length > visible && (
          <button
            onClick={() => setOpen(false)}
            className="rounded-md border border-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500 hover:text-zinc-300"
          >
            collapse
          </button>
        )}
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {shown.map((it) => (
          <button
            key={it.id}
            onClick={() => onPick(it.id)}
            className={`${chip(active === it.id)} max-w-56 truncate`}
            title={it.id}
          >
            {it.id} · {it.count}
          </button>
        ))}
      </div>
    </div>
  );
}

function Plot({ rows, capped }: { rows: BenchmarkObservation[]; capped: boolean }) {
  const years = Array.from(new Set(rows.map((r) => r.year))).sort((a, b) => a - b);
  const maxV = Math.max(...rows.map((r) => r.value), 0.001);
  const yMax = maxV * 1.05;
  const colW = PLOT_W / Math.max(years.length, 1);
  const yOf = (v: number) => T + PLOT_H - (v / yMax) * PLOT_H;
  const ticks = [0, 1, 2, 3, 4].map((i) => (yMax * i) / 4);

  return (
    <div className="overflow-x-auto">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full min-w-[640px] rounded-lg border border-zinc-800 bg-black/30"
      >
        {ticks.map((v, i) => (
          <g key={i}>
            <line
              x1={L}
              x2={L + PLOT_W}
              y1={yOf(v)}
              y2={yOf(v)}
              stroke="#3f3f46"
              strokeWidth={i === 0 ? 1 : 0.5}
            />
            <text
              x={L - 6}
              y={yOf(v) + 3}
              fontSize={9}
              fill="#71717a"
              textAnchor="end"
              fontFamily="monospace"
            >
              {v >= 10 ? v.toFixed(0) : v.toFixed(1)}
            </text>
          </g>
        ))}

        {years.map((y, xi) => {
          const group = rows.filter((r) => r.year === y);
          const bw = Math.min(30, (colW * 0.72) / Math.max(group.length, 1));
          const total = bw * group.length;
          const x0 = L + (xi + 0.5) * colW - total / 2;
          return (
            <g key={y}>
              <text
                x={L + (xi + 0.5) * colW}
                y={H - 14}
                fontSize={9}
                fill="#71717a"
                textAnchor="middle"
                fontFamily="monospace"
              >
                {y}
              </text>
              {group.map((o, i) => {
                const bwFull = bw * 0.86;
                const bx = x0 + i * bw + bw * 0.07;
                const by = yOf(o.value);
                const bh = T + PLOT_H - by;
                const paper = PAPER_BY_ID[o.paperId];
                return (
                  <g key={`${o.paperId}-${i}`}>
                    <rect
                      x={bx}
                      y={by}
                      width={bwFull}
                      height={Math.max(bh, 1)}
                      rx={1.5}
                      fill={capped ? "#34d399" : "#10b981"}
                      fillOpacity={0.8}
                    >
                      <title>
                        {o.paperId}
                        {paper ? ` · ${paper.shortTitle}` : ""} · {o.raw} · {o.year}
                        {o.split ? ` · ${o.split}` : ""}
                      </title>
                    </rect>
                    {bwFull >= 15 && (
                      <text
                        x={bx + bwFull / 2}
                        y={by - 3}
                        fontSize={8}
                        fill="#a1a1aa"
                        textAnchor="middle"
                        fontFamily="monospace"
                      >
                        {o.raw}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function BenchmarkView() {
  const [dataset, setDataset] = useState("COCO");
  const [metric, setMetric] = useState(() => topMetricFor("COCO"));
  const [page, setPage] = useState(0);

  const datasetCounts = useMemo(() => countBy(BENCHMARKS, (o) => o.dataset), []);
  const datasets = useMemo(
    () =>
      [...BENCHMARK_DATASETS]
        .filter((d) => datasetCounts.has(d))
        .sort((a, b) => (datasetCounts.get(b) ?? 0) - (datasetCounts.get(a) ?? 0) || a.localeCompare(b)),
    [datasetCounts],
  );

  const metrics = useMemo(() => {
    const m = countBy(
      BENCHMARKS.filter((o) => o.dataset === dataset),
      (o) => o.metric,
    );
    return [...m.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([k, n]) => ({ id: k, count: n }));
  }, [dataset]);

  const rows = useMemo(
    () =>
      BENCHMARKS.filter((o) => o.dataset === dataset && o.metric === metric).sort(
        (a, b) => a.year - b.year || b.value - a.value,
      ),
    [dataset, metric],
  );

  const capped = rows.length > CAP;
  const plotRows = useMemo(() => {
    if (!capped) return rows;
    const byYear = new Map<number, BenchmarkObservation>();
    for (const o of rows) {
      const cur = byYear.get(o.year);
      if (!cur || o.value > cur.value) byYear.set(o.year, o);
    }
    return [...byYear.values()].sort((a, b) => a.year - b.year);
  }, [rows, capped]);

  const best = useMemo(
    () => BENCHMARK_BEST.find((b) => b.dataset === dataset && b.metric === metric),
    [dataset, metric],
  );

  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const cur = Math.min(page, pageCount - 1);
  const slice = rows.slice(cur * PAGE_SIZE, cur * PAGE_SIZE + PAGE_SIZE);
  const valid = metric !== "" && rows.length > 0;

  const pickDataset = (d: string) => {
    setDataset(d);
    setMetric(topMetricFor(d));
    setPage(0);
  };
  const pickMetric = (m: string) => {
    setMetric(m);
    setPage(0);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">
          Benchmark maps — dataset × metric × year
        </h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          {BENCHMARKS.length} reported observations from the corpus. Pick a dataset and a
          metric to see every figure the papers state, with the source table location.
        </p>
        <p className="mt-2 max-w-3xl font-mono text-[10px] leading-4 text-zinc-500">
          UNITS: values are raw as reported by papers — no normalisation across papers or
          evaluation protocols.
        </p>
      </div>

      {/* controls */}
      <Card>
        <div className="space-y-4">
          <ChipRow
            label="dataset"
            items={datasets.map((d) => ({ id: d, count: datasetCounts.get(d) ?? 0 }))}
            active={dataset}
            onPick={pickDataset}
            visible={DATASET_VISIBLE}
          />
          <ChipRow
            label="metric"
            items={metrics}
            active={metric}
            onPick={pickMetric}
            visible={METRIC_VISIBLE}
          />
        </div>
      </Card>

      {!valid ? (
        <InsufficientEvidence
          what={
            metric === ""
              ? `any reported ${dataset} metric`
              : `observations for ${dataset} × ${metric}`
          }
        />
      ) : (
        <>
          {/* chart */}
          <Card>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <SectionTitle>
                {dataset} · {metric}
              </SectionTitle>
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge tone="zinc">{rows.length} observations</Badge>
                <Badge tone="zinc">{plotRows.length} plotted</Badge>
                {capped && <Badge tone="amber">year-max cap ({CAP})</Badge>}
              </div>
            </div>
            <p className="mt-1 text-[11px] leading-4 text-zinc-500">
              x = year (categorical), y = value as reported. Hover a bar for the paper and
              split. Units note: values are raw as reported by papers.
            </p>
            <div className="mt-3">
              <Plot rows={plotRows} capped={capped} />
            </div>
          </Card>

          {/* best callout */}
          {best && (
            <Card tone="accent">
              <div className="flex flex-wrap items-center gap-3">
                <Badge tone="emerald">best reported</Badge>
                <span className="font-mono text-xl text-emerald-300">{best.value}</span>
                <span className="font-mono text-[11px] text-zinc-400">
                  {best.metric} · {best.dataset} · {best.year}
                </span>
                <PaperLink id={best.paperId} />
                <span className="ml-auto font-mono text-[10px] text-zinc-500">
                  from BENCHMARK_BEST
                </span>
              </div>
            </Card>
          )}

          {/* table */}
          <Card>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <SectionTitle>All observations</SectionTitle>
              <span className="font-mono text-[10px] text-zinc-500">
                {rows.length} rows · raw value as printed
              </span>
            </div>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full border-collapse text-[11px]">
                <thead>
                  <tr className="border-b border-zinc-800">
                    {["year", "paper", "value", "location", "split"].map((h) => (
                      <th
                        key={h}
                        className="p-1.5 text-left font-mono text-[10px] uppercase text-zinc-500"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {slice.map((o, i) => {
                    const p = PAPER_BY_ID[o.paperId];
                    return (
                      <tr key={`${o.paperId}-${i}`} className="border-b border-zinc-900">
                        <td className="p-1.5 font-mono text-zinc-400">{o.year}</td>
                        <td className="p-1.5">
                          <PaperLink id={o.paperId} />
                          {p && <span className="ml-1.5 text-zinc-500">{p.shortTitle}</span>}
                        </td>
                        <td className="p-1.5 font-mono text-emerald-300">{o.raw}</td>
                        <td className="p-1.5 font-mono text-[10px] text-zinc-500">
                          {o.location ?? "—"}
                        </td>
                        <td className="p-1.5 text-[10px] text-zinc-500">{o.split ?? "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {pageCount > 1 && (
              <div className="mt-3 flex items-center justify-center gap-3">
                <button
                  onClick={() => setPage(cur - 1)}
                  disabled={cur === 0}
                  className="rounded-md border border-zinc-800 px-3 py-1 font-mono text-[10px] text-zinc-400 hover:text-zinc-200 disabled:opacity-40"
                >
                  ← prev
                </button>
                <span className="font-mono text-[10px] text-zinc-500">
                  page {cur + 1} / {pageCount}
                </span>
                <button
                  onClick={() => setPage(cur + 1)}
                  disabled={cur >= pageCount - 1}
                  className="rounded-md border border-zinc-800 px-3 py-1 font-mono text-[10px] text-zinc-400 hover:text-zinc-200 disabled:opacity-40"
                >
                  next →
                </button>
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
