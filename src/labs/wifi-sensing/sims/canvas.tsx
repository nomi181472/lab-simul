"use client";

import type { Complex } from "@/labs/wifi-sensing/sim/math";

/* SVG primitives for signal-domain visualisation.
 * No canvas / d3 / charting dependency: plots are hand-built SVG polylines,
 * matching the object-detection lab's BoxCanvas approach.
 */

export const VIEW_W = 460;
export const VIEW_H = 190;
const PAD_L = 42;
const PAD_R = 10;
const PAD_T = 10;
const PAD_B = 24;

function scaleX(i: number, n: number) {
  return PAD_L + (i / Math.max(n - 1, 1)) * (VIEW_W - PAD_L - PAD_R);
}

function scaleY(v: number, min: number, max: number) {
  const span = Math.max(max - min, 1e-9);
  return PAD_T + (1 - (v - min) / span) * (VIEW_H - PAD_T - PAD_B);
}

function extent(ys: number[]): [number, number] {
  if (!ys.length) return [0, 1];
  const min = Math.min(...ys);
  const max = Math.max(...ys);
  if (min === max) return [min - 0.5, max + 0.5];
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}

/** Shared frame: dark background, axes, optional zero line. */
export function PlotFrame({
  yLabel,
  xLabel,
  yTicks = 4,
  domain,
  zeroLine,
}: {
  yLabel: string;
  xLabel: string;
  yTicks?: number;
  domain?: [number, number];
  zeroLine?: boolean;
}) {
  return (
    <g>
      <rect
        x={0}
        y={0}
        width={VIEW_W}
        height={VIEW_H}
        rx={4}
        className="fill-zinc-900"
      />
      {domain
        ? Array.from({ length: yTicks + 1 }, (_, i) => {
            const v = domain[0] + ((domain[1] - domain[0]) * i) / yTicks;
            const y = scaleY(v, domain[0], domain[1]);
            return (
              <g key={i}>
                <line
                  x1={PAD_L}
                  y1={y}
                  x2={VIEW_W - PAD_R}
                  y2={y}
                  stroke="#27272a"
                  strokeWidth={0.5}
                />
                <text
                  x={PAD_L - 4}
                  y={y + 3}
                  fontSize={7}
                  fill="#71717a"
                  textAnchor="end"
                  fontFamily="monospace"
                >
                  {Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)}
                </text>
              </g>
            );
          })
        : null}
      {zeroLine && domain ? (
        <line
          x1={PAD_L}
          y1={scaleY(0, domain[0], domain[1])}
          x2={VIEW_W - PAD_R}
          y2={scaleY(0, domain[0], domain[1])}
          stroke="#3f3f46"
          strokeWidth={1}
          strokeDasharray="3 3"
        />
      ) : null}
      <text x={6} y={12} fontSize={7} fill="#a1a1aa" fontFamily="monospace">
        {yLabel}
      </text>
      <text
        x={VIEW_W - PAD_R}
        y={VIEW_H - 6}
        fontSize={7}
        fill="#a1a1aa"
        textAnchor="end"
        fontFamily="monospace"
      >
        {xLabel}
      </text>
    </g>
  );
}

/** Generic 2-D line plot of one or more series. */
export function LinePlot({
  series,
  yLabel,
  xLabel,
  domain,
  zeroLine,
  markers,
}: {
  series: { ys: number[]; color: string; width?: number; dashed?: boolean; label?: string }[];
  yLabel: string;
  xLabel: string;
  domain?: [number, number];
  zeroLine?: boolean;
  markers?: { i: number; color: string; label: string }[];
}) {
  const all = series.flatMap((s) => s.ys);
  const d = domain ?? extent(all);
  const n = Math.max(...series.map((s) => s.ys.length), 2);
  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      className="w-full"
      preserveAspectRatio="xMidYMid meet"
    >
      <PlotFrame yLabel={yLabel} xLabel={xLabel} domain={d} zeroLine={zeroLine} />
      {series.map((s, si) => (
        <polyline
          key={si}
          points={s.ys
            .map((v, i) => `${scaleX(i, s.ys.length)},${scaleY(v, d[0], d[1])}`)
            .join(" ")}
          fill="none"
          stroke={s.color}
          strokeWidth={s.width ?? 1.5}
          strokeDasharray={s.dashed ? "4 3" : undefined}
          strokeLinejoin="round"
        />
      ))}
      {(markers ?? []).map((m, i) => (
        <g key={i}>
          <line
            x1={scaleX(m.i, n)}
            y1={PAD_T}
            x2={scaleX(m.i, n)}
            y2={VIEW_H - PAD_B}
            stroke={m.color}
            strokeWidth={1}
            strokeDasharray="3 3"
          />
          <text
            x={scaleX(m.i, n) + 3}
            y={PAD_T + 9}
            fontSize={8}
            fill={m.color}
            fontFamily="monospace"
          >
            {m.label}
          </text>
        </g>
      ))}
      {series.map((s, si) =>
        s.label ? (
          <text
            key={`lg${si}`}
            x={VIEW_W - PAD_R - 4}
            y={PAD_T + 10 + si * 10}
            fontSize={8}
            fill={s.color}
            textAnchor="end"
            fontFamily="monospace"
          >
            {s.label}
          </text>
        ) : null,
      )}
    </svg>
  );
}

/** Magnitude/phase pair against subcarrier index or packet index. */
export function CsiPlot({
  cs,
  xLabel,
  noiseSd = 0,
  seed = 5,
}: {
  cs: Complex[];
  xLabel: string;
  noiseSd?: number;
  seed?: number;
}) {
  const mags = cs.map((z) => Math.hypot(z.re, z.im) + (noiseSd ? (seed % 7) * 1e-3 : 0));
  const phs = cs.map((z) => Math.atan2(z.im, z.re));
  const mDom: [number, number] = [0, Math.max(...mags, 1e-6) * 1.1];
  const pDom: [number, number] = [-Math.PI, Math.PI];
  return (
    <div className="space-y-1">
      <svg
        viewBox={`0 0 ${VIEW_W} ${VIEW_H / 1.7}`}
        className="w-full"
        preserveAspectRatio="xMidYMid meet"
      >
        <PlotFrame yLabel="|CSI|" xLabel={xLabel} domain={mDom} yTicks={2} />
        <polyline
          points={mags
            .map(
              (v, i) =>
                `${scaleX(i, mags.length)},${
                  PAD_T +
                  (1 - (v - mDom[0]) / (mDom[1] - mDom[0])) *
                    (VIEW_H / 1.7 - PAD_T - PAD_B)
                }`,
            )
            .join(" ")}
          fill="none"
          stroke="#22d3ee"
          strokeWidth={1.5}
        />
      </svg>
      <svg
        viewBox={`0 0 ${VIEW_W} ${VIEW_H / 1.7}`}
        className="w-full"
        preserveAspectRatio="xMidYMid meet"
      >
        <PlotFrame yLabel="phase" xLabel={xLabel} domain={pDom} yTicks={4} zeroLine />
        <polyline
          points={phs
            .map(
              (v, i) =>
                `${scaleX(i, phs.length)},${
                  PAD_T +
                  (1 - (v - pDom[0]) / (pDom[1] - pDom[0])) *
                    (VIEW_H / 1.7 - PAD_T - PAD_B)
                }`,
            )
            .join(" ")}
          fill="none"
          stroke="#fbbf24"
          strokeWidth={1.5}
        />
      </svg>
    </div>
  );
}

/**
 * Room plan: TX and RX with the direct link, the 60% Fresnel clearance
 * ellipse, and the target's position. Used by the Fresnel and path-loss sims.
 */
export function RoomPlan({
  tx,
  rx,
  target,
  zone,
}: {
  tx: { x: number; y: number };
  rx: { x: number; y: number };
  target?: { x: number; y: number };
  zone?: { d1: number; d2: number; radius: number };
}) {
  const S = 1; // 1 world unit == 1 svg unit
  const cx = (p: { x: number; y: number }) => ({ x: PAD_L + p.x * S, y: VIEW_H - PAD_B - p.y * S });
  const t = cx(tx);
  const r = cx(rx);
  const tz = target ? cx(target) : null;
  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      className="w-full"
      preserveAspectRatio="xMidYMid meet"
    >
      <rect x={0} y={0} width={VIEW_W} height={VIEW_H} rx={4} className="fill-zinc-900" />
      {zone ? (
        <line
          x1={t.x}
          y1={t.y}
          x2={r.x}
          y2={r.y}
          stroke="#22d3ee"
          strokeWidth={zone.radius * S}
          strokeOpacity={0.16}
          strokeLinecap="round"
        />
      ) : null}
      <line x1={t.x} y1={t.y} x2={r.x} y2={r.y} stroke="#22d3ee" strokeWidth={1} strokeDasharray="4 3" />
      {tz ? (
        <>
          <line x1={t.x} y1={t.y} x2={tz.x} y2={tz.y} stroke="#fbbf24" strokeWidth={1.5} />
          <line x1={r.x} y1={r.y} x2={tz.x} y2={tz.y} stroke="#fbbf24" strokeWidth={1.5} />
          <circle cx={tz.x} cy={tz.y} r={4} fill="#fbbf24" />
          <text x={tz.x + 6} y={tz.y - 4} fontSize={8} fill="#fbbf24" fontFamily="monospace">
            target
          </text>
        </>
      ) : null}
      <circle cx={t.x} cy={t.y} r={4} fill="#22d3ee" />
      <text x={t.x - 14} y={t.y + 14} fontSize={8} fill="#22d3ee" fontFamily="monospace">
        TX
      </text>
      <circle cx={r.x} cy={r.y} r={4} fill="#a78bfa" />
      <text x={r.x + 6} y={r.y + 12} fontSize={8} fill="#a78bfa" fontFamily="monospace">
        RX
      </text>
      <text x={6} y={12} fontSize={7} fill="#a1a1aa" fontFamily="monospace">
        room plan (m)
      </text>
    </svg>
  );
}

/** Room-plan variant that also draws the occupied cells of an RSSI grid. */
export function GridPlan({
  w,
  h,
  ap,
  truePos,
  estPos,
}: {
  w: number;
  h: number;
  ap: { x: number; y: number };
  truePos?: { x: number; y: number };
  estPos?: { x: number; y: number };
}) {
  const S = 40;
  const ox = PAD_L;
  const oy = PAD_T;
  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      className="w-full"
      preserveAspectRatio="xMidYMid meet"
    >
      <rect x={0} y={0} width={VIEW_W} height={VIEW_H} rx={4} className="fill-zinc-900" />
      {Array.from({ length: w }, (_, c) =>
        Array.from({ length: h }, (_, r) => (
          <rect
            key={`${c},${r}`}
            x={ox + c * S}
            y={oy + (h - 1 - r) * S}
            width={S}
            height={S}
            fill="#18181b"
            stroke="#27272a"
            strokeWidth={0.5}
          />
        )),
      )}
      <circle cx={ox + ap.x * S + S / 2} cy={oy + (h - 1 - ap.y) * S + S / 2} r={4} fill="#22d3ee" />
      <text
        x={ox + ap.x * S + S / 2 + 6}
        y={oy + (h - 1 - ap.y) * S + S / 2}
        fontSize={8}
        fill="#22d3ee"
        fontFamily="monospace"
      >
        AP
      </text>
      {truePos ? (
        <g>
          <circle
            cx={ox + truePos.x * S + S / 2}
            cy={oy + (h - 1 - truePos.y) * S + S / 2}
            r={5}
            fill="none"
            stroke="#34d399"
            strokeWidth={1.5}
          />
          <text
            x={ox + truePos.x * S + S / 2 + 7}
            y={oy + (h - 1 - truePos.y) * S + S / 2 - 2}
            fontSize={8}
            fill="#34d399"
            fontFamily="monospace"
          >
            true
          </text>
        </g>
      ) : null}
      {estPos ? (
        <g>
          {truePos ? (
            <line
              x1={ox + truePos.x * S + S / 2}
              y1={oy + (h - 1 - truePos.y) * S + S / 2}
              x2={ox + estPos.x * S + S / 2}
              y2={oy + (h - 1 - estPos.y) * S + S / 2}
              stroke="#fbbf24"
              strokeWidth={1}
              strokeDasharray="3 2"
            />
          ) : null}
          <circle
            cx={ox + estPos.x * S + S / 2}
            cy={oy + (h - 1 - estPos.y) * S + S / 2}
            r={4}
            fill="#fbbf24"
          />
          <text
            x={ox + estPos.x * S + S / 2 + 6}
            y={oy + (h - 1 - estPos.y) * S + S / 2 + 12}
            fontSize={8}
            fill="#fbbf24"
            fontFamily="monospace"
          >
            est
          </text>
        </g>
      ) : null}
      <text x={6} y={12} fontSize={7} fill="#a1a1aa" fontFamily="monospace">
        grid (m)
      </text>
    </svg>
  );
}

/** Horizontal share bar — the "share of signal" idiom used for contributions. */
export function ShareBar({
  parts,
}: {
  parts: { label: string; value: number; color: string }[];
}) {
  const total = Math.max(
    parts.reduce((s, p) => s + Math.max(p.value, 0), 0),
    1e-9,
  );
  return (
    <div className="space-y-1">
      <div className="flex h-6 w-full overflow-hidden rounded-md border border-zinc-800">
        {parts.map((p) => (
          <div
            key={p.label}
            style={{ width: `${(Math.max(p.value, 0) / total) * 100}%` }}
            className={p.color}
            title={`${p.label}: ${p.value.toFixed(3)}`}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-3">
        {parts.map((p) => (
          <div key={p.label} className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-sm" style={{ background: p.color }} />
            <span className="font-mono text-[10px] text-zinc-400">{p.label}</span>
            <span className="font-mono text-[10px] text-zinc-200">{p.value.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Vertical bar chart, e.g. a spectrum. */
export function BarPlot({
  ys,
  xLabel,
  yLabel,
  color = "#22d3ee",
  highlight,
}: {
  ys: number[];
  xLabel: string;
  yLabel: string;
  color?: string;
  highlight?: number;
}) {
  const d = extent(ys);
  const n = ys.length;
  const bw = (VIEW_W - PAD_L - PAD_R) / Math.max(n, 1);
  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      className="w-full"
      preserveAspectRatio="xMidYMid meet"
    >
      <PlotFrame yLabel={yLabel} xLabel={xLabel} domain={d} yTicks={3} />
      {ys.map((v, i) => {
        const y = scaleY(v, d[0], d[1]);
        const base = scaleY(Math.max(d[0], 0), d[0], d[1]);
        return (
          <rect
            key={i}
            x={PAD_L + i * bw}
            y={y}
            width={Math.max(bw - 0.5, 0.5)}
            height={Math.max(base - y, 0.5)}
            fill={highlight === i ? "#fbbf24" : color}
            fillOpacity={highlight === i ? 1 : 0.55}
          />
        );
      })}
    </svg>
  );
}

/** A compact metric tile used across the sims. */
export function Stat({
  label,
  value,
  tone = "zinc",
  hint,
}: {
  label: string;
  value: string;
  tone?: "zinc" | "cyan" | "emerald" | "amber" | "rose";
  hint?: string;
}) {
  const tones = {
    zinc: "border-zinc-800 text-zinc-200",
    cyan: "border-cyan-700/60 bg-cyan-950/20 text-cyan-200",
    emerald: "border-emerald-700/60 bg-emerald-950/20 text-emerald-200",
    amber: "border-amber-700/60 bg-amber-950/20 text-amber-200",
    rose: "border-rose-700/60 bg-rose-950/20 text-rose-200",
  };
  return (
    <div className={`rounded-lg border p-2 text-center ${tones[tone]}`} title={hint}>
      <div className="font-mono text-[9px] uppercase text-zinc-500">{label}</div>
      <div className="mt-1 font-mono text-sm">{value}</div>
    </div>
  );
}