"use client";

import type {
  Architecture,
  ArchitectureLayer,
  Locus,
  LocusChange,
  LayerKind,
} from "@/labs/neuroevolution/data/types";
import {
  LAYER_COLOR,
  LAYER_LABEL,
  LOCUS_COLOR,
  LOCUS_LABEL,
} from "@/labs/neuroevolution/data/types";

/* Colourful architecture + modification diagrams.
 *
 * Two things a neuroevolution reader cannot get from prose:
 *   1. the *shape* of the network the search produced, and
 *   2. *which part of the candidate the operators actually changed*.
 * These components draw both. Everything is deterministic SVG — no chart
 * library — and when a paper does not state something, the diagram says so
 * rather than inventing a plausible network.
 */

const LAYER_W = 74;
const LAYER_H = 54;
const LAYER_GAP = 26;
const PADDING = 16;

/** Scale bar for "how wide is this layer", in arbitrary relative units. */
function layerScale(layers: ArchitectureLayer[]): number {
  const units = layers.map((l) => l.units ?? 0);
  const max = Math.max(...units, 0);
  return max > 0 ? max : 0;
}

export function LayerLegend({ kinds }: { kinds?: LayerKind[] }) {
  const used = (kinds ?? (Object.keys(LAYER_COLOR) as LayerKind[])).filter((k) =>
    k === "unknown" ? false : true,
  );
  return (
    <div className="flex flex-wrap gap-1.5">
      {used.map((k) => (
        <span
          key={k}
          className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900/60 px-1.5 py-0.5 text-[10px] text-zinc-400"
        >
          <span
            aria-hidden
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ background: LAYER_COLOR[k] }}
          />
          <span className="font-mono">{LAYER_LABEL[k]}</span>
        </span>
      ))}
    </div>
  );
}

/**
 * The evolved network as a left-to-right chain of coloured blocks. Block width
 * encodes node/channel count when the paper states one, so a 512-unit layer
 * reads as visibly wider than a 16-unit one.
 */
export function ArchitectureDiagram({
  architecture,
  compact = false,
}: {
  architecture: Architecture;
  compact?: boolean;
}) {
  const layers = architecture.layers;

  if (layers.length === 0) {
    return (
      <div className="flex flex-col gap-2 rounded-xl border border-dashed border-zinc-700 bg-zinc-900/40 p-4">
        <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
          architecture diagram
        </div>
        <div className="text-[12px] leading-5 text-zinc-400">
          {architecture.summary || "No summary recorded."}
        </div>
        <div className="font-mono text-[11px] text-amber-400/80">
          Layer-by-layer topology is not enumerated in the retrieved text — this
          paper describes its network in prose only, so no diagram is drawn
          rather than guessing one.
        </div>
      </div>
    );
  }

  const scale = layerScale(layers);
  const widths = layers.map((l) => {
    if (!scale || !l.units) return LAYER_W;
    // log scale: 8..1024 units stays legible, unlike a linear one
    const t = Math.log2(Math.max(l.units, 2)) / Math.log2(Math.max(scale, 2));
    return LAYER_W * (0.6 + 0.9 * t);
  });
  const total =
    widths.reduce((a, b) => a + b, 0) + LAYER_GAP * (layers.length - 1) + PADDING * 2;
  const height = LAYER_H + (compact ? PADDING + 6 : PADDING * 2 + 14);

  return (
    <div className="flex flex-col gap-2">
      <svg
        viewBox={`0 0 ${total} ${height}`}
        width="100%"
        height={height}
        role="img"
        aria-label={`Evolved architecture: ${layers.map((l) => l.label).join(" then ")}`}
        preserveAspectRatio="xMinYMid meet"
      >
        <defs>
          <marker
            id="arch-arrow"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M0,0 L8,4 L0,8 z" fill="#52525b" />
          </marker>
        </defs>
        {layers.map((l, i) => {
          const x = PADDING + widths.slice(0, i).reduce((a, b) => a + b, 0) + LAYER_GAP * i;
          const w = widths[i];
          const c = LAYER_COLOR[l.kind];
          return (
            <g key={i}>
              <rect
                x={x}
                y={PADDING}
                width={w}
                height={LAYER_H}
                rx="8"
                fill={`${c}22`}
                stroke={c}
                strokeWidth="1.5"
              />
              <text
                x={x + w / 2}
                y={PADDING + 20}
                textAnchor="middle"
                fontSize="9"
                fontFamily="ui-monospace, monospace"
                fill={c}
              >
                {LAYER_LABEL[l.kind]}
              </text>
              <text
                x={x + w / 2}
                y={PADDING + 34}
                textAnchor="middle"
                fontSize="9"
                fontFamily="ui-monospace, monospace"
                fill="#d4d4d8"
              >
                {l.label.length > 16 ? `${l.label.slice(0, 15)}…` : l.label}
              </text>
              {l.units ? (
                <text
                  x={x + w / 2}
                  y={PADDING + 46}
                  textAnchor="middle"
                  fontSize="8"
                  fontFamily="ui-monospace, monospace"
                  fill="#71717a"
                >
                  {l.units} units
                </text>
              ) : null}
              {i < layers.length - 1 ? (
                <line
                  x1={x + w}
                  y1={PADDING + LAYER_H / 2}
                  x2={x + w + LAYER_GAP}
                  y2={PADDING + LAYER_H / 2}
                  stroke="#52525b"
                  strokeWidth="1.5"
                  markerEnd="url(#arch-arrow)"
                />
              ) : null}
            </g>
          );
        })}
      </svg>
      <LayerLegend kinds={layers.map((l) => l.kind)} />
    </div>
  );
}

/* ------------------------------------------------------------------ genome */

/**
 * A genome as a strip of genes, grouped by locus and coloured by locus.
 * `changed` marks the loci an operator touched, which is how the modification
 * view shows *what* a mutation actually did rather than just that one happened.
 */
export function GenomeStrip({
  loci,
  changed,
  height = 26,
}: {
  loci: Locus[];
  changed?: Locus[];
  height?: number;
}) {
  const hot = new Set(changed ?? []);
  const all = loci.length ? loci : (Object.keys(LOCUS_COLOR) as Locus[]);
  const perGene = 7;
  const gap = 3;
  const w = all.length * (perGene + gap) - gap;

  return (
    <svg
      viewBox={`0 0 ${w} ${height}`}
      width="100%"
      height={height}
      preserveAspectRatio="xMinYMid meet"
      role="img"
      aria-label={`Genome with ${all.length} loci${
        hot.size ? `, ${hot.size} modified` : ""
      }`}
    >
      {all.map((locus, i) => {
        const c = LOCUS_COLOR[locus];
        const on = hot.has(locus);
        return (
          <g key={locus}>
            <rect
              x={i * (perGene + gap)}
              y={on ? 0 : 3}
              width={perGene}
              height={on ? height : height - 6}
              rx="2"
              fill={on ? c : `${c}55`}
              stroke={on ? "#fff" : c}
              strokeWidth={on ? 1.2 : 0.8}
            />
            {on ? (
              <text
                x={i * (perGene + gap) + perGene / 2}
                y={height + 9}
                textAnchor="middle"
                fontSize="7"
                fontFamily="ui-monospace, monospace"
                fill={c}
              >
                ▲
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

/**
 * What the operators modify. Rendered as one coloured row per locus the paper
 * actually changes, with the operator's own words and a quote when available.
 */
export function ModificationDiagram({ changes }: { changes: LocusChange[] }) {
  if (changes.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-zinc-700 bg-zinc-900/40 p-4 font-mono text-[11px] text-amber-400/80">
        No locus-level modification is described in the retrieved text.
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-1.5">
      {changes.map((c, i) => {
        const color = LOCUS_COLOR[c.locus];
        return (
          <div
            key={i}
            className="flex items-start gap-2 rounded-lg border border-zinc-800 bg-zinc-900/50 px-2.5 py-1.5"
          >
            <span
              aria-hidden
              className="mt-0.5 inline-block h-3.5 w-1.5 shrink-0 rounded-sm"
              style={{ background: color }}
            />
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span
                  className="font-mono text-[11px] font-semibold"
                  style={{ color }}
                >
                  {LOCUS_LABEL[c.locus]}
                </span>
                <span className="text-[11px] text-zinc-400">{c.operator}</span>
              </div>
              {c.quote ? (
                <div className="mt-0.5 truncate text-[10px] italic text-zinc-600">
                  “{c.quote}”
                </div>
              ) : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** Locus colour key, for the modification view. */
export function LocusLegend() {
  return (
    <div className="flex flex-wrap gap-1.5">
      {(Object.keys(LOCUS_COLOR) as Locus[]).map((l) => (
        <span
          key={l}
          className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900/60 px-1.5 py-0.5 text-[10px] text-zinc-400"
        >
          <span
            aria-hidden
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ background: LOCUS_COLOR[l] }}
          />
          <span className="font-mono">{LOCUS_LABEL[l]}</span>
        </span>
      ))}
    </div>
  );
}