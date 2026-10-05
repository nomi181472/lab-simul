"use client";

/* Deterministic evolutionary-search simulators.
 *
 * Dependency-free (divs + SVG + range inputs) to match the other labs, and
 * fully deterministic: every run is driven by a small LCG seeded from the
 * controls, so the same settings always produce the same population, the same
 * curve and the same genome strip. Nothing here is random per render.
 *
 * The genome is a real genotype/phenotype pair so the diagrams in
 * architecture.tsx can be driven by it directly: mutate genes, decode to a
 * network, draw the network.
 */

/* ---------------------------------------------------------------- rng */

export function lcg(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    // Numerical Recipes LCG; 32-bit so results are identical everywhere.
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function gaussian(rand: () => number) {
  // Box-Muller, single output (the pair is discarded for determinism's sake).
  const u = Math.max(rand(), 1e-9);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export type Gene = number;

export type Genome = Gene[];

export type NetworkSpec = {
  /** hidden widths, evolved */
  widths: number[];
  inputDim: number;
  outputDim: number;
  kind: "mlp" | "recurrent" | "conv";
};

/* ------------------------------------------------- fitness landscapes */

/** Fit an evolved MLP's width vector to a target curve. */
export function targetCurve(n: number, freq: number, seed: number) {
  const rand = lcg(seed);
  const noise = Array.from({ length: n }, () => gaussian(rand) * 0.08);
  return (t: number) =>
    Math.sin(2 * Math.PI * freq * t) * 0.6 + Math.sin(2 * Math.PI * freq * 2.7 * t) * 0.25 + noise[Math.floor(t * n) % n];
}

/** Score a width vector: smaller is better, with a sweet spot around 24. */
export function widthFitness(widths: number[], budget: number) {
  const total = widths.reduce((a, b) => a + b, 0);
  const over = Math.max(0, total - budget);
  const under = Math.max(0, budget * 0.35 - total);
  // Prefer enough capacity, punish excess hard: a real compute-accuracy tradeoff.
  return over * 0.6 + under * 0.4 + total * 0.02;
}

export function encodeWidths(rand: () => number, count: number, max: number): Genome {
  return Array.from({ length: count }, () => Math.round(rand() * max));
}

export function mutateWidths(g: Genome, rate: number, max: number, rand: () => number): Genome {
  return g.map((v) => {
    if (rand() > rate) return v;
    if (rand() < 0.5) {
      return Math.max(1, Math.min(max, v + Math.round(1 + rand() * 4)));
    }
    return Math.max(1, Math.min(max, v - Math.round(1 + rand() * 4)));
  });
}

export function crossoverWidths(a: Genome, b: Genome, rand: () => number): Genome {
  const cut = 1 + Math.floor(rand() * (a.length - 1));
  return [...a.slice(0, cut), ...b.slice(cut)];
}

export type RunConfig = {
  population: number;
  generations: number;
  mutationRate: number;
  crossoverRate: number;
  elite: number;
  layers: number;
  maxWidth: number;
  budget: number;
  seed: number;
};

export type RunResult = {
  /** best-so-far fitness per generation, index 0 = generation 1 */
  curve: number[];
  mean: number[];
  /** diversity: mean pairwise L1 distance, scaled to 0..1 */
  diversity: number[];
  best: Genome;
  finalPop: Genome[];
  /** per-locus mutation counts across the run, for the modification panel */
  locusHits: number[];
};

export function runGA(cfg: RunConfig): RunResult {
  const rand = lcg(cfg.seed);
  let pop: Genome[] = Array.from({ length: cfg.population }, () =>
    encodeWidths(rand, cfg.layers, cfg.maxWidth),
  );
  const curve: number[] = [];
  const mean: number[] = [];
  const diversity: number[] = [];
  const locusHits = new Array(cfg.layers).fill(0);
  let best: Genome = pop[0];
  let bestFit = Infinity;

  const fitness = (g: Genome) => widthFitness(g, cfg.budget);

  for (let gen = 0; gen < cfg.generations; gen++) {
    const scored = pop.map((g) => ({ g, f: fitness(g) })).sort((x, y) => x.f - y.f);
    if (scored[0].f < bestFit) {
      bestFit = scored[0].f;
      best = scored[0].g;
    }
    curve.push(bestFit);
    mean.push(scored.reduce((a, s) => a + s.f, 0) / scored.length);

    // diversity: mean pairwise distance normalised by the genotype range
    let d = 0;
    for (let i = 0; i < scored.length; i++) {
      for (let j = i + 1; j < scored.length; j++) {
        d += scored[i].g.reduce((a, v, k) => a + Math.abs(v - scored[j].g[k]), 0);
      }
    }
    const pairs = (scored.length * (scored.length - 1)) / 2 || 1;
    diversity.push(d / pairs / (cfg.maxWidth * cfg.layers));

    const next: Genome[] = scored.slice(0, cfg.elite).map((s) => s.g.slice());
    while (next.length < cfg.population) {
      const a = scored[Math.floor(rand() * Math.max(1, scored.length / 2))].g;
      const b = scored[Math.floor(rand() * scored.length)].g;
      let child = rand() < cfg.crossoverRate ? crossoverWidths(a, b, rand) : a.slice();
      const before = child.slice();
      child = mutateWidths(child, cfg.mutationRate, cfg.maxWidth, rand);
      for (let k = 0; k < child.length; k++) {
        if (child[k] !== before[k]) locusHits[k]++;
      }
      next.push(child);
    }
    pop = next;
  }
  return { curve, mean, diversity, best, finalPop: pop, locusHits };
}

/* ------------------------------------------------------ quality diversity */

/** MAP-Elites archive over a 2-D behaviour descriptor. */
export type ArchiveCell = { x: number; y: number; quality: number; genome: Genome };

export function runQD(cfg: RunConfig, cells: number): ArchiveCell[] {
  const rand = lcg(cfg.seed + 77);
  const grid = new Map<string, ArchiveCell>();
  const n = Math.max(cfg.population, 24);

  // behaviour descriptor: two thresholds on the first two hidden widths
  const behaviour = (g: Genome) => {
    const a = Math.min(1, g[0] / cfg.maxWidth);
    const b = Math.min(1, (g[1] ?? g[0]) / cfg.maxWidth);
    return [a, b];
  };
  const objective = (g: Genome) => {
    const total = g.reduce((s, v) => s + v, 0);
    return 1 / (1 + Math.abs(total - cfg.budget) / cfg.budget);
  };

  for (let iter = 0; iter < cfg.generations * n; iter++) {
    const g = encodeWidths(rand, cfg.layers, cfg.maxWidth);
    const [x, y] = behaviour(g);
    const key = `${Math.floor(x * cells)}:${Math.floor(y * cells)}`;
    const q = objective(g);
    const cur = grid.get(key);
    if (!cur || q > cur.quality) grid.set(key, { x, y, quality: q, genome: g });
  }
  return [...grid.values()];
}

/** Novelty score of a candidate against a reference set. */
export function novelty(cand: Genome, refs: Genome[], k: number) {
  if (refs.length === 0) return 0;
  const d = (a: Genome, b: Genome) =>
    a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);
  const dists = refs.map((r) => d(cand, r)).sort((x, y) => x - y);
  const take = dists.slice(0, Math.max(1, Math.min(k, dists.length)));
  return take.reduce((a, b) => a + b, 0) / take.length;
}

/* ------------------------------------------------------------- svg plots */

export type Series = { label: string; color: string; values: number[] };

export function Sparkline({
  series,
  height = 96,
  yLabel,
}: {
  series: Series[];
  height?: number;
  yLabel?: string;
}) {
  const w = 320;
  const pad = 6;
  const all = series.flatMap((s) => s.values);
  if (all.length === 0) return null;
  let min = Math.min(...all);
  let max = Math.max(...all);
  if (max - min < 1e-9) {
    min -= 0.5;
    max += 0.5;
  }
  const sx = (i: number, len: number) => pad + (i / Math.max(1, len - 1)) * (w - 2 * pad);
  const sy = (v: number) => height - pad - ((v - min) / (max - min)) * (height - 2 * pad);

  return (
    <svg
      viewBox={`0 0 ${w} ${height}`}
      className="h-auto w-full"
      role="img"
      aria-label={yLabel ?? "curve"}
    >
      {series.map((s) => (
        <polyline
          key={s.label}
          fill="none"
          stroke={s.color}
          strokeWidth={1.6}
          strokeLinejoin="round"
          strokeLinecap="round"
          points={s.values.map((v, i) => `${sx(i, s.values.length)},${sy(v)}`).join(" ")}
        />
      ))}
    </svg>
  );
}

/** Render an evolved MLP as stacked bars sized by the decoded widths. */
export function NetworkBars({ widths, height = 74 }: { widths: number[]; height?: number }) {
  if (!widths.length) return null;
  const max = Math.max(...widths);
  const cols = [
    "bg-sky-500/80",
    "bg-violet-500/80",
    "bg-emerald-500/80",
    "bg-amber-500/80",
    "bg-cyan-500/80",
  ];
  return (
    <div className="flex items-end gap-1.5" style={{ height }}>
      {widths.map((w, i) => (
        <div key={i} className="flex flex-1 flex-col items-center gap-1">
          <div
            className={`w-full rounded-t ${cols[i % cols.length]}`}
            style={{ height: Math.max(3, (w / max) * (height - 22)) }}
            title={`layer ${i + 1}: ${w} units`}
          />
          <span className="font-mono text-[9px] text-zinc-500">{w}</span>
        </div>
      ))}
    </div>
  );
}

export const DEFAULT_CONFIG: RunConfig = {
  population: 40,
  generations: 60,
  mutationRate: 0.14,
  crossoverRate: 0.7,
  elite: 2,
  layers: 3,
  maxWidth: 48,
  budget: 72,
  seed: 7,
};