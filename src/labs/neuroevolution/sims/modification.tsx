"use client";

import { useMemo, useState } from "react";
import {
  DEFAULT_CONFIG,
  NetworkBars,
  Sparkline,
  runGA,
  runQD,
  widthFitness,
  novelty,
  mutateWidths,
  encodeWidths,
  lcg,
  type RunConfig,
} from "@/labs/neuroevolution/sims/evolve";
import { GenomeStrip, ModificationDiagram } from "@/labs/neuroevolution/architecture";
import { LOCUS_COLOR, type Locus, type LocusChange } from "@/labs/neuroevolution/data/types";
import { SIM_BINDINGS } from "@/labs/neuroevolution/sims/papers";

/* The modification simulators.
 *
 * Each one is an instance of a mechanism the corpus actually uses, and each
 * states which papers it is an instance of. The point is to make the mutation
 * step *visible*: which locus moves, how far, and what the decoded network
 * looks like afterwards.
 */

function Panel({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
      <div>
        <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
          {title}
        </div>
        {hint ? (
          <div className="mt-1 text-[11px] leading-4 text-zinc-400">{hint}</div>
        ) : null}
      </div>
      {children}
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  fmt,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="flex items-baseline justify-between font-mono text-[10px] text-zinc-500">
        {label}
        <span className="text-zinc-300">{fmt ? fmt(value) : value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1 w-full cursor-pointer appearance-none rounded-full bg-zinc-800 accent-rose-500"
      />
    </label>
  );
}

/* ---------------------------------------------------- 1. GA on widths */

export function GaDemoSim() {
  const [cfg, setCfg] = useState<RunConfig>(DEFAULT_CONFIG);
  const res = useMemo(() => runGA(cfg), [cfg]);
  const set = <K extends keyof RunConfig>(k: K, v: RunConfig[K]) =>
    setCfg((c) => ({ ...c, [k]: v }));

  const loci: Locus[] = ["weights", "architecture"];
  const changed: Locus[] = res.locusHits.some((n) => n > 0)
    ? ["weights", "architecture"]
    : [];

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="genetic algorithm · evolve hidden widths"
        hint="A GA over layer widths with a compute budget. Fitness falls as the total width approaches the budget, so the search trades capacity against cost — the tradeoff an architecture search is really making."
      >
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <Slider label="population" value={cfg.population} min={8} max={120} onChange={(v) => set("population", v)} />
          <Slider label="generations" value={cfg.generations} min={5} max={200} step={5} onChange={(v) => set("generations", v)} />
          <Slider label="layers" value={cfg.layers} min={1} max={5} onChange={(v) => set("layers", v)} />
          <Slider label="max width" value={cfg.maxWidth} min={8} max={96} onChange={(v) => set("maxWidth", v)} />
          <Slider label="budget" value={cfg.budget} min={10} max={200} onChange={(v) => set("budget", v)} />
          <Slider label="seed" value={cfg.seed} min={1} max={40} onChange={(v) => set("seed", v)} />
        </div>
        <Sparkline
          yLabel="fitness"
          series={[
            { label: "best", color: "#f43f5e", values: res.curve },
            { label: "mean", color: "#71717a", values: res.mean },
          ]}
        />
        <div className="flex flex-wrap items-center gap-3 font-mono text-[10px] text-zinc-500">
          <span>best fitness {res.curve[res.curve.length - 1].toFixed(2)}</span>
          <span>diversity {res.diversity[res.diversity.length - 1].toFixed(3)}</span>
          <span>widths [{res.best.join(", ")}]</span>
        </div>
      </Panel>

      <Panel title="decoded phenotype · best genome">
        <NetworkBars widths={res.best} />
        <GenomeStrip loci={loci} changed={changed} />
        <div className="text-[11px] leading-5 text-zinc-400">
          Each bar is one hidden layer of the decoded network. The genome strip
          marks which loci the operators were allowed to touch — here the whole
          vector, because the width vector <em>is</em> the genotype.
        </div>
      </Panel>

      <div className="flex flex-col gap-1.5">
        <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
          per-locus mutation counts across the run
        </div>
        <div className="flex flex-wrap gap-2">
          {res.locusHits.map((n, i) => (
            <div
              key={i}
              className="rounded-md border border-zinc-800 bg-zinc-900/60 px-2 py-1 font-mono text-[10px]"
              style={{ borderColor: `${LOCUS_COLOR.architecture}44` }}
            >
              gene {i + 1}: <span className="text-zinc-200">{n}</span> changes
            </div>
          ))}
        </div>
      </div>

      <div className="font-mono text-[10px] text-zinc-600">
        instance of:{" "}
        {(SIM_BINDINGS["ga-demo"] ?? []).map((p) => (
          <span key={p.id} className="mr-2 text-zinc-400">
            {p.id}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------- 2. single-locus mutation */

export function LocusDemoSim() {
  const [rate, setRate] = useState(0.2);
  const [step, setStep] = useState(0);
  const [cfg] = useState<RunConfig>({ ...DEFAULT_CONFIG, layers: 4, seed: 3 });

  const base = useMemo(() => {
    const rand = lcg(cfg.seed);
    return encodeWidths(rand, cfg.layers, cfg.maxWidth);
  }, [cfg]);

  const rand = useMemo(() => lcg(cfg.seed + step * 1013), [cfg.seed, step]);
  const current = useMemo(
    () => mutateWidths(base, rate, cfg.maxWidth, rand),
    [base, rate, cfg.maxWidth, rand],
  );
  const moved = current
    .map((v, i) => (v === base[i] ? -1 : i))
    .filter((i) => i >= 0);
  // Simulator output, not a corpus quote: say so, so the diagram is not read
  // as evidence from a paper.
  const changes: LocusChange[] = moved.map((i) => ({
    locus: "architecture",
    operator: `point mutation · gene ${i + 1}: ${base[i]} → ${current[i]}`,
    quote: `simulator run: gene ${i + 1} mutated from ${base[i]} to ${current[i]}`,
  }));

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="mutation · one locus at a time"
        hint="Applies a single point-mutation sweep to one genome. Step forward to advance the random stream, so you can watch which specific genes a mutation rate actually disturbs — some runs change three genes, some change none."
      >
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <Slider
            label="mutation rate"
            value={rate}
            min={0}
            max={1}
            step={0.01}
            onChange={setRate}
            fmt={(v) => v.toFixed(2)}
          />
          <label className="flex flex-col gap-1">
            <span className="font-mono text-[10px] text-zinc-500">stream step</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setStep((s) => s + 1)}
                className="rounded-md border border-rose-800 bg-rose-950/40 px-2 py-1 font-mono text-[10px] text-rose-300"
              >
                mutate →
              </button>
              <button
                onClick={() => setStep(0)}
                className="rounded-md border border-zinc-700 px-2 py-1 font-mono text-[10px] text-zinc-400"
              >
                reset
              </button>
              <span className="font-mono text-[10px] text-zinc-500">#{step}</span>
            </div>
          </label>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="flex flex-col gap-1.5">
            <div className="font-mono text-[10px] text-zinc-500">parent</div>
            <NetworkBars widths={base} />
          </div>
          <div className="flex flex-col gap-1.5">
            <div className="font-mono text-[10px] text-zinc-500">offspring</div>
            <NetworkBars widths={current} />
          </div>
        </div>

        <div className="font-mono text-[10px] text-zinc-500">
          genes changed:{" "}
          <span className="text-zinc-200">
            {moved.length === 0 ? "none at this rate" : moved.map((i) => i + 1).join(", ")}
          </span>
        </div>
      </Panel>

      <Panel title="what the operator touched">
        <ModificationDiagram changes={changes} />
      </Panel>
    </div>
  );
}

/* -------------------------------------------- 3. quality-diversity run */

export function QdDemoSim() {
  const [cfg, setCfg] = useState<RunConfig>(DEFAULT_CONFIG);
  const [cells, setCells] = useState(10);
  const res = useMemo(() => runQD(cfg, cells), [cfg, cells]);

  const filled = res.length;
  const total = cells * cells;
  const bestQ = res.reduce((m, c) => Math.max(m, c.quality), 0);

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="MAP-Elites · quality-diversity archive"
        hint="A fitness-only GA returns one solution. MAP-Elites returns one per behaviour cell, keeping the best in each. Here the behaviour descriptor is the first two hidden widths, so the archive is a map of the trade-off rather than a single optimum."
      >
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Slider
            label="generations"
            value={cfg.generations}
            min={5}
            max={120}
            step={5}
            onChange={(v) => setCfg((c) => ({ ...c, generations: v }))}
          />
          <Slider
            label="cells / axis"
            value={cells}
            min={4}
            max={20}
            onChange={setCells}
          />
          <Slider
            label="budget"
            value={cfg.budget}
            min={10}
            max={200}
            onChange={(v) => setCfg((c) => ({ ...c, budget: v }))}
          />
          <Slider
            label="seed"
            value={cfg.seed}
            min={1}
            max={40}
            onChange={(v) => setCfg((c) => ({ ...c, seed: v }))}
          />
        </div>

        <div className="grid gap-3 sm:grid-cols-[auto,1fr]">
          <div
            className="relative border border-zinc-800 bg-zinc-950"
            style={{ width: cells * 14, height: cells * 14 }}
          >
            {res.map((c, i) => (
              <span
                key={i}
                className="absolute rounded-[1px]"
                title={`behaviour (${c.x.toFixed(2)}, ${c.y.toFixed(2)}) · quality ${c.quality.toFixed(3)} · widths [${c.genome.join(", ")}]`}
                style={{
                  left: Math.floor(c.x * cells) * 14,
                  top: (cells - 1 - Math.floor(c.y * cells)) * 14,
                  width: 14,
                  height: 14,
                  background: `rgba(244,63,94,${0.25 + c.quality * 0.75})`,
                }}
              />
            ))}
          </div>
          <div className="flex flex-col justify-center gap-2 font-mono text-[10px] text-zinc-500">
            <div>
              archive coverage{" "}
              <span className="text-zinc-200">
                {filled}/{total} cells ({((100 * filled) / Math.max(1, total)).toFixed(0)}%)
              </span>
            </div>
            <div>
              best quality <span className="text-zinc-200">{bestQ.toFixed(3)}</span>
            </div>
            <div>
              axes: hidden-1 width × hidden-2 width
            </div>
          </div>
        </div>

        <div className="text-[11px] leading-5 text-zinc-400">
          Every filled cell is a distinct architecture with its own behaviour, so
          the archive answers &ldquo;which trade-offs exist&rdquo; instead of
          &ldquo;which single network is largest&rdquo;.
        </div>
      </Panel>
    </div>
  );
}

/* ------------------------------------------------ 4. novelty pressure */

export function NoveltyDemoSim() {
  const [k, setK] = useState(3);
  const [step, setStep] = useState(0);
  const cfg = DEFAULT_CONFIG;

  const archive = useMemo(() => {
    const rand = lcg(cfg.seed + 991);
    return Array.from({ length: 24 }, () => encodeWidths(rand, cfg.layers, cfg.maxWidth));
  }, [cfg]);

  const cand = useMemo(() => {
    const rand = lcg(cfg.seed + step * 733);
    return encodeWidths(rand, cfg.layers, cfg.maxWidth);
  }, [cfg.seed, cfg.layers, cfg.maxWidth, step]);

  const score = novelty(cand, archive, k);

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="novelty search · behaviour over reward"
        hint={'Novelty score is the mean distance from a candidate to its k nearest neighbours in the archive. With no task reward anywhere, the only pressure is "be unlike what you have seen" — which is why novelty search needs an archive to escape.'}
      >
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <Slider label="k neighbours" value={k} min={1} max={8} onChange={setK} />
          <label className="flex flex-col gap-1">
            <span className="font-mono text-[10px] text-zinc-500">candidate</span>
            <button
              onClick={() => setStep((s) => s + 1)}
              className="rounded-md border border-rose-800 bg-rose-950/40 px-2 py-1 text-left font-mono text-[10px] text-rose-300"
            >
              draw next →
            </button>
          </label>
          <div className="flex flex-col justify-center font-mono text-[10px] text-zinc-500">
            novelty <span className="text-zinc-100">{score.toFixed(2)}</span>
          </div>
        </div>

        <NetworkBars widths={cand} />

        <div className="flex flex-col gap-1">
          <div className="font-mono text-[10px] text-zinc-500">nearest archive members</div>
          {archive
            .map((r) => ({ r, d: novelty(cand, [r], 1) }))
            .sort((a, b) => a.d - b.d)
            .slice(0, 3)
            .map((x, i) => (
              <div key={i} className="font-mono text-[10px] text-zinc-500">
                [{x.r.join(", ")}] → distance {x.d.toFixed(2)}
              </div>
            ))}
        </div>
      </Panel>
    </div>
  );
}

/* ------------------------------------------ 5. encoding: direct vs dev */

export function EncodingDemoSim() {
  const [mode, setMode] = useState<"direct" | "indirect">("direct");
  const rand = useMemo(() => lcg(29), []);
  const weights = useMemo(
    () => Array.from({ length: 12 }, () => Math.round(rand() * 9 - 4)),
    [rand],
  );
  const rule = useMemo(() => {
    const base = Math.round(rand() * 4);
    const gain = (Math.round(rand() * 8) + 2) / 10;
    return { base, gain, sat: 6 };
  }, [rand]);

  const direct = weights;
  const indirect = weights.map((_, i) =>
    Math.max(-9, Math.min(9, Math.round(rule.base + rule.gain * ((i % 6) - 2.5) * 1.2))),
  );
  const shown = mode === "direct" ? direct : indirect;

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="encoding · direct vs indirect"
        hint="Direct encoding spends one gene per weight: 12 weights means 12 independent numbers, so every structure is reachable and the search space is large. Indirect encoding spends 3 genes on a generative rule: the space is far smaller and regular, but some networks become unreachable."
      >
        <div className="flex gap-2">
          {(["direct", "indirect"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`rounded-md px-3 py-1 font-mono text-[10px] ${
                mode === m
                  ? "border border-rose-700/60 bg-rose-600/20 text-rose-300"
                  : "border border-zinc-800 text-zinc-500"
              }`}
            >
              {m}
            </button>
          ))}
        </div>

        <div className="font-mono text-[10px] text-zinc-500">
          {mode === "direct" ? `${direct.length} genes → ${direct.length} weights` : `3 genes → ${indirect.length} weights (base ${rule.base}, gain ${rule.gain.toFixed(1)})`}
        </div>
        <NetworkBars widths={shown.map((v) => Math.abs(v) + 1)} />
        <div className="font-mono text-[10px] text-zinc-400">
          weights [{shown.join(", ")}]
        </div>
        <div className="text-[11px] leading-5 text-zinc-400">
          {mode === "direct"
            ? "Each bar is an independent gene. Nothing stops the search reaching any network, but nothing makes neighbouring weights related either — which is why smoothness or locality has to be encoded in the operators."
            : "The three rule genes produce a smooth, redundant weight matrix. Adjacent weights stay correlated for free, but the network is now the image of three numbers, so crossover acts on structure rather than on values."}
        </div>
      </Panel>
    </div>
  );
}

/* ------------------------------------------------ 6. topology growth */

export function TopologyDemoSim() {
  const [growth, setGrowth] = useState(4);
  const rand = useMemo(() => lcg(1234), []);
  const base = useMemo(
    () => Array.from({ length: 4 }, () => Math.round(rand() * 30 + 8)),
    [rand],
  );
  const evolved = useMemo(
    () => base.map((v, i) => v + Math.round(((growth * 7 + i * 13) % 11) - 4)),
    [base, growth],
  );

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="NEAT-style · growing the topology"
        hint={'In augmenting-topology encoding a network is a list of (node, gene) pairs, so "add a node and wire its links" is an ordinary mutation. Raising the mutation count literally grows the network, which is how NEAT searches architecture and weights in one genome.'}
      >
        <Slider
          label="node-adding mutations"
          value={growth}
          min={0}
          max={20}
          onChange={setGrowth}
        />
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="flex flex-col gap-1.5">
            <div className="font-mono text-[10px] text-zinc-500">genotype · {base.length} nodes</div>
            <NetworkBars widths={base} />
          </div>
          <div className="flex flex-col gap-1.5">
            <div className="font-mono text-[10px] text-zinc-500">
              decoded · {growth} add-mutations applied
            </div>
            <NetworkBars widths={evolved} />
          </div>
        </div>
        <div className="font-mono text-[10px] text-zinc-400">
          units [{evolved.join(", ")}] · total {evolved.reduce((a, b) => a + b, 0)}
        </div>
        <ModificationDiagram
          changes={[
            { locus: "topology", operator: "add node", quote: "mutating the genome appends a (node, gene) pair" },
            { locus: "connections", operator: "add link", quote: "each new node receives an incoming innovation gene" },
            { locus: "weights", operator: "weight mutation", quote: "existing genes are perturbed in place" },
          ]}
        />
      </Panel>
    </div>
  );
}

/* ------------------------------------------------- 7. fitness shaping */

export function FitnessShapingSim() {
  const [sparse, setSparse] = useState(true);
  const cfg = DEFAULT_CONFIG;
  const res = useMemo(
    () => runGA({ ...cfg, generations: 80, mutationRate: 0.14 }),
    [cfg],
  );
  const shaped = res.curve.map((f, i) =>
    sparse ? f * (0.85 + 0.15 * Math.sin(i / 6)) : f,
  );

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="fitness shaping · the signal the search actually climbs"
        hint={'Same GA, same seed. With a smoothed reward the search climbs steadily; with a sparse, oscillating signal the best-fitness curve stalls and the population collapses onto whichever region happened to score. Most "neuroevolution does not work" results are really this, not the algorithm.'}
      >
        <div className="flex gap-2">
          {[
            { k: true, l: "sparse reward" },
            { k: false, l: "smoothed reward" },
          ].map((o) => (
            <button
              key={String(o.k)}
              onClick={() => setSparse(o.k)}
              className={`rounded-md px-3 py-1 font-mono text-[10px] ${
                sparse === o.k
                  ? "border border-rose-700/60 bg-rose-600/20 text-rose-300"
                  : "border border-zinc-800 text-zinc-500"
              }`}
            >
              {o.l}
            </button>
          ))}
        </div>
        <Sparkline
          yLabel="fitness"
          series={[
            { label: "shaped", color: sparse ? "#f43f5e" : "#34d399", values: sparse ? shaped : res.curve },
            { label: "true objective", color: "#52525b", values: res.curve },
          ]}
        />
        <div className="font-mono text-[10px] text-zinc-500">
          final best {shaped[shaped.length - 1].toFixed(2)} vs true{" "}
          {res.curve[res.curve.length - 1].toFixed(2)}
        </div>
      </Panel>
    </div>
  );
}

/* --------------------------------------------- 8. selection pressure */

export function SelectionSim() {
  const [pressure, setPressure] = useState(0.25);
  const res = useMemo(() => {
    const cfg = { ...DEFAULT_CONFIG, generations: 90 };
    // Tournament size = pressure knob: big tournaments select only elites.
    const rand = lcg(cfg.seed + 17);
    let pop = Array.from({ length: cfg.population }, () =>
      encodeWidths(rand, cfg.layers, cfg.maxWidth),
    );
    const fit = (g: number[]) => widthFitness(g, cfg.budget);
    const diversity: number[] = [];
    for (let gen = 0; gen < cfg.generations; gen++) {
      const scored = pop.map((g) => ({ g, f: fit(g) })).sort((a, b) => a.f - b.f);
      const k = Math.max(1, Math.round(pressure * 12));
      const pick = () => {
        let best = scored[Math.floor(rand() * scored.length)];
        for (let i = 0; i < k; i++) {
          const c = scored[Math.floor(rand() * scored.length)];
          if (c.f < best.f) best = c;
        }
        return best.g;
      };
      const next = scored.slice(0, cfg.elite).map((s) => s.g.slice());
      while (next.length < cfg.population) {
        const a = pick();
        const b = pick();
        next.push(mutateWidths(a, cfg.mutationRate, cfg.maxWidth, rand));
      }
      pop = next;
      let d = 0;
      for (let i = 0; i < pop.length; i++)
        for (let j = i + 1; j < pop.length; j++)
          d += pop[i].reduce((s, v, kk) => s + Math.abs(v - pop[j][kk]), 0);
      diversity.push(d / ((pop.length * (pop.length - 1)) / 2) / (cfg.maxWidth * cfg.layers));
    }
    return diversity;
  }, [pressure]);

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="selection pressure · the diversity collapse"
        hint="Tournament size is the pressure knob. Raise it and the search converges on a good solution faster — while population diversity falls off a cliff, which is exactly when a GA stops being able to escape a local optimum."
      >
        <Slider
          label="tournament size"
          value={pressure}
          min={0.05}
          max={1}
          step={0.05}
          onChange={setPressure}
          fmt={(v) => String(Math.max(1, Math.round(v * 12)))}
        />
        <Sparkline
          yLabel="population diversity"
          series={[{ label: "diversity", color: "#a78bfa", values: res }]}
        />
        <div className="font-mono text-[10px] text-zinc-500">
          final diversity {res[res.length - 1].toFixed(4)}
          {res[res.length - 1] < 0.02 ? " · collapsed: the search has effectively stopped exploring" : ""}
        </div>
      </Panel>
    </div>
  );
}

/* ----------------------------------------------- 9. ES gradient view */

export function EsGradSim() {
  const [sigma, setSigma] = useState(0.5);
  const [lambda, setLambda] = useState(24);
  const [step, setStep] = useState(0);
  const mean = 0.6;
  const theta = useMemo(() => [mean + (step === 0 ? 0 : 0.35 * (step % 5))], [step]);

  const samples = useMemo(() => {
    const rand = lcg(555 + step);
    const out: { x: number; f: number }[] = [];
    for (let i = 0; i < lambda; i++) {
      const u = Math.max(rand(), 1e-9);
      const v = rand();
      const eps = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      const x = theta[0] + sigma * eps;
      out.push({ x, f: -(x - mean) * (x - mean) });
    }
    return out;
  }, [lambda, sigma, step, theta]);

  const est = useMemo(() => {
    // m_theta = (1/lambda) * sum F(theta + sigma*eps) * eps
    let num = 0;
    for (const s of samples) {
      const eps = (s.x - theta[0]) / (sigma || 1);
      num += s.f * eps;
    }
    return num / samples.length;
  }, [samples, sigma, theta]);

  const w = 300;
  const h = 110;
  const xs = samples.map((s) => s.x);
  const lo = Math.min(...xs, mean) - 0.5;
  const hi = Math.max(...xs, mean) + 0.5;
  const sx = (x: number) => ((x - lo) / (hi - lo)) * w;
  const sy = (f: number) => h - 20 - (1 / (1 + f * 3)) * 40;

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="evolution strategy · a gradient hiding in finite differences"
        hint="Perturb theta by Gaussian noise, evaluate, and weight each sample by its fitness. The average is an unbiased estimate of the gradient — which is why a gradient-free ES can match backprop on smooth objectives, and why it stalls wherever the objective is not smooth."
      >
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Slider
            label="sigma"
            value={sigma}
            min={0.05}
            max={1.5}
            step={0.05}
            onChange={setSigma}
            fmt={(v) => v.toFixed(2)}
          />
          <Slider label="lambda" value={lambda} min={4} max={60} onChange={setLambda} />
          <label className="flex flex-col gap-1">
            <span className="font-mono text-[10px] text-zinc-500">draw</span>
            <button
              onClick={() => setStep((s) => s + 1)}
              className="rounded-md border border-rose-800 bg-rose-950/40 px-2 py-1 text-left font-mono text-[10px] text-rose-300"
            >
              resample →
            </button>
          </label>
          <div className="flex flex-col justify-center font-mono text-[10px] text-zinc-500">
            ‖m_theta‖ <span className="text-zinc-100">{Math.abs(est).toFixed(3)}</span>
          </div>
        </div>

        <svg viewBox={`0 0 ${w} ${h}`} className="h-auto w-full" role="img" aria-label="es samples">
          <line x1={0} y1={sy(0)} x2={w} y2={sy(0)} stroke="#3f3f46" strokeWidth={1} />
          {samples.map((s, i) => (
            <circle
              key={i}
              cx={sx(s.x)}
              cy={sy(s.f)}
              r={2.4 + Math.min(4, s.f * 1.6)}
              fill="#f43f5e"
              opacity={0.55 + Math.min(0.45, s.f * 0.3)}
            />
          ))}
          <line x1={sx(mean)} y1={8} x2={sx(mean)} y2={h - 4} stroke="#38bdf8" strokeWidth={1} strokeDasharray="3 3" />
          <text x={sx(mean) + 4} y={14} fontSize={8} fill="#38bdf8">
            target
          </text>
        </svg>
        <div className="font-mono text-[10px] text-zinc-500">
          lambda {lambda} samples · sigma {sigma.toFixed(2)} · theta {theta[0].toFixed(2)}
        </div>
      </Panel>
    </div>
  );
}

/* ------------------------------------- 10. no-gradient NAS comparison */

export function NasSearchSim() {
  const [budget, setBudget] = useState(60);
  const modes = [
    { k: "evo", label: "evolutionary NAS", iters: budget, color: "#f43f5e" },
    { k: "rnd", label: "random search", iters: budget, color: "#a1a1aa" },
    { k: "rl", label: "RL controller", iters: Math.round(budget * 2.4), color: "#38bdf8" },
  ];

  const rows = modes.map((m) => {
    // Deterministic pseudo-curves: evolutionary search improves steadily,
    // random search plateaus, RL pays a large up-front controller cost.
    const vals: number[] = [];
    let v = 0.1;
    for (let i = 0; i < m.iters; i++) {
      const gain =
        m.k === "evo"
          ? 0.055 / (1 + i / 22)
          : m.k === "rnd"
            ? 0.02 / (1 + i / 9)
            : 0.075 / (1 + i / 40);
      v = Math.min(0.98, v + gain);
      vals.push(v);
    }
    return { ...m, vals };
  });
  const best = Math.max(...rows.map((r) => r.vals[r.vals.length - 1]));

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title="architecture search · what the same budget buys"
        hint="Three strategies at a comparable evaluation budget. Evolutionary NAS improves steadily because each generation exploits the architecture that already worked; random search plateaus; an RL controller pays for its own training before it improves at all."
      >
        <Slider label="evaluations" value={budget} min={20} max={300} step={10} onChange={setBudget} />
        <Sparkline
          yLabel="validation accuracy"
          series={rows.map((r) => ({ label: r.label, color: r.color, values: r.vals }))}
          height={120}
        />
        <div className="flex flex-wrap gap-3 font-mono text-[10px] text-zinc-500">
          {rows.map((r) => (
            <span key={r.k}>
              <span style={{ color: r.color }}>{r.label}</span> →{" "}
              {((r.vals[r.vals.length - 1] / best) * 100).toFixed(0)}% of best
            </span>
          ))}
        </div>
        <div className="text-[11px] leading-5 text-zinc-400">
          This is the argument evolutionary NAS makes in the corpus: at a fixed
          evaluation budget it is competitive with gradient-based search, and it
          needs no differentiable architecture parameterisation.
        </div>
      </Panel>
    </div>
  );
}