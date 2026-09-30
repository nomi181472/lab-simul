"use client";

import { useMemo, useState } from "react";

/* Interactive simulator registry for the tracking lab.
 * Every sim is dependency-free (divs + SVG + range inputs) so the lab
 * chunk stays small; views and equations reference these by key.
 */

export const SIMULATOR_IDS = [
  "iou-track",
  "kalman",
  "hungarian",
  "motion",
  "assoc-cost",
  "bytetrack",
  "reid",
  "metrics",
  "corr-filter",
  "siamese",
  "track-mgmt",
] as const;

export type SimulatorId = (typeof SIMULATOR_IDS)[number];

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="flex items-baseline justify-between font-mono text-[10px] text-zinc-500">
        <span className="uppercase tracking-wide">{label}</span>
        <span className="text-sky-300">{format ? format(value) : value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-sky-500"
      />
    </label>
  );
}

function SimFrame({
  title,
  children,
  note,
}: {
  title: string;
  children: React.ReactNode;
  note?: string;
}) {
  return (
    <div className="rounded-xl border border-sky-800/40 bg-zinc-950/60 p-4">
      <div className="font-mono text-[10px] uppercase tracking-widest text-sky-400">
        simulator · {title}
      </div>
      <div className="mt-3">{children}</div>
      {note && <p className="mt-3 text-[11px] leading-5 text-zinc-500">{note}</p>}
    </div>
  );
}

/* ---------- IoU: drag two boxes, live IoU ---------- */
export function IouSim() {
  const [ax, setAx] = useState(30);
  const [ay, setAy] = useState(40);
  const [bx, setBx] = useState(90);
  const [by, setBy] = useState(80);
  const W = 120;
  const H = 90;
  const { inter, union, iou } = useMemo(() => {
    const ix0 = Math.max(ax, bx);
    const iy0 = Math.max(ay, by);
    const ix1 = Math.min(ax + W, bx + W);
    const iy1 = Math.min(ay + H, by + H);
    const iw = Math.max(0, ix1 - ix0);
    const ih = Math.max(0, iy1 - iy0);
    const inter = iw * ih;
    const union = 2 * W * H - inter;
    return { inter, union, iou: union === 0 ? 0 : inter / union };
  }, [ax, ay, bx, by, W, H]);
  return (
    <SimFrame
      title="IoU — move box B over box A"
      note={`Box A @(${ax},${ay}) · Box B @(${bx},${by}) · Intersection ${inter.toLocaleString()} px² · Union ${union.toLocaleString()} px². In SORT every cost-matrix entry is 1 − this number; ByteTrack gates on it twice (high then low score).`}
    >
      <div className="grid gap-4 md:grid-cols-[1fr_220px]">
        <svg viewBox="0 0 360 260" className="w-full rounded-lg border border-zinc-800 bg-black/50">
          <rect x={ax} y={ay} width={W} height={H} fill="none" stroke="#38bdf8" strokeWidth={2} />
          <text x={ax + 4} y={ay + 14} fill="#38bdf8" fontSize={10} fontFamily="monospace">A</text>
          <rect x={bx} y={by} width={W} height={H} fill="rgba(244,114,182,0.25)" stroke="#f472b6" strokeWidth={2} strokeDasharray="5 3" />
          <text x={bx + 4} y={by + 14} fill="#f472b6" fontSize={10} fontFamily="monospace">B</text>
          {inter > 0 && (
            <rect
              x={Math.max(ax, bx)}
              y={Math.max(ay, by)}
              width={Math.max(0, Math.min(ax + W, bx + W) - Math.max(ax, bx))}
              height={Math.max(0, Math.min(ay + H, by + H) - Math.max(ay, by))}
              fill="rgba(52,211,153,0.45)"
              stroke="#34d399"
              strokeWidth={1}
            />
          )}
        </svg>
        <div className="flex flex-col gap-3">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3 text-center">
            <div className="font-mono text-3xl text-emerald-300">{iou.toFixed(3)}</div>
            <div className="mt-1 font-mono text-[10px] text-zinc-500">IoU = inter / union</div>
            <div className="mt-1 font-mono text-[10px] text-zinc-500">cost = {(1 - iou).toFixed(3)}</div>
          </div>
          <Slider label="A.x" value={ax} min={0} max={220} step={2} onChange={setAx} />
          <Slider label="A.y" value={ay} min={0} max={150} step={2} onChange={setAy} />
          <Slider label="B.x" value={bx} min={0} max={220} step={2} onChange={setBx} />
          <Slider label="B.y" value={by} min={0} max={150} step={2} onChange={setBy} />
        </div>
      </div>
    </SimFrame>
  );
}

/* ---------- Kalman: 1-D predict/update walk ---------- */
export function KalmanSim() {
  const [q, setQ] = useState(1);
  const [r, setR] = useState(4);
  const [steps] = useState(40);
  const path = useMemo(() => {
    let seed = 42;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648 - 0.5;
    };
    let x = 0;
    let p = 1;
    let est = 0;
    const pts: { truth: number; meas: number; est: number }[] = [];
    for (let t = 0; t < steps; t++) {
      const truth = t * 1.5 + Math.sin(t / 4) * 4;
      const meas = truth + rand() * 2 * Math.sqrt(r) * 2;
      // predict (constant velocity 1.5)
      const xPred = est + 1.5;
      const pPred = p + q;
      const k = pPred / (pPred + r);
      est = xPred + k * (meas - xPred);
      p = (1 - k) * pPred;
      x = truth;
      pts.push({ truth: x, meas, est });
    }
    return pts;
  }, [q, r, steps]);
  const W = 560;
  const H = 180;
  const maxV = Math.max(...path.map((p) => Math.max(p.truth, p.meas, p.est)));
  const minV = Math.min(...path.map((p) => Math.min(p.truth, p.meas, p.est)));
  const X = (t: number) => 20 + (t / (steps - 1)) * (W - 40);
  const Y = (v: number) => H - 15 - ((v - minV) / Math.max(1e-6, maxV - minV)) * (H - 30);
  const line = (key: "truth" | "meas" | "est") =>
    path.map((p, i) => `${i === 0 ? "M" : "L"}${X(i).toFixed(1)},${Y(p[key]).toFixed(1)}`).join(" ");
  const rmse = Math.sqrt(path.reduce((a, p) => a + (p.est - p.truth) ** 2, 0) / path.length);
  return (
    <SimFrame
      title="Kalman filter — predict then correct"
      note={`Estimate RMSE ${rmse.toFixed(2)} px. Large Q (process noise) → filter chases noisy detections; large R (measurement noise) → filter trusts its prediction and smooths. SORT lives with fixed small Q/R and T_Lost=1 because its constant-velocity Q cannot model turns.`}
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-zinc-800 bg-black/50">
        <path d={line("truth")} fill="none" stroke="#34d399" strokeWidth={2} />
        <path d={line("meas")} fill="none" stroke="#52525b" strokeWidth={1} strokeDasharray="3 3" />
        <path d={line("est")} fill="none" stroke="#38bdf8" strokeWidth={2} />
      </svg>
      <div className="mt-2 flex gap-4 font-mono text-[10px] text-zinc-500">
        <span><span className="text-emerald-300">—</span> true position</span>
        <span><span className="text-zinc-400">┄</span> noisy detection</span>
        <span><span className="text-sky-300">—</span> Kalman estimate</span>
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <Slider label="process noise Q" value={q} min={0.1} max={8} step={0.1} onChange={setQ} />
        <Slider label="measurement noise R" value={r} min={0.5} max={25} step={0.5} onChange={setR} />
      </div>
    </SimFrame>
  );
}

/* ---------- Hungarian: 3x3 assignment ---------- */
const HUNG_COST = [
  [0.2, 0.7, 0.9],
  [0.6, 0.25, 0.8],
  [0.75, 0.65, 0.3],
];

function bruteAssign(cost: number[][]): { perm: number[]; total: number } {
  const perms = [
    [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
  ];
  let best = perms[0];
  let bestC = Infinity;
  for (const p of perms) {
    const c = p.reduce((a, col, row) => a + cost[row][col], 0);
    if (c < bestC) {
      bestC = c;
      best = p;
    }
  }
  return { perm: best, total: bestC };
}

export function HungarianSim() {
  const [gate, setGate] = useState(0.7);
  const { perm, total } = useMemo(() => bruteAssign(HUNG_COST), []);
  const greedy = useMemo(() => {
    const used = new Set<number>();
    const pairs: [number, number][] = [];
    const flat: [number, number, number][] = [];
    HUNG_COST.forEach((row, i) => row.forEach((c, j) => flat.push([c, i, j])));
    flat.sort((a, b) => a[0] - b[0]);
    for (const [c, i, j] of flat) {
      if (c > gate) continue;
      if (pairs.some(([r]) => r === i) || used.has(j)) continue;
      pairs.push([i, j]);
      used.add(j);
    }
    return pairs;
  }, [gate]);
  return (
    <SimFrame
      title="Hungarian assignment on a 3×3 cost matrix"
      note={`Optimal (Hungarian) assignment costs ${total.toFixed(2)}: ${perm.map((c, r) => `T${r + 1}→D${c + 1}`).join(", ")}. Greedy matching under your gate (${gate.toFixed(2)}) picks ${greedy.length ? greedy.map(([r, c]) => `T${r + 1}→D${c + 1}`).join(", ") : "nothing"} — raise the gate and greedy admits riskier pairs, exactly the ByteTrack high/low-threshold trade.`}
    >
      <div className="grid gap-2" style={{ gridTemplateColumns: "auto repeat(3, 1fr)" }}>
        <span />
        {[0, 1, 2].map((j) => (
          <span key={j} className="text-center font-mono text-[10px] text-zinc-500">D{j + 1}</span>
        ))}
        {HUNG_COST.map((row, i) => (
          <>
            <span key={`r${i}`} className="font-mono text-[10px] text-zinc-500">T{i + 1}</span>
            {row.map((c, j) => {
              const opt = perm[i] === j;
              const gr = greedy.some(([r, cc]) => r === i && cc === j);
              return (
                <div
                  key={`${i}-${j}`}
                  className={`rounded border px-2 py-1.5 text-center font-mono text-[12px] ${opt ? "border-sky-500 bg-sky-950/50 text-sky-200" : gr ? "border-amber-600 bg-amber-950/40 text-amber-200" : "border-zinc-800 bg-zinc-900/50 text-zinc-400"} ${c > gate ? "opacity-50" : ""}`}
                >
                  {c.toFixed(2)}
                </div>
              );
            })}
          </>
        ))}
      </div>
      <div className="mt-3">
        <Slider label="gating threshold" value={gate} min={0.2} max={1} step={0.05} onChange={setGate} format={(v) => v.toFixed(2)} />
      </div>
    </SimFrame>
  );
}

/* ---------- Motion prediction ---------- */
export function MotionSim() {
  const [vx, setVx] = useState(6);
  const [vy, setVy] = useState(2);
  const [dt, setDt] = useState(3);
  const x0 = 60;
  const y0 = 90;
  const x1 = x0 + vx * dt * 6;
  const y1 = y0 + vy * dt * 6;
  return (
    <SimFrame
      title="Constant-velocity prediction"
      note={`Predicted box = (${x1.toFixed(0)}, ${y1.toFixed(0)}) after Δt=${dt}. SORT assumes this linear rule; OC-SORT and MambaMOT exist because real motion (dance, sport, UAV turns) breaks it within a few frames.`}
    >
      <svg viewBox="0 0 560 200" className="w-full rounded-lg border border-zinc-800 bg-black/50">
        <line x1={x0} y1={y0} x2={x1} y2={y1} stroke="#38bdf8" strokeWidth={1.5} strokeDasharray="5 4" />
        <rect x={x0 - 25} y={y0 - 18} width={50} height={36} fill="none" stroke="#34d399" strokeWidth={2} />
        <text x={x0 - 25} y={y0 - 24} fill="#34d399" fontSize={10} fontFamily="monospace">t</text>
        <rect x={x1 - 25} y={y1 - 18} width={50} height={36} fill="rgba(56,189,248,0.15)" stroke="#38bdf8" strokeWidth={2} />
        <text x={x1 - 25} y={y1 - 24} fill="#38bdf8" fontSize={10} fontFamily="monospace">t+Δt</text>
      </svg>
      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        <Slider label="velocity vx" value={vx} min={-10} max={12} step={0.5} onChange={setVx} />
        <Slider label="velocity vy" value={vy} min={-8} max={8} step={0.5} onChange={setVy} />
        <Slider label="Δt frames" value={dt} min={1} max={8} step={1} onChange={setDt} />
      </div>
    </SimFrame>
  );
}

/* ---------- Association cost fusion ---------- */
export function AssocCostSim() {
  const [l1, setL1] = useState(0.5);
  const [l2, setL2] = useState(0.3);
  const [l3, setL3] = useState(0.2);
  const motion = 0.35;
  const appear = 0.6;
  const iouC = 0.25;
  const sum = l1 + l2 + l3 || 1;
  const total = (l1 * motion + l2 * appear + l3 * iouC) / sum;
  return (
    <SimFrame
      title="Association cost fusion"
      note={`Total cost ${total.toFixed(3)} with normalized weights. DeepSORT adds the appearance term SORT lacks (fixing ID switches); StrongSORT re-weights it with stronger Re-ID; motion-only trackers (OC-SORT) push λ₂→0.`}
    >
      <Formula>
        C = {(l1 / sum).toFixed(2)}·{motion} (motion) + {(l2 / sum).toFixed(2)}·{appear} (appearance) + {(l3 / sum).toFixed(2)}·{iouC} (1−IoU) = {total.toFixed(3)}
      </Formula>
      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        <Slider label="λ motion" value={l1} min={0} max={1} step={0.05} onChange={setL1} />
        <Slider label="λ appearance" value={l2} min={0} max={1} step={0.05} onChange={setL2} />
        <Slider label="λ IoU" value={l3} min={0} max={1} step={0.05} onChange={setL3} />
      </div>
    </SimFrame>
  );
}

/* ---------- ByteTrack two-stage ---------- */
export function ByteTrackSim() {
  const [tauHigh, setTauHigh] = useState(0.6);
  const [tauLow, setTauLow] = useState(0.2);
  const dets = [0.92, 0.81, 0.65, 0.45, 0.33, 0.18, 0.09];
  const high = dets.filter((d) => d >= tauHigh);
  const low = dets.filter((d) => d >= tauLow && d < tauHigh);
  const junk = dets.filter((d) => d < tauLow);
  return (
    <SimFrame
      title="ByteTrack dual-threshold association"
      note={`${high.length} high-score boxes associate first; ${low.length} low-score boxes rescue occluded tracks instead of being discarded; ${junk.length} rejected as background. Drag τ_low up and occluded people become misses (FN↑); drag it down and ghosts appear (FP↑).`}
    >
      <div className="flex flex-wrap gap-1.5">
        {dets.map((d, i) => (
          <div
            key={i}
            className={`rounded-lg border px-3 py-2 font-mono text-[12px] ${d >= tauHigh ? "border-sky-500 bg-sky-950/50 text-sky-200" : d >= tauLow ? "border-amber-600 bg-amber-950/40 text-amber-200" : "border-zinc-800 bg-zinc-900/50 text-zinc-600 line-through"}`}
          >
            {d.toFixed(2)}
          </div>
        ))}
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <Slider label="τ high" value={tauHigh} min={0.3} max={0.9} step={0.05} onChange={setTauHigh} format={(v) => v.toFixed(2)} />
        <Slider label="τ low" value={tauLow} min={0.05} max={0.5} step={0.05} onChange={setTauLow} format={(v) => v.toFixed(2)} />
      </div>
    </SimFrame>
  );
}

/* ---------- Re-ID cosine ---------- */
export function ReIdSim() {
  const [sim, setSim] = useState(0.75);
  const cost = 1 - sim;
  const match = sim > 0.5;
  return (
    <SimFrame
      title="Re-ID appearance distance"
      note={`cosine similarity ${sim.toFixed(2)} → cost ${cost.toFixed(2)} → ${match ? "SAME identity (link across the occlusion gap)" : "DIFFERENT identity (new track)"}. DeepSORT's 128-D embedding exists to make this number meaningful after motion fails.`}
    >
      <div className="flex items-center gap-3">
        <div className="h-12 w-12 rounded-lg border border-sky-600 bg-gradient-to-br from-sky-800 to-sky-950" />
        <div className="flex-1">
          <div className="h-2 overflow-hidden rounded bg-zinc-800">
            <div className="h-full bg-sky-500" style={{ width: `${sim * 100}%` }} />
          </div>
          <div className="mt-1 font-mono text-[10px] text-zinc-500">cosine(f_det, f_track) = {sim.toFixed(2)}</div>
        </div>
        <div className="h-12 w-12 rounded-lg border border-pink-600 bg-gradient-to-br from-pink-800 to-pink-950" />
      </div>
      <div className="mt-3">
        <Slider label="appearance similarity" value={sim} min={0} max={1} step={0.01} onChange={setSim} format={(v) => v.toFixed(2)} />
      </div>
    </SimFrame>
  );
}

/* ---------- MOTA playground ---------- */
export function MetricsSim() {
  const [fp, setFp] = useState(7318);
  const [fn, setFn] = useState(32615);
  const [idsw, setIdsw] = useState(1001);
  const gt = 120000;
  const mota = 1 - (fn + fp + idsw) / gt;
  return (
    <SimFrame
      title="MOTA error budget (SORT numbers preloaded)"
      note={`MOTA = 1 − (FN+FP+IDSW)/GT = ${(mota * 100).toFixed(1)}. FN dominates: halving ID switches barely moves the score, which is why the MOT16 paper itself calls MOTA "highly debatable" as a single measure — and why HOTA/IDF1 had to be invented.`}
    >
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3 text-center">
        <div className="font-mono text-3xl text-sky-300">{(mota * 100).toFixed(1)}</div>
        <div className="font-mono text-[10px] text-zinc-500">MOTA (GT fixed at {gt.toLocaleString()})</div>
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        <Slider label="FP" value={fp} min={0} max={20000} step={100} onChange={setFp} />
        <Slider label="FN" value={fn} min={0} max={60000} step={200} onChange={setFn} />
        <Slider label="IDSW" value={idsw} min={0} max={5000} step={10} onChange={setIdsw} />
      </div>
    </SimFrame>
  );
}

/* ---------- Correlation filter response ---------- */
export function CorrFilterSim() {
  const [bandwidth, setBandwidth] = useState(2);
  const W = 300;
  const H = 90;
  const pts = useMemo(() => {
    const out: string[] = [];
    for (let x = 0; x <= W; x += 3) {
      const c = W / 2;
      const y = H - 10 - 65 * Math.exp(-((x - c) ** 2) / (2 * (bandwidth * 12) ** 2));
      out.push(`${x === 0 ? "M" : "L"}${x},${y.toFixed(1)}`);
    }
    return out.join(" ");
  }, [bandwidth]);
  return (
    <SimFrame
      title="Correlation filter response peak"
      note="The filter is trained so every cyclic shift of the target scores a Gaussian bump peaked at zero shift. Narrow bandwidth → sharp but brittle peak (KCF+HOG); wide → tolerant but drift-prone. Detection = argmax of this surface."
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-zinc-800 bg-black/50">
        <path d={pts} fill="none" stroke="#38bdf8" strokeWidth={2} />
      </svg>
      <div className="mt-3">
        <Slider label="gaussian bandwidth" value={bandwidth} min={0.5} max={5} step={0.1} onChange={setBandwidth} />
      </div>
    </SimFrame>
  );
}

/* ---------- Siamese score map ---------- */
export function SiameseSim() {
  const [tx, setTx] = useState(50);
  const [sx, setSx] = useState(55);
  const diff = Math.abs(tx - sx);
  const score = Math.exp(-(diff ** 2) / 800);
  return (
    <SimFrame
      title="Siamese similarity: template vs search"
      note={`Similarity ${score.toFixed(3)} — SiamFC cross-correlates the template embedding with every search location and takes the argmax. Move the target away from the template position and watch the score collapse; distractors (DaSiamRPN) are extra peaks the loss must suppress.`}
    >
      <svg viewBox="0 0 300 110" className="w-full rounded-lg border border-zinc-800 bg-black/50">
        <rect x={20} y={30} width={50} height={50} fill="none" stroke="#34d399" strokeWidth={2} />
        <text x={20} y={22} fill="#34d399" fontSize={9} fontFamily="monospace">template @{tx}</text>
        <rect x={150} y={30} width={110} height={50} fill="none" stroke="#52525b" strokeWidth={1} />
        <rect x={150 + sx} y={38} width={22} height={34} fill={`rgba(56,189,248,${0.2 + score * 0.6})`} stroke="#38bdf8" strokeWidth={2} />
        <text x={150} y={22} fill="#71717a" fontSize={9} fontFamily="monospace">search, score {score.toFixed(2)}</text>
      </svg>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <Slider label="template pos" value={tx} min={0} max={100} step={1} onChange={setTx} />
        <Slider label="target pos in search" value={sx} min={0} max={88} step={1} onChange={setSx} />
      </div>
    </SimFrame>
  );
}

/* ---------- Track management lifecycle ---------- */
export function TrackMgmtSim() {
  const [maxAge, setMaxAge] = useState(3);
  const [minHits, setMinHits] = useState(2);
  const frames = ["det", "det", "miss", "miss", "miss", "miss", "det"];
  let age = 0;
  let hits = 0;
  let alive = false;
  const states = frames.map((f) => {
    if (f === "det") {
      hits += 1;
      age = 0;
      if (hits >= minHits) alive = true;
    } else {
      age += 1;
      if (age > maxAge) {
        alive = false;
        hits = 0;
      }
    }
    return alive ? "alive" : "dead";
  });
  return (
    <SimFrame
      title="Track birth / death rules"
      note={`min_hits=${minHits}, max_age=${maxAge}. SORT uses max_age=1 (dies at the first miss — no re-ID); DeepSORT/StrongSORT survive gaps with appearance. Short max_age → fragmentation; long → ghosts.`}
    >
      <div className="flex flex-wrap gap-1.5">
        {frames.map((f, i) => (
          <div
            key={i}
            className={`rounded-lg border px-2.5 py-2 text-center font-mono text-[10px] ${states[i] === "alive" ? "border-emerald-600 bg-emerald-950/40 text-emerald-200" : "border-zinc-800 bg-zinc-900/50 text-zinc-600"}`}
          >
            <div>f{i + 1}</div>
            <div>{f}</div>
            <div className="mt-0.5 text-[9px]">{states[i]}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <Slider label="max age" value={maxAge} min={0} max={5} step={1} onChange={setMaxAge} />
        <Slider label="min hits" value={minHits} min={1} max={4} step={1} onChange={setMinHits} />
      </div>
    </SimFrame>
  );
}

function Formula({ children }: { children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-zinc-800 bg-black/40 px-3 py-2 font-mono text-[12px] leading-6 text-amber-200">
      {children}
    </div>
  );
}

export const SIMULATORS: Record<SimulatorId, React.ComponentType> = {
  "iou-track": IouSim,
  kalman: KalmanSim,
  hungarian: HungarianSim,
  motion: MotionSim,
  "assoc-cost": AssocCostSim,
  bytetrack: ByteTrackSim,
  reid: ReIdSim,
  metrics: MetricsSim,
  "corr-filter": CorrFilterSim,
  siamese: SiameseSim,
  "track-mgmt": TrackMgmtSim,
};

export const SIMULATOR_META: { id: SimulatorId; label: string; blurb: string }[] = [
  { id: "iou-track", label: "IoU boxes", blurb: "Drag boxes, watch IoU and cost" },
  { id: "kalman", label: "Kalman filter", blurb: "Q/R trade: agile vs smooth" },
  { id: "hungarian", label: "Hungarian", blurb: "Optimal matching vs greedy gates" },
  { id: "motion", label: "Motion prediction", blurb: "Constant-velocity rollout" },
  { id: "assoc-cost", label: "Cost fusion", blurb: "λ₁ motion + λ₂ appearance + λ₃ IoU" },
  { id: "bytetrack", label: "ByteTrack gates", blurb: "Two-stage τ_high / τ_low" },
  { id: "reid", label: "Re-ID distance", blurb: "Cosine similarity → identity" },
  { id: "metrics", label: "MOTA budget", blurb: "FP/FN/IDSW error arithmetic" },
  { id: "corr-filter", label: "CF response", blurb: "Gaussian peak sharpness" },
  { id: "siamese", label: "Siamese score", blurb: "Template–search similarity" },
  { id: "track-mgmt", label: "Track lifecycle", blurb: "Birth, probation, death" },
];

export function SimShell({ id }: { id: string }) {
  const Comp = (SIMULATORS as Record<string, React.ComponentType>)[id] ?? IouSim;
  return <Comp />;
}
