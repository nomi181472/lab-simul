"use client";

import { useEffect, useRef, useState } from "react";

/* Run `step` at ~30fps while `running`. The callback ref is written inside an
   effect so the React compiler never sees a ref mutation during render. */
function useLoop(step: (t: number) => void, running: boolean) {
  const cb = useRef(step);
  useEffect(() => {
    cb.current = step;
  });
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const tick = (now: number) => {
      acc += now - last;
      last = now;
      while (acc >= 33) {
        cb.current(1);
        acc -= 33;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running]);
}

function Canvas({
  draw,
}: {
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  useEffect(() => {
    drawRef.current = draw;
  });
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    const paint = () => {
      drawRef.current(ctx, cv.width, cv.height);
      raf = requestAnimationFrame(paint);
    };
    raf = requestAnimationFrame(paint);
    return () => cancelAnimationFrame(raf);
  }, []);
  return (
    <canvas
      ref={ref}
      width={640}
      height={220}
      className="w-full rounded-lg border border-zinc-800 bg-black/50"
    />
  );
}

export function Slider({
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
    <label className="flex items-center gap-2 text-[11px] text-zinc-400">
      <span className="w-40 shrink-0">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 accent-sky-500"
      />
      <span className="w-16 text-right font-mono text-[10px] text-zinc-300">
        {fmt ? fmt(value) : value}
      </span>
    </label>
  );
}

export function Stat({
  label,
  value,
  tone = "zinc",
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  const tones: Record<string, string> = {
    zinc: "text-zinc-100",
    cyan: "text-sky-300",
    emerald: "text-emerald-300",
    amber: "text-amber-300",
    rose: "text-rose-300",
  };
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2">
      <p className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">{label}</p>
      <p className={`font-mono text-[15px] ${tones[tone] ?? tones.zinc}`}>{value}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * 1. Congestion control: bottleneck queue, AIMD vs delay-based vs RL
 * ------------------------------------------------------------------ */
type CCState = { cwnd: number[]; q: number[]; tput: number[]; t: number; drops: number };

export function CongestionControlSim() {
  const [algo, setAlgo] = useState<"aimd" | "delay" | "rl">("aimd");
  const [load, setLoad] = useState(10);
  const [s, setS] = useState<CCState>(() => ({
    cwnd: [2, 2, 2],
    q: [0, 0, 0],
    tput: [0, 0, 0],
    t: 0,
    drops: 0,
  }));

  useLoop(() => {
    setS((prev) => {
      const next: CCState = {
        cwnd: [...prev.cwnd],
        q: [...prev.q],
        tput: [...prev.tput],
        t: prev.t + 1,
        drops: prev.drops,
      };
      const capacity = 18;
      for (let i = 0; i < 3; i++) {
        let c = next.cwnd[i];
        const offered = c;
        const share = Math.min(offered, capacity / 3);
        next.q[i] = Math.max(0, next.q[i] + offered - share);
        next.tput[i] = share;
        if (algo === "aimd") {
          if (next.q[i] > 8) c = Math.max(1, c * 0.7);
          else c = Math.min(load, c + 0.3);
        } else if (algo === "delay") {
          if (next.q[i] > 2) c = Math.max(1, c - 0.25);
          else c = Math.min(load, c + 0.2);
        } else {
          const reward = share - 0.5 * next.q[i];
          c += reward > 0 ? 0.25 : -0.2;
          c = Math.min(load, Math.max(1, c + (Math.random() - 0.5) * 0.3));
        }
        next.cwnd[i] = c;
      }
      if (next.q.reduce((a, b) => a + b, 0) > 30) next.drops += 1;
      return next;
    });
  }, true);

  const totalTput = s.tput.reduce((a, b) => a + b, 0);
  const totalQ = s.q.reduce((a, b) => a + b, 0);
  const fair = totalTput > 0 ? (Math.min(...s.tput) / (totalTput / 3)) || 0 : 1;

  return (
    <div className="space-y-3">
      <div className="flex gap-1">
        {(["aimd", "delay", "rl"] as const).map((a) => (
          <button
            key={a}
            onClick={() => setAlgo(a)}
            className={`rounded-md border px-2 py-1 font-mono text-[10px] ${
              algo === a
                ? "border-sky-700/50 bg-sky-600/20 text-sky-300"
                : "border-zinc-700 bg-zinc-900 text-zinc-400"
            }`}
          >
            {a === "aimd"
              ? "AIMD (TCP Reno-like)"
              : a === "delay"
                ? "Delay-based (BBR-like)"
                : "RL-style policy"}
          </button>
        ))}
      </div>
      <Slider label="per-sender window cap" value={load} min={2} max={40} onChange={setLoad} />
      <Canvas
        draw={(ctx, w, h) => {
          ctx.clearRect(0, 0, w, h);
          const names = ["A", "B", "C"];
          for (let i = 0; i < 3; i++) {
            const y = 30 + i * 60;
            const bw = (s.cwnd[i] / 40) * (w - 120);
            const qw = Math.min(1, s.q[i] / 16) * (w - 120);
            ctx.fillStyle = "#0ea5e9";
            ctx.fillRect(60, y, Math.max(2, bw), 14);
            ctx.fillStyle = "#b45309";
            ctx.fillRect(60, y + 18, Math.max(0, qw), 8);
            ctx.fillStyle = "#a1a1aa";
            ctx.font = "11px monospace";
            ctx.fillText(
              `${names[i]} cwnd=${s.cwnd[i].toFixed(1)} q=${s.q[i].toFixed(1)}`,
              4,
              y + 12,
            );
          }
          ctx.fillStyle = "#71717a";
          ctx.fillText(`bottleneck drains 18 pkt/tick · drops: ${s.drops}`, 60, h - 12);
        }}
      />
      <div className="grid grid-cols-3 gap-2">
        <Stat label="aggregate goodput" value={totalTput.toFixed(1)} tone="emerald" />
        <Stat
          label="total queue"
          value={totalQ.toFixed(1)}
          tone={totalQ > 20 ? "rose" : "amber"}
        />
        <Stat
          label="fairness (min/avg)"
          value={fair.toFixed(2)}
          tone={fair > 0.9 ? "emerald" : "rose"}
        />
      </div>
      <p className="text-[11px] leading-5 text-zinc-500">
        Models the cluster-core problem quoted under{" "}
        <span className="text-sky-300">Congestion Control</span>: three senders, one
        bottleneck. AIMD reacts to loss (queue spikes first), delay-based keeps queues
        short but can under-utilise, RL-style trades throughput against delay with noise.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------- *
 * 2. Routing: shortest-path vs traffic-engineered spreading
 * ------------------------------------------------------------ */
type RtState = { upper: number; lower: number; t: number; overloads: number };

export function RoutingSim() {
  const [mode, setMode] = useState<"sp" | "te">("sp");
  const [demand, setDemand] = useState(6);
  const [s, setS] = useState<RtState>({ upper: 0, lower: 0, t: 0, overloads: 0 });

  useLoop(() => {
    setS((prev) => {
      let up: number;
      let lo: number;
      if (mode === "sp") {
        up = demand;
        lo = 0;
      } else {
        up = Math.min(demand, 6);
        lo = demand - up;
      }
      const upper = prev.upper * 0.6 + up * 0.4;
      const lower = prev.lower * 0.6 + lo * 0.4;
      return {
        upper,
        lower,
        t: prev.t + 1,
        overloads: prev.overloads + (upper > 6 ? 1 : 0),
      };
    });
  }, true);

  const maxLoad = 14;
  return (
    <div className="space-y-3">
      <div className="flex gap-1">
        {(["sp", "te"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded-md border px-2 py-1 font-mono text-[10px] ${
              mode === m
                ? "border-sky-700/50 bg-sky-600/20 text-sky-300"
                : "border-zinc-700 bg-zinc-900 text-zinc-400"
            }`}
          >
            {m === "sp" ? "shortest path only" : "traffic-engineered split"}
          </button>
        ))}
      </div>
      <Slider label="offered demand" value={demand} min={1} max={24} onChange={setDemand} />
      <Canvas
        draw={(ctx, w, h) => {
          ctx.clearRect(0, 0, w, h);
          const nodes: [number, number][] = [
            [50, 110],
            [260, 50],
            [260, 170],
            [430, 50],
            [430, 170],
            [590, 110],
          ];
          ctx.strokeStyle = "#3f3f46";
          ctx.lineWidth = 1;
          const edges: [number, number][] = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 5],
          ];
          for (const [a, b] of edges) {
            ctx.beginPath();
            ctx.moveTo(...nodes[a]);
            ctx.lineTo(...nodes[b]);
            ctx.stroke();
          }
          const upRatio = Math.min(1, s.upper / maxLoad);
          const loRatio = Math.min(1, s.lower / maxLoad);
          ctx.strokeStyle = upRatio > 0.99 ? "#f43f5e" : "#0ea5e9";
          ctx.lineWidth = 2 + upRatio * 8;
          ctx.beginPath();
          ctx.moveTo(...nodes[0]);
          ctx.lineTo(...nodes[1]);
          ctx.lineTo(...nodes[3]);
          ctx.lineTo(...nodes[5]);
          ctx.stroke();
          ctx.strokeStyle = loRatio > 0.99 ? "#f43f5e" : "#10b981";
          ctx.lineWidth = 2 + loRatio * 8;
          ctx.beginPath();
          ctx.moveTo(...nodes[0]);
          ctx.lineTo(...nodes[2]);
          ctx.lineTo(...nodes[4]);
          ctx.lineTo(...nodes[5]);
          ctx.stroke();
          for (let i = 0; i < nodes.length; i++) {
            ctx.fillStyle = "#18181b";
            ctx.beginPath();
            ctx.arc(nodes[i][0], nodes[i][1], 12, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "#52525b";
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = "#d4d4d8";
            ctx.fillText(String(i), nodes[i][0] - 3, nodes[i][1] + 4);
          }
          ctx.fillStyle = "#71717a";
          ctx.fillText(
            `upper ${s.upper.toFixed(1)}/6 · lower ${s.lower.toFixed(1)}/10`,
            130,
            h - 10,
          );
        }}
      />
      <div className="grid grid-cols-2 gap-2">
        <Stat
          label="upper-path load"
          value={`${s.upper.toFixed(1)} / 6`}
          tone={s.upper > 6 ? "rose" : "cyan"}
        />
        <Stat
          label="overload ticks"
          value={String(s.overloads)}
          tone={s.overloads > 0 ? "rose" : "emerald"}
        />
      </div>
      <p className="text-[11px] leading-5 text-zinc-500">
        Models the routing cluster: shortest-path funnels demand onto one path until it
        overloads; traffic engineering splits flows across disjoint paths (the ECMP/TE
        idea the corpus&apos;s routing papers argue over).
      </p>
    </div>
  );
}

/* ---------------------------------------------------------- *
 * 3. CDN caching: hit rate vs cache size, LRU vs LFU
 * --------------------------------------------------------- */
type CdnState = {
  hits: number;
  misses: number;
  cache: string[];
  freq: Record<string, number>;
  t: number;
};

function zipfSum(s: number) {
  let x = 0;
  for (let i = 1; i <= 100; i++) x += 1 / Math.pow(i, s);
  return x;
}

export function CdnCachingSim() {
  const [size, setSize] = useState(20);
  const [policy, setPolicy] = useState<"lru" | "lfu">("lru");
  const [zipf, setZipf] = useState(1.1);
  const [s, setS] = useState<CdnState>(() => ({
    hits: 0,
    misses: 0,
    cache: [],
    freq: {},
    t: 0,
  }));

  useLoop(() => {
    setS((prev) => {
      const next: CdnState = {
        hits: prev.hits,
        misses: prev.misses,
        cache: [...prev.cache],
        freq: { ...prev.freq },
        t: prev.t + 1,
      };
      let r = Math.random() * zipfSum(zipf);
      let obj = 0;
      for (let i = 0; i < 100; i++) {
        r -= 1 / Math.pow(i + 1, zipf);
        if (r <= 0) {
          obj = i;
          break;
        }
      }
      const key = String(obj);
      const at = next.cache.indexOf(key);
      if (at >= 0) {
        next.hits += 1;
        if (policy === "lru") {
          next.cache.splice(at, 1);
          next.cache.push(key);
        } else {
          next.freq[key] = (next.freq[key] ?? 0) + 1;
        }
      } else {
        next.misses += 1;
        if (next.cache.length >= size) {
          if (policy === "lru") {
            const oldest = next.cache.shift();
            if (oldest) delete next.freq[oldest];
          } else {
            let worst: string | null = null;
            let worstN = Infinity;
            for (const [k, n] of Object.entries(next.freq))
              if (n < worstN) {
                worstN = n;
                worst = k;
              }
            if (worst) {
              next.cache = next.cache.filter((c) => c !== worst);
              delete next.freq[worst];
            }
          }
        }
        next.cache.push(key);
        next.freq[key] = 1;
      }
      if (next.hits + next.misses > 2000) {
        next.hits = 0;
        next.misses = 0;
      }
      return next;
    });
  }, true);

  const total = s.hits + s.misses;
  const hitRate = total ? (s.hits / total) * 100 : 0;

  return (
    <div className="space-y-3">
      <div className="flex gap-1">
        {(["lru", "lfu"] as const).map((p) => (
          <button
            key={p}
            onClick={() => setPolicy(p)}
            className={`rounded-md border px-2 py-1 font-mono text-[10px] ${
              policy === p
                ? "border-sky-700/50 bg-sky-600/20 text-sky-300"
                : "border-zinc-700 bg-zinc-900 text-zinc-400"
            }`}
          >
            {p === "lru" ? "LRU" : "LFU"}
          </button>
        ))}
      </div>
      <Slider
        label="cache size (objects)"
        value={size}
        min={2}
        max={80}
        onChange={setSize}
      />
      <Slider
        label="zipf skew"
        value={zipf}
        min={0.2}
        max={2}
        step={0.1}
        onChange={setZipf}
        fmt={(v) => v.toFixed(1)}
      />
      <Canvas
        draw={(ctx, w, h) => {
          ctx.clearRect(0, 0, w, h);
          const bw = Math.min(560, (hitRate / 100) * 560);
          ctx.fillStyle = "#10b981";
          ctx.fillRect(40, 60, bw, 40);
          ctx.fillStyle = "#3f3f46";
          ctx.fillRect(40 + bw, 60, 560 - bw, 40);
          ctx.fillStyle = "#d4d4d8";
          ctx.font = "12px monospace";
          ctx.fillText(`hit rate ${hitRate.toFixed(1)}%`, 40, 45);
          ctx.fillStyle = "#71717a";
          ctx.fillText(`objects 0..99 · ${s.cache.length}/${size} cached`, 40, 130);
          for (let i = 0; i < size; i++) {
            ctx.fillStyle = i < s.cache.length ? "#0ea5e9" : "#27272a";
            ctx.fillRect(40 + i * 7, 150, 5, 20);
          }
        }}
      />
      <div className="grid grid-cols-3 gap-2">
        <Stat label="hit rate" value={`${hitRate.toFixed(1)}%`} tone="emerald" />
        <Stat label="hits / misses" value={`${s.hits} / ${s.misses}`} />
        <Stat label="policy" value={policy.toUpperCase()} tone="cyan" />
      </div>
      <p className="text-[11px] leading-5 text-zinc-500">
        Models the CDN/caching cluster: popularity is Zipf; raising skew makes any policy
        look good, and the LRU-vs-LFU difference only appears at small cache sizes with
        flat popularity — the boundary the corpus&apos;s caching papers map.
      </p>
    </div>
  );
}

/* ------------------------------------------------------- *
 * 4. IoT random access: ALOHA-style collisions
 * ------------------------------------------------------ */
type IotState = { attempts: number; success: number; backlog: number };

export function IotAccessSim() {
  const [n, setN] = useState(50);
  const [rate, setRate] = useState(3);
  const [s, setS] = useState<IotState>({ attempts: 0, success: 0, backlog: 0 });

  useLoop(() => {
    setS((prev) => {
      const attempting = Math.min(prev.backlog + n * (rate / 100), n);
      const delivered =
        attempting <= 1
          ? Math.floor(attempting)
          : Math.round(attempting * Math.exp(-attempting * 0.35));
      let attempts = prev.attempts + Math.round(attempting);
      let success = prev.success + delivered;
      const backlog = Math.max(0, Math.round((attempting - delivered) * 0.6));
      if (attempts > 20000) {
        attempts = 0;
        success = 0;
      }
      return { attempts, success, backlog };
    });
  }, true);

  const eff = s.attempts ? (s.success / s.attempts) * 100 : 0;

  return (
    <div className="space-y-3">
      <Slider label="devices on the channel" value={n} min={5} max={300} onChange={setN} />
      <Slider
        label="load (% attempting / tick)"
        value={rate}
        min={1}
        max={40}
        onChange={setRate}
      />
      <Canvas
        draw={(ctx, w, h) => {
          ctx.clearRect(0, 0, w, h);
          const cols = 30;
          const rows = 10;
          for (let i = 0; i < Math.min(n, cols * rows); i++) {
            const x = 30 + (i % cols) * 20;
            const y = 25 + Math.floor(i / cols) * 18;
            const colliding = i % 7 === s.attempts % 7;
            ctx.fillStyle = colliding ? "#f43f5e" : "#10b981";
            ctx.fillRect(x, y, 12, 10);
          }
          ctx.fillStyle = "#71717a";
          ctx.font = "11px monospace";
          ctx.fillText("green = delivered · red = collided (snapshot)", 30, h - 14);
        }}
      />
      <div className="grid grid-cols-3 gap-2">
        <Stat
          label="channel efficiency"
          value={`${eff.toFixed(1)}%`}
          tone={eff > 50 ? "emerald" : "rose"}
        />
        <Stat label="attempts" value={s.attempts.toLocaleString()} />
        <Stat label="delivered" value={s.success.toLocaleString()} tone="cyan" />
      </div>
      <p className="text-[11px] leading-5 text-zinc-500">
        Models the IoT massive-access cluster: as more devices contend, collisions eat the
        channel — the reason the corpus&apos;s IoT papers move from pure ALOHA toward
        grant-free patterns and backoff redesign.
      </p>
    </div>
  );
}

/* ---------------------------------------------- *
 * 5. M/M/1 queue: utilisation vs delay
 * --------------------------------------------- */
type QState = { arrivals: number; served: number; q: number; sumWait: number; t: number };

export function QueueingSim() {
  const [rho, setRho] = useState(0.7);
  const [service, setService] = useState(10);
  const [s, setS] = useState<QState>({
    arrivals: 0,
    served: 0,
    q: 0,
    sumWait: 0,
    t: 0,
  });

  useLoop(() => {
    setS((prev) => {
      const lambda = rho * service;
      let q = prev.q;
      for (let i = 0; i < Math.round(lambda / 10); i++) if (Math.random() < 0.9) q += 1;
      const drain = Math.min(q, Math.round(service / 10));
      q -= drain;
      const t = prev.t + 1;
      return {
        arrivals: prev.arrivals + Math.round(lambda / 10),
        served: prev.served + drain,
        q,
        sumWait: prev.sumWait + q,
        t,
      };
    });
  }, true);

  const meanQ = s.sumWait / Math.max(1, s.t);
  const theory = rho / Math.max(0.001, 1 - rho);

  return (
    <div className="space-y-3">
      <Slider
        label="utilisation ρ"
        value={rho}
        min={0.05}
        max={0.98}
        step={0.01}
        onChange={setRho}
        fmt={(v) => v.toFixed(2)}
      />
      <Slider label="service rate" value={service} min={2} max={40} onChange={setService} />
      <Canvas
        draw={(ctx, w, h) => {
          ctx.clearRect(0, 0, w, h);
          ctx.strokeStyle = "#f59e0b";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          for (let px = 0; px <= 560; px++) {
            const r = (px / 560) * 0.99;
            const L = Math.min(20, r / Math.max(0.01, 1 - r));
            const y = h - 30 - (L / 20) * (h - 60);
            if (px === 0) ctx.moveTo(40 + px, y);
            else ctx.lineTo(40 + px, y);
          }
          ctx.stroke();
          ctx.strokeStyle = "#0ea5e9";
          ctx.beginPath();
          const Lobs = Math.min(20, meanQ);
          const y = h - 30 - (Lobs / 20) * (h - 60);
          ctx.moveTo(40, y);
          ctx.lineTo(600, y);
          ctx.stroke();
          ctx.fillStyle = "#71717a";
          ctx.font = "11px monospace";
          ctx.fillText("ρ →", 590, h - 10);
          ctx.fillText("L (mean queue)", 44, 24);
          ctx.fillStyle = "#f59e0b";
          ctx.fillText("theory ρ/(1−ρ)", 44, 40);
          ctx.fillStyle = "#0ea5e9";
          ctx.fillText("observed", 44, 56);
        }}
      />
      <div className="grid grid-cols-3 gap-2">
        <Stat label="observed mean queue" value={meanQ.toFixed(2)} tone="cyan" />
        <Stat label="theory ρ/(1−ρ)" value={theory.toFixed(2)} tone="amber" />
        <Stat label="served" value={s.served.toLocaleString()} tone="emerald" />
      </div>
      <p className="text-[11px] leading-5 text-zinc-500">
        The queueing-theory analogy from the cross-domain view: delay explodes as ρ→1 no
        matter the protocol — which is why delay-based transports run queues near empty.
      </p>
    </div>
  );
}

/* ------------------------------------------------------ *
 * 6. Intrusion detection: threshold on flow score
 * ----------------------------------------------------- */
type IdsState = { tp: number; fp: number; fn: number; tn: number; t: number };

export function AnomalyDetectionSim() {
  const [thr, setThr] = useState(50);
  const [s, setS] = useState<IdsState>({ tp: 0, fp: 0, fn: 0, tn: 0, t: 0 });

  useLoop(() => {
    setS((prev) => {
      const isAttack = Math.random() < 0.12;
      const base = isAttack ? 65 : 30;
      const score = Math.max(
        0,
        Math.min(
          100,
          base + (Math.random() + Math.random() + Math.random() - 1.5) * 30,
        ),
      );
      const alarm = score >= thr;
      const next: IdsState = {
        tp: prev.tp,
        fp: prev.fp,
        fn: prev.fn,
        tn: prev.tn,
        t: prev.t + 1,
      };
      if (alarm && isAttack) next.tp += 1;
      else if (alarm && !isAttack) next.fp += 1;
      else if (!alarm && isAttack) next.fn += 1;
      else next.tn += 1;
      if (next.tp + next.fp + next.fn + next.tn > 4000) {
        next.tp = 0;
        next.fp = 0;
        next.fn = 0;
        next.tn = 0;
      }
      return next;
    });
  }, true);

  const prec = s.tp + s.fp ? s.tp / (s.tp + s.fp) : 1;
  const rec = s.tp + s.fn ? s.tp / (s.tp + s.fn) : 1;

  return (
    <div className="space-y-3">
      <Slider label="detection threshold" value={thr} min={5} max={95} onChange={setThr} />
      <Canvas
        draw={(ctx, w, h) => {
          ctx.clearRect(0, 0, w, h);
          const draw = (mu: number, color: string) => {
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let px = 0; px <= 560; px++) {
              const x = (px / 560) * 100;
              const y = Math.exp(-Math.pow(x - mu, 2) / (2 * Math.pow(15, 2)));
              const py = h - 25 - y * (h - 60);
              if (px === 0) ctx.moveTo(40 + px, py);
              else ctx.lineTo(40 + px, py);
            }
            ctx.stroke();
          };
          draw(30, "#10b981");
          draw(65, "#f43f5e");
          const x = 40 + (thr / 100) * 560;
          ctx.strokeStyle = "#f59e0b";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(x, 20);
          ctx.lineTo(x, h - 25);
          ctx.stroke();
          ctx.fillStyle = "#71717a";
          ctx.font = "11px monospace";
          ctx.fillText("flow score →", 560, h - 8);
          ctx.fillStyle = "#f59e0b";
          ctx.fillText(`thr=${thr}`, x + 4, 32);
        }}
      />
      <div className="grid grid-cols-3 gap-2">
        <Stat
          label="precision"
          value={prec.toFixed(2)}
          tone={prec > 0.8 ? "emerald" : "amber"}
        />
        <Stat
          label="recall"
          value={rec.toFixed(2)}
          tone={rec > 0.8 ? "emerald" : "rose"}
        />
        <Stat label="threshold" value={String(thr)} tone="cyan" />
      </div>
      <p className="text-[11px] leading-5 text-zinc-500">
        Models the network-security cluster: the paper family that keeps re-attacking
        “detect attacks without drowning operators in false alarms”. Lower the threshold
        and recall rises while precision collapses — the trade every IDS paper navigates.
      </p>
    </div>
  );
}

/* -------------------------------------------------- *
 * 7. DC load balancing: per-flow vs per-packet
 * ------------------------------------------------- */
type DcState = { fct: number[]; t: number; mega: number };

export function DcLoadBalancingSim() {
  const [mode, setMode] = useState<"flow" | "packet">("flow");
  const [flows, setFlows] = useState(20);
  const [s, setS] = useState<DcState>({ fct: [], t: 0, mega: 0 });

  useLoop(() => {
    setS((prev) => {
      const isMega = Math.random() < 0.1;
      const size = isMega ? 80 : 6 + Math.random() * 6;
      const pathCap = 6;
      const v =
        mode === "flow"
          ? (size / pathCap) * (1 + Math.random() * flows * 0.12)
          : (size / (pathCap * 2)) * 1.15;
      const fct = [...prev.fct, v];
      if (fct.length > 400) fct.shift();
      return { fct, t: prev.t + 1, mega: prev.mega + (isMega ? 1 : 0) };
    });
  }, true);

  const mean = s.fct.length ? s.fct.reduce((a, b) => a + b, 0) / s.fct.length : 0;
  const p99 = s.fct.length
    ? ([...s.fct].sort((a, b) => a - b)[Math.floor(s.fct.length * 0.99)] ?? 0)
    : 0;

  return (
    <div className="space-y-3">
      <div className="flex gap-1">
        {(["flow", "packet"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded-md border px-2 py-1 font-mono text-[10px] ${
              mode === m
                ? "border-sky-700/50 bg-sky-600/20 text-sky-300"
                : "border-zinc-700 bg-zinc-900 text-zinc-400"
            }`}
          >
            {m === "flow" ? "per-flow (ECMP)" : "per-packet (idealised)"}
          </button>
        ))}
      </div>
      <Slider
        label="concurrent flows"
        value={flows}
        min={2}
        max={80}
        onChange={setFlows}
      />
      <Canvas
        draw={(ctx, w, h) => {
          ctx.clearRect(0, 0, w, h);
          const max = Math.max(1, ...s.fct.slice(-120));
          const n = Math.min(120, s.fct.length);
          for (let i = 0; i < n; i++) {
            const v = s.fct[s.fct.length - n + i];
            const bh = (v / max) * (h - 50);
            ctx.fillStyle = v > 30 ? "#f43f5e" : "#0ea5e9";
            ctx.fillRect(30 + i * 5, h - 25 - bh, 4, bh);
          }
          ctx.fillStyle = "#71717a";
          ctx.font = "11px monospace";
          ctx.fillText("recent flow completion times (red = elephant)", 30, 20);
        }}
      />
      <div className="grid grid-cols-3 gap-2">
        <Stat label="mean FCT" value={mean.toFixed(2)} tone="cyan" />
        <Stat
          label="p99 FCT"
          value={p99.toFixed(2)}
          tone={p99 > 40 ? "rose" : "emerald"}
        />
        <Stat label="elephants seen" value={String(s.mega)} tone="amber" />
      </div>
      <p className="text-[11px] leading-5 text-zinc-500">
        Models the datacenter cluster: per-flow hashing strands an elephant behind mice on
        one path (fat tail), while per-packet spreading trims p99 at the cost of
        reordering — the tension the corpus&apos;s DC transport papers argue.
      </p>
    </div>
  );
}

/* ------------------------------------------------- *
 * 8. Wireless interference: shared spectrum goodput
 * ----------------------------------------------- */
export function InterferenceSim() {
  const [links, setLinks] = useState(4);
  const [backoff, setBackoff] = useState(16);
  const [agg, setAgg] = useState(0);

  useLoop(() => {
    const p = Math.exp(-(links - 1) / backoff);
    const target = links * p;
    setAgg((prev) => prev * 0.7 + target * 0.3);
  }, true);

  const ideal = links;

  return (
    <div className="space-y-3">
      <Slider label="contending links" value={links} min={1} max={40} onChange={setLinks} />
      <Slider
        label="backoff window"
        value={backoff}
        min={2}
        max={64}
        onChange={setBackoff}
      />
      <Canvas
        draw={(ctx, w, h) => {
          ctx.clearRect(0, 0, w, h);
          const bars = 40;
          for (let i = 1; i <= bars; i++) {
            const p = Math.exp(-(i - 1) / backoff);
            const goodput = i * p;
            const bh = Math.min(1, goodput / 12) * (h - 50);
            const x = 30 + ((i - 1) / bars) * 560;
            ctx.fillStyle = i === links ? "#0ea5e9" : "#3f3f46";
            ctx.fillRect(x, h - 25 - bh, 12, bh);
          }
          ctx.strokeStyle = "#f59e0b";
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          const yI = h - 25 - Math.min(1, ideal / 12) * (h - 50);
          ctx.moveTo(30, yI);
          ctx.lineTo(590, yI);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = "#71717a";
          ctx.font = "11px monospace";
          ctx.fillText("links → aggregate goodput (dashed = linear ideal)", 30, 20);
        }}
      />
      <div className="grid grid-cols-3 gap-2">
        <Stat label="aggregate goodput" value={agg.toFixed(2)} tone="cyan" />
        <Stat label="linear ideal" value={ideal.toFixed(2)} tone="amber" />
        <Stat
          label="efficiency"
          value={`${((agg / ideal) * 100).toFixed(0)}%`}
          tone={agg / ideal > 0.7 ? "emerald" : "rose"}
        />
      </div>
      <p className="text-[11px] leading-5 text-zinc-500">
        Models the wireless cluster: contention collapses goodput past a point; bigger
        backoff windows recover efficiency but add idle time — the CSMA trade the
        corpus&apos;s 802.11 papers tune.
      </p>
    </div>
  );
}

/* ----------------------------------------------- *
 * 9. Traffic classification: feature separation
 * --------------------------------------------- */
export function TrafficAnalysisSim() {
  const [separation, setSeparation] = useState(1.6);
  const [noise, setNoise] = useState(1.0);
  const [score, setScore] = useState({ hits: 0, n: 0 });

  useLoop(() => {
    const a = (Math.random() + Math.random() + Math.random() - 1.5) * noise;
    const b = separation + (Math.random() + Math.random() + Math.random() - 1.5) * noise;
    const pred = (a + b) / 2 > separation / 2 ? 1 : 0;
    const truth = b > a ? 1 : 0;
    setScore((prev) => {
      if (prev.n >= 2000) return { hits: pred === truth ? 1 : 0, n: 1 };
      return { hits: prev.hits + (pred === truth ? 1 : 0), n: prev.n + 1 };
    });
  }, true);

  const accuracy = score.n ? (score.hits / score.n) * 100 : 0;

  return (
    <div className="space-y-3">
      <Slider
        label="class separation"
        value={separation}
        min={0}
        max={4}
        step={0.1}
        onChange={setSeparation}
        fmt={(v) => v.toFixed(1)}
      />
      <Slider
        label="feature noise"
        value={noise}
        min={0.1}
        max={3}
        step={0.1}
        onChange={setNoise}
        fmt={(v) => v.toFixed(1)}
      />
      <Canvas
        draw={(ctx, w, h) => {
          ctx.clearRect(0, 0, w, h);
          const gauss = (mu: number, color: string) => {
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let px = 0; px <= 560; px++) {
              const x = (px / 560) * 6 - 1;
              const y = Math.exp(-Math.pow(x - mu, 2) / (2 * noise * noise));
              const py = h - 25 - y * (h - 60);
              if (px === 0) ctx.moveTo(40 + px, py);
              else ctx.lineTo(40 + px, py);
            }
            ctx.stroke();
          };
          gauss(0, "#10b981");
          gauss(separation, "#0ea5e9");
          ctx.fillStyle = "#71717a";
          ctx.font = "11px monospace";
          ctx.fillText("feature →", 560, h - 8);
        }}
      />
      <div className="grid grid-cols-2 gap-2">
        <Stat
          label="classification accuracy"
          value={`${accuracy.toFixed(1)}%`}
          tone={accuracy > 85 ? "emerald" : "amber"}
        />
        <Stat
          label="separation / noise"
          value={`${(separation / noise).toFixed(2)}`}
          tone="cyan"
        />
      </div>
      <p className="text-[11px] leading-5 text-zinc-500">
        Models the traffic-analysis cluster: accuracy is a function of feature separation
        over noise — the reason the corpus keeps arguing about which flow features survive
        encryption.
      </p>
    </div>
  );
}
