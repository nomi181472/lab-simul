"use client";

import { useMemo, useState } from "react";
import {
  binaryF1,
  bpmError,
  crossEntropy,
  detectionRate,
  erf,
  estimateBpm,
  gaussian,
  kalmanTrack,
  knnLocalize,
  mulberry32,
  nearestCentroidAcc,
  rssiGrid,
  snrDb,
  softmax,
  trackRmse,
} from "@/labs/wifi-sensing/sim/math";
import {
  AP_POSITIONS,
  chestDisplacement,
  classClusters,
  fingerprintScene,
} from "@/labs/wifi-sensing/sim/mock";
import { Slider, Segmented, Seeded, Sim } from "./shared";
import { BarPlot, GridPlan, LinePlot, ShareBar, Stat } from "./canvas";
import { SIM_PAPERS } from "./papers";

/* ========================= Activity classification ========================= */

export function ActivityClfSim() {
  const [sep, setSep] = useState(2.2);
  const [classes, setClasses] = useState(4);
  const [noise, setNoise] = useState(1);
  const [seed, setSeed] = useState(2);
  const perClass = 40;

  const { acc, f1, probs } = useMemo(() => {
    const { data, labels: lb } = classClusters(classes, perClass, sep, noise, seed);
    return {
      acc: nearestCentroidAcc(data, lb),
      f1: binaryF1(
        lb.map((l) => (l === 0 ? 1 : 0)),
        nearestCentroidLabels(data, lb, classes),
      ).f1,
      labels: lb,
      probs: softmax(data.slice(0, classes)),
    };
  }, [sep, classes, noise, seed]);

  const logitScale = sep * noise;
  const loss = crossEntropy(probs, 0);

  return (
    <Sim
      name="Class separability is the whole game"
      identifies={`${classes} activity classes separated by ${sep.toFixed(2)}σ: nearest-centroid accuracy ${(acc * 100).toFixed(1)}%. Drop separation toward 0 and the classes stop being distinguishable regardless of model capacity.`}
      math={[
        "p = softmax(z),   z = model output logits",
        "CE loss = −log p_{y}          (→ ∞ as p_y → 0)",
        "Bayes error is set by class separation d/σ, not by architecture:",
        "for 2 Gaussians at separation k·σ,  P(error) = Φ(−k/2)",
      ]}
      need="A model's ceiling is the separability of the features, not its depth. If walking and sitting produce overlapping CSI distributions, no architecture separates them — the paper needs better features, not more layers."
      real="Reported activity-recognition accuracies in the 90s on 5–8 classes are consistent with roughly 2–3σ separation; the jump to 99% usually reflects an easier class set, not a better model."
      papers={SIM_PAPERS["activity-clf"].papers}
      params={
        <>
          <Slider label="class separation" value={sep} min={0} max={4} step={0.1} onChange={setSep} fmt={(v) => `${v.toFixed(2)} σ`} />
          <Slider label="noise σ" value={noise} min={0.2} max={3} step={0.1} onChange={setNoise} />
          <Segmented
            label="num classes"
            value={String(classes)}
            onChange={(v) => setClasses(Number(v))}
            options={[
              { value: "2", label: "2" },
              { value: "4", label: "4" },
              { value: "8", label: "8" },
            ]}
          />
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
        </>
      }
    >
      <BarPlot
        ys={Array.from({ length: 81 }, (_, i) =>
          0.5 * (1 + erf((i / 40 - 1) / Math.max(sep / 2, 0.05))),
        )}
        yLabel="accuracy"
        xLabel="separation (σ)"
        color="#22d3ee"
      />
      <div className="mt-3">
        <ShareBar
          parts={[
            { label: "correct", value: acc, color: "bg-emerald-500/80" },
            { label: "error", value: 1 - acc, color: "bg-rose-500/70" },
          ]}
        />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="accuracy" value={`${(acc * 100).toFixed(1)}%`} tone={acc > 0.9 ? "emerald" : acc > 0.7 ? "amber" : "rose"} />
        <Stat label="binary F1" value={f1.toFixed(3)} />
        <Stat label="separation" value={`${(sep * noise).toFixed(2)} units`} tone="cyan" />
        <Stat label="CE loss (k=0)" value={loss.toFixed(3)} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        {logitScale > 0.2
          ? `At ${sep.toFixed(2)}σ the classes are cleanly separable and accuracy saturates — adding layers buys nothing. `
          : `Below roughly 1σ the distributions overlap and accuracy collapses toward chance (${
              (100 / classes).toFixed(0)
            }% for ${classes} classes). `}
        This is the shape behind most “we improved accuracy” claims: they moved separability, not
        the model.
      </p>
    </Sim>
  );
}

function nearestCentroidLabels(data: number[], labels: number[], classes: number): number[] {
  const cent: number[] = [];
  for (let k = 0; k < classes; k++) {
    const vals = data.filter((_, i) => labels[i] === k);
    cent.push(vals.reduce((s, v) => s + v, 0) / Math.max(vals.length, 1));
  }
  return data.map((v) => {
    let best = 0;
    let bd = Infinity;
    cent.forEach((c, j) => {
      if (Math.abs(v - c) < bd) {
        bd = Math.abs(v - c);
        best = j;
      }
    });
    return best;
  });
}

/* ======================= RSSI fingerprint localization ======================= */

const TRUE_POS = { x: 4.3, y: 3.6 };

export function RssiFingerprintSim() {
  const [k, setK] = useState(3);
  const [noise, setNoise] = useState(0.8);
  const [seed, setSeed] = useState(11);
  const w = 6;
  const h = 6;

  const { fingerprints, query } = useMemo(
    () => fingerprintScene(w, h, AP_POSITIONS[0], TRUE_POS, noise, seed),
    [noise, seed],
  );
  const est = knnLocalize(query, fingerprints, k);
  const err = Math.hypot(est.x - TRUE_POS.x, est.y - TRUE_POS.y);

  return (
    <Sim
      name="k-NN over RSSI fingerprints"
      identifies={`A scan taken at (${TRUE_POS.x.toFixed(1)}, ${TRUE_POS.y.toFixed(1)}) with noise σ=${noise.toFixed(2)} dB is matched by k=${k} nearest fingerprints to (${est.x.toFixed(2)}, ${est.y.toFixed(2)}) — error ${err.toFixed(2)} m.`}
      math={[
        "d(q, f) = ‖ RSSI(q) − RSSI(f) ‖₂    (Euclidean over the per-AP vector)",
        "poŝ = mean of the k nearest fingerprint positions",
        "RSSI(d) = RSSI(d₀) − 20·log₁₀(d/d₀)  (log-distance model that shapes the grid)",
      ]}
      need="Fingerprinting sidesteps multipath modelling entirely: measure the RSSI map once, then match. Accuracy is therefore limited by how well the map's shadowing pattern generalises and by measurement noise — not by any physical model."
      real="This is the localization baseline nearly every model-based paper compares against, and the reason many report sub-metre errors only on a map built in the same room they test in."
      papers={SIM_PAPERS["rssi-fingerprint"].papers}
      params={
        <>
          <Slider label="k neighbours" value={k} min={1} max={8} step={1} onChange={setK} fmt={(v) => String(v)} />
          <Slider label="RSSI noise" value={noise} min={0} max={4} step={0.1} onChange={setNoise} />
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
        </>
      }
    >
      <GridPlan w={w} h={h} ap={AP_POSITIONS[0]} truePos={TRUE_POS} estPos={est} />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="position error" value={`${err.toFixed(2)} m`} tone={err < 1 ? "emerald" : err < 2.5 ? "amber" : "rose"} />
        <Stat label="mean NN distance" value={est.meanDist.toFixed(2)} />
        <Stat label="fingerprint count" value={String(fingerprints.length)} />
        <Stat label="grid cell" value="1.0 m" />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Raise the noise and the estimate slides across cell boundaries in discrete jumps — k-NN
        returns a fingerprint centroid, never a continuous position, so its error is quantised by
        the grid. Lower k toward 1 and the estimate becomes the single closest cell, which is
        high-variance; raise k and it smooths but drifts toward the map’s dense region. The
        shadowed corner is deliberately unmodelled: no amount of averaging recovers a region the
        map never characterised.
      </p>
    </Sim>
  );
}

/* ========================= Kalman trajectory ========================= */

export function KalmanTrajectorySim() {
  const [seed, setSeed] = useState(3);
  const [measVar, setMeasVar] = useState(1.2);
  const [accelVar, setAccelVar] = useState(0.08);
  const [velocity, setVelocity] = useState(1);
  const fs = 20;
  const n = 90;

  const { truth, noisy, smoothed } = useMemo(() => {
    const rand = mulberry32(seed);
    const t: number[] = [];
    let d = 1;
    for (let i = 0; i < n; i++) {
      t.push(d);
      d += velocity / fs;
      if (d > 7) d = 7;
    }
    const z = t.map((v) => v + gaussian(rand, 0, Math.sqrt(measVar)));
    return { truth: t, noisy: z, smoothed: kalmanTrack(z, accelVar, measVar).smoothed };
  }, [seed, measVar, accelVar, velocity, n, fs]);

  const rawRmse = trackRmse(noisy, truth);
  const smRmse = trackRmse(smoothed, truth);
  const lag = Math.abs(smoothed[smoothed.length - 1] - truth[truth.length - 1]);

  return (
    <Sim
      name="Smoothing a noisy CSI-derived track"
      identifies={`True path vs CSI-derived estimates. Raw RMSE ${rawRmse.toFixed(3)} m → Kalman-smoothed ${smRmse.toFixed(3)} m, at the cost of ${lag.toFixed(3)} m of end-point lag.`}
      math={[
        "state x = [position, velocity]ᵀ,  constant-velocity motion",
        "predict:  x⁻ = F x,      P⁻ = F P Fᵀ + Q",
        "update:   K = P⁻ / (P⁻ + R),   x = x⁻ + K (z − x⁻)",
        "Q = process noise (how fast the target may change),  R = measurement noise (CSI quality)",
      ]}
      need="CSI-derived positions are noisy and bursty. A Kalman filter trades responsiveness for stability by trusting the motion model — the Q/R ratio is the entire tuning, and it is a real trade rather than a free win."
      real="Trajectory and localization papers smooth for exactly this reason; the ones that report only raw estimates are often reporting a noisier number than their competitors, not a better method."
      papers={SIM_PAPERS["kalman-trajectory"].papers}
      params={
        <>
          <Slider label="measurement noise R" value={measVar} min={0.05} max={6} step={0.05} onChange={setMeasVar} />
          <Slider label="process noise Q" value={accelVar} min={0.005} max={1} step={0.005} onChange={setAccelVar} />
          <Slider label="target speed" value={velocity} min={0.2} max={2.5} step={0.1} onChange={setVelocity} fmt={(v) => `${v.toFixed(1)} m/s`} />
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
        </>
      }
    >
      <LinePlot
        series={[
          { ys: truth, color: "#34d399", width: 2, label: "truth" },
          { ys: noisy, color: "#71717a", width: 1, label: "raw CSI estimate" },
          { ys: smoothed, color: "#22d3ee", width: 2, label: "Kalman smoothed" },
        ]}
        yLabel="distance (m)"
        xLabel="packet index"
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="raw RMSE" value={`${rawRmse.toFixed(3)} m`} tone="rose" />
        <Stat label="smoothed RMSE" value={`${smRmse.toFixed(3)} m`} tone="emerald" />
        <Stat label="improvement" value={`${(100 * (1 - smRmse / Math.max(rawRmse, 1e-9))).toFixed(0)}%`} tone="cyan" />
        <Stat label="end lag" value={`${lag.toFixed(3)} m`} tone={lag > 0.8 ? "amber" : "zinc"} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Drop Q toward zero and the smoothed line stops responding — it trusts the motion model so
        much that a direction change passes it by. Raise Q and it chases the noise again. That
        trade is visible in the end-point lag tile, and it is why a filter tuned for a walking
        person fails on a falling one.
      </p>
    </Sim>
  );
}

/* ============================ Vital signs ============================ */

export function VitalSignsSim() {
  const [bpm, setBpm] = useState(15);
  const [amp, setAmp] = useState(0.01);
  const [noise, setNoise] = useState(0.004);
  const [fs, setFs] = useState(10);
  const seconds = 30;

  const { est, rel, spec } = useMemo(() => {
    const x = chestDisplacement(fs, seconds, bpm, amp, 0, 0, noise, 5);
    const e = estimateBpm(x, fs);
    return { est: e, rel: bpmError(e, bpm), spec: x };
  }, [bpm, amp, noise, fs]);

  const snr = snrDb(amp * amp, noise * noise);
  const breathHz = bpm / 60;
  const usable = rel < 0.1;

  return (
    <Sim
      name="Recovering respiration rate from a tiny displacement"
      identifies={`True ${bpm.toFixed(1)} bpm (${breathHz.toFixed(3)} Hz) at displacement amplitude ${amp.toFixed(3)} and noise σ=${noise.toFixed(4)} — feature SNR ${snr.toFixed(1)} dB, estimate ${est.toFixed(1)} bpm, error ${(rel * 100).toFixed(1)}%.`}
      math={[
        "chest displacement d(t) ~ A sin(2π f_b t),  f_b = bpm/60",
        "CSI phase swing Δφ = d(t) · sensitivity   (small-angle, calibrated per setup)",
        "estimate:  b̂pm = 60 · argmax_k |FFT(Δφ)| / (N / f_s)",
        "search band 6–48 bpm, so the estimate cannot leave the physiological range",
      ]}
      need="Respiration produces millimetre chest motion, so the signal lives far below the CSI noise floor and must be found by spectral search inside a narrow plausible band. The band restriction is doing most of the work, which is why a paper must report how often it locks onto a harmonic instead of the fundamental."
      real="Vital-sign papers report median absolute error in bpm. The dominant failure is not estimation noise but body movement, which adds broadband energy that swamps the respiratory band entirely."
      papers={SIM_PAPERS["vital-signs"].papers}
      params={
        <>
          <Slider label="true rate" value={bpm} min={6} max={40} step={0.5} onChange={setBpm} fmt={(v) => `${v.toFixed(1)} bpm`} />
          <Slider label="chest amplitude" value={amp} min={0.0005} max={0.05} step={0.0005} onChange={setAmp} />
          <Slider label="noise σ" value={noise} min={0} max={0.03} step={0.001} onChange={setNoise} />
          <Slider label="sample rate" value={fs} min={2} max={40} step={1} onChange={setFs} fmt={(v) => `${v} Hz`} />
        </>
      }
    >
      <LinePlot
        series={[{ ys: spec.slice(0, 200), color: "#22d3ee", label: "CSI phase swing" }]}
        yLabel="displacement"
        xLabel="sample index"
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
        <Stat label="feature SNR" value={`${snr.toFixed(1)} dB`} tone={snr > 10 ? "emerald" : snr > 0 ? "amber" : "rose"} />
        <Stat label="estimate" value={`${est.toFixed(1)} bpm`} tone="cyan" />
        <Stat label="abs error" value={`${Math.abs(est - bpm).toFixed(2)} bpm`} />
        <Stat label="rel error" value={`${(rel * 100).toFixed(1)}%`} tone={usable ? "emerald" : "rose"} />
        <Stat label="verdict" value={usable ? "locked" : "lost"} tone={usable ? "emerald" : "rose"} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Shrink the chest amplitude and inflate the noise until the estimate leaves the true rate
        entirely. The estimator still returns a number inside 6–48 bpm because it must choose some
        bin — that is the dangerous failure mode: a confidently wrong vital sign rather than an
        explicit failure. Robust systems detect it by requiring the peak to clear a threshold.
      </p>
    </Sim>
  );
}

/* ============================= SNR margin ============================= */

export function SnrMarginSim() {
  const [sep, setSep] = useState(3);
  const [noiseSd, setNoiseSd] = useState(1);
  const rate = detectionRate(sep, noiseSd);
  const curve = useMemo(
    () => Array.from({ length: 60 }, (_, i) => detectionRate(sep, i * 0.1 + 0.05)),
    [sep],
  );
  const db = snrDb(1, noiseSd * noiseSd);

  return (
    <Sim
      name="Detection reliability is an SNR statement"
      identifies={`Two classes separated by ${sep.toFixed(2)}σ with feature noise ${noiseSd.toFixed(2)} (SNR ${db.toFixed(1)} dB): balanced detection rate ${(rate * 100).toFixed(1)}%.`}
      math={[
        "SNR_dB = 10·log₁₀( P_signal / P_noise )",
        "balanced accuracy of a fixed threshold = Φ( k / (2σ) ),  k = class separation",
        "⇒ halving σ is worth as much as doubling the true separation",
      ]}
      need="Most WiFi sensing claims are SNR claims wearing a task label. Improving a classifier cannot help once the feature SNR is too low to separate the classes at all, which puts a hard ceiling on accuracy that is set by radio conditions."
      real="This is why papers report performance conditioned on distance and obstruction — the SNR floor moves with the environment, and so does the achievable accuracy."
      papers={SIM_PAPERS["snr-margin"].papers}
      params={
        <>
          <Slider label="class separation k" value={sep} min={0} max={6} step={0.1} onChange={setSep} />
          <Slider label="noise σ" value={noiseSd} min={0.1} max={4} step={0.1} onChange={setNoiseSd} />
        </>
      }
    >
      <BarPlot ys={curve} yLabel="detection rate" xLabel="noise σ" highlight={Math.min(59, Math.round((noiseSd - 0.05) / 0.1))} />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="SNR" value={`${db.toFixed(1)} dB`} tone="cyan" />
        <Stat label="detection rate" value={`${(rate * 100).toFixed(1)}%`} tone={rate > 0.9 ? "emerald" : rate > 0.7 ? "amber" : "rose"} />
        <Stat label="miss rate" value={`${(100 * (1 - rate)).toFixed(1)}%`} tone="rose" />
        <Stat label="chance" value="50.0%" />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Sweep noise and watch the rate fall from near-perfect to chance. Note the symmetry: the
        curve is identical whether you reduce separation or increase noise, because the detector
        only sees the ratio. A paper reporting 60% accuracy on a 4-class task is close to this
        curve’s 50% floor, and no architecture moves it.
      </p>
    </Sim>
  );
}

/** rssiGrid re-exported for the Compare view's grid rendering. */
export { rssiGrid };