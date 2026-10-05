/* Deterministic scene generators for the WiFi sensing simulators.
 * Every generator takes a seed so a sim is reproducible across reloads and
 * sessions — same contract as object-detection/sim/mock.ts.
 *
 * Scenes are built from the propagation kernels in ./math.ts, so what a sim
 * draws is the output of the same math its Formula panel displays.
 */

import {
  csiPacket,
  excessTau,
  gaussian,
  mulberry32,
  type Complex,
} from "./math";

/* ---------- geometry ---------- */

export type Geometry = {
  /** TX–RX distance in metres */
  l0: number;
  /** TX–RX height separation in metres */
  height: number;
  /** carrier centre frequency in Hz */
  fc: number;
};

/**
 * A two-wall room: TX and RX on opposite sides, target free to move inside.
 * `distances` are target distances from the receiver.
 */
export function makeGeometry(l0 = 5, height = 1.2, fc = 2.4e9): Geometry {
  return { l0, height, fc };
}

/**
 * Build the multipath path list for a target at distance d.
 * Static paths model walls/furniture; the dynamic path carries the target.
 */
export function scenePaths(
  g: Geometry,
  d: number,
  dynamicAmp = 1,
  staticAmp = 0.6,
): { amp: number; tau: number }[] {
  const cLight = 3e8;
  const staticTau = g.l0 / cLight;
  return [
    // direct / static path
    { amp: staticAmp, tau: staticTau },
    // one-bounce reflection off a far wall (static clutter)
    { amp: staticAmp * 0.4, tau: (g.l0 * 1.35) / cLight },
    // dynamic target path — this is the part the classifier keys on
    { amp: dynamicAmp, tau: excessTau(d, g.height, g.l0, cLight) },
  ];
}

/* ---------- time series ---------- */

/**
 * CSI time series at one subcarrier while the target walks a given path.
 * `dists` are the per-packet target distances.
 */
export function csiSeries(
  g: Geometry,
  dists: number[],
  dynamicAmp = 1,
  staticAmp = 0.6,
  noiseSd = 0,
  seed = 1,
): Complex[] {
  const rand = mulberry32(seed);
  const out: Complex[] = [];
  for (const d of dists) {
    const h = csiPacket(scenePaths(g, d, dynamicAmp, staticAmp))[15];
    if (noiseSd > 0) {
      out.push({ re: h.re + gaussian(rand, 0, noiseSd), im: h.im + gaussian(rand, 0, noiseSd) });
    } else {
      out.push(h);
    }
  }
  return out;
}

/** Magnitude series from a complex series. */
export function mags(cs: Complex[]): number[] {
  return cs.map((z) => Math.hypot(z.re, z.im));
}

/** Raw (wrapped) phase series. */
export function phases(cs: Complex[]): number[] {
  return cs.map((z) => Math.atan2(z.im, z.re));
}

/**
 * Constant-velocity target trajectory sampled at `fs`, bouncing off the far
 * wall so the Doppler sign flips mid-run — the real-world case where a
 * single-direction assumption breaks.
 */
export function walkTrajectory(
  fs: number,
  seconds: number,
  dStart: number,
  velocity: number,
  dMin = 1,
  dMax = 8,
): number[] {
  const n = Math.max(1, Math.round(fs * seconds));
  const out: number[] = [];
  let d = dStart;
  for (let i = 0; i < n; i++) {
    out.push(d);
    d += velocity / fs;
    if (d > dMax) {
      d = dMax;
      velocity = -Math.abs(velocity);
    }
    if (d < dMin) {
      d = dMin;
      velocity = Math.abs(velocity);
    }
  }
  return out;
}

/* ---------- vital signs ---------- */

/**
 * Chest-displacement proxy: a breathing waveform plus optional heartbeat
 * component and body-movement noise. This is the signal a phase-based vital
 * sign paper tries to recover from CSI.
 */
export function chestDisplacement(
  fs: number,
  seconds: number,
  bpm: number,
  amp = 0.01,
  heartbeatBpm = 0,
  hbAmp = 0,
  noiseSd = 0,
  seed = 3,
): number[] {
  const rand = mulberry32(seed);
  const n = Math.max(1, Math.round(fs * seconds));
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    let v = amp * Math.sin(2 * Math.PI * (bpm / 60) * t);
    if (heartbeatBpm > 0) v += hbAmp * Math.sin(2 * Math.PI * (heartbeatBpm / 60) * t);
    if (noiseSd > 0) v += gaussian(rand, 0, noiseSd);
    out.push(v);
  }
  return out;
}

/* ---------- classification scenes ---------- */

/**
 * 1-D class clusters whose separation is controlled by `sep` in noise
 * standard deviations. Used by the activity-recognition sim: as `sep` shrinks
 * toward 0, the classes become indistinguishable and accuracy falls.
 */
export function classClusters(
  classes: number,
  perClass: number,
  sep: number,
  noiseSd = 1,
  seed = 7,
): { data: number[]; labels: number[] } {
  const rand = mulberry32(seed);
  const data: number[] = [];
  const labels: number[] = [];
  for (let k = 0; k < classes; k++) {
    const centre = k * sep * noiseSd;
    for (let i = 0; i < perClass; i++) {
      data.push(centre + gaussian(rand, 0, noiseSd));
      labels.push(k);
    }
  }
  return { data, labels };
}

/* ---------- localization scenes ---------- */

export type Fingerprint = { pos: { x: number; y: number }; rssi: number[] };

/**
 * A grid of RSSI fingerprints plus the true position, with measurement noise
 * added to the query so the estimator sees what a real scan would.
 */
export function fingerprintScene(
  w: number,
  h: number,
  tx: { x: number; y: number },
  truePos: { x: number; y: number },
  noiseSd = 0,
  seed = 11,
): { fingerprints: Fingerprint[]; query: number[] } {
  const rand = mulberry32(seed);
  const fingerprints: Fingerprint[] = [];
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const rssi: number[] = [];
      for (const ap of AP_POSITIONS.slice(0, 3)) {
        const d = Math.max(0.6, Math.hypot(c - ap.x, r - ap.y));
        let v = -40 - 20 * Math.log10(d);
        if (c > w * 0.7 && r < h * 0.3) v -= 12;
        rssi.push(v);
      }
      fingerprints.push({ pos: { x: c, y: r }, rssi });
    }
  }
  const query: number[] = [];
  for (const ap of AP_POSITIONS.slice(0, 3)) {
    const d = Math.max(0.6, Math.hypot(truePos.x - ap.x, truePos.y - ap.y));
    let v = -40 - 20 * Math.log10(d);
    if (truePos.x > w * 0.7 && truePos.y < h * 0.3) v -= 12;
    query.push(v + (noiseSd > 0 ? gaussian(rand, 0, noiseSd) : 0));
  }
  return { fingerprints, query };
}

/** Three APs used by the fingerprint scenes. */
export const AP_POSITIONS = [
  { x: 0, y: 0 },
  { x: 6, y: 0 },
  { x: 3, y: 6 },
];

/* ---------- domain shift scenes ---------- */

/**
 * Two feature domains for the same physical activity, separated by `shift`.
 * This is the cross-device / cross-room problem: the model is trained on
 * domain A and evaluated on domain B.
 */
export function domainScenes(
  shift: number,
  spread: number,
  perDomain = 40,
  seed = 13,
): { a: number[]; b: number[] } {
  const rand = mulberry32(seed);
  const draw = (off: number) =>
    Array.from({ length: perDomain }, () => gaussian(rand, off, spread));
  return { a: draw(0), b: draw(shift) };
}

/* ---------- wall materials ---------- */

export const WALL_MATERIALS: { label: string; perWallDb: number }[] = [
  { label: "drywall", perWallDb: 3 },
  { label: "wood", perWallDb: 5 },
  { label: "glass", perWallDb: 2 },
  { label: "concrete", perWallDb: 9 },
  { label: "brick", perWallDb: 11 },
  { label: "metal door", perWallDb: 18 },
];