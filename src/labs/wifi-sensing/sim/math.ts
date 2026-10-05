/* Pure math kernels for the WiFi sensing simulators.
 * All functions are deterministic and unit-tested in __tests__/wifi-math.test.ts.
 *
 * Notation follows the rest of the project: plain unicode, no LaTeX runtime.
 * Complex numbers are represented as {re, im} pairs because the simulators
 * must run in a client component with no dependencies.
 */

export type Complex = { re: number; im: number };

export const EPS = 1e-12;

/* ---------- complex helpers ---------- */

export function c(re: number, im: number): Complex {
  return { re, im };
}

export function cAdd(a: Complex, b: Complex): Complex {
  return { re: a.re + b.re, im: a.im + b.im };
}

export function cMul(a: Complex, b: Complex): Complex {
  return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}

export function cAbs(a: Complex): number {
  return Math.hypot(a.re, a.im);
}

export function cArg(a: Complex): number {
  return Math.atan2(a.im, a.re);
}

export function cScale(a: Complex, k: number): Complex {
  return { re: a.re * k, im: a.im * k };
}

/** e^{-j2πfτ} — the phase term of a single multipath component. */
export function expNegJ2PiFTau(f: number, tau: number): Complex {
  const a = -2 * Math.PI * f * tau;
  return { re: Math.cos(a), im: Math.sin(a) };
}

/* ---------- seeded RNG (deterministic scenes) ---------- */

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller, used for the additive-noise models. */
export function gaussian(rand: () => number, mean = 0, sd = 1): number {
  const u = Math.max(rand(), 1e-12);
  const v = rand();
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ---------- multipath channel & CSI ---------- */

export const SUBFREQ_HZ = 312500; // 20 MHz channel → 64 subcarriers of 312.5 kHz

/** ITU subcarrier indices actually reported by 802.11n tools (1..56). */
export const CSI_SUBCARRIERS = Array.from({ length: 30 }, (_, i) => i + 1);

/**
 * Channel frequency response at one subcarrier: the coherent sum of every
 * path's amplitude and its e^{-j2πfτ} phase term. `paths` are
 * {amp, tau} pairs where tau is the excess path length in seconds.
 */
export function cAt(paths: { amp: number; tau: number }[], f: number): Complex {
  let acc = c(0, 0);
  for (const p of paths) {
    acc = cAdd(acc, cScale(expNegJ2PiFTau(f, p.tau), p.amp));
  }
  return acc;
}

/** CSI across all reported subcarriers for one packet. */
export function csiPacket(
  paths: { amp: number; tau: number }[],
  subcarriers = CSI_SUBCARRIERS,
): Complex[] {
  return subcarriers.map((k) => cAt(paths, k * SUBFREQ_HZ));
}

/**
 * Excess path length of a reflector at distance d from the receiver, given the
 * straight-line TX→RX distance. tau = (sqrt(d² + h²) − L0) / c.
 * This is the whole reason a walking target modulates the channel: its excess
 * path length changes as it moves.
 */
export function excessTau(d: number, height: number, l0: number, cLight = 3e8): number {
  return (Math.hypot(d, height) - l0) / cLight;
}

/**
 * Doppler frequency shift caused by a target moving at `velocity` m/s at a
 * cos(angle) projection of the path change: f_d = v·cosθ·f_c / c.
 * Positive shift = approaching, negative = receding.
 */
export function dopplerHz(velocity: number, cosTheta: number, fc = 2.4e9): number {
  return (velocity * cosTheta * fc) / 3e8;
}

/** Inverse: the velocity a Doppler shift implies. */
export function velocityFromDoppler(fd: number, cosTheta: number, fc = 2.4e9): number {
  return (fd * 3e8) / (cosTheta * fc);
}

/** Speeds (m/s) that map to observable Doppler at 100 Hz packet rate. */
export function maxUnambiguousVelocity(packetRate = 100, fc = 2.4e9): number {
  // Nyquist: |f_d| < packetRate/2 → v_max = (fs/2)·c / f_c
  return ((packetRate / 2) * 3e8) / fc;
}

/* ---------- RSSI vs CSI ---------- */

/**
 * RSSI is the *magnitude only* of the summed channel, so every path's phase is
 * discarded before it reaches the radio. That is why RSSI cannot be used for
 * phase-based sensing and why it jitters when any path length changes.
 */
export function rssiFromPaths(paths: { amp: number; tau: number }[], f: number): number {
  return cAbs(cAt(paths, f));
}

/** dBm from a linear amplitude. */
export function ampToDbm(amp: number): number {
  return 10 * Math.log10(Math.max(amp, EPS));
}

/** Inverse of ampToDbm. */
export function dbmToAmp(dbm: number): number {
  return Math.pow(10, dbm / 10);
}

/* ---------- phase processing ---------- */

/** Wrap a phase into (-π, π]. */
export function wrapPhase(ph: number): number {
  let x = ph;
  while (x > Math.PI) x -= 2 * Math.PI;
  while (x <= -Math.PI) x += 2 * Math.PI;
  return x;
}

/**
 * Linear unwrapping by adding ±2π whenever consecutive samples jump more than
 * π. This is the standard first step before any CSI phase work, because a
 * receiver only reports phase modulo 2π.
 */
export function unwrapSeries(ph: number[]): number[] {
  const out: number[] = [];
  if (!ph.length) return out;
  out.push(ph[0]);
  for (let i = 1; i < ph.length; i++) {
    let d = ph[i] - ph[i - 1];
    if (d > Math.PI) d -= 2 * Math.PI;
    if (d < -Math.PI) d += 2 * Math.PI;
    out.push(out[i - 1] + d);
  }
  return out;
}

/**
 * Sanitize FiLiMS-style sign correction: when the phase *slope* flips sign
 * between consecutive packets the CSI sample is inverted, so the slope is
 * negated. Models the classic SFRA artifact that appears as spikes.
 */
export function sanitizeFiLiMS(unwrapped: number[]): number[] {
  const out = [...unwrapped];
  // FiLiMS removes a linear-fit outlier at index i only once two consecutive
  // slopes exist, so the scan must start at i = 2 (at i = 1 there is no
  // previous slope and out[i - 2] would be out[-1]).
  for (let i = 2; i < out.length; i++) {
    const dPrev = out[i - 1] - out[i - 2];
    const dNow = out[i] - out[i - 1];
    if (dPrev * dNow < 0 && Math.abs(dPrev) > EPS) {
      out[i] = 2 * out[i - 1] - out[i];
    }
  }
  return out;
}

/* ---------- path loss ---------- */

/**
 * Log-distance path loss: PL(d) = PL(d0) + 10·n·log10(d/d0).
 * Exponent n is ~2 in free space, higher indoors (2.4 walls), ~4 through walls.
 */
export function pathLossDb(d: number, n: number, d0 = 1, pl0 = 40): number {
  return pl0 + 10 * n * Math.log10(Math.max(d, d0) / d0);
}

export const PATH_LOSS_EXPONENTS: { label: string; n: number }[] = [
  { label: "free space", n: 2 },
  { label: "indoor office", n: 2.4 },
  { label: "dense clutter", n: 3 },
  { label: "through one wall", n: 4 },
  { label: "through two walls", n: 5.2 },
];

/** First Fresnel zone radius at the midpoint of a TX→RX link. */
export function fresnelRadius(dTotal: number, wavelength = 0.125): number {
  // r1 = sqrt(λ d1 d2 / (d1 + d2)) at d1 = d2 = dTotal/2
  const d1 = dTotal / 2;
  return Math.sqrt((wavelength * d1 * d1) / (d1 + d1));
}

/** The 60% clearance ellipsoid semiaxes (Fresnel zone) for a link. */
export function fresnelZone(
  dTotal: number,
  clearance = 0.6,
  wavelength = 0.125,
): { d1: number; d2: number; radius: number } {
  const d1 = (dTotal / 2) * (1 - Math.sqrt(1 - clearance * clearance));
  const d2 = dTotal - d1;
  return { d1, d2, radius: fresnelRadius(dTotal, wavelength) };
}

/* ---------- spectral / time-frequency ---------- */

/** One DFT bin centre for a window of `n` samples at `fs` Hz. */
export function binHz(n: number, fs: number, k: number): number {
  return (k * fs) / n;
}

/** Frequency resolution of an N-point window. */
export function freqResolution(n: number, fs: number): number {
  return fs / n;
}

/** Velocity resolution implied by a given frequency resolution. */
export function velocityResolution(df: number, cosTheta: number, fc = 2.4e9): number {
  return Math.abs(velocityFromDoppler(df, cosTheta, fc));
}

/**
 * Hann-windowed periodogram of a real signal. `out[k]` is the magnitude of bin
 * k, k = 0..n/2. Windows are deliberately not normalised to unit energy so the
 * sim can show how window length trades resolution against leakage.
 */
export function periodogram(x: number[]): number[] {
  const n = x.length;
  if (!n) return [];
  const half = Math.floor(n / 2);
  const out: number[] = [];
  for (let k = 0; k <= half; k++) {
    let re = 0;
    let im = 0;
    for (let i = 0; i < n; i++) {
      const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1 || 1)));
      const a = (2 * Math.PI * k * i) / n;
      const xr = x[i] * w;
      re += xr * Math.cos(a);
      im -= xr * Math.sin(a);
    }
    out.push(Math.hypot(re, im) / n);
  }
  return out;
}

/** Index of the dominant bin (excluding DC). */
export function peakBin(spec: number[]): number {
  let best = 1;
  for (let k = 2; k < spec.length; k++) if (spec[k] > spec[best]) best = k;
  return best;
}

/** Estimate dominant frequency in Hz from a periodogram. */
export function estimateFreqHz(x: number[], fs: number): number {
  const spec = periodogram(x);
  if (spec.length <= 2) return 0;
  return binHz(x.length, fs, peakBin(spec));
}

/* ---------- classical WiFi features ---------- */

/** Root Mean Square of a series. */
export function rms(x: number[]): number {
  if (!x.length) return 0;
  return Math.sqrt(x.reduce((s, v) => s + v * v, 0) / x.length);
}

/**
 * Doppler Frequency Ratio (DFR): the ratio of the dominant Doppler frequency of
 * the moving path to that of the static path. Unaffected by overall TX power.
 */
export function dfr(movingSpecHz: number, staticSpecHz: number): number {
  if (Math.abs(staticSpecHz) < EPS) return 0;
  return movingSpecHz / staticSpecHz;
}

/**
 * Fluctuation of CSI (FCS): mean CSI fluctuation over the window. A coarse
 * motion-presence feature that needs no training.
 */
export function fcs(x: number[]): number {
  if (x.length < 2) return 0;
  let s = 0;
  for (let i = 1; i < x.length; i++) s += Math.abs(x[i] - x[i - 1]);
  return s / (x.length - 1);
}

/** Windowed variance. */
export function variance(x: number[]): number {
  if (x.length < 2) return 0;
  const m = x.reduce((s, v) => s + v, 0) / x.length;
  return x.reduce((s, v) => s + (v - m) * (v - m), 0) / (x.length - 1);
}

/** Shannon entropy of a normalised spectrum, in bits. */
export function spectralEntropy(spec: number[]): number {
  const tot = spec.reduce((s, v) => s + v, 0);
  if (tot <= EPS) return 0;
  let h = 0;
  for (const v of spec) {
    const p = v / tot;
    if (p > EPS) h -= p * Math.log2(p);
  }
  return h;
}

/* ---------- quantization & oscillator error ---------- */

/**
 * What a NIC's fixed-point sample budget does to the reported CSI. The NIC
 * reports magnitude/phase with `bits` of resolution over a fixed dynamic range.
 */
export function quantize(x: number, bits: number, fullScale = 1): number {
  const levels = Math.pow(2, bits);
  const step = fullScale / levels;
  return Math.round(x / step) * step;
}

/** Quantisation error magnitude bound for `bits` of resolution. */
export function quantErrorBound(bits: number, fullScale = 1): number {
  return fullScale / Math.pow(2, bits + 1);
}

/** Angle error introduced by a magnitude noise level on a complex sample. */
export function phaseNoiseFromAmpNoise(angle: number, ampNoise: number, amp = 1): number {
  return Math.atan2(ampNoise, Math.max(Math.abs(amp * Math.cos(angle)) + ampNoise, EPS));
}

/**
 * Carrier frequency offset expressed as a per-sample phase ramp: a 1 Hz error
 * rotates the reported phase by 2π/fs radians every packet. Sampling frequency
 * offset leaks into the same axis and cannot be separated by phase alone.
 */
export function cfoPhaseRampHz(offsetHz: number, fs: number): number {
  return offsetHz / fs;
}

/* ---------- classification ---------- */

export type Logits = number[];

/** Numerically-stable softmax. */
export function softmax(z: Logits): number[] {
  if (!z.length) return [];
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / Math.max(s, EPS));
}

/** Cross-entropy loss for a one-hot target. */
export function crossEntropy(probs: number[], target: number): number {
  const p = probs[target] ?? EPS;
  return -Math.log(Math.max(p, EPS));
}

/**
 * Accuracy of a nearest-centroid classifier on 1-D class clusters.
 * `data` and `labels` are parallel arrays; centroids are learned from them.
 */
export function nearestCentroidAcc(data: number[], labels: number[]): number {
  const classes = [...new Set(labels)];
  const cent: number[] = classes.map(
    (k) =>
      data.reduce((s, v, i) => (labels[i] === k ? s + v : s), 0) /
      data.filter((_, i) => labels[i] === k).length,
  );
  let ok = 0;
  for (let i = 0; i < data.length; i++) {
    let best = 0;
    let bd = Infinity;
    for (let j = 0; j < cent.length; j++) {
      const d = Math.abs(data[i] - cent[j]);
      if (d < bd) {
        bd = d;
        best = j;
      }
    }
    if (classes[best] === labels[i]) ok++;
  }
  return data.length ? ok / data.length : 0;
}

/** Confusion-matrix accuracy plus per-class F1 for a 2-class problem. */
export function binaryF1(truth: number[], pred: number[]): { acc: number; f1: number } {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (let i = 0; i < truth.length; i++) {
    if (pred[i] === 1 && truth[i] === 1) tp++;
    else if (pred[i] === 1 && truth[i] === 0) fp++;
    else if (pred[i] === 0 && truth[i] === 1) fn++;
  }
  const acc = truth.length ? (tp + (truth.length - tp - fp - fn)) / truth.length : 0;
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
  return { acc, f1 };
}

/* ---------- localization: RSSI fingerprinting ---------- */

export type Grid = number[]; // row-major W*H

/**
 * Synthetic RSSI grid from inverse-square falloff with one shadowed corner,
 * which is what makes fingerprinting non-trivial rather than a clean Voronoi
 * diagram. `x`,`y` in grid units.
 */
export function rssiGrid(
  w: number,
  h: number,
  tx: { x: number; y: number },
  d0 = 1,
  pl0 = -40,
): Grid {
  const out: number[] = [];
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const d = Math.hypot(c - tx.x, r - tx.y);
      let v = pl0 - 20 * Math.log10(Math.max(d, d0));
      if (c > w * 0.7 && r < h * 0.3) v -= 12; // shadowed corner
      out.push(v);
    }
  }
  return out;
}

/** Euclidean distance between two RSSI vectors (the k-NN metric). */
export function rssiDist(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

/**
 * k-NN localization: find the `k` closest fingerprints and return their
 * centroid. This is the dominant classical WiFi localization baseline.
 */
export function knnLocalize(
  query: number[],
  fingerprints: { pos: { x: number; y: number }; rssi: number[] }[],
  k = 3,
): { x: number; y: number; meanDist: number } {
  if (!fingerprints.length) return { x: 0, y: 0, meanDist: Infinity };
  const scored = fingerprints
    .map((f) => ({ f, d: rssiDist(query, f.rssi) }))
    .sort((a, b) => a.d - b.d);
  const top = scored.slice(0, Math.max(1, Math.min(k, scored.length)));
  const x = top.reduce((s, t) => s + t.f.pos.x, 0) / top.length;
  const y = top.reduce((s, t) => s + t.f.pos.y, 0) / top.length;
  return { x, y, meanDist: top.reduce((s, t) => s + t.d, 0) / top.length };
}

/* ---------- trajectory smoothing ---------- */

export type KalmanState = { x: number; v: number; p: number };

/**
 * Constant-velocity Kalman filter for a 1-D position track. `accelVar` is
 * process noise (how quickly the target may change velocity); `measVar` is
 * the CSI-derived position noise. Large measVar → heavy smoothing.
 */
export function kalmanStep(
  st: KalmanState,
  z: number,
  accelVar: number,
  measVar: number,
): { state: KalmanState; innovation: number } {
  // predict
  const x = st.x + st.v;
  const p = st.p + accelVar;
  // update
  const k = p / (p + measVar);
  const nx = x + k * (z - x);
  const np = (1 - k) * p;
  const nv = st.v + (k * (z - x)) / (st.p + accelVar || 1);
  return { state: { x: nx, v: nv, p: np }, innovation: z - x };
}

/** Run the filter over a measurement series. */
export function kalmanTrack(
  zs: number[],
  accelVar: number,
  measVar: number,
): { smoothed: number[]; states: KalmanState[] } {
  let st: KalmanState = { x: zs[0] ?? 0, v: 0, p: 1 };
  const smoothed: number[] = [];
  const states: KalmanState[] = [];
  for (const z of zs) {
    const r = kalmanStep(st, z, accelVar, measVar);
    st = r.state;
    smoothed.push(st.x);
    states.push(st);
  }
  return { smoothed, states };
}

/** Root-mean-square position error of a track vs ground truth. */
export function trackRmse(est: number[], truth: number[]): number {
  const n = Math.min(est.length, truth.length);
  if (!n) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += (est[i] - truth[i]) ** 2;
  return Math.sqrt(s / n);
}

/* ---------- vital signs ---------- */

/** Chest displacement amplitude → CSI phase swing, via small-angle sensitivity. */
export function phaseSwingForDisplacement(
  displacement: number,
  sensitivity: number,
): number {
  return displacement * sensitivity;
}

/**
 * Estimate a periodic vital rate (breaths or heartbeats per minute) from a
 * displacement-like signal, by finding the dominant FFT bin and converting it
 * to cycles per minute. Only searches the plausible physiological band.
 */
export function estimateBpm(x: number[], fs: number, minBpm = 6, maxBpm = 48): number {
  const spec = periodogram(x);
  const binOf = (bpm: number) => Math.round((bpm / 60) * x.length) / fs;
  const lo = Math.max(1, binOf(minBpm));
  const hi = Math.min(spec.length - 1, binOf(maxBpm));
  if (hi <= lo) return 0;
  let best = lo;
  for (let k = lo; k <= hi; k++) if (spec[k] > spec[best]) best = k;
  return (best * fs) / x.length * 60;
}

/** Relative error of a rate estimate against the true rate. */
export function bpmError(est: number, truth: number): number {
  if (!truth) return 0;
  return Math.abs(est - truth) / truth;
}

/* ---------- robustness ---------- */

/**
 * Cross-domain accuracy: train and test distributions are separated by `shift`
 * in the same units the features live in. With no adaptation, a linear
 * threshold learned on the source stops separating the target domain.
 */
export function domainShiftAccuracy(shift: number, spread: number): number {
  // Decision boundary learned at 0; after shifting the mean by `shift`, the
  // fraction of target samples that stay on the correct side decays with the
  // offset measured in spreads.
  const z = shift / Math.max(spread, EPS);
  // Gaussian tail approximation via erf
  return 0.5 * (1 + erf(z / Math.SQRT2));
}

/** Abramowitz & Stegun 7.1.26 error function approximation. */
export function erf(x: number): number {
  const s = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-ax * ax);
  return s * y;
}

/** SNR in dB from signal and noise powers. */
export function snrDb(signalPower: number, noisePower: number): number {
  return 10 * Math.log10(Math.max(signalPower, EPS) / Math.max(noisePower, EPS));
}

/**
 * Two-class detection reliability from a 1-D Gaussian score: returns the
 * balanced accuracy of a fixed threshold as noise grows. `sep` is the class
 * separation in noise standard deviations.
 */
export function detectionRate(sep: number, noiseSd: number): number {
  const z = sep / Math.max(2 * noiseSd, EPS);
  return 0.5 * (1 + erf(z / Math.SQRT2));
}

/** Through-wall attenuation in dB for a wall count and material factor. */
export function throughWallDb(walls: number, perWallDb: number): number {
  return walls * perWallDb;
}

/** Whether the post-wall SNR stays above a usable margin. */
export function usableAfterWalls(
  txDbm: number,
  pathLoss: number,
  walls: number,
  perWallDb: number,
  noiseDbm: number,
  marginDb = 6,
): boolean {
  return txDbm - pathLoss - throughWallDb(walls, perWallDb) - noiseDbm >= marginDb;
}

/* ---------- feature-window helpers used by the Conv1D sim ---------- */

/** How many packets a 1-D conv kernel of width k covers at stride s. */
export function convReceptiveField(k: number, layers: number, stride = 1): number {
  let rf = 1;
  let jump = 1;
  for (let i = 0; i < layers; i++) {
    rf += (k - 1) * jump;
    jump *= stride;
  }
  return rf;
}

/**
 * Temporal receptive field expressed in seconds, given the packet rate. A
 * conv that sees 40 packets at 100 Hz covers 0.4 s — enough for a gait cycle
 * but not for a full respiration cycle.
 */
export function receptiveFieldSeconds(rf: number, packetRate: number): number {
  return rf / packetRate;
}