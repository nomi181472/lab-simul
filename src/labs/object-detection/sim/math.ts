/* Pure math kernels for the simulators.
 * All functions are deterministic and unit-tested in __tests__/math.test.ts.
 */

export type Box = { x1: number; y1: number; x2: number; y2: number };

export type NumArray = number[];

export const EPS = 1e-9;

/* ---------- boxes ---------- */

export function area(b: Box): number {
  return Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
}

export function intersection(a: Box, b: Box): number {
  const w = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const h = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  return w * h;
}

export function union(a: Box, b: Box): number {
  return area(a) + area(b) - intersection(a, b);
}

export function iou(a: Box, b: Box): number {
  const inter = intersection(a, b);
  const uni = union(a, b);
  return uni <= EPS ? 0 : inter / uni;
}

export function bboxCenter(b: Box): { cx: number; cy: number; w: number; h: number } {
  return { cx: (b.x1 + b.x2) / 2, cy: (b.y1 + b.y2) / 2, w: b.x2 - b.x1, h: b.y2 - b.y1 };
}

function enclosingBox(a: Box, b: Box): Box {
  return {
    x1: Math.min(a.x1, b.x1),
    y1: Math.min(a.y1, b.y1),
    x2: Math.max(a.x2, b.x2),
    y2: Math.max(a.y2, b.y2),
  };
}

/** Generalized IoU: IoU minus the non-overlap in the convex hull. */
export function giou(a: Box, b: Box): number {
  const inter = intersection(a, b);
  const uni = union(a, b);
  const c = enclosingBox(a, b);
  const cArea = area(c);
  if (cArea <= EPS) return 0;
  const iouVal = uni <= EPS ? 0 : inter / uni;
  return iouVal - (cArea - uni) / cArea;
}

/** Distance-IoU: IoU minus normalized center distance. */
export function diou(a: Box, b: Box): number {
  const ac = bboxCenter(a);
  const bc = bboxCenter(b);
  const inter = intersection(a, b);
  const uni = union(a, b);
  const iouVal = uni <= EPS ? 0 : inter / uni;
  const c = enclosingBox(a, b);
  const cDiag2 = (c.x2 - c.x1) ** 2 + (c.y2 - c.y1) ** 2;
  const centerDist2 = (ac.cx - bc.cx) ** 2 + (ac.cy - bc.cy) ** 2;
  const rho = cDiag2 <= EPS ? 0 : centerDist2 / cDiag2;
  return iouVal - rho;
}

/** Complete-IoU: DIoU plus an aspect-ratio consistency penalty. */
export function ciou(a: Box, b: Box): number {
  const ac = bboxCenter(a);
  const bc = bboxCenter(b);
  const inter = intersection(a, b);
  const uni = union(a, b);
  const iouVal = uni <= EPS ? 0 : inter / uni;
  const c = enclosingBox(a, b);
  const cDiag2 = (c.x2 - c.x1) ** 2 + (c.y2 - c.y1) ** 2;
  const centerDist2 = (ac.cx - bc.cx) ** 2 + (ac.cy - bc.cy) ** 2;
  const rho = cDiag2 <= EPS ? 0 : centerDist2 / cDiag2;
  const w = Math.max(EPS, ac.w);
  const h = Math.max(EPS, ac.h);
  const wt = Math.max(EPS, bc.w);
  const ht = Math.max(EPS, bc.h);
  const v = (4 / Math.PI ** 2) * (Math.atan(w / h) - Math.atan(wt / ht)) ** 2;
  const alpha = v <= EPS ? 0 : v / (1 - iouVal + v + EPS);
  return iouVal - rho - alpha * v;
}

/** IoU-loss family values as losses (0 = perfect). */
export function iouFamily(pred: Box, gt: Box) {
  return {
    iou: iou(pred, gt),
    giou: giou(pred, gt),
    diou: diou(pred, gt),
    ciou: ciou(pred, gt),
    lossIou: 1 - iou(pred, gt),
    lossGioU: 1 - giou(pred, gt),
    lossDioU: 1 - diou(pred, gt),
    lossCioU: 1 - ciou(pred, gt),
  };
}

/* ---------- regression losses ---------- */

export function smoothL1(x: number, beta = 1): number {
  const ax = Math.abs(x);
  return ax < beta ? 0.5 * (ax * ax) / beta : ax - 0.5 * beta;
}

/** Coordinate-wise box regression loss on (cx, cy, w, h) normalized deltas. */
export function boxRegressionLoss(
  pred: Box,
  gt: Box,
  kind: "l1" | "smooth-l1" | "l2",
): { loss: number; perCoord: number[] } {
  const p = bboxCenter(pred);
  const g = bboxCenter(gt);
  const deltas = [
    g.cx - p.cx,
    g.cy - p.cy,
    g.w - p.w,
    g.h - p.h,
  ];
  const perCoord = deltas.map((d) => {
    if (kind === "l1") return Math.abs(d);
    if (kind === "l2") return d * d;
    return smoothL1(d);
  });
  return { loss: perCoord.reduce((s, v) => s + v, 0), perCoord };
}

/* ---------- classification losses ---------- */

export function binaryCE(p: number, target: 0 | 1): number {
  const q = Math.min(Math.max(p, EPS), 1 - EPS);
  return target === 1 ? -Math.log(q) : -Math.log(1 - q);
}

/** Focal loss for a single (positive) sample: FL = -alpha (1-p)^gamma log(p). */
export function focalLoss(
  p: number,
  target: 0 | 1,
  gamma: number,
  alpha: number,
): number {
  const q = Math.min(Math.max(p, EPS), 1 - EPS);
  const pt = target === 1 ? q : 1 - q;
  const a = target === 1 ? alpha : 1 - alpha;
  return -a * Math.pow(1 - pt, gamma) * Math.log(pt);
}

export function focalLossStats(
  samples: { p: number; target: 0 | 1 }[],
  gamma: number,
  alpha: number,
) {
  let ce = 0;
  let fl = 0;
  const fg = samples.filter((s) => s.target === 1);
  const bg = samples.filter((s) => s.target === 0);
  for (const s of samples) {
    ce += binaryCE(s.p, s.target);
    fl += focalLoss(s.p, s.target, gamma, alpha);
  }
  const sum = samples.length;
  const fgCE = fg.reduce((a, s) => a + binaryCE(s.p, 1), 0);
  const fgFL = fg.reduce((a, s) => a + focalLoss(s.p, 1, gamma, alpha), 0);
  const bgCE = bg.reduce((a, s) => a + binaryCE(s.p, 0), 0);
  const bgFL = bg.reduce((a, s) => a + focalLoss(s.p, 0, gamma, alpha), 0);
  return {
    n: sum,
    fg: fg.length,
    bg: bg.length,
    ce,
    fl,
    fgCE,
    fgFL,
    bgCE,
    bgFL,
    ratioFg: fg.length / Math.max(1, bg.length),
  };
}

/* ---------- softmax / attention ---------- */

export function softmax(xs: NumArray): NumArray {
  const max = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - max));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

/** Scaled dot-product attention. Returns weights (over keys) and output. */
export function attention(
  q: NumArray,
  keys: NumArray[],
  values: NumArray[],
): { logits: number[]; weights: number[]; output: NumArray } {
  if (keys.length === 0) return { logits: [], weights: [], output: [] };
  const d = Math.max(1, q.length);
  const logits = keys.map((k) => q.reduce((s, qv, i) => s + qv * k[i], 0) / Math.sqrt(d));
  const weights = softmax(logits);
  const dim = values[0]?.length ?? 0;
  const output = Array.from({ length: dim }, (_, j) =>
    values.reduce((s, v, i) => s + weights[i] * (v[j] ?? 0), 0),
  );
  return { logits, weights, output };
}

/** Positional Gaussian-similarity attention for the self-attn vs conv sim. */
export function gaussianAttention(
  positions: NumArray,
  sigma: number,
): number[][] {
  return positions.map((p) => {
    const raw = positions.map((q) => Math.exp(-((p - q) ** 2) / (2 * sigma * sigma)));
    const s = raw.reduce((a, b) => a + b, 0);
    return raw.map((r) => r / s);
  });
}

/** 1D convolution (valid mode) for a fixed kernel over a signal. */
export function conv1d(signal: NumArray, kernel: NumArray, stride = 1): NumArray {
  const out: NumArray = [];
  const k = kernel.length;
  for (let i = 0; i + k <= signal.length; i += stride) {
    let s = 0;
    for (let j = 0; j < k; j++) s += signal[i + j] * kernel[j];
    out.push(s);
  }
  return out;
}

/* ---------- NMS ---------- */

export type NmsCandidate = {
  id: number;
  box: Box;
  score: number;
  cls: string;
};

export type NmsStep = {
  pick: number;
  suppressed: number[];
};

export function nms(
  candidates: NmsCandidate[],
  confThreshold: number,
  iouThreshold: number,
): { kept: number[]; steps: NmsStep[] } {
  let list = candidates
    .filter((c) => c.score >= confThreshold)
    .sort((a, b) => b.score - a.score);
  const kept: number[] = [];
  const steps: NmsStep[] = [];
  while (list.length > 0) {
    const [best, ...rest] = list;
    kept.push(best.id);
    const suppressed: number[] = [];
    const remaining: NmsCandidate[] = [];
    for (const c of rest) {
      if (c.cls === best.cls && iou(best.box, c.box) > iouThreshold) {
        suppressed.push(c.id);
      } else {
        remaining.push(c);
      }
    }
    steps.push({ pick: best.id, suppressed });
    list = remaining;
  }
  return { kept, steps };
}

/** DIoU-NMS (Zheng et al.): same loop, but suppression compares DIoU, which
 * additionally rewards spatially-separated centers, keeping distinct nearby objects. */
export function nmsDiou(
  candidates: NmsCandidate[],
  confThreshold: number,
  iouThreshold: number,
): { kept: number[]; steps: NmsStep[] } {
  let list = candidates
    .filter((c) => c.score >= confThreshold)
    .sort((a, b) => b.score - a.score);
  const kept: number[] = [];
  const steps: NmsStep[] = [];
  while (list.length > 0) {
    const [best, ...rest] = list;
    kept.push(best.id);
    const suppressed: number[] = [];
    const remaining: NmsCandidate[] = [];
    for (const c of rest) {
      if (c.cls === best.cls && diou(best.box, c.box) > iouThreshold) {
        suppressed.push(c.id);
      } else {
        remaining.push(c);
      }
    }
    steps.push({ pick: best.id, suppressed });
    list = remaining;
  }
  return { kept, steps };
}

/* ---------- Hungarian (exact, small n) ---------- */

/** Minimum-cost bipartite assignment: rows (predictions) to cols (GT).
 * Exact DP over subsets. Returns assignment rows->col, -1 for unmatched rows. */
export function hungarianMinCost(m: number[][], allowUnmatched = true): {
  assignment: number[];
  cost: number;
} {
  const rows = m.length;
  const cols = m[0]?.length ?? 0;
  const memo = new Map<string, { cost: number; assign: number[] }>();

  function dfs(r: number, used: number): { cost: number; assign: number[] } {
    if (r === rows) return { cost: 0, assign: [] };
    const key = `${r},${used}`;
    if (memo.has(key)) return memo.get(key)!;
    let best: { cost: number; assign: number[] } | null = null;
    for (let c = 0; c < cols; c++) {
      if ((used >> c) & 1) continue;
      const rest = dfs(r + 1, used | (1 << c));
      const cost = m[r][c] + rest.cost;
      if (!best || cost < best.cost) best = { cost, assign: [c, ...rest.assign] };
    }
    // allow leaving row unmatched with a high cost (rows can exceed cols)
    if (allowUnmatched && cols > 0) {
      const restUnmatched = dfs(r + 1, used);
      const cost = 10 + restUnmatched.cost; // large unmatched penalty
      if (!best || cost < best.cost) best = { cost, assign: [-1, ...restUnmatched.assign] };
    }
    if (!best) best = { cost: 10 * (rows - r), assign: Array.from({ length: rows - r }, () => -1) };
    memo.set(key, best);
    return best;
  }

  const { cost, assign } = dfs(0, 0);
  return { assignment: assign.length ? assign : Array.from({ length: rows }, () => -1), cost: parseFloat(cost.toFixed(4)) };
}

/* ---------- receptive field ---------- */

export type RfLayer = { kernel: number; stride: number; dilation: number };

export function receptiveField(layers: RfLayer[]): {
  rf: number;
  stride: number;
  perLayer: number[];
} {
  let rf = 1;
  let strideTotal = 1;
  const perLayer: number[] = [];
  for (const l of layers) {
    const effK = (l.kernel - 1) * l.dilation + 1;
    rf = rf + (effK - 1) * strideTotal;
    strideTotal *= l.stride;
    perLayer.push(rf);
  }
  return { rf, stride: strideTotal, perLayer };
}

/* ---------- anchors ---------- */

export type Anchor = { x1: number; y1: number; x2: number; y2: number; scale: number; ratio: number };

export function generateAnchors(
  gridW: number,
  gridH: number,
  cellSize: number,
  scales: number[],
  ratios: number[],
): Anchor[] {
  const anchors: Anchor[] = [];
  const base = cellSize;
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const cx = gx * cellSize + cellSize / 2;
      const cy = gy * cellSize + cellSize / 2;
      for (const s of scales) {
        for (const r of ratios) {
          const w = base * s * Math.sqrt(r);
          const h = (base * s) / Math.sqrt(r);
          anchors.push({
            x1: cx - w / 2,
            y1: cy - h / 2,
            x2: cx + w / 2,
            y2: cy + h / 2,
            scale: s,
            ratio: r,
          });
        }
      }
    }
  }
  return anchors;
}

export function bestAnchorsForBox(anchors: Anchor[], box: Box, topK = 3) {
  const scored = anchors
    .map((a, i) => ({ a, i, iou: iou(a, box) }))
    .filter((s) => s.iou > 0)
    .sort((a, b) => b.iou - a.iou)
    .slice(0, topK);
  return scored;
}

/* ---------- distribution box regression (D-FINE style) ---------- */

/** Decode a box coordinate from a distribution over W bins. */
export function distributionDecode(
  probs: NumArray,
  lower: number,
  upper: number,
): number {
  const W = probs.length;
  const step = (upper - lower) / W;
  let sum = 0;
  for (let k = 0; k < W; k++) sum += probs[k] * (lower + (k + 0.5) * step);
  return sum;
}

export function uniformTarget(dist: NumArray, target: number, lower: number, upper: number): NumArray {
  const W = dist.length;
  const step = (upper - lower) / W;
  const k = Math.max(0, Math.min(W - 1, Math.floor((target - lower) / step)));
  const out = dist.map(() => 0);
  // distribute probability over the two nearest bins for a soft target
  const frac = (target - (lower + k * step)) / step;
  out[k] = 1 - frac;
  if (k + 1 < W) out[k + 1] = frac;
  return out;
}

/* ---------- seeded RNG (mulberry32) ---------- */

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

/* ---------- dynamic label assignment (SimOTA-style topK) ---------- */

/** For every GT column, its top-k predictions (lowest cost) become positives.
 *  k=1 reproduces a one-to-one regime; k>1 admits many-to-one dense supervision. */
export function topKAssignment(m: number[][], k: number): {
  positives: boolean[][];
  assignment: number[];
  nPairs: number;
  collisions: number;
} {
  const rows = m.length;
  const cols = m[0]?.length ?? 0;
  const positives: boolean[][] = Array.from({ length: rows }, () => Array(cols).fill(false));
  for (let c = 0; c < cols; c++) {
    const ranked = Array.from({ length: rows }, (_, r) => r).sort((a, b) => m[a][c] - m[b][c]);
    const take = Math.min(k, rows);
    for (let i = 0; i < take; i++) positives[ranked[i]][c] = true;
  }
  const assignment = positives.map((row, r) => {
    let best = -1;
    let bv = Infinity;
    for (let c = 0; c < cols; c++) {
      if (row[c] && m[r][c] < bv) {
        bv = m[r][c];
        best = c;
      }
    }
    return best;
  });
  const perGt = Array(cols).fill(0);
  for (const c of assignment) if (c >= 0) perGt[c]++;
  const collisions = perGt.filter((n) => n > 1).length;
  const nPairs = positives.reduce((s, row) => s + row.filter(Boolean).length, 0);
  return { positives, assignment, nPairs, collisions };
}

/* ---------- anchor vs anchor-free location strands ---------- */

export type LocationStrand = {
  cx: number;
  cy: number;
  w: number;
  h: number;
  anchorCost: number; // 1 - IoU(anchor, gt)
  freeCost: number; // normalized center distance cost
  centerInGt: boolean;
};

/** Anchor-based strands score every tiled anchor by IoU with the GT;
 *  anchor-free strands score the center cell only (plus scale reach). */
export function locationStrands(gt: Box, anchors: Anchor[]): {
  strands: LocationStrand[];
  anchorPositives: number; // IoU > 0.5
  freePositives: number; // centers inside the GT
  gtArea: number;
} {
  const g = bboxCenter(gt);
  const diag = Math.hypot(gt.x2 - gt.x1, gt.y2 - gt.y1) || 1;
  const strands: LocationStrand[] = anchors.map((a) => {
    const ac = bboxCenter(a);
    const dist = Math.hypot(ac.cx - g.cx, ac.cy - g.cy) / diag;
    const inGt =
      gt.x1 <= ac.cx && ac.cx <= gt.x2 && gt.y1 <= ac.cy && ac.cy <= gt.y2;
    return {
      cx: ac.cx,
      cy: ac.cy,
      w: a.x2 - a.x1,
      h: a.y2 - a.y1,
      anchorCost: 1 - iou(a, gt),
      freeCost: dist,
      centerInGt: inGt,
    };
  });
  return {
    strands,
    anchorPositives: strands.filter((s) => s.anchorCost < 0.5).length,
    freePositives: strands.filter((s) => s.centerInGt).length,
    gtArea: area(gt),
  };
}

/* ---------- cascade refinement (Cascade R-CNN) ---------- */

export type CascadeStage = {
  stage: number;
  iouVal: number;
  threshold: number;
  deltaArea: number;
  box: Box;
};

const CASCADE_THRESHOLDS = [0.5, 0.6, 0.7];

/** Each stage is trained at a higher IoU threshold and refines the box,
 *  with noisy regression scaled by how far (1 - IoU) still is. */
export function cascadeRefine(init: Box, gt: Box, seed: number): CascadeStage[] {
  const rng = mulberry32(seed);
  const out: CascadeStage[] = [];
  let cur = { ...init };
  for (let s = 0; s < CASCADE_THRESHOLDS.length; s++) {
    const before = iou(cur, gt);
    const progress = Math.min(0.92, 0.45 + (1 - before) * 0.55);
    const noise = 0.18 * (1 - before) * (0.6 + rng() * 0.8);
    const g = bboxCenter(gt);
    const c = bboxCenter(cur);
    const nx = c.cx + (g.cx - c.cx) * progress + (rng() - 0.5) * noise * Math.abs(g.cx - c.cx);
    const ny = c.cy + (g.cy - c.cy) * progress + (rng() - 0.5) * noise * Math.abs(g.cy - c.cy);
    const nw = c.w + (g.w - c.w) * progress;
    const nh = c.h + (g.h - c.h) * progress;
    cur = {
      x1: nx - nw / 2,
      y1: ny - nh / 2,
      x2: nx + nw / 2,
      y2: ny + nh / 2,
    };
    const after = iou(cur, gt);
    out.push({
      stage: s + 1,
      iouVal: after,
      threshold: CASCADE_THRESHOLDS[s],
      deltaArea: after - before,
      box: cur,
    });
  }
  return out;
}

/* ---------- query denoising (DN-DETR) ---------- */

export type DenoisingResult = {
  baselineCost: number; // random queries vs GT
  denoisedCost: number; // corrupted-GT queries vs GT
  reductionPct: number;
  matchedIou: number; // matched-pair IoU in the denoised regime
};

/** Hungarian matching cost of random queries vs corrupted-GT queries against
 *  the ground truth; corruption is a fraction of the box size. */
export function denoisingEffect(gtBoxes: Box[], noise: number, seed: number): DenoisingResult {
  const rng = mulberry32(seed);

  const randomQueries = gtBoxes.map((g) => {
    const c = bboxCenter(g);
    const w = g.x2 - g.x1;
    const cx = c.cx + (rng() - 0.5) * 220;
    const cy = c.cy + (rng() - 0.5) * 140;
    const nw = w * (0.6 + rng() * 1.2);
    const nh = nw * 0.8;
    return { x1: cx - nw / 2, y1: cy - nh / 2, x2: cx + nw / 2, y2: cy + nh / 2 };
  });
  const corrupted = gtBoxes.map((g) => {
    const c = bboxCenter(g);
    const w = g.x2 - g.x1;
    const h = g.y2 - g.y1;
    const off = noise * Math.sqrt(w * h);
    const cx = c.cx + (rng() * 2 - 1) * off;
    const cy = c.cy + (rng() * 2 - 1) * off;
    const nw = w * (1 + (rng() * 2 - 1) * 0.3 * noise);
    const nh = h * (1 + (rng() * 2 - 1) * 0.3 * noise);
    return { x1: cx - nw / 2, y1: cy - nh / 2, x2: cx + nw / 2, y2: cy + nh / 2 };
  });

  const oneToOneCost = (a: Box[], b: Box[]) =>
    a.map((x) => b.map((y) => 1 - iou(x, y)));

  const { cost: baselineCost } = hungarianMinCost(oneToOneCost(randomQueries, gtBoxes));
  const { cost: denoisedCost } = hungarianMinCost(oneToOneCost(corrupted, gtBoxes));

  const matchedIou =
    corrupted.map((q, i) => iou(q, gtBoxes[i])).reduce((a, b) => a + b, 0) /
    Math.max(1, corrupted.length);
  const reductionPct = baselineCost <= 0 ? 0 : ((baselineCost - denoisedCost) / baselineCost) * 100;
  return { baselineCost, denoisedCost, reductionPct, matchedIou };
}

/* ---------- deformable attention (sparse sampled) ---------- */

export type DeformablePoint = { p: NumArray; label: string };

/** Sample the top-K of N potential points for a query; dense attention would
 *  weigh all N. Returns weights on the sample, the FLOP-cost of each mode and
 *  the concentration (max weight) of the sparse focus. */
export function deformableAttention(
  q: NumArray,
  points: DeformablePoint[],
  K: number,
): { sampled: number[]; weights: number[]; costDense: number; costSparse: number; focus: number } {
  const d = Math.max(1, q.length);
  const score = points.map((pt) => q.reduce((s, qv, i) => s + qv * (pt.p[i] ?? 0), 0) / Math.sqrt(d));
  const order = points.map((_, i) => i).sort((a, b) => score[b] - score[a]);
  const sampled = order.slice(0, Math.min(K, points.length)).sort((a, b) => a - b);
  const weights = softmax(sampled.map((i) => score[i]));
  return {
    sampled,
    weights,
    costDense: points.length,
    costSparse: sampled.length,
    focus: Math.max(...weights),
  };
}

/* ---------- mosaic augmentation ---------- */

export type MosaicResult = {
  boxes: { box: Box; q: number; w: number }[];
  smallCount: number;
  total: number;
  densityRatio: number; // objects compared to the average quadrant before mixing
};

/** Stitch 2×2 quadrants into one larger canvas, halving each quadrant and its
 *  boxes, then reporting small-box counts and density. */
export function mosaicAugment(
  quadrants: Box[][],
  W: number,
  H: number,
): MosaicResult {
  const tw = W / 2;
  const th = H / 2;
  const offsets = [
    { x: 0, y: 0 },
    { x: tw, y: 0 },
    { x: 0, y: th },
    { x: tw, y: th },
  ];
  const boxes = quadrants.slice(0, 4).flatMap((qs, q) =>
    qs.map((b) => {
      const ocx = b.x1 + (b.x2 - b.x1) / 2;
      const ocy = b.y1 + (b.y2 - b.y1) / 2;
      const nw = (b.x2 - b.x1) / 2;
      const nh = (b.y2 - b.y1) / 2;
      const cx = offsets[q].x + ocx / 2;
      const cy = offsets[q].y + ocy / 2;
      return { box: { x1: cx - nw / 2, y1: cy - nh / 2, x2: cx + nw / 2, y2: cy + nh / 2 }, q, w: nw };
    }),
  );
  const small = boxes.filter((b) => b.w < 8).length;
  const avgPerQuadrant = quadrants.slice(0, 4).reduce((s, qs) => s + qs.length, 0) / Math.max(1, Math.min(4, quadrants.length));
  return {
    boxes,
    smallCount: small,
    total: boxes.length,
    densityRatio: boxes.length / Math.max(1, avgPerQuadrant),
  };
}