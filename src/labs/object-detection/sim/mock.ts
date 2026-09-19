import type { Box } from "./math";
import { mulberry32, type NmsCandidate } from "./math";

/* Deterministic mock-data engine. Every generator accepts a seed so all
 * simulators are reproducible across reloads / user sessions.
 */

export type MockSceneOptions = {
  seed: number;
  numObjects?: number;
  noise?: number;
  objectSize?: number;
  overlap?: number;
  confidence?: number;
};

export type EdgeScene = {
  W: number;
  H: number;
  objects: {
    id: number;
    box: Box;
    cls: string;
    visible: number; // 0..1 occlusion factor
    confidence: number;
  }[];
  difficulty: string[];
};

const CLASSES = ["car", "person", "bike", "truck", "dog", "boat", "plane", "cup"];

export function randBox(rng: () => number, minW: number, maxW: number): Box {
  const w = minW + rng() * (maxW - minW);
  const h = w * (0.7 + rng() * 0.7);
  const x = 12 + rng() * (200 - w - 24);
  const y = 12 + rng() * (120 - h - 24);
  return { x1: x, y1: y, x2: x + w, y2: y + h };
}

export function generateObjects(opts: MockSceneOptions) {
  const { seed, numObjects = 6, objectSize = 0.5 } = opts;
  const rng = mulberry32(seed);
  const W = 220;
  const H = 140;
  const minW = 14 + objectSize * 40;
  const maxW = 26 + objectSize * 70;
  const objects: { id: number; box: Box; cls: string; confidence: number }[] = [];
  for (let i = 0; i < numObjects; i++) {
    objects.push({
      id: i,
      box: randBox(rng, minW, maxW),
      cls: CLASSES[Math.floor(rng() * CLASSES.length)],
      confidence: 0.5 + rng() * 0.48,
    });
  }
  return { W, H, objects };
}

export function generateOcclusionScene(opts: MockSceneOptions): EdgeScene {
  const { seed, numObjects = 4, objectSize = 0.6 } = opts;
  const rng = mulberry32(seed);
  const W = 220;
  const H = 140;
  const base = generateObjects({ seed, numObjects, objectSize });
  const objects = base.objects.map((o, i) => {
    const visible = i === 0 ? 1 : 0.25 + rng() * 0.7;
    return { ...o, visible: Math.max(0, Math.min(1, visible)) };
  });
  const difficulty = [
    "Occluders reduce the visible footprint of boxes.",
    "Features vanish progressively with the visible fraction.",
    "Research responses: better NMS (DIoU/Matrix), context fusion, and occlusion-robust tracking.",
  ];
  return { W, H, objects, difficulty };
}

export function generateCrowdedScene(opts: MockSceneOptions): EdgeScene {
  const { seed, numObjects = 9 } = opts;
  const rng = mulberry32(seed);
  const W = 220;
  const H = 140;
  const objects = [];
  for (let i = 0; i < numObjects; i++) {
    const w = 26 + rng() * 30;
    const h = w * (0.8 + rng() * 0.3);
    const x = 8 + rng() * (W - w - 16);
    const y = 8 + rng() * (H - h - 16);
    objects.push({
      id: i,
      box: { x1: x, y1: y, x2: x + w, y2: y + h },
      cls: rng() > 0.4 ? "person" : "car",
      visible: 1,
      confidence: 0.55 + rng() * 0.4,
    });
  }
  const difficulty = [
    "Many overlapping instances confound NMS.",
    "One-to-many dense overlaps: duplicates suppressed too aggressively.",
    "Research responses: DIoU-NMS, Matrix NMS, then NMS-free one-to-one set prediction.",
  ];
  return { W, H, objects, difficulty };
}

export function generateSmallObjectScene(opts: MockSceneOptions): EdgeScene {
  const { seed, numObjects = 5 } = opts;
  const rng = mulberry32(seed);
  const W = 220;
  const H = 140;
  const objects = [];
  for (let i = 0; i < numObjects; i++) {
    const w = 4 + rng() * 6;
    const h = 4 + rng() * 6;
    const x = 8 + rng() * (W - w - 16);
    const y = 8 + rng() * (H - h - 16);
    objects.push({
      id: i,
      box: { x1: x, y1: y, x2: x + w, y2: y + h },
      cls: "tiny",
      visible: 1,
      confidence: 0.4 + rng() * 0.4,
    });
  }
  const difficulty = [
    "Tiny boxes occupy few feature-map cells after stride.",
    "SSD: 'small objects may not even have any information' in deep maps.",
    "Research responses: multi-scale maps, feature pyramids, dense supervision, STAL coverage.",
  ];
  return { W, H, objects, difficulty };
}

export function generateClassificationSamples(
  seed: number,
  fg: number,
  bg: number,
  easy: number,
): { p: number; target: 0 | 1 }[] {
  const rng = mulberry32(seed);
  const samples: { p: number; target: 0 | 1 }[] = [];
  for (let i = 0; i < fg; i++) {
    samples.push({ p: 0.2 + rng() * 0.45, target: 1 });
  }
  for (let i = 0; i < bg; i++) {
    samples.push({ p: (1 - easy) + rng() * 0.2, target: 0 });
  }
  return samples;
}

export function generateDetectionCandidates(seed: number): NmsCandidate[] {
  const rng = mulberry32(seed);
  const defs = [
    { cls: "car", cx: 55, cy: 55, w: 46, score: 0.95 },
    { cls: "car", cx: 62, cy: 57, w: 44, score: 0.91 },
    { cls: "car", cx: 70, cy: 62, w: 48, score: 0.83 },
    { cls: "person", cx: 140, cy: 60, w: 26, score: 0.87 },
    { cls: "person", cx: 148, cy: 66, w: 30, score: 0.7 },
  ];
  return defs.map((d, i) => ({
    id: i,
    box: {
      x1: d.cx - d.w / 2 + rng() * 6 - 3,
      y1: d.cy - d.w * 1.1 / 2 + rng() * 6 - 3,
      x2: d.cx + d.w / 2 + rng() * 6 - 3,
      y2: d.cy + d.w * 1.1 / 2 + rng() * 6 - 3,
    },
    score: Math.min(0.99, d.score),
    cls: d.cls,
  }));
}

export function generateCostMatrix(
  seed: number,
  nPreds: number,
  nGts: number,
): number[][] {
  const rng = mulberry32(seed);
  const m: number[][] = [];
  for (let r = 0; r < nPreds; r++) {
    const row: number[] = [];
    for (let c = 0; c < nGts; c++) {
      row.push(parseFloat((rng() * 0.75 + 0.05).toFixed(2)));
    }
    m.push(row);
  }
  return m;
}

export function generateTokens(
  seed: number,
  n: number,
  dim: number,
): { tokens: number[][]; labels: string[] } {
  const rng = mulberry32(seed);
  const tokens = Array.from({ length: n }, () =>
    Array.from({ length: dim }, () => parseFloat((rng() * 2 - 1).toFixed(3))),
  );
  const labels = ["sky", "street", "car", "pedestrian", "sign", "building"].slice(0, n);
  return { tokens, labels };
}

export function generateFeatureMap(
  seed: number,
  W: number,
  H: number,
  cells = 8,
): { grid: number[][]; scaleByRows: number[] } {
  const rng = mulberry32(seed);
  const grid = Array.from({ length: cells }, () =>
    Array.from({ length: Math.floor((cells * W) / H) }, () => rng() * 2 - 1),
  );
  return { grid, scaleByRows: grid.map((_, i) => Math.pow(2, i % 3)) };
}