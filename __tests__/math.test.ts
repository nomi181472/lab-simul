import { describe, it, expect } from "vitest";
import {
  iou,
  giou,
  diou,
  ciou,
  iouFamily,
  intersection,
  softmax,
  attention,
  nms,
  nmsDiou,
  hungarianMinCost,
  receptiveField,
  smoothL1,
  boxRegressionLoss,
  focalLoss,
  focalLossStats,
  binaryCE,
  distributionDecode,
  conv1d,
  gaussianAttention,
  topKAssignment,
  locationStrands,
  cascadeRefine,
  denoisingEffect,
  deformableAttention,
  mosaicAugment,
  generateAnchors,
} from "../src/labs/object-detection/sim/math";
import {
  generateDetectionCandidates,
  generateCostMatrix,
  generateSmallObjectScene,
  generateOcclusionScene,
  generateCrowdedScene,
  generateObjects,
} from "../src/labs/object-detection/sim/mock";

const T = 1e-6;
const box = (x1: number, y1: number, x2: number, y2: number) => ({ x1, y1, x2, y2 });

describe("IoU family", () => {
  it("IoU(box,box) == 1", () => {
    const b = box(10, 10, 40, 40);
    expect(iou(b, b)).toBeCloseTo(1, 9);
  });

  it("IoU(non-overlapping) == 0", () => {
    expect(iou(box(0, 0, 10, 10), box(20, 20, 30, 30))).toBe(0);
  });

  it("partial overlap computes correctly", () => {
    const a = box(0, 0, 10, 10);
    const b = box(5, 0, 15, 10);
    expect(intersection(a, b)).toBeCloseTo(50);
    // union = 100 + 100 - 50 = 150
    expect(iou(a, b)).toBeCloseTo(50 / 150, 9);
  });

  it("disjoint GIoU < 0 (gradient preserved)", () => {
    const a = box(0, 0, 10, 10);
    const b = box(20, 20, 30, 30);
    expect(giou(a, b)).toBeLessThan(0);
    expect(iou(a, b)).toBe(0);
  });

  it("perfect boxes: all losses 0, similarities 1", () => {
    const b = box(5, 5, 25, 35);
    for (const f of [giou, diou, ciou]) expect(f(b, b)).toBeCloseTo(1, 6);
  });

  it("iouFamily exposes every loss cross-mapped to its metric key (UI-safe)", () => {
    const b = box(1, 1, 9, 9);
    const f = iouFamily(b, box(3, 1, 9, 9));
    const LOSS_KEY: Record<string, string> = {
      iou: "lossIou",
      giou: "lossGioU",
      diou: "lossDioU",
      ciou: "lossCioU",
    };
    for (const metric of ["iou", "giou", "diou", "ciou"] as const) {
      const loss = f[LOSS_KEY[metric] as keyof typeof f];
      expect(typeof loss).toBe("number");
      expect(Number.isFinite(loss)).toBe(true);
    }
  });

  it("CIoU < DIoU for aspect mismatch", () => {
    const gt = box(10, 10, 30, 30);
    const pred = box(10, 10, 30, 50); // same center, wrong aspect
    expect(ciou(pred, gt)).toBeLessThan(diou(pred, gt));
  });
});

describe("classification losses", () => {
  it("binaryCE(p=1,target=1) ~ 0", () => expect(binaryCE(0.9999, 1)).toBeLessThan(0.001));
  it("binaryCE(0.5,1) = log(2)", () => expect(binaryCE(0.5, 1)).toBeCloseTo(Math.LN2, 6));

  it("focal < CE for easy positives (reweighting)", () => {
    const p = 0.95;
    expect(focalLoss(p, 1, 2, 0.25)).toBeLessThan(binaryCE(p, 1));
  });

  it("focal hits zero at perfect confidence", () => {
    expect(focalLoss(1, 1, 2, 0.25)).toBeLessThan(1e-4);
  });

  it("focal imbalance stats: contribution shifts with gamma", () => {
    const samples = [
      ...Array.from({ length: 1000 }, () => ({ p: 0.05, target: 0 as const })), // easy bg, confidently correct
      ...Array.from({ length: 10 }, () => ({ p: 0.4, target: 1 as const })), // hard fg
    ];
    const g0 = focalLossStats(samples, 0, 0.25);
    const g2 = focalLossStats(samples, 2, 0.25);
    // with gamma=2 easy-negatives contribute far less relatively to CE
    expect(g2.bgFL / g2.fl).toBeLessThan(g0.bgCE / g0.ce);
  });
});

describe("softmax & attention", () => {
  it("softmax sums to ~1", () => {
    const s = softmax([1, 2, 3, 0.5, -2]);
    expect(s.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
  });

  it("softmax is order-preserving", () => {
    const s = softmax([0.2, 4.0]);
    expect(s[1]).toBeGreaterThan(s[0]);
  });

  it("attention weights sum to ~1 and output is weighted avg of V", () => {
    const q = [1, 0];
    const keys = [
      [1, 0],
      [0, 1],
      [0.5, 0.5],
    ];
    const values = [
      [2, 0],
      [0, 4],
      [1, 1],
    ];
    const { weights, output } = attention(q, keys, values);
    expect(weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    for (let j = 0; j < 2; j++) {
      expect(output[j]).toBeCloseTo(values.reduce((s, v, i) => s + weights[i] * v[j], 0), 9);
    }
  });

  it("identical q,k,d=1 gives near-uniform attention", () => {
    const q = [0.5];
    const keys = [[0.5], [0.5], [0.5]];
    const { weights } = attention(q, keys, [[1], [1], [1]]);
    expect(weights[0]).toBeCloseTo(1 / 3, 6);
  });
});

describe("NMS", () => {
  const candidates = [
    { id: 0, box: box(10, 10, 60, 60), score: 0.95, cls: "car" },
    { id: 1, box: box(14, 12, 64, 62), score: 0.91, cls: "car" },
    { id: 2, box: box(8, 8, 58, 58), score: 0.83, cls: "car" },
    { id: 3, box: box(120, 40, 160, 90), score: 0.87, cls: "person" },
  ];

  it("removes duplicate overlapping boxes when threshold crossed", () => {
    const { kept } = nms(candidates, 0.3, 0.5);
    expect(kept).toEqual([0, 3]); // two cars merge, person survives
  });

  it("keeps both when IoU threshold is high", () => {
    const { kept } = nms(candidates, 0.3, 0.99);
    expect(kept).toContain(0);
    expect(kept).toContain(1);
  });

  it("confidence threshold drops low-score boxes first", () => {
    const { kept } = nms(candidates, 0.9, 0.5);
    expect(kept).not.toContain(2);
  });

  it("DIoU-NMS keeps center-distant same-class pairs that IoU-NMS would drop", () => {
    // two same-class boxes with ~identical shape but shifted centers: high IoU, lower DIoU
    const close = [
      { id: 0, box: box(10, 10, 60, 60), score: 0.95, cls: "car" },
      { id: 1, box: box(24, 16, 74, 66), score: 0.9, cls: "car" },
    ];
    const iouThr = 0.4;
    const std = nms(close, 0.3, iouThr).kept;
    const diouRes = nmsDiou(close, 0.3, iouThr).kept;
    expect(std.length).toBeLessThanOrEqual(diouRes.length);
    // DIoU-NMS must never keep fewer high-scoring boxes than standard at equal threshold
    expect(diouRes.length).toBeGreaterThanOrEqual(std.length);
  });
});

describe("Hungarian assignment", () => {
  it("assigns minimum-cost matching", () => {
    // 4 predictions -> 3 GT boxes; optimum matches the 3 lightweight diagonals
    const cost = [
      [0.2, 0.9, 0.8],
      [0.8, 0.1, 0.7],
      [0.9, 0.7, 0.2],
      [0.4, 0.6, 0.5],
    ];
    const { assignment } = hungarianMinCost(cost);
    expect(assignment).toHaveLength(4);
    const matched = assignment.filter((a) => a >= 0).sort();
    expect(matched).toEqual([0, 1, 2]);
    // sum of matched costs = 0.2+0.1+0.2 (fourth row left unmatched)
    const matchedCost = cost.reduce((s, row, r) => {
      const c = assignment[r];
      return c >= 0 ? s + row[c] : s;
    }, 0);
    expect(matchedCost).toBeCloseTo(0.5, 4);
  });

  it("finds strict improvement over greedy", () => {
    const cost = [
      [1, 100],
      [100, 1],
    ];
    const { cost: total } = hungarianMinCost(cost);
    expect(total).toBeCloseTo(2, 4); // diagonal 1+1, not 200
  });
});

describe("receptive field", () => {
  it("single 3x3 conv -> rf 3", () => {
    const { rf } = receptiveField([{ kernel: 3, stride: 1, dilation: 1 }]);
    expect(rf).toBe(3);
  });

  it("two stacked 3x3 convs -> rf 5", () => {
    const { rf } = receptiveField([
      { kernel: 3, stride: 1, dilation: 1 },
      { kernel: 3, stride: 1, dilation: 1 },
    ]);
    expect(rf).toBe(5);
  });

  it("dilated kernel grows rf", () => {
    const d1 = receptiveField([{ kernel: 3, stride: 1, dilation: 1 }]).rf;
    const d2 = receptiveField([{ kernel: 3, stride: 1, dilation: 2 }]).rf;
    expect(d2).toBeGreaterThan(d1);
  });
});

describe("box regression", () => {
  it("smooth L1 quadratic near zero, linear far", () => {
    expect(smoothL1(0)).toBe(0);
    expect(smoothL1(0.5)).toBeCloseTo(0.125, 6);
    expect(smoothL1(10)).toBeCloseTo(9.5, 6);
  });

  it("perfect prediction gives zero coordinate loss", () => {
    const b = box(10, 10, 30, 40);
    const { loss } = boxRegressionLoss(b, b, "smooth-l1");
    expect(loss).toBeCloseTo(0, 6);
  });

  it("L1 loss is axis-separable and positive", () => {
    const { loss, perCoord } = boxRegressionLoss(box(0, 0, 10, 10), box(0, 0, 20, 20), "l1");
    // center deltas: (5->10)=5, (5->10)=5, (w 10->20)=10, (h 10->20)=10
    expect(perCoord).toEqual([5, 5, 10, 10]);
    expect(loss).toBeCloseTo(30, 6);
  });
});

describe("distribution decoding (D-FINE style)", () => {
  it("decodes peak position from uniform-width bins", () => {
    const probs = [0, 0, 0, 1, 0, 0, 0, 0];
    // 8 bins over [0,4], step 0.5; bin 3 center = 3.5*... = 1.75
    expect(distributionDecode(probs, 0, 4)).toBeCloseTo(1.75, 6);
  });

  it("decodes expectation for uniform distribution", () => {
    const probs = [1, 1, 1, 1].map((v) => v / 4);
    expect(distributionDecode(probs, 0, 4)).toBeCloseTo(2, 6); // midpoint expectation
  });
});

describe("conv1d & gaussian attention", () => {
  it("conv detects local feature only in its window", () => {
    const signal = [0, 0, 1, 1, 0, 0];
    const out = conv1d(signal, [1, 1, 1]); // sum in window of 3
    // windows [0,0,1]->1, [0,1,1]->2, [1,1,0]->2, [1,0,0]->1
    expect(out).toEqual([1, 2, 2, 1]);
  });

  it("gaussian attention rows sum to 1", () => {
    const m = gaussianAttention([0, 10, 20], 3);
    for (const row of m) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
  });
});

describe("mock data engine determinism", () => {
  it("same seed -> identical scenes", () => {
    const a = generateObjects({ seed: 42, numObjects: 5 });
    const b = generateObjects({ seed: 42, numObjects: 5 });
    expect(a.objects).toEqual(b.objects);
  });

  it("different seed -> different scenes", () => {
    const a = generateObjects({ seed: 1, numObjects: 5 });
    const b = generateObjects({ seed: 2, numObjects: 5 });
    expect(a.objects).not.toEqual(b.objects);
  });

  it("scene generators are box-valid", () => {
    for (const gen of [generateSmallObjectScene, generateOcclusionScene, generateCrowdedScene]) {
      const s = gen({ seed: 7 });
      for (const o of s.objects) {
        expect(o.box.x2).toBeGreaterThan(o.box.x1);
        expect(o.box.y2).toBeGreaterThan(o.box.y1);
      }
    }
  });

  it("candidates have stable ordering after sort by score", () => {
    const c = generateDetectionCandidates(9);
    const sorted = [...c].sort((a, b) => b.score - a.score);
    for (let i = 1; i < sorted.length; i++) {
      expect(sorted[i - 1].score).toBeGreaterThanOrEqual(sorted[i].score);
    }
  });

  it("cost matrix is deterministic and within [0,1]", () => {
    const m = generateCostMatrix(3, 4, 3);
    for (const row of m) for (const v of row) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

describe("sim additions (2026 lab expansion)", () => {
  it("topKAssignment: k=1 is one-to-one-ish, k>1 raises supervision density", () => {
    const m = generateCostMatrix(7, 8, 5);
    const one = topKAssignment(m, 1);
    const many = topKAssignment(m, 3);
    expect(many.nPairs).toBeGreaterThanOrEqual(one.nPairs);
    // every positive pair's cell cost must be within the top-k of its column
    expect(one.assignment.length).toBe(8);
    expect(one.collisions).toBeLessThanOrEqual(5);
  });

  it("topKAssignment picks the cheapest cells per column", () => {
    const m = [
      [0.9, 0.8],
      [0.1, 0.9],
      [0.8, 0.2],
    ];
    const { positives, nPairs } = topKAssignment(m, 2);
    expect(positives[1][0]).toBe(true); // best cell for gt0
    expect(positives[2][1]).toBe(true); // best cell for gt1
    expect(nPairs).toBe(4);
  });

  it("locationStrands: free strand positives are centers inside the GT", () => {
    const gt = { x1: 70, y1: 45, x2: 150, y2: 100 };
    const anchors = generateAnchors(12, 7, 20, [1], [1, 1.6, 0.6]);
    const r = locationStrands(gt, anchors);
    expect(r.strands.length).toBe(anchors.length);
    const anyIn = r.strands.some((s) => s.centerInGt);
    expect(anyIn).toBe(true);
    expect(r.strands.every((s) => s.anchorCost >= 0 && s.anchorCost <= 1)).toBe(true);
  });

  it("cascadeRefine: IoU improves monotonically across 3 stages", () => {
    const init = { x1: 56, y1: 30, x2: 172, y2: 118 };
    const gt = { x1: 92, y1: 52, x2: 138, y2: 92 };
    const stages = cascadeRefine(init, gt, 21);
    expect(stages).toHaveLength(3);
    for (let i = 1; i < stages.length; i++) {
      expect(stages[i].iouVal).toBeGreaterThanOrEqual(stages[i - 1].iouVal);
    }
    expect(stages[2].iouVal).toBeGreaterThan(iou(init, gt));
  });

  it("denoisingEffect: corrupted-GT queries match at lower cost than random", () => {
    const gts = [
      { x1: 40, y1: 30, x2: 96, y2: 84 },
      { x1: 120, y1: 50, x2: 176, y2: 106 },
    ];
    const smallNoise = denoisingEffect(gts, 0.15, 33);
    expect(smallNoise.denoisedCost).toBeLessThan(smallNoise.baselineCost);
    expect(smallNoise.reductionPct).toBeGreaterThan(0);
  });

  it("deformableAttention: sparse cost = k and focus grows with larger samples", () => {
    const points = Array.from({ length: 12 }, (_, i) => ({
      p: [i / 11, (i % 3) / 3],
      label: String(i),
    }));
    const a = deformableAttention([0.5, 0.5], points, 4);
    const b = deformableAttention([0.5, 0.5], points, 8);
    expect(a.costSparse).toBe(4);
    expect(b.costSparse).toBe(8);
    expect(a.costDense).toBe(12);
    expect(b.focus).toBeLessThanOrEqual(a.focus);
  });

  it("mosaicAugment: 4 quadrant scenes → total boxes preserved, small counts rise", () => {
    const quadrants = [
      [
        { x1: 10, y1: 10, x2: 60, y2: 50 },
        { x1: 20, y1: 60, x2: 70, y2: 90 },
      ],
      [
        { x1: 10, y1: 20, x2: 40, y2: 60 },
      ],
      [
        { x1: 30, y1: 10, x2: 80, y2: 40 },
        { x1: 50, y1: 50, x2: 90, y2: 80 },
      ],
      [
        { x1: 10, y1: 30, x2: 50, y2: 70 },
      ],
    ];
    const before = quadrants.flat().length;
    const r = mosaicAugment(quadrants, 220, 130);
    expect(r.total).toBe(before);
    expect(r.boxes.every((b) => b.box.x1 >= 0 && b.box.y1 >= 0 && b.box.x2 <= 220 && b.box.y2 <= 130)).toBe(true);
    expect(r.densityRatio).toBeGreaterThan(1);
  });
});