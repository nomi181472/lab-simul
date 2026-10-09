"use client";

import { useMemo, useState } from "react";
import { PAPERS, PAPER_BY_ID } from "@/labs/object-detection/data/papers";
import { SOLUTION_BY_ID } from "@/labs/object-detection/data/solutions";
import { PROBLEM_BY_ID } from "@/labs/object-detection/data/problems";
import { Card, PaperLink, Badge, InsufficientEvidence } from "@/labs/object-detection/ui";

type Answer = {
  text: string;
  papers: string[];
  source: "rule" | "search" | "empty";
};

/* Rules are keyed by lowercase substrings; their papers are never hardcoded —
 * they are the union of the cited solutions' paperIds and the cited problems'
 * occurrence papers, capped at 6 and sorted newest first. The fallback searches
 * the papers' own titles, summaries and contributions. */

type Rule = {
  keys: string[];
  text: string;
  solutionIds?: string[];
  problemIds?: string[];
};

const RULES: Rule[] = [
  {
    keys: ["focal loss", "focal", "class imbalance", "class-imbalance", "imbalance"],
    text: "Focal loss reshapes cross-entropy with a modulating factor (1 − pt)^γ (α-balanced in practice) that decays the loss of easy examples toward zero, so training over roughly 100k dense anchors is dominated by hard examples without any sampling heuristic. The corpus sets it against the class-imbalance problem: one-stage detectors applied over dense samplings of locations, scales and aspect ratios trailed their two-stage counterparts.",
    solutionIds: ["focal-loss"],
    problemIds: ["class-imbalance"],
  },
  {
    keys: ["nms", "non-maximum", "non maximum", "suppression", "duplicate"],
    text: "NMS drops every detection overlapping a kept box above a fixed IoU threshold — so an object actually present in that overlap region is missed, which is how the corpus states the NMS problem. Dense one-stage decoders still rely on it as their only duplicate filter, later variants re-score the suppression itself, and the set-prediction lineage removes it entirely (NMS-free one-to-one decoding).",
    solutionIds: ["post-processing-nms"],
    problemIds: ["nms"],
  },
  {
    keys: ["iou", "giou", "diou", "ciou", "complete-iou", "generalized-iou", "distance-iou", "box regression loss"],
    text: "The IoU family replaces independent per-coordinate L2 box regression with a loss computed from overlap: the corpus's IoU loss regresses all four bounds jointly as L = −ln(IoU), against a regression-limits problem that notes the four box bounds are correlated yet are optimized separately under L2. Later solutions in the same line (including CIoU-flavoured recipes) keep refining the overlap-based objective.",
    solutionIds: ["iou-loss", "ciou"],
    problemIds: ["regression-limits"],
  },
  {
    keys: ["denois", "dn-detr", "convergence", "accelerat", "faster training", "training speed"],
    text: "Denoising training feeds noised ground-truth box/label pairs as extra decoder queries and reconstructs them with a matching-free reconstruction loss added on top of the standard Hungarian loss. That hands the one-to-one matcher cheap positives before it learns anything, which is the corpus's lever for DETR-style convergence.",
    solutionIds: ["denoising-training"],
  },
  {
    keys: ["deformable", "sparse attention", "area attention", "sampling locations"],
    text: "Deformable attention attends only to a small fixed set of learned sampling locations around a reference point instead of all pixels, aggregated across the four feature levels inside a DETR encoder-decoder — long-range, multi-scale context at a fraction of dense attention's cost.",
    solutionIds: ["deformable-attention"],
  },
  {
    keys: ["cascade", "refinement", "refine", "staged"],
    text: "The corpus's cascade solution chains a binary anchor-refinement stage and a multi-class detection stage inside one single-stage network, linked by transfer connection blocks: two-step cascade regression plus negative-anchor filtering, delivered at single-stage speed rather than as a separate heavy second network.",
    solutionIds: ["cascade"],
  },
  {
    keys: ["mosaic", "mixup", "augment", "data aug", "data-aug", "bag of freebies", "free training tricks"],
    text: "Augmentation sits inside a bag-of-freebies training recipe: the mosaic-augmentation solution combines a CSP backbone, a PAN path-aggregation neck and a YOLOv3 head with empirically validated free training tricks (image-combining mosaic among them) and low-cost inference plugins, aimed at maximizing COCO AP at real-time FPS on a conventional GPU.",
    solutionIds: ["mosaic-augmentation"],
  },
  {
    keys: ["open vocabulary", "open-vocabulary", "zero-shot", "vision-language", "clip", "grounding", "phrases"],
    text: "The open-vocabulary problem is that most detection methods are constrained to a small set of objects, because detection datasets are limited compared to classification datasets. The corpus's answer grounds every region against phrases in a text prompt with a unified loss, fuses vision and language deeply across encoder layers, and self-trains on detection, gold grounding and pseudo-labeled image-text data.",
    solutionIds: ["open-vocabulary-clip"],
    problemIds: ["open-vocabulary"],
  },
  {
    keys: ["hungarian", "bipartite", "set prediction", "set-prediction", "one-to-one", "set loss"],
    text: "DETR-style detectors pose detection as set prediction: a fixed-size set of boxes and classes is produced in one pass by a transformer decoder over learned object queries, supervised by a Hungarian-matching set loss. Replacing per-anchor supervision with one-to-one matching is what makes end-to-end, anchor-free, NMS-free decoding possible.",
    solutionIds: ["set-prediction"],
  },
  {
    keys: ["transformer", "detr", "queries", "query", "attention", "encoder-decoder"],
    text: "Transformers replace hand-designed localization with learned object queries: a decoder over image features predicts a fixed-size set of boxes and classes in one pass under a Hungarian set loss, removing anchors and NMS entirely. Attention itself is extended with geometric weights over relative box coordinates so recognition is conditioned on where a region sits, not only what it contains.",
    solutionIds: ["transformer-query", "attention"],
  },
  {
    keys: ["anchor-free", "anchor free", "anchor", "anchors", "center-based", "center point"],
    text: "Anchor-based heads tile preset boxes at every location and regress offsets to them; the corpus's anchor-design problem is that predefined anchor strategies fix anchor choices and anchor functions in both training and inference. Anchor-free heads instead emit dense per-location outputs (score plus box distances) or detect objects as center points on a keypoint heatmap, with peak extraction replacing NMS.",
    solutionIds: ["anchors", "anchor-free", "center-based"],
    problemIds: ["anchor-design"],
  },
  {
    keys: ["small object", "small-object", "tiny object", "feature pyramid", "pyramid", "multi-scale", "scales"],
    text: "Small objects are low-resolution and context-poor, and pooling descriptors only from the last convolutional layer discards the high-resolution features they need. The corpus answers with pyramid-shaped prediction: 3×3 predictors applied at several progressively smaller feature maps so detections at multiple scales come out of a single forward pass.",
    solutionIds: ["feature-pyramid", "multi-scale"],
    problemIds: ["small-objects"],
  },
  {
    keys: ["real-time", "realtime", "speed", "latency", "fps", "throughput"],
    text: "Speed is a first-class corpus problem: external region-proposal methods (Selective Search, EdgeBoxes) dominate the runtime of otherwise fast region-based detectors. The real-time-backbone answer is a thin-but-deep lightweight backbone whose multi-scale features are concatenated into a lightweight proposal/classifier pipeline — the accuracy–speed curve re-centered, not moved by one fixed trick.",
    solutionIds: ["real-time-backbone"],
    problemIds: ["speed-latency"],
  },
  {
    keys: ["label assignment", "assignment", "assigned", "simota", "positive samples", "positive/negative"],
    text: "Label assignment decides which locations supervise which head, and the corpus's problem statement is that a fixed IoU threshold is needed to define positives and negatives, with 0.5 a loose requirement that admits close false positives. Dynamic assignment answers by marking anchors outside a per-scale validity range as invalid (zeroing their gradients during training) and by assigning positives globally — SimOTA-style — with decoupled classification and regression heads.",
    solutionIds: ["dynamic-assignment", "label-assignment-simota"],
    problemIds: ["label-assignment"],
  },
];

function rulePapers(r: Rule): string[] {
  const ids = new Set<string>();
  for (const sid of r.solutionIds ?? []) {
    const s = SOLUTION_BY_ID[sid];
    if (s) for (const pid of s.paperIds) ids.add(pid);
  }
  for (const pid of r.problemIds ?? []) {
    const problem = PROBLEM_BY_ID[pid];
    if (problem) for (const o of problem.occurrences) for (const p of o.papers) ids.add(p);
  }
  return [...ids]
    .filter((id) => PAPER_BY_ID[id])
    .sort((a, b) => PAPER_BY_ID[b].year - PAPER_BY_ID[a].year || a.localeCompare(b))
    .slice(0, 6);
}

function ruleFor(q: string): Answer | null {
  const s = q.toLowerCase();
  for (const r of RULES) {
    if (r.keys.some((k) => s.includes(k))) {
      return { text: r.text, papers: rulePapers(r), source: "rule" };
    }
  }
  return null;
}

function searchFor(q: string): Answer {
  const terms = q.toLowerCase().split(/\s+/).filter((t) => t.length > 2);
  if (terms.length === 0) return { text: "", papers: [], source: "empty" };
  const hits = PAPERS.map((p) => {
    const hay = `${p.title} ${p.summary} ${p.contributions.join(" ")}`.toLowerCase();
    const score = terms.reduce((s, t) => s + (hay.includes(t) ? 1 : 0), 0);
    return { p, score };
  })
    .filter((h) => h.score > 0)
    .sort((a, b) => b.score - a.score || b.p.year - a.p.year);
  if (hits.length === 0) return { text: "", papers: [], source: "empty" };
  return {
    text: `No curated rule matched, so I searched the papers themselves. ${terms.join(" / ")} appears in ${hits.length} paper summary(ies)/contribution(s). Closest matches from the 121-paper corpus:`,
    papers: hits.slice(0, 6).map((h) => h.p.id),
    source: "search",
  };
}

const EXAMPLES = [
  "How does focal loss fix class imbalance?",
  "How does the IoU family of losses work?",
  "Why is NMS-free detection possible?",
  "How do transformers replace anchors?",
  "How are small objects handled?",
  "What accelerates DETR training?",
];

export function AskView() {
  const [q, setQ] = useState("");
  const [asked, setAsked] = useState<string | null>(null);

  const answer = useMemo<Answer | null>(() => {
    if (!asked) return null;
    return ruleFor(asked) ?? searchFor(asked);
  }, [asked]);

  const submit = (text: string) => {
    const t = text.trim();
    if (t) setAsked(t);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">Ask the corpus</h1>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          Ask a question in plain English. Answers come only from the 121-paper corpus —
          either a curated rule, whose papers are derived at runtime from the cited
          solutions and problems, or a full-text search over the papers&apos; own summaries
          and contributions. If nothing is supported, we say so.
        </p>
      </div>

      <Card>
        <form
          className="flex flex-wrap gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            submit(q);
          }}
        >
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="e.g. How does focal loss fix class imbalance?"
            className="min-w-0 flex-1 rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-[13px] text-zinc-200 placeholder:text-zinc-600 focus:border-emerald-600 focus:outline-none"
          />
          <button
            type="submit"
            className="rounded-md border border-emerald-700/60 bg-emerald-950/40 px-3 py-2 text-[12px] text-emerald-200 hover:bg-emerald-900/40"
          >
            ask
          </button>
        </form>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => {
                setQ(ex);
                submit(ex);
              }}
              className="rounded-md border border-zinc-800 px-2 py-1 text-[11px] text-zinc-400 hover:border-zinc-600 hover:text-zinc-200"
            >
              {ex}
            </button>
          ))}
        </div>
      </Card>

      {answer && (
        <Card tone={answer.source === "empty" ? "warn" : "default"}>
          {answer.source === "empty" ? (
            <InsufficientEvidence what={asked ?? ""} />
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={answer.source === "rule" ? "emerald" : "sky"}>
                  {answer.source === "rule" ? "curated rule · corpus-derived papers" : "full-text search"}
                </Badge>
                <span className="font-mono text-[10px] text-zinc-500">
                  {answer.papers.length} paper{answer.papers.length === 1 ? "" : "s"} cited
                </span>
              </div>
              <p className="mt-3 text-[13px] leading-6 text-zinc-300">{answer.text}</p>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {answer.papers.map((p) => (
                  <PaperLink key={p} id={p} />
                ))}
              </div>
            </>
          )}
        </Card>
      )}
    </div>
  );
}
