"use client";

import { useMemo, useState } from "react";
import { PAPERS, PAPER_BY_ID } from "@/labs/object-detection/data/papers";
import { Card, PaperLink, Badge, EvidenceLine } from "@/labs/object-detection/ui";
import type { Evidence } from "@/labs/object-detection/data/types";

type Answer = {
  text: string;
  papers: string[];
  source: "rule" | "search" | "empty";
};

type Rule = {
  keys: string[];
  text: string;
  papers: string[];
};

/* Rules ≡ grounded claims already present across the simulators/views; they only
 * cite the same corpus IDs those pages cite. The fallback is a full-text search
 * over actual paper summaries + contributions — never fabricated. */
const RULES: Rule[] = [
  {
    keys: ["focal loss", "focal", "class imbalance", "class-imbalance"],
    text: "Focal loss down-weights easy, correctly-classified background so the few hard examples dominate the gradient; the corpus uses it against foreground/background imbalance in one-stage dense detectors.",
    papers: ["P015", "P017", "P024", "P021"],
  },
  {
    keys: ["nms", "non-maximum", "suppression"],
    text: "NMS greedily removes duplicate detections above an IoU threshold; DIoU-NMS and Matrix NMS improve it, and the transformer lineage removes it entirely (NMS-free one-to-one set prediction).",
    papers: ["P001", "P016", "P017", "P022", "P029", "P033", "P046"],
  },
  {
    keys: ["iou", "giou", "diou", "ciou", "complete-iou", "generalized-iou", "distance-iou"],
    text: "The IoU family as regression losses: GIoU adds a hull penalty when boxes do not overlap, DIoU penalizes center distance, CIoU adds aspect-ratio consistency — the corpus’s box regression workhorse.",
    papers: ["P001", "P016", "P017", "P024", "P038"],
  },
  {
    keys: ["hungarian", "matching", "assignment", "bipartite", "set loss", "set-loss"],
    text: "DETR-style detectors replace per-anchor supervision with Hungarian one-to-one matching between queries and ground truth, which is what makes end-to-end NMS-free decoding possible.",
    papers: ["P029", "P037", "P033", "P046", "P038"],
  },
  {
    keys: ["anchor", "anchors", "anchor-free", "anchor free"],
    text: "Anchor-based detection tiles preset boxes and assigns by IoU; anchor-free heads predict from a center cell and scale level, decoupling location from scale — both strands are documented in the corpus.",
    papers: ["P001", "P003", "P017", "P024", "P027", "P038"],
  },
  {
    keys: ["small object", "small-object", "tiny", "scales"],
    text: "Small objects occupy few feature cells after stride; the field responded with multi-scale feature maps, feature pyramids and dense supervision across levels.",
    papers: ["P001", "P004", "P005", "P008", "P027", "P048"],
  },
  {
    keys: ["real-time", "realtime", "speed", "latency", "fps"],
    text: "The real-time thread trades localization cost for throughput, and the corpus repeatedly re-centers the accuracy-speed curve — including transformer detectors at YOLO-class latency.",
    papers: ["P001", "P016", "P042", "P047", "P048"],
  },
  {
    keys: ["transformer", "detr", "query", "queries", "attention"],
    text: "Transformers replaced hand-designed localization with learned query-to-feature attention, enabling NMS-free set prediction; deformable and area attention recover the quadratic cost.",
    papers: ["P029", "P037", "P041", "P038"],
  },
  {
    keys: ["denois", "dn-detr", "convergence"],
    text: "Denoising queries (corrupted ground-truth boxes fed as auxiliary queries) give the one-to-one matcher early, cheap positives and accelerate DETR-style training convergence.",
    papers: ["P038", "P037"],
  },
  {
    keys: ["cascade", "refinement", "refine", "stage"],
    text: "Cascade refinement trains successive box heads at rising IoU thresholds so each stage operates on tighter positives — the two-stage answer to hard localization.",
    papers: ["P016", "P017"],
  },
  {
    keys: ["deformable", "sparse attention", "area-attention", "area attention"],
    text: "Deformable/area attention samples a small learned set of positions per query instead of attending densely, cutting cost while keeping long-range context for multi-scale fusion.",
    papers: ["P029", "P037", "P038", "P041"],
  },
  {
    keys: ["mosaic", "mixup", "augment", "data aug", "data-aug"],
    text: "Mosaic augments by stitching several images, multiplying instance density and small-object coverage; the corpus documents bag-of-freebies training recipes around such augmentation.",
    papers: ["P016", "P042", "P047"],
  },
  {
    keys: ["distribution", "dfl", "dfine", "soft label", "soft-label"],
    text: "Distribution-based box regression predicts a probability vector over discretized offsets and decodes the continuous value — D-FINE-style refinement of the regressed box.",
    papers: ["P037", "P038"],
  },
  {
    keys: ["open vocabulary", "open-vocabulary", "zero-shot", "vlm", "grounding"],
    text: "Open-vocabulary detection extends the class set beyond training by aligning visual features with text embeddings from vision-language models.",
    papers: ["P031", "P032", "P033"],
  },
  {
    keys: ["eval", "ap", "map", "metric", "coco"],
    text: "The corpus evaluates with COCO-style average precision and reports both accuracy and latency; our Evidence policy shows numeric figures only where a paper states them.",
    papers: ["P001", "P015", "P016"],
  },
];

function ruleFor(q: string): Answer | null {
  const s = q.toLowerCase();
  for (const r of RULES) {
    if (r.keys.some((k) => s.includes(k))) {
      return { text: r.text, papers: r.papers, source: "rule" };
    }
  }
  return null;
}

function searchFor(q: string): Answer {
  const terms = q.toLowerCase().split(/\s+/).filter((t) => t.length > 2);
  if (terms.length === 0) return { text: "", papers: [], source: "empty" };
  const hits = PAPERS
    .map((p) => {
      const hay = `${p.title} ${p.summary} ${p.contributions.join(" ")}`.toLowerCase();
      const score = terms.reduce((s, t) => s + (hay.includes(t) ? 1 : 0), 0);
      return { p, score };
    })
    .filter((h) => h.score > 0)
    .sort((a, b) => b.score - a.score);
  if (hits.length === 0) return { text: "", papers: [], source: "empty" };
  const top = hits.slice(0, 6);
  return {
    text: `No rule matched, so I searched the papers themselves. ${terms.join(" / ")} appears in ${hits.length} paper summary(ies). These were the closest matches:`,
    papers: top.map((h) => h.p.id),
    source: "search",
  };
}

const EXAMPLES = [
  "What causes the foreground/background imbalance?",
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
          Ask a question in plain English. Answers come only from the 48-paper corpus —
          either a curated, cited rule or a full-text search over the papers’ own
          summaries and contributions. If nothing is supported, we say so.
        </p>
      </div>

      <Card>
        <form
          className="flex flex-wrap gap-2"
          onSubmit={(e) => { e.preventDefault(); submit(q); }}
        >
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="e.g. How does focal loss fix class imbalance?"
            className="min-w-0 flex-1 rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-[13px] text-zinc-200 placeholder:text-zinc-600"
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
              onClick={() => { setQ(ex); submit(ex); }}
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
            <>
              <Badge tone="rose">insufficient evidence</Badge>
              <p className="mt-2 text-[12px] leading-5 text-zinc-400">
                No corpus rule or paper summary supports “{asked}”. The answer is
                withheld rather than guessed.
              </p>
            </>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={answer.source === "rule" ? "emerald" : "sky"}>
                  {answer.source === "rule" ? "curated from corpus" : "full-text search"}
                </Badge>
                <div className="flex flex-wrap gap-1.5">
                  {answer.papers.map((p) => {
                    const paper = PAPER_BY_ID[p];
                    if (!paper) return null;
                    const ev: Evidence = {
                      paperIds: [p],
                      kind: answer.source === "rule" ? "paperSays" : "crossPaper",
                      confidence: "HIGH",
                    };
                    return <EvidenceLine key={p} evidence={ev} compact />;
                  })}
                </div>
              </div>
              <p className="mt-3 text-[13px] leading-6 text-zinc-300">{answer.text}</p>
              {answer.source === "search" && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {answer.papers.map((p) => (
                    <PaperLink key={p} id={p} />
                  ))}
                </div>
              )}
            </>
          )}
        </Card>
      )}
    </div>
  );
}