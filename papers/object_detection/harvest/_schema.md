# Object Detection Harvest Schema (v1)

You are harvesting ONE paper into ONE JSON file:
`papers/object_detection/harvest/<ID>.json` (e.g. `OD001.json`).

The full text is at the `textFile` path given in the batch brief
(e.g. `.cache/od_text/1512.02325v5.txt`). Read the ENTIRE text before
writing (use Read with offset/limit until the end; ~50k chars each).

## Grounding rules (NON-NEGOTIABLE)

1. Every `quote` must be a **verbatim substring** of the paper text.
   - Copy/paste exactly: same words, punctuation, numbers, casing.
   - Max 300 chars. Whitespace/newlines may differ (checker normalizes).
   - Never fix typos inside a quote. Never paraphrase inside `quotes`.
2. `location` = where it came from: "Abstract", "Introduction", "Sec 3.2",
   "Table 2", "Fig 4", "Conclusion", etc.
3. If a section genuinely has no support in the paper, use an empty
   array / `"INSUFFICIENT_EVIDENCE"` — do NOT invent content.
4. Everything else (`summary`, `statements`, `items`, ...) is a
   paraphrase — accurate, conservative, no hype.
5. Tag slugs: prefer the seed vocabulary below. If none fits, create a
   new slug: lowercase-kebab-case, ≤32 chars (`crowded-scenes`).
6. Do not copy large text blocks outside `quotes`.

## Seed problem tags (prefer these)

`small-objects` `tiny-objects` `localization` (box precision) `class-imbalance`
`long-tail` `speed-latency` `nms` (post-processing cost/errors) `anchor-design`
`label-assignment` `convergence` `occlusion` `open-vocabulary` `domain-shift`
`deployability` `crowded-scenes` `class-confusion` `one-stage-accuracy`
`two-stage-speed` `annotation-cost` `weak-supervision` `domain-adaptation`
`background-false-positives` `training-instability` `scale-variance`
`real-time-edge` `forgetting-catastrophic` `benchmark-inconsistency`

## Seed solution tags (prefer these)

`one-stage` `two-stage` `transformer-query` `anchor-free` `anchors`
`feature-pyramid` `feature-fusion` `attention` `self-attention`
`deformable-attention` `nms-free` `set-prediction` `label-assignment-simota`
`focal-loss` `iou-loss` `ciou` `distribution-fl` `focal-loss-quality`
`gaussian-heatmap` `keypoint-based` `multi-scale` `dynamic-assignment`
`distillation` `reparameterization` `nas` `open-vocabulary-clip`
`post-processing-nms` `sparse-sampling` `cascade` `real-time-backbone`
`quantization-pruning` `mosaic-augmentation` `self-training`
`contrastive-learning` `dense-prediction` `sliding-window-cnns`
`center-based` `edge-detection-based`

## Exact JSON shape

```json
{
  "id": "OD001",
  "arxiv": "1512.02325",
  "title": "<exact title from batch brief>",
  "year": 2015,
  "venue": "venue if stated in the PDF (e.g. ECCV 2016), else null",
  "problem": {
    "summary": "1-3 sentence paraphrase of what the paper says is unsolved/wrong.",
    "statements": ["specific, self-contained problem statement", "..."],
    "problemTags": ["small-objects", "speed-latency"],
    "evidence": [
      { "quote": "verbatim quote", "location": "Abstract" }
    ]
  },
  "background": {
    "summary": "What prior work the paper builds on and the gap it claims.",
    "relatedWork": [
      { "name": "Fast R-CNN", "relation": "baseline / prior stage", "quote": "verbatim", "location": "Introduction" }
    ],
    "evidence": [ { "quote": "verbatim", "location": "Introduction" } ]
  },
  "solution": {
    "summary": "1-3 sentence paraphrase of the proposed method.",
    "approach": "the core mechanism in one sentence",
    "components": ["multi-scale feature maps", "..."],
    "solutionTags": ["one-stage", "anchors", "feature-pyramid"],
    "evidence": [ { "quote": "verbatim", "location": "Sec 3" } ]
  },
  "experiments": {
    "summary": "1-3 sentences on how the paper evaluates itself.",
    "benchmarks": [
      { "dataset": "PASCAL VOC", "metric": "mAP", "value": "76.8", "split": "test 2012", "location": "Table 3" }
    ],
    "baselines": ["Faster R-CNN", "YOLO"],
    "evidence": [ { "quote": "verbatim of a headline result", "location": "Abstract" } ]
  },
  "limitations": {
    "summary": "1-3 sentences: what the authors admit or what clearly remains.",
    "items": ["admitted or evident limitation", "..."],
    "evidence": [ { "quote": "verbatim", "location": "Conclusion" } ]
  },
  "futureDirections": [
    { "direction": "paraphrase of a stated future direction", "quote": "verbatim future-work quote", "location": "Conclusion" }
  ],
  "technical": {
    "datasets": ["COCO", "PASCAL VOC"],
    "backbones": ["VGG-16"],
    "models": ["named detector / architecture variants"],
    "algorithms": ["NMS", "k-means anchor clustering"],
    "metrics": ["mAP", "AP50", "FPS"],
    "equations": [
      { "name": "IoU", "expr": "|A ∩ B| / |A ∪ B|", "location": "Sec 2" }
    ]
  },
  "industry": {
    "applications": [
      { "domain": "autonomous driving", "detail": "paraphrase", "quote": "verbatim (optional)", "location": "Introduction" }
    ],
    "deploymentClaims": ["e.g. 45 FPS on a Titan X, stated by authors"],
    "evidence": [ { "quote": "verbatim", "location": "Abstract" } ]
  },
  "citations": {
    "doi": "<from batch brief or paper, else null>",
    "arxivUrl": "https://arxiv.org/abs/<id>" or null,
    "venue": "published venue if known, else null",
    "pdfUrl": "<from batch brief>"
  }
}
```

Field notes:
- `futureDirections`: `[]` if the paper states none.
- `industry.applications` / `deploymentClaims` / `evidence`: `[]` if no
  industry/application evidence exists — this is a real signal the
  analysis phase uses; never pad it.
- `experiments.benchmarks`: include the paper's OWN reported numbers
  (headline table rows), not every cell. Aim 3-10 entries.
- `problem.problemTags` / `solution.solutionTags`: 1-6 tags each.
- `technical.equations`: only if the paper's method rests on equations;
  0-6 entries, `expr` is a compact text rendering.
- Keep JSON < 12KB. Prefer 2-4 evidence quotes per section over many.
