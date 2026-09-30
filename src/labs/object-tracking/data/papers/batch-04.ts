import type { PaperRecord } from "../types";

/* batch-04 — T026..T037 */

export const BATCH_04: PaperRecord[] = [
  {
    id: "T026",
    arxiv: "1811.07628",
    title: "ATOM: Accurate Tracking by Overlap Maximization",
    shortTitle: "ATOM",
    year: 2018,
    authors: ["Martin Danelljan", "Goutam Bhat", "Fahad Shahbaz Khan", "Michael Felsberg"],
    fileName: "1811.07628v2.pdf",
    task: "single-object",
    tags: ["iou-prediction", "overlap-maximization", "siamese", "gauss-newton", "online-learning", "real-time"],
    difficulty: "intermediate",
    summary:
      "ATOM splits tracking into two dedicated modules: an offline-trained IoU-prediction network that regresses the overlap between any candidate box and the target (conditioned on the first frame), and a two-layer classification head trained online with an unrolled Gauss-Newton/Conjugate-Gradient optimizer. By maximizing predicted overlap instead of searching scales, it sets a new state of the art on NFS, UAV123, TrackingNet, LaSOT and VOT2018 while running at over 30 FPS.",
    problem:
      "Most trackers before ATOM did not estimate the target state at all: a sliding-window classifier did double duty, and the box was found by brute-force search over a few discrete scales at a fixed aspect ratio, which cannot follow pose or aspect-ratio changes. Learning accurate box estimation online from a single starting frame was considered infeasible, so bounding-box accuracy rather than classification was the bottleneck — exactly what the VOT2018 challenge results exposed.",
    background: ["sot", "bounding-box", "iou", "deep-detection", "correlation-filter", "siamese", "success-plot"],
    previousWork: [
      {
        name: "Correlation-filter trackers without state estimation (UPDT, ECO, SRDCF)",
        limitation:
          "They rely on the classifier response for target estimation via a brute-force multi-scale search (5 scales, ratio 1.02) and do not handle aspect-ratio changes at all — Figure 1 shows UPDT failing exactly on that.",
        whyThisPaper:
          "A dedicated overlap-prediction network regresses box quality directly and beats the multi-scale strategy by 8.6% AUC while nearly doubling OP0.75 (26.0 → 48.4).",
      },
      {
        name: "IoU-Net (Jiang et al.)",
        limitation:
          "Predicts IoU overlap but is class-specific, so it cannot be used for generic tracking where the object category is unknown.",
        whyThisPaper:
          "ATOM re-architects IoU prediction around a modulation vector computed from the first frame, making the estimator target-specific and category-agnostic, trainable offline on large detection datasets.",
      },
      {
        name: "Siamese box regression (SiamRPN, DaSiamRPN)",
        limitation:
          "Box regression comes from extensive offline training on anchor priors, yet these trackers struggle at target classification and still fail on out-of-plane rotation and deformation (Figure 1).",
        whyThisPaper:
          "ATOM pairs its overlap predictor with an online-trained classifier, so both tasks are handled by the component designed for each.",
      },
    ],
    researchGap:
      "Before this paper, no generic tracker had an offline-trained component that predicts target-specific bounding-box overlap, so box estimation still relied on brute-force multi-scale search.",
    contribution: [
      "Overlap-prediction network (IoU-Net made class-agnostic): predicts IoU(B) = g(c(x₀, B₀) ⊙ z(x, B)) between the target and a candidate box, with target appearance injected through a modulation vector c computed once from the first frame.",
      "Classification head: a two-layer fully convolutional network with PELU nonlinearities, trained exclusively online, hardened by hard-negative mining (a distractor peak below score 0.25 doubles that sample's learning rate and triggers an immediate optimization round).",
      "Efficient online optimization: unrolled Gauss-Newton iterations whose quadratic subproblem is solved by Conjugate Gradient using two BackProp calls per CG iteration — virtually parameter free and clearly better than tuned gradient descent.",
      "Tracking loop: classify to get an initial box, jitter 10 random proposals around it, run 5 gradient-ascent steps per proposal on predicted IoU, keep the mean of the 3 highest-IoU boxes, then update the classifier.",
      "State of the art on five benchmarks (NFS, UAV123, TrackingNet, LaSOT, VOT2018) at over 30 FPS, with a +15% relative gain on TrackingNet over the previous best.",
    ],
    method: {
      pipeline: ["extract", "classify", "predict-iou", "refine-box", "mine-negatives", "update"],
      architecture:
        "Shared ImageNet-pretrained ResNet-18 processing one 288×288 patch per frame (a region five times the estimated target size). Target estimation feeds Block3+Block4 features through a PrPool-based modulation network (modulation vector × pooled features) and a 3-layer MLP that emits one IoU score; classification takes Block4 features through a 1×1 conv to 64 channels followed by a 2-class fully convolutional head.",
      motionModel:
        "None: the search patch is centred on the previous state and the box comes from maximizing predicted IoU around the classifier's peak, not from a motion prior.",
      appearanceModel:
        "Two appearance models: (i) an offline, target-conditioned modulation vector c(x₀, B₀) precomputed in the first frame and kept fixed for IoU prediction; (ii) an online classification filter with per-sample weights γⱼ (learning rate 0.01), reinforced by hard negatives.",
      detectionDependency: "None — a class-agnostic single-object tracker with no external detector.",
      trackManagement:
        "The target is declared lost when the classification score falls below 0.25; distractor peaks are treated as hard negatives and re-weighted by doubling their learning rate before re-optimizing.",
      loss:
        "Classification: L(w) = Σⱼ γⱼ‖f(xⱼ;w) − yⱼ‖² + Σₖ λₖ‖wₖ‖² with Gaussian pseudo-labels yⱼ centered at the estimated target location. The IoU branch is trained offline by regression against annotated overlaps.",
      optimization:
        "Online: N_GN Gauss-Newton outer iterations, each running N_CG Conjugate-Gradient iterations on the quadratic subproblem (a heavy schedule on the first frame, a lighter one every 10th frame later). Offline: ADAM, lr 1e-3 decayed ×0.2 at 15 epochs, 40 epochs, batch 64 image pairs, trained on LaSOT-train, TrackingNet-train and COCO.",
    },
    equations: [
      {
        id: "atom-iou",
        label: "Target-specific IoU prediction",
        formula: "IoU(B) = g( c(x₀, B₀) ⊙ z(x, B) )",
        variables: [
          { symbol: "c(x₀, B₀)", meaning: "modulation vector computed from the reference (first-frame) features inside box B₀ — fixed after the first frame" },
          { symbol: "z(x, B)", meaning: "pooled features of the current frame x inside the candidate box B" },
          { symbol: "⊙", meaning: "element-wise modulation of the candidate features by the target descriptor" },
          { symbol: "g", meaning: "small multi-layer perceptron mapping modulated features to a single overlap score" },
        ],
        intuition:
          "Ask a tiny network 'how much would this candidate box overlap the target as I remember it?' — the target's own first-frame features are mixed into the candidate features so one network works for any object.",
        why:
          "Overlap is the quantity tracking actually cares about; predicting it directly replaces the multi-scale search and lets the tracker maximize IoU instead of classification score.",
        where: "Section 3.1 (Target Estimation); evaluated every frame during box refinement.",
        params:
          "The modulation vector is precomputed once in the first frame, so the estimator never changes online — only the classifier adapts.",
        paperIds: ["T026"],
      },
      {
        id: "atom-classifier",
        label: "Classification head",
        formula: "f(x; w) = φ₂( w₂ ∗ φ₁( w₁ ∗ x ) )",
        variables: [
          { symbol: "w₁", meaning: "1×1 convolution reducing the Block4 feature dimensionality to 64 channels" },
          { symbol: "w₂", meaning: "second convolutional layer of the 2-class head (foreground/background)" },
          { symbol: "φ₁, φ₂", meaning: "PELU nonlinearities (parametric ELUs) used in place of ReLU" },
        ],
        intuition:
          "A deliberately shallow two-layer convolutional scorer on top of frozen backbone features — cheap enough to be re-optimized online every frame.",
        why:
          "Only the classifier must adapt to the sequence (distractors, appearance drift), and a tiny head makes the unrolled Gauss-Newton updates fast enough for real time.",
        where: "Section 3.2 (Target Classification); run on the Block4 features of every search patch.",
        params:
          "Hard negatives (distractor peaks below 0.25) are up-weighted by doubling their sample weight γⱼ.",
        paperIds: ["T026"],
      },
      {
        id: "atom-loss",
        label: "Regularized classification objective",
        formula: "L(w) = Σⱼ γⱼ ‖ f(xⱼ; w) − yⱼ ‖² + Σₖ λₖ ‖wₖ‖²",
        variables: [
          { symbol: "yⱼ", meaning: "Gaussian regression target centered on the estimated target location for sample j" },
          { symbol: "γⱼ", meaning: "per-sample weight — increased for hard negatives" },
          { symbol: "λₖ", meaning: "L2 weight-decay strengths per layer" },
        ],
        intuition:
          "Fit a smooth blob of confidence over the target while keeping the two-layer head small and stable.",
        why:
          "A regression-style L2 loss (rather than cross-entropy) gives the Gauss-Newton linearization the least-squares structure it needs for the unrolled optimizer.",
        where: "Section 3.2, Eq. 3; minimized online each time the classifier updates.",
        params:
          "Gaussian label width controls how peaked the confidence map is: too wide blurs localization, too narrow overfits single frames.",
        paperIds: ["T026"],
      },
      {
        id: "atom-gauss-newton",
        label: "Gauss-Newton quadratic model (online optimizer)",
        formula: "L̃_w(Δw) = Δwᵀ Jᵀ J Δw + 2 Δwᵀ Jᵀ r + rᵀ r",
        variables: [
          { symbol: "J", meaning: "Jacobian of the residuals at the current weights (computed with BackProp tricks)" },
          { symbol: "r", meaning: "residual vector f(x;w) − y at the current estimate" },
          { symbol: "Δw", meaning: "weight update solved inside each unrolled iteration" },
        ],
        intuition:
          "Replace the hard least-squares problem with a local quadratic bowl whose minimum can be found directly, then step there.",
        why:
          "Plain gradient descent needs hand-tuned learning rate and momentum and still converges slowly; the Gauss-Newton/Conjugate-Gradient loop is essentially parameter-free and reaches the same speed as the whole tracker budget.",
        where: "Section 3.2 and Algorithm 1; runs every time the classification model is updated.",
        params:
          "N_GN outer Gauss-Newton iterations each contain N_CG Conjugate-Gradient iterations; each CG iteration costs two BackProp calls.",
        paperIds: ["T026"],
      },
    ],
    datasets: ["others", "uav123", "trackingnet", "lasot", "vot"],
    metrics: ["success-auc", "precision", "norm-precision", "eao", "fps"],
    baselines: ["UPDT", "ECO", "CCOT", "MDNet", "DaSiamRPN", "SiamRPN"],
    results: [
      "NFS (30 FPS version): success AUC 59.0 — ahead of UPDT 54.2, CCOT 49.2, ECO 47.0 and MDNet 42.5.",
      "UAV123: success 65.0 vs DaSiamRPN 58.4, SiamRPN 57.1, UPDT 55.0, ECO 53.7 and SRDCF 47.3.",
      "TrackingNet test: success 70.3 and precision 64.8; the abstract claims a 15% relative gain over the previous best while running at over 30 FPS.",
      "LaSOT (Table 4): success 51.5 and normalized precision 57.6 vs DaSiamRPN 41.5 / 49.6 — an absolute gain of 10% cited in the introduction.",
      "VOT2018: EAO 0.401, accuracy 0.590, robustness 0.204 — +3% relative EAO over LADCF (0.389), best in the compared table.",
      "Target-estimation quality on combined NFS+UAV123: OP0.50 76.3%, OP0.75 48.4%, AUC 62.3%.",
    ],
    ablations: [
      "IoU-prediction architecture (Table 1, NFS+UAV123): no-reference baseline 56.7 AUC, feature concatenation 56.3, siamese scalar product 61.7, feature modulation 62.3; using only Block3 gives 60.3 and only Block4 58.5 — modulation over both blocks wins.",
      "Component ablation (Table 2): ATOM 62.3 AUC vs brute-force multi-scale 53.7 (+8.6), target estimation without the classification module 43.0, gradient descent instead of Gauss-Newton 60.9 (GD) or 61.1 with 5× more iterations (GD++), no hard-negative mining 61.9.",
      "The gain concentrates in precise boxes: OP0.75 rises from 26.0 (multi-scale) to 48.4 (ATOM), i.e. highly accurate predictions nearly double.",
    ],
    limitations: {
      authorStated: [
        "The hard-negative strategy is explicitly described as 'not fundamental to our framework' — it only adds some extra robustness.",
        "The IoU module cannot be learned online: the paper argues box estimation needs high-level pose knowledge that cannot be learned from a single starting frame, so that component stays frozen after offline training.",
      ],
      evident: [
        "Box refinement is a local search: 10 jittered proposals × 5 gradient-ascent steps around the classifier peak cannot recover once the classifier locks onto the wrong object.",
        "The modulation vector is computed once in the first frame, so long-term appearance change can only be absorbed by the classifier, never by the box estimator.",
      ],
    },
    assumptions: [
      "A ground-truth box for the first frame is given; one target per sequence.",
      "The target stays inside a 288×288 search patch (5× the estimated target size) around its previous position.",
    ],
    computation:
      "Over 30 FPS on an Nvidia GT-1080 in PyTorch; one shared ResNet-18 forward pass per frame on a single 288×288 patch; refinement runs 10 proposals × 5 gradient-ascent steps through the IoU head.",
    relations: [
      {
        to: "T011",
        type: "uses-as-baseline",
        note: "ECO is compared on NFS (47.0 vs ATOM 59.0), UAV123, TrackingNet and VOT2018.",
      },
      {
        to: "T022",
        type: "uses-as-baseline",
        note: "DaSiamRPN is the strongest compared box-regressing tracker (UAV123 58.4 vs ATOM 65.0); Figure 1 shows it failing on out-of-plane rotation where overlap prediction succeeds.",
      },
      {
        to: "T010",
        type: "uses-as-baseline",
        note: "SRDCF appears in the state-of-the-art tables, e.g. UAV123 success 47.3 vs ATOM 65.0.",
      },
      {
        to: "T003",
        type: "uses-as-baseline",
        note: "MDNet is compared on NFS (42.5 vs 59.0) and in the VOT2018 table.",
      },
      {
        to: "T024",
        type: "uses-as-baseline",
        note: "ATOM is evaluated on the LaSOT benchmark and posts its best absolute numbers there (success 51.5).",
      },
    ],
    concepts: ["sot", "bounding-box", "iou", "deep-detection", "appearance-features", "success-plot"],
    impact:
      "ATOM made 'classification + explicit overlap prediction' the template for the next generation of SOT trackers: DiMP (T031) keeps ATOM's box branch unchanged and replaces only the classification model, PrDiMP (T040) re-trains the same IoU head as a probabilistic density, and its class-agnostic use of IoU-Net became the standard way to do box regression in tracking.",
  },
  {
    id: "T027",
    arxiv: "1812.05050",
    title: "Fast Online Object Tracking and Segmentation: A Unifying Approach",
    shortTitle: "SiamMask",
    year: 2018,
    authors: ["Qiang Wang", "Li Zhang", "Luca Bertinetto", "Weiming Hu", "Philip H.S. Torr"],
    fileName: "1812.05050v2.pdf",
    task: "segmentation-tracking",
    tags: ["siamese", "segmentation-tracking", "mask-prediction", "multi-task-learning", "real-time"],
    difficulty: "intermediate",
    summary:
      "SiamMask trains one Siamese network with three branches — classification score, bounding-box regression and a 63×63 binary mask — so a single model tracks an object, regresses its box, and segments it pixel by pixel. Converting the predicted mask back into a box (Min-max, MBR or Opt) beats previous box trackers on VOT2016/2018 at 55 FPS, and the same model does first-frame-box video object segmentation on DAVIS and YouTube-VOS.",
    problem:
      "Tracking and video object segmentation were separate problems: trackers output boxes but no pixel-accurate shape, while VOS methods either fine-tune on the first-frame mask for every sequence (far too slow for online use) or need more supervision than a plain bounding box. No single fast model could start from a rectangle and produce both a robust box and a segmentation mask.",
    background: ["siamese", "sot", "iou", "detection", "deep-detection", "appearance-features"],
    previousWork: [
      {
        name: "SiamFC / SiamRPN-style matching trackers",
        limitation:
          "They produce only a box from a score map or anchor regression — no pixel-level representation of the target, so box quality is capped by the coarse similarity peak.",
        whyThisPaper:
          "Adding a mask branch gives a dense per-pixel target representation; the box can then be derived from the predicted mask, raising mAP@0.5 from 76.20 (SiamRPN) to 85.4.",
      },
      {
        name: "Video object segmentation with per-sequence fine-tuning (OSVOS-style)",
        limitation:
          "Fine-tuning on the first-frame mask at test time is slow and prevents online operation; the segmentation runs detached from the tracking model.",
        whyThisPaper:
          "SiamMask is trained offline once and runs at 55 FPS with no adaptation to the test sequence, covering segmentation and tracking with one set of weights.",
      },
    ],
    researchGap:
      "Before this paper, no online tracker learned classification, box regression and mask prediction jointly, so fast box tracking and pixel-accurate segmentation could not live in one model.",
    contribution: [
      "Unified three-branch Siamese network: score branch (2k), box branch (4k) and mask branch (k² × 63²) on shared ResNet-50 features with depth-wise cross-correlation, trained end-to-end offline.",
      "Mask branch supervised by a binary logistic loss over positive RoIs only (anchor IoU ≥ 0.6), weighted with λ₁ = 32 alongside the RoI classification score (λ₂ = λ₃ = 1) — the mask-detector multi-task recipe applied to tracking.",
      "Box-from-mask strategies (Min-max, MBR/optimal rotated rectangle, Opt): MBR is the accuracy/ speed compromise, Opt the quality ceiling.",
      "One model covers three tasks: box tracking, box-from-mask tracking, and semi-supervised VOS from a first-frame bounding box.",
      "State of the art on VOT2016/2018 at 55 FPS while simultaneously producing masks competitive with dedicated VOS methods.",
    ],
    method: {
      pipeline: ["extract", "correlate", "score", "regress-box", "predict-mask", "convert-mask"],
      architecture:
        "ResNet-50 backbone (dilated, effective stride 8) shared by a 127×127 exemplar and 255×255 search branch; depth-wise cross-correlation produces k response maps consumed by three heads: score (2k), box (4k offsets) and mask (k² × 63² coefficients decoded to a 127×127 mask via RoI-align inside each proposal).",
      motionModel:
        "None: the search patch is centred on the previous box; scale comes from RPN-style box refinement over 5 anchors per location.",
      appearanceModel:
        "First-frame template features only — no online update; robustness comes entirely from offline training on large video detection data.",
      detectionDependency:
        "None external, but the RPN proposal mechanism inside the search region is what the mask branch consumes (mask loss applies only to positive proposals).",
      loss:
        "Multi-task: L2B = λ₁·Lmask + λ₂·Lsim (two-branch) or L3B = λ₁·Lmask + λ₂·Lscore + λ₃·Lbox (three-branch), with λ₁ = 32, λ₂ = λ₃ = 1; Lmask is a binary logistic regression over all pixels of positive RoIs.",
      optimization:
        "Offline training only; inference-time box-from-mask conversion is non-parametric (Min-max, MBR or Opt), so no per-sequence adaptation of any kind.",
    },
    equations: [
      {
        id: "siammask-maskloss",
        label: "Mask branch loss (per-RoW logistic regression)",
        formula:
          "Lmask = Σₙ [ (1 + yₙ) / (2 · w · h) ] · Σ_{i,j} log( 1 + e^{ −c_ij · m_ij } )",
        variables: [
          { symbol: "yₙ", meaning: "binary label of the n-th candidate RoI ({+1, −1}); only positive RoIs contribute" },
          { symbol: "c_ij", meaning: "predicted mask coefficient at pixel (i, j) for that RoI" },
          { symbol: "m_ij", meaning: "ground-truth binary mask value at pixel (i, j)" },
          { symbol: "w, h", meaning: "mask resolution (the branch predicts a 63×63 coefficient map)" },
        ],
        intuition:
          "For every proposal that really contains the target, push each pixel's score toward the true mask; ignore proposals without the object.",
        why:
          "Mask quality feeds the box estimate (box-from-mask) and the VOS task alike, and restricting supervision to positive RoIs keeps the extremely unbalanced foreground/background ratio manageable.",
        where: "Section 3, Eq. 3; applies to positive anchors (IoU ≥ 0.6 with ground truth).",
        params: "λ₁ = 32 follows the mask-detector convention cited by the paper; λ₂ = λ₃ = 1.",
        paperIds: ["T027"],
      },
      {
        id: "siammask-multitask",
        label: "Multi-task training objectives",
        formula:
          "L2B = λ₁ · Lmask + λ₂ · Lsim\nL3B = λ₁ · Lmask + λ₂ · Lscore + λ₃ · Lbox",
        variables: [
          { symbol: "Lsim", meaning: "logistic loss on the score branch (same as SiamFC/SiamRPN)" },
          { symbol: "Lscore, Lbox", meaning: "RoI classification and bounding-box regression losses from the mask-detector recipe" },
          { symbol: "λ₁, λ₂, λ₃", meaning: "task weights, set to 32, 1, 1" },
        ],
        intuition:
          "Train box quality, objectness and segmentation together so the shared backbone must serve all three at once.",
        why:
          "The paper shows both variants beat their plain counterparts (SiamFC 0.251 → 0.265 and SiamRPN 0.359 → 0.363 EAO on VOT2018), isolating the value of the extra branches.",
        where: "Section 3, Eqs. 4–5; L2B for the two-branch variant, L3B for the full three-branch model.",
        params: "λ₁ = 32, λ₂ = λ₃ = 1, set exactly as in the cited mask-detection work.",
        paperIds: ["T027"],
      },
    ],
    datasets: ["vot", "others"],
    metrics: ["eao", "success-auc", "precision", "fps"],
    baselines: ["SiamFC", "SiamRPN", "DaSiamRPN", "ECO", "OnAVOS"],
    results: [
      "Box-from-mask (Table 1): mIOU Min-max 65.05, MBR 67.15, Opt 71.68 vs SiamFC 50.48 and SiamRPN 60.02 (fixed aspect-ratio oracles 73.43 / 77.70 / 84.07); mAP@0.5 for MBR reaches 85.4 vs SiamRPN 76.20 and SiamFC 56.42.",
      "VOT2018 (Table 2): SiamMask EAO 0.380, accuracy 0.609, robustness 0.276 at 55 FPS — vs DaSiamRPN EAO 0.326; SiamMask-box 0.363 and SiamMask-Opt 0.387 (≈5 FPS).",
      "VOT2016 (Table 3): SiamMask EAO 0.433, accuracy 0.639, robustness 0.214; box-only variant 0.428, Opt variant 0.465.",
      "DAVIS2016 validation: J 71.7, F 86.8 at 55 FPS vs OnAVOS J 86.1 / F 84.9 at 0.08 FPS; DAVIS2017: J 54.3, F 58.5.",
      "YouTube-VOS: overall 52.8 (J-seen 60.2, J-unseen 45.1) at 55 FPS — best speed among the compared VOS methods.",
      "Branch contributions on VOT2018: two-branch model lifts SiamFC from 0.251 to 0.265 EAO, three-branch model lifts SiamRPN from 0.359 to 0.363, with the full model reaching 0.380.",
    ],
    ablations: [
      "Box-from-mask conversion strategy (Table 1): Min-max 65.05 → MBR 67.15 → Opt 71.68 mIOU; MBR is chosen for the accuracy/speed trade-off since Opt is computationally expensive.",
      "Backbone and mask refinement (Table 7): swapping AlexNet for ResNet-50 and adding the mask-refinement stage both improve mask quality, mirroring the box-tracking results.",
      "Score-only two-branch vs three-branch variants quantify what each branch adds: 0.265 (two-branch) vs 0.363/0.380 (three-branch family) EAO on VOT2018.",
      "Mask supervision uses only positive RoIs (anchor IoU ≥ 0.6); λ₁ = 32 follows the mask-detector convention rather than a tuned value.",
    ],
    limitations: {
      authorStated: [
        "Failure cases are motion blur and 'non-object' instances, attributed to a lack of similar training samples.",
        "The Opt box-from-mask strategy is computationally expensive, which is why the paper recommends MBR for practical use.",
      ],
      evident: [
        "No online appearance update: the template is fixed at the first frame, so long occlusions and appearance drift are handled only by offline training.",
        "Box quality is capped by mask quality — since the box is derived from the mask, segmentation errors propagate straight into localization.",
      ],
    },
    assumptions: [
      "A first-frame bounding box is given (plus the first-frame mask for the VOS task).",
      "One target per sequence; the target remains inside the 255×255 search region.",
    ],
    computation:
      "55 FPS on a single Nvidia RTX 2080 (feature extractor dominates the cost); the Opt box conversion drops to about 5 FPS; ResNet-50 backbone, 127×127 exemplar / 255×255 search.",
    relations: [
      {
        to: "T009",
        type: "builds-on",
        note: "Keeps SiamFC's fully-convolutional Siamese template/search design and cross-correlation, adding box and mask branches on top of the same score-map machinery.",
      },
      {
        to: "T022",
        type: "uses-as-baseline",
        note: "DaSiamRPN is the strongest compared box tracker (VOT2018 EAO 0.326 vs SiamMask 0.380).",
      },
    ],
    concepts: ["siamese", "sot", "iou", "detection", "deep-detection", "appearance-features"],
    impact:
      "SiamMask turned mask output into a standard add-on for Siamese trackers and gave the field its reference box-from-mask study; later anchor-free trackers inherit its dense prediction idea, and Siam R-CNN (T037) cites it as the mask-based competitor it must beat on the VOS benchmarks.",
  },
  {
    id: "T028",
    arxiv: "1812.11703",
    title: "SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks",
    shortTitle: "SiamRPN++",
    year: 2018,
    authors: ["Bo Li", "Wei Wu", "Qiang Wang", "Fangyi Zhang", "Junliang Xing", "Junjie Yan"],
    fileName: "1812.11703v1.pdf",
    task: "single-object",
    tags: ["siamese", "rpn-head", "deep-networks", "translation-invariance", "depthwise-correlation", "real-time"],
    difficulty: "intermediate",
    summary:
      "SiamRPN++ diagnoses why deep backbones broke Siamese tracking — padding destroys the strict translation invariance the matching function relies on — and repairs it with a spatial-aware sampling strategy, then adds layer-wise feature aggregation and depth-wise cross-correlation to scale the RPN design onto ResNet, VGG and MobileNet. It is the first Siamese tracker to overtake the correlation-filter era on VOT2018, TrackingNet, LaSOT, OTB and UAV123, at 35 FPS.",
    problem:
      "Siamese trackers were kept deliberately shallow (modified AlexNet) because their design assumes strict translation invariance, and modern deep networks violate that assumption through padding — producing a spatial bias that collapses responses toward the image centre. Separately, the RPN correlation head explodes in parameters when applied naively to deep, high-channel features, and deep features of a single layer proved insufficient for tracking.",
    background: ["siamese", "rpn-head", "deep-detection", "bounding-box", "sot", "appearance-features"],
    previousWork: [
      {
        name: "SiamFC-style shallow Siamese trackers",
        limitation:
          "Their architecture is intrinsically restricted to small, shallow networks to preserve translation invariance, so performance saturated with AlexNet-era backbones.",
        whyThisPaper:
          "Spatial-aware sampling restores the invariance property under padding, making ResNet-50/VGG/MobileNet usable and lifting EAO through layer-wise aggregation alone.",
      },
      {
        name: "SiamRPN / DaSiamRPN (anchor-based correlation head)",
        limitation:
          "Works only with small networks: the RPN requires asymmetrical features for classification and regression, which conflicts with the symmetric Siamese structure and with deep padded backbones.",
        whyThisPaper:
          "The paper's analysis of the two intrinsic restrictions plus spatial-aware sampling and depth-wise correlation make deep RPN tracking work at all.",
      },
      {
        name: "VOT2018 leaders (LADCF and the challenge winner)",
        limitation:
          "Siamese trackers still trailed correlation-filter and MDNet-era methods on the benchmarks that mattered most.",
        whyThisPaper:
          "SiamRPN++ reaches EAO 0.414, exceeding LADCF (0.389) by 6.4% relative and beating the challenge winner by 2.5% absolute.",
      },
    ],
    researchGap:
      "Before this paper, no Siamese tracker could use a genuinely deep backbone without breaking the strict translation invariance on which its matching function depends.",
    contribution: [
      "Analysis: strict translation invariance exists only in no-padding networks such as modified AlexNet; padding in deeper networks creates a spatial bias, verified by shift-range simulations that expose a centre bias when the training shift range is zero.",
      "Spatial-aware sampling: random translation augmentation (a ±64 px shift range is reported as vital — zero-shift training collapses to 0.14 EAO on VOT2018) restores the invariance property for padded deep networks.",
      "Layer-wise feature aggregation: conv3/conv4/conv5 features are merged with learnable weighted sums (separate weights for classification and regression) into the RPN, worth +4.0% EAO on VOT2018 over a single-layer conv4.",
      "Depth-wise cross-correlation: replaces the full cross-correlation with a depth-wise version, adding +2.3% EAO on VOT2018 and +0.8% success on OTB with far fewer parameters.",
      "State of the art at 35 FPS: VOT2018 EAO 0.414, TrackingNet AUC 73.3, LaSOT 49.6, OTB-2015 0.696, UAV123 0.613; a MobileNetV2 variant runs at 70 FPS.",
    ],
    method: {
      pipeline: ["extract", "aggregate-levels", "correlate", "classify", "regress", "refine", "smooth"],
      architecture:
        "Shared backbone (modified AlexNet, VGG-16, ResNet-50 or MobileNetV2) for the 127×127 template and 255×255 search patches; layer-wise aggregation sums conv3/conv4/conv5 (each reduced to 256 channels) with learnable weights into the Siamese RPN; depth-wise separable correlation feeds heads predicting 2k classification and 4k regression maps with 5 anchors per location.",
      motionModel:
        "None beyond search centring: the selected box is smoothed with a cosine window and scale-change penalty inherited from SiamFC, then refined by RPN proposal regression.",
      appearanceModel:
        "First-frame template features only, no online update — all adaptation is pushed into offline training with heavy spatial-aware augmentation.",
      detectionDependency: "None — class-agnostic one-shot tracking with no external detector.",
      loss:
        "Sum of the classification loss and the standard smooth L1 regression loss of the RPN branch (as stated in the training details), trained end-to-end with the Siamese matching objective.",
      optimization:
        "SGD on 8 GPUs with synchronized gradients, 128 pairs per minibatch (16 per GPU), about 12 hours to converge; warmup lr 0.001 for the first 5 epochs, then exponential decay from 0.005 to 0.0005 over 15 epochs; weight decay 5e-4, momentum 0.9; the last 15 epochs fine-tune the whole network — backbone fine-tuning under spatial-aware sampling is reported as critical.",
    },
    equations: [
      {
        id: "siamrpnpp-match",
        label: "Siamese similarity function",
        formula: "f(z, x) = φ(z) ∗ φ(x) + b",
        variables: [
          { symbol: "z", meaning: "exemplar patch cropped from the first frame (127×127)" },
          { symbol: "x", meaning: "search patch centred on the previous target position (255×255)" },
          { symbol: "φ", meaning: "shared feature embedding (backbone + layer-wise aggregation)" },
          { symbol: "b", meaning: "learned offset modelling the similarity baseline" },
        ],
        intuition:
          "Embed template and search image with the same network and slide one over the other to get a similarity map.",
        why:
          "This is the function whose translation behaviour the whole paper studies — everything else (sampling, aggregation, correlation) exists to keep it well-behaved under deep, padded backbones.",
        where: "Section 3.1, Eq. 1; the definition of the Siamese tracking problem as the paper poses it.",
        params:
          "With correlation (not concatenation) the layers must be symmetric, which the paper lists as the second intrinsic restriction f(z, x₀) = f(x₀, z).",
        paperIds: ["T028"],
      },
      {
        id: "siamrpnpp-invariance",
        label: "Strict translation invariance condition",
        formula: "f(z, x[τ↓]) = f(z, x)[τ↓]",
        variables: [
          { symbol: "[τ↓]", meaning: "translation shift sub-window operator that crops/shifts by τ" },
          { symbol: "x", meaning: "search image before the shift" },
        ],
        intuition:
          "Shifting the input must shift the response map identically — the network cannot care where in the image the target appears.",
        why:
          "It justifies the efficient dense training/inference of Siamese trackers, and the paper shows deep padded networks violate it (creating spatial bias), which is precisely what spatial-aware sampling repairs.",
        where: "Section 3.1 (Analysis on Siamese Networks for Tracking), stated as the first intrinsic restriction.",
        params:
          "Simulation: training with zero shift learns a strong centre bias; with shift range 32 the aggregate heatmap matches the test distribution, and zero-shift ResNet-50 training scores only 0.14 EAO on VOT2018.",
        paperIds: ["T028"],
      },
    ],
    datasets: ["otb", "vot", "trackingnet", "lasot", "uav123"],
    metrics: ["eao", "success-auc", "precision", "norm-precision", "fps"],
    baselines: ["SiamFC", "DaSiamRPN", "ECO", "MDNet", "LADCF", "CCOT"],
    results: [
      "VOT2018: EAO 0.414, accuracy 0.600, robustness 0.234 — +6.4% relative EAO over LADCF (0.389) and +10.3% over DaSiamRPN in robustness.",
      "TrackingNet test: AUC 73.3, precision 69.4, normalized precision 80.0 vs DaSiamRPN 63.8 / 59.1 / 73.4.",
      "LaSOT: AUC 49.6, a +24.9% relative gain over MDNet, the previous LaSOT leader.",
      "OTB-2015 success AUC 0.696 with ResNet-50 and depth-wise correlation; UAV123 success 0.613 vs DaSiamRPN 0.586 and ECO 0.525.",
      "Speed: 35 FPS with ResNet-50; the MobileNetV2 variant reaches 70 FPS.",
      "Long-term tracking section: enlarging the search window and enlarging training crops for large objects — presented as the paper's approach to disappearing/reappearing targets.",
    ],
    ablations: [
      "Layer-wise feature aggregation (VOT2018): a single conv4 layer already reaches 0.374 EAO, and aggregating deeper layers lifts it to 0.414 (+4.0%) — while lower-layer-only combinations do not help.",
      "Depth-wise vs cross-channel correlation: +2.3% EAO on VOT2018 and +0.8% success on OTB at far lower parameter count.",
      "Spatial-aware sampling: zero-shift training collapses to a centre bias and scores 0.14 EAO on VOT2018; a ±64 px shift range is essential, and heatmap simulations show the learned position distribution matching the test data only when shifts are used.",
      "Backbone fine-tuning: the last 15 epochs fine-tune the whole network; the paper reports backbone fine-tuning (with spatial-aware sampling) as critical to all downstream gains.",
    ],
    limitations: {
      authorStated: [
        "Robustness still lags behind correlation filters that update their model online — SiamRPN++ does not update the template during the run, so long-term appearance change is unaddressed.",
      ],
      evident: [
        "Five hand-designed anchors per location remain fixed hyper-parameters; the anchor/classification ambiguity they create is exactly what SiamFC++ (T035) later attacks.",
        "Padding-induced spatial bias is mitigated through training-time augmentation rather than removed architecturally, so the guarantee is statistical, not structural.",
      ],
    },
    assumptions: [
      "The first-frame template stays representative (there is no template update).",
      "The target remains inside the 255×255 search patch centred on the previous position.",
    ],
    computation:
      "35 FPS (ResNet-50) / 70 FPS (MobileNetV2) at inference; training: 8 GPUs × 16 pairs per GPU, 128 pairs per minibatch, ~12 hours.",
    relations: [
      {
        to: "T009",
        type: "builds-on",
        note: "Extends SiamFC's fully-convolutional Siamese template/search formulation with an RPN head, deep backbones and layer-wise aggregation.",
      },
      {
        to: "T022",
        type: "uses-as-baseline",
        note: "DaSiamRPN is the direct competitor it beats on VOT2018 (robustness) and TrackingNet (63.8 vs 73.3 AUC).",
      },
      {
        to: "T011",
        type: "uses-as-baseline",
        note: "ECO is compared across VOT2018, OTB and UAV123 tables.",
      },
      {
        to: "T003",
        type: "uses-as-baseline",
        note: "MDNet is the LaSOT leader that SiamRPN++ surpasses by +24.9% relative AUC.",
      },
      {
        to: "T024",
        type: "uses-as-baseline",
        note: "Evaluated on the LaSOT benchmark (AUC 49.6).",
      },
    ],
    concepts: ["siamese", "rpn-head", "deep-detection", "sot", "bounding-box", "appearance-features"],
    impact:
      "SiamRPN++ made deep backbones the default for Siamese tracking and its three fixes (spatial-aware sampling, layer-wise aggregation, depth-wise correlation) were absorbed into the whole subsequent family; SiamBAN (T038) and Ocean (T046) keep its skeleton while removing its anchors, and SiamFC++ (T035) exists to attack exactly the anchor ambiguity it left behind.",
  },
  {
    id: "T029",
    arxiv: "1901.01660",
    title: "Deeper and Wider Siamese Networks for Real-Time Visual Tracking",
    shortTitle: "SiamFC+/SiamRPN+",
    year: 2019,
    authors: ["Zhipeng Zhang", "Houwen Peng"],
    fileName: "1901.01660v3.pdf",
    task: "single-object",
    tags: ["siamese", "backbone-design", "residual-units", "receptive-field", "real-time"],
    difficulty: "intermediate",
    summary:
      "SiamDW asks why deeper and wider backbones degraded Siamese tracking instead of improving it, and answers by measuring the internal factors that actually matter — receptive field, network stride, output feature size and convolution padding. It then designs backbones for tracking: crop-inside-residual (CIR) units strip padding-contaminated features inside the residual branch, giving CIResNet-22/29/43, CIResNeXt and CIResIncep backbones that lift the original SiamFC and SiamRPN by up to 9.8%/5.7% AUC on OTB and 24.4%/25.0% EAO on VOT at 150 FPS.",
    problem:
      "Siamese trackers still used a 5-layer AlexNet while the rest of vision moved to deeper and wider networks, yet naively swapping in ResNet, VGG or Inception made tracking worse — counter to the general belief that deeper and wider is better. The paper identifies the cause: receptive field controls how much context a neuron sees, network stride controls localization precision and output feature size, and padding in fully-convolutional architectures induces a position bias that appears whenever the target moves near the search-window boundary.",
    background: ["siamese", "deep-detection", "bounding-box", "sot", "appearance-features"],
    previousWork: [
      {
        name: "SiamFC and SiamRPN on an AlexNet-era backbone",
        limitation:
          "Accuracy is capped by the shallow classical backbone, and Figure 1 shows that simply replacing AlexNet with deeper or wider standard networks degrades both trackers instead of helping them.",
        whyThisPaper:
          "Keeping the two frameworks untouched and swapping in CIResNet-22 yields 9.8% (SiamFC+) and 5.7% (SiamRPN+) relative AUC on OTB and 23.3%/8.8% relative EAO on VOT-16.",
      },
      {
        name: "Off-the-shelf residual backbones (ResNet, ResNeXt, Inception)",
        limitation:
          "Their receptive field, stride, output size and padding are tuned for classification, not localization; standard residual blocks let padding signals propagate (Table 2, Table 1 padding sweep).",
        whyThisPaper:
          "CIR units crop the padding-affected features before BN/ReLU, and the architecture search over RF/OFS/STR restores stride-8 localization: the plain residual unit gives 0.213 EAO where CIR gives 0.301 on CIResNet-22.",
      },
      {
        name: "Correlation-filter and hybrid trackers SRDCF, ECO-HC, CFNet (Table 5)",
        limitation:
          "Real-time but accuracy-capped hand-crafted/hybrid approaches (OTB-2013 AUC 0.61–0.65, precision up to 0.87).",
        whyThisPaper:
          "SiamFC+ and SiamRPN+ reach 0.67 AUC and up to 0.92 precision on OTB-2013 while running at 70 and 150 FPS.",
      },
    ],
    researchGap:
      "Before this paper, it was unknown which internal network factors made deep or wide backbones hurt Siamese tracking, so no backbone had been designed for tracking's receptive-field, stride and padding requirements.",
    contribution: [
      "Factor study: receptive field, network stride, output feature size and padding are swept across AlexNet, VGG, Inception and ResNet (Tables 1, 2 and 8), showing stride 16 and padding cause the largest drops and that RF/OFS/STR choices interact.",
      "CIR (crop-inside-residual) and CIR-D (downsampling variant) units: crop the input inside the residual branch before BN/ReLU so convolution filters cannot learn position bias from padding signals.",
      "A tracking-oriented backbone zoo with controlled receptive field (the design goal is 60–80% of the exemplar image) and stride 8: CIResNet-16/19/22/43, CIResNeXt-22 and CIResIncep-22, with width increased 2× and 32× for the wider families.",
      "SiamFC+ and SiamRPN+ built on CIResNet-22: up to 9.8%/5.7% relative AUC on OTB, 23.3%/8.8% EAO on VOT-16 and 24.4%/25.0% EAO on VOT-17 over the originals, at 70 and 150 FPS.",
    ],
    method: {
      pipeline: ["extract", "correlate", "classify", "regress", "smooth"],
      architecture:
        "The unmodified SiamFC and SiamRPN frameworks (127×127 exemplar, 255×255 search; three-scale search with factor 1.0482^{−1,0,1} and linear-interpolation damping 0.3629 for SiamFC+, single-scale RPN search with a size/aspect penalty for SiamRPN+) whose AlexNet is replaced by a CIR-based backbone, best result with CIResNet-22 at stride 8.",
      motionModel:
        "None: the search patch is centred on the previous position; scale is handled by SiamFC+'s three-scale search with interpolation damping, or by SiamRPN+'s proposal refinement and change penalty.",
      appearanceModel:
        "First-frame template only, no online update — the paper's claim is that the gains come 'solely due to the proposed network architectures'.",
      detectionDependency: "None — class-agnostic one-shot tracking with no external detector.",
      loss:
        "The original logistic losses of SiamFC and SiamRPN (Eq. 2: minimizing E ℓ(y, fθ(z, x)) over random training pairs) — unchanged; only the backbone differs.",
      optimization:
        "Offline training identical to the two base trackers: backbones initialized from ImageNet-pretrained weights, SiamFC pairs sampled from ImageNet VID, SiamRPN pairs from ImageNet VID and Youtube-BB; inference follows the official SiamFC/SiamRPN protocols for fair comparison.",
    },
    equations: [
      {
        id: "siamdw-match",
        label: "Siamese similarity function",
        formula: "fθ(z, x) = ϕθ(z) ⋆ ϕθ(x) + b · 1",
        variables: [
          { symbol: "z", meaning: "exemplar patch centred on the target in the first frame (127×127)" },
          { symbol: "x", meaning: "candidate search patch centred on the previous position (255×255)" },
          { symbol: "ϕθ", meaning: "shared ConvNet embedding — here the CIR-based backbone under study" },
          { symbol: "b · 1", meaning: "a scalar bias b ∈ R broadcast to every location of the response map" },
        ],
        intuition:
          "Slide the exemplar's features over the search image with the same network and read off a similarity map; the argmax is the predicted target location.",
        why:
          "This is the function whose behaviour the whole paper studies: receptive field, stride and padding all change what this response map looks like, which is why a backbone swap can help or hurt.",
        where: "Section 3 (matching formulation), Eq. 1; the definition inherited from SiamFC/SiamRPN.",
        params:
          "Eq. 1 is an exhaustive search of the pattern z over x — a larger receptive field gives more context, stride 8 keeps localization finer than stride 16.",
        paperIds: ["T029"],
      },
      {
        id: "siamdw-objective",
        label: "Offline training objective",
        formula: "θ* = arg min_θ E(z,x,y) ℓ(y, fθ(z, x))",
        variables: [
          { symbol: "(z, x, y)", meaning: "random image pair from training videos with its ground-truth label" },
          { symbol: "ℓ", meaning: "logistic loss used to train the similarity function" },
          { symbol: "θ", meaning: "all ConvNet parameters, including the CIR backbone" },
        ],
        intuition:
          "Train the shared embedding so that matching pairs score high and non-matching pairs score low — no online learning at test time.",
        why:
          "Because everything is learned offline, architecture quality (the paper's subject) is directly visible in benchmark results with no update mechanism to hide behind.",
        where: "Section 3 (matching formulation), Eq. 2; minimized during offline training only.",
        params:
          "Pair construction differs per framework: ImageNet VID pairs for SiamFC, ImageNet VID and Youtube-BB pairs for SiamRPN.",
        paperIds: ["T029"],
      },
    ],
    datasets: ["otb", "vot"],
    metrics: ["success-auc", "precision", "eao", "fps"],
    baselines: ["SiamFC", "SiamRPN", "ECO-HC", "CFNet", "MDNet", "LSART"],
    results: [
      "Abstract claim: SiamFC+/SiamRPN+ gain up to 9.8%/5.7% AUC on OTB and 23.3%/8.8% (VOT-16) and 24.4%/25.0% (VOT-17) relative EAO over the original SiamFC and SiamRPN.",
      "Backbone swap in fixed frameworks (Table 4): SiamFC OTB AUC 0.608 → 0.662 and VOT-17 EAO 0.188 → 0.234 from AlexNet to CIResNet-22; SiamRPN 0.637 → 0.662 and 0.244 → 0.301, at 70/150 FPS versus 101/190 FPS.",
      "OTB (Table 5): SiamFC+ AUC 0.67 / precision 0.88 (OTB-2013) and 0.64 / 0.85 (OTB-2015); SiamRPN+ 0.67 / 0.92 and 0.67 / 0.90 — versus SiamFC 0.61/0.81, SiamRPN 0.64/0.85, ECO-HC 0.65/0.87 and CFNet 0.61/0.80.",
      "VOT (Table 5, EAO): SiamRPN+ 0.38 (VOT-15), 0.37 (VOT-16), 0.30 (VOT-17); SiamFC+ 0.31 / 0.30 / 0.23 — versus SiamRPN 0.35 / 0.34 / 0.24 and SiamFC 0.29 / 0.24 / 0.19.",
      "VOT-16: SiamRPN+ ranks first in EAO, beating challenge winner CCOT by 3.9 points and VITAL by 4.7 points, while SiamFC+ improves its baseline by 6.0 EAO points; on VOT-15 SiamRPN+ is best, 'slightly better than MDNet' which runs at about 1 FPS.",
      "VOT-17: SiamRPN+ is slightly inferior to LSART but runs at 150 FPS — 150× faster — and ranks first among real-time trackers; plain SiamFC+ still beats CSRDCF++ by 2.2 points.",
    ],
    ablations: [
      "CIR versus plain residual unit (Table 6, VOT-16 EAO): CIResNet-20 0.204 → 0.271, CIResNet-22 0.213 → 0.301, CIResIncep-22 0.227 → 0.282 — removing the crop costs 8.8 points (0.301 → 0.213).",
      "Downsampling placement (Table 7): CIR-D gives 0.271 / 0.301 / 0.282 on the three backbones, ahead of Setting 1 (0.264 / 0.292 / 0.266) and Setting 2 (0.259 / 0.287 / 0.275).",
      "Internal factors (Table 8): sweeping RF/OFS/STR around CIResNet-22 moves VOT-16 EAO between 0.23 and 0.30 — stride 16 is the worst setting, matching Table 1's observation that moving stride from 4 or 8 to 16 drops performance sharply and that padding consistently hurts.",
      "Depth and width limits: CIResNet-43 does not improve over CIResNet-22 because stride becomes 4 (overlapping receptive fields) and output channels halve (256 vs 512); CIResNeXt-22 underperforms CIResNet-22 because its model size is smaller despite more branches.",
    ],
    limitations: {
      authorStated: [
        "CIResNet-43 fails to improve on shallower variants: with stride 4 neighbouring receptive fields overlap so localization is less precise, and halved channels shrink the model (256 vs 512).",
        "CIResNeXt-22 is inferior to CIResNet-22 and CIResIncep-22 — the paper attributes this to its smaller model size rather than to grouped convolutions per se.",
        "On VOT-17 SiamRPN+ is 'slightly inferior to the best performing LSART', so accuracy still trails the correlation-filter leader.",
      ],
      evident: [
        "Nothing adapts at test time: the template is fixed after frame one, so long-term appearance change is unhandled exactly as in SiamFC/SiamRPN.",
        "Gains are shown only by swapping backbones inside two fixed frameworks, so architecture quality cannot be separated from framework-specific training choices.",
      ],
    },
    assumptions: [
      "The first-frame target appearance stays representative for the whole sequence.",
      "The target remains inside the 255×255 search patch centred on the previous position.",
    ],
    computation:
      "150 FPS for SiamRPN+ and 70 FPS for SiamFC+ with CIResNet-22 (Table 4); 150× faster than LSART on VOT-17.",
    relations: [
      {
        to: "T009",
        type: "builds-on",
        note: "SiamFC+ is SiamFC unchanged except for the CIR backbone; the paper's whole ablation runs inside the SiamFC framework.",
      },
      {
        to: "T011",
        type: "uses-as-baseline",
        note: "ECO-HC is compared in Table 5 (OTB-2013 AUC 0.65 / precision 0.87 versus SiamFC+ 0.67 / 0.88).",
      },
      {
        to: "T014",
        type: "uses-as-baseline",
        note: "CFNet appears in Table 5 (OTB-2013 0.61 / 0.80, OTB-2015 0.59 / 0.78) and is explicitly named as a beaten competitor.",
      },
      {
        to: "T003",
        type: "uses-as-baseline",
        note: "MDNet is the VOT-15 reference SiamRPN+ surpasses — 'slightly better than MDNet', which runs at about 1 FPS.",
      },
    ],
    concepts: ["siamese", "deep-detection", "bounding-box", "sot", "appearance-features"],
    impact:
      "SiamDW made 'design the backbone for tracking' an accepted line of work: its CIR units and stride-8 recipe became the default deeper alternative to AlexNet inside Siamese trackers, and its factor study (RF, stride, output size, padding) is the standard explanation cited whenever a detection backbone is transplanted into a tracker.",
  },
  {
    id: "T030",
    arxiv: "1903.05625",
    title: "Tracking without bells and whistles",
    shortTitle: "Tracktor",
    year: 2019,
    authors: ["Philipp Bergmann", "Tim Meinhardt", "Laura Leal-Taixé"],
    venue: "ICCV 2019",
    fileName: "1903.05625v3.pdf",
    task: "multi-object",
    tags: ["tracking-by-detection", "box-regression", "online", "reid", "oracle-analysis"],
    difficulty: "intro",
    summary:
      "Tracktor turns an off-the-shelf object detector into a tracker: the Faster R-CNN box-regression branch is applied to each existing track's previous box to predict its position in the next frame, tracks are killed by classification score or NMS overlap, and unmatched detections spawn new tracks. Two small extensions — a short-term Siamese re-identifier and camera-motion compensation — form Tracktor++, which reaches 61.9 MOTA and 64.7 IDF1 on the MOT17 test set, while an oracle study quantifies how much each remaining subsystem would still buy.",
    problem:
      "Multi-object tracking had nearly stalled (+2.4% MOTA on MOT16 over two years) while trackers accumulated motion models, appearance models, graph optimizers, plotters and offline smoothing with many hyper-parameters. Nobody had measured how far the detector's own box regression can carry tracking, nor isolated with ground-truth oracles how much MOTA and IDF1 each assumed-essential component actually contributes.",
    background: ["mot", "tracking-by-detection", "deep-detection", "bounding-box", "iou", "reid", "mota"],
    previousWork: [
      {
        name: "Complex tracking-by-detection pipelines (MHT-DAM, jCC, FWT, D&T, eHAF)",
        limitation:
          "They layer motion models, appearance models, graph association and post-processing on top of the detections, yet on MOT17 public detections they score only 47–55 MOTA and 47–55 IDF1.",
        whyThisPaper:
          "A bare detector with regression reaches 61.5 MOTA and 61.1 IDF1 on exactly the same detections, with no tracking-specific training.",
      },
      {
        name: "Re-identification and camera-motion as entangled add-ons",
        limitation:
          "Their contribution is folded into full systems, so what they are worth — and what perfect versions of them could add — was never measured.",
        whyThisPaper:
          "Tracktor adds each in isolation (+reID cuts ID switches 1747 → 921, +CMC → 458) and the oracle study bounds their potential (Oracle-reID +10.0 IDF1, Oracle-MM +5.2 IDF1).",
      },
    ],
    researchGap:
      "Before this paper, nobody had used a detector's box-regression head as the tracker itself, nor measured with ground-truth oracles how much MOTA and IDF1 perfect detection, perfect regression, motion, re-identification and interpolation would each add.",
    contribution: [
      "Tracktor: a Faster R-CNN (ResNet-101 with FPN, trained on MOT17Det) whose regression branch predicts each track's next box; a track dies when its regressed box's classification score drops below the active threshold or when NMS overlap says it duplicates a fresh detection; a detection starts a track when it survives NMS against all existing boxes.",
      "Tracktor++: two extensions — short-term Siamese re-identification over cropped detections, and camera-motion compensation by ECC image registration (euclidean for rotating cameras, affine where translation matters), plus a constant-velocity option for low-frame-rate sequences.",
      "Oracle study (Table 4): ground-truth substitutions for kill rule, regression, motion model, re-ID and interpolation, ending in Oracle-ALL at +10.7 MOTA / +22.5 IDF1 — the paper's way of showing where the remaining headroom is.",
      "Coverage analysis by object visibility, height and detection-gap length: Tracktor stays superior even for boxes with visibility as low as 0.3 and covers gaps longer than 15 frames better than dedicated trackers.",
      "A new tracking paradigm: 'a trained Faster R-CNN detector is enough to solve most' of tracking, with no training or optimization on tracking ground-truth data.",
    ],
    method: {
      pipeline: ["detect", "regress", "prune", "spawn", "extend", "kill"],
      architecture:
        "Faster R-CNN pedestrian detector (ResNet-101 backbone with FPN, crop-and-resize RoI pooling instead of RoI pooling, 256 extracted regions per image) run per frame; the tracking layer is bookkeeping over its boxes — classification scores, IoU/NMS overlap and per-track history — with no learned tracking component.",
      motionModel:
        "None in Tracktor: the next position is one step of the detector's own box regression on the previous box. Tracktor++ adds camera-motion compensation via ECC image registration (euclidean or affine) and, for low frame rates, a constant-velocity model that shifts boxes before regression.",
      appearanceModel:
        "Optional short-term re-identification: a Siamese CNN embeds detection crops and re-attaches broken tracklets by distance threshold; the network is trained on tracking ground-truth data.",
      detectionDependency:
        "Total — every decision is a function of detector output; comparisons use the MOT17 public detections, and the paper's premise is that the detector, not the tracker, dominates performance.",
      trackManagement:
        "Kill by classification score (deactivate when the regressed box's score falls below the active threshold) or by NMS overlap with a fresh detection; unassigned detections are suppressed by NMS and the survivors spawn new tracks.",
      loss: "None at tracking time — the tracker has no objective of its own; the detector's classification and box losses were learned offline on MOT17Det.",
      optimization:
        "No online optimization: 'our approach requires no dedicated training or optimization on tracking ground truth data' — only the detector is trained offline (ResNet-101 FPN, 256 regions per image, crop-and-resize pooling).",
    },
    equations: [],
    datasets: ["mot16", "mot17", "mot15"],
    metrics: ["mota", "idf1", "fp", "fn", "idsw", "mt-ml"],
    baselines: ["MHT-DAM", "jCC", "FWT", "MOTDT17", "D&T", "eHAF"],
    results: [
      "MOT17 test, public detections (Table 2): Tracktor MOTA 61.5, IDF1 61.1, MT 33.5, ML 20.7, FP 367, FN 42903, ID switches 1747; Tracktor++ 61.9 / 64.7 with FP 323, FN 42454 and ID switches cut to 326.",
      "Extensions measured separately on MOT17: +reID 61.5 / 62.8 (IDs 1747 → 921), +CMC 61.9 / 64.1 (IDs → 458), Tracktor-no-FPN 57.4 / 58.7.",
      "Same detections, far more complex trackers: FWT 51.3 / 47.6, jCC 51.2 / 54.5, MOTDT17 50.9 / 52.7, MHT-DAM 50.7 / 47.2, eHAF 51.8 / 54.7 — Tracktor leads each by roughly 10 MOTA points.",
      "MOT16 test: Tracktor++ MOTA 54.4, IDF1 52.5, FP 3280, FN 79149, ID switches 682; 2D MOT15: MOTA 44.1, IDF1 46.7, ID switches 1318.",
      "Oracle ladder (Table 4, ΔMOTA / ΔIDF1 versus Tracktor): +Tracktor++ +0.4 / +3.6, Oracle-Kill +0.7 / −0.7, Oracle-REG +1.4 / +5.6, Oracle-MM +0.9 / +5.2, Oracle-reID 0.0 / +10.0, Oracle-MM-reID +0.9 / +13.9, Oracle-MM-reID-INTER +2.6 / +15.9, Oracle-ALL +10.7 / +22.5.",
      "Analysis: Tracktor has the best coverage even for boxes with visibility as low as 0.3, where neither MHT-DAM's identity handling nor jCC's offline interpolation helps; high MOTA is largely an artefact of the unbalanced visibility distribution.",
    ],
    ablations: [
      "FPN removal: 61.5 → 57.4 MOTA — detector feature quality, not tracking logic, dominates the outcome.",
      "Extension ablation (Table 2): reID and CMC are added one at a time; each cuts ID switches sharply while MOTA moves by at most 0.4 points.",
      "Oracle substitution: each oracle swaps ground truth for one subsystem (kill rule, regression, motion, re-ID, interpolation) to bound its potential — the paper's substitute for a component ablation.",
    ],
    limitations: {
      authorStated: [
        "Tracktor is not expected to excel on the scenarios other trackers target — heavy occlusions and small objects — and the authors call for a better motion model to carry targets through long occlusions.",
        "The gap analysis shows that no tracker, including Tracktor, convincingly covers detection gaps longer than about 15 frames without offline interpolation or motion prediction.",
        "Large camera motion and low frame rate are the two scenarios where the detector-only assumption does not hold.",
      ],
      evident: [
        "All errors are inherited from the detector: FN 42903 on MOT17 means most misses can never be recovered by association, and Oracle-ALL says +10.7 MOTA is still unclaimed.",
        "The base tracker keeps no appearance or motion memory, so identity through occlusion depends entirely on the optional extensions (ID switches 1747 → 326 with both).",
      ],
    },
    assumptions: [
      "A pedestrian detector trained on MOT17Det is available and runs every frame.",
      "Objects move smoothly enough frame-to-frame that one regression step predicts the next box.",
    ],
    computation:
      "One Faster R-CNN forward pass per frame plus cheap bookkeeping; comparisons reuse the shared MOT17 public detections, so the tracking layer itself adds negligible cost.",
    relations: [
      {
        to: "T006",
        type: "uses-as-baseline",
        note: "Evaluates on the MOT16 benchmark under the public and private detection protocols (Tracktor++: MOTA 54.4, IDF1 52.5).",
      },
    ],
    concepts: ["tracking-by-detection", "deep-detection", "bounding-box", "iou", "mota", "occlusion"],
    impact:
      "Tracktor is the reference 'regression-as-tracker': it showed that the detector's own box head is a motion model, and its oracle ladder became the standard template for arguing what a better detector, motion model or re-identification would still buy on MOT benchmarks.",
  },
  {
    id: "T031",
    arxiv: "1904.07220",
    title: "Learning Discriminative Model Prediction for Tracking",
    shortTitle: "DiMP",
    year: 2019,
    authors: ["Goutam Bhat", "Martin Danelljan", "Luc Van Gool", "Radu Timofte"],
    fileName: "1904.07220v2.pdf",
    task: "single-object",
    tags: ["meta-learning", "unrolled-optimization", "correlation-filter", "online-learning", "real-time"],
    difficulty: "advanced",
    summary:
      "DiMP embeds model prediction directly in the architecture: an initializer network plus an unrolled steepest-descent module with learned Gauss-Newton step lengths predict a discriminative convolutional filter from a memory of sequence samples in one forward pass, and the loss the filter is trained against (a learned blend of least-squares and hinge) is itself part of the network. Keeping ATOM's IoU box branch, it reaches EAO 0.440 on VOT2018 and state-of-the-art results on LaSOT, TrackingNet, GOT-10k and NFS at 57 FPS.",
    problem:
      "Discriminative filter trackers update their model by iterative online optimization with fixed hyper-parameters — slow enough that frames must be subsampled and prone to drift when samples are contaminated — and the filters themselves are fit by hand rather than learned. Siamese trackers avoid the problem by never updating, which caps robustness. Neither camp learns how to update as part of the network.",
    background: ["sot", "correlation-filter", "appearance-features", "siamese", "deep-detection", "bounding-box"],
    previousWork: [
      {
        name: "ATOM (T026)",
        limitation:
          "Its classification head is trained online by optimizing a hand-designed loss, so adaptation is slow and the loss cannot be tuned for generalization to future frames.",
        whyThisPaper:
          "DiMP keeps ATOM's box branch and replaces classification with a meta-learned model predictor, lifting VOT2018 EAO from 0.401 to 0.440 with a 34% lower failure rate.",
      },
      {
        name: "Discriminative correlation-filter trackers (SRDCF, ECO, CCOT, LADCF)",
        limitation:
          "Filters are fitted online by fixed iterative solvers with hand-set regularization, sample weights and update schedules — not learned, not predicted, and unable to exploit multiple samples jointly.",
        whyThisPaper:
          "The filter is predicted by a network trained end-to-end with the tracker, so no solver runs at test time and the update policy is learned from data.",
      },
      {
        name: "Siamese trackers without model update (SiamFC, SiamRPN++)",
        limitation:
          "A fixed first-frame template cannot adapt, which surfaces as weaker robustness (SiamRPN++ VOT2018 robustness 0.234 versus DiMP-50's 0.153).",
        whyThisPaper:
          "DiMP maintains a memory of sequence samples and re-predicts the model from it, so the target model adapts while learned sample weighting keeps contamination bounded.",
      },
    ],
    researchGap:
      "Before this paper, no tracker learned to predict its own discriminative target model through an unrolled, meta-trained optimizer, so online model update remained a slow, hand-designed component.",
    contribution: [
      "Model predictor as a network: initializer (convolution + precise ROI pooling, averaged over the sample set) feeding unrolled steepest-descent iterations with closed-form step length α = ∇Lᵀ∇L/(∇LᵀQ∇L) using Gauss-Newton Q = JᵀJ — all implemented as network operations and trained end-to-end.",
      "Learned discriminative loss: residual r(s, c) = v_c·(m_c·s + (1−m_c)·max(0, s) − y_c) behaves as least-squares at the target and as a hinge in the background, with label y_c, target mask m_c, spatial weight v_c and regularization λ all predicted through radial basis functions (N = 100 triangular bases, knot spacing Δ = 0.1).",
      "Set-based training: model prediction is trained on pairs of sample sets (M_train, M_test) drawn from TrackingNet, LaSOT, GOT-10k and COCO segments, with the loss averaged over every unrolled iteration for intermediate supervision; 50 epochs × 20,000 videos, under 24 hours on a single TITAN X.",
      "Memory-based online update: 15 augmented samples and 10 recursions in the first frame, then two recursions every 20 frames or one whenever a distractor peak is detected, with memory capped at 50 samples (oldest discarded).",
      "State of the art on seven benchmarks at over 40 FPS: VOT2018 EAO 0.440, LaSOT 56.9, TrackingNet AUC 74.0, GOT-10k AO 61.1, NFS 61.9, OTB-100 68.4, UAV123 65.3.",
    ],
    method: {
      pipeline: ["extract", "predict-model", "score", "update-memory", "refine-box"],
      architecture:
        "Shared ResNet backbone (features from the third block, stride 16, for the predictor; kernel size 4×4 for the target model) feeding three modules: an initializer network (convolution + precise ROI pooling), an unrolled optimizer module that produces the filter f = D(S_train), and ATOM's IoU-prediction box head modulated by a first-frame reference appearance.",
      motionModel:
        "None: the search patch is centred on the previous point estimate, taken as the argmax of the classifier response.",
      appearanceModel:
        "A discriminative convolutional filter predicted from a memory of samples (first-frame augmentations plus score-ranked later frames); the RBF-parameterized classifier gives a non-linear decision function whose label, mask and weight functions are predicted from the distance to the target centre.",
      detectionDependency: "None — class-agnostic single-object tracking with no external detector.",
      trackManagement:
        "Target loss is detectable from calibrated confidences (the hinge form lets background scores go freely negative while target confidences stay accurate); memory is refreshed only when the target is predicted with sufficient confidence.",
      loss:
        "L_cls = (1/N_iter) Σ_i Σ_(x,c) ℓ(x ∗ f⁽ⁱ⁾, z_c)² with the hinge ℓ(s, z) = s − z if z > T else max(0, s), averaged over all iterations including the initializer's output; combined with the box loss as L_tot = β·L_cls + L_bb with β = 10².",
      optimization:
        "End-to-end offline training of initializer, optimizer module and box head together (ADAM, learning rate decay ×0.2 every 15th epoch, 50 epochs, 20,000 videos per epoch, ImageNet-initialized backbone, N_iter = 5 recursions during training); at test time the forward pass predicts the filter, so no gradient steps are required on a test sequence.",
    },
    equations: [
      {
        id: "dimp-residual",
        label: "Learned residual defining the discriminative loss",
        formula: "r(s, c) = v_c · ( m_c · s + (1 − m_c) · max(0, s) − y_c )",
        variables: [
          { symbol: "s", meaning: "classifier response score at a spatial location" },
          { symbol: "c", meaning: "target centre coordinate the sample is relative to" },
          { symbol: "m_c", meaning: "learned target mask in [0, 1]: 1 at the target, 0 in the background" },
          { symbol: "y_c", meaning: "learned regression label (confidence target) at that location" },
          { symbol: "v_c", meaning: "learned spatial weight, increased at the target centre" },
        ],
        intuition:
          "Treat the target neighbourhood as a regression problem (fit the score to a label) and the background as a hinge (only penalize positive scores), and blend the two with a learned mask.",
        why:
          "Least-squares near the target gives Gauss-Newton the structure it needs for fast convergence; the hinge in the background avoids punishing easy negatives, and learning m, v, y replaces hand-tuned loss design.",
        where: "Section 3.1 (Discriminative Model Prediction), Eq. 2; defines the loss minimized by the unrolled optimizer.",
        params:
          "m ≈ 1 at the target and m ≈ 0 far away; y_c, m_c, v_c and the regularization λ are all learned through RBF coefficients φ with N = 100 bases and knot spacing Δ = 0.1.",
        paperIds: ["T031"],
      },
      {
        id: "dimp-update",
        label: "Filter update inside the optimizer module",
        formula: "f⁽ⁱ⁺¹⁾ = f⁽ⁱ⁾ − α · ∇L(f⁽ⁱ⁾)",
        variables: [
          { symbol: "f⁽ⁱ⁾", meaning: "filter estimate at iteration i of the unrolled optimizer" },
          { symbol: "∇L", meaning: "gradient of the discriminative loss with respect to the filter" },
          { symbol: "α", meaning: "step length, computed in closed form rather than learned as a scalar" },
        ],
        intuition:
          "Each iteration takes one step down the loss surface; because the steps are unrolled, the whole sequence is a differentiable network.",
        why:
          "Iterative online optimization is exactly what made previous filter trackers slow and hand-tuned — unrolling it lets the network learn how to take those steps.",
        where: "Section 3.2 (Optimization-Based Architecture), Eq. 3; runs N_iter times per prediction.",
        params:
          "Trained with N_iter = 5 recursions; at test time 10 recursions on the first frame, then two every 20 frames or one on a distractor peak.",
        paperIds: ["T031"],
      },
      {
        id: "dimp-step",
        label: "Steepest-descent step length (Gauss-Newton)",
        formula: "α = ∇L(f⁽ⁱ⁾)ᵀ ∇L(f⁽ⁱ⁾) / ( ∇L(f⁽ⁱ⁾)ᵀ Q⁽ⁱ⁾ ∇L(f⁽ⁱ⁾) )",
        variables: [
          { symbol: "Q⁽ⁱ⁾", meaning: "positive-definite matrix defining the local quadratic model; set to JᵀJ" },
          { symbol: "J", meaning: "Jacobian of the residuals r at f⁽ⁱ⁾, implemented as network operations" },
        ],
        intuition:
          "Instead of a hand-set learning rate, compute the step size that minimizes a local quadratic approximation of the loss along the gradient direction.",
        why:
          "A constant step length makes gradient descent converge far too slowly for the few iterations available; Gauss-Newton gives second-order convergence using only first-order derivatives.",
        where: "Section 3.2, Eq. 4 (quadratic model) and Eq. 5 (step length); evaluated every unrolled iteration.",
        params:
          "Q = scaled identity recovers plain gradient descent with fixed step β; Q = JᵀJ is the Gauss-Newton choice used in the paper.",
        paperIds: ["T031"],
      },
      {
        id: "dimp-hinge",
        label: "Hinge loss for background samples",
        formula: "ℓ(s, z) = s − z  if z > T,  else  max(0, s)",
        variables: [
          { symbol: "s", meaning: "predicted confidence score" },
          { symbol: "z", meaning: "label confidence (a Gaussian centred on the target during offline training)" },
          { symbol: "T", meaning: "threshold separating target from background labels (T = 0.05 in training)" },
        ],
        intuition:
          "In the target region match the label exactly; in the background only require the score to be non-positive, leaving easy negatives free.",
        why:
          "Accurately calibrated target confidences are needed for loss detection, while the background must not force large negative values that would distort the filter.",
        where: "Section 3.6 (Offline Training), Eq. 8; applied to test samples inside the classification loss (Eq. 9).",
        params: "T defines the target/background split; only positive scores are penalized when z ≤ T.",
        paperIds: ["T031"],
      },
    ],
    datasets: ["vot", "lasot", "trackingnet", "got10k", "otb", "uav123", "others"],
    metrics: ["eao", "success-auc", "precision", "norm-precision", "fps"],
    baselines: ["SiamRPN++", "ATOM", "LADCF", "UPDT", "ECO", "MDNet"],
    results: [
      "VOT2018: DiMP-50 EAO 0.440, accuracy 0.597, robustness 0.153 (DiMP-18: 0.402 / 0.594 / 0.182) versus SiamRPN++ 0.414 with robustness 0.234, ATOM 0.401 and LADCF 0.389 — +6.3% relative EAO and a 34% lower failure rate at similar accuracy.",
      "TrackingNet test: precision 68.7, normalized precision 80.1, AUC 74.0 (DiMP-18: 66.6 / 78.5 / 72.3) versus SiamRPN++ 69.4 / 80.0 / 73.3.",
      "LaSOT: DiMP-50 success 56.9 and DiMP-18 53.2, +3.3% relative over ATOM's 51.5 with the same ResNet-18 backbone; the paper notes online adaptation is crucial on its 2,500-frame average sequences.",
      "GOT-10k test: AO 61.1, SR0.5 71.7, SR0.75 49.2 — best in the compared table (ATOM 55.6).",
      "NFS 30-FPS: AUC 61.9 versus ATOM 58.4 (+4.4% and +6.0% relative for ResNet-18 and ResNet-50); OTB-100 68.4 versus UPDT 70.4; UAV123 65.3 (DiMP-18 64.3) versus ATOM 64.2.",
      "Speed: 57 FPS with ResNet-18 and 43 FPS with ResNet-50 on a single Nvidia GTX 1080 (abstract: over 40 FPS).",
    ],
    ablations: [
      "Model prediction architecture (Table 1, pooled OTB-100 + NFS + UAV123): initializer only 58.2 AUC, gradient descent 61.6, steepest descent 63.8 — SD beats GD by 2.2 points and the initializer alone is 5.6 behind.",
      "Incremental components (Table 2): baseline steepest descent 58.7 → +initializer 60.0 → +backbone training 62.6 → +extra convolutional block 63.3 → +learning the loss 63.8.",
      "Update strategy (Table 3): no update 61.7, naive model averaging 61.7 (no gain over no update), DiMP's memory-based update 63.8 — about +2 AUC over both.",
    ],
    limitations: {
      authorStated: [
        "On OTB-100 the paper reports DiMP-50 at 68.4 AUC behind UPDT's 70.4 and calls its own result merely 'competitive'.",
        "Plain gradient descent with a learned (even per-coefficient) step length is explicitly described as insufficient — it needs a vast increase in iterations, which harms efficiency and complicates offline learning.",
      ],
      evident: [
        "The box branch is inherited unchanged from ATOM, so localization accuracy is bounded by that offline-trained IoU head; only the classification model is new.",
        "Memory is capped at 50 samples with the oldest discarded and updates are heuristic (triggered by confidence or a distractor peak), so long-term appearance change is only summarized, not stored.",
      ],
    },
    assumptions: [
      "A ground-truth box for the first frame is given and the target stays inside the search patch around the previous position.",
      "Past samples that scored well remain useful references for re-predicting the model.",
    ],
    computation:
      "57 FPS (ResNet-18) and 43 FPS (ResNet-50) on a single GTX 1080; inference is one forward pass through the predictor plus two optimizer recursions every 20 frames (or one on a distractor peak); training takes under 24 hours on one TITAN X.",
    relations: [
      {
        to: "T026",
        type: "builds-on",
        note: "Keeps ATOM's overlap-prediction box branch unchanged ('we utilize the overlap maximization strategy introduced in [6]', same network architecture) and replaces only classification.",
      },
      {
        to: "T028",
        type: "uses-as-baseline",
        note: "SiamRPN++ is the strongest compared tracker on VOT2018 (EAO 0.414, robustness 0.234) and TrackingNet (73.3 AUC).",
      },
      {
        to: "T021",
        type: "uses-as-baseline",
        note: "LADCF is the best previous discriminative-filter entry in the VOT2018 table (EAO 0.389) that DiMP-50 surpasses by 0.051.",
      },
      {
        to: "T024",
        type: "uses-as-baseline",
        note: "Evaluated on the LaSOT benchmark (success 56.9), where its online adaptation is presented as decisive.",
      },
      {
        to: "T025",
        type: "uses-as-baseline",
        note: "Evaluated on the GOT-10k benchmark (AO 61.1) under its train-only protocol.",
      },
    ],
    concepts: ["sot", "correlation-filter", "appearance-features", "online-vs-offline", "deep-detection"],
    impact:
      "DiMP turned 'learn the update rule' into the standard formulation for model-based tracking: its unrolled optimizer, learned loss and sample memory became the template that PrDiMP (T040) inherits wholesale (changing only the output distribution), and the predicted-filter-plus-IoU-head split defines the 2020-era tracker skeleton.",
  },
  {
    id: "T032",
    arxiv: "1905.02857",
    title: "Learning Cascaded Siamese Networks for High Performance Visual Tracking",
    shortTitle: "Cascaded Siamese Network",
    year: 2019,
    authors: ["Peng Gao", "Yipeng Ma", "Ruyue Yuan", "Liyi Xiao", "Fei Wang"],
    fileName: "1905.02857v1.pdf",
    task: "single-object",
    tags: ["siamese", "cascaded-network", "online-update", "deep-detection"],
    difficulty: "intermediate",
    summary:
      "A cascaded Siamese network splits tracking into two specialists: a matching subnetwork (ResNet plus cross-correlation) proposes candidate locations from every significant peak of its similarity map, and an MDNet-style classification subnetwork selects the best candidate among several peaks and scales. A gate built from the similarity and classification score histories decides whether the classifier may be updated online, so ambiguous frames cannot corrupt it — the result ranks second overall in EAO on VOT-2016, ahead of the challenge winner CCOT.",
    problem:
      "Siamese trackers simply take the argmax of one similarity map, which fails when similar objects or background clutter create multiple peaks — the true target may sit at any of them. Adding a classifier on top immediately raises a second problem: online updates on unreliable tracking results can break the classification subnetwork, and no rule existed for deciding when the current result is trustworthy enough to learn from.",
    background: ["siamese", "appearance-features", "deep-detection", "sot", "occlusion"],
    previousWork: [
      {
        name: "SiamFC-style single-peak Siamese localization",
        limitation:
          "The similarity measurement is disturbed by similar objects and background noise, so multiple peaks appear and the argmax may not be the target (Figure 2).",
        whyThisPaper:
          "All peaks exceeding a ratio γp of the maximum become candidates, which are re-scored by a classification subnetwork; the matching subnetwork itself is built directly on the SiamFC formulation.",
      },
      {
        name: "MDNet (classification subnetwork design)",
        limitation:
          "Accurate but slow: it evaluates every candidate with several forward passes and has no cheap localization prior to lean on.",
        whyThisPaper:
          "Only the handful of candidates proposed by the matching subnetwork are classified, and the paper retrains MDNet on ImageNet so the comparison against it is fair.",
      },
      {
        name: "Online-update trackers with fixed update schedules",
        limitation:
          "Inappropriate updates may break down the classification subnetwork due to ambiguous tracking results — updating on every frame teaches the model its own mistakes.",
        whyThisPaper:
          "A gate compares the current similarity and classification scores with their six-frame histories (γp = 0.75, γm = 0.8, γc = 0.6) before any of the last three layers are fine-tuned.",
      },
    ],
    researchGap:
      "Before this paper, multi-peak similarity maps had no principled way to pick the true target among candidates, and online classifier updates had no principled way to decide whether the current frame was trustworthy enough to learn from.",
    contribution: [
      "Cascaded architecture: a matching subnetwork (ResNet + cross-correlation, offline-pretrained on the ImageNet video object detection dataset) for localization and candidate generation, plus an MDNet-style classification subnetwork (three convolutional layers, two 512-unit fully connected layers, 2-class softmax) for final selection.",
      "Multi-peak, multi-scale candidate generation: peaks whose score exceeds γp = 0.75 of the highest peak become candidate positions, cropped from a patch about four times the target size at three scales 1.02^{−1,0,1}.",
      "Model-update decision method: current similarity and classification scores are compared with their averages over the previous n = 6 frames through γm = 0.8 and γc = 0.6; only then are the last three layers of the classification subnetwork fine-tuned.",
      "Second overall in EAO on VOT-2016, ahead of CCOT (the challenge winner), with competitive success/precision ranking plots on OTB-2013 and OTB-2015.",
    ],
    method: {
      pipeline: ["extract", "correlate", "propose-candidates", "classify", "gate-update"],
      architecture:
        "Matching subnetwork: ResNet backbone followed by a cross-correlation layer, taking a 127×127 exemplar and a 255×255 candidate (the paper's notation: x exemplar, z candidate) centred on the previous position. Classification subnetwork: MDNet-like stack whose convolutional layers follow VGG-M, two fully connected layers with 512 units and a 2-class softmax head trained with cross-entropy.",
      motionModel:
        "None: candidates are generated around the previous position inside a patch of roughly four times the target size, rescaled by three factors 1.02^{−1,0,1}.",
      appearanceModel:
        "Two appearance models: the fixed matching-subnetwork template (never updated) and the classification subnetwork, whose last three layers alone are fine-tuned online when the score-history gate approves.",
      detectionDependency: "None — class-agnostic one-shot tracking with no external detector.",
      trackManagement:
        "Peak gating (γp = 0.75) decides how many candidate locations survive; the highest target-class classification score selects the box; the gate (γm, γc over n = 6 frames) decides whether this frame may update the model.",
      loss:
        "Offline: both subnetworks trained end-to-end with softmax cross-entropy on sample pairs drawn from the ImageNet video object detection dataset, exemplar and candidate from the same video. Online: the same cross-entropy applied to the surviving candidates.",
      optimization:
        "Offline SGD with learning rate from 10⁻³ to 10⁻⁴ and momentum 0.9, initializing from the pretrained networks; online only the last three layers of the classification subnetwork are fine-tuned while the matching subnetwork stays fixed.",
    },
    equations: [
      {
        id: "cascadesiam-match",
        label: "Matching subnetwork similarity map",
        formula: "f(z, x) = ϕ(z) ∗ ϕ(x) + b · 1",
        variables: [
          { symbol: "x", meaning: "exemplar image of 127×127 centred on the previous target position" },
          { symbol: "z", meaning: "candidate image of 255×255 covering the search region" },
          { symbol: "ϕ", meaning: "shared ResNet feature extractor of the matching subnetwork" },
          { symbol: "b · 1", meaning: "bias term broadcast over the response map" },
        ],
        intuition:
          "Cross-correlate the exemplar features with the search region to get a score map whose peaks are possible target locations.",
        why:
          "This map is where ambiguity enters: several peaks may be similar, so the cascade exists precisely to choose among them.",
        where: "Section 3.1 (Matching Subnetwork), Eq. 1; evaluated once per frame.",
        params:
          "The bias b is a single scalar; the top of the map localizes the target only when the peak is unambiguous.",
        paperIds: ["T032"],
      },
      {
        id: "cascadesiam-gate",
        label: "Score-history averages used by the update gate",
        formula: "S̄M = (1/n) Σ_{T=1}^{n} S_M^{t−T},   S̄C = (1/n) Σ_{T=1}^{n} S_C^{t−T}",
        variables: [
          { symbol: "S_M^t, S_C^t", meaning: "similarity and classification scores of the current optimal result at frame t" },
          { symbol: "S̄M, S̄C", meaning: "averages over the previous n frames of those scores" },
          { symbol: "n", meaning: "history length, set to 6" },
        ],
        intuition:
          "Compare how good this frame looks with how good the last six frames looked — a sudden drop means the result is probably wrong.",
        why:
          "Updating on an ambiguous frame corrupts the classifier; the gate turns 'should I learn from this frame?' into an explicit, testable decision.",
        where: "Section 3.3 (Updating Method); unnumbered formula driving the γm and γc thresholds.",
        params: "γp = 0.75 (candidate peak ratio), γm = 0.8, γc = 0.6, n = 6.",
        paperIds: ["T032"],
      },
    ],
    datasets: ["otb", "vot"],
    metrics: ["success-auc", "precision", "eao"],
    baselines: ["SiamFC", "MDNet", "ECO", "CCOT", "Staple", "KCF"],
    results: [
      "VOT-2016: the tracker ranks second overall in the EAO measure, beating CCOT — the winner of the original VOT-2016 challenge — as well as re-detection trackers MLCFT and CACT, while remaining 'less effective than ECO'.",
      "VOT-2015: the paper reports its EAO against the challenge trackers in Figure 5 (top), with SiamFC and MDNet as its own baselines.",
      "OTB-2013 and OTB-2015 (Figure 4): success–precision ranking plots place the approach in the upper-right group among nine state-of-the-art trackers, ahead of SiamFC, Staple, MDNet, DSST and KCF on both axes.",
      "Ablation (Figure 3, OTB-2015): variants that swap or drop components all score below the full model, and only the final implementation employs the update method — 'each component in our tracking framework is helpful to improve performance'.",
      "The paper's qualitative claim: 'surprisingly excellent performance both in terms of accuracy and robustness' across OTB and VOT, with numerical tables deliberately omitted in favour of ranking plots.",
    ],
    ablations: [
      "Component ablation on OTB-2015 (Figure 3): matching-subnetwork and classification-subnetwork variants (legends such as w-M-ResNet and w-C-VGG) and the no-update version all fall short of the full model; only 'Ours' employs the update method.",
      "Update decision itself is the subject of the third contribution: the paper argues updates must be conditional and demonstrates the gate against schedules without it (the no-update variant in Figure 3).",
    ],
    limitations: {
      authorStated: [
        "'Inappropriate updates may break down the classification subnetwork due to the ambiguous tracking results' — the method cannot trust its own output and must gate every update.",
        "On VOT-2016 the tracker is 'less effective than ECO which exploits continuous convolutional filters'.",
      ],
      evident: [
        "The matching subnetwork is frozen during tracking, so all adaptation flows through the last three layers of the classifier — a narrow channel for real appearance change.",
        "Benchmarks are reported only as ranking plots (no numeric tables), so the claimed margins cannot be read off precisely.",
      ],
    },
    assumptions: [
      "A ground-truth box for the first frame is given.",
      "The target stays within a patch of about four times its previous size around the previous position.",
    ],
    computation:
      "One matching forward pass plus classification of the surviving candidates per frame; implemented in MXNet on an Amazon EC2 instance with an Intel Xeon E5 CPU, 61 GB RAM and one NVIDIA K80 GPU with 12 GB VRAM.",
    relations: [
      {
        to: "T009",
        type: "builds-on",
        note: "The matching subnetwork adopts SiamFC's fully-convolutional cross-correlation formulation (Eq. 1) with ResNet features.",
      },
      {
        to: "T003",
        type: "uses-as-baseline",
        note: "MDNet is one of the two explicit baselines on VOT (and the source of the classification subnetwork's design); the paper re-trains it on ImageNet for fairness.",
      },
      {
        to: "T011",
        type: "uses-as-baseline",
        note: "ECO is the tracker the paper concedes is better on VOT-2016 ('less effective than ECO').",
      },
      {
        to: "T001",
        type: "uses-as-baseline",
        note: "KCF appears in the OTB-2013/2015 ranking plots (Figure 4) that the approach outperforms.",
      },
    ],
    concepts: ["siamese", "appearance-features", "deep-detection", "online-vs-offline"],
    impact:
      "The paper's two-tier structure — a cheap matcher proposing candidates and an expensive classifier verifying them — anticipates proposal-plus-verification trackers, and its score-history gate is an early, hand-crafted answer to 'when should a tracker update?', the question DiMP (T031) later answers with a meta-learned predictor.",
  },
  {
    id: "T033",
    arxiv: "1906.01551",
    title: "Learning Rotation Adaptive Correlation Filters in Robust Visual Object Tracking",
    shortTitle: "RACF",
    year: 2019,
    authors: ["Litu Rout", "Priya Mariam Raju", "Deepak Mishra", "Rama Krishna Sai Subrahmanyam Gorthi"],
    fileName: "1906.01551v1.pdf",
    task: "single-object",
    tags: ["correlation-filter", "rotation-adaptive", "illumination", "false-positive-elimination"],
    difficulty: "advanced",
    summary:
      "RACF asks what real video does to a correlation filter — light changes, object rotation, equal-height false peaks and abrupt motion — and adds four mechanisms to the standard DCF formulation: an Illumination Correction front-end (contrast stretching plus unsharp masking), filters learned from deterministically oriented samples with orientation optimized at detection time, a false-positive-elimination cost that rewards the peak closest to the previous centroid, and displacement consistency on the sub-grid motion. Integrated into SRDCF and ECO, the components lift VOT-2016 AEO by 11.4% and 1.7% relative.",
    problem:
      "Correlation filters assume a fixed, axis-aligned, uniformly lit target and that the highest response peak is the target. Real sequences violate all of it: variable illumination corrupts the input features, rotating objects are cut by axis-aligned boxes, similar objects produce equal-height peaks, and abrupt displacement breaks the smoothness the sub-grid detector relies on.",
    background: ["correlation-filter", "appearance-features", "sot", "bounding-box", "occlusion"],
    previousWork: [
      {
        name: "SRDCF (T010)",
        limitation:
          "Spatial regularization suppresses background but nothing in the formulation handles illumination change, rotation, ambiguous peaks or motion jumps.",
        whyThisPaper:
          "RIDF-SRDCF raises VOT-2016 AEO from 0.1981 to 0.2207 (+11.41% relative) and reduces the failure rate from 2.07 to 1.80.",
      },
      {
        name: "ECO (T011)",
        limitation:
          "Strong overall with deep features, but the paper's own OTB study finds it weak on background clutter, deformation and out-of-plane rotation, where it is 'definitely not better than MDNet'.",
        whyThisPaper:
          "RIDF-ECO improves ECO by +1.71% AEO and +6.41% robustness on VOT-2016 and lifts OTB-100 overall success from 0.691 to 0.702.",
      },
      {
        name: "Rotation-adaptive tracking by template search (RAJSSC, SiameseFC-DSR)",
        limitation:
          "Orientation is sought by exhaustive search over joint scale/spatial space or handled in a separate Siamese formulation — costly and outside the DCF.",
        whyThisPaper:
          "Orientation is optimized on a coarse grid inside the standard DCF detection stage (δ = 5°, A = 2, so five candidate orientations) with a single learned model.",
      },
    ],
    researchGap:
      "Before this paper, the standard DCF formulation had no explicit treatment of illumination change, object rotation, false-positive peaks or displacement continuity — each was left to heuristic preprocessing or to whatever the feature extractor happened to tolerate.",
    contribution: [
      "Illumination Correction (IC) filter: contrast stretching of each frame followed by unsharp masking, suppressing low-frequency interference and enhancing high-frequency detail before feature extraction.",
      "Rotation adaptiveness: training samples are rotated by a deterministic orientation θ_k, and at detection the orientation is chosen by maximizing the rotation-normalized response energy over Φ = {θ_k ± aδ} with δ = 5° and A = 2 (five orientations).",
      "False Positive Elimination: the sub-grid detection cost becomes s(u,v)/‖(u − u*, v − v*)‖ — maximize response subject to minimum deviation from the previous centroid, which separates equal-height peaks.",
      "Displacement consistency: the sub-grid location update blends current and previous displacement and heading with ωd = ωa = 0.9 (Eq. 13), damping abrupt speed and angular jumps.",
      "Delivered as a generic framework and instantiated in two CF trackers: RDF-/RIDF-SRDCF and RDF-/RIDF-ECO, improving AEO by 11.4% and 1.71% and robustness by 14.7% and 6.41% over the respective baselines.",
    ],
    method: {
      pipeline: ["enhance-illumination", "orient-samples", "correlate", "score", "eliminate-false-positives", "smooth-displacement"],
      architecture:
        "The full SRDCF (or ECO) pipeline with four insertions: a frame-level IC front-end; rotation of training samples before filter learning in the Fourier domain; a detection stage that first searches orientation, then optimizes the sub-grid location with the FPE cost; and a displacement-consistency update of the winning coordinate.",
      motionModel:
        "Displacement consistency on the sub-grid: the new location is the old one plus an exponentially blended displacement and heading (ωd = ωa = 0.9), which restricts abrupt transitions while leaving the filter's own Newton sub-grid optimization in charge.",
      appearanceModel:
        "Multi-channel correlation filters over hand-crafted or deep features with the SRDCF spatial (Tikhonov) regularizer, learned from oriented training samples — a single model, no bank of per-orientation filters.",
      detectionDependency: "None — conventional single-object correlation-filter tracking.",
      trackManagement:
        "No explicit track bookkeeping; FPE acts as a soft persistence prior (the target is expected near its previous centroid), and orientation continuity is enforced by the small orientation step δ = 5°.",
      loss:
        "ε(f) = Σ_k α_k‖S_f(x_k) − y_k‖² + Σ_l ‖(w/(MN))·f_l‖² (Eq. 2), minimized in the Fourier domain via Parseval's theorem with the per-location regularizer w as in SRDCF.",
      optimization:
        "Filter fitting uses the base tracker's Fourier-domain solver; detection adds a coarse orientation search (five orientations), Newton sub-grid optimization of the FPE cost, and Gauss-Seidel optimization of the FPE energy.",
    },
    equations: [
      {
        id: "racf-filter-learn",
        label: "Spatially regularized filter objective",
        formula: "ε(f) = Σ_k α_k ‖ S_f(x_k) − y_k ‖² + Σ_l ‖ (w / (M·N)) · f_l ‖²",
        variables: [
          { symbol: "S_f(x_k) = Σ_l x_l ∗ f_l", meaning: "multi-channel circular-convolution response on training sample k" },
          { symbol: "y_k", meaning: "scalar label map (typically a Gaussian) for sample k" },
          { symbol: "α_k", meaning: "per-sample weight" },
          { symbol: "w", meaning: "spatial Tikhonov regularizer emphasizing the target region over the background" },
          { symbol: "f_l", meaning: "layer l of the d-channel filter, on an M×N grid" },
        ],
        intuition:
          "Fit the filter to reproduce each label while forcing its coefficients to be small away from the target centre.",
        why:
          "It is the standard SRDCF objective that RACF extends — every contribution (rotation, FPE, displacement) modifies how samples are oriented or how detection is done on top of this fit.",
        where: "Section 4.1 (SRDCF Training and Detection), Eq. 2; solved in the Fourier domain (Eq. 3).",
        params: "The regularizer w (learned per location) controls how much background is suppressed; α_k weights samples.",
        paperIds: ["T033"],
      },
      {
        id: "racf-response",
        label: "Continuous response interpolation",
        formula: "s(u, v) = (1/(M·N)) Σ_{m=0}^{M−1} Σ_{n=0}^{N−1} ŝ(m, n) · e^{ i·2π·( m·u/M + n·v/N ) }",
        variables: [
          { symbol: "ŝ(m, n)", meaning: "DFT of the convolution response on the test sample" },
          { symbol: "(u, v)", meaning: "continuous sub-grid location in [0, M) × [0, N)" },
        ],
        intuition:
          "Reconstruct the response at any fractional pixel from its Fourier coefficients, so localization is not limited to the grid.",
        why:
          "Sub-grid localization is where both false-positive elimination and displacement consistency operate; without it there is no (u*, v*) to smooth or to ratio-normalize.",
        where: "Section 4.1, Eq. 4; evaluated during detection before Newton optimization.",
        params: "M×N is the search window resolution; the maximum of s is refined from the best grid point (u(0), v(0)).",
        paperIds: ["T033"],
      },
      {
        id: "racf-fpe",
        label: "False-positive-elimination detection cost",
        formula: "max_{(u,v)}  s(u, v) / ‖ (u − u*, v − v*) ‖",
        variables: [
          { symbol: "s(u, v)", meaning: "interpolated response at the candidate location" },
          { symbol: "(u*, v*)", meaning: "previous target centroid" },
        ],
        intuition:
          "Among several equally strong peaks, prefer the one closest to where the target just was — a distractor across the image must offer a much higher response to win.",
        why:
          "Equal-height peaks (gloves, leaves, similar objects) make argmax localization ambiguous; dividing by distance resolves the tie using motion continuity instead of appearance alone.",
        where: "Section 4.4 (Fast Sub-grid Detection); the cost optimized jointly with orientation.",
        params: "Unnumbered in the paper; used inside the Gauss-Seidel optimization of the detection energy.",
        paperIds: ["T033"],
      },
      {
        id: "racf-displacement",
        label: "Displacement-consistent sub-grid update",
        formula: "u*_{k+1} = (u*_k, v*_k) + d₁ₙ ∠ϕ₁ₙ ,  d₁ₙ = ωd·d₁ + (1−ωd)·d₀ ,  ϕ₁ₙ = ωa·ϕ₁ + (1−ωa)·ϕ₀",
        variables: [
          { symbol: "d₀, d₁", meaning: "previous and current displacement magnitudes of the sub-grid solution" },
          { symbol: "ϕ₀, ϕ₁", meaning: "previous and current displacement headings (arctangent of the displacement vector)" },
          { symbol: "ωd, ωa", meaning: "exponential moving-average weights, both set to 0.9" },
        ],
        intuition:
          "Update the position with a smoothed step and direction so the tracker cannot teleport between frames.",
        why:
          "Abrupt speed and heading changes are a main cause of tracking jumps; smoothing the sub-grid solution adds motion-level robustness without touching the filter.",
        where: "Section 5 (Displacement Consistency), Eq. 13, applied to the sub-grid location from Eq. 12.",
        params: "With ωd = ωa = 1 the update reduces to the raw optimal solution of Eq. 12; at 0.9 it is slightly damped.",
        paperIds: ["T033"],
      },
    ],
    datasets: ["vot", "otb"],
    metrics: ["eao", "success-auc", "precision"],
    baselines: ["SRDCF", "ECO", "MDNet", "CCOT", "KCF"],
    results: [
      "VOT-2016 development ablation (Table 1, 16 videos): AEO ECO 0.357 → D-ECO 0.360 → DF-ECO 0.362 → R-ECO 0.383 → RF-ECO 0.386 → RD-ECO 0.395 → RDF-ECO 0.402 → RIDF-ECO 0.433, i.e. +21.3% over the baseline.",
      "VOT-2016 full benchmark (Table 2): SRDCF 0.1981 → I-SRDCF 0.2051 (+3.53%) → RDF-SRDCF 0.2191 (+10.60%) → RIDF-SRDCF 0.2207 (+11.41%), with failure rate 2.07 → 1.80; ECO 0.3563 → RIDF-ECO 0.3624 (+1.71% AEO, +6.41% robustness), ahead of TCNN 0.3249, CCOT 0.3310 and MDNet 0.3584.",
      "Abstract claim: 14.7% and 6.41% improvement in robustness and 11.4% and 1.71% in Average Expected Overlap over the SRDCF and ECO baselines.",
      "OTB-100 (Table 3): RIDF-ECO overall success 0.702 and precision 0.937 versus ECO 0.691 / 0.910 and MDNet 0.678 / 0.909; largest gains over ECO on low resolution (0.734 vs 0.652), illumination variation (0.702 vs 0.662) and out-of-view (0.767 vs 0.726).",
      "OTB-100 (Table 3): RIDF-SRDCF overall success 0.641 versus SRDCF 0.598 and DeepSRDCF 0.635 — the paper's evidence that its contributions beat simply using deep features.",
    ],
    ablations: [
      "Component ladder (Table 1): each mechanism contributes when combined — displacement (D), false-positive elimination (F), rotation (R) and illumination (I) — from ECO's 0.357 AEO to RIDF-ECO's 0.433 (+21.3%).",
      "Baseline dependence (Table 2): the same contributions give +11.41% on SRDCF but only +1.71% on ECO, showing ECO already absorbs much of what they offer.",
      "Per-category OTB study: the contributions strengthen ECO on occlusion (0.710 → 0.721), illumination variation (0.662 → 0.702) and low resolution (0.652 → 0.734), while MDNet still wins background clutter (0.697), deformation (0.722) and out-of-plane rotation (0.707).",
      "IC front-end alone: the paper reports that 'the performance of the baseline trackers improves by a considerable amount just by enhancing the input images'.",
    ],
    limitations: {
      authorStated: [
        "'Our method finds difficulties in dealing with Background Clutter, Deformation and Out-of-plane rotation… it is definitely not better than MDNet' on those OTB categories.",
        "The framework is demonstrated only by integrating with SRDCF and ECO, though the authors argue it is generic and can be combined with other CF trackers.",
        "DeepSRDCF lags RIDF-SRDCF on OTB-100 success and precision — reported as a win for the method, but it also shows the gains are tied to the CF formulation rather than to features.",
      ],
      evident: [
        "Detection now costs an orientation search (five orientations) plus Newton/Gauss-Seidel sub-grid optimization on every frame, on top of the otherwise Fourier-cheap DCF pipeline.",
        "The IC front-end is a fixed enhancement pipeline (contrast stretch plus unsharp mask) with no learning, so its benefit depends on the sequence's input statistics.",
      ],
    },
    assumptions: [
      "Object orientation changes little between consecutive frames (δ = 5° with A = 2 gives five candidate orientations per frame).",
      "The target stays close enough to its previous centroid that distance from it is a useful prior.",
    ],
    computation:
      "Correlation filtering stays in the Fourier domain as in SRDCF/ECO; detection adds a coarse search over 2A+1 = 5 orientations plus sub-grid Newton optimization with the FPE cost; the paper reports no frame rate.",
    relations: [
      {
        to: "T010",
        type: "extends",
        note: "Builds directly on SRDCF's spatially regularized formulation ('identical notations as in SRDCF'), delivering RDF-SRDCF and RIDF-SRDCF.",
      },
      {
        to: "T011",
        type: "extends",
        note: "Inserts the same four mechanisms into ECO, producing RDF-ECO and RIDF-ECO (AEO 0.357 → 0.433).",
      },
      {
        to: "T003",
        type: "uses-as-baseline",
        note: "MDNet is compared on VOT-2016 (AEO 0.3584 versus RIDF-ECO 0.3624) and is the tracker the authors beat on the categories where ECO fails.",
      },
      {
        to: "T001",
        type: "uses-as-baseline",
        note: "KCF appears in the OTB-100 comparison table (overall success 0.477 versus RIDF-ECO 0.702).",
      },
    ],
    concepts: ["correlation-filter", "appearance-features", "sot", "motion-model", "occlusion"],
    impact:
      "RACF is one of the last major attempts to make correlation filters invariant to the physics of the camera and scene — illumination, rotation, jitter — rather than only to their features; its four-part recipe quantifies how much was still left on the table inside DCF trackers, and its per-category diagnosis (background clutter, deformation, out-of-plane rotation) is the standard statement of what correlation filters still could not do as deep trackers took over.",
  },
  {
    id: "T034",
    arxiv: "1909.12605",
    title: "Towards Real-Time Multi-Object Tracking",
    shortTitle: "JDE",
    year: 2019,
    authors: ["Zhongdao Wang", "Liang Zheng", "Yixuan Liu", "Yali Li", "Shengjin Wang"],
    fileName: "1909.12605v2.pdf",
    task: "multi-object",
    tags: ["end-to-end-mot", "joint-detection-embedding", "reid", "single-stage", "real-time"],
    difficulty: "intermediate",
    summary:
      "JDE folds the appearance embedding into a single-shot detector so one network emits bounding boxes and their re-ID features in the same forward pass, then associates them with a lightweight Kalman-plus-Hungarian scheme whose cost fuses cosine appearance and Mahalanobis motion. It is presented as the first (near) real-time MOT system — 22 to 40 FPS depending on resolution — at 64.4 MOTA on the MOT-16 test set, within 1.7 points of far slower separate-detection-and-embedding pipelines.",
    problem:
      "Tracking-by-detection spends its time on two heavy networks: a detector re-run on every frame and an embedding model re-run on re-sampled crops, whose runtimes roughly add up and grow with the number of targets. Two-stage joint designs save some computation but still run below 10 FPS because their second stage also scales with target count — so a competitive-accuracy MOT system at real-time speed did not exist.",
    background: ["mot", "tracking-by-detection", "deep-detection", "reid", "end-to-end-mot", "appearance-features"],
    previousWork: [
      {
        name: "Separate Detection and Embedding (SDE) systems such as Faster R-CNN + QAN",
        limitation:
          "Two compute-intensive networks: <6 FPS at 66.1 MOTA on MOT-16, with runtime growing with target count; their training data is detection-heavy but ID-poor (429K boxes with only 1.2K–16K identities in Table 4).",
        whyThisPaper:
          "JDE trains on 270K boxes with 8.7K identities and reaches 64.4 MOTA at 20.2 FPS with a single network.",
      },
      {
        name: "Two-stage joint detector-embedding (Faster R-CNN with an embedding stage)",
        limitation:
          "The Fast R-CNN stage must re-compute features per proposal and usually runs at fewer than 10 FPS; its runtime also grows with the number of targets.",
        whyThisPaper:
          "A single-shot head predicts detections and a dense embedding map together, eliminating the second stage entirely.",
      },
      {
        name: "Triplet-loss appearance embeddings (earlier MOT works)",
        limitation:
          "Unstable training and slow convergence, with weak discriminability: 42.2 TPR@FAR=0.1 and 59.5 MOTA on the validation set.",
        whyThisPaper:
          "Replacing it with a class-wise cross-entropy over learned proxies raises TPR to 88.2 and MOTA to 64.3 (90.4 / 65.8 with uncertainty weighting).",
      },
    ],
    researchGap:
      "Before this paper, no real-time MOT system learned detection and appearance embeddings jointly inside a single-shot network, so high accuracy at real-time speed was unattainable.",
    contribution: [
      "JDE: a single-shot detector (DarkNet-53, 1088×608 input) that outputs bounding boxes and a dense embedding map at multiple scales in one pass; embeddings with label −1 (foreground boxes without identity annotations) are ignored in the embedding loss.",
      "Embedding objective: class-wise cross-entropy over learned proxies (L_CE) rather than the triplet loss or its smooth upper bound, lifting TPR@FAR=0.1 from 42.2 to 88.2 and validation MOTA from 59.5 to 64.3.",
      "Automatic loss balancing: the multi-head objective L_total = Σ_i Σ_j w_j^i L_j^i with learnable task-dependent uncertainties s_j^i, replacing weight search — uniform weights collapse to 36.9 MOTA.",
      "Fast association: cosine appearance and Mahalanobis motion affinities fused as C = λA_e + (1−λ)A_m, solved once per frame by Hungarian assignment with a Kalman motion state and EMA-smoothed embeddings (α = 0.9).",
      "Joint training set and first (near) real-time system: 54K images / 270K boxes / 8.7K identities across six datasets, 22–40 FPS at 64.4 MOTA on MOT-16.",
    ],
    method: {
      pipeline: ["detect", "embed", "associate", "update", "terminate"],
      architecture:
        "DarkNet-53 backbone with heads for foreground/background classification, box regression and a dense embedding map at three scales, input resolution 1088×608; embeddings from all scales pass through a shared fully-connected layer so identities are consistent across scales.",
      motionModel:
        "Kalman filter maintaining a per-tracklet motion state (including velocity along x), updated for matched tracklets and consumed only through the Mahalanobis motion affinity.",
      appearanceModel:
        "Dense embeddings predicted by the detector, trained against class-wise proxy weights; per-track appearance is an exponential moving average of matched embeddings, compared to detections by cosine similarity.",
      detectionDependency:
        "Intrinsic — JDE is itself the detector, so association operates only on its own output; this is the paper's point (one network, one forward pass).",
      trackManagement:
        "A tracklet is initialized when an unassigned detection appears in 2 consecutive frames, terminated after 30 frames without update, and briefly lost tracklets are handled through a buffer pool; assignment runs once per frame.",
      loss:
        "L_total = Σ_i Σ_{j∈{α,β,γ}} w_j^i L_j^i over classification, box regression and embedding cross-entropy heads, with weights learned as task-dependent uncertainties s_j^i; embeddings labelled −1 are excluded from the embedding loss.",
      optimization:
        "Standard SGD for 30 epochs with learning rate 1e-2 decayed by 0.1 at epochs 15 and 23; random rotation, scale and color-jitter augmentation; input 1088×608 (JDE1088) or 864 (JDE864).",
    },
    equations: [
      {
        id: "jde-affinity",
        label: "Fused affinity cost for assignment",
        formula: "C = λ · A_e + (1 − λ) · A_m",
        variables: [
          { symbol: "A_e", meaning: "appearance affinity matrix computed by cosine similarity between tracklets and observations" },
          { symbol: "A_m", meaning: "motion affinity matrix computed by Mahalanobis distance against the Kalman state" },
          { symbol: "λ", meaning: "trade-off between appearance and motion evidence" },
        ],
        intuition:
          "Score every tracklet–detection pair as a blend of 'does it look right?' and 'does it move right?', then hand the matrix to the Hungarian algorithm.",
        why:
          "The joint model only pays off if association stays cheap; one weighted cost keeps the association step simple enough to run in milliseconds.",
        where: "Section 3.6 (Online Association); the cost matrix solved by Hungarian assignment each frame.",
        params: "λ controls the appearance/motion balance; the paper notes its scheme is simpler than SORT's cascade matching (single assignment plus a buffer pool).",
        paperIds: ["T034"],
      },
      {
        id: "jde-embed-loss",
        label: "Class-wise cross-entropy embedding loss",
        formula: "L_CE = − log[ exp(fᵀg⁺) / ( exp(fᵀg⁺) + Σ_i exp(fᵀg_i⁻) ) ]",
        variables: [
          { symbol: "f", meaning: "embedding extracted from the dense embedding map for the anchor box" },
          { symbol: "g⁺", meaning: "learnable class-wise weight of the positive class (the anchor's identity)" },
          { symbol: "g_i⁻", meaning: "learnable weights of all negative classes" },
        ],
        intuition:
          "Treat identity as classification: pull the embedding toward its identity's learned proxy and away from every other proxy, not just sampled hard negatives.",
        why:
          "The triplet loss only repels sampled negatives and trains unstably; using all classes as negatives at once gives far more discriminative embeddings (TPR 42.2 → 88.2).",
        where: "Section 3.4, Eq. 4; applied to embeddings of positive samples at all scales.",
        params: "Embeddings with label −1 (no identity annotation) are ignored; embeddings from different scales share the same classifier, so identities are comparable across scales.",
        paperIds: ["T034"],
      },
      {
        id: "jde-total-loss",
        label: "Multi-head joint objective",
        formula: "L_total = Σ_{i=1}^{M} Σ_{j∈{α,β,γ}} w_j^i · L_j^i",
        variables: [
          { symbol: "M", meaning: "number of prediction heads (one per scale)" },
          { symbol: "L_α, L_β, L_γ", meaning: "classification, box regression and embedding losses of head i" },
          { symbol: "w_j^i", meaning: "loss weight per head and task, learned through task-dependent uncertainty" },
        ],
        intuition:
          "One objective for detection and embedding at every scale, with the weights between them decided during training rather than by grid search.",
        why:
          "Joint training only works if the tasks are balanced: the paper shows uniform weighting collapses to 36.9 MOTA while uncertainty weighting reaches 65.8.",
        where: "Section 3.5 (Automatic Loss Balancing), Eq. 5; the objective minimized during offline training.",
        params: "The uncertainties s_j^i are learnable parameters (Eq. 6 of the paper); searching weights by hand is explicitly criticized as far from optimal.",
        paperIds: ["T034"],
      },
      {
        id: "jde-ema",
        label: "Exponential moving average of track appearance",
        formula: "e_i^t = α · e_i^{t−1} + (1 − α) · f_i^t",
        variables: [
          { symbol: "e_i^t", meaning: "appearance state of tracklet i after observing frame t" },
          { symbol: "f_i^t", meaning: "embedding of the current matched observation" },
          { symbol: "α", meaning: "momentum, set to 0.9" },
        ],
        intuition:
          "Smooth each track's appearance over time so a single corrupted detection does not redefine its identity.",
        why:
          "Cosine matching against a stable history is more reliable than against the last frame alone, and it costs one line of arithmetic.",
        where: "Section 3.6, Eq. 7; updated for every matched tracklet after assignment.",
        params: "α = 0.9 means roughly the last ten observations dominate; α = 0 would mean identity equals the latest embedding.",
        paperIds: ["T034"],
      },
    ],
    datasets: ["mot15", "mot16", "others"],
    metrics: ["mota", "idf1", "idsw", "mt-ml", "fp", "fn", "fps"],
    baselines: ["SORT", "DeepSORT", "POI", "CNNMTT", "TAP", "MOTDT"],
    results: [
      "MOT-16 test, private protocol (Table 4): JDE1088 MOTA 64.4, IDF1 55.8, MT 35.4, ML 20.0, ID switches 1,544, 24.5 FPS detection + 236.5 FPS association = 22.2 FPS overall; JDE864 62.1 / 56.9 / 1,608 IDs at 30.3 FPS.",
      "Headline comparison: 20.2 FPS with MOTA 64.4 on MOT-16 versus Faster R-CNN + QAN at <6 FPS with MOTA 66.1; the abstract claims the first (near) real-time MOT system at 22–40 FPS depending on input resolution.",
      "Against SDE systems on the same benchmark: POI 66.1 / 65.1, CNNMTT 65.2 / 62.2, TAP 64.8 / 73.5, DeepSORT 61.4 / 62.2, RAR16 63.0 / 63.8 (MOTA / IDF1).",
      "Association ablation (Table 1, same JDE model): low density (MOT-15) ours 46.2 FPS / 67.5 MOTA / 67.6 IDF1 versus SORT 44.1 / 66.9 / 55.8; high density (CVPR19-01, density 61.1) ours 33.9 / 35.4 / 35.5 versus SORT 26.4 / 35.0 / 32.4.",
      "Embedding losses (Table 3): triplet 42.2 TPR / 59.5 MOTA / 375 IDs, smooth upper bound 44.3 / 59.8 / 346, cross-entropy 88.2 / 64.3 / 223, cross-entropy + uncertainty weighting 90.4 / 65.8 / 207; uniform weighting collapses to 6.8 AP and 36.9 MOTA.",
      "Low-density quality/speed: JDE-DN53 reaches 66.2 IDF1 at 22 FPS versus Cascade R-CNN ResNet-101 + PCB at 69.6 IDF1 and 7.6 FPS (MOTA 65.8 versus 66.2 for a ~6 FPS combination).",
    ],
    ablations: [
      "Embedding loss (Table 3): the loss choice dominates — cross-entropy over proxies beats the triplet loss and its upper bound by about 44 TPR points and lifts MOT validation MOTA by roughly 5.",
      "Loss weighting (Table 3): uniform 36.9, MGDA-UB 38.3, loss normalization 57.9, uncertainty weighting 65.8 MOTA — automatic balancing is not optional for joint training.",
      "Input resolution (Table 4): JDE864 versus JDE1088 trades 2.3 MOTA for about 8 FPS.",
      "Association strategy (Table 1): the single-assignment scheme beats SORT's cascade matching in both density regimes while running faster.",
    ],
    limitations: {
      authorStated: [
        "'JDE has a lower IDF1 score and more ID switches than existing methods' — the authors trace this to inaccurate detection when pedestrians overlap heavily, not to the joint embedding (swapping in a separately trained embedding changes nothing).",
        "In high-density crowds IDF1 degrades slightly more than in SDE combinations, because single-stage detections drift under overlap and box misalignment makes the embedding ambiguous.",
        "Future work is stated explicitly: more accurate box prediction under significant pedestrian overlap, and the time–accuracy trade-off.",
      ],
      evident: [
        "Association is deliberately minimal (one Hungarian assignment, buffer pool, no cascade matching), and it shows: IDF1 55.8 versus TAP's 73.5 on MOT-16.",
        "Identity supervision only exists where labels are present — embeddings from boxes with label −1 contribute no gradient, so crowded frames with missing IDs teach the embedding nothing.",
      ],
    },
    assumptions: [
      "All training identities are known, so embedding learning is a classification problem over the training set's identities.",
      "Per-track motion is adequately described by the Kalman filter's state (including constant velocity along x).",
    ],
    computation:
      "20.2 FPS end-to-end on MOT-16 test (24.5 FPS detection + 236.5 FPS association) with DarkNet-53 at 1088×608, 30.3 FPS at 864; abstract claims 22–40 FPS depending on input resolution; SGD for 30 epochs offline.",
    relations: [
      {
        to: "T005",
        type: "uses-as-baseline",
        note: "Table 1 compares the paper's association against SORT on the same JDE detections (low and high density), beating it in MOTA, IDF1 and FPS.",
      },
      {
        to: "T013",
        type: "uses-as-baseline",
        note: "DeepSORT is one of the SDE systems in the MOT-16 comparison (61.4 MOTA / 62.2 IDF1 versus JDE's 64.4 / 55.8).",
      },
      {
        to: "T006",
        type: "uses-as-baseline",
        note: "Evaluates on the MOT-16 benchmark under the private-data protocol, reporting 64.4 MOTA at 20.2 FPS.",
      },
    ],
    concepts: ["mot", "tracking-by-detection", "end-to-end-mot", "reid", "appearance-features", "learned-association"],
    impact:
      "JDE established joint detection-and-embedding as the efficiency blueprint for real-time MOT: FairMOT (T043) inherits its dense-embedding single-shot layout and adds a fairness analysis of the two heads, while its diagnosis — that ID switches come from boxes, not from the embedding — shaped how later one-stage trackers allocate capacity between detection and association.",
  },
  {
    id: "T035",
    arxiv: "1911.06188",
    title: "SiamFC++: Towards Robust and Accurate Visual Tracking with Target Estimation Guidelines",
    shortTitle: "SiamFC++",
    year: 2019,
    authors: ["Yinda Xu", "Zeyu Wang", "Zuoxin Li", "Ye Yuan", "Gang Yu"],
    fileName: "1911.06188v4.pdf",
    venue: "AAAI 2020",
    task: "single-object",
    tags: ["siamese", "anchor-free", "target-estimation", "quality-assessment", "real-time"],
    difficulty: "intermediate",
    summary:
      "SiamFC++ turns tracker design into four explicit guidelines for target state estimation — separate classification from state estimation, score without anchor ambiguity, keep the tracker free of scale/ratio prior knowledge, and assess estimation quality independently — then builds a tracker that follows all four. The result is an anchor-free SiamFC-style network with a per-pixel regression head and a Prior Spatial Score quality branch that reaches state of the art on OTB2015, VOT2018, LaSOT, GOT-10k and TrackingNet (a previously unseen 75.4 AUC on TrackingNet) while running at 90 to 160 FPS.",
    problem:
      "There were three incompatible ways to estimate the target box: brute-force multi-scale search (SiamFC and the correlation filters), iterative refinement from random initializations (ATOM), and anchor-based RPN regression (the SiamRPN family). Each brings its own cost — inaccurate scale selection, many tuned hyperparameters, or a score that measures anchor-to-object similarity plus a hand-tuned prior on scale and aspect ratio — and none of the designs were derived from what makes tracking different from detection, so detector components that help detection end up hurting tracking robustness.",
    background: ["sot", "siamese", "bounding-box", "iou", "anchor-free-head", "success-plot"],
    previousWork: [
      {
        name: "SiamFC and correlation filters with brute-force multi-scale test",
        limitation:
          "Scale is chosen by rescaling the search patch into several scales and taking the highest-scoring one, which is inaccurate (it ignores object pose), expensive (a mini-batch of forward passes), and built on an assumption that scale changes at a fixed rate between frames.",
        whyThisPaper:
          "A regression head predicts the four side distances at every feature pixel in one pass; the ablation shows this branch is the single largest gain (+0.094 EAO on VOT2018).",
      },
      {
        name: "ATOM",
        limitation:
          "Iterative refinement needs many random box initializations and backpropagation steps plus extra hyperparameters (number and distribution of initial boxes), which slows the tracker considerably.",
        whyThisPaper:
          "SiamFC++ reaches EAO 0.400 on VOT2018 with a one-shot anchor-free head at over 100 FPS, versus ATOM's 0.401 at 30 FPS.",
      },
      {
        name: "SiamRPN++ and the SiamRPN family",
        limitation:
          "Anchors make the classification score a similarity between anchors and objects rather than between the template and the image (violates G2), require prior knowledge of scale/ratio distribution (G3), and box selection uses classification confidence instead of an estimation-quality score (G4) — the paper measures SiamRPN++'s SR0.5 and SR0.75 on GOT-10k as 7.7 and 15.4 points below SiamFC++'s.",
        whyThisPaper:
          "Direct template-to-image matching gives one unambiguous score per pixel, and a dedicated quality branch multiplies the classification score before box selection.",
      },
    ],
    researchGap:
      "Before this paper, no tracker derived its target-state-estimation design from the specific characteristics of the tracking problem, so anchor ambiguity and brute-force multi-scale search persisted even in state-of-the-art trackers.",
    contribution: [
      "Four design guidelines: G1 decompose tracking into classification and state estimation, G2 make the classification score represent target existence in the sub-window of the corresponding pixel (no anchor matching), G3 remove prior knowledge such as scale/ratio distributions, G4 select boxes with an estimation-quality score independent of classification.",
      "SiamFC++: siamese backbone followed by task-specific ψ_cls and ψ_reg layers over cross-correlated features, predicting per-pixel foreground/background and a 4-sided distance box at total stride 8 — anchor-free, so exactly one prediction per location.",
      "Prior Spatial Score (PSS): a 1×1 quality branch predicts PSS* = sqrt( min(l*,r*)/max(l*,r*) * min(t*,b*)/max(t*,b*) ), and the final box score is classification score × PSS; predicting IoU instead of PSS is evaluated as a variant (AO 78.0 versus 77.8 on GOT-10k val, PSS chosen for stability).",
      "Ablation hierarchy: the regression branch contributes the largest single gain (+0.094 EAO), then training-data diversity (+0.063 and +0.010), a stronger backbone (+0.026) and deeper head structure (+0.020) — the same ingredients SiamRPN++ adds over SiamFC, but with far better robustness.",
      "State of the art on five benchmarks: VOT2018 EAO 0.426 (GoogLeNet) and 0.400 at over 100 FPS (AlexNet), TrackingNet AUC 75.4, GOT-10k AO 59.5, LaSOT success 54.4, at 90/160 FPS.",
    ],
    method: {
      pipeline: ["extract", "cross-correlate", "classify", "regress", "score-quality", "penalize", "select"],
      architecture:
        "Shared ImageNet-pretrained siamese backbone (modified AlexNet or GoogLeNet, total stride 8) embeds a 127×127 template and a 303×303 search image; after cross-correlation, two task-specific stacks of conv3×3 layers (ψ_cls, ψ_reg) emit per-pixel 2-class scores and 4-distance boxes, with a parallel 1×1 convolution producing the quality (PSS) score.",
      motionModel:
        "No filter. Test-phase post-processing follows the SiamRPN family: scores are multiplied by a penalty p = e^(k·max(r/r₀,r₀/r)·max(s/s₀,s₀/s)) for box ratio and size change, damped by a cosine window with influence coefficient Ω, the argmax pixel x* is taken, and the target size is updated by confidence-weighted linear interpolation B_pred.size = (1−α₀)·B_prev.size + α₀·B_curr.size with α₀ = s[x*]·α.",
      appearanceModel:
        "Purely offline: the template is the first-frame crop and stays fixed for the whole sequence; there is no online filter or model update.",
      detectionDependency: "None — an anchor-free, class-agnostic single-object tracker with no external detector.",
      loss:
        "L = (1/N_pos)·Σ L_cls + (λ/N_pos)·Σ 1{c*>0}·L_quality + (λ/N_pos)·Σ 1{c*>0}·L_reg, where L_cls is focal loss over foreground/background, L_quality is binary cross entropy on PSS and L_reg is IoU loss on the 4-sided box; a location (x, y) is positive when the center of its corresponding image patch falls inside the ground-truth box.",
      optimization:
        "AlexNet version: conv1–conv3 frozen, conv4–conv5 fine-tuned, 5 warm-up epochs with the learning rate raised linearly from 1e-7 to 2e-3, then cosine annealing for 45 epochs, 600k image pairs per epoch, SGD with momentum 0.9. GoogLeNet version: stage 1–2 frozen, base learning rate 2e-2 with the backbone multiplied by 0.1, 300k pairs per epoch for 20 epochs, backbone unfrozen at epoch 10; the LaSOT protocol-II model freezes the backbone and trains on 150k pairs per epoch.",
    },
    equations: [
      {
        id: "siamfcpp-match",
        label: "Task-specific siamese matching",
        formula: "f_i(z, x) = ψ_i( φ(z) ) ⋆ ψ_i( φ(x) ),  i ∈ {cls, reg}",
        variables: [
          { symbol: "φ", meaning: "shared siamese backbone that embeds template z and search image x into a common feature space" },
          { symbol: "ψ_i", meaning: "task-specific layer for sub-task i: classification (cls) or regression (reg)" },
          { symbol: "⋆", meaning: "cross-correlation between the template and search feature maps" },
        ],
        intuition:
          "Correlate the template with the search image once per task, so the classifier and the box regressor each get their own response map.",
        why:
          "Separating the two tasks after a shared embedding is guideline G1 — one branch answers 'is it there?', the other 'where are its sides?', instead of one score doing both jobs.",
        where: "Section 3, Eq. 1; the input stage of every forward pass.",
        params: "Template 127×127, search image 303×303, backbone total stride s = 8.",
        paperIds: ["T035"],
      },
      {
        id: "siamfcpp-pss",
        label: "Prior Spatial Score (quality branch)",
        formula: "PSS* = sqrt( min(l*, r*)/max(l*, r*) × min(t*, b*)/max(t*, b*) )",
        variables: [
          { symbol: "l*, t*, r*, b*", meaning: "ground-truth distances from the matched location to the four sides of the target box" },
          { symbol: "PSS*", meaning: "target quality label in [0,1]: how centered the location is inside the box" },
        ],
        intuition:
          "A location near the box center has near-equal left/right and top/bottom distances, so its PSS approaches 1; a location near an edge gets a low score.",
        why:
          "Classification confidence is poorly correlated with localization accuracy (guideline G4), so a separate score that rewards well-centered boxes is needed before picking the final box.",
        where: "Section 3, Eq. 3; produced by a 1×1 convolution in parallel with the classification head and multiplied into the score at inference.",
        params: "An IoU-prediction variant (Eq. 4) scores 78.0 AO versus PSS's 77.8 on GOT-10k val; PSS is chosen for its observed stability across datasets.",
        paperIds: ["T035"],
      },
      {
        id: "siamfcpp-loss",
        label: "Three-term training objective",
        formula: "L = (1/N_pos) Σ L_cls + (λ/N_pos) Σ 1{c*>0} L_quality + (λ/N_pos) Σ 1{c*>0} L_reg",
        variables: [
          { symbol: "L_cls", meaning: "focal loss on the per-pixel foreground/background classification score p" },
          { symbol: "L_quality", meaning: "binary cross entropy between the predicted quality q and the PSS target q*" },
          { symbol: "L_reg", meaning: "IoU loss between the predicted 4-sided box t and the target t*" },
          { symbol: "λ", meaning: "weight applied to the positive-only quality and regression terms" },
        ],
        intuition:
          "Train the three heads together: classify every pixel, but only ask the quality and box heads to learn at pixels that actually cover the object.",
        why:
          "The guidelines only pay off if quality and localization are learned explicitly rather than piggybacking on the classification score.",
        where: "Section 3, Training Objective, Eq. 5; minimized during offline training.",
        params: "Positive label c* = 1 when the patch center falls inside the ground-truth box; the term 1{c*>0} masks negative pixels out of quality and regression supervision.",
        paperIds: ["T035"],
      },
      {
        id: "siamfcpp-penalty",
        label: "Score penalty for box size and ratio change",
        formula: "p = e^( k · max(r/r₀, r₀/r) · max(s/s₀, s₀/s) )",
        variables: [
          { symbol: "r, r₀", meaning: "aspect ratios of the proposed box and of the previous prediction" },
          { symbol: "s, s₀", meaning: "sizes of the proposed box and of the previous prediction" },
          { symbol: "k", meaning: "hyperparameter controlling the magnitude of the penalization" },
        ],
        intuition:
          "Shrink the score of boxes whose shape or size changes a lot relative to the last frame, since real targets usually change little in one step.",
        why:
          "The tracker has no motion model, so this hand-written prior plus a cosine window is what keeps the per-pixel argmax temporally coherent.",
        where: "Appendix B (Test Phase Behavior), Eq. 6; applied to the score map before the argmax.",
        params: "Combined with a cosine window ω(x) = 0.5 − 0.5·cos(2π·dist(x,x_c)/((N−1)/2 − 1)) of influence Ω, then the box size is updated by linear interpolation.",
        paperIds: ["T035"],
      },
    ],
    datasets: ["otb", "vot", "lasot", "got10k", "trackingnet"],
    metrics: ["success-auc", "eao", "precision", "norm-precision", "fps"],
    baselines: ["SiamFC", "ECO", "MDNet", "SiamRPN++", "ATOM"],
    results: [
      "OTB2015 success (Table 2): 68.3 for SiamFC++-GoogLeNet and 65.6 for SiamFC++-AlexNet, versus SiamFC 58.2, ECO 70.0, MDNet 67.8, SiamRPN++ 69.6, ATOM 66.9.",
      "VOT2018 (Table 2): EAO 0.426 (GoogLeNet) and 0.400 (AlexNet) versus SiamRPN++ 0.414, ATOM 0.401, ECO 0.280, SiamFC 0.188; robustness R 0.183 versus 0.234 (SiamRPN++) and 0.204 (ATOM) — the paper claims the first tracker reaching EAO 0.400 on VOT2018 while running over 100 FPS.",
      "GOT-10k under the restricted protocol (trained on the train subset only): AO 59.5 for the GoogLeNet version versus ATOM 55.6 and SiamRPN++ 51.8, with SR0.5 69.5 and SR0.75 47.9 versus ATOM's 63.4 and 40.2; the text reports the AlexNet version at AO 53.5, 1.7 points above SiamRPN++ (Table 2 lists 49.3).",
      "TrackingNet test with YouTube-BB excluded from training: success (AUC) 75.4, precision 70.5, normalized precision 80.0 for GoogLeNet versus SiamRPN++ 73.3 / 69.4 / 80.0 — the first tracker above 75 AUC on TrackingNet.",
      "LaSOT Protocol II: success 54.4 (GoogLeNet) and 50.1 (AlexNet) versus ATOM 51.5, SiamRPN++ 49.6, MDNet 39.7; SR0.5 69.5 and SR0.75 47.9 for the GoogLeNet version.",
      "Speed and size: 160 FPS (AlexNet) and about 90 FPS (GoogLeNet) on an NVIDIA RTX 2080Ti; the GoogLeNet model uses 9.16M parameters and 15.9G MACs at 303×303 against 45.5M parameters and 44.7G MACs for the ResNet-50 SiamRPN++ at 255×255 (Table 3).",
    ],
    ablations: [
      "Table 1, from SiamFC to SiamFC++ on VOT2018: EAO 0.213 → 0.276 (extra training data) → 0.296 (3×conv3 head) → 0.306 (COCO&DET + LaSOT&GOT) → 0.400 (cls+reg+PSS) → 0.426 (GoogLeNet); the regression branch alone contributes +0.094 EAO, data diversity +0.063 and +0.010, backbone +0.026, head structure +0.020, and robustness improves from R 0.566 to 0.183.",
      "Table 4, quality branch on GOT-10k val: no quality score 77.1 AO, PSS 77.8, IoU 78.0 — PSS is selected for cross-dataset stability; head depth 1×conv3 → 3×conv3 raises AO 76.2 → 78.1 and SR0.5 88.4 → 90.0, so the paper ships the two-layer head as the speed/accuracy balance.",
      "Table 4 data/diversity lines: adding COCO&DET and LaSOT&GOT to VID+Youtube lifts VOT2018 EAO from 0.352 to 0.378 and 0.400 (AlexNet) — richer and more diverse offline data monotonically helps a tracker with no online updates.",
    ],
    limitations: {
      authorStated: [],
      evident: [
        "Only the 303×303 search window is examined and there is no re-detection, so after a long occlusion recovery depends entirely on the inherited SiamRPN-style penalty and cosine-window heuristics.",
        "The test-phase penalty (k), cosine-window weight (Ω) and size-update rate (α) are untuned-by-the-method hyperparameters carried over from the SiamRPN family.",
      ],
    },
    assumptions: [
      "Target scale and aspect ratio change smoothly between adjacent frames, which is why box size is updated by confidence-weighted linear interpolation.",
      "The object stays inside the search window each frame and motion is smooth enough for a cosine window to suppress large displacements.",
    ],
    computation:
      "160 FPS (AlexNet) and about 90 FPS (GoogLeNet) on an NVIDIA RTX 2080Ti; 9.16M parameters / 15.9G MACs for SiamFC++-GoogLeNet at 303×303 versus 45.5M / 44.7G for ResNet-50 SiamRPN++ at 255×255; training is offline SGD on ILSVRC-VID/DET, COCO, YouTube-BB, LaSOT and GOT-10k.",
    relations: [
      {
        to: "T009",
        type: "builds-on",
        note: "SiamFC++ is explicitly built on SiamFC: the ablation in Table 1 starts from the SiamFC baseline and adds data, head and regression components until it reaches SiamFC++.",
      },
      {
        to: "T028",
        type: "addresses-limitation",
        note: "Section 4 dissects SiamRPN++ as violating guidelines G2, G3 and G4 (anchor-object scoring, anchor priors, no quality score) and shows its GOT-10k SR0.5/SR0.75 lie 7.7/15.4 points below SiamFC++.",
      },
      {
        to: "T026",
        type: "uses-as-baseline",
        note: "ATOM is one of the five trackers in Table 2 (OTB success 66.9, VOT2018 EAO 0.401, GOT-10k AO 55.6 versus SiamFC++ 68.3 / 0.426 / 59.5).",
      },
      {
        to: "T011",
        type: "uses-as-baseline",
        note: "ECO appears in Table 2 (OTB success 70.0, VOT2018 EAO 0.280) as the correlation-filter representative of the multi-scale-search generation.",
      },
      {
        to: "T025",
        type: "uses-as-baseline",
        note: "Evaluates on the GOT-10k benchmark under its restricted protocol, training only on the provided train subset (AO 59.5 for the GoogLeNet version).",
      },
    ],
    concepts: ["sot", "siamese", "anchor-free-head", "iou", "bounding-box", "success-plot"],
    impact:
      "SiamFC++ crystallized the anchor-free argument for single-object tracking: dropping anchors while adding an independent quality score beats anchor-based RPN trackers at 90–160 FPS, and its per-pixel classification/regression decomposition with a separate quality term is the same design that SiamCAR (T036) realizes with a center-ness branch, also reaching first place on GOT-10k.",
  },
  {
    id: "T036",
    arxiv: "1911.07241",
    title: "SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking",
    shortTitle: "SiamCAR",
    year: 2019,
    authors: ["Dongyan Guo", "Jun Wang", "Ying Cui", "Zhenhua Wang", "Shengyong Chen"],
    fileName: "1911.07241v2.pdf",
    task: "single-object",
    tags: ["anchor-free", "siamese", "center-ness", "per-pixel-prediction", "real-time"],
    difficulty: "intermediate",
    summary:
      "SiamCAR decomposes tracking into per-pixel classification (is this location the object?) and regression (what box does it own?), solved by a two-subnetwork fully convolutional Siamese model with a center-ness branch that suppresses off-center outliers. Both proposal-free and anchor-free, it takes first place on GOT-10k (AO 0.569, SR0.5 0.670, 52.27 FPS) and leads LaSOT, UAV123 and the evaluated OTB-50 attributes without any anchor tuning.",
    problem:
      "Anchor-based trackers such as SiamRPN and SiamRPN++ are sensitive to the number, size and aspect ratio of their anchor boxes, so expertise is needed to tune hyperparameters and the classification score measures anchor-object similarity rather than object existence. A tracker that removes anchors and proposals entirely while still predicting a precise box — and stays real-time — did not exist, and the anchor-free trackers of the time (ECO and friends) lagged far behind on large-scale benchmarks.",
    background: ["sot", "siamese", "anchor-free-head", "bounding-box", "iou", "success-plot"],
    previousWork: [
      {
        name: "SiamRPN, DaSiamRPN and SiamRPN++ (RPN-based siamese trackers)",
        limitation:
          "Anchors make these trackers sensitive to anchor numbers, sizes and aspect ratios so that hand-tuned hyperparameters are required, and they struggle against distractors with similar appearance to the target.",
        whyThisPaper:
          "SiamCAR classifies and regresses directly at each location of the response map, removing anchors and proposals while beating SiamRPN++ by 5.2% AO on GOT-10k.",
      },
      {
        name: "SiamFC and multi-scale siamese tracking",
        limitation:
          "One single similarity map contains limited spatial information, so scale invariance is obtained by re-running matching over multiple search scales — time-consuming and labor-intensive.",
        whyThisPaper:
          "The regression branch predicts all four side distances at every pixel in a single forward pass, and scale is handled by averaging the boxes of the top-3 neighboring locations.",
      },
      {
        name: "ECO and other anchor-free correlation-filter trackers",
        limitation:
          "The paper notes the accuracy and speed of anchor-free trackers like ECO still has a gap with anchor-based trackers on challenging benchmarks such as GOT-10k (ECO AO 0.316).",
        whyThisPaper:
          "SiamCAR closes and reverses that gap with a plain anchor-free architecture: AO 0.569 versus ECO's 0.316 on GOT-10k at 52.27 FPS versus 2.62 FPS.",
      },
    ],
    researchGap:
      "Before this paper, anchor-free trackers lagged far behind RPN trackers on large-scale benchmarks, so anchors were considered necessary for accurate state estimation.",
    contribution: [
      "Decomposition of tracking into per-pixel classification and bounding-box regression, trained end-to-end by a Siamese classification-regression subnetwork — the framework is both proposal-free and anchor-free.",
      "Depth-wise cross-correlation producing a multi-channel response map (channel-by-channel, R = ϕ(X) ⋆ ϕ(Z)), compressed by a 1×1 convolution to 256 channels, preserving per-channel semantic information that SiamFC's single-channel correlation throws away.",
      "Center-ness branch: C(i,j) = sqrt(min(l̃,r̃)/max(l̃,r̃) · min(t̃,b̃)/max(t̃,b̃)) scores how centered a location is inside its box and removes off-center outliers, trained with a binary cross-entropy loss.",
      "Tracking phase: scale-change penalty p_ij re-ranks classification scores with a cosine window (Eq. 9), and the final box is the weighted average of the regression boxes of the top-3 neighbors of the winning location (n = 8 neighborhoods, k = 3).",
      "First place on GOT-10k with AO 0.569, SR0.5 0.670, SR0.75 0.415 at 52.27 FPS (5.2%, 5.4% and 9.0% relative gains over SiamRPN++), plus best results on LaSOT and UAV123 among the compared trackers.",
    ],
    method: {
      pipeline: ["extract", "cross-correlate", "classify", "regress", "score-center-ness", "penalize", "select"],
      architecture:
        "Modified ResNet-50 (as in SiamRPN++) as the shared Siamese backbone, concatenating the last three residual blocks into ϕ with 3×256 channels; a depth-wise cross-correlation layer builds the multi-channel response map, a 1×1 convolution reduces it to 256 channels, and the classification-regression subnetwork emits three parallel 25×25 heads — foreground/background (2 channels), center-ness (1) and four side distances (4). Template 127 pixels, search region 255 pixels.",
      motionModel:
        "No explicit motion model; temporal coherence comes from the SiamRPN-style scale-change penalty p_ij applied to the classification score with a cosine window H (Eq. 9), plus averaging the regression boxes of the top-3 neighboring locations of the argmax.",
      appearanceModel:
        "Fully offline: the template is the initial-frame crop, the target branch is pre-computed and fixed for the whole sequence, and only the first-frame object is used as reference — no online model updates.",
      detectionDependency: "None — a proposal-free, anchor-free single-object tracker with no external detector.",
      loss:
        "L = L_cls + λ1·L_cen + λ2·L_reg with λ1 = 1 and λ2 = 3: cross-entropy for foreground/background classification, binary cross-entropy for center-ness over positive locations, and IoU loss on the four-distance regression masked by the indicator I(t̃(i,j)) (all four distances > 0).",
      optimization:
        "SGD with initial learning rate 0.001, batch size 96, 20 epochs: the Siamese subnetwork is frozen for the first 10 epochs while the classification-regression subnetwork trains, then the last 3 blocks of ResNet-50 are unfrozen for the final 10 epochs; about 42 hours on 4 RTX2080ti in PyTorch, with no data augmentation; trained on COCO, ImageNet DET, ImageNet VID and YouTube-BB (official splits only for GOT-10k and LaSOT).",
    },
    equations: [
      {
        id: "siamcar-corr",
        label: "Depth-wise siamese cross-correlation",
        formula: "R = ϕ(X) ⋆ ϕ(Z)",
        variables: [
          { symbol: "ϕ(X)", meaning: "search-region features: concatenation of the last three ResNet-50 residual blocks F3, F4, F5 (256 channels each)" },
          { symbol: "ϕ(Z)", meaning: "template features from the same backbone" },
          { symbol: "⋆", meaning: "channel-by-channel (depth-wise) cross-correlation" },
          { symbol: "R", meaning: "multi-channel response map retaining one map per semantic channel" },
        ],
        intuition:
          "Correlate each template channel with the matching search channel instead of collapsing everything into a single compressed map.",
        why:
          "A single-channel response map carries limited information for decoding location and scale; keeping channels preserves semantic detail for the classification-regression head.",
        where: "Section 3.1, Eq. 1; a 1×1 convolution afterwards reduces R to 256 channels before the prediction head.",
        params: "ϕ(X) = Cat(F3(X), F4(X), F5(X)) with 3×256 channels (Eq. 2).",
        paperIds: ["T036"],
      },
      {
        id: "siamcar-reg",
        label: "Masked IoU regression loss",
        formula: "L_reg = Σ_{i,j} I(t̃(i,j)) · L_IOU( A_reg(i,j,:), t̃(x,y) ) / Σ_{i,j} I(t̃(i,j))",
        variables: [
          { symbol: "t̃(i,j)", meaning: "regression target (l̃, t̃, r̃, b̃): distances from location (x, y) to the four sides of the ground-truth box" },
          { symbol: "I(t̃(i,j))", meaning: "indicator equal to 1 only when all four distances t̃k(i,j) > 0, i.e. the location lies inside the object" },
          { symbol: "L_IOU", meaning: "IoU loss between the predicted box and the target box" },
        ],
        intuition:
          "Average the overlap loss over the pixels that genuinely sit inside the object and ignore every background pixel.",
        why:
          "Locations outside the object have no meaningful four-sided target, so masking them keeps the regression head focused and the IoU loss optimizes the overlap metric that tracking actually measures.",
        where: "Section 3.2, Eq. 4–5; computed over the 25×25 response map during training.",
        params: "The regression head outputs a 4D vector t(i,j) = (l, t, r, b) per location.",
        paperIds: ["T036"],
      },
      {
        id: "siamcar-centerness",
        label: "Center-ness score",
        formula: "C(i, j) = I(t̃(i,j)) · sqrt( min(l̃, r̃)/max(l̃, r̃) × min(t̃, b̃)/max(t̃, b̃) )",
        variables: [
          { symbol: "l̃, r̃, t̃, b̃", meaning: "target distances from the location to the left, right, top and bottom box sides" },
          { symbol: "C(i, j)", meaning: "center-ness label in [0,1] for the location; 0 for background locations" },
        ],
        intuition:
          "A location near the object's center has almost equal left/right and top/bottom distances, so its score approaches 1; edge locations score low and background scores 0.",
        why:
          "Locations far from the object center produce low-quality boxes, so down-weighting them removes outliers before the final argmax.",
        where: "Section 3.2, Eq. 6; supervised by the center-ness loss of Eq. 7 (a binary cross entropy against the predicted center-ness map).",
        params: "The center-ness branch runs in parallel with classification and its score multiplies the re-ranked classification score during tracking.",
        paperIds: ["T036"],
      },
      {
        id: "siamcar-loss",
        label: "Overall training loss",
        formula: "L = L_cls + λ1 · L_cen + λ2 · L_reg",
        variables: [
          { symbol: "L_cls", meaning: "cross-entropy classification loss for foreground/background at each location" },
          { symbol: "L_cen", meaning: "center-ness loss of Eq. 7 over positive locations" },
          { symbol: "L_reg", meaning: "masked IoU regression loss of Eq. 4" },
          { symbol: "λ1, λ2", meaning: "loss weights, empirically set to 1 and 3" },
        ],
        intuition:
          "Train the three heads with one objective, weighting box overlap three times as heavily as center-ness.",
        why:
          "The final box comes from the argmax of re-ranked regression outputs, so overlap accuracy must dominate the optimization.",
        where: "Section 3.2, Eq. 8; minimized for all 20 training epochs.",
        params: "λ1 = 1, λ2 = 3 as stated by the paper.",
        paperIds: ["T036"],
      },
      {
        id: "siamcar-select",
        label: "Tracking-phase location selection",
        formula: "q = arg max_{i,j} [ (1 − λd) · cls_ij · p_ij + λd · H ]",
        variables: [
          { symbol: "cls_ij", meaning: "classification score at location (i, j), already combined with center-ness in the 6D output vector" },
          { symbol: "p_ij", meaning: "scale-change penalty re-ranking the score (as introduced in SiamRPN)" },
          { symbol: "H", meaning: "cosine window suppressing large displacements" },
          { symbol: "λd", meaning: "balance weight between the penalized score and the window" },
          { symbol: "q", meaning: "the winning location, whose neighborhood supplies the final box" },
        ],
        intuition:
          "Pick the pixel with the highest window-softened, scale-penalized score, then average the boxes predicted around it.",
        why:
          "Scale and aspect ratio change little between consecutive frames, so penalizing jumps and smoothing over neighbors keeps the per-pixel prediction stable.",
        where: "Section 3.3, Eq. 9; executed once per frame during tracking.",
        params: "The final prediction is the weighted average of the top-3 regression boxes among n = 8 neighborhoods of q; n = 8 and k = 3 were chosen empirically.",
        paperIds: ["T036"],
      },
    ],
    datasets: ["got10k", "lasot", "uav123", "otb"],
    metrics: ["success-auc", "precision", "norm-precision", "fps"],
    baselines: ["SiamRPN++", "SiamRPN", "SiamFC", "ECO", "MDNet"],
    results: [
      "GOT-10k (Table 1, official evaluation): AO 0.569, SR0.5 0.670, SR0.75 0.415 at 52.27 FPS — first on every indicator, ahead of SiamRPN++ (0.517 / 0.616 / 0.325 at 49.83 FPS) and SPM (0.513 / 0.593 / 0.359).",
      "LaSOT test set (Fig. 6): success 0.507, normalized precision 0.600 and precision 0.510 versus SiamRPN++ 0.496 / 0.569 / 0.491 — relative gains of 3.1%, 1.9% and 1.1%, and over 14%, 13.7% and 11% against the official baselines.",
      "UAV123 (Fig. 8): precision 0.760 and success 0.614, ranked first among the 10 compared trackers versus SiamRPN++ 0.752 / 0.610 and DaSiamRPN 0.724 / 0.569.",
      "OTB-50 per-attribute (Fig. 7): first on all four plotted attributes — low resolution precision 0.873 / success 0.677, background clutter 0.786 / 0.627, out-of-plane rotation 0.782 / 0.601, deformation 0.740 / 0.588 (SiamRPN++: 0.822 / 0.649, 0.711 / 0.586, 0.746 / 0.598, 0.732 / 0.573).",
      "Run-time (Table 1): 52.27 FPS on a single RTX 2080ti, ahead of SiamRPN++ at 49.83 FPS and far above ECO at 2.62 FPS and CCOT at 0.68 FPS.",
    ],
    ablations: [
      "Number of averaged neighbors k (Fig. 3): sweeping k = 0…6 on GOT-10k moves AO only within the plotted 0.556–0.57 band, so the paper fixes n = 8 neighborhoods and k = 3 as the setting that delivers stable tracking; there is no other component ablation table.",
    ],
    limitations: {
      authorStated: [
        "Conclusion: 'Since the present framework is simple and neat, it can be easily to be modified with specific modules to make further improvement in the future' — the authors present SiamCAR as a plain baseline rather than a complete solution.",
        "Related work: the tracking drift problem for template updating still need to be solved — a motivation for SiamCAR's fully offline design, which never updates the template.",
      ],
      evident: [
        "The template is frozen after the first frame, so appearance drift over long sequences can only be countered by the scale-change penalty and cosine window, not by model adaptation.",
        "OTB is evaluated on OTB-50 and only four per-attribute plots are printed, so no overall OTB-100 headline number is directly comparable to other trackers.",
      ],
    },
    assumptions: [
      "The target stays inside the 255×255 search region and its scale and aspect ratio change little between adjacent frames (the basis of the penalty and box averaging).",
      "The first-frame crop is a sufficient appearance model for the entire sequence.",
    ],
    computation:
      "52.27 FPS on one RTX 2080ti (GOT-10k official speed evaluation); training batch 96 for 20 epochs in about 42 hours on 4 RTX2080ti; template 127 pixels and search region 255 pixels.",
    relations: [
      {
        to: "T028",
        type: "uses-as-baseline",
        note: "SiamRPN++ is the primary comparison on every benchmark: GOT-10k AO 0.517 versus 0.569, LaSOT success 0.496 versus 0.507, UAV123 0.752/0.610 versus 0.760/0.614.",
      },
      {
        to: "T009",
        type: "uses-as-baseline",
        note: "SiamFC is named as the pioneering fully convolutional siamese tracker the framework follows and is compared on GOT-10k (AO 0.374), LaSOT and UAV123.",
      },
      {
        to: "T011",
        type: "uses-as-baseline",
        note: "ECO is the anchor-free correlation-filter representative whose gap to RPN trackers the paper sets out to close (GOT-10k AO 0.316 versus 0.569, 2.62 FPS versus 52.27).",
      },
      {
        to: "T025",
        type: "uses-as-baseline",
        note: "Evaluates on the GOT-10k benchmark using only its officially provided training split for a fair comparison (AO 0.569, first rank).",
      },
      {
        to: "T024",
        type: "uses-as-baseline",
        note: "Evaluates on the LaSOT benchmark, reporting the best success, normalized precision and precision among the top-20 compared trackers (Fig. 6).",
      },
    ],
    concepts: ["sot", "siamese", "anchor-free-head", "bounding-box", "iou", "success-plot"],
    impact:
      "SiamCAR proved that a plain anchor-free, proposal-free siamese head could top GOT-10k at real-time speed, consolidating the anchor-free wave of 2019–2020 alongside SiamFC++ (T035) and carrying the FCOS-style center-ness term into tracking heads; its per-pixel classification/regression layout became the default shape for one-stage siamese trackers.",
  },
  {
    id: "T037",
    arxiv: "1911.12836",
    title: "Siam R-CNN: Visual Tracking by Re-Detection",
    shortTitle: "Siam R-CNN",
    year: 2019,
    authors: ["Paul Voigtlaender", "Jonathon Luiten", "Philip H.S. Torr", "Bastian Leibe"],
    fileName: "1911.12836v2.pdf",
    task: "single-object",
    tags: ["re-detection", "two-stage-detector", "long-term-tracking", "hard-example-mining", "dynamic-programming"],
    difficulty: "advanced",
    summary:
      "Siam R-CNN turns a frozen COCO-pretrained Faster R-CNN into a siamese re-detector that asks, for every region proposal in the whole image, whether it is the same object as the first-frame template. A Tracklet Dynamic Programming Algorithm then combines re-detections of the first frame and of the previous frame over the full history of target and distractor tracklets, giving the best published results on ten benchmarks — 81.2 success on TrackingNet, 64.8 on LaSOT, MaxGM 72.3 on OxUvA — especially for long-term tracking, at 4.7 FPS.",
    problem:
      "Single-stage siamese trackers search only a small window around the previous prediction, so they drift when a similar distractor enters that window and cannot recover once the object disappears; the alternatives — strong spatial priors or online model updates — drift as well. Full-image re-detection is how tracking started, but no re-detector had been trained to be robust to the specific distractors of a given video, and no method explicitly modelled the history of both the target and its distractors.",
    background: ["sot", "siamese", "detection", "rpn-head", "occlusion", "appearance-features"],
    previousWork: [
      {
        name: "SiamRPN-style local-window trackers (SiamRPN++, DaSiamRPN, SiamMask)",
        limitation:
          "They search only within a small window of the previous prediction and suppress distractors via spatial priors, so they drift and cannot re-detect the object after disappearance; long-term variants merely enlarge that window when confidence drops.",
        whyThisPaper:
          "Siam R-CNN re-detects over the whole image with a two-stage head and resolves ambiguity with the explicit tracklet history, reaching F 66.8 on LTB35 and MaxGM 72.3 on OxUvA.",
      },
      {
        name: "DiMP-50 and ATOM (previous state of the art on short-term benchmarks)",
        limitation:
          "Both still predict within a local region around the last state (meta-learned or online-updated appearance models), which caps their long-term re-detection ability — they were the previous best on TrackingNet (74.0), GOT-10k (61.1) and LaSOT (56.8).",
        whyThisPaper:
          "Whole-image re-detection plus TDPA beats DiMP-50 by 7.2 points on TrackingNet, 3.8 on GOT-10k and 8.0 on LaSOT.",
      },
      {
        name: "Online-learned re-detection classifiers (Avidan's SVM tracker, Grabner et al., Struck)",
        limitation:
          "They learn the target classifier from the video itself during tracking, which is slow and prone to drift, and they had no mechanism to mine the hard distractors that re-detection must reject.",
        whyThisPaper:
          "Siam R-CNN learns appearance variation offline on 18k videos and mines hard negatives from other videos with an embedding network plus an approximate-nearest-neighbour index.",
      },
    ],
    researchGap:
      "Before this paper, no tracker combined a powerful offline-trained two-stage re-detector with an explicit model of the full target-and-distractor history, so both long-term recovery after occlusion and distractor suppression were weak.",
    contribution: [
      "Siam R-CNN re-detector: a COCO-pretrained Faster R-CNN (ResNet-101-FPN, group normalization, cascade) with frozen backbone and RPN, whose re-detection head concatenates RoI-Aligned features of each proposal with those of the first-frame ground-truth box and classifies 'same object / not' through a three-stage cascade.",
      "Video hard example mining: a PReMVOS embedding network (batch-hard triplet loss) plus an ANN index retrieves the 10,000 nearest neighbour boxes from other videos and samples 100 of them as negatives each training step — worth up to 1.7 AUC points.",
      "Tracklet Dynamic Programming Algorithm (TDPA): detections are grouped into tracklets with similarity threshold α and ambiguity margin β, and dynamic programming over the array θ maximizes the unary-plus-location score of a tracklet sequence starting at the first-frame tracklet, allowing an explicit 'object absent' decision.",
      "State of the art on ten benchmarks: TrackingNet 81.2 success / 85.4 normalized precision, LaSOT 64.8, GOT-10k 64.9, NfS 63.9, OxUvA MaxGM 72.3, LTB35 F 66.8, DAVIS 2017 J&F 70.6 with box-only initialization, YouTube-VOS 68.3.",
      "Speed variants: 4.7 FPS for the full model, 13.6 FPS at half input resolution with 100 RoIs, 15.2 FPS with a ResNet-50 backbone.",
    ],
    method: {
      pipeline: ["propose", "re-detect", "score-pairs", "build-tracklets", "dynamic-program", "select"],
      architecture:
        "Faster R-CNN with a ResNet-101-FPN backbone, group normalization and cascade heads, pre-trained on COCO with backbone and RPN frozen; RoI Align features of each proposal are concatenated with the first-frame ground-truth box features, halved by a 1×1 convolution, and passed to a three-stage cascade re-detection head with two classes. RoIs come from the RPN union up to 100 previous-frame boxes.",
      motionModel:
        "No filter. Motion enters only through (i) the previous-frame boxes added to the RoI set to compensate RPN false negatives, (ii) a spatial gate γ that sets pairwise similarity scores to −∞ when the L∞ distance between box centers and sizes exceeds γ, and (iii) the L1 location score that penalizes spatial jumps between tracklets.",
      appearanceModel:
        "Two references blended with weight w_ff: re-detection against the first-frame ground-truth box (ff_score) and against the most recent detection of the first-frame tracklet (ff_tracklet_score); pairwise similarity between current and previous detections re-uses the same head with detections as reference. Hard negatives are retrieved with a PReMVOS embedding network trained by batch-hard triplet loss.",
      detectionDependency:
        "Self-contained: proposals come from its own frozen RPN plus previous-frame boxes; no external detector and no first-frame mask are required (segmentation, when needed, comes from the off-the-shelf Box2Seg network).",
      trackManagement:
        "Algorithm 1: extend a tracklet only when the similarity to a previous detection is above α and no other detection or tracklet is within a margin β of that score; otherwise start a new tracklet. Tracklets terminate when not extended, θ[a] stores the best sequence score ending at tracklet a, the maximum temporal gap between tracklets is 1500 frames, and if the best tracklet has no detection in the current frame the object is declared absent (its most recent box is emitted with score 0 for benchmarks requiring a prediction).",
      loss:
        "Standard Faster R-CNN detection losses (classification and box regression across the three cascade stages) on the 2-class re-detection head, trained on frame pairs of ImageNet VID (4000 videos), YouTube-VOS 2018 (3471), GOT-10k (9335) and LaSOT (1120) with 100 sampled hard negatives per step.",
      optimization:
        "Backbone and RPN stay frozen; only the re-detection head after concatenation is trained, on pairs of frames where one frame is the reference and the other the target; augmentations are motion blur, grayscale, gamma, flip and scale; 18k videos and 119k static COCO images (versus SiamRPN++'s 384k videos and 1867k images); TDPA hyper-parameters are tuned once on the DAVIS 2017 training set and used unchanged on all benchmarks.",
    },
    equations: [
      {
        id: "siamrcnn-tdpa-score",
        label: "Tracklet sequence score (TDPA)",
        formula: "score(A) = Σ_{i=1..N} unary(a_i) + Σ_{i=1..N−1} w_loc · loc_score(a_i, a_{i+1})",
        variables: [
          { symbol: "A", meaning: "a sequence of N non-overlapping tracklets (end(a_i) < start(a_{i+1})) between the first frame and now" },
          { symbol: "unary(a_i)", meaning: "quality of an individual tracklet, summed over its detections" },
          { symbol: "loc_score", meaning: "penalty for the spatial jump between consecutive tracklets" },
          { symbol: "w_loc", meaning: "weight balancing tracklet quality against spatial continuity" },
        ],
        intuition:
          "Score a candidate history by how confident each of its segments is plus how small the jumps between segments are, then choose the best history with dynamic programming.",
        why:
          "Per-frame decisions drift on distractors; pooling evidence over the whole tracklet history resolves ambiguities that no pairwise visual similarity can.",
        where: "Section 3.3, Eq. 1; maximized online through the array θ whose entries hold the best sequence score ending at each tracklet.",
        params: "θ[a_ff] = 0 for the first-frame tracklet, all tracks must start there, and the maximum temporal gap between tracklets is 1500 frames.",
        paperIds: ["T037"],
      },
      {
        id: "siamrcnn-unary",
        label: "Tracklet unary score",
        formula: "unary(a_i) = Σ_{t=start(a_i)}^{end(a_i)} [ w_ff · ff_score(a_{i,t}) + (1 − w_ff) · ff_tracklet_score(a_{i,t}) ]",
        variables: [
          { symbol: "ff_score", meaning: "re-detection confidence of detection a_{i,t} against the first-frame ground-truth box" },
          { symbol: "ff_tracklet_score", meaning: "confidence using the most recent detection of the first-frame tracklet as the reference" },
          { symbol: "w_ff", meaning: "blend weight between the fixed first-frame reference and the recent reference" },
        ],
        intuition:
          "Average how well each detection matches the original template and how well it matches the most confident recent observation of the target.",
        why:
          "The first-frame reference anchors identity, while the tracklet reference absorbs appearance change; blending both keeps identity stable across long videos.",
        where: "Section 3.3, Eq. 2; part of the DP recurrence for θ.",
        params: "One tracklet always contains the first-frame ground-truth box (a_ff); detections in that tracklet are almost certainly correct because ambiguous extensions start new tracklets.",
        paperIds: ["T037"],
      },
      {
        id: "siamrcnn-loc-score",
        label: "Location score between tracklets",
        formula: "loc_score(a_i, a_j) = − ‖ end_bbox(a_i) − start_bbox(a_j) ‖₁",
        variables: [
          { symbol: "end_bbox(a_i)", meaning: "bounding box (x, y, w, h) of the last detection of tracklet a_i, normalized to [0,1]" },
          { symbol: "start_bbox(a_j)", meaning: "bounding box of the first detection of tracklet a_j" },
          { symbol: "‖·‖₁", meaning: "L1 norm over the four box parameters" },
        ],
        intuition:
          "Punish histories in which the object teleports between two consecutive tracklets.",
        why:
          "Spatial continuity is the only cue besides appearance, and it is what suppresses distractor tracklets that take over for a few frames.",
        where: "Section 3.3 (Scoring); used inside Eq. 1 and weighted by w_loc.",
        params: "Box coordinates are normalized by image width and height; pairwise similarity itself is gated by the L∞ spatial distance γ before scoring.",
        paperIds: ["T037"],
      },
      {
        id: "siamrcnn-fscore",
        label: "F-score for long-term presence decisions",
        formula: "F = 2 · Pr · Re / (Pr + Re)",
        variables: [
          { symbol: "Pr", meaning: "precision of presence decisions at a confidence threshold" },
          { symbol: "Re", meaning: "recall of presence decisions at the same threshold" },
        ],
        intuition:
          "Combine how often the tracker says 'present' correctly with how many present frames it catches into a single number.",
        why:
          "Long-term benchmarks require deciding whether the object exists at all, so trackers are ranked by the maximum F over confidence thresholds rather than by overlap.",
        where: "Section 4.2, LTB35 evaluation protocol; the tracker outputs a presence confidence every frame.",
        params: "LTB35 sequences average 12.4 disappearances per video, each about 40.6 frames long.",
        paperIds: ["T037"],
      },
    ],
    datasets: ["otb", "vot", "got10k", "lasot", "trackingnet", "uav123", "others"],
    metrics: ["success-auc", "eao", "precision", "norm-precision", "fps", "recall"],
    baselines: ["DiMP-50", "SiamRPN++", "ATOM", "UPDT", "SiamMask", "DaSiamRPN"],
    results: [
      "TrackingNet (Table 2): success 81.2, precision 80.0, normalized precision 85.4 versus DiMP-50 74.0 / 68.7 / 80.1 — +7.2 success and more than +10 precision over the previous best.",
      "GOT-10k test (Fig. 6): 64.9 success versus DiMP-50 61.1 (+3.8), with a model trained only on the GOT-10k training set; NfS success 63.9 versus DiMP-50 62.0 (Table 1).",
      "VOT2018 (Table 3): the full TDPA variant reaches EAO 0.140 with the highest accuracy (0.624) but frequent resets, the short-term variant reaches 0.408 EAO, and estimating rotated boxes from Box2Seg masks raises EAO to 0.423 with accuracy 0.684.",
      "LaSOT (Fig. 8): success 64.8 and normalized precision 72.2 versus DiMP-50 56.8 / 64.8 (+8 and +7.4 points); LTB35 F-score 66.8 (+3.9 over the previous best); OxUvA MaxGM 72.3 with TPR 70.1 and TNR 74.5 (Table 4).",
      "Video object segmentation with box-only initialization: DAVIS 2017 J&F 70.6 (J 66.1, F 75.0, Jbox 78.3) at 0.32 s per frame, +14.8 points over SiamMask's box-only result; YouTube-VOS 68.3 overall, +15.5 points ahead of other methods that use only the first-frame bounding box.",
      "Short-term breadth: OTB2015 70.1 AUC (tying UPDT, the previous best) and UAV123 64.9 AUC versus DiMP-50 65.4; 4.7 FPS for the full model.",
    ],
    ablations: [
      "Video hard example mining (Table 7): removing it costs 1.7 AUC on OTB2015 (70.1 → 68.4), 1.6 on LaSOT (64.8 → 63.2) and 0.3 on LTB35 F (66.8 → 66.5).",
      "Association strategy (Table 7): TDPA beats per-frame argmax on all three datasets (OTB2015 70.1 vs 63.8, LaSOT 64.8 vs 62.9, LTB35 66.8 vs 65.5), while the short-term algorithm collapses on the long-term sets (LaSOT 55.7, LTB35 57.2).",
      "Speed/accuracy trade-off (Table 7): half input resolution with 100 RoIs gives 13.6 FPS at 69.1 / 63.2 / 66.0, and ResNet-50 gives 5.1 FPS at 68.0 / 62.3 / 64.4 — with the same backbone Siam R-CNN still beats DiMP-50 on LaSOT (62.3 vs 56.8), so the gain is not just model size.",
      "RPN recall (Fig. 10): with 1000 proposals the frozen COCO RPN recalls only 69.1% of objects outside the 80 COCO classes versus 98.2% for known classes; adding up to 100 previous-frame boxes lifts unknown recall to 95.5%, while 10000 proposals reach 98.7% at about 1 FPS.",
    ],
    limitations: {
      authorStated: [
        "'Even the fastest variant is not real-time and our work focuses on accuracy' — the full model runs at 4.7 FPS.",
        "The backbone and RPN are frozen after COCO pre-training on 80 object classes, which the paper identifies as the reason the RPN recalls only 69.1% of unknown objects with 1000 proposals.",
        "VOT2018's extreme short-term reset protocol is 'not what Siam R-CNN with the TDPA was designed for', which is why the full variant scores only 0.140 EAO there.",
      ],
      evident: [
        "4.7 FPS is an order of magnitude below the 30–90 FPS single-stage siamese trackers it outperforms, so the accuracy gain costs real-time capability.",
        "Training requires 18k videos, 119k static images and a separate PReMVOS embedding network, making reproduction far heavier than for single-stage trackers.",
      ],
    },
    assumptions: [
      "The object is visible in the first frame and its RoI-Aligned features remain a usable reference for the whole video.",
      "Every true detection is proposed by the RPN or arrives among the previous-frame boxes — recall, not classification, is the bottleneck (Fig. 10).",
      "Target and distractor boxes can be separated by pooling appearance and spatial continuity over tracklets rather than by a single-frame score.",
    ],
    computation:
      "4.7 FPS with a ResNet-101 backbone, 1000 RPN proposals per frame and TDPA on a V100 GPU; 13.6 FPS at half input resolution with 100 RoIs, 15.2 FPS with ResNet-50; Box2Seg segmentation adds 0.025 s per object per frame.",
    relations: [
      {
        to: "T031",
        type: "uses-as-baseline",
        note: "DiMP-50 is the previous best on the headline tables: TrackingNet 74.0 vs 81.2, GOT-10k 61.1 vs 64.9, LaSOT 56.8 vs 64.8, NfS 62.0 vs 63.9.",
      },
      {
        to: "T028",
        type: "uses-as-baseline",
        note: "SiamRPN++ is compared on OTB2015 (0.696 vs 0.701 success), UAV123, VOT2018 (EAO 0.414) and in the DAVIS/YouTube-VOS Box2Seg experiments.",
      },
      {
        to: "T027",
        type: "uses-as-baseline",
        note: "SiamMask is the previous best box-only VOS method and is beaten by 14.8 points of J&F on DAVIS 2017 (70.6 vs 55.8), including a variant where Box2Seg is applied to SiamMask's output.",
      },
      {
        to: "T026",
        type: "uses-as-baseline",
        note: "ATOM appears throughout the short-term comparisons (OTB2015 success 0.671 vs 0.701, UAV123 0.643 vs 0.649, NfS 58.4 vs 63.9).",
      },
      {
        to: "T025",
        type: "uses-as-baseline",
        note: "Evaluates on the GOT-10k benchmark under its restricted protocol with a model trained only on the GOT-10k training set (64.9 success, +3.8 over the previous best).",
      },
    ],
    concepts: ["sot", "siamese", "detection", "rpn-head", "occlusion", "appearance-features", "track-management"],
    impact:
      "Siam R-CNN revived full-image re-detection as a serious tracking paradigm: its TDPA and video hard-example mining showed that modelling the whole target-and-distractor history beats local-window trackers by up to 10 points on long-term benchmarks, and it shares the detection-community thread with Tracktor (T030), which also converts a detector into a tracker rather than training a siamese similarity matcher.",
  },
];
