import type { PaperRecord } from "../types";

/* batch-05 — T038..T049 */

export const BATCH_05: PaperRecord[] = [
  {
    id: "T038",
    arxiv: "2003.06761",
    title: "Siamese Box Adaptive Network for Visual Tracking",
    shortTitle: "SiamBAN",
    year: 2020,
    authors: ["Zedu Chen", "Bineng Zhong", "Guorong Li", "Shengping Zhang", "Rongrong Ji"],
    fileName: "2003.06761v2.pdf",
    task: "single-object",
    tags: ["siamese", "anchor-free", "classification-regression", "multi-level-prediction", "real-time"],
    difficulty: "intermediate",
    summary:
      "SiamBAN removes both scale-estimation crutches of Siamese trackers — exhaustive multi-scale search and hand-designed anchor boxes — by classifying and regressing the box directly on the correlation map. Every position predicts a foreground score plus four distances to the box sides in a fully convolutional, end-to-end trained network, reaching state-of-the-art on VOT2018/2019, OTB100, NFS, UAV123 and LaSOT at 40 FPS.",
    problem:
      "Accurate scale and aspect ratio estimation was previously handled either by running the network on many scaled crops (multi-scale search: slow, coarse, fixed aspect ratio) or by pre-defined anchor boxes whose scales and ratios need tedious heuristic tuning per dataset. Neither option lets a tracker regress a target box directly from the correlation features, and the anchor design adds many hyper-parameters and computational cost.",
    background: [
      "siamese",
      "rpn-head",
      "anchor-free-head",
      "bounding-box",
      "iou",
      "deep-detection",
      "appearance-features",
      "success-plot",
    ],
    previousWork: [
      {
        name: "SiamFC and correlation-filter trackers (ECO, SRDCF)",
        limitation:
          "Target scale is estimated by evaluating the model over a bank of scaled search crops (and aspect ratio stays fixed), which is compute-hungry and only samples a discrete set of scales.",
        whyThisPaper:
          "A single forward pass regresses all four box sides (left/top/right/bottom distances) at the winning location, so scale and aspect ratio change continuously with no re-scaling loop.",
      },
      {
        name: "Anchor-based Siamese trackers (SiamRPN, SiamRPN++)",
        limitation:
          "Anchor boxes must be carefully designed and fixed from heuristic knowledge; their scales/ratios are dataset-specific hyper-parameters, and classification runs at the anchor centre while regression moves the box — an inconsistency between the two tasks.",
        whyThisPaper:
          "The no-prior-box design outputs 5× fewer variables per location (no 5 anchors), and classification and regression are both performed at the same unshifted location with only offset regression, keeping the two tasks aligned.",
      },
      {
        name: "Anchor-free detectors (FCOS, FSAF, RepPoints)",
        limitation:
          "They assume the object categories are known in advance and only separate classes, but tracking must decide whether two unknown-category observations are the same object.",
        whyThisPaper:
          "SiamBAN grafts the Siamese template branch (first-frame appearance) onto dense anchor-free prediction, turning the question into foreground/background against the template.",
      },
    ],
    researchGap:
      "Before this paper, no Siamese tracker could estimate scale and aspect ratio by direct dense box regression without anchor boxes or multi-scale search.",
    contribution: [
      "Siamese box adaptive network with multiple box adaptive heads, trainable end-to-end offline with deep networks on ImageNet VID/DET, COCO, YouTube-BB, GOT-10k and LaSOT.",
      "No-prior-box design: each correlation-map location regresses a 4D vector of side distances, removing all anchor-related hyper-parameters and making the tracker flexible and general.",
      "State-of-the-art results at 40 FPS on six benchmarks: VOT2018, VOT2019, OTB100, NFS, UAV123 and LaSOT.",
    ],
    method: {
      pipeline: ["extract", "correlate", "classify", "regress", "fuse-levels", "decode", "smooth"],
      architecture:
        "ResNet-50 backbone (downsampling removed from the last two blocks; stride 1 in conv4/conv5 with atrous rates 2 and 4; 1×1 conv to 256 channels) shared by a template branch (centre 7×7 region, 127×127 patch) and a search branch (255×255 patch). Per-level box adaptive heads apply depth-wise cross-correlation: a classification module (2 channels: fg/bg) and a regression module (4 channels: side distances), adaptively fused across conv3/conv4/conv5 with learned weights αₗ, βₗ.",
      motionModel:
        "No explicit motion model: the search patch is centred on the previous target position; the selected box is smoothed with a cosine window and scale-change penalty (carried over from SiamRPN), then its size is updated by linear interpolation with the previous state.",
      appearanceModel:
        "Siamese template branch features ϕ(z) from the first frame act as the appearance reference; the correlation layer measures location-wise similarity against search features ϕ(x).",
      detectionDependency:
        "None — class-agnostic one-shot tracking; no external detector or model update.",
      loss:
        "Multi-task loss L = λ₁·L_cls + λ₂·L_reg with λ₁ = λ₂ = 1 (no hyper-parameter search): cross-entropy classification and IoU loss L_IoU = 1 − IoU for joint regression of all four side distances.",
      optimization:
        "ImageNet-pretrained backbone with the first two layers frozen; SGD, minibatch 28 pairs, 20 epochs; warmup lr 0.001→0.005 for 5 epochs then exponential decay 0.005→0.00005 over 15; heads trained alone in the first 10 epochs, backbone fine-tuned at 1/10 learning rate in the last 10; weight decay 1e-4, momentum 0.9; PyTorch on a GTX 1080Ti.",
    },
    equations: [
      {
        id: "siamban-corr",
        label: "Depth-wise cross-correlation heads",
        formula:
          "P_cls(w×h×2) = [ϕ(x)]_cls ⋆ [ϕ(z)]_cls\nP_reg(w×h×4) = [ϕ(x)]_reg ⋆ [ϕ(z)]_reg",
        variables: [
          { symbol: "ϕ(z), ϕ(x)", meaning: "template-branch and search-branch feature maps" },
          { symbol: "⋆", meaning: "convolution with the template features as kernel (depth-wise cross-correlation)" },
          { symbol: "P_cls", meaning: "classification map, 2 channels per location (foreground/background)" },
          { symbol: "P_reg", meaning: "regression map, 4 channels per location (distances to the four box sides)" },
        ],
        intuition:
          "Slide the template features over the search image; at every position ask two questions — 'is this the target?' and 'how far is the box edge in each direction?'",
        why:
          "This is what makes the tracker anchor-free: one correlation pass replaces both anchor enumeration and multi-scale search, outputting 5× fewer variables than anchor-based trackers with 5 anchors.",
        where: "Section 3.2 (Box Adaptive Head); one head per feature level (conv3/conv4/conv5).",
        params:
          "A 1×1 conv first cuts channels to 256 to control the correlation cost; only the template centre 7×7 region is used as kernel.",
        paperIds: ["T038"],
      },
      {
        id: "siamban-multilevel",
        label: "Adaptive multi-level fusion",
        formula:
          "P_cls-all(w×h×2) = Σ_{l=3}^{5} αₗ · P_l^cls\nP_reg-all(w×h×4) = Σ_{l=3}^{5} βₗ · P_l^reg",
        variables: [
          { symbol: "P_l", meaning: "classification/regression map predicted from feature level l (conv3/conv4/conv5)" },
          { symbol: "αₗ, βₗ", meaning: "learnable weights for each level's map, optimized jointly with the network" },
        ],
        intuition:
          "Early layers give fine localization, later layers give semantic robustness — learn how much of each to trust instead of hand-picking a level.",
        why:
          "All three levels share the same spatial resolution (atrous convolutions), so their maps can be summed directly; classification and regression are fused independently so each can weight levels for its own need.",
        where: "Section 3.3 (Multi-level Prediction); applied before decoding.",
        params:
          "Ablation on OTB100: conv4 alone is the best single level, conv4+conv5 the best pair, all three levels the best overall.",
        paperIds: ["T038"],
      },
      {
        id: "siamban-labels",
        label: "Ellipse label assignment",
        formula:
          "(pᵢ − g_xc)² / (g_w/2)² + (p_j − g_yc)² / (g_h/2)² = 1   (E1, outer)\n(pᵢ − g_xc)² / (g_w/4)² + (p_j − g_yc)² / (g_h/4)² = 1   (E2, inner)",
        variables: [
          { symbol: "(pᵢ, p_j)", meaning: "centre of the receptive field of a location on the search patch" },
          { symbol: "(g_xc, g_yc)", meaning: "centre of the ground-truth bounding box" },
          { symbol: "g_w, g_h", meaning: "ground-truth box width and height" },
        ],
        intuition:
          "Positive if well inside the target (inner ellipse), negative if clearly outside (outer ellipse), ignored in the ambiguous ring between them — and the ellipses stretch with the target's real shape.",
        why:
          "A plain centre-radius circle treats different-sized/elongated targets the same; the ellipse rule makes label assignment respect scale and aspect ratio, exactly the properties the regressor must learn.",
        where: "Section 3.4 (Ground-truth and Loss); labels sampled per image pair (≤16 positive, ≤48 negative).",
        params:
          "E2/E1 semi-axes are fixed fractions (1/4 and 1/2) of the box size; the band between them is an ignore buffer for ambiguous samples.",
        paperIds: ["T038"],
      },
      {
        id: "siamban-loss",
        label: "Classification + IoU loss",
        formula: "L = λ₁ · L_cls + λ₂ · L_reg ,  λ₁ = λ₂ = 1\nL_IoU = 1 − IoU",
        variables: [
          { symbol: "L_cls", meaning: "cross-entropy over foreground/background labels from the ellipse rule" },
          { symbol: "L_reg", meaning: "IoU loss between the decoded box and the ground-truth box" },
          { symbol: "IoU", meaning: "intersection-over-union of predicted and ground-truth boxes" },
        ],
        intuition:
          "Train the head to both spot the target and make the box actually overlap it; because positives lie inside E2, IoU is always > 0 so the loss stays in [0, 1).",
        why:
          "IoU loss couples all four side distances into one overlap objective (unlike per-side L2), so the box converges as a shape rather than four independent edges.",
        where: "Section 3.4; applied at every positively labelled location.",
        params: "λ₁ = λ₂ = 1 set without any hyper-parameter search, per the paper.",
        paperIds: ["T038"],
      },
      {
        id: "siamban-decode",
        label: "Box decoding",
        formula:
          "p_x1 = pᵢ − d_l^reg ,  p_y1 = p_j − d_t^reg\np_x2 = pᵢ + d_r^reg ,  p_y2 = p_j + d_b^reg",
        variables: [
          { symbol: "d_l, d_t, d_r, d_b", meaning: "predicted distances from the location to the left, top, right, bottom box sides" },
          { symbol: "(pᵢ, p_j)", meaning: "the location's receptive-field centre on the search patch" },
          { symbol: "p_x1..p_y2", meaning: "decoded top-left and bottom-right corners of the predicted box" },
        ],
        intuition:
          "Grow the box outward from the winning location by the four predicted side distances.",
        why:
          "Regression targets are positive real numbers, so exp(x) is applied at the head output to keep distances in (0, +∞); decoding then needs no anchor reference at all.",
        where: "Section 3.4, Eq. 8; run on the fused maps, followed by cosine-window/scale-penalty smoothing.",
        params:
          "Only the highest-scoring location is decoded; its box size is linearly interpolated with the previous frame's state.",
        paperIds: ["T038"],
      },
    ],
    datasets: ["otb", "vot", "got10k", "lasot", "uav123", "others"],
    metrics: ["eao", "success-auc", "precision", "norm-precision", "fps"],
    baselines: ["SiamRPN++", "DiMP", "ATOM", "ECO", "SiamMask", "DaSiamRPN"],
    results: [
      "VOT2018 (Table 1): EAO 0.452 (best of the 10 trackers compared), accuracy 0.597, robustness 0.178 — vs SiamRPN++ the failure rate drops 23.9% and EAO rises 8.4%; vs DiMP-50 (EAO 0.441) SiamBAN wins on EAO with equal accuracy and no online update.",
      "VOT2019 real-time (Table 2): EAO 0.327, accuracy 0.602 (highest), robustness 0.396 — failure rate 17.8% lower and EAO 14.7% (relative) higher than SiamRPN++.",
      "OTB100: AUC 0.696 (tied with SiamRPN++ for best), precision 0.910 (second to SiamRPN++'s 0.915).",
      "LaSOT test set (280 videos): success 0.514 (3rd, behind DiMP 0.568 and ATOM 0.515), normalized precision 0.598 (2nd, behind DiMP 0.648) — 5.1% above SiamRPN++ (0.569).",
      "UAV123: success 0.631, precision 0.833; NFS (30 FPS version): AUC 0.594, ranking 2nd and 40.8% above the best tracker reported in the original NFS paper.",
      "Runs at 40 FPS on the six tracking benchmarks.",
    ],
    ablations: [
      "Multi-level prediction (Table 4, OTB100): among single levels conv4 is best; conv4+conv5 is the best two-level pair; aggregating all three levels gives the top AUC (0.696 with ellipse labels).",
      "Label assignment (Table 4, Figure 10): ellipse vs circle vs rectangle labels under identical training — ellipse labels win (best row 0.696 AUC vs 0.662–0.689 for the alternatives), because they adapt positive/negative regions to the target's scale and aspect ratio; an ignore buffer between E2 and E1 discards ambiguous samples.",
      "Sampling balance: with far fewer negatives than anchor-based trackers, at most 16 positive and 48 negative samples are collected per image pair.",
    ],
    limitations: {
      authorStated: [],
      evident: [
        "No re-detection: the search patch is always centred on the previous frame's position, so after a long occlusion or fast target escape the tracker cannot look elsewhere (visible as LaSOT failures vs re-detection trackers).",
        "Anchor priors are gone but post-hoc heuristics remain — the cosine window and scale-change penalty from SiamRPN still shape every output.",
        "The ellipse rule's semi-axis fractions (1/2, 1/4) and the 16/48 sample caps are fixed hyper-parameters of the training recipe.",
      ],
    },
    assumptions: [
      "The target stays inside the search patch centred on its previous position.",
      "The first-frame template patch remains representative of the target's appearance.",
      "One target per sequence; no explicit motion model between frames.",
    ],
    computation:
      "40 FPS on VOT/OTB/NFS/UAV123/LaSOT (axis-aligned boxes without mask head); 127×127 template and 255×255 search inputs; ResNet-50; training on one Nvidia GTX 1080Ti (PyTorch, Intel Xeon 4108, 64 GB RAM).",
    relations: [
      {
        to: "T028",
        type: "addresses-limitation",
        note:
          "Drops SiamRPN++'s 5 hand-designed anchor boxes per location for a 4-side offset regression, keeping the depth-wise correlation — 5× fewer output variables and no anchor hyper-parameters.",
      },
      {
        to: "T009",
        type: "addresses-limitation",
        note:
          "Replaces SiamFC's multi-scale search with direct box regression, so scale and aspect ratio are predicted in one forward pass.",
      },
      {
        to: "T031",
        type: "uses-as-baseline",
        note:
          "DiMP-50 is the strongest EAO competitor on VOT2018 (0.441 vs SiamBAN 0.452) and is compared again on LaSOT, UAV123 and NFS.",
      },
      {
        to: "T026",
        type: "uses-as-baseline",
        note: "ATOM is compared on all six benchmarks, e.g. LaSOT success 0.515 vs SiamBAN 0.514.",
      },
    ],
    concepts: ["siamese", "anchor-free-head", "rpn-head", "iou", "sot"],
    impact:
      "SiamBAN became the standard box-adaptive (anchor-free) Siamese baseline: dense 4-side regression on the correlation map, plus its ellipse-vs-circle-vs-rectangle label study, was picked up by the anchor-free tracking line that Ocean (T046) later extended with object-aware features.",
  },
  {
    id: "T039",
    arxiv: "2003.09003",
    title: "MOT20: A benchmark for multi object tracking in crowded scenes",
    shortTitle: "MOT20",
    year: 2020,
    authors: [
      "Patrick Dendorfer",
      "Hamid Rezatofighi",
      "Anton Milan",
      "Javen Shi",
      "Daniel Cremers",
      "Ian Reid",
      "a.o.",
    ],
    venue: "4th BMTT MOT Challenge Workshop, CVPR 2019",
    fileName: "2003.09003v1.pdf",
    task: "benchmark",
    tags: ["benchmark", "tracking-by-detection", "crowded-scene", "pedestrian", "evaluation-protocol"],
    difficulty: "intro",
    summary:
      "MOT20 extends MOTChallenge with 8 new sequences from 3 extremely crowded scenes (up to 246 pedestrians per frame, ~10× denser than the first MOT release), annotated under the MOT16 protocol and evaluated with the CLEAR and track-quality metric families. It ships public Faster R-CNN detections as the only accepted detection input for the challenge and was first presented at the 4th BMTT MOT Challenge Workshop at CVPR 2019.",
    problem:
      "MOT15/16/17 scenes contain a moderate number of pedestrians, but the failure modes that matter most in real deployments — bodies fully overlapping, dozens of near-identical targets crossing at once — were never standardized: no benchmark measured how trackers behave when a frame contains hundreds of mutually occluding people, and no shared protocol separated 'track the person' from 'ignore the cyclist/sitting bystander' in such density.",
    background: ["mot", "tracking-by-detection", "benchmark-design", "mota", "occlusion", "detection", "bounding-box"],
    previousWork: [
      {
        name: "MOT16 / MOT17 (MOTChallenge)",
        limitation:
          "Carefully annotated and protocol-clean, but their scenes have far lower pedestrian density; detectors and trackers tuned on them degrade in crowds where occlusion is the norm rather than the exception.",
        whyThisPaper:
          "MOT20 keeps the exact annotation/evaluation protocol yet raises mean density to 127 pedestrians per frame (peak 246) and triples the number of boxes relative to MOT17, so crowd handling becomes measurable.",
      },
      {
        name: "Target-class handling in previous MOT challenges",
        limitation:
          "Whether to score sitting people, reflections, or people on bicycles is ambiguous: rewarding their tracking can inflate scores, penalizing correct filtering punishes sensible trackers.",
        whyThisPaper:
          "MOT20 formalizes three annotation classes (moving pedestrians to track; static/ambiguous people and vehicles/occluders to annotate but not score) and excludes result boxes overlapping >75% with distractor classes, so methods are neither penalized nor rewarded for them.",
      },
    ],
    researchGap:
      "Before this paper, MOTChallenge had no benchmark under extreme pedestrian density, so crowded-scene tracking was never standardized, annotated, or fairly measured.",
    contribution: [
      "8 new sequences from 3 crowded scenes (4 train / 4 test, one entire scene held back for testing), 13,410 frames, 3,457 annotated pedestrian trajectories and 1,652,040 pedestrian boxes — roughly 3× the boxes of MOT17.",
      "Annotation protocol inherited from MOT16 with a three-way class split (moving pedestrians; static/ambiguous people; vehicles and occluders incl. ground occluders and crowds), 13 label classes in total.",
      "Public baseline detections from a Faster R-CNN (ResNet-101) trained on the MOT20 training sequences; the challenge accepts results on public detections only.",
      "Evaluation framework: CLEAR MOTA/MOTP plus track-quality measures, with crowd-specific handling (75% distractor-overlap exclusion) and relative error ratios IDSW/Recall and FM/Recall.",
      "Per-sequence detector statistics and the full data format (9-column CSV: frame, id, box, confidence/flag, class, visibility) released for reproducible comparison.",
    ],
    method: {
      pipeline: ["collect", "annotate", "protocol", "evaluate"],
      architecture:
        "Benchmark system: 8 sequences (25 FPS, 1173×880 to 1920×1080) split 4/4 into train and test; images as 6-digit JPEGs, annotations and results as 9-column CSV files; a Faster R-CNN with ResNet-101 backbone supplies the public detections used for all tracking submissions.",
      association:
        "Evaluation-side matching: Hungarian (Munkres) assignment with temporal correspondence — if GT i was matched to hypothesis j at t−1 and their distance is below t_d, that correspondence carries into frame t even if another hypothesis is closer; t_d is IoU ≥ 0.5 (Jaccard index).",
      trackManagement:
        "Scoring assumes each GT trajectory has one unique start and end; re-entering targets are treated as new IDs (re-identification is explicitly not handled). Trajectories are classified Mostly Tracked (≥80% recovered), Mostly Lost (<20%), else Partially Tracked; fragmentations count tracked→untracked→tracked transitions.",
      detectionDependency:
        "Central by design: only public Faster R-CNN detections are accepted for the challenge, isolating tracker quality from detector choice.",
      optimization:
        "Detector training only: Faster R-CNN with ResNet-101 trained on the four MOT20 training sequences; no tracker-side optimization in the benchmark itself.",
    },
    equations: [
      {
        id: "mot20-mota",
        label: "MOTA (Multiple Object Tracking Accuracy)",
        formula: "MOTA = 1 − Σ_t (FN_t + FP_t + IDSW_t) / Σ_t GT_t",
        variables: [
          { symbol: "FN_t, FP_t", meaning: "missed and false hypotheses in frame t" },
          { symbol: "IDSW_t", meaning: "identity switches in frame t (assignment changed vs the previous known mapping)" },
          { symbol: "GT_t", meaning: "number of ground-truth objects in frame t" },
        ],
        intuition:
          "Sum every tracking error over the whole test set relative to the number of real objects; the score lives in (−∞, 100] as a percentage.",
        why:
          "MOTA is the benchmark's headline number because it combines the three error sources (misses, false alarms, identity changes) in one expression.",
        where: "Section 4.1.4, Eq. 1; reported over all test sequences concatenated rather than averaged per sequence.",
        params:
          "Can go negative when a tracker makes more errors than there are objects; that is why the paper also reports MOTA's standard deviation across sequences as a robustness measure.",
        simulator: "metrics",
        paperIds: ["T039"],
      },
      {
        id: "mot20-motp",
        label: "MOTP (Multiple Object Tracking Precision)",
        formula: "MOTP = Σ_{t,i} d_{t,i} / Σ_t c_t",
        variables: [
          { symbol: "d_{t,i}", meaning: "overlap of matched hypothesis i with its ground-truth object in frame t" },
          { symbol: "c_t", meaning: "number of matches in frame t" },
        ],
        intuition:
          "Average localization overlap over all correct matches — between t_d (50%) and 100%.",
        why:
          "It separates 'how well are boxes placed' from 'how many are found'; the paper stresses it measures detector localization, not precision/recall relevance.",
        where: "Section 4.1.5, Eq. 2; reported alongside MOTA for every sequence.",
        params:
          "Mostly quantifies detector localization accuracy in practice, so it says little about association quality.",
        simulator: "metrics",
        paperIds: ["T039"],
      },
    ],
    datasets: ["mot20"],
    metrics: ["mota", "motp", "fp", "fn", "idsw", "fragments", "mt-ml", "faf", "precision", "recall"],
    baselines: ["Faster R-CNN (public detections)"],
    results: [
      "Dataset scale: 13,410 frames (8,931 train / 4,479 test), 3,457 pedestrian trajectories, 1,652,040 pedestrian boxes; 2,102,385 total annotations including 335,891 static persons, 87,786 ground occluders, 22,365 non-motorized vehicles and 4,303 crowd labels.",
      "Crowd density: mean 127.04 pedestrians/frame on training sequences and 115.52 on test; MOT20-05 reaches 194.98 with 1,169 tracks and 646,344 boxes; the paper cites peaks of 246 pedestrians per frame — about 10× the first MOT release.",
      "Public Faster R-CNN detector (Table 6): 1,022,941 detections overall (76.28/frame); per-sequence recall ranges 55.2% (MOT20-08, FAR 13.93) to 86.5% (MOT20-01, FAR 0.14), with AP 0.38–0.82 — detection, not association, collapses in the densest test scenes.",
      "MOTChallenge at large: >1,000 active users, 44 sequences, 2.7M boxes, 36k seconds across five challenges; MOT20 adds ~3× more boxes than MOT17 for training and testing.",
      "Generalization protocol: test data contains sequences from known scenes plus one scene never seen at training time, to measure detector and tracker generalization.",
    ],
    ablations: [],
    limitations: {
      authorStated: [
        "The evaluation procedure does not explicitly handle target re-identification: a target that leaves the field of view and reappears is scored as a new identity.",
        "Whether one number (MOTA) alone can serve as a single performance measure is described as 'highly debatable' — hence the multi-metric report (MOTA, MOTP, MT/ML, FM, relative IDSW/Recall, FM/Recall, MOTA standard deviation, average rank).",
        "GT annotations for the test sequences are withheld to avoid (over)fitting to those specific sequences.",
      ],
      evident: [
        "The public-detection-only rule couples every tracker ranking to one Faster R-CNN: on MOT20-06/08 (recall 57.9/55.2, FAR 12.64/13.93) no association method can recover the missing ground truth.",
        "Distractor exclusion (>75% overlap with static-person/reflection/vehicle classes) is a hand-tuned threshold that silently changes which errors count.",
      ],
    },
    assumptions: [
      "Every visible moving pedestrian is annotated; each GT trajectory has a single start and end (no GT fragmentation).",
      "IoU ≥ 0.5 defines a valid match; one-to-one Hungarian assignment with temporal correspondence is the scoring rule.",
      "Static/ambiguous people and vehicles are annotated but carry no score, so trackers are agnostic to them.",
    ],
    computation:
      "Public Faster R-CNN (ResNet-101) detections precomputed on all sequences; evaluation devkit scripts published at motchallenge.net/devkit; results submitted as per-sequence CSVs zipped for server-side scoring.",
    relations: [
      {
        to: "T006",
        type: "extends",
        note:
          "Adopts MOT16's annotation/evaluation protocol unchanged, then extends it to extreme crowding with a formalized three-class target/distractor split and 75% distractor-overlap exclusion.",
      },
    ],
    concepts: ["mot", "benchmark-design", "mota", "tracking-by-detection", "occlusion"],
    impact:
      "MOT20 became MOTChallenge's crowded-scene stress test: its public-detection-only rule, distractor-exclusion logic and relative identity ratios (IDSW/Recall, FM/Recall) are the standardized way subsequent corpus MOT work reports identity stability under heavy occlusion.",
  },
  {
    id: "T040",
    arxiv: "2003.12565",
    title: "Probabilistic Regression for Visual Tracking",
    shortTitle: "PrDiMP",
    year: 2020,
    authors: ["Martin Danelljan", "Luc Van Gool", "Radu Timofte"],
    fileName: "2003.12565v1.pdf",
    task: "single-object",
    tags: ["probabilistic", "uncertainty", "meta-learning", "box-regression", "online-update"],
    difficulty: "advanced",
    summary:
      "PrDiMP reframes tracking regression as density estimation: the network predicts the conditional probability density p(y|x) of the target state, trained by KL divergence against a label distribution p(y|yi) that explicitly models annotation noise and task ambiguity. Integrated into both branches of DiMP, the formulation gives calibrated uncertainty (multi-modal densities under distractors) and a new state of the art — 59.8% AUC on LaSOT and 75.8% Success on TrackingNet.",
    problem:
      "Every dominant tracker — DCF filters, Siamese correlation, IoU-Net box heads, DiMP — maximizes a confidence score s(y, x) whose value depends on the chosen loss and pseudo-label scheme and has no probabilistic interpretation. Trackers therefore cannot compute absolute probabilities ('is the target lost?', 'how uncertain is this box?') even though such reasoning is exactly what update logic, failure detection and occlusion handling need; meanwhile the regression task itself is ambiguous (ground-truth box centres jump when the object deforms) and label noise is ignored.",
    background: ["sot", "bounding-box", "correlation-filter", "siamese", "deep-detection", "success-plot"],
    previousWork: [
      {
        name: "Confidence-based regression in DCF trackers (KCF, ECO) and Siamese trackers (SiamFC, SiamRPN++)",
        limitation:
          "They learn s(y, x) and take argmax; the score's range depends on the loss (squared, cross-entropy) and pseudo-label design (Gaussian peaks, binary labels), so uncertainty estimates built from it are ad hoc and uncalibrated.",
        whyThisPaper:
          "Predicts a normalized density p(y|x) — a continuous generalization of SoftMax — so probabilities are absolute and no loss/pseudo-label choice is needed (training minimizes KL divergence directly).",
      },
      {
        name: "ATOM's IoU-Net bounding box regression",
        limitation:
          "Trains an IoU confidence head with plain squared error on IoU pseudo-labels, sampling boxes during training; it cannot express that an annotation itself is noisy or that several boxes are equally 'correct'.",
        whyThisPaper:
          "The probabilistic BBR head models label uncertainty with an isotropic Gaussian p(y bb|yi) (σ²_bb = 0.05) and trains with the KL loss, adding +0.8% AUC over the previous NLL formulation on top of +1.2% from NLL alone.",
      },
      {
        name: "DiMP (T031)",
        limitation:
          "Its target-centre regression uses a robust-L2 confidence objective with Gaussian pseudo-labels, and the unrolled optimizer is derived from a Gauss-Newton approximation that only exists for least-squares objectives.",
        whyThisPaper:
          "Both DiMP branches are converted to the probabilistic KL objective, and the optimizer module is re-derived with a second-order Taylor (Newton) expansion with closed-form gradient and Hessian, since the KL loss is no longer least-squares.",
      },
    ],
    researchGap:
      "Before this paper, no tracker predicted a normalized probability density over target states or modeled the annotation noise that makes box regression ambiguous, so confidence outputs could not be interpreted or used to reason about uncertainty.",
    contribution: [
      "A general probabilistic regression formulation: the network parametrizes the conditional density p(y|x, θ) = e^{sθ(y,x)} / Zθ(x) directly through its architecture, trained by minimizing KL divergence to a label-conditional distribution p(y|yi) — replacing both the loss function and the pseudo-label function.",
      "Label-distribution modeling of annotation noise and task ambiguity (e.g. the ground-truth centre of a dog jumps to the background when its tail moves), with p(y|yi) estimated as a Gaussian whose σ could in principle be measured from repeated annotations.",
      "Two training approximations: grid sampling for fully convolutional centre regression (yielding a SoftMax cross-entropy loss) and Monte Carlo importance sampling with a Gaussian-mixture proposal for the 4-D bounding-box density.",
      "Probabilistic DiMP (PrDiMP): both the target-centre and bounding-box branches of DiMP made probabilistic, with a Newton-step optimizer module (closed-form gradient/Hessian) for the non-least-squares objective.",
      "State-of-the-art on six datasets, e.g. LaSOT +2.9% AUC and TrackingNet +1.8% Success over DiMP, with ResNet-18/50 versions running at ~40/30 FPS.",
    ],
    method: {
      pipeline: ["predict-model", "localize", "predict-density", "argmax", "box-regress", "update"],
      architecture:
        "DiMP (via PyTracking) with two probabilistic branches: (1) Target Centre Regression — a linear convolution filter wθ, predicted by a meta-learned unrolled optimizer from support samples, densely scores the search region and the scores are normalized into a density (Gaussian label, σ_tc = 1/14 of target size, grid-sampled KL loss); (2) Bounding Box Regression — ATOM's IoU-Net-style head repredicts the 4-D box density p(y bb|x, θ), trained with MC-sampled KL loss against an isotropic Gaussian label (σ_bb = 0.05) with proposal q = ½N(0.05²) + ½N(0.5²). ResNet-18 or ResNet-50 backbone.",
      motionModel:
        "No explicit motion model: the centre branch scores a wide search region around the previous target location; the point estimate is the argmax of the predicted density.",
      appearanceModel:
        "Meta-learned discriminative filter wθ over backbone features φθ(x), predicted from a memory of sequence samples (first-frame augmentations plus gradual memory updates), as in DiMP.",
      detectionDependency: "None — standalone tracker, no external detector.",
      trackManagement:
        "Inherited from DiMP with minimal changes: only the missing-target threshold and the gradient step length of the box regression were adjusted for the different output scaling.",
      loss:
        "KL divergence KL(p(·|yi), p(·|xi, θ)) ≈ log ∫ e^{sθ(y,xi)} dy − ∫ sθ(y, xi) p(y|yi) dy, approximated by grid sampling (centre branch, +L2 regularizer in the optimizer module) or Monte Carlo importance sampling (box branch).",
      optimization:
        "End-to-end joint training as DiMP: LaSOT/GOT10k/TrackingNet/COCO training splits, 50 epochs × 1000 iterations. The optimizer module performs steepest descent with Newton step α⁽ⁱ⁾ = (∇L)ᵀ∇L / ((∇L)ᵀH∇L), where the Hessian uses the SoftMax Jacobian diag(p̂) − p̂p̂ᵀ; the original Gauss-Newton approximation was replaced by a second-order Taylor expansion because the KL objective is not least-squares.",
    },
    equations: [
      {
        id: "prdim-density",
        label: "Predictive density (continuous SoftMax)",
        formula: "p(y|x, θ) = e^{sθ(y,x)} / Zθ(x) ,   Zθ(x) = ∫_Y e^{sθ(y,x)} dy",
        variables: [
          { symbol: "sθ(y, x)", meaning: "raw network score for state y on image x (unnormalized log-density)" },
          { symbol: "Zθ(x)", meaning: "partition function normalizing the density over the output space Y" },
          { symbol: "y", meaning: "target state: centre coordinate ∈ R² or box ∈ R⁴" },
        ],
        intuition:
          "Exponentiate the score the network already produces and divide by its integral — the confidence map becomes a genuine probability distribution over where the target can be.",
        why:
          "Only a normalized density supports absolute probabilities ('target lost', 'two equally likely boxes'); the paper stresses this is a direct generalization of SoftMax to an arbitrary output space, with no assumed Gaussian family.",
        where: "Section 3.1, Eq. 6; applied to both the centre and box branches at inference; the argmax of p equals the argmax of s.",
        params:
          "Flexible: multi-modal densities arise naturally (e.g. cat vs mirror), which confidence scores cannot express meaningfully.",
        paperIds: ["T040"],
      },
      {
        id: "prdim-kl-loss",
        label: "KL-divergence training loss",
        formula:
          "KL(p(·|yi), p(·|xi, θ)) = ∫ p(y|yi) log[ p(y|yi) / p(y|xi, θ) ] dy\n∼ log ∫ e^{sθ(y,xi)} dy − ∫ sθ(y, xi) p(y|yi) dy",
        variables: [
          { symbol: "p(y|yi)", meaning: "label-conditional ground-truth distribution modeling annotation noise and ambiguity (isotropic Gaussian in practice)" },
          { symbol: "yi", meaning: "the single annotated target state used as the sample" },
          { symbol: "θ", meaning: "network parameters" },
        ],
        intuition:
          "Instead of forcing the prediction onto the annotated box, match the whole predicted distribution to a small cloud of 'plausible correct values' around the annotation.",
        why:
          "Replaces both pseudo-label design and loss choice: p(y|yi) carries the ambiguity (centre-of-mass jumps, annotator disagreement) that ad hoc wide Gaussian pseudo-labels were imitating without justification.",
        where: "Section 3.2, Eq. 8; derived in Appendix A; the discarded constant is the negative entropy of p(y|yi).",
        params:
          "σ² of the Gaussian label distribution controls assumed label noise: too large → over-uncertain predictions, too small → overconfident overfitting (Figure 5).",
        paperIds: ["T040"],
      },
      {
        id: "prdim-grid",
        label: "Grid-sampling approximation (centre branch)",
        formula:
          "L_i = log( A · Σ_{k=1}^{K} e^{sθ(y⁽ᵏ⁾, xi)} ) − A · Σ_{k=1}^{K} sθ(y⁽ᵏ⁾, xi) · p(y⁽ᵏ⁾|yi)",
        variables: [
          { symbol: "y⁽ᵏ⁾", meaning: "uniform grid locations produced by the fully convolutional output map" },
          { symbol: "A", meaning: "area of a single grid cell" },
          { symbol: "K", meaning: "number of grid locations (= H·W)" },
        ],
        intuition:
          "The two integrals become sums over the CNN's output grid — free, because the network already evaluates every location.",
        why:
          "For 2-D centre regression the grid given by the convolution is exactly the sampling needed; the result is equivalent to a SoftMax cross-entropy loss against the label density.",
        where: "Section 3.3, Eq. 9; used for the target-centre branch (with an added L2 term inside the optimizer module, Eq. 14/16).",
        params: "A = 1 without loss of generality (Eq. 16 shows the SoftMax-cross-entropy form).",
        paperIds: ["T040"],
      },
      {
        id: "prdim-mc",
        label: "Monte Carlo importance-sampling approximation (box branch)",
        formula:
          "L_i = log( (1/K) Σ_k e^{sθ(y_i⁽ᵏ⁾, x_i)} / q(y_i⁽ᵏ⁾|yi) ) − (1/K) Σ_k sθ(y_i⁽ᵏ⁾, x_i) · p(y_i⁽ᵏ⁾|yi) / q(y_i⁽ᵏ⁾|yi)",
        variables: [
          { symbol: "y_i⁽ᵏ⁾ ∼ q(·|yi)", meaning: "samples drawn from a proposal distribution q centered at the annotation" },
          { symbol: "q", meaning: "proposal, here ½N(yi, 0.05²) + ½N(yi, 0.5²) — a Gaussian mixture covering the label cloud and high-density regions" },
          { symbol: "K", meaning: "number of Monte Carlo samples" },
        ],
        intuition:
          "For the 4-D box space a grid is infeasible, so estimate both integrals from random samples, reweighted by 1/q.",
        why:
          "The proposal must cover both the label distribution and regions where the predicted density is high; a single Gaussian centered at the annotation sufficed (following DCTD) for bounding-box regression.",
        where: "Section 3.3, Eq. 10; used for the bounding-box branch.",
        params:
          "More samples → lower variance but more forward passes; sharing the backbone feature extractor φθ(x) across all samples keeps the cost manageable.",
        paperIds: ["T040"],
      },
      {
        id: "prdim-newton",
        label: "Newton-step optimizer module (target model prediction)",
        formula:
          "wθ⁽ⁱ⁺¹⁾ = wθ⁽ⁱ⁾ − α⁽ⁱ⁾ ∇L(wθ⁽ⁱ⁾) ,   α⁽ⁱ⁾ = (∇L)ᵀ∇L / ((∇L)ᵀ H ∇L)\nH(wθ) = Σ_j γⱼ [z̃ⱼ∗]ᵀ (diag(p̂ⱼ) − p̂ⱼ p̂ⱼᵀ) [z̃ⱼ∗] + λI",
        variables: [
          { symbol: "wθ", meaning: "convolution filter weights of the centre regressor, predicted by the unrolled module" },
          { symbol: "∇L, H", meaning: "gradient and Hessian of the KL-based objective (Eqs. 19, 20) at the current estimate" },
          { symbol: "p̂ⱼ", meaning: "spatial SoftMax of the predicted scores on support sample j" },
          { symbol: "z̃ⱼ = φθ(zⱼ)", meaning: "backbone features of support image j" },
          { symbol: "γⱼ, λ", meaning: "per-sample weights and L2 regularization strength" },
        ],
        intuition:
          "Take the steepest-descent direction, then choose the step length by fitting a local quadratic (Newton) model instead of Gauss-Newton — which only works for least-squares losses.",
        why:
          "The KL loss is not least-squares, so DiMP's Gauss-Newton step length is unavailable; the SoftMax-cross-entropy structure still yields closed-form gradient and Hessian, keeping the optimizer a stack of differentiable layers.",
        where: "Section 4.2 and Appendix B, Eqs. 17–21; Algorithm 1 runs N_iter unrolled steps to produce wθ.",
        params:
          "λ regularizes for generalization to unseen frames; the step is convex in wθ for the linear predictor sθ = (wθ ∗ φθ(x))(y).",
        paperIds: ["T040"],
      },
    ],
    datasets: ["lasot", "got10k", "trackingnet", "vot", "otb", "uav123", "others"],
    metrics: ["success-auc", "norm-precision", "precision", "eao", "fps"],
    baselines: ["DiMP", "ATOM", "SiamRPN++", "UPDT", "ECO", "MDNet"],
    results: [
      "Regression-model comparison (Table 1, OTB100+NFS+UAV123, ResNet-18): full tracker L2 63.1 → robust-L2 (DiMP) 63.8 → NLL 63.0 → Ours 65.5 AUC (+1.7 over DiMP); BBR-only 63.8/65.0/65.8; TCR-only L2 64.8, NLL 63.2, Ours 65.5 (+2.3 over L2).",
      "LaSOT test: PrDiMP50 59.8 AUC vs DiMP50 56.9 (+2.9), PrDiMP18 56.4 vs DiMP18 53.2 (+3.2); normalized precision 68.8 vs DiMP50 65.0 and SiamRPN++ 56.9 (+11.9).",
      "TrackingNet test (Table 2): PrDiMP50 Success 75.8, Precision 70.4, Norm. Precision 81.6 — vs DiMP50 74.0/68.7/80.1 and SiamRPN++ 73.3/69.4/80.0.",
      "VOT2018 (Table 3): PrDiMP50 EAO 0.442, accuracy 0.618 (both best in the table), robustness 0.165 — vs DiMP50 0.440/0.597/0.153.",
      "GOT10k test (Table 4): PrDiMP50 AO 63.4, SR0.50 73.8, SR0.75 54.3 vs DiMP50 61.2/71.7/50.4 (+2.3 AO); PrDiMP18 61.2 vs DiMP18 57.9 (+3.3).",
      "UAV123 68.0 AUC (vs DiMP50 65.3, ATOM 64.2); OTB-100 69.6 (vs DiMP50 68.4); NFS 63.5 (vs DiMP50 62.0); ~40 FPS (ResNet-18), ~30 FPS (ResNet-50).",
    ],
    ablations: [
      "Four regression losses on identical DiMP-ResNet-18 (Table 1): L2, robust-L2, probabilistic NLL, and the proposed KL loss, evaluated on the whole tracker and per branch — NLL helps BBR (+1.2 AUC) but hurts TCR (63.2 vs 64.8) because it ignores label ambiguity; explicitly modeling p(y|yi) gives the best score in every configuration.",
      "Label-uncertainty sweep (Figure 5): varying σ_bb ∈ [0.015, 0.1] and σ_tc ∈ [1/12, 1/2] shows too-large σ forces over-uncertain outputs and too-small σ overfits into over-confident predictions; the proposed model beats baseline DiMP-18 across the whole range, so it is not tuned to a single σ.",
      "All state-of-the-art numbers are averaged over 5 runs to establish significance.",
    ],
    limitations: {
      authorStated: [
        "σ² of the label distribution is treated as a hyper-parameter; the paper notes it could in principle be estimated as the empirical variance of repeated annotations, but does not do so.",
        "Monte Carlo training requires multiple network evaluations per sample — mitigated only by sharing the backbone feature extractor across samples.",
      ],
      evident: [
        "The density is normalized over a finite grid or sample set, so reported probabilities are approximations of the true integral.",
        "Inherits DiMP's two-stage design: robustness of the meta-learned centre filter and the memory-update schedule still dominate failure cases; uncertainty is predicted but the tracker still commits to the argmax state.",
        "Distributions over the 4-D box cannot be inspected as a whole (the paper visualizes only centre and size slices).",
      ],
    },
    assumptions: [
      "The single annotation yi is a noisy draw from a conditional distribution p(y|yi) (isotropic Gaussian in the experiments).",
      "The network output is an unnormalized log-density whose partition function is computable by grid integral or importance sampling.",
      "Target remains inside the search region around the previous estimate; one target per sequence.",
    ],
    computation:
      "PrDiMP18 ≈ 40 FPS, PrDiMP50 ≈ 30 FPS; training 50 epochs × 1000 iterations on LaSOT/GOT10k/TrackingNet/COCO splits in PyTracking; MC box-branch training evaluates the score head K times per sample with shared backbone features.",
    relations: [
      {
        to: "T031",
        type: "builds-on",
        note:
          "Uses DiMP as the baseline tracker and converts both of its branches (meta-learned centre filter and IoU-style box head) to the probabilistic KL objective, replacing the Gauss-Newton optimizer step with a Newton expansion.",
      },
      {
        to: "T026",
        type: "builds-on",
        note:
          "The bounding-box branch keeps ATOM's IoU-Net architecture but trains it as a probabilistic density with a Gaussian label distribution instead of squared-error IoU regression.",
      },
      {
        to: "T028",
        type: "uses-as-baseline",
        note: "SiamRPN++ is compared on LaSOT, TrackingNet, GOT10k, OTB-100 and VOT2018 (e.g. LaSOT normalized precision 56.9 vs PrDiMP50 68.8).",
      },
    ],
    concepts: ["sot", "bounding-box", "deep-detection", "success-plot"],
    impact:
      "PrDiMP made predictive uncertainty a standard output of SOT trackers — later work cites its densities for failure detection and its KL-to-label-distribution training as the principled replacement for ad hoc pseudo-label design, and PrDiMP50 became a default strong baseline in single-object tracking comparisons.",
  },
  {
    id: "T041",
    arxiv: "2003.12949",
    title: "AutoTrack: Towards High-Performance Visual Tracking for UAV with Automatic Spatio-Temporal Regularization",
    shortTitle: "AutoTrack",
    year: 2020,
    authors: ["Yiming Li", "Changhong Fu", "Fangqiang Ding", "Ziyuan Huang", "Geng Lu"],
    fileName: "2003.12949v1.pdf",
    task: "single-object",
    tags: ["correlation-filter", "uav", "adaptive-regularization", "online-learning", "real-time"],
    difficulty: "intermediate",
    summary:
      "AutoTrack is a DCF tracker that sets its own regularization: the local variation of the response map becomes a spatial penalty (don't learn from unreliable pixels), and the global response variation sets the temporal penalty and update rate (stop learning when the response looks aberrant, learn faster when appearance genuinely changes). It runs at ~60 FPS on a single CPU, beats all compared CPU trackers on four UAV benchmarks, and additionally powers a vision-based UAV localization system with 3.44 cm RMSE.",
    problem:
      "DCF trackers improve themselves with regularization terms — a fixed spatial bowl (SRDCF/STRCF), a fixed temporal penalty (STRCF's θ = 15), fixed aberrance thresholds (ARCF) — whose hyper-parameters require tedious tuning and cannot react to situations the designer never saw, such as the drastic illumination changes and fast camera motion of UAV flight. Deep trackers handle those variations but are unusable on the CPU-only onboard computers of small UAVs.",
    background: ["correlation-filter", "sot", "appearance-features", "bounding-box", "success-plot"],
    previousWork: [
      {
        name: "STRCF (spatio-temporal regularized correlation filters)",
        limitation:
          "Uses a constant bowl-shaped spatial penalty borrowed from SRDCF and an unchanged temporal penalty strength (θ = 15) — neither adapts to appearance variation in unforeseeable aerial scenes, so the filter drifts once contaminated.",
        whyThisPaper:
          "Keeps STRCF's objective but replaces both fixed terms with response-driven ones: ũ is recomputed from local variation every frame and θt is optimized jointly with the filter in closed form.",
      },
      {
        name: "ARCF (aberrance-repressed correlation filters)",
        limitation:
          "Exploits only the spatially global response variation with a fixed suppression parameter, ignoring per-location credibility inside the bounding box.",
        whyThisPaper:
          "Adds the local response-variation vector for pixel-level credibility (automatic spatial regularization) and jointly optimizes the temporal hyper-parameter, running 3.1× faster than ARCF-HC with better accuracy.",
      },
      {
        name: "Deep trackers (MDNet, ADNet, SiamFC, ECO, TADT)",
        limitation:
          "Even feed-forward Siamese trackers need a high-end GPU to meet real-time needs; on a CPU-only mobile device they cannot run at the 30 FPS UAVs require.",
        whyThisPaper:
          "A hand-crafted-feature DCF with automatic regularization reaches ~60 FPS on one CPU while outperforming the deep trackers on DTB70 and UAVDT.",
      },
    ],
    researchGap:
      "Before this paper, every DCF regularizer (spatial penalty shape, temporal penalty strength, aberrance threshold) was a hand-set constant, so no tracker could automatically decide where and how fast to update its filter in unforeseen aerial scenes.",
    contribution: [
      "A novel spatio-temporal regularization term that simultaneously exploits local response variation (per-pixel credibility inside the box) and global response variation (overall update-rate control) hidden in DCF response maps.",
      "A DCF tracker that tunes the regularization hyper-parameters automatically and on the fly — the temporal strength θt is jointly optimized with the filter each frame instead of fixed at STRCF's 15.",
      "State-of-the-art evaluation on 278 difficult UAV sequences (DTB70, UAVDT, UAV123@10fps, VisDrone2018-test-dev; 119,830 frames) against both CPU- and GPU-based trackers, at ~60 FPS on a single CPU.",
      "First application of visual object tracking to UAV self-localization: four tracked markers feed a correspondence-plus-reprojection-pose solver, giving 3.44 cm RMSE over 2,666 indoor frames and working with any annotated object rather than infrared LEDs only.",
    ],
    method: {
      pipeline: ["extract", "detect", "response-variation", "auto-regularize", "admm-solve", "localize", "update"],
      architecture:
        "STRCF-style linear correlation filter over K hand-crafted feature channels, solved in the Fourier domain every frame. Two new modules wrap the filter: an automatic spatial term ũ = P> δ log(Π+1) + u built from the local response-variation vector Π (motion-aligned with a shift operator so both peaks coincide), and an automatic temporal term θt optimized jointly with the filter against a reference θ̃ derived from the global variation ‖Π‖₂. Four ADMM subproblems (G, H, θt, multiplier) are solved per frame; the response-map peak localizes the target.",
      motionModel:
        "No explicit motion model; temporal regularization on the filter itself plays the smoothness role — the filter is pulled toward last frame's solution with strength θt that shrinks when recent filter change was large.",
      appearanceModel:
        "Online-updated linear filter on hand-crafted features; local response variation gates learning per pixel (dramatic variation → spatially punished), global variation gates the overall learning rate (aberrant → freeze, large-but-plausible → learn faster).",
      detectionDependency: "None — standalone DCF, detection is the correlation response itself.",
      trackManagement:
        "No track lifecycle; an aberrance monitor acts as failure handling: when global response variation exceeds threshold φ = 3000 the correlation filter ceases learning entirely, preventing contamination of the model.",
      loss:
        "Joint objective (Eq. 5): Gaussian-fit data term + automatic spatial term (ũ-weighted) + temporal term (θt/2)·Σ‖hₜᵏ − hₜ₋₁ᵏ‖² + reference-tracking term (1/2)‖θt − θ̃‖².",
      optimization:
        "ADMM in the frequency domain with augmented Lagrangian; per-frame subproblems have closed-form solutions (Sherman-Morrison for the G subproblem; diagonal closed form for H; scalar argmin for θt); γ update γ⁽ⁱ⁺¹⁾ = min(γmax, βγ⁽ⁱ⁾) with β = 10, γmax = 10000, 4 ADMM iterations per frame; hyper-parameters δ = 0.2, ν = 2×10⁻⁵, ζ = 13, φ = 3000.",
    },
    equations: [
      {
        id: "autotrack-local-var",
        label: "Local response variation",
        formula: "Πᵢ = ( R_t[ψ∆]ᵢ − R_{t−1,ᵢ} ) / R_{t−1,ᵢ}",
        variables: [
          { symbol: "R_t, R_{t−1}", meaning: "response maps of the current and previous frame" },
          { symbol: "[ψ∆]", meaning: "shift operator aligning the two response peaks so motion does not masquerade as variation" },
          { symbol: "Πᵢ", meaning: "relative change of response at pixel i — the credibility signal" },
        ],
        intuition:
          "How much did the response at this pixel change since last frame, once both maps are slid so the targets line up? Big jumps mean the appearance there cannot be trusted.",
        why:
          "It converts the response map — which the tracker already computes — into a per-pixel reliability map, the raw material for both regularizers, without any learned detector.",
        where: "Section 4.1, Eq. 2; computed once per frame before optimization.",
        params:
          "Drastic local variation → low credibility → that filter location is spatially restricted; the alignment step removes pure motion influence.",
        paperIds: ["T041"],
      },
      {
        id: "autotrack-spatial",
        label: "Automatic spatial regularization",
        formula: "ũ = P> δ · log(Π + 1) + u",
        variables: [
          { symbol: "P>", meaning: "crop operator keeping the central filter region where the object lies" },
          { symbol: "δ", meaning: "constant weighting local variation (set to 0.2)" },
          { symbol: "u", meaning: "bowl-shaped spatial penalty inherited from STRCF/SRDCF for boundary effects" },
        ],
        intuition:
          "Add a log-scaled penalty wherever the response changed a lot: the filter stops absorbing appearance from pixels that just became unreliable.",
        why:
          "Replaces SRDCF's static bowl with a data-driven map — the paper's fix for 'fixed spatial regularization cannot address unforeseeable appearance variation'.",
        where: "Section 4.1, Eq. 3; feeds the spatial term of the joint objective (Eq. 5).",
        params: "δ controls how harshly local variation is punished; u still handles the boundary effect as before.",
        paperIds: ["T041"],
      },
      {
        id: "autotrack-temporal-ref",
        label: "Temporal reference value from global variation",
        formula: "θ̃ = ζ / (1 + log(ν·‖Π‖₂ + 1)) ,   ‖Π‖₂ ≤ φ",
        variables: [
          { symbol: "‖Π‖₂", meaning: "global (L2) response variation over the map" },
          { symbol: "ζ, ν", meaning: "hyper-parameters (13 and 2×10⁻⁵) scaling the reference" },
          { symbol: "φ", meaning: "aberrance threshold (3000): beyond it the filter stops learning" },
        ],
        intuition:
          "The bigger the global response change, the smaller the reference temporal weight — so the filter is allowed to move faster when the appearance really is changing; past a hard threshold, learning stops altogether.",
        why:
          "It is a reference, not the parameter itself: θt is then optimized around θ̃ each frame, turning STRCF's fixed θ = 15 into an adaptive quantity.",
        where: "Section 4.1, Eq. 4; used by the θt subproblem of the joint objective.",
        params:
          "Two regimes: ‖Π‖₂ > φ → freeze (gross failure); below φ → continuous loosening of the temporal restriction as variation grows.",
        paperIds: ["T041"],
      },
      {
        id: "autotrack-objective",
        label: "Joint objective (filter + temporal weight)",
        formula:
          "E(H_t, θ_t) = Σ_k ‖y − xₜᵏ ∗ hₜᵏ‖₂²/2 + Σ_k ‖ũ ⊙ hₜᵏ‖₂²/2 + (θ_t/2) Σ_k ‖hₜᵏ − hₜₜ₋₁ᵏ‖₂² + (1/2)‖θ_t − θ̃‖₂²",
        variables: [
          { symbol: "hₜᵏ", meaning: "filter of channel k at frame t" },
          { symbol: "y", meaning: "desired Gaussian-shaped response" },
          { symbol: "ũ", meaning: "automatic spatial map (Eq. 3)" },
          { symbol: "θ_t, θ̃", meaning: "optimized temporal strength and its reference (Eq. 4)" },
        ],
        intuition:
          "Fit the Gaussian response, but shrink the filter where pixels are unreliable, keep it close to last frame's filter with a strength that itself has a target value θ̃.",
        why:
          "The extra term (1/2)‖θt − θ̃‖² is what makes the hyper-parameter a learnable variable: the optimizer now chooses θt every frame instead of reading it from a config file.",
        where: "Section 4.2, Eq. 5; converted to the frequency domain (Eq. 6) and solved by ADMM.",
        params:
          "Closed-form θt update (Eq. 14): θt* = θ̃ − ½Σ_k‖ĝₜᵏ − ĝₜ₋₁ᵏ‖² — if the filter just moved a lot, the restriction eases further.",
        paperIds: ["T041"],
      },
      {
        id: "autotrack-theta-update",
        label: "Closed-form temporal weight update",
        formula: "θ_t* = argmin_θ { (θ/2) Σ_k ‖ĝₜᵏ − ĝₜ₋₁ᵏ‖₂² + (1/2)‖θ − θ̃‖₂² } = θ̃ − (1/2) Σ_k ‖ĝₜᵏ − ĝₜ₋₁ᵏ‖₂²",
        variables: [
          { symbol: "ĝₜᵏ", meaning: "Fourier-domain filter of channel k at frame t" },
          { symbol: "θ̃", meaning: "reference from global response variation" },
        ],
        intuition:
          "A one-dimensional least-squares problem: move θ away from the reference in proportion to how much the filter actually changed.",
        why:
          "Gives an exact subproblem solution inside the ADMM loop — adaptive temporal strength costs one scalar computation, not a search.",
        where: "Section 4.2, Eq. 14; one of the four per-frame ADMM subproblems.",
        params:
          "Large recent filter change → θt below θ̃ → looser temporal restriction → faster adaptation; identical filters → θt = θ̃.",
        paperIds: ["T041"],
      },
    ],
    datasets: ["uav123", "uavdt", "others"],
    metrics: ["precision", "success-auc", "fps"],
    baselines: ["STRCF", "ARCF-H", "ECO-HC", "BACF", "MCCT-H", "ASRCF"],
    results: [
      "Top-ten CPU trackers averaged over the four UAV benchmarks (Table 2): AutoTrack precision 72.4 (1st) at 59.2 FPS (2nd, behind ECO-HC 69.5 FPS) — +4.8% precision over ECO-HC, +7.9% precision and +108.5% speed over baseline STRCF (67.1 @ 28.4 FPS).",
      "Per-dataset: DTB70 precision 0.717 / success 0.479 (1st vs both deep and CPU trackers); UAVDT precision 0.718 / success 0.450; UAV123@10fps precision 0.671 (+5.8% over ECO-HC) / success 0.477; VisDrone2018 precision 0.788 (2nd to MCCT-H 0.811) / success 0.573.",
      "UAVDT deep-tracker comparison (Table 1): 71.8 precision at 65.4 FPS on CPU, ahead of GPU trackers ASRCF (70.0 @ 24.1 FPS*), ECO (70.0 @ 16.4 FPS*), SiamFC (68.1 @ 37.9 FPS*).",
      "Attribute gains over STRCF: deformation +15.9%, in-plane rotation +15.5%, large occlusion +15.7%, illumination variation +7.0%, viewpoint change +4.6% success.",
      "UAV localization (Quanser motion-capture platform, 6 datasets / 2,666 frames): mean errors x 2.17 cm, y 2.32 cm, z 1.32 cm, overall RMSE 3.44 cm.",
    ],
    ablations: [
      "Module ablation on the four-benchmark average (Table 3): STRCF 0.671 precision / 0.468 AUC / 28.4 FPS → +automatic spatial regularization 0.716 / 0.489 / 53.7 → +automatic temporal regularization 0.714 / 0.492 / 60.0 → both (AutoTrack) 0.724 / 0.495 / 59.2 — each module adds accuracy, and ATR additionally boosts speed by skipping training on contaminated samples.",
      "Qualitative ablation (Fig. 4): AutoTrack's response maps show far less distraction than STRCF's on bird1_3, car18, MountainBike5 and person12_2 because regularized pixels refuse the wrong appearance.",
    ],
    limitations: {
      authorStated: [],
      evident: [
        "Only the spatial offset and temporal weight are self-tuned: δ = 0.2, ν = 2×10⁻⁵, ζ = 13 and the aberrance gate φ = 3000 remain hand-set hyper-parameters (sensitivity analysis relegated to supplementary material).",
        "Hand-crafted features cap appearance discrimination — on VisDrone2018 the tracker trails deep MCCT-H in precision (0.788 vs 0.811).",
        "The φ-gate is binary: past threshold the filter freezes completely, so recovery depends on the response normalizing in later frames rather than on a re-detection mechanism.",
      ],
    },
    assumptions: [
      "Response-map variation is a faithful proxy for appearance credibility (drastic local change = untrustworthy pixel).",
      "Desired response is Gaussian-shaped; the target stays inside the Fourier-domain search window between frames.",
      "Hand-crafted features plus linear filtering suffice for the UAV scenes targeted (vehicles, people, small aerial objects).",
    ],
    computation:
      "~60 FPS (59.2 average) on a single CPU (i7-8700K 3.7 GHz, 32 GB RAM, MATLAB R2018a, no GPU for tracking); 4 ADMM iterations per frame; localization runs in ROS/C++ at real-time rates with an Intel RealSense R200 camera at 10 FPS.",
    relations: [
      {
        to: "T019",
        type: "builds-on",
        note:
          "Takes STRCF's objective as the baseline and replaces its two fixed hyper-parameters (bowl spatial penalty, θ = 15 temporal penalty) with response-driven, jointly optimized ones.",
      },
      { to: "T001", type: "uses-as-baseline", note: "KCF is one of the 12 real-time CPU trackers compared on all four UAV benchmarks." },
      { to: "T011", type: "uses-as-baseline", note: "ECO-HC is the closest CPU competitor (Table 2: 69.1 precision @ 69.5 FPS vs AutoTrack 72.4 @ 59.2)." },
      { to: "T012", type: "uses-as-baseline", note: "BACF is compared as a real-time CPU tracker on DTB70, UAVDT, UAV123@10fps and VisDrone." },
      { to: "T004", type: "uses-as-baseline", note: "Staple and its context-aware variant STAPLE-CA are among the compared real-time CPU trackers." },
    ],
    concepts: ["correlation-filter", "sot", "appearance-features", "online-vs-offline"],
    impact:
      "AutoTrack established response-variation-driven update gating (learn where credible, freeze on aberrance) as the recipe for lightweight aerial trackers, and its tracking-for-localization system opened the vision-based UAV self-localization application thread on top of standard SOT.",
  },
  {
    id: "T042",
    arxiv: "2004.01177",
    title: "Tracking Objects as Points",
    shortTitle: "CenterTrack",
    year: 2020,
    authors: ["Xingyi Zhou", "Vladlen Koltun", "Philipp Krähenbühl"],
    fileName: "2004.01177v2.pdf",
    task: "multi-object",
    tags: ["end-to-end-mot", "center-point", "anchor-free-head", "online", "real-time"],
    difficulty: "intermediate",
    summary:
      "CenterTrack represents every object by the centre point of its bounding box and tracks those points: a CenterNet detector is conditioned on the previous frame and a heatmap of prior tracklets, and is trained to also predict the 2-D offset from each current centre to its previous-frame centre; greedy matching on that offset does the association. One network, online and real-time, sets new state of the art on MOT17 (67.8 MOTA), KITTI (89.4 MOTA) and nuScenes monocular 3D tracking (28.3 AMOTA@0.2).",
    problem:
      "The dominant tracking-by-detection pipelines detect per frame and then link boxes in a separate stage with slow, complex association (Re-ID features, graph matching) — or discard appearance entirely. Joint trackers built on Faster R-CNN propagate identities through box-regression overlap, which assumes consecutive boxes overlap largely and breaks at low frame rates. There was no simple, fully learned, online formulation that detects and associates in one pass.",
    background: [
      "mot",
      "tracking-by-detection",
      "deep-detection",
      "data-association",
      "greedy-matching",
      "motion-model",
      "online-vs-offline",
      "mota",
    ],
    previousWork: [
      {
        name: "SORT and DeepSORT (tracking-by-detection with Kalman + matching)",
        limitation:
          "Association is a separate stage: it either discards appearance (SORT) or requires an expensive feature extractor (DeepSORT, Re-ID-based methods), and detection stays disconnected from tracking.",
        whyThisPaper:
          "Association is learned jointly with detection and costs almost nothing — a greedy lookup on the predicted centre offset, with no Re-ID network and no Hungarian step.",
      },
      {
        name: "Tracktor (T030) and other joint-detection trackers",
        limitation:
          "They reuse Faster R-CNN box regression to propagate IDs, which presumes large overlap between consecutive boxes — false at low frame rates (Tracktor then needs an external motion model).",
        whyThisPaper:
          "Prior tracklets enter the network as a point heatmap, letting it match objects anywhere in its receptive field even when the boxes do not overlap at all.",
      },
      {
        name: "CenterNet (Objects as Points) detector",
        limitation:
          "A purely single-frame detector: it cannot recover objects that are invisible or occluded in the current frame and has no temporal coherence between predictions.",
        whyThisPaper:
          "Adds just four input channels (previous frame + rendered tracklet heatmap) and two output channels (centre offset) to turn the detector into a tracker that can keep occluded objects alive from evidence at t−1.",
      },
    ],
    researchGap:
      "Before this paper, no tracker could detect and associate in a single end-to-end online network conditioned only on two frames and a point heatmap — association was either a separate expensive stage or depended on inter-frame box overlap.",
    contribution: [
      "Point-based joint detection and tracking: each object is one centre point; a heatmap of prior centres plus two-frame input conditions the detector, and a learned 2-D offset links centres across frames.",
      "Tracking-conditioned detection that reasons about occluded objects jointly with visible ones, and can even be trained on static images with 'hallucinated' motion (random scale/translate of the current frame as the previous one).",
      "Aggressive training-time noise on the prior heatmap (centre jitter λjt = 0.05, random false positives λfp, random false negatives λfn) that prevents the shortcut of copying the previous frame — without it accuracy collapses (66.1 → 34.4 MOTA).",
      "Greedy offset matching with a box-size radius instead of Hungarian/Re-ID association, keeping the whole tracker online, differentiable and real time.",
      "State of the art on three test servers: MOT17 67.8 MOTA (private), KITTI 89.4 MOTA, nuScenes 28.3 AMOTA@0.2 monocular 3D, plus an easy extension to monocular 3D via extra depth/rotation/extent heads.",
    ],
    method: {
      pipeline: ["render-heatmap", "detect", "predict-offset", "associate", "update"],
      architecture:
        "CenterNet with DLA backbone extended by 4 input channels (previous-frame image and the Gaussian-rendered, τ-thresholded class-agnostic heatmap of prior tracklet centres) and 2 output channels (offset map D̂). Outputs: centre heatmap Ŷ, box size map Ŝ, offset map D̂ (plus 4-channel amodal border distances for MOT's out-of-frame centres, and depth/rotation/extent heads for 3D). Peaks of Ŷ above θ give detections with offsets; prior heatmap renders only detections with confidence > τ.",
      motionModel:
        "Learned 2-D centre offset d̂ = p̂⁽ᵗ⁾ − p̂⁽ᵗ⁻¹⁾, analogous to sparse optical flow but conditioned on the prior detections and trained jointly — beats SORT's Kalman filter and FlowNet2 optical flow at low frame rates (Table 6).",
      association:
        "Greedy: process detections in descending confidence; each is matched to the closest unmatched prior detection at p̂ − D̂p̂ if within radius κ = min(√(ŵᵢĥᵢ), √(wⱼhⱼ)) (geometric mean of the two box sizes); otherwise a new tracklet spawns. Hungarian is evaluated and performs no better (IDF1 61.0 vs 64.2).",
      reid: "None — purely local tracker, no appearance features.",
      trackManagement:
        "Unmatched tracks become 'inactive' for K consecutive frames (rebirth, K = 32 on the MOT test set): they can regain their ID by matching later but never appear in the prior heatmap or output while inactive; K = 0 elsewhere. Rendering threshold τ gates which predictions feed the next heatmap (false-positive control).",
      detectionDependency:
        "Intrinsic — detection and tracking are the same network; in MOT public-detection mode track creation is additionally restricted to boxes near the provided public detections.",
      loss:
        "CenterNet multi-task objective (focal loss on the centre heatmap, L1 size loss, centre-refinement loss) plus L1 offset regression L_off = (1/N)Σ‖D̂⁽ᵗ⁾ − (pᵢ⁽ᵗ⁻¹⁾ − pᵢ⁽ᵗ⁾)‖; 4-D amodal size and 2-D-to-3D centre-offset L1 losses for the extensions.",
      optimization:
        "Fine-tuned from a pretrained CenterNet (copied weights for the detection pipeline; new channels random); Adam, lr 1.25e−4, batch 32, 70 epochs, lr ÷10 at epoch 60; flip / resized-crop / color-jitter plus heatmap-noise augmentation; optional CrowdHuman static-image pretraining; MOT inputs resized and padded to 960×544 (KITTI 1280×384, nuScenes 800×448).",
    },
    equations: [
      {
        id: "centertrack-render",
        label: "Gaussian heatmap rendering",
        formula: "R_q({p₀, p₁, ...}) = max_i exp( −(pᵢ − q)² / (2σᵢ²) )",
        variables: [
          { symbol: "pᵢ", meaning: "centre point of object i (ground-truth during training, prior detection at inference)" },
          { symbol: "q", meaning: "candidate output location on the low-resolution map (stride R = 4)" },
          { symbol: "σᵢ", meaning: "Gaussian kernel width, a function of object size" },
        ],
        intuition:
          "Turn a set of points into a smooth blob map: every object becomes a small hill on the heatmap the network can both read (input tracklets) and write (detections).",
        why:
          "This is what makes 'tracking as points' tractable — the whole constellation of tracked objects compresses into one channel the detector can condition on, instead of a variable-length set of boxes.",
        where: "Preliminaries (training target for Ŷ) and Section 4.1 (rendering prior tracklets H⁽ᵗ⁻¹⁾ with only confidence > τ).",
        params:
          "τ = 0.5 (MOT) gates prior tracklets into the heatmap; σ scales with object size so small objects get tight peaks.",
        paperIds: ["T042"],
      },
      {
        id: "centertrack-offset-loss",
        label: "Centre-offset regression loss",
        formula: "L_off = (1/N) Σ_{i=1}^{N} ‖ D̂⁽ᵗ⁾ − (pᵢ⁽ᵗ⁻¹⁾ − pᵢ⁽ᵗ⁾) ‖",
        variables: [
          { symbol: "D̂⁽ᵗ⁾", meaning: "predicted 2-channel offset map at frame t" },
          { symbol: "pᵢ⁽ᵗ⁾, pᵢ⁽ᵗ⁻¹⁾", meaning: "current and previous ground-truth centres of tracked object i" },
          { symbol: "N", meaning: "number of objects" },
        ],
        intuition:
          "Supervise the network to say, at every detected centre, where that object's centre was in the previous frame — a per-object displacement arrow.",
        why:
          "The arrow is all the association stage needs: subtracting it from the current centre lands exactly on the prior detection, replacing motion models and cost matrices with one regression.",
        where: "Section 4.2, Eq. 3; same L1 style as the size and location-refinement losses.",
        params:
          "Learned jointly with detection; effectiveness depends on frame rate — at 2 FPS nuScenes it contributes +10.5 AMOTA, at 25–30 FPS MOT only +0.3 MOTA (Table 4).",
        paperIds: ["T042"],
      },
      {
        id: "centertrack-greedy-cost",
        label: "Greedy association cost and gate",
        formula: "W_ij = ‖ p̂ᵢ⁽ᵗ⁾ − d̂ᵢ⁽ᵗ⁾ , pⱼ⁽ᵗ⁻¹⁾ ‖₂ ,  match if W_ij < κ ,  κ = min( √(ŵᵢĥᵢ) , √(wⱼhⱼ) )",
        variables: [
          { symbol: "p̂ᵢ⁽ᵗ⁾ − d̂ᵢ⁽ᵗ⁾", meaning: "predicted previous-frame position of detection i" },
          { symbol: "pⱼ⁽ᵗ⁻¹⁾", meaning: "centre of prior tracklet j" },
          { symbol: "κ", meaning: "radius: geometric mean of the box sizes of the two candidates" },
        ],
        intuition:
          "Walk the detections from most to least confident; each one grabs the nearest unclaimed prior position pointed at by its own offset, but only if it's within roughly one box size away.",
        why:
          "Points make this sufficient — no global optimum needed (Hungarian scores worse on IDF1) and no appearance embedding; the radius prevents wild matches when offsets are bad.",
        where: "Section 4.2 and Algorithm 1/2 (supplement); unmatched detections spawn new tracklets.",
        params:
          "κ scales with object size, so large vehicles associate over larger distances than pedestrians; θ = 0.4 output threshold and τ = 0.5 rendering threshold are the tuned knobs (Table 11).",
        paperIds: ["T042"],
      },
      {
        id: "centertrack-mota",
        label: "MOTA (evaluation)",
        formula: "MOTA = 1 − Σ_t (FP_t + FN_t + IDSW_t) / Σ_t GT_t",
        variables: [
          { symbol: "FP_t, FN_t", meaning: "false positives and missed objects in frame t" },
          { symbol: "IDSW_t", meaning: "identity switches in frame t" },
          { symbol: "GT_t", meaning: "ground-truth boxes in frame t" },
        ],
        intuition: "One number counting every tracking error relative to the number of real objects.",
        why: "The common headline metric on MOT17/KITTI; the paper also reports FP/FN/IDSW rates because MOTA is sensitive to the output threshold θ.",
        where: "Section 5.1 (metrics); reported on all 2D benchmarks.",
        params:
          "True positive needs IoU > 0.5 (2D); ranking ignores confidence, hence the per-dataset threshold search (Table 11).",
        simulator: "metrics",
        paperIds: ["T042"],
      },
      {
        id: "centertrack-amota",
        label: "AMOTA (threshold-robust 3D metric)",
        formula:
          "AMOTA = (1/(n−1)) Σ_{r ∈ {1/(n−1), ..., 1}} MOTA_r\nMOTA_r = max(0, 1 − α · (IDSW_r + FP_r + FN_r − (1−r)·P) / (r·P))",
        variables: [
          { symbol: "r", meaning: "recall threshold being evaluated" },
          { symbol: "P", meaning: "total number of annotated objects across all frames" },
          { symbol: "α", meaning: "benchmark weight (0.2 for AMOTA@0.2, 1 for AMOTA@1)" },
          { symbol: "n", meaning: "number of sampled thresholds (40)" },
        ],
        intuition:
          "Average MOTA over many confidence thresholds so the score cannot be gamed by picking one good output threshold.",
        why:
          "nuScenes adopted AMOTA because plain MOTA is threshold-sensitive; CenterTrack reports 28.3 AMOTA@0.2 against the monocular baseline's 6.9.",
        where: "Section 5.1; nuScenes evaluation over the 7 object categories.",
        params: "True positive for 3D = box-centre distance < 2 m on the ground plane.",
        simulator: "metrics",
        paperIds: ["T042"],
      },
    ],
    datasets: ["mot17", "mot16", "kitti-tracking", "others"],
    metrics: ["mota", "idf1", "motp", "idsw", "fp", "fn", "mt-ml", "fps"],
    baselines: ["Tracktor", "DeepSORT", "SORT", "3DT", "AB3D", "mmMOT"],
    results: [
      "MOT17 test (Table 1): private detection 67.8 MOTA / 64.7 IDF1 / MT 34.6 / ML 24.6 / FP 18,498 / FN 160,332 / IDSW 3,039 at 57 ms; public detection 61.5 MOTA / 59.6 IDF1 — a 5-point (8.6% relative) MOTA gain over Tracktor v2 on equal detections.",
      "KITTI test (Table 2): 89.44 MOTA / 85.05 MOTP / MT 82.31 / ML 2.31 / IDSW 116 / FRAG 334 at 82 ms with flip testing (45 ms, 88.7 MOTA without) — first place over all published entries including 3DT, mmMOT and MOTSFusion.",
      "nuScenes test (Table 3): AMOTA@0.2 27.8, AMOTA@1 4.6, AMOTP 1.5 at 45 ms vs the monocular Mapillary+AB3D baseline 6.9 / 1.8 / 1.8 (abstract reports 28.3 AMOTA@0.2, ~3× the baseline).",
      "MOT16 test (supplement, Table 7): 69.6 MOTA / 60.7 IDF1 at 57 ms — 2nd on the leaderboard behind LMP (71.0 MOTA at 2000+D ms), beating POI, KNDT and DeepSORT.",
      "Training without MOT data: CrowdHuman-only static-image pretraining reaches 52.2 MOTA on MOT validation; from-scratch 60.7 vs full pipeline 66.1.",
      "Abstract claims 22 FPS on MOT17 and 15 FPS on KITTI; nuScenes 3D tracking runs at 28 FPS.",
    ],
    ablations: [
      "Core components (Table 4, validation sets): detection-only 63.6 → +heatmap (w/o offset) 65.8 → +offset (w/o heatmap) 63.9 → full 66.1 MOTA on MOT17; KITTI 84.3 → 87.1 / 85.4 → 88.7; nuScenes 18.1 → 17.8 / 26.5 → 28.3 AMOTA@0.2 — heatmap conditioning gives ~2–3 MOTA and fewer IDSW, offsets matter most at low frame rate.",
      "Noisy-heatmap training (Table 5): removing noise injection drops MOT17 val from 66.1 to 34.4 MOTA with a huge FN rate — the model can't discover objects missing from an (initially empty) prior heatmap.",
      "Static-image training (Table 5): equals video training on MOT17 (66.1 vs 66.1, IDF1 65.4 vs 64.2) but is weaker on KITTI (86.8 vs 88.7) and nuScenes where motion is large.",
      "Matching and rebirth (Table 5): Hungarian instead of greedy does not help (same MOTA, worse IDF1 61.0 vs 64.2); rebirth K=32 keeps MOTA (66.2) while IDF1 rises to 69.4 and IDSW falls from 1.0% to 0.4% — used for the MOT test submission.",
      "Motion models (Table 6): no-motion 65.8 / Kalman (SORT's) 66.1 / FlowNet2 66.1 / learned offset 66.1 on MOT17, but 87.1 / 87.9 / 88.4 / 88.7 on KITTI and 17.8 / 18.3 / 26.6 / 28.3 on nuScenes — learned offsets beat hand-crafted and flow motion exactly where inter-frame motion is large (FlowNet2 also costs ~150 ms/pair).",
    ],
    limitations: {
      authorStated: [
        "Purely local: objects are only associated between adjacent frames; a target leaving or being occluded long-term gets a new identity — the paper explicitly trades long-range reconnection for simplicity, speed and local accuracy.",
        "IDF1 and ID-switch are weaker than offline methods such as LSST17; the paper calls combining local trackers with offline long-range models an 'exciting avenue for future work'.",
        "On low-framerate data (nuScenes) static-image training is less effective than video training.",
      ],
      evident: [
        "Error propagation through the rendered heatmap: a wrong prior peak biases the next frame's detection until τ-gating or noise augmentation suppresses it.",
        "Greedy confidence-ordered matching is not globally optimal — a confident wrong offset can steal a prior tracklet from the correct detection.",
        "Both thresholds (θ output, τ rendering) are tuned per dataset; MOTA's threshold sensitivity makes the reported numbers depend on that search (Table 11 spans 62.6–66.2 MOTA).",
      ],
    },
    assumptions: [
      "Objects persist between nearby frames (previous frame sampled with |k − t| < 3) with moderate motion the offset head can absorb.",
      "Prior tracklet centres remain informative even when consecutive boxes have zero overlap.",
      "A single greedy pass over confidence-sorted detections approximates the correct assignment well enough for online use.",
    ],
    computation:
      "MOT17: 57 ms/frame (≈22 FPS claimed in abstract; 17 FPS end-to-end including detection in private mode); KITTI: 82 ms with flip testing, 45 ms without; nuScenes: 45 ms (28 FPS claimed). DLA backbone on Titan Xp + i7-8086K; the optical-flow alternative alone would add ~150 ms per image pair.",
    relations: [
      {
        to: "T030",
        type: "uses-as-baseline",
        note:
          "Tracktor v2 is the closest MOT17 public-detection competitor (56.5 vs CenterTrack 61.5 MOTA), and CenterTrack adopts Tracktor's rule of only creating tracks near provided public detections.",
      },
      { to: "T005", type: "uses-as-baseline", note: "SORT's Kalman filter implementation is one of the three motion models compared in Table 6." },
      { to: "T013", type: "uses-as-baseline", note: "DeepSORT is compared on the MOT16 test leaderboard (61.4 MOTA vs CenterTrack 69.6)." },
      { to: "T006", type: "uses-as-baseline", note: "Evaluates on the MOT16/17 benchmark introduced by that paper, with its metrics, splits and test server." },
    ],
    concepts: ["mot", "end-to-end-mot", "learned-association", "greedy-matching", "deep-detection", "motion-model"],
    impact:
      "      CenterTrack turned 'objects as points' into a standard MOT design — heatmap-conditioned detection plus learned-offset association — and its finding that learned offsets beat Kalman/flow at low frame rates underpins the motion-predictor line of later MOT work; the recipe extends directly to monocular 3D.",
  },
  {
    id: "T043",
    arxiv: "2004.01888",
    title:
      "FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking",
    shortTitle: "FairMOT",
    year: 2020,
    authors: ["Yifu Zhang", "Chunyu Wang", "Xinggang Wang", "Wenjun Zeng", "Wenyu Liu"],
    fileName: "2004.01888v6.pdf",
    task: "multi-object",
    tags: ["one-shot-mot", "joint-detection-reid", "anchor-free-head", "center-point", "deep-reid", "online", "real-time"],
    difficulty: "intermediate",
    summary:
      "FairMOT asks why one-shot (joint detection + re-ID) trackers lag two-step pipelines and answers with three diagnosed unfairnesses — anchors, feature sharing and embedding dimension — then rebuilds the design on CenterNet: homogeneous anchor-free detection branch and a 128-channel re-ID branch whose embeddings are sampled only at object centres, balanced by uncertainty-weighted multi-task losses. Ranks first among all trackers on the 2DMOT15, MOT16, MOT17 and MOT20 test servers (73.7 MOTA on MOT17) while running at 30 FPS.",
    problem:
      "One-shot trackers such as JDE and Track R-CNN use anchors and cascade detection ahead of re-ID, so the network is biased toward detection: re-ID embeddings are learned from inaccurate proposals, one anchor may cover multiple identities and multiple anchors may cover one identity, shared features conflict across tasks, and the usual 512-d re-ID dimension steals capacity from detection. Two-step trackers stay more accurate but cannot run at video rate because they re-extract re-ID features per box with a separate network.",
    background: [
      "mot",
      "tracking-by-detection",
      "reid",
      "deep-detection",
      "appearance-features",
      "data-association",
      "online-vs-offline",
      "mota",
    ],
    previousWork: [
      {
        name: "JDE — Towards Real-Time Multi-Object Tracking",
        limitation:
          "Anchor-based YOLOv3 detector with 512-d re-ID features: anchors create identity ambiguity (one anchor ↔ multiple IDs), the tasks compete for features, and ID switches remain high (1544 on MOT16).",
        whyThisPaper:
          "Anchor-free centre sampling gives exactly one embedding per object, lower-dimensional features hurt detection less, and fairness-oriented loss balancing lifts MOT16 test from 64.4 to 74.9 MOTA.",
      },
      {
        name: "Track R-CNN (detection first, re-ID secondary)",
        limitation:
          "Cascaded Mask R-CNN: re-ID features come from ROI-Align over proposals, so training is dominated by proposal quality and re-ID is learned unfairly — IDF1 only 49.4 even with segmentation-labelled training data.",
        whyThisPaper:
          "Two homogeneous branches from the same stride-4 features with the embedding read at the detected centre remove the proposal dependence; IDF1 rises to 64.0 at 15× the speed.",
      },
      {
        name: "Two-step re-ID trackers (POI, MOTDT, MHT-DAM etc.)",
        limitation:
          "Most accurate on MOT leaderboards but slow: detection and re-ID run as separate models without shared features, per-box re-ID extraction does not scale to crowded scenes.",
        whyThisPaper:
          "A single shared-backbone network reaches the same accuracy regime at video rate (25.9–30 FPS including detection), making real deployment possible.",
      },
    ],
    researchGap:
      "Before this paper nobody had identified *why* joint detection-and-re-ID networks systematically underperform two-step trackers — the unfair treatment of re-ID through anchors, feature sharing and embedding dimension was undocumented and uncorrected.",
    contribution: [
      "Empirical demonstration that anchor-based one-shot MOT architectures cannot learn effective re-ID features (identity ambiguity, proposal-biased training) — the core 'unfairness' diagnosis.",
      "FairMOT itself: anchor-free homogeneous branches with centre-sampled re-ID, uncertainty-based multi-task balancing and multi-layer feature fusion — small design changes, large tracking gains.",
      "The rule that low-dimensional re-ID features suit MOT (few one-to-one matches per frame) unlike the re-ID task (query-vs-gallery) — 64/128-d beats 512-d on MOTA, AP and FPS.",
      "Single-image training: assign every box a unique identity so box-only datasets (CrowdHuman, COCO) pretrain the full tracker — 71.1 MOTA on MOT17 val from CrowdHuman + MOT17, no identity labels needed for pretraining.",
      "State of the art on all four public test servers (2DMOT15, MOT16, MOT17, MOT20) under the private-detector protocol, at near-video rate.",
    ],
    method: {
      pipeline: ["detect", "embed", "associate", "update"],
      architecture:
        "DLA-34 (CenterNet variant with deformable convolutions in the up-sampling modules, stride-4 output, input 1088×608 → feature map 272×152) feeds two homogeneous branches: a detection branch of heatmap / centre-offset / box-size heads (3×3 conv ×256 + 1×1), and a re-ID branch — one conv layer with 128 kernels producing E ∈ R^{128×H×W}, whose vector at the detected object centre is the embedding. Detection peaks come from 3×3 max-pooling NMS on the heatmap.",
      motionModel:
        "Kalman filter predicts tracklet locations for stage-1 association; the Mahalanobis distance is capped to infinity beyond a threshold to reject large-motion matches (following JDE).",
      association:
        "MOTDT-style hierarchical two-stage matching: stage 1 fuses appearance and motion, D = λ·D_reid + (1−λ)·D_mahal with λ = 0.98, solved by Hungarian at threshold τ₁ = 0.4; stage 2 matches leftovers by box IoU at τ₂ = 0.5. Tracklet appearance features are updated online each frame (KCF-style).",
      reid:
        "Identity classification: all training-set instances of an identity form one class; cross-entropy over K identities, using only embeddings at ground-truth/predicted object centres — never regions or anchors.",
      trackManagement:
        "Unmatched detections start new tracks; unmatched tracklets are kept for 30 frames in case they reappear. First-frame detections seed the initial tracklets.",
      detectionDependency:
        "Intrinsic (one-shot): detections and embeddings come from the same forward pass; experiments use the private-detector protocol.",
      loss:
        "Focal loss on the centre heatmap (Eq. 1), L1 losses for centre offset and box size with weight λs = 0.1 (Eq. 2), identity cross-entropy (Eq. 3), combined with homoscedastic uncertainty weights: L_total = ½(e^{−w₁}·L_detection + e^{−w₂}·L_identity + w₁ + w₂), w₁, w₂ learnable (Eqs. 4–5).",
      optimization:
        "Initialized from COCO-pretrained DLA-34; Adam, lr 10⁻⁴ decaying to 10⁻⁵ at epoch 20, 30 epochs, batch 12; rotation / scale / color-jitter augmentation; ~30 h on two RTX 2080 Ti. Single-image pretraining treats every box as its own class on box-only images.",
    },
    equations: [
      {
        id: "fairmot-focal",
        label: "Heatmap focal loss",
        formula:
          "L_heat = −(1/N) Σ_{xy} { (1−M̂_xy)^α log(M̂_xy)  if M_xy = 1\n                (1−M_xy)^β (M̂_xy)^α log(1−M̂_xy)  otherwise }",
        variables: [
          { symbol: "M̂_xy", meaning: "estimated objectness heatmap value at (x, y)" },
          { symbol: "M_xy", meaning: "ground-truth Gaussian-rendered heatmap (1 at object centres)" },
          { symbol: "α, β", meaning: "focal-loss hyperparameters (from CenterNet)" },
        ],
        intuition:
          "Punish the network mostly for missing the true centre peak and for confident false peaks, ignoring the vast sea of easy background pixels.",
        why: "Standard CenterNet detection objective; keeps the detection half of the multi-task loss well-behaved alongside the identity loss.",
        where: "Section 4.2.1, Eq. 1; part of L_detection = L_heat + L_box (Eq. 4).",
        params: "N = number of objects in the image; rendered with σ_c per object size.",
        paperIds: ["T043"],
      },
      {
        id: "fairmot-box-l1",
        label: "Box offset and size loss",
        formula: "L_box = Σ_i ‖ o_i − ô_i ‖₁ + λ_s ‖ s_i − ŝ_i ‖₁",
        variables: [
          { symbol: "o_i", meaning: "ground-truth sub-pixel centre offset on the stride-4 map" },
          { symbol: "s_i", meaning: "ground-truth box size (width, height)" },
          { symbol: "λ_s", meaning: "size-loss weight, set to 0.1 as in CenterNet" },
        ],
        intuition: "At every positive centre pixel, regress the fine position and the box extents with L1 penalties.",
        why: "Corrects the up-to-4-pixel quantization error of the stride-4 map so boxes (and hence IoU stage-2 matching) stay accurate.",
        where: "Section 4.2.2, Eq. 2.",
        params: "λ_s = 0.1.",
        paperIds: ["T043"],
      },
      {
        id: "fairmot-identity-ce",
        label: "Identity classification loss",
        formula: "L_identity = − Σ_{i=1}^{N} Σ_{k=1}^{K} L^i(k) · log(p(k))",
        variables: [
          { symbol: "p(k)", meaning: "softmax probability of identity class k from the FC layer on the centre embedding" },
          { symbol: "L^i(k)", meaning: "one-hot ground-truth identity of object i" },
          { symbol: "K", meaning: "total number of identities in the training data" },
        ],
        intuition:
          "Treat re-ID as classification: the embedding at an object's centre must name that person among all training identities — pulling same-identity embeddings together and pushing others apart.",
        why:
          "Only centre embeddings participate, so the gradient teaches exactly the features association will read at inference, with no anchor/ROI contamination.",
        where: "Section 4.3.1, Eq. 3.",
        params: "K classes closed over the training set; generalizes to unseen test identities at inference.",
        paperIds: ["T043"],
      },
      {
        id: "fairmot-uncertainty",
        label: "Uncertainty-weighted multi-task total loss",
        formula: "L_total = ½ ( e^{−w₁}·L_detection + e^{−w₂}·L_identity + w₁ + w₂ ),  L_detection = L_heat + L_box",
        variables: [
          { symbol: "w₁, w₂", meaning: "learnable log-variance parameters for detection and re-ID" },
          { symbol: "L_detection, L_identity", meaning: "the two task losses (Eqs. 1–2 and Eq. 3)" },
        ],
        intuition:
          "Each task's loss is automatically down-weighted when its uncertainty is high; the linear terms stop the trivial zero-loss solution.",
        why:
          "Fixes the fairness problem directly: fixed weights bias the model toward detection (best MOTA, worst TPR), MGDA-UB over-corrects toward re-ID (worst MOTA); uncertainty lands in between.",
        where: "Section 4.4, Eqs. 4–5 (Kendall et al. uncertainty weighting); chosen over GradNorm, which scores slightly higher IDF1 but trains longer.",
        params:
          "Ablated alternatives: fixed grid-search weights (69.6 MOTA / 71.6 IDF1), GradNorm (69.5 / 73.8), MGDA-UB (63.6 / 67.9); chosen setting 69.1 / 72.8.",
        paperIds: ["T043"],
      },
      {
        id: "fairmot-fused-distance",
        label: "Fused appearance-motion association distance",
        formula: "D = λ·D_reid + (1−λ)·D_mahalanobis ,  λ = 0.98",
        variables: [
          { symbol: "D_reid", meaning: "cosine distance between re-ID embeddings of detection and tracklet" },
          { symbol: "D_mahalanobis", meaning: "Kalman-filter Mahalanobis distance between predicted and detected boxes" },
          { symbol: "λ", meaning: "fusion weight, set to 0.98 — appearance dominates" },
        ],
        intuition:
          "Almost trust appearance alone, keeping a sliver of motion to break ties and reject impossible jumps (Mahalanobis set to ∞ past a threshold).",
        why:
          "With fair, centre-sampled embeddings the appearance term is strong enough to lead; Table 7 shows appearance-only already beats IoU-only, and adding Kalman trims ID switches further.",
        where: "Section 4.5.2 (stage-1 matching before Hungarian at τ₁ = 0.4); style follows MOTDT/JDE.",
        params: "λ = 0.98; τ₁ = 0.4 (stage 1), τ₂ = 0.5 (IoU stage 2).",
        paperIds: ["T043"],
      },
    ],
    datasets: ["mot15", "mot16", "mot17", "mot20", "others"],
    metrics: ["mota", "idf1", "idsw", "fp", "fn", "mt-ml", "fps"],
    baselines: ["JDE", "Track R-CNN", "CenterTrack", "DeepSORT", "SORT", "POI"],
    results: [
      "MOT17 test (private, Table 10): 73.7 MOTA / 72.3 IDF1 / MT 43.2% / ML 17.3% / IDSW 3303 at 25.9 FPS — first among all trackers, ahead of CenterTrack (67.8 / 64.7), CTracker (66.6), POI (66.1).",
      "MOT16 test: 74.9 MOTA / 72.8 IDF1 / IDSW 1074 at 25.9 FPS, beating JDE (64.4 / 55.8), TAP (64.8 / 73.5 IDF1), CNNMTT (65.2) and DeepSORT (61.4 / 62.2).",
      "MOT15 test: 60.6 MOTA / 64.7 IDF1; MOT20 test: 61.8 MOTA / 67.3 IDF1 / MT 68.8% / ML 7.6% / IDSW 5243 at 13.2 FPS — first on both leaderboards.",
      "Training-data ablation (MOT17 test, Table 11): MOT17-only 69.8 MOTA → +MIX 72.9 → +CrowdHuman single-image pretraining 73.7 with fewer IDSW (3996 → 3303).",
      "Versus Track R-CNN on the 4 segmentation-labelled MOT17 videos: IDF1 64.0 vs 49.4, IDSW 96 vs 294, 30.9 vs 2.0 FPS; versus JDE on 2DMOT15 (MIX): 77.2 vs 67.5 MOTA, IDSW 80 vs 218.",
      "Abstract claim: ~30 FPS on a single RTX 2080 Ti including detection and association.",
    ],
    ablations: [
      "re-ID feature sampling (Table 1, MOT17 val): Centre 69.1 MOTA / 72.8 IDF1 / TPR 94.4 beats ROI-Align (Track R-CNN style, 68.7 / 71.0 / 93.1), POS-Anchor (JDE style, 69.0 / 70.3 / TPR 93.9, 434 IDSW) and two-stage ROI-Align (69.0 / 68.2 / TPR 90.5); Centre-BI pushes TPR to 94.9 and IDF1 to 74.3.",
      "Loss balancing (Table 2): Fixed weights get best MOTA 69.6 but worst TPR 93.8 and 387 IDSW (detection bias); MGDA-UB the reverse (63.6 MOTA, TPR 97.0); GradNorm 69.5 / 73.8 but slower; Uncertainty-task (69.1 / 72.8) selected.",
      "Backbones (Table 3): ResNet-34 63.6 → ResNet-50 63.7 (bigger net does not help; TPR 90.9 → 91.9 but IDSW 435 → 501); ResNet-34-FPN 64.4 with TPR 94.2; DLA-34 69.1 / 72.8; HarDNet-85 71.2 / 74.5 / 198 IDSW. Multi-layer feature fusion, not capacity, fixes the feature conflict; replacing DLA's deformable convs drops it to 65.0 MOTA / 78.1 AP.",
      "Feature conflict proof (Table 4): training detection only (ResNet-34-det) raises AP 75.1 → 76.1 while TPR collapses 90.9 → 36.7 — the two tasks genuinely compete; DLA fusion relieves it (69.1 / TPR 94.4).",
      "Feature dimension (Table 6): FairMOT 64-d beats 512-d on MOTA (69.2 vs 68.5), AP (81.3 vs 80.9) and FPS (26.8 vs 24.1) despite slightly lower IDF1/TPR; JDE 64-d beats 512-d on every column — a generic one-shot rule.",
      "Association ingredients (Table 7): IoU only 67.8 / 67.2 / 648 IDSW; re-ID only 68.1 / 70.3 / 435; +Kalman 68.9 / 71.8 / 342; all three 69.1 / 72.8 / 299 — appearance leads, motion smooths, IoU rescues occlusion.",
      "Single-image training (Table 8, MOT17 val): CrowdHuman-only (no identities) 64.1 MOTA / TPR 79.9; MOT17-only 67.5; CrowdHuman+MOT17 71.1 / 75.6 — the best of all, beating MIX+MOT17 69.1.",
    ],
    limitations: {
      authorStated: [
        "Runtime analysis shows re-ID matching time grows linearly with scene density because every tracklet's appearance feature must be updated each frame (detection+re-ID forward pass itself is density-insensitive).",
        "GradNorm produces slightly better tracking accuracy (IDF1 73.8 vs 72.8) but is not used because it takes longer to train — a cost/accuracy trade-off acknowledged in Section 5.3.2.",
      ],
      evident: [
        "MOT20's 5243 ID switches (vs 3303 on MOT17) shows centre-sampled appearance alone still struggles in extreme crowds — the densest benchmark is where the design is weakest.",
        "Association thresholds (λ = 0.98, τ₁ = 0.4, τ₂ = 0.5, 30-frame persistence) are fixed hand-tuned constants, not learned.",
        "The 'fairness' argument is empirical (Tables 1–4) rather than a formal characterization of task interference.",
      ],
    },
    assumptions: [
      "Heatmap peaks align closely enough with true object centres that the centre embedding represents the instance (Centre sampling).",
      "Identities can be enumerated as K training classes with cross-entropy yet transfer to unseen test identities.",
      "Adjacent-frame motion stays within the Kalman/Mahalanobis gate so appearance + motion stage-1 matching suffices before IoU stage 2.",
    ],
    computation:
      "≈30 FPS on one RTX 2080 Ti end-to-end (paper claim); test-server FPS 25.9 (MOT15/16/17) and 13.2 (MOT20, larger frames). Joint detection+re-ID forward pass is roughly density-independent; Kalman and IoU matching cost ~1–2 ms; re-ID matching scales linearly with crowd density. Training: ~30 h on two 2080 Ti, batch 12, 30 epochs.",
    relations: [
      { to: "T042", type: "uses-as-baseline", note: "CenterTrack is the direct MOT17 competitor in Table 10 (67.8 MOTA / 64.7 IDF1 vs FairMOT 73.7 / 72.3); discussed in related work as centre-based but appearance-free." },
      { to: "T039", type: "uses-as-baseline", note: "Evaluated on the MOT20 benchmark introduced by that paper, ranking first on its leaderboard (61.8 MOTA / 67.3 IDF1)." },
      { to: "T013", type: "uses-as-baseline", note: "DeepSORT is both compared (MOT16 test 61.4 MOTA) and its Mahalanobis distance formulation is reused inside stage-1 association." },
      { to: "T005", type: "uses-as-baseline", note: "SORT (with HPD detections) appears as a baseline in Table 10 on MOT16 (59.8 MOTA); IoU-only association in Table 7 approximates its setting." },
    ],
    concepts: ["mot", "end-to-end-mot", "reid", "learned-association", "data-association", "online-vs-offline"],
    impact:
      "      FairMOT became the default strong real-time baseline on the MOT15/16/17/20 leaderboards and the reference one-shot design — centre-sampled embeddings, the fairness analysis and the low-dimension-for-MOT rule shaped subsequent joint detection-tracking work, while single-image training on box-only data became a standard pretraining recipe.",
  },
  {
    id: "T044",
    arxiv: "2004.06711",
    title: "Deformable Siamese Attention Networks for Visual Object Tracking",
    shortTitle: "SiamAttn",
    year: 2020,
    authors: ["Yuechen Yu", "Yilei Xiong", "Weilin Huang", "Matthew R. Scott"],
    fileName: "2004.06711v2.pdf",
    task: "single-object",
    tags: ["siamese-tracker", "self-attention", "cross-attention", "deformable-conv", "mask-refinement"],
    difficulty: "intermediate",
    summary:
      "SiamAttn injects a deformable Siamese attention module into a ResNet-50 SiamRPN++ tracker: spatial and channel self-attention strengthen each branch's context while cross-attention lets template and search features condition each other — an implicit template update without any online model update — and a lightweight region refinement module predicts box and mask for the single top proposal. State of the art on six benchmarks: EAO 0.464 → 0.537 on VOT2016 and 0.415 → 0.470 on VOT2018, at 33–45 FPS.",
    problem:
      "Siamese trackers are trained fully offline, so the target template is frozen after frame one and drifts under large appearance change, deformation and occlusion. Worse, the template and search branches compute features independently: background context that would separate the target from distractors is discarded, and prior attention trackers (RASNet, FlowTrack) still compute the two branches' attentions separately.",
    background: ["sot", "siamese", "attention", "rpn-head", "deep-detection", "occlusion", "success-plot"],
    previousWork: [
      {
        name: "SiamRPN++ (the baseline)",
        limitation:
          "Fixed offline template and independently computed branch features limit discriminability against close distractors, complex backgrounds and large appearance variations — EAO capped at 0.464 (VOT2016) / 0.415 (VOT2018).",
        whyThisPaper:
          "The DSA module modulates the last three stages of both branches with joint self- and cross-attention, lifting EAO to 0.537 / 0.470 while keeping the same ResNet-50, depth-wise correlation and multi-layer aggregation.",
      },
      {
        name: "Online template-update trackers (MLT, UpdateNet, GradNet)",
        limitation:
          "They bolt an explicit update network onto the fixed template — extra training and machinery, yet still no background context flowing between the branches.",
        whyThisPaper:
          "Cross-attention encodes the search image's contextual interdependencies into the template representation each frame — an implicit, parameter-light template refresh inside the matching itself.",
      },
      {
        name: "Siamese attention trackers (RASNet, FlowTrack)",
        limitation:
          "Attention is computed separately for template and search (RASNet uses only template information), so the two branches never exchange evidence and the architecture's potential is capped.",
        whyThisPaper:
          "Deformable cross-attention makes the branches collaborate directly, plus deformable conv/pooling sample features at variable locations to handle non-rigid deformation.",
      },
    ],
    researchGap:
      "Siamese trackers had neither context-aware template features nor any mechanism for the two branches to attend to each other; nobody had combined self-attention, cross-attention and deformable sampling inside the Siamese matching pipeline.",
    contribution: [
      "Deformable Siamese attention (DSA): deformable self-attention (spatial + channel) and deformable cross-attention computed jointly over template and search features.",
      "Region refinement module: depth-wise cross-correlations across stages, fusion, deformable RoI pooling, then a lightweight single-region box head and mask head — unlike ATOM/SiamMask it refines only one proposal.",
      "New state of the art on OTB-2015, UAV123, VOT2016, VOT2018, LaSOT and TrackingNet with real-time speed (45 FPS box mode, 33 FPS with mask-based rotated boxes).",
    ],
    method: {
      pipeline: ["extract", "attend", "propose", "refine"],
      architecture:
        "Five-stage ResNet-50 (dilated last two blocks → effective stride 8; last three blocks reduced to 256 channels via 1×1 conv). Stage features of both branches are modulated by the DSA module into two-stream attentional features; three SiamRPN blocks (depth-wise cross-correlation, cls + reg heads) produce three response maps combined by weighted sum; the single highest-score region goes to the region refinement module — depth-wise cross-correlations of the two-stream features across stages → fusion block (spatial/channel alignment by up/down-sampling + 1×1 conv, element-wise sum, first-two-stage features added for mask detail) → deformable RoI pooling → box head (two FC-512, input 4×4, outputs t = (tx, ty, tw, th) at 25×25 feature level) and mask head (four conv + deconv, input 16×16, outputs 64×64 class-agnostic binary mask at 64×64 feature level). No classification head — tracking is class-agnostic.",
      motionModel:
        "None — localization uses the SiamFC-style inference penalties (cosine window penalty, scale-change penalty) plus linear interpolation of the state; no explicit motion prediction.",
      association: "Not applicable (single-object tracking: match template against search region only).",
      reid: "None — appearance is handled entirely by the Siamese matching and attentional features.",
      trackManagement:
        "Single target persisted across frames by the interpolated bounding-box state; the template itself is never replaced (cross-attention supplies per-frame adaptation).",
      detectionDependency: "Not applicable.",
      loss:
        "L = L_rpn-cls + λ₁·L_rpn-reg + λ₂·L_refine-box + λ₃·L_refine-mask with λ₁ = 0.2, λ₂ = 0.2, λ₃ = 0.1; negative log-likelihood for RPN classification, smooth L1 for RPN and refinement box regression, binary cross-entropy for refinement mask.",
      optimization:
        "SGD momentum 0.9, weight decay 1e-5, 20 epochs, batch 12; learning-rate warm-up then exponential decay 5e-3 → 5e-4 over the last 15 epochs; backbone frozen for the first 10 epochs and trained with 20× smaller lr afterwards. Trained on COCO + YouTube-VOS + LaSOT + TrackingNet (ImageNet-pretrained backbone); exemplar 127×127, search 255×255; anchors: 5 aspect ratios [0.33, 0.5, 1, 2, 3], positive if IoU > 0.6, negative if IoU < 0.3; 16 regions with IoU > 0.5 per image train the refinement module.",
    },
    equations: [
      {
        id: "siamattn-spatial-self-attn",
        label: "Spatial self-attention map",
        formula: "A_ss = softmax_col( Q̄ K̄ᵀ ) ∈ R^{N×N} ,  N = H·W ,  C′ = C/8",
        variables: [
          { symbol: "Q, K", meaning: "query/key features from two 1×1 convolutions on the input feature map X ∈ R^{C×H×W}, reshaped to R^{C′×N}" },
          { symbol: "N", meaning: "number of spatial positions (H·W)" },
          { symbol: "A_ss", meaning: "pairwise spatial affinity, column-wise softmax normalized" },
        ],
        intuition:
          "Every location looks at every other location and reweights them — recovering global context that the convolutional receptive field cannot see.",
        why:
          "Self-attention over positions lets the tracker model relationships like target-vs-distractor co-occurrence; applied separately within each branch (spatial + channel variants).",
        where: "Section 3.2, Eq. 1 (self-attention sub-module, spatial branch); channel variant computed analogously as softmax_row(Z̄ Z̄ᵀ).",
        params: "Channels reduced to C′ = C/8 for the spatial path; value features V̄ from a 1×1 conv.",
        paperIds: ["T044"],
      },
      {
        id: "siamattn-spatial-residual",
        label: "Residual attentional feature update",
        formula: "X̄ˢ = α · V̄ · A_ss + X̄ ∈ R^{C×N}",
        variables: [
          { symbol: "V̄", meaning: "value features (1×1 conv on X, reshaped)" },
          { symbol: "α", meaning: "learnable scalar gate on the attention contribution" },
          { symbol: "X̄", meaning: "input features, kept via residual connection" },
        ],
        intuition:
          "Blend the context-weighted summary back into the original features — attention refines rather than replaces.",
        why:
          "Residual blending preserves the strong local descriptors Siamese matching relies on while adding context; α lets training learn how much attention to trust.",
        where: "Section 3.2, Eq. 2; output reshaped back to R^{C×H×W} as the spatial self-attentional features.",
        params: "α scalar; channel self-attentional features combined with this by element-wise sum → Xˢ.",
        paperIds: ["T044"],
      },
      {
        id: "siamattn-cross-attn",
        label: "Cross-attention: encoding template context into search features",
        formula: "A_c = softmax_row( Z̄ Z̄ᵀ ) ∈ R^{C×C} ;  X̄ᶜ = γ · A_c · X̄ + X̄ ∈ R^{C×N}",
        variables: [
          { symbol: "Z̄", meaning: "reshaped template features Z ∈ R^{C×h×w} → R^{C×n}, n = h·w" },
          { symbol: "A_c", meaning: "channel attention computed from the target branch (row-wise softmax)" },
          { symbol: "X̄", meaning: "reshaped search features (R^{C×N}, N = H·W)" },
          { symbol: "γ", meaning: "learnable scalar gating the cross-branch contribution" },
        ],
        intuition:
          "The template tells the search branch which channels matter for the current target, and the search branch's context flows back into the template representation — mutual conditioning instead of independent processing.",
        why:
          "This is the paper's implicit template update: discriminability against distractors improves (confidence maps in Fig. 4 focus on the true object) without any online model-update machinery.",
        where: "Section 3.2, Eqs. 3–4; self- and cross-attentional features of each branch are added element-wise to form the final attentional features.",
        params: "γ scalar; cross-attention computed bidirectionally (search from template, template from search).",
        paperIds: ["T044"],
      },
      {
        id: "siamattn-total-loss",
        label: "Joint RPN + region-refinement loss",
        formula: "L = L_rpn-cls + λ₁·L_rpn-reg + λ₂·L_refine-box + λ₃·L_refine-mask",
        variables: [
          { symbol: "L_rpn-cls, L_rpn-reg", meaning: "negative log-likelihood classification and smooth-L1 regression losses in the Siamese RPN blocks" },
          { symbol: "L_refine-box, L_refine-mask", meaning: "smooth-L1 box regression and binary cross-entropy mask losses in the refinement module" },
          { symbol: "λ₁, λ₂, λ₃", meaning: "task-balance weights" },
        ],
        intuition: "One end-to-end objective couples proposal scoring, localization refinement and mask segmentation.",
        why:
          "Joint supervision lets the refinement head specialize on the attentional features — ablating refinement drops EAO by +2.2% relative to baseline alone (Table 5).",
        where: "Section 3.4, Eq. 5.",
        params: "λ₁ = 0.2, λ₂ = 0.2, λ₃ = 0.1 (empirically set).",
        paperIds: ["T044"],
      },
    ],
    datasets: ["otb", "vot", "uav123", "lasot", "trackingnet"],
    metrics: ["precision", "success-auc", "eao", "norm-precision", "fps"],
    baselines: ["SiamRPN++", "SiamMask", "DiMP-50", "ATOM", "DaSiamRPN", "SiamFC"],
    results: [
      "VOT2016 (Table 1): A 0.68 / R 0.14 / EAO 0.537 — best on every metric, +7.3% EAO over SiamRPN++ (0.464) and +9.5% over SiamMask-Opt (0.442).",
      "VOT2018: A 0.63 / R 0.16 / EAO 0.470, top EAO, +5.5% over SiamRPN++ (0.415) and +3.0% over DiMP-50 (0.440).",
      "OTB-2015 (Fig. 5): best precision and success among compared methods — plots show 0.926 precision / 0.712 success AUC (the prose reports the pair transposed), ahead of SiamRPN++ (0.914 / 0.696) and DiMP-50 (0.899 / 0.684).",
      "UAV123 (Table 2): precision 0.845 (from SiamRPN++ 0.807), AUC 0.650 — comparable to DiMP-50 (0.654).",
      "LaSOT (Table 3): success 56.0 / normalized precision 64.8 vs SiamRPN++ 49.5 / 56.9 — largest normalized-precision jump in the table; TrackingNet (Table 4): success 75.2 / norm-precision 81.7, +1.2% / +1.6% over DiMP-50.",
      "Speed: 45 FPS with axis-aligned boxes (OTB/UAV/LaSOT/TrackingNet), 33 FPS on VOT where rotated boxes are extracted from the predicted mask (RTX 2080 Ti, PyTorch).",
    ],
    ablations: [
      "Component build-up on VOT2016 (Table 5, SiamRPN++ baseline 0.464 EAO): +mask learning 0.477 (+1.3%), +region refinement 0.486 (+2.2%), +self-attention 0.511 (+4.7%), +cross-attention 0.513 (+4.9%), full model 0.537 (+7.3%) — cross-attention alone contributes slightly more than self-attention.",
      "Deformable layers (Table 6): replacing deformable conv → EAO 0.520, deformable pooling → 0.531, both → 0.516; the model still beats the 0.464 baseline without any deformable layer, so attention + refinement are the primary contributors and deformable sampling adds the last +0.021.",
      "Training data (Table 7): recent large-scale sets (COCO + YouTube-VOS + LaSOT + TrackingNet) give EAO 0.537 vs the SiamMask-style mix (VID + YTB-BB + COCO + DET + YTB-VOS) 0.525 — +1.2% EAO, and the design stays SOTA either way.",
    ],
    limitations: {
      authorStated: [
        "No explicit limitations section; the paper does note that SiamMask-Opt attains the best VOT2018 accuracy only by optimizing a rotated rectangle from the mask, which cuts its speed to 5 FPS — a cost SiamAttn avoids at slightly lower accuracy.",
      ],
      evident: [
        "The template is still frozen from frame one — cross-attention injects current search context but the model never stores or refreshes an explicit template bank for long-term appearance drift.",
        "Only the single top-scoring SiamRPN proposal is refined, so refinement cannot rescue a target the proposal stage misranks.",
        "Deformable layers contribute only +0.021 EAO of the +0.073 total gain — the headline 'deformable' mechanism is secondary to plain attention.",
      ],
    },
    assumptions: [
      "The first-frame template (127×127) plus the current search region (255×255) carry enough information to re-localize the target every frame — offline training needs no online supervision.",
      "Channel maps of high-level features are class-selective, so channel (re)weighting can specialize the representation to the single tracked class despite the task being class-agnostic.",
      "The top-1 proposal from the SiamRPN blocks is correct enough that refining only it suffices.",
    ],
    computation:
      "45 FPS for axis-aligned box tracking, 33 FPS on VOT with mask-derived rotated boxes (RTX 2080 Ti, PyTorch). Training: 20 epochs, batch 12, backbone frozen for the first 10 epochs with 20× smaller learning rate; effective backbone stride reduced to 8 (dilated last two blocks) and channels cut to 256 to keep the dense attention affordable.",
    relations: [
      {
        to: "T049",
        type: "uses-as-baseline",
        note:
          "Trains on the LaSOT training set and evaluates on the LaSOT test benchmark introduced by that paper (56.0 success / 64.8 normalized precision, Table 3).",
      },
    ],
    concepts: ["sot", "siamese", "attention", "occlusion"],
    impact:
      "      SiamAttn showed that cross-branch attention inside a Siamese tracker substitutes for explicit template updating — the implicit-template-update idea and the single-region refinement pattern fed into later attention-based SOT designs, and its EAO figures became the VOT2016/2018 numbers to beat for real-time trackers.",
  },
  {
    id: "T045",
    arxiv: "2006.06664",
    title: "Quasi-Dense Similarity Learning for Multiple Object Tracking",
    shortTitle: "QDTrack",
    year: 2020,
    authors: ["Jiangmiao Pang", "Linlu Qiu", "Xia Li", "Haofeng Chen", "Qi Li", "Trevor Darrell", "a.o."],
    fileName: "2006.06664v4.pdf",
    task: "multi-object",
    tags: ["contrastive-learning", "appearance-features", "tracking-by-detection", "bi-directional-softmax", "end-to-end-mot"],
    difficulty: "intermediate",
    summary:
      "QDTrack argues MOT similarity learning wastes supervision on sparse ground-truth pairs: quasi-dense similarity learning instead densely matches hundreds of RPN RoIs across a pair of frames with a multi-positive contrastive objective, then associates objects by plain nearest-neighbour search in that embedding space — no motion models, no displacement regression. Bi-directional softmax, backdrops and duplicate removal close the loop: 68.7 MOTA on MOT17 at 20.3 FPS without external data, and new records on BDD100K, Waymo and TAO.",
    problem:
      "Existing trackers learn appearance from only the handful of ground-truth boxes (n-class identity classification or triplet loss), leaving most image regions unused, so embeddings are too weak to distinguish instances and appearance is demoted to a secondary cue with local search windows. Meanwhile IoU/centre-proximity heuristics break in crowds and occlusions, and motion/displacement machinery (Kalman, optical flow, D&T-style regression) adds complexity — while humans can associate objects from appearance alone.",
    background: [
      "mot",
      "tracking-by-detection",
      "appearance-features",
      "reid",
      "data-association",
      "motion-model",
      "occlusion",
      "mota",
    ],
    previousWork: [
      {
        name: "Location/motion-cue trackers (SORT, IoU trackers, Tracktor, D&T, CenterTrack)",
        limitation:
          "Spatial proximity and motion priors only work in simple scenes — crowds, occlusions and low frame rates defeat them — and displacement-regression methods still bolt on a separate re-ID model to re-identify vanished objects.",
        whyThisPaper:
          "Table 5 shows appearance alone (36.6 mMOTA) beats IoU+motion+regression (28.6) and that adding those cues to appearance changes nothing — a well-learned embedding space makes association a nearest-neighbour lookup.",
      },
      {
        name: "Sparse similarity learning (DeepSORT-style identity classification, triplet re-ID)",
        limitation:
          "Only sparse GT pairs supervise the embedding: classification scales poorly with the number of identities, and triplet loss compares each sample against just two others — 'rudimentary training samples and objectives leave instance similarity learning not fully explored'.",
        whyThisPaper:
          "Quasi-dense matching gives each sample 128 key-frame and 256 reference-frame RoIs as contrastive targets with multiple positives — +4.8 IDF1 on BDD100K over sparse matching, mostly from extra negative pairs.",
      },
      {
        name: "Joint embedding-head trackers (JDE, FairMOT, RetinaTrack)",
        limitation:
          "They add an embedding head to the detector but still train it image-retrieval style (classification or cosine on sparse pairs), and continue to rely on motion models and displacement predictions for association.",
        whyThisPaper:
          "One lightweight 4conv-1fc 256-d head trained quasi-densely is sufficient: QDTrack drops every motion prior and still outperforms these methods with a simpler pipeline.",
      },
    ],
    researchGap:
      "Nobody had densely matched region proposals for MOT similarity learning — nor shown that the resulting feature space removes the need for motion models, displacement regression and duplicate re-ID modules entirely.",
    contribution: [
      "Quasi-dense similarity learning: hundreds of RoIs on frame pairs matched densely, optimized with a multi-positive extension of the non-parametric softmax contrastive loss plus an L2 auxiliary loss.",
      "Inference recipe that makes pure-appearance association work: bi-directional softmax for one-to-one consistent matches, backdrops (unmatched last-frame objects) to cut false positives, and inter-class duplicate removal.",
      "QDTrack itself: Faster R-CNN + embedding head, end-to-end trainable, SOTA on four benchmarks (MOT16/17, BDD100K, Waymo, TAO) with no bells and whistles.",
      "Extension to instance-segmentation tracking on BDD100K (mMOTSA 30.8 vs SORT 12.8), demonstrating the generality of quasi-dense similarity.",
    ],
    method: {
      pipeline: ["detect", "embed", "associate", "update"],
      architecture:
        "Faster R-CNN + FPN with ResNet-50 backbone (ResNet-101 + deformable convs for Waymo); an extra lightweight embedding head (4 conv + 1 fc with group normalization, 256-d output) runs parallel to the bounding-box head; RoI Align pulls features from the matching FPN level by scale. Training samples: 128 RoIs from the key frame, 256 RoIs from the reference frame (positive:negative = 1.0, IoU-balanced sampling), reference frame sampled within k ∈ [−3, 3] frames; an RoI is positive to an object at IoU > α₁ = 0.7, negative at IoU < α₂ = 0.3.",
      motionModel: "None — association is purely appearance-based nearest-neighbour search (the paper's explicit design choice).",
      association:
        "Bi-directional softmax: f(i,j) = [softmax over rows (det-to-candidate) + softmax over columns (candidate-to-det)] / 2 on cosine similarities between the N current detections and M matching candidates (tracks of the past frames + vanished tracks + backdrops); match if f exceeds match_score_thr = 0.5 and categories agree. Inter-class duplicate removal via NMS (IoU 0.7 for score > 0.5, 0.3 for score ≤ 0.5).",
      reid:
        "256-d embeddings trained contrastively; track embeddings updated with momentum E ← m·E₁ + (1−m)·E₀, m = 0.8 (Table 10 shows the value barely matters).",
      trackManagement:
        "New track if detection confidence > init_score_thr 0.8; track continues while obj_score_thr 0.5 holds; tracks kept memo_tracklet_frames = 10 frames; backdrops kept 1 frame only; vanished tracks stay in the candidate pool for re-identification.",
      detectionDependency:
        "Loosely coupled: detection and embedding are trained jointly (one multi-task loss) but association operates on the detector's box outputs, and the method 'can be easily coupled with most existing detectors'.",
      loss:
        "Detection: L_det = L_rpn + λ₁L_cls + λ₂L_reg (Faster R-CNN losses, λ₁ = λ₂ = 1.0). Embedding: multi-positive non-parametric softmax contrastive loss (Eqs. 2–5) + L2 auxiliary loss constraining logit magnitude and cosine similarity (Eq. 6). Total: L = L_det + γ₁L_embed + γ₂L_aux with γ₁ = 0.25, γ₂ = 1.0; all positive pairs and 3× more negative pairs feed L_aux.",
      optimization:
        "ImageNet-pretrained backbone; batch 16, initial lr 0.02 for 12 epochs, decayed ×0.1 after epochs 8 and 11. MOT17: longer side resized to 1088 (following recent practice), horizontal flip + color jitter. No extra data beyond COCO-pretrained init (allowed by official rules). TAO: severe over-fitting on train videos → detection model frozen, only the embedding head fine-tuned (LVIS-pretrained init, shorter side 640–800).",
    },
    equations: [
      {
        id: "qdtrack-embed-single",
        label: "Single-positive non-parametric softmax (base loss)",
        formula: "L_embed = − log [ exp(v·k⁺) / ( exp(v·k⁺) + Σ_{k⁻} exp(v·k⁻) ) ]",
        variables: [
          { symbol: "v", meaning: "embedding of the training sample (key-frame RoI)" },
          { symbol: "k⁺", meaning: "positive target embedding (same object on reference frame)" },
          { symbol: "k⁻", meaning: "negative target embeddings on the reference frame" },
        ],
        intuition:
          "Pull the sample toward its true match and push it away from every other reference-frame region — a contrastive objective over the actual candidate set rather than a fixed class list.",
        why:
          "Non-parametric (no learned class weights) so it transfers to unseen identities at test time; the base form from instance-discrimination literature, then extended for MOT's multiple positives.",
        where: "Section 3.2, Eq. 2; extended to Eq. 3 for multiple positives.",
        params: "128 key-frame samples × 256 reference-frame targets per iteration, IoU-balanced sampling.",
        paperIds: ["T045"],
      },
      {
        id: "qdtrack-embed-multi",
        label: "Multi-positive contrastive loss",
        formula: "L_embed = log [ 1 + Σ_{k⁺} Σ_{k⁻} exp( v·k⁻ − v·k⁺ ) ]",
        variables: [
          { symbol: "k⁺", meaning: "all positive targets (every RoI on the reference frame covering the same object)" },
          { symbol: "k⁻", meaning: "all negative targets" },
          { symbol: "v", meaning: "key-frame sample embedding" },
        ],
        intuition:
          "Each sample must beat *every* other reference region against *each* of its true matches simultaneously — one sample has many positives, unlike classic contrastive learning.",
        why:
          "The multi-positive form (reformulated from Eq. 2→4→5) treats positives and negatives fairly; ablating from one-positive to multi-positive adds +1 point IDF1 on BDD100K (66.8 → 67.8).",
        where: "Section 3.2, Eqs. 4–5 (derivation from the single-positive Eq. 2 that 'does not treat positive and negative targets fairly').",
        params: "Dense matching: each key-frame sample matched to ALL reference-frame samples, not just GT pairs.",
        paperIds: ["T045"],
      },
      {
        id: "qdtrack-aux-l2",
        label: "L2 auxiliary loss",
        formula: "L_aux = ( (v·k) / (‖v‖·‖k‖) − c )² ,  c = 1 if positive else 0",
        variables: [
          { symbol: "c", meaning: "1 for positive pairs, 0 for negative pairs" },
          { symbol: "v, k", meaning: "the pair's embeddings" },
        ],
        intuition: "Directly regress the cosine similarity to ±1, pinning the logit magnitude the softmax loss leaves free.",
        why:
          "Constrains similarity magnitude and cosine similarity — without it the bi-softmax at inference (which operates on these logits) sees poorly calibrated scores.",
        where: "Section 3.2, Eq. 6; weighted by γ₂ = 1.0 in the total loss.",
        params: "All positives + 3× negatives sampled for L_aux; γ₂ is insensitive to tuning (Appendix D).",
        paperIds: ["T045"],
      },
      {
        id: "qdtrack-bisoftmax",
        label: "Bi-directional softmax similarity",
        formula: "f(i, j) = [ exp(nᵢ·mⱼ) / Σ_{k=0}^{M−1} exp(nᵢ·m_k) + exp(nᵢ·mⱼ) / Σ_{k=0}^{N−1} exp(n_k·mⱼ) ] / 2",
        variables: [
          { symbol: "nᵢ", meaning: "embedding of detection i in the current frame (N detections)" },
          { symbol: "mⱼ", meaning: "embedding of matching candidate j (M candidates: tracks, vanished tracks, backdrops)" },
          { symbol: "f(i, j)", meaning: "symmetrized match confidence in [0, 1]" },
        ],
        intuition:
          "Score a pair only if it is the best match in BOTH directions — detection's best candidate and candidate's best detection — so one-to-many and no-match cases get penalized automatically.",
        why:
          "Turns the ambiguous nearest-neighbour problem into consistent one-to-one matching: objects with no true target fail both directions (low score → new track / false positive), and replacing plain cosine adds +2.2 IDF1 (+4.5 on pedestrians).",
        where: "Section 3.3, Eq. 8; the paper's 'main inference strategy', with duplicate removal and backdrops around it.",
        params: "match_score_thr = 0.5; candidates span past 10-frame tracks + backdrops (1 frame).",
        paperIds: ["T045"],
      },
      {
        id: "qdtrack-total-loss",
        label: "Joint detection–embedding objective",
        formula: "L = L_det + γ₁·L_embed + γ₂·L_aux ,  L_det = L_rpn + λ₁·L_cls + λ₂·L_reg",
        variables: [
          { symbol: "L_rpn, L_cls, L_reg", meaning: "standard Faster R-CNN proposal, classification and box regression losses" },
          { symbol: "γ₁, γ₂", meaning: "weights on the contrastive embedding loss and L2 auxiliary loss" },
          { symbol: "λ₁, λ₂", meaning: "detection loss weights" },
        ],
        intuition: "One end-to-end objective trains detection and instance similarity together instead of treating re-ID as a post-hoc stage.",
        why:
          "Joint optimization is what lets the quasi-dense pairs shape the detector's shared FPN features; γ₁ > 0.5 hurts performance, γ₂ barely matters (Appendix D).",
        where: "Sections 3.1 and 3.2, Eqs. 1 and 7.",
        params: "λ₁ = λ₂ = 1.0, γ₁ = 0.25, γ₂ = 1.0.",
        paperIds: ["T045"],
      },
    ],
    datasets: ["mot16", "mot17", "others"],
    metrics: ["mota", "idf1", "motp", "idsw", "fp", "fn", "mt-ml", "fps"],
    baselines: ["CenterTrack", "Tracktor", "RetinaTrack", "SORT", "Lif_T", "TubeTK"],
    results: [
      "MOT17 test, private detector (Table 1): 68.7 MOTA / 66.3 IDF1 / MT 957 / ML 516 / FP 26,589 / FN 146,643 / IDSW 3378 with no external data — beats CenterTrack (67.8 / 64.7) by 0.9 / 1.6 points.",
      "MOT16 test: 69.8 MOTA / 67.1 IDF1 / MT 316 — best row in Table 1; MOT17 public detector (Table 7): 64.6 MOTA / 65.1 IDF1 vs CenterTrack public 61.5 / 59.6 (+3.1 / +5.5).",
      "BDD100K (Table 2): val 63.5 MOTA / 71.5 IDF1 / 36.6 mMOTA / 50.8 mIDF1, test 64.3 / 72.3 / 35.5 / 52.3 — +9.2 mMOTA over the benchmark baseline (Yu et al.) and far ahead of the 2020 challenge champion madamada (33.6 mMOTA test) despite a simpler detector.",
      "Waymo (Table 3): val vehicle 55.6 MOTA (+10.7 over RetinaTrack, +13.0 Tracktor++); test with ResNet-101+DCN MOTA/L1 51.18, MOTA/L2 45.09 — on par with Waymo 2020 challenge champion HorizonMOT (51.01 / 45.13) using a single simple model.",
      "TAO (Table 8): AP50 16.1 val / 12.4 test vs SORT_TAO 13.2 / 10.2 (+2.9 / +2.2), driven by frequent classes (person 38.6 vs baseline 18.5 AP50).",
      "Speed: 20.3 FPS on MOT17 (1088×608, ResNet-50) and 16.4 FPS on BDD100K (1296×720) on a Tesla V100 — the embedding head adds only marginal cost over Faster R-CNN.",
    ],
    ablations: [
      "Quasi-dense matching (Table 4 top, BDD100K val, cosine metric): sparse baseline 60.4 MOTA / 63.0 IDF1 → one-positive quasi-dense 61.5 / 66.8 → multi-positive 62.5 / 67.8 — extra negatives contribute 70% of the +4.8 IDF1 gain, multi-positive adds another +1.0.",
      "Inference strategy (Table 4 bottom): cosine → bi-softmax +2.2 IDF1 (67.8 → 70.0, pedestrian +4.5); +duplicate removal 70.1; +backdrops 71.5 — full pipeline +3.1 MOTA / +8.5 IDF1 over the base, pedestrian +9.1 MOTA / +10.5 IDF1, ID switches cut by 30%.",
      "Cues (Table 5): without appearance, IoU alone 26.3 mMOTA, +motion 27.7, +regression 28.6; appearance alone is already best at 36.6 / 50.8, and adding IoU, motion or regression to appearance does not improve it (36.3–36.4) — motion priors are redundant once embeddings are good.",
      "Oracle analysis (Appendix E): detection oracle reaches >94 MOTA on all 8 classes and 88.8 IDF1 (+38 over the real result) while tracking-oracle only adds +4.3 mIDF1 — QDTrack is 'bounded more by detection performance than tracking performance'.",
    ],
    limitations: {
      authorStated: [
        "Failure analysis: inaccurate detection confidences (small/occluded objects as FN, mirrors and billboards as FP) are the main distraction because they destroy the one-to-one matching constraint — pointing at video object detection as the next lever.",
        "Association requires category consistency: a 'rider' reclassified as 'pedestrian' under occlusion cannot be matched.",
        "Instances covering totally different object regions (before/after occlusion) or highly truncated objects lack sufficient appearance evidence for matching.",
        "On TAO the gains concentrate on frequent classes; tail-class tracking still needs dedicated work (zero/few-shot) and is buried by the hundreds-class average.",
      ],
      evident: [
        "Higher recall trades into more identity switches — the paper itself notes MOT17's 3378 IDSW partly stems from detecting more tracks (FP 26,589 vs CenterTrack's 18,498).",
        "All association thresholds (init 0.8, match 0.5, NMS 0.7/0.3, memo 10 frames) are fixed constants, robust on MOT/BDD but tuned per dataset (TAO overrides them).",
        "Pure appearance with no motion prior is untested on motion-blur / fast-motion corner cases; its sufficiency claim rests on BDD100K and MOT statistics.",
      ],
    },
    assumptions: [
      "Objects in an image are rarely identical, so a properly learned embedding space makes nearest-neighbour search sufficient without bells and whistles.",
      "A reference frame within ±3 frames yields valid positive pairs for contrastive training.",
      "True associations are same-category, letting inter-class NMS and category gates prune the candidate set safely.",
    ],
    computation:
      "20.3 FPS MOT17 (1088×608, ResNet-50), 16.4 FPS BDD100K (1296×720), Tesla V100; embedding head adds only marginal inference cost to Faster R-CNN. Training: batch 16, lr 0.02, 12 epochs with ×0.1 decay at epochs 8 and 11; TAO fine-tunes only the embedding head (detection frozen) to avoid over-fitting.",
    relations: [
      { to: "T042", type: "uses-as-baseline", note: "CenterTrack is the MOT17 state of the art QDTrack displaces on both private (68.7 vs 67.8 MOTA) and public (64.6 vs 61.5) protocols; cited as a displacement-regression method." },
      { to: "T030", type: "uses-as-baseline", note: "Tracktor appears as a baseline in the public-detector MOT17 table and defines the comparison protocol (same detector, same track-initialization rules) the paper follows." },
      { to: "T005", type: "uses-as-baseline", note: "SORT is the appearance-free baseline in the BDD100K segmentation-tracking tables (mMOTSA 12.8 vs QDTrack 24.0/30.8)." },
      { to: "T006", type: "uses-as-baseline", note: "Evaluates on the MOT16/17 benchmarks introduced by that paper (Table 1) with its CLEAR/IDF1 metrics and splits." },
    ],
    concepts: ["mot", "appearance-features", "reid", "data-association", "learned-association", "tracking-by-detection"],
    impact:
      "      QDTrack established dense contrastive similarity as a viable replacement for motion models in tracking-by-detection — its 'appearance alone beats IoU+motion' result (Table 5) and bi-softmax/backdrop inference became reference design points for driving-scenario MOT on BDD100K/Waymo/TAO and fed into later query-based MOT systems.",
  },
  {
    id: "T046",
    arxiv: "2006.10721",
    title: "Ocean: Object-aware Anchor-free Tracking",
    shortTitle: "Ocean",
    year: 2020,
    authors: ["Zhipeng Zhang", "Houwen Peng", "Jianlong Fu", "Bing Li", "Weiming Hu"],
    fileName: "2006.10721v2.pdf",
    task: "single-object",
    tags: ["anchor-free-head", "siamese-tracker", "feature-alignment", "online-update", "scale-invariance"],
    difficulty: "intermediate",
    summary:
      "Ocean traces anchor-based Siamese trackers' drift to a training gap: the regression head only ever sees positive anchors (IoU ≥ 0.6), so it cannot repair boxes whose overlap has collapsed. The fix is anchor-free — every pixel inside the ground-truth box regresses its four side distances, giving the tracker a 'rectification capacity' that recovers predictions shifted up to 48 px — plus a feature-alignment module that samples classification features from the predicted box itself (object-aware global feature) alongside the fixed-grid local feature. EAO 0.467 offline / 0.489 with plug-in online update on VOT2018 at 58 FPS; SOTA on five benchmarks.",
    problem:
      "Anchor-based regression trained only on IoU ≥ 0.6 anchors is blind to weak predictions (IoU < 0.3), so once error accumulates the tracker cannot amend itself and drifts. Separately, classification scores are read from fixed regular sampling grids that ignore object scale and position, making foreground/background decisions unreliable against complex backgrounds; existing feature-alignment tricks additionally trust only high-score boxes, so a confident background hit poisons the features.",
    background: ["sot", "siamese", "anchor-free-head", "rpn-head", "bounding-box", "occlusion", "success-plot"],
    previousWork: [
      {
        name: "SiamRPN / SiamRPN++ (anchor-based Siamese trackers)",
        limitation:
          "The regression network learns offsets only from positive anchors (IoU ≥ 0.6) and is 'previously unseen' at low overlap — predictions degraded by error accumulation (e.g. fast motion) can never be rectified, so the tracker drifts.",
        whyThisPaper:
          "Anchor-free per-pixel distance regression trains on every pixel of the GT box, so even a small foreground region suffices to re-predict the full box — mIoU stays 0.71 under 40 px induced drift where SiamRPN++ collapses to 0.28 (Table 8).",
      },
      {
        name: "FCOS (anchor-free detection)",
        limitation:
          "FCOS uses identical training samples for classification and regression and computes objectness from a fixed regular-region feature — no tracking-specific sampling, no global object description.",
        whyThisPaper:
          "Asymmetric sampling tailored to tracking (regression: all GT-box pixels; classification: only pixels within R ≤ 16 px of the centre) plus an object-aware aligned feature that FCOS lacks (Sec. 3.4).",
      },
      {
        name: "Existing feature-alignment approaches (e.g. SPM-tracker, Cascade RPN-style ROI alignment)",
        limitation:
          "They align features only to high-classification-score boxes; when the top scores indicate background, the aligned features mislead detection instead of correcting it.",
        whyThisPaper:
          "Alignment is decoupled from classification confidence — features are sampled from the predicted box regardless of score, and the resulting global object-aware feature feeds back to make classification more reliable.",
      },
    ],
    researchGap:
      "No tracker could rectify its own low-overlap predictions (anchor regression had no training signal there), and no classification head read scale-adaptive, object-level features independent of confidence — the two missing pieces for robust anchor-free tracking.",
    contribution: [
      "Object-aware anchor-free network: direct position/scale regression from every GT-box pixel (rectifies inexact predictions) + feature-alignment module producing an object-aware feature from the predicted box that enhances classification.",
      "Tracking framework combining the anchor-free model with an efficient feature-combination module (three irregular dilated convolutions + depth-wise cross-correlation on a single backbone stage) and a plug-in online update branch.",
      "State of the art on five benchmarks (VOT2018, VOT2019, OTB-100, GOT-10k, LaSOT) at real-time speed (58 FPS), with controlled experiments showing the gains come from the model, not extra training data.",
    ],
    method: {
      pipeline: ["extract", "combine", "localize", "update"],
      architecture:
        "Modified ResNet-50 backbone (last stage cut; stage-4 down-sampling stride 2→1, 3×3 convs dilated ×2 for resolution/receptive field). Feature combination: single-scale (stage-4) features through three parallel 3×3 dilated convolutions Φ_ab with (a, b) ∈ {(1,1), (1,2), (2,1)}, channels 1024→256, depth-wise cross-correlation between exemplar and search branches, fused by point-wise sum (Eq. 9). Anchor-free regression head: four 3×3 convs (256 ch) + one 3×3 conv (4 ch) predicting (l, t, r, b) distances per pixel. Classification head: regular-region branch (four 3×3 convs 256 ch + one 3×3, 1 ch) over the fixed-grid feature f′, and object-aware branch (3×3 conv 'OA.Conv') over the alignment-sampled feature f; scores fused as p_cls = ω·p_o + (1−ω)·p_r with ω = 0.07. Inference adds a scale-change penalty α (k = 0.021) and smooth scale interpolation ŝ = β·s′ + (1−β)·s (β = 0.7).",
      motionModel:
        "None beyond SiamFC-style penalties: scale-change penalty α = exp(k·max(r₀/r, r/r₀)·max(s₀/s, s/s₀)) suppressing size/aspect jumps, and linear interpolation of box scale between frames.",
      association: "Not applicable (single-object tracking).",
      reid: "None.",
      trackManagement:
        "Offline mode: exemplar feature computed once at the first frame and matched against every search image. Online mode: an ATOM/DiMP-style branch (inheriting backbone stages 1–3, stage 4 initialized with DiMP's pretraining) is trained during inference with the fast conjugate-gradient algorithm and fused as p = ω′·p_onl + (1−ω′)·p̂_cls, ω′ = 0.5; no IoUNet.",
      detectionDependency: "Not applicable.",
      loss:
        "L = L_reg + λ₁·L_o + λ₂·L_r (Eq. 8), λ₁ = 1, λ₂ = 1.2: IoU loss −Σ ln(IoU(preg, T*)) on the regression head; BCE on the object-aware branch with soft label p*_o = IoU(predicted box, GT); BCE on the regular-region branch with binary label p*_r = 1 for pixels within R ≤ 16 px of the target centre (Eq. 7).",
      optimization:
        "ImageNet-initialized backbone; trained on Youtube-BB, ImageNet VID/DET, GOT-10k and COCO; inputs 127×127 / 255×255; synchronized SGD on 8 GPUs × 32 images = batch 256, 50 epochs × 6×10⁵ pairs; first 5 epochs warm-up lr 10⁻³ with backbone frozen, then end-to-end with exponential lr decay 5×10⁻³ → 10⁻⁵; weight decay 10⁻³, momentum 0.9. Test hyper-parameters (ω = 0.07, ω′ = 0.5, k = 0.021, β = 0.7) selected with the TracKit automated parameter tuner.",
    },
    equations: [
      {
        id: "ocean-ltrb-labels",
        label: "Anchor-free distance labels",
        formula: "l* = x − x₀ ,  t* = y − y₀ ,  r* = x₁ − x ,  b* = y₁ − y",
        variables: [
          { symbol: "(x, y)", meaning: "a training pixel inside the ground-truth box B = (x₀, y₀, x₁, y₁)" },
          { symbol: "l*, t*, r*, b*", meaning: "distances from the pixel to the four sides of B" },
        ],
        intuition:
          "Every pixel inside the object votes for the whole box by reporting how far each edge is — no anchors, no keypoints.",
        why:
          "Because ALL GT-box pixels (not just IoU ≥ 0.6 anchors) are supervised, the head can still predict the box when only a sliver of the object is foreground — the rectification capacity.",
        where: "Section 3.1, Eq. 1; predicted by the final 3×3 conv with 4 channels.",
        params: "Regression samples: all pixels in GT box; classification positives: separate, only pixels within R = 16 px of centre.",
        paperIds: ["T046"],
      },
      {
        id: "ocean-feature-alignment",
        label: "Object-aware feature alignment",
        formula: "f[u] = Σ_{g∈G, Δt∈T} w[g] · x[u + g + Δt] ,   T = { (m_x, m_y) + B } − { (d_x, d_y) + G }",
        variables: [
          { symbol: "G", meaning: "regular k×k convolution sampling grid around location (d_x, d_y)" },
          { symbol: "B", meaning: "grid of offsets spanning the predicted box M = (m_x, m_y, m_w, m_h)" },
          { symbol: "Δt", meaning: "spatial transformation shifting regular samples onto the predicted box" },
          { symbol: "f", meaning: "output object-aware feature map" },
        ],
        intuition:
          "Deform the convolution's sampling grid so the kernel reads the whole predicted object instead of a fixed local patch — a global, scale-adaptive description.",
        why:
          "Object-aware features give classification a whole-object view (complementing the centre-focused regular feature); ablating them costs 2.9 EAO points, and alignment is independent of confidence so background scores can't corrupt sampling.",
        where: "Section 3.2, Eqs. 2–3; Δt = 0 degenerates to regular-region sampling (Eq. 2 special case).",
        params: "Sampling positions adapt to each predicted box per location on the classification map.",
        paperIds: ["T046"],
      },
      {
        id: "ocean-iou-loss",
        label: "IoU regression loss",
        formula: "L_reg = − Σ_i ln( IoU( p_reg , T* ) )",
        variables: [
          { symbol: "p_reg", meaning: "predicted (l, t, r, b) box" },
          { symbol: "T*", meaning: "ground-truth distance labels (Eq. 1)" },
          { symbol: "i", meaning: "index over training samples (pixels)" },
        ],
        intuition: "Maximize overlap directly rather than penalizing coordinate errors — the loss matches the evaluation criterion.",
        why: "IoU loss is regression-scale invariant, which suits per-pixel distance prediction across object sizes.",
        where: "Section 3.3, Eq. 4; paired with BCE classification losses (Eqs. 5–6).",
        params: "Applied to every GT-box pixel sample.",
        paperIds: ["T046"],
      },
      {
        id: "ocean-total-loss",
        label: "Joint training objective",
        formula: "L = L_reg + λ₁·L_o + λ₂·L_r",
        variables: [
          { symbol: "L_o", meaning: "BCE on the object-aware score map with soft IoU labels p*_o" },
          { symbol: "L_r", meaning: "BCE on the regular-region score map with binary centre-proximity labels p*_r (R ≤ 16 px)" },
          { symbol: "λ₁, λ₂", meaning: "trade-off hyperparameters" },
        ],
        intuition:
          "One objective trains rectification (regression), global objectness (aligned branch) and centreness-like matching (regular branch) together — asymmetric by design.",
        why:
          "The asymmetric sampling — dense regression labels vs centre-only classification labels — is exactly what separates Ocean from FCOS and is validated by the +3.8 EAO 'centralized sampling' ablation.",
        where: "Section 3.3, Eqs. 7–8.",
        params: "λ₁ = 1, λ₂ = 1.2, R = 16 pixels.",
        paperIds: ["T046"],
      },
      {
        id: "ocean-feature-combination",
        label: "Multi-dilation feature combination",
        formula: "S = Σ_{ab} Φ_ab( f_e ) ∗ Φ_ab( f_s )",
        variables: [
          { symbol: "f_e, f_s", meaning: "exemplar and search branch features (single backbone stage)" },
          { symbol: "Φ_ab", meaning: "3×3 dilated convolution with strides (a, b) on X/Y axes, channels 1024→256" },
          { symbol: "∗", meaning: "depth-wise cross-correlation" },
        ],
        intuition:
          "Correlate template and search through three differently dilated kernels and sum — anisotropic dilations read elongated regions at different scales.",
        why:
          "Irregular dilations (1,2) and (2,1) add +1.3/+1.0 EAO over isotropic ones (Table 5) at negligible cost — multi-scale modelling without multi-stage correlation.",
        where: "Section 4.1, Eq. 9; strides chosen as (1,1), (1,2), (2,1).",
        params: "Single-scale (stage-4) features only, unlike SiamRPN++'s multi-level correlation.",
        paperIds: ["T046"],
      },
    ],
    datasets: ["vot", "otb", "got10k", "lasot"],
    metrics: ["eao", "precision", "success-auc", "fps"],
    baselines: ["SiamRPN++", "ATOM", "DiMP", "ECO", "SiamMask", "LADCF"],
    results: [
      "VOT2018 (Table 1): offline EAO 0.467 / A 0.598 / R 0.169, online 0.489 / 0.592 / 0.117 — beats VOT2018 champion LADCF (0.389) by 7.8 points, SiamRPN++ (0.414) by 5.3 (mainly +27.8% relative robustness), DiMP (0.440), and the online branch adds +2.2 EAO at 58 FPS.",
      "VOT2019 (Table 2): online EAO 0.350, offline 0.327 — second overall behind DiMP's baseline run (0.379), but the offline real-time model beats real-time DiMP (0.321) by 0.6 and SiamRPN++ (0.292) by 3.5.",
      "GOT-10k test (Table 3): offline AO 0.592 / SR₀.₅ 0.695; online AO 0.611 / SR₀.₅ 0.721 — +4.5 AO over ATOM (0.556), +0.9 SR over DiMP (0.712).",
      "OTB-100 (Fig. 4): online tracker achieves the best precision 0.920 among compared methods (DiMP holds best AUC 0.686).",
      "LaSOT test: offline SUC 0.527 vs SiamRPN++ 0.496; online improves +4.6 SUC over ATOM (comparable to DiMP-50) and sets the best precision 0.566.",
      "Rectification capacity (Table 8): under simulated drift of 8–56 px, mIoU stays 0.71–0.73 out to 48 px (0.54 at 56 px) while SiamRPN++ falls 0.65 → 0.21; under identical SiamRPN++ training settings Ocean still wins 0.455 vs 0.414 EAO (Table 6).",
    ],
    ablations: [
      "Component build-up on VOT2018 (Table 4): baseline 0.358 → +centralized classification sampling 0.396 (+3.8) → +feature combination 0.438 (+4.2) → +object-aware classification 0.467 (+2.9) → +online update 0.489 (+2.2) — every component contributes.",
      "Dilated kernels in feature combination (Table 5): Φ₁₁ alone 0.425; parallel Φ₁₁ 0.433; +anisotropic Φ₁₂ 0.446 / Φ₂₁ 0.443; all three 0.467 — irregular dilations are the point, not extra parameters.",
      "Training data (Table 6): under SiamRPN++'s exact schedule Ocean 0.455 vs SiamRPN++ 0.414 (+4.1 from the model); adding GOT-10k +1.2 (0.467); adding LaSOT does not help further (0.462 — its categories are already covered).",
      "Depth of anchor-free heads (Table 7, OTB-100): AUC saturates at 3–4 conv layers (0.672 / 0.673 vs 0.645 with 0), the speed/accuracy balance chosen for the model.",
    ],
    limitations: {
      authorStated: [
        "Future work: update the object-aware classification network's parameters directly, instead of integrating an additional online branch — acknowledging the current online mode needs a whole extra network.",
        "Future work: extend the framework to other online video tasks (video object detection, segmentation).",
      ],
      evident: [
        "The object-aware score enters the final classification with ω = 0.07 — tiny at inference, so most of its +2.9 EAO ablation gain is likely training-time regularization rather than test-time signal.",
        "Offline Ocean trails DiMP on VOT2019 (0.327 vs 0.379); without the plug-in branch the tracker is not benchmark-leading there.",
        "Test-time penalties (k, β) and fusion weights are tuned per benchmark with an automated tuner, so headline numbers include protocol-specific tuning.",
      ],
    },
    assumptions: [
      "The first-frame exemplar feature remains valid for the whole sequence in offline mode (computed once).",
      "Predictions stay close enough to the object that sampling inside the predicted box still covers it — alignment follows the prediction, errors included.",
      "Centre-proximity positives (R ≤ 16 px) yield discriminative similarity features for region matching.",
    ],
    computation:
      "58 FPS on VOT2018 (abstract/Fig. 1); the feature-combination module is 'extremely lightweight' (three 3×3 dilated convs on one stage, channels 1024→256). Training: 8× Tesla V100, batch 256, 50 epochs × 6×10⁵ pairs; inference on Xeon E5-2690; three-run standard deviation ±0.5%.",
    relations: [
      {
        to: "T049",
        type: "uses-as-baseline",
        note:
          "Trains on the LaSOT training set and evaluates on the LaSOT test benchmark introduced by that paper (offline SUC 0.527 vs SiamRPN++ 0.496).",
      },
    ],
    concepts: ["sot", "siamese", "anchor-free-head", "rpn-head"],
    impact:
      "Ocean imported the FCOS-style anchor-free formulation into Siamese SOT and identified regression rectification as the robustness lever — its per-pixel distance regression, asymmetric sampling and confidence-independent feature alignment became the template for later anchor-free Siamese trackers, distributed through the TracKit codebase.",
  },
  {
    id: "T047",
    arxiv: "2008.00836",
    title: "LSOTB-TIR: A Large-Scale High-Diversity Thermal Infrared Object Tracking Benchmark",
    shortTitle: "LSOTB-TIR",
    year: 2020,
    authors: [
      "Qiao Liu",
      "Xin Li",
      "Zhenyu He",
      "Chenglong Li",
      "Jun Li",
      "Zikun Zhou",
      "a.o.",
    ],
    venue: "28th ACM International Conference on Multimedia (MM '20)",
    fileName: "2008.00836v1.pdf",
    task: "benchmark",
    tags: ["benchmark", "thermal-infrared", "tir-tracking", "training-data", "attribute-evaluation", "dataset-curation"],
    difficulty: "intro",
    summary:
      "LSOTB-TIR is the first large-scale thermal-infrared (TIR) SOT benchmark with a released training split: 1,400 sequences (>600K frames, >730K boxes) split into a 120-sequence evaluation set (82K frames, 22 classes) and a 1,280-sequence training set (47 classes). It defines 4 scenario and 12 challenge attributes, evaluates 33 public trackers, and re-trains ECO/SiamFC/CFNet/HSSNet on its TIR training data — every re-trained variant improves on three benchmarks, evidence that RGB-pretrained deep features underperform on thermal imagery.",
    problem:
      "Existing TIR benchmarks (LTIR, VOT-TIR16, PTB-TIR, RGB-T) are too small, contain too few object classes (PTB-TIR is pedestrians only), cover too few scenarios and challenges, and ship no training dataset — so TIR-specific deep features could not be learned and fair evaluation was impossible. Meanwhile the deep trackers used on TIR data are trained on RGB datasets: TIR images have no color and little texture, so texture-biased RGB features are less discriminative for recognizing TIR objects and distinguishing distractors.",
    background: ["sot", "benchmark-design", "appearance-features", "success-plot", "occlusion"],
    previousWork: [
      {
        name: "LTIR / VOT-TIR16 / PTB-TIR / RGB-T (TIR benchmarks)",
        limitation:
          "Small scale (20–60 sequences for LTIR/VOT-TIR16/PTB-TIR), few object classes, few scenarios/challenges (only VOT-TIR16 has challenge subsets; none has scenario attributes), and no training dataset — parameter fine-tuning overfits easily and comparisons stay fragmented.",
        whyThisPaper:
          "LSOTB-TIR provides 1,400 sequences, 47 classes, 4 scenario + 12 challenge attributes and, critically, a 1,280-sequence training split released with the benchmark.",
      },
      {
        name: "RGB-pretrained deep TIR trackers (MCFTS, MLSSNet, HSSNet, Gao et al., ECO-stir)",
        limitation:
          "Their deep feature models are learned from RGB data, which biases them toward texture/color cues that TIR images lack; the paper's t-SNE study shows RGB-trained features fail to separate intra-class TIR distractors that TIR-trained features separate.",
        whyThisPaper:
          "The benchmark releases enough TIR training data to re-train these very architectures, and the re-training results (ECO-TIR, SiamFC-TIR, CFNet-TIR, HSSNet-TIR) isolate the feature-training domain as the missing ingredient.",
      },
      {
        name: "Hand-crafted-feature TIR trackers (KCF, DSST, SRDCF, BACF, Struck, L1APG)",
        limitation:
          "Limited by hand-crafted representations (HoG, Haar, raw pixels, covariance); online-learning alternatives gain accuracy at unusable speed (MDNet 1 fps, VITAL 3 fps).",
        whyThisPaper:
          "The 33-tracker evaluation quantifies the gap directly: the top ten are almost all deep-feature methods, and the re-trained TIR deep variants top the table at real-time speed (ECO-TIR 18 fps, SiamFC-TIR 45 fps).",
      },
    ],
    researchGap:
      "Before this paper, no large-scale, high-diversity TIR tracking benchmark with a released training set existed, so learning TIR-specific deep features was impossible and evaluation on TIR scenarios and challenges stayed locked to tiny datasets.",
    contribution: [
      "LSOTB-TIR benchmark: 1,400 TIR sequences, >600K frames and >730K annotated boxes — an evaluation subset of 120 sequences (82K frames, 22 classes, selected as the hardest of 200 candidates) plus a training subset of 1,280 sequences (47 classes, 524K frames, >650K boxes).",
      "Attribute taxonomy for TIR tracking: 4 scenario attributes (video surveillance, drone-mounted, hand-held, vehicle-mounted) and 12 challenge attributes (thermal crossover, intensity variation, distractor, background clutter, deformation, occlusion, out of view, scale change, fast motion, motion blur, low resolution, aspect-ratio variation), each with an evaluation subset.",
      "High-quality annotation pipeline: semi-automatic label tool built on ECO-HC generating boxes in short (≤10-frame) tracking windows, manual correction of inaccurate boxes, and an 8-PhD annotator team verifying every annotation frame by frame twice.",
      "Evaluation of 33 public trackers (sparse, correlation-filter, hand-crafted, deep-CF, matching-based, classification-based), each annotated with its representation, search strategy, category and venue for analysis.",
      "Four re-trained deep trackers (ECO-TIR, SiamFC-TIR, CFNet-TIR, HSSNet-TIR) on the released TIR training set, with gains confirmed on VOT-TIR16, PTB-TIR and LSOTB-TIR itself.",
    ],
    method: {
      pipeline: ["collect", "annotate", "attribute", "split", "evaluate", "retrain"],
      architecture:
        "Benchmark construction: ~600 TIR videos collected from YouTube (white-hot palette only; iron/rainbow/cyan filtered out) plus 150 sequences from existing datasets (BU-TIV, RGB-T); fragments cut under three rules — object must be active, full occlusion/out-of-view gap ≤ 1 s, fragment ≤ 4K frames — yielding 1,400 sequences over 47 object sub-classes across 5 target kinds (person, animal, vehicle, aircraft, boat) and 4 camera scenarios.",
      appearanceModel:
        "Analysis instrument, not a tracker: t-SNE visualization of CFNet backbone features trained on ImageNet VID vs LSOTB-TIR — the TIR-trained features separate both intra-class distractors (30 persons from different sequences) and inter-class objects, while the RGB-trained features collapse them; the re-trained ECO-TIR then uses SiamFC-TIR's backbone as its feature extractor.",
      detectionDependency:
        "Annotation-time dependency: ground-truth boxes are pre-generated by a semi-automatic label tool based on the ECO-HC tracker run in short windows (accurate within ~10 frames), with manual adjustment whenever appearance or scale changes drastically; tracking evaluation itself has no detector dependency.",
      trackManagement:
        "Evaluation-set selection: 200 candidate sequences each containing at least one challenging factor are first screened by running all evaluated trackers, then the 120 with the lowest average success score become the final evaluation subset; local labels per frame record class, position, occlusion (true when occluded or out-of-view above 50%) and identity.",
      optimization:
        "Evaluation protocol only: trackers run with the authors' original parameters (no per-tracker tuning); re-trained variants swap RGB training data for the released TIR training set within each tracker's own training scheme.",
    },
    equations: [
      {
        id: "lsotb-tir-precision",
        label: "Precision (center location error)",
        formula: "precision(θ) = #{frames with CLE < θ} / #{frames},  CLE = ‖p_pred − p_gt‖₂",
        variables: [
          { symbol: "CLE", meaning: "center location error: Euclidean distance between predicted and ground-truth box centers" },
          { symbol: "θ", meaning: "threshold in pixels (e.g., 20 px) for calling a frame successful" },
        ],
        intuition:
          "How often the tracker's box center lands within θ pixels of the true center — position accuracy only, blind to box size.",
        why:
          "Together with success it is one of the two base criteria the benchmark reports for every tracker (One Pass Evaluation).",
        where: "Section 4.1, evaluation criteria; Table 3 precision column (e.g., ECO-TIR 0.768).",
        params:
          "θ is a fixed pixel count, so it is sensitive to image resolution and target size — the motivation for normalized precision.",
        simulator: "metrics",
        paperIds: ["T047"],
      },
      {
        id: "lsotb-tir-norm-precision",
        label: "Normalized precision (AUC)",
        formula: "AUC of normalized precision over the threshold range [0, 0.5], center error normalized by the ground-truth bounding-box size",
        variables: [
          { symbol: "normalized error", meaning: "center error divided by the size of the ground-truth box, as in TrackingNet and LaSOT" },
          { symbol: "AUC", meaning: "area under the precision curve between thresholds 0 and 0.5 — the ranking score in Table 3" },
        ],
        intuition:
          "The same precision curve but with errors measured relative to target size, so a 10 px miss on a small target counts as worse than on a large one.",
        why:
          "Raw precision mixes tracker quality with resolution and object scale; normalizing makes sequences with very different target sizes comparable, and the [0, 0.5] AUC gives the single ranking number.",
        where: "Section 4.1 (normalization 'as that in TrackingNet and LaSOT'); used to rank trackers in Table 3.",
        params: "Range [0, 0.5] because normalized errors beyond half the box size are already complete failures.",
        simulator: "metrics",
        paperIds: ["T047"],
      },
      {
        id: "lsotb-tir-success",
        label: "Success (overlap ratio / AUC)",
        formula: "OR = area(B_pred ∩ B_gt) / area(B_pred ∪ B_gt);  success(τ) = #{frames with OR > τ} / #{frames};  score = AUC over τ ∈ [0, 1]",
        variables: [
          { symbol: "OR", meaning: "overlap rate (IoU) between predicted and ground-truth bounding boxes" },
          { symbol: "τ", meaning: "overlap threshold swept across [0, 1] to build the success plot" },
        ],
        intuition:
          "Sweep how much overlap you demand and count how often the tracker clears it; the curve's area is the headline success score.",
        why:
          "Success AUC is the paper's primary ranking criterion — trackers in Table 3 are ordered by it (ECO-TIR 0.631 first).",
        where: "Section 4.1, success under One Pass Evaluation; Table 3 and all attribute-based success plots (Fig. 6).",
        params: "Single-threshold overlap would be arbitrary; the AUC aggregates all strictness levels in one number.",
        simulator: "metrics",
        paperIds: ["T047"],
      },
    ],
    datasets: ["lsotb-tir", "others"],
    metrics: ["precision", "norm-precision", "success-auc", "eao", "fps"],
    baselines: ["ECO", "SiamFC", "CFNet", "SiamRPN++", "MDNet", "ATOM"],
    results: [
      "Dataset scale: 1,400 sequences, 606K total frames (524K train + 82K evaluation), >730K boxes; evaluation subset 120 sequences (max 2,110 / min 105 / mean 684 frames, 22 classes) and training subset 1,280 sequences (max 3,056 / min 47 / mean 410 frames, 47 classes, >650K boxes).",
      "Overall ranking of 33 trackers (Table 3, success / precision / normalized precision): ECO-TIR (re-trained) 0.631 / 0.768 / 0.695 first, ECO-stir 0.616, ECO 0.609, SiamRPN++ 0.604, MDNet 0.601, VITAL 0.597, ATOM 0.595; best hand-crafted tracker ECO-HC 0.561; KCF fastest (272 fps) but only 0.321 success — the top ten are almost all deep-feature methods.",
      "Re-training (Table 4, VOT-TIR16 EAO): SiamFC 0.225 → SiamFC-TIR 0.250, CFNet 0.254 → CFNet-TIR 0.289 (+3.5), HSSNet 0.262 → HSSNet-TIR 0.271, ECO 0.267 → ECO-TIR 0.290 (+2.3); PTB-TIR success SiamFC 0.480 → 0.566, CFNet 0.449 → 0.530 (+8.1); LSOTB-TIR success SiamFC 0.517 → 0.554 (+3.7) with normalized precision 0.587 → 0.626 (+3.9).",
      "Attribute-based evaluation (Fig. 6): ECO-TIR is best on distractor (0.629, +4.6 over MDNet) and background clutter (0.621, +2.7) — its strength is TIR-specific discrimination; MDNet wins thermal crossover (0.612, +10.5 over ECO-TIR); ATOM wins scale variation and aspect-ratio variation; rankings shift across scenarios (ECO-TIR +7.1 over MDNet on surveillance but −1.3 on drone-mounted).",
      "Feature study (Fig. 2, t-SNE): CFNet features trained on LSOTB-TIR separate both intra-class distractors and inter-class TIR objects, where the same architecture trained on ImageNet VID does not — the qualitative evidence behind the re-training gains.",
      "Annotation and protocol: 730K boxes from a semi-automatic ECO-HC labeler corrected and double-verified by 8 PhD annotators; evaluation subset chosen as the 120 hardest of 200 candidates by average tracker success; all trackers run unmodified on one i7 4.0 GHz + GTX-1080 machine.",
    ],
    ablations: [
      "Training-data swap (Table 4): identical architectures, only the training set changes from RGB (ImageNet VID) to LSOTB-TIR — every -TIR variant improves on all three benchmarks (e.g., ECO 0.267 → 0.290 EAO, CFNet 0.254 → 0.289), so the gain comes from TIR-specific features rather than architecture.",
      "Data-scale counterpoint: CFNet-TIR is trained on the TIR set even though CFNet's original RGB model used ImageNet VID — four times larger — yet still gains 8.1% success on PTB-TIR and 3.5% EAO on VOT-TIR16; domain match beats dataset size.",
      "Backbone transfer: ECO-TIR simply reuses SiamFC-TIR's backbone as its feature extractor inside the ECO framework (0.609 → 0.631 success), showing the TIR features are portable across tracking frameworks without touching the tracker itself.",
    ],
    limitations: {
      authorStated: [
        "Conclusions frame deep TIR tracking as unfinished: 'learning TIR-specific deep features for improving TIR object tracking is one of the main ways in the future' — even the re-trained trackers are presented as a starting point, not a solved problem.",
        "Attribute-based results are shown only for the top-10 trackers in the paper; remaining per-attribute results are deferred to supplementary material.",
      ],
      evident: [
        "The evaluation subset was chosen by running the very trackers being compared and keeping the 120 sequences where they struggle most — benchmark difficulty is defined relative to 2020-era trackers, favoring sequences that defeat their failure modes.",
        "The authors' own re-trained variant (ECO-TIR) tops every main table of the benchmark it introduces.",
        "Collection keeps only white-hot TIR renders; iron/rainbow/cyan palettes were filtered out, so the results do not transfer to other thermal renderings.",
        "Speed numbers are measured on one desktop configuration (i7 4.0 GHz + GTX-1080) with authors' original parameters, so cross-tracker fps comparisons are indicative rather than controlled.",
      ],
    },
    assumptions: [
      "The 5 target kinds with prominent thermal radiation (person, animal, vehicle, aircraft, boat) cover the objects of interest in civilian TIR tracking.",
      "White-hot is the representative TIR rendering; other palettes are noise for benchmark purposes.",
      "Average success of the evaluated trackers is a valid difficulty signal for selecting the final 120 evaluation sequences.",
      "Gaps of full occlusion or out-of-view shorter than one second are still trackable without resetting the protocol.",
    ],
    computation:
      "Annotation: ECO-HC-based semi-automatic label tool pre-generates boxes in ≤10-frame windows, manually corrected, double frame-by-frame verified by an 8-PhD team. Evaluation: all 33 trackers run with authors' original parameters on the same PC (Intel i7 4.0 GHz CPU, GTX-1080 GPU); speed reported as average frame rate (1–272 fps across the table). Dataset and code released at github.com/QiaoLiuHit/LSOTB-TIR.",
    relations: [
      {
        to: "T011",
        type: "uses-as-baseline",
        note:
          "ECO (RGB-pretrained features) is the baseline the re-trained ECO-TIR improves: LSOTB-TIR success 0.609 → 0.631, VOT-TIR16 EAO 0.267 → 0.290 (+2.3), PTB-TIR success 0.633 → 0.650 (+1.7).",
      },
      {
        to: "T009",
        type: "uses-as-baseline",
        note:
          "SiamFC is re-trained into SiamFC-TIR on the released training set: LSOTB-TIR success 0.517 → 0.554 (+3.7), normalized precision 0.587 → 0.626, PTB-TIR precision 0.623 → 0.758 (+13.5).",
      },
      {
        to: "T014",
        type: "uses-as-baseline",
        note:
          "CFNet is re-trained into CFNet-TIR: despite CFNet's RGB model using ImageNet VID (4× larger than the TIR set), CFNet-TIR gains 8.1% success on PTB-TIR and 3.5% EAO on VOT-TIR16.",
      },
      {
        to: "T024",
        type: "builds-on",
        note:
          "Adopts LaSOT's normalized-precision protocol (center error normalized by ground-truth box size, ranked by AUC over [0, 0.5]) and follows its large-scale benchmark design — evaluation split plus released training split — for the TIR domain.",
      },
    ],
    concepts: ["benchmark-design", "sot", "appearance-features", "success-plot"],
    impact:
      "LSOTB-TIR became the standard large-scale TIR SOT benchmark: its 1,280-sequence training split made TIR-specific deep-feature learning possible for the first time, its re-training results are the reference evidence that RGB-pretrained features underperform on thermal imagery, and its 4-scenario / 12-challenge attribute taxonomy is the vocabulary later TIR tracking work reports against.",
  },
  {
    id: "T048",
    arxiv: "2008.03467",
    title: "RPT: Learning Point Set Representation for Siamese Visual Tracking",
    shortTitle: "RPT",
    year: 2020,
    authors: ["Ziang Ma", "Linyuan Wang", "Haitao Zhang", "Wei Lu", "Jun Yin"],
    fileName: "2008.03467v2.pdf",
    task: "single-object",
    tags: ["siamese", "point-set-representation", "anchor-free-head", "online-classifier", "state-estimation"],
    difficulty: "intermediate",
    summary:
      "RPT argues that tracking accuracy is capped by the bounding-box state itself — a coarse four-number extent with no way to model geometric transformation — and replaces it with a set of 9 representative points, uniformly initialized at each correlation-map location and refined by deformable-convolution offsets in an offline-trained Siamese subnet. A min-max pseudo box turns the box annotations into IoU supervision for the points; a second, online-trained classifier labels target presence by average deviation to the point set and is fused with the offline scores, while features from the last three residual blocks are aggregated for both branches. State of the art at publication: OTB2015 AUC 0.715, VOT2018 EAO 0.510, VOT2019 EAO 0.417, GOT-10k AO 0.624, at over 20 FPS.",
    problem:
      "Target state estimation lags the foreground/background classification side of tracking: RPN-based trackers regress from pre-defined anchors (hurting generalization and efficiency), anchor-free trackers only predict offsets to box corners or sides, and positives are assigned by a preset center area that ignores target appearance and geometry. The bounding box itself provides only a coarse spatial extent and cannot express geometric transformations (scale, aspect ratio, rotation), so localization accuracy is structurally limited however good the features become.",
    background: ["sot", "siamese", "bounding-box", "iou", "anchor-free-head"],
    previousWork: [
      {
        name: "SiamRPN series (anchor-based RPN trackers)",
        limitation:
          "Regressing the target region from a pre-defined set of anchor boxes injects prior assumptions about scale/ratio distribution and, per the paper, 'leads to an apparent degeneration in tracking efficiency and generalization'.",
        whyThisPaper:
          "RPT needs no anchors: every correlation-map location is a candidate whose state is a point set initialized at (i, j) and moved by learned deformable offsets.",
      },
      {
        name: "Anchor-free Siamese trackers (SiamFC++, SiamBAN, SiamCAR)",
        limitation:
          "They free the tracker from anchors but still regress relative offsets to the box corners or sides — the state remains a box — and they label a location positive if it falls in a preset center area of the GT box, 'ignoring the target appearance and geometric structure'.",
        whyThisPaper:
          "The point set is a finer state than four numbers: supervised through a pseudo box (so existing box annotations suffice), it learns points on object boundaries and semantically prominent regions for fine-grained localization.",
      },
      {
        name: "Other state refinements (ATOM, GOTURN, SAMF, LDES, SiamMask, D3S)",
        limitation:
          "Multi-scale search (SAMF) and Laplace-based box regression (GOTURN) are restricted to small motions and limited scale changes; mask-based trackers (SiamMask, D3S) need an extra branch and pixel-level supervision from YouTube-VOS.",
        whyThisPaper:
          "Nine deformable points model scale, aspect-ratio and rotation through the offsets of a 3×3 deformable convolution, trained from plain bounding-box annotations — no masks, no multi-scale enumeration.",
      },
    ],
    researchGap:
      "Classification advanced with discriminative filters and online learning, but the state representation everyone estimated was still the bounding box; no tracker had learned a finer, annotation-compatible target state that directly attacks localization accuracy.",
    contribution: [
      "Point set target state: n = 9 representative points per correlation-map location, refined by deformable-convolution offsets (two stages), with a min-max pseudo box mapping the point set back to the box annotations for IoU-based supervision.",
      "Point-guided online classifier: a lightweight 2-layer FCN trained online whose Gaussian-style training labels are replaced by per-pixel scores built from average position deviation to the offline-learned point set, fused with the offline response map by weight α.",
      "Multi-level aggregation over the last three residual blocks: weighted-fusion of the three classification response maps (Eq. 7) and union of the three point sets (Eq. 8), with down-sampling removed from conv4/conv5 and dilated convolutions restoring receptive field.",
      "State of the art at publication on OTB2015 (AUC 0.715 / precision 0.936), VOT2018 (EAO 0.510) and VOT2019 (EAO 0.417), competitive on GOT-10k (AO 0.624), running at over 20 FPS; extended to VOT2020 with EAO 0.539.",
    ],
    method: {
      pipeline: ["correlate", "refine-points", "pseudo-box", "online-classify", "aggregate", "fuse"],
      architecture:
        "Shared ResNet-50 backbone (per the deeper-and-wider design guidelines) extracting hierarchical features from the last three residual blocks; per-level depth-wise cross-correlation between template z and search region x feeds two parallel subnets — an offline target-estimation subnet with an RP head (classification + two-stage point regression using deformable convs) and a 2-layer fully-convolutional online classification subnet.",
      appearanceModel:
        "The online branch is the appearance/distractor defense: trained on target regions from the last few frames with labels derived from the representative point set (average position deviation per pixel instead of an isotropic Gaussian that ignores geometry), then combined with the offline response f_l = α·f_online + (1−α)·f_offline.",
      association:
        "SOT, no data association: multi-level response fusion R = Σ_{l=3..5} w_l ∗ f_l with learned end-to-end weights, and the dense state R = ∪_{l=3..5} R_l unioning each head's 9 points into a larger point set for the final pseudo box.",
      loss:
        "Offline multi-task loss: regression = overlap-over-union ratio between the min-max pseudo box and ground-truth box; classification = focal loss for foreground/background. Stage-1 positives are the search locations nearest the target center; stage-2 positives/negatives are pseudo-box IoU > 0.5 / < 0.4 (in between ignored); only positives supervise regression.",
      optimization:
        "SGD on 8 GPUs, 128 image pairs per mini-batch, 20 epochs; linear warm-up 0.001 → 0.005 for the first 5 epochs then exponential decay 0.005 → 0.0005 for the remaining 15; heads only for the first 10 epochs, whole network fine-tuned in the last 10 with backbone learning rate 10× smaller; online subnet trained with Conjugate Gradient.",
    },
    equations: [
      {
        id: "rpt-cross-corr",
        label: "Depth-wise cross-correlation per level",
        formula: "g_l(z, x) = ϕ_l(z) ∗ ϕ_l(x)",
        variables: [
          { symbol: "z, x", meaning: "template patch and search-region patch" },
          { symbol: "ϕ_l(·)", meaning: "feature output of the l-th (of the last three) residual-block level" },
        ],
        intuition:
          "Match template against search region at each feature scale with cheap depth-wise correlation, giving one response map per level as the canvas on which points are placed.",
        why:
          "It is the Siamese matching substrate: every candidate location (i, j) of g_l becomes a potential target state to be refined into a point set.",
        where: "Section 3.1, Eq. 1; computed for levels l = 3, 4, 5 feeding the RP heads.",
        params: "Levels come from the last three residual blocks with down-sampling removed (dilated convolutions preserve spatial resolution).",
        paperIds: ["T048"],
      },
      {
        id: "rpt-point-set",
        label: "Representative point set and its refinement",
        formula: "R = {(x_k, y_k)}_{k=1..n}, x_k = i, y_k = j;  R_r = {(x_k + Δx_k, y_k + Δy_k)}_{k=1..n}",
        variables: [
          { symbol: "n", meaning: "capacity of the point set — set to 9, matching the 3×3 deformable-convolution kernel" },
          { symbol: "(i, j)", meaning: "location of the candidate on the correlation map, where all points start" },
          { symbol: "(Δx_k, Δy_k)", meaning: "learned per-point offsets, predicted alongside a deformable convolution" },
        ],
        intuition:
          "Nine points begin stacked on the candidate and are individually nudged outward; because deformable convolution offsets are unconstrained, the spread can take any shape — modeling scale, aspect ratio and rotation.",
        why:
          "This is the paper's central representational change: the target state becomes a shape instead of four numbers, which is what makes finer localization possible.",
        where: "Section 3.1, Eqs. 2–4; first-stage regression per level, second stage refines the first point set.",
        params: "n = 9 fixed by the 3×3 kernel; only positive samples drive the state regression in both stages.",
        paperIds: ["T048"],
      },
      {
        id: "rpt-pseudo-box",
        label: "Min-max pseudo box (box supervision for points)",
        formula: "B_p = (min_k(x_k + Δx_k), min_k(y_k + Δy_k), max_k(x_k + Δx_k), max_k(y_k + Δy_k));  L_reg = area(B_p ∩ B_gt) / area(B_p ∪ B_gt)",
        variables: [
          { symbol: "B_p", meaning: "pseudo box induced by taking the bounding rectangle of the refined point set" },
          { symbol: "B_gt", meaning: "ground-truth bounding box annotation" },
        ],
        intuition:
          "Wrap the points in the tightest rectangle and score it against the annotated box; the loss only cares about the overlap of the induced extent, so the points themselves are free to sit on boundaries and key parts.",
        why:
          "Tracking annotations are boxes, not points — the pseudo box is the adapter that lets ordinary box datasets supervise a point-set state end-to-end.",
        where: "Section 3.1, Eq. 5; the overlap/union ratio is the regression loss, focal loss handles classification.",
        params: "Stage-2 sample labelling uses the same pseudo box: IoU > 0.5 positive, < 0.4 negative, otherwise ignored.",
        paperIds: ["T048"],
      },
      {
        id: "rpt-score-fusion",
        label: "Online/offline score fusion",
        formula: "f_l = α · f_online + (1 − α) · f_offline",
        variables: [
          { symbol: "f_online", meaning: "response map of the online-trained classifier at level l" },
          { symbol: "f_offline", meaning: "classification-head output of the offline target-estimation subnet at level l" },
          { symbol: "α", meaning: "weight balancing the two confidence sources" },
        ],
        intuition:
          "Blend the fast offline matching score with the online classifier that has seen the last few frames — offline knows what the target is, online knows what looks like it right now.",
        why:
          "The offline head alone 'lacks the capacity to discriminate the target from similar surrounding instances'; the online branch supplies distractor robustness at the cost of continuous training.",
        where: "Section 3.2, Eq. 6; per level, before multi-level fusion.",
        params: "α controls how much the tracker trusts the continually retrained classifier vs the frozen embedding.",
        paperIds: ["T048"],
      },
      {
        id: "rpt-multi-level",
        label: "Multi-level aggregation",
        formula: "R_cls-all = Σ_{l=3..5} w_l ∗ f_l;  R_est-all = ∪_{l=3..5} R_l",
        variables: [
          { symbol: "w_l", meaning: "learned per-level weights for the pixel-wise weighted fusion of classification responses" },
          { symbol: "R_l", meaning: "the n representative points predicted by the head at level l" },
        ],
        intuition:
          "Earlier layers carry fine spatial detail, later layers semantics — sum the responses and pool the point sets so the final state sees both detail and context.",
        why:
          "A single head's 9 points are 'insufficient when handling complicated object structures' and yield an inaccurate pseudo box; stacking three heads' points densifies the state.",
        where: "Section 3.3, Eqs. 7–8; same spatial resolution across levels makes the fusion a pixel-wise sum.",
        params: "w_l are optimized end-to-end with the network rather than hand-tuned.",
        paperIds: ["T048"],
      },
    ],
    datasets: ["otb", "vot", "got10k", "others"],
    metrics: ["success-auc", "precision", "eao", "fps"],
    baselines: ["SiamFC++", "SiamBAN", "SiamRPN++", "DiMP-50", "SiamR-CNN", "ATOM"],
    results: [
      "OTB2015 (Fig. 5): AUC 0.715 and precision 0.936 — +1.4 pts over SiamR-CNN (0.701 AUC) and +2.2 pts over SiamRPN++ (0.914 precision); ahead of SiamBAN 0.696, SiamRPN++ 0.696, DiMP-50 0.684, SiamFC++ 0.682 and ATOM 0.663.",
      "VOT2018 (Table 2): EAO 0.510, robustness 0.103, accuracy 0.629 — best EAO and best robustness of all compared methods, ahead of D3S 0.489 / SiamAttn 0.470 / SiamBAN 0.452 / DiMP-50 0.440; D3S retains higher accuracy (0.640) thanks to YouTube-VOS mask supervision.",
      "VOT2019 (Table 2): EAO 0.417, R 0.186, A 0.623 — outperforms DRNet (0.395, leading the VOT2019 public challenge) and DiMP-50 (0.379) on all three metrics.",
      "GOT-10k (Table 1): AO 0.624, SR0.5 0.730, SR0.75 0.504 — described by the authors as 'competitive': above DiMP-50 (AO 0.611) and SiamFC++ (0.595), below SiamR-CNN (AO 0.649, SR0.75 0.597).",
      "VOT2020: the framework complemented with a modified D3S (target location channel enhanced by average position deviations from the point set) reaches EAO 0.539 under the new segmentation-mask protocol.",
      "Training/throughput: offline subnet trained on image pairs from YouTube-BoundingBox, COCO and ImageNet VID (SGD, 8 GPUs, batch 128, 20 epochs); inference over 20 FPS on a GeForce GTX 1080Ti.",
    ],
    ablations: [
      "Component build-up on OTB2015 (Table 3), baseline = modified SiamFC++ with ResNet-50: 0.675 AUC / 0.898 precision → +point set representation 0.700 / 0.925 (+2.5 AUC) → +online classification 0.709 / 0.928 (+0.9) → +multi-level aggregation 0.715 / 0.936 (+0.6) — the representational change contributes nearly three times either add-on, confirming the paper's thesis that the state, not the classifier, was the bottleneck.",
    ],
    limitations: {
      authorStated: [],
      evident: [
        "The abstract claims state-of-the-art performance, but on GOT-10k the paper only claims 'competitive': Table 1 shows SiamR-CNN ahead on both AO (0.649 vs 0.624) and SR0.75 (0.597 vs 0.504), and whether RPT respected the benchmark's train-only-on-GOT-10k rule (its offline training uses YouTube-BB/COCO/VID) is not stated.",
        "Point count (n = 9), stage-2 IoU thresholds (0.5 / 0.4) and the deformable kernel size (3×3) are hand-set constants with no sensitivity study.",
        "The online classifier trains on target regions 'obtained from the last few frames' — the window length and retraining schedule are unspecified, so the α trade-off cannot be reproduced from the text alone.",
        "VOT2020's EAO 0.539 comes from a hybrid with a modified D3S segmentation branch, not from the pure box-output RPT evaluated elsewhere.",
      ],
    },
    assumptions: [
      "Bounding-box annotations can supervise a point set: the min-max pseudo box's IoU with the GT box is a sufficient training signal for where the points should land.",
      "All points initialized at the same location can be moved by learned deformable offsets to geometrically significant positions during offline training.",
      "The candidate whose search-region location is closest to the target center is the correct stage-1 positive.",
      "Features from the last three residual blocks jointly carry fine localization detail and semantic robustness; fusing them is better than any single level.",
      "An online 2-layer FCN trained on recent frames generalizes to distractors without overfitting to the current background.",
    ],
    computation:
      "Offline training: SGD, 8 GPUs, 128 pairs/mini-batch, 20 epochs (linear warm-up 0.001→0.005 for 5 epochs, exponential decay 0.005→0.0005 for 15; heads only for the first 10 epochs, whole network fine-tuned for the last 10 with backbone lr ÷10); data from YouTube-BoundingBox, COCO, ImageNet VID. Online subnet: Conjugate Gradient optimization on recent-frame samples. Inference: PyTorch on a single GeForce GTX 1080Ti, over 20 FPS.",
    relations: [
      {
        to: "T029",
        type: "uses-as-baseline",
        note:
          "SiamFC++ is both the ablation baseline (its modified ResNet-50 version scores 0.675 AUC vs RPT's 0.715, Table 3) and a comparison method on OTB2015 (0.682) and VOT2018 (EAO 0.426).",
      },
      {
        to: "T028",
        type: "uses-as-baseline",
        note:
          "SiamRPN++ represents the anchor-based formulation RPT argues against and is compared directly (OTB2015 precision 0.914 vs 0.936; VOT2018 EAO 0.414 vs 0.510).",
      },
      {
        to: "T038",
        type: "uses-as-baseline",
        note:
          "SiamBAN is the anchor-free relative-offset baseline RPT replaces (Fig. 2 juxtaposes them): OTB2015 AUC 0.696 vs 0.715, VOT2018 EAO 0.452 vs 0.510.",
      },
      {
        to: "T044",
        type: "uses-as-baseline",
        note:
          "SiamAttn appears in the VOT2018 comparison (EAO 0.470, A 0.630) — RPT beats it on EAO and robustness while trailing slightly on accuracy.",
      },
    ],
    concepts: ["sot", "siamese", "anchor-free-head", "online-vs-offline", "appearance-features"],
    impact:
      "RPT imported the RepPoints point-set formulation into Siamese tracking and demonstrated that box annotations can train a finer state through min-max pseudo boxes — a recipe later state-estimation trackers reuse — while its point-derived soft labels for the online classifier showed how a learned shape can replace the hand-drawn Gaussian in online tracking; it stood as one of the strongest VOT2018/2019 performers at publication.",
  },
  {
    id: "T049",
    arxiv: "2009.03465",
    title: "LaSOT: A High-quality Large-scale Single Object Tracking Benchmark",
    shortTitle: "LaSOT",
    year: 2020,
    authors: [
      "Heng Fan",
      "Hexin Bai",
      "Liting Lin",
      "Fan Yang",
      "Peng Chu",
      "Ge Deng",
      "a.o.",
    ],
    fileName: "2009.03465v3.pdf",
    task: "benchmark",
    tags: ["benchmark", "long-term-tracking", "one-shot-evaluation", "dataset-curation", "language-annotation", "category-balance"],
    difficulty: "intro",
    summary:
      "LaSOT is a high-quality large-scale SOT benchmark: 85 object classes, 1,550 YouTube videos totaling 3.87M frames, every frame manually annotated with a double-checked bounding box, with an average video length of ~2,500 frames (min 1,000, max 11,397) making long-term tracking assessable for the first time at scale. Each video also carries a natural-language specification, each class has an equal number of videos (no category bias), and the benchmark defines two protocols — full-overlap and one-shot (15 classes held out, entirely outside ImageNet) — under which 48 trackers are evaluated. This journal version extends the CVPR 2019 conference paper with 15 new unseen classes, protocol details and deeper analysis.",
    problem:
      "Deep-era tracking had no benchmark that simultaneously satisfied the needs of training and evaluation: existing datasets are small (<400 videos) so tracking-specific deep representations cannot be learned, most average under 600 frames so they measure only short-term tracking, most have <30 categories and no unseen-class split so category bias and generalization go untested, and the large ones (TrackingNet, OxUvA) are sparsely or semi-automatically annotated so per-frame evaluation and motion cues are unreliable. Researchers were forced to train on detection video data (ImageNet VID, YT-BB) whose targets are often static, partially out of view initially, and sparsely labeled.",
    background: ["sot", "benchmark-design", "success-plot", "appearance-features", "online-vs-offline"],
    previousWork: [
      {
        name: "OTB / TC-128 / VOT / NUS-PRO / UAV123 / UAV20L / NfS / CDTB (dense evaluation benchmarks)",
        limitation:
          "Precisely annotated but short (average video length under 600 frames — ~20 s at 30 fps), small (<400 videos), evaluation-only (no training split), with mostly <30 categories and no unseen-class protocol; targets almost always stay in view, so results may not reflect real-world performance.",
        whyThisPaper:
          "LaSOT keeps per-frame manual annotation but raises scale to 1,550 videos / 3.87M frames with average length 2,502 frames (83 s), 85 balanced categories, and releases a training split alongside evaluation.",
      },
      {
        name: "TrackingNet / OxUvA / ALOV (large but sparse or semi-automatic benchmarks)",
        limitation:
          "TrackingNet labels every video with a tracker (reliable only over ~1 s windows, average video <500 frames), OxUvA labels every 30 frames, ALOV every 5 frames — sparse or machine-generated ground truth prevents accurate per-frame evaluation and deprives trackers of temporal/motion training cues.",
        whyThisPaper:
          "Every LaSOT frame is drawn/edited by a human, double-checked by a validation team with unanimity, and absence (full occlusion / out-of-view) frames are explicitly labeled — dense, reliable ground truth at 3.87M-frame scale.",
      },
      {
        name: "GOT-10k (one-shot evaluation)",
        limitation:
          "First to propose one-shot evaluation and diverse categories, but it is short-term focused (average 149 frames), uses a different class-counting granularity, is not rigorously balanced per category, and provides no full-overlap counterpart; other large benchmarks have train/test category overlap.",
        whyThisPaper:
          "LaSOT adopts the one-shot idea but pairs it with a full-overlap protocol, enforces equal videos per class (first rigorously category-balanced benchmark), holds its 15 one-shot classes entirely outside ImageNet, and targets long-term rather than short-term tracking.",
      },
    ],
    researchGap:
      "Before this paper, no benchmark combined scale, per-frame manual dense annotation, long videos, category balance, unseen-category evaluation and a tracking-specific training split — so deep trackers trained on detection data and were judged on short, small, biased evaluation sets.",
    contribution: [
      "Large-scale dense benchmark: 85 object classes, 1,550 videos, 3.87M frames, every frame manually labeled with an axis-aligned box and visually double-checked (corrected when needed) — by the authors' knowledge the largest precisely annotated tracking benchmark.",
      "Long-term evaluation: shortest sequence 1,000 frames, longest 11,397, average 2,502 (83 s at 30 fps), with target disappearing/re-appearing events for assessing long-term trackers.",
      "Comprehensive labeling: bounding boxes plus a natural-language specification per video (color, behavior, surroundings — 1,550 sentences), intended to enable lingual-feature tracking; plus 14 attributes with ≥200 videos each for attribute-based analysis.",
      "Two evaluation protocols: full-overlap (1,120 train / 280 test from 70 shared classes, 80/20 per class) and one-shot (train on 1,400 sequences of 70 classes, test on 150 sequences of 15 classes with zero overlap, chosen outside ImageNet).",
      "Rigorous category balance: equal numbers of videos per class in both parts — the first benchmark, to the authors' knowledge, rigorously balanced for equal category size.",
      "Journal-extension contributions over the CVPR 2019 version: 15 extra classes with 150 manually annotated sequences (>350K frames) for one-shot testing, full-overlap/one-shot protocol pair, extended construction details, and deeper experimental analysis of 48 trackers.",
    ],
    method: {
      pipeline: ["collect", "annotate", "validate", "language", "split", "evaluate", "retrain"],
      architecture:
        "Dataset construction: >6,000 YouTube sequences (Creative Commons) searched per class, filtered by video quality and design principles down to 1,550 (one usable clip retained per video after removing tracking-irrelevant content); part-1 = 1,400 sequences from 70 classes (mostly ImageNet categories, a few like drone chosen for applications), part-2 = 150 sequences from 15 classes selected entirely outside ImageNet with zero overlap to part-1; 20 videos per class in part-1, 10 in part-2.",
      appearanceModel:
        "Each sequence carries a natural-language sentence describing the target's color, behavior and surroundings (1,550 sentences), meant to be extracted as lingual features and used as global semantic guidance to suppress background distractors in the search region.",
      trackManagement:
        "Per-frame state labels: if the target is visible, a labeler draws/edits a tight axis-aligned box over any visible part; otherwise an absence label — full occlusion or out-of-view — is assigned, so evaluation skips frames without the target and enables occlusion/out-of-view-aware algorithms. Special exclusion rules apply to 12 categories with undesired thin/long parts (e.g., tails of cat/tiger/mouse, bird legs, guitar handlebar), identified in advance by a three-PhD expert group.",
      optimization:
        "Evaluation design: each of the 48 trackers is tested exactly as published (no retraining or tuning) because trackers need different training strategies, are presumed optimal in their original papers, and pre-trained backbones are hard to fine-tune; the retraining experiments (SiamFC, CFNet) keep the original ImageNet-VID training settings and only swap the training data.",
    },
    equations: [
      {
        id: "lasot-precision",
        label: "Precision (PRE)",
        formula: "PRE(θ) = #{frames with ‖p_pred − p_gt‖₂ < θ} / #{frames},  θ typically 20 px",
        variables: [
          { symbol: "p_pred, p_gt", meaning: "centers of the predicted and ground-truth bounding boxes" },
          { symbol: "θ", meaning: "distance threshold in pixels, typically 20" },
        ],
        intuition:
          "The fraction of frames where the tracker's box center is within 20 pixels of the true center — position accuracy only.",
        why:
          "It is the first of the three OPE metrics LaSOT ranks trackers with, following the OTB protocol.",
        where: "Section 4.1; reported per protocol (e.g., DiMP 0.563 PRE under full overlap, LTMU 0.473 under one-shot).",
        params: "Sensitive to target size and image resolution — the stated motivation for normalized precision.",
        simulator: "metrics",
        paperIds: ["T049"],
      },
      {
        id: "lasot-norm-precision",
        label: "Normalized precision (N-PRE)",
        formula: "N-PRE = precision computed after normalizing center error by the target scale (as in TrackingNet)",
        variables: [
          { symbol: "normalized error", meaning: "center distance divided by ground-truth target scale, so errors are comparable across object sizes" },
        ],
        intuition:
          "The same precision curve with distances measured relative to target size — a 20 px miss on a 30 px object is not the same as on a 300 px one.",
        why:
          "PRE alone 'is sensitive to target size and image resolution'; normalization 'can ensure the consistency of evaluation across different target scales'.",
        where: "Section 4.1 (normalization strategy adopted from TrackingNet); reported alongside PRE for both protocols.",
        params: "No free parameter in the paper — the normalization follows TrackingNet's definition.",
        simulator: "metrics",
        paperIds: ["T049"],
      },
      {
        id: "lasot-success",
        label: "Success rate (SUC)",
        formula: "SUC(τ) = #{frames with IoU(B_pred, B_gt) > τ} / #{frames},  τ typically 0.5",
        variables: [
          { symbol: "IoU", meaning: "intersection-over-union between predicted and ground-truth boxes" },
          { symbol: "τ", meaning: "overlap threshold, typically 0.5" },
        ],
        intuition:
          "What fraction of frames the predicted box overlaps the truth enough to count as tracked; success plots sweep the threshold.",
        why:
          "Success is the headline score for ranking on LaSOT (Figures 8 and 11) because it combines localization and scale accuracy in one number.",
        where: "Section 4.1; primary ranking metric under both full-overlap and one-shot protocols, also used per attribute.",
        params: "Trackers are compared by their success scores; the drop between protocols (0.037–0.18) is the paper's domain-gap measure.",
        simulator: "metrics",
        paperIds: ["T049"],
      },
    ],
    datasets: ["lasot", "otb", "others"],
    metrics: ["precision", "norm-precision", "success-auc"],
    baselines: ["DiMP", "LTMU", "DaSiamRPN", "ATOM", "GlobalTrack", "SiamRPN++"],
    results: [
      "Scale and quality: 1,550 videos, 3.87M frames, 85 classes (70 + 15); min/mean/max video length 1,000 / 2,502 / 11,397 frames; 14 attributes each with ≥200 videos (scale variation >1,400, low resolution 765, out-of-view 509); ~40% of initial annotations fail the first validation round and many frames are revised at least three times before acceptance.",
      "Protocol splits: full overlap — 1,120 train (2.83M frames) / 280 test (690K), 16 of 20 videos per class for training; one-shot — 1,400 train sequences (3.52M frames, 70 classes) vs 150 test sequences (350K frames, 15 unseen classes outside ImageNet).",
      "Full-overlap overall (48 trackers, Fig. 8): DiMP best at 0.563 PRE / 0.642 N-PRE / 0.560 SUC; LTMU 0.535 / 0.621 / 0.539; DaSiamRPN 0.529 / 0.605 / 0.515; GlobalTrack 0.528 / 0.597 / 0.517; ATOM 0.497 / 0.57 / 0.499; SiamRPN++ 0.493 / 0.57 / 0.495; SiamFC 0.339 / 0.42 / 0.336 — yet still beats StructSiam, DSiam, PTAV and HCFT, which the paper attributes to those overfitting small datasets.",
      "One-shot overall (Fig. 11): LTMU best at 0.473 / 0.499 / 0.414; DiMP 0.451 / 0.476 / 0.392; ATOM 0.43 / 0.459 / 0.376; DaSiamRPN 0.42 / 0.443 / 0.356; every algorithm drops 0.037–0.18 SUC versus full overlap — a quantified cross-category domain gap.",
      "Qualitative patterns: all top 18 trackers use deep features; all top seven (DiMP, LTMU, GlobalTrack, DaSiamRPN, ATOM, SiamRPN++, SiamMask) use ResNet-18/50; DiMP is best on 13/14 attributes full-overlap and 10/14 one-shot, LTMU second-best on 11/14; GlobalTrack and LTMU beat DiMP on out-of-view thanks to full-image search, but fail on fast motion (yoyo-7) and small targets (pool-3).",
      "Retraining on LaSOT (Table 6, same settings as VID training): SiamFC OTB-13 SUC 0.588 → 0.608 (full-overlap split) / 0.614 (one-shot split, +2.6), OTB-15 0.565 → 0.582 / 0.589; CFNet OTB-13 0.589 → 0.615 / 0.622, OTB-15 0.568 → 0.593 / 0.598; LaSOT-test gains are smaller (SiamFC 0.336 → 0.342 full, 0.230 → 0.237 one-shot) because LaSOT sequences are long-term while these trackers were designed short-term.",
      "Backbone study (Table 7): SiamRPN++ AlexNet 0.433 → ResNet-18 0.472 → ResNet-50 0.495 (full overlap) and 0.245 → 0.316 → 0.340 (one-shot); DiMP ResNet-18 0.534 → ResNet-50 0.560 / 0.381 → 0.392; swapping ResNet-50 for AlexNet costs 0.039 on full overlap but 0.071 on one-shot — deeper features matter disproportionately for unseen categories.",
      "Model-update analysis (Sec. 5.4): without updates — GlobalTrack 0.517, SiamRPN++ 0.495, SiamMask 0.467, C-RPN 0.455 (full overlap) vs 0.356 / 0.340 / 0.332 / 0.275 (one-shot); with updates — DiMP 0.560, LTMU 0.539, ATOM 0.515, DaSiamRPN 0.499 vs 0.392 / 0.414 / 0.376 / 0.356; model updating is essential for robustness even though the specific online-learning scheme is not.",
    ],
    ablations: [
      "Training-data ablation (Table 6): identical architectures and hyperparameters, only the training set changes from ImageNet VID to LaSOT's splits — every cell improves for both SiamFC and CFNet on OTB-13, OTB-15 and LaSOT test, demonstrating the value of large-scale tracking-specific training data over detection video data.",
      "Backbone-depth ablation (Table 7): deeper backbones (AlexNet → ResNet-18 → ResNet-50) monotonically improve success under both protocols, and the AlexNet penalty is ~2× larger under one-shot (0.071) than full overlap (0.039) — depth is especially required when categories are unseen.",
      "Protocol ablation: the same 48 trackers evaluated under full overlap vs one-shot drop 0.037–0.18 SUC, isolating the cross-category domain gap that category-overlap benchmarks hide.",
      "Model-update comparison (Sec. 5.4): with-update trackers (DiMP, LTMU, ATOM, DaSiamRPN) consistently outperform matching-based no-update trackers (GlobalTrack, SiamRPN++, SiamMask, C-RPN) on both protocols and most attributes.",
    ],
    limitations: {
      authorStated: [
        "Discussion 5.1: 'existing trackers do not fully address the domain gap between different object categories' — the 0.037–0.18 one-shot drop is stated as an open problem, with category/target-level domain adaptation proposed as future direction.",
        "Discussion 5.4: model updating is 'an extremely complex process' — it is difficult to determine when and how to use current information, and inappropriate updates increase drift risk; the paper concludes updates are essential but the online-learning scheme is not the key to gains.",
        "Section 4.4.2: SiamMask fails to beat DaSiamRPN/SiamRPN++ under one-shot — the authors conjecture this is caused by 'the lack of mask annotation for training SiamMask on our benchmark', acknowledging the box-only labels limit mask-based trackers.",
        "Conclusions: despite 48 evaluations there 'still exists significant room for future improvement', and lingual-feature tracking is called out as unexplored future research enabled by the released language specifications.",
      ],
      evident: [
        "Each tracker is evaluated 'as it is' without uniform retraining, so the ranking mixes architecture quality with whatever data each tracker was originally trained on — a deliberate but confounding design choice.",
        "Attribute sequences carry multiple attributes simultaneously (the paper admits the ideal one-attribute-per-sequence set is 'almost impossible' to build), so per-attribute causality remains ambiguous despite ≥200 videos per attribute.",
        "The one-shot test set is only 150 sequences (10 per held-out class), so its per-category statistics — the protocol's entire point — rest on small samples.",
        "The full-overlap test split is just 4 videos per class (280/70), which makes per-category performance estimates noisy even though the split is balanced.",
        "Retraining experiments cover only two Siamese CF trackers (SiamFC, CFNet); the claim that LaSOT beats VID training is not demonstrated for modern trackers like DiMP or SiamRPN++.",
      ],
    },
    assumptions: [
      "Axis-aligned boxes provide sufficient information for stable tracker initialization and reliable evaluation (argued in Section 2.2.1), so rotated annotation is unnecessary.",
      "Absence labels (full occlusion, out-of-view) should be excluded from scoring — evaluation is more accurate and occlusion-aware algorithms can use the signal.",
      "Collecting equal numbers of videos per class eliminates category bias and makes comparisons fairer.",
      "Each published tracker is optimally trained as released, so evaluating unmodified is the fairest comparison across 48 heterogeneous methods.",
      "YouTube in-the-wild footage under Creative Commons license represents real-world tracking conditions, and one usable clip per video suffices after manual inspection.",
    ],
    computation:
      "Annotation: two teams — a labeling team (volunteer draws/edits each frame, PhD expert inspects and adjusts) and a validation team of typically three experts requiring unanimous agreement (non-unanimous results return to the labeling team with detailed comments); ~40% first-round failure rate, many frames revised ≥3 times; undesired-part rules set by a three-PhD expert group; 1,550 language sentences written per sequence. Evaluation: 48 trackers run under OPE with each tracker's published settings; results, data and analysis released at vision.cs.stonybrook.edu/~lasot/.",
    relations: [
      {
        to: "T024",
        type: "extends",
        note:
          "Journal extension of the authors' CVPR 2019 LaSOT: adds 15 new classes outside ImageNet (150 sequences, >350K frames) for one-shot testing, the full-overlap/one-shot protocol pair, richer construction details and deeper analysis of 48 trackers.",
      },
      {
        to: "T025",
        type: "builds-on",
        note:
          "Adopts GOT-10k's one-shot evaluation idea (zero train/test category overlap) and its category-bias motivation, then extends it with a full-overlap counterpart, rigorous equal-videos-per-class balance and a long-term (avg 2,502-frame) rather than short-term focus.",
      },
      {
        to: "T009",
        type: "uses-as-baseline",
        note:
          "SiamFC is one of the 48 evaluated trackers (full-overlap SUC 0.336) and is retrained on LaSOT in Table 6: OTB-13 0.588 → 0.608 / 0.614 vs ImageNet-VID training.",
      },
      {
        to: "T014",
        type: "uses-as-baseline",
        note:
          "CFNet is retrained on LaSOT in Table 6 with unchanged settings: OTB-13 SUC 0.589 → 0.615 / 0.622 and OTB-15 0.568 → 0.593 / 0.598 versus ImageNet-VID training.",
      },
      {
        to: "T028",
        type: "uses-as-baseline",
        note:
          "SiamRPN++ is evaluated on both protocols (full overlap 0.493 PRE / 0.495 SUC) and is the backbone of the depth study (Table 7): AlexNet 0.433 → ResNet-50 0.495 full-overlap SUC.",
      },
    ],
    concepts: ["benchmark-design", "sot", "success-plot", "appearance-features"],
    impact:
      "LaSOT became the standard long-term SOT training-and-evaluation platform: its released train split is the default large-scale training set for Siamese trackers, its one-shot protocol (with classes outside ImageNet) is the field's unseen-category test, and its measured 0.037–0.18 cross-category drop made the domain gap a named research problem — while its language specifications seeded later vision-language tracking work.",
  },
];
