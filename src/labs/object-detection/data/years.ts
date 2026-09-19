import type { YearResearchState } from "./types";

/* YEARLY RESEARCH STATE 2015-2026
 * One entry per year present in the corpus (papers have years 2015..2026).
 * Every item is grounded in the papers cited in supportingPapers[]
 * (see papers.ts / evidence extraction).
 *
 * Evidence labels used throughout:
 *   - "paper says": statement from one paper's own text
 *   - "cross-paper": consistent across 2+ corpus papers
 *   - "lab": research-lab synthesis (marked MEDIUM/LOW confidence)
 */

export const YEAR_STATES: Record<number, YearResearchState> = {
  2015: {
    year: 2015,
    dominantApproaches: [
      "Two-stage pipelines (region proposal + classifier) remain the accuracy reference (cited by SSD/YOLO).",
      "One-stage unified detection arrives: detection-as-regression (YOLO) and anchor-based single-shot multibox scoring (SSD).",
    ],
    emergingIdeas: [
      "Unified single-network detection (no proposal stage).",
      "Default-box / anchor priors with IoU-based matching to ground truth.",
      "Multi-scale feature maps to cover object size variation.",
      "Real-time detection as a first-class goal.",
    ],
    decliningIdeas: ["Per-component hand-tuned pipelines yield to end-to-end networks."],
    persistentProblems: [
      "Small objects: SSD states 'small objects may not even have any information' in the deepest feature maps.",
      "Localization accuracy: YOLO underperforms two-stage methods on precise boxes.",
      "Class imbalance: SSD matches thousands of default boxes against few objects.",
      "Anchor/grid design sensitivity.",
      "Dependence on post-processing (NMS/thresholding).",
    ],
    newProblems: [
      "Anchor prior design now a first-class hyper-parameter (shapes, scales, aspect ratios).",
      "Speed/accuracy trade-off becomes the organizing axis of the field.",
    ],
    solvedOrReducedProblems: ["End-to-end single-pass prediction removes the proposal bottleneck."],
    unsolvedProblems: ["Small objects in deep feature maps. Precise localization.", "Grouped/overlapping objects (YOLO)."],
    mathematicalDevelopments: [
      "Bounding-box regression with (smooth) L1-style losses.",
      "IoU matching to assign anchors/default boxes to ground truth.",
      "Multi-class softmax over grid cells / default boxes.",
    ],
    architecturalDevelopments: [
      "CNN backbone (VGG16 / custom) + multi-scale feature maps.",
      "Grid-cell structure with B predictors per cell (YOLO).",
      "Prediction heads attached to several feature scales (SSD).",
    ],
    computationalDevelopments: [
      "Real-time GPU detection (~45 fps region for YOLO).",
      "Single forward pass replaces sliding-window/proposal cost.",
    ],
    applications: [
      "General-purpose real-time detection (YOLO marketed for live/embedded settings).",
      "Detection on VOC/COCO-class benchmarks as the standard evaluation.",
    ],
    edgeCases: [
      "Small objects vanishing in low-resolution deep features.",
      "Groups of nearby objects confounding grid-cell prediction.",
    ],
    researchDirections: [
      "One-stage vs two-stage accuracy race.",
      "Multi-scale feature engineering.",
      "Real-time efficiency.",
    ],
    supportingPapers: ["P001", "P002"],
  },

  2016: {
    year: 2016,
    dominantApproaches: ["One-stage real-time detectors (YOLO/SSD family) mature; two-stage remains reference."],
    emergingIdeas: [
      "Joint detection + classification training (WordTree hierarchy).",
      "k-means dimension clusters as data-driven anchor priors (replacing hand-picked).",
      "Multi-scale / multi-resolution training.",
      "Long-tail class recognition without full box labels.",
    ],
    decliningIdeas: ["Hand-selected anchor shapes."],
    persistentProblems: [
      "Small objects (YOLO still weaker on small objects).",
      "Localization precision vs two-stage.",
      "Speed-accuracy trade-off.",
    ],
    newProblems: [
      "Long-tail categories lack bounding-box annotations.",
      "Semantic hierarchy design (WordTree) and misclassification across hierarchy levels.",
      "Co-occurring/rare context for thousands of classes.",
    ],
    solvedOrReducedProblems: [
      "Real-time + strong classification jointly; anchor priors now learned (fewer hand choices).",
    ],
    unsolvedProblems: ["Small/grouped objects.", "Accurate boxes at high speed."],
    mathematicalDevelopments: [
      "Iterative k-means with an IoU-distance objective for anchor priors.",
      "Joint softmax over hierarchy + multi-label classification.",
      "Multi-scale train-time input augmentation.",
    ],
    architecturalDevelopments: [
      "Batch-normalized backbone, higher-resolution inputs, dimension clustering in anchor design.",
      "WordTree-structured output space.",
    ],
    computationalDevelopments: ["9000-class real-time detection with one model/one pass."],
    applications: [
      "Massively multi-class detection (9000 categories) from combined detection+classification data.",
    ],
    edgeCases: [
      "Rare classes and unseen-at-finetune categories (precursor of open-vocabulary).",
      "Small objects and groups of objects.",
      "Hierarchy ambiguity (e.g., classes overlapping semantically).",
    ],
    researchDirections: [
      "Joint classification/detection supervision as a data-efficiency path.",
      "Data-driven priors.",
      "Higher resolution inputs for accuracy.",
    ],
    supportingPapers: ["P003"],
  },

  2017: {
    year: 2017,
    dominantApproaches: [
      "One-stage multi-scale detectors dominate the corpus; SSD is the main lineage.",
      "Two-stage pipelines still the accuracy reference in citations.",
    ],
    emergingIdeas: [
      "Context injection by deconvolution (DSSD) and recurrent rolling convolution (RRC).",
      "Cascade refinement of anchors within a single-shot network (RefineDet).",
      "From-scratch training with deep supervision (DSOD), removing pre-trained backbone dependency.",
      "Feature fusion across heterogeneous maps (FSSD).",
      "Oriented / rotatable bounding boxes (RotBox).",
    ],
    decliningIdeas: [
      "Dependence on ImageNet-pretrained backbones (explicitly challenged by DSOD).",
    ],
    persistentProblems: [
      "Small objects: this year's corpus is focused on it (DSSD, RRC, FSSD, RF-UNet all target small/overlapping objects).",
      "Localization of small or overlapping objects.",
      "Context insufficiency in shallow features.",
    ],
    newProblems: [
      "From-scratch optimization instability (addressed by deep supervision).",
      "Feature misalignment across fused scales.",
      "Orientation/rotation: axis-aligned boxes poor for rotated aerial objects.",
    ],
    solvedOrReducedProblems: ["Small-object accuracy partially improved via context fusion (DSSD, FSSD)."],
    unsolvedProblems: [
      "Small objects remain hard (only partially reduced).",
      "Efficient training without pre-trained feature extractors.",
      "Accurate boxes on crowded rotated scenes.",
    ],
    mathematicalDevelopments: [
      "Deep supervision losses across stages (DSOD).",
      "Deconvolution/fusion feature combination.",
      "Anchor refinement losses + transfer-connection features (RefineDet).",
      "Rotatable IoU / oriented box regression (RotBox).",
    ],
    architecturalDevelopments: [
      "Deconvolution modules (DSSD), recurrent rolling convolution (RRC), transfer-connection blocks (RefineDet).",
      "Dense + deeply supervised design (DSOD).",
      "Feature fusion modules (FSSD), rotatable-bbox heads (RotBox).",
      "FPN-style structure cited and adopted for context (RF-UNet).",
    ],
    computationalDevelopments: ["From-scratch training frees detectors from pre-trained feature extraction."],
    applications: [
      "Aerial and satellite imagery (RotBox).",
      "General real-time single-shot detection.",
    ],
    edgeCases: [
      "Aerial/satellite rotation-dense scenes.",
      "Small and overlapping objects in cluttered imagery.",
    ],
    researchDirections: [
      "Context engineering (deconv, recurrent, fusion).",
      "From-scratch learning.",
      "Oriented detection for aerial imagery.",
    ],
    supportingPapers: ["P004", "P005", "P006", "P007", "P008", "P009", "P010"],
  },

  2018: {
    year: 2018,
    dominantApproaches: [
      "Multi-scale one-stage detectors; feature pyramids formalized as a core construct.",
    ],
    emergingIdeas: [
      "Mobile/efficient backbones with detection heads (MobileNetV2 + SSDLite).",
      "Multi-level feature pyramid networks (M2Det) - object-scale-aware feature assembly.",
      "YOLOv3: three-scale prediction + per-class logistic outputs - small-object 'reversal'.",
    ],
    decliningIdeas: ["Single-scale or shallow-feature-only heads."],
    persistentProblems: [
      "Small objects again central (YOLOv3 reversal narrative; ExtSSD extends shallow layers for small objects).",
      "Efficiency: mobile/edge compute.",
    ],
    newProblems: [
      "Effective receptive field vs object scale mismatch in deep networks.",
      "Fusion of heterogeneous feature levels without degradation.",
    ],
    solvedOrReducedProblems: [
      "Small-object weakness of early YOLO reduced by multi-scale prediction (YOLOv3's own claim).",
    ],
    unsolvedProblems: ["Small objects remain the hardest category.", "Efficient backbone capacity for detection."],
    mathematicalDevelopments: [
      "Feature pyramid construction (top-down upsampling + lateral connections).",
      "Binary cross-entropy per class (multi-label) on YOLOv3 heads.",
      "Anchor clustering per feature scale.",
      "Inverted residuals / linear bottleneck design (MobileNetV2).",
    ],
    architecturalDevelopments: [
      "Three-scale YOLOv3 heads with independent logistic classifiers.",
      "U-shape multi-level feature pyramid (M2Det).",
      "Inverted-residual blocks + SSDLite detection head (MobileNetV2).",
      "Extended shallow feature layers (ExtSSD).",
    ],
    computationalDevelopments: [
      "Mobile/edge detection feasibility (SSDLite).",
      "Real-time multi-scale YOLO.",
    ],
    applications: [
      "Mobile/embedded object detection.",
      "Real-time detection on CPUs/low-power GPUs.",
    ],
    edgeCases: [
      "Small objects at low resolution.",
      "Mobile/embedded compute budgets.",
      "Multi-label scenes (object + context categories).",
    ],
    researchDirections: ["Efficiency-first backbones.", "Deep multi-level feature design."],
    supportingPapers: ["P011", "P012", "P013", "P014"],
  },

  2019: {
    year: 2019,
    dominantApproaches: ["One-stage detectors with fusion/attention refinements; FPN-based two-stage still reference."],
    emergingIdeas: [
      "Attention-based feature fusion in one-stage detectors (ASSD), including Transformer-style global dependencies.",
      "Recognition that naive fusion alone does not improve accuracy - attention needed over content.",
    ],
    decliningIdeas: ["Pure concatenation/summation fusion as a universal fix."],
    persistentProblems: [
      "Small objects; insufficient expressive shallow features.",
      "Effective long-range context integration.",
    ],
    newProblems: [
      "How to fuse global context without injecting irrelevant/noise (attention weighting needed).",
      "Attention module placement/cost in one-stage heads.",
    ],
    solvedOrReducedProblems: ["Context modeling improved by attention-weighted fusion (ASSD)."],
    unsolvedProblems: ["Small-object accuracy gap remains.", "Efficient global context modeling."],
    mathematicalDevelopments: [
      "Attention weights over feature maps (channel/spatial).",
      "Transformer-style self-attention imported for global dependencies.",
    ],
    architecturalDevelopments: [
      "Attention-augmented single-shot detectors.",
      "FPN-based necks retained + attention fusion.",
    ],
    computationalDevelopments: ["Order-of-magnitude cost control for attention in one-stage heads."],
    applications: ["General detection; mobile-friendly one-stage models."],
    edgeCases: ["Small objects.", "Cluttered context needing global reasoning."],
    researchDirections: ["Attention in detection.", "Global-context modeling."],
    supportingPapers: ["P015"],
  },

  2020: {
    year: 2020,
    dominantApproaches: [
      "CSP backbone + neck (FPN/PAN/SPP) real-time one-stage detectors.",
      "Curated 'bag of freebies' training tricks and 'bag of specials' modules (YOLOv4).",
      "Industrial trick-stacking (PP-YOLO) without new architectural invention.",
      "Scalable single-architecture design (Scaled-YOLOv4).",
    ],
    emergingIdeas: [
      "IoU-family regression losses: GIoU -> DIoU -> CIoU (YOLOv4 adopts CIoU + DIoU-NMS).",
      "IoU-aware classification branch (PP-YOLO).",
      "Matrix NMS (parallel) and DIoU-based suppression (PP-YOLO).",
      "Anchor-free/anchor-based taxonomy discussed (YOLOv4, PP-YOLO, Scaled-YOLOv4).",
      "Network-architecture-search surveys enter efficiency arguments (Scaled-YOLOv4, YOLOv4 cite NAS).",
    ],
    decliningIdeas: ["Hyper-parameter-heavy custom modules of a few years earlier (systematized)."],
    persistentProblems: [
      "Small objects still hardest.",
      "NMS heuristics and latency.",
      "Localization metric-loss mismatch (IoU vs L1-style).",
      "Foreground/background and easy/hard sample imbalance.",
    ],
    newProblems: [
      "NMS latency in low-latency pipelines.",
      "Hand-design complexity vs NAS alternatives.",
    ],
    solvedOrReducedProblems: [
      "Regression targets better aligned with evaluation metric (IoU losses).",
      "Suppression quality improved (DIoU-NMS, Matrix NMS).",
    ],
    unsolvedProblems: [
      "Small objects, NMS dependence, class imbalance.",
      "Interaction of dozens of training tricks (combinatorial).",
    ],
    mathematicalDevelopments: [
      "GIoU / DIoU / CIoU loss formulations (documented in YOLOv4).",
      "CIoU loss for training; DIoU as NMS criterion.",
      "IOU-aware weighting between classification and localization.",
      "Focal loss (cited) for one-stage class imbalance.",
      "Mish activation, CmBN, DropBlock, Mosaic augmentation as freebies.",
    ],
    architecturalDevelopments: [
      "CSPDarknet53 + SPP + PAN neck (YOLOv4).",
      "Deformable convolution / CoordConv in heads (PP-YOLO).",
      "CSP bottleneck family scaled across size tiers (Scaled-YOLOv4).",
    ],
    computationalDevelopments: [
      "Real-time on edge GPUs (V100/T4 regimes) as the accepted metric context.",
      "Scaled variants (tiny..large) from one architecture.",
    ],
    applications: [
      "Industrial real-time detection pipelines (PP-YOLO).",
      "Deployment on constrained GPUs.",
    ],
    edgeCases: ["Crowded overlapping detections (NMS stress).", "Small objects in low-compute settings."],
    researchDirections: [
      "IoU-loss and NMS redesign.",
      "Training-trick systematization.",
      "Anchor-free comparison/transition.",
      "Scaling laws within one architecture.",
    ],
    supportingPapers: ["P016", "P017", "P018", "P019"],
  },

  2021: {
    year: 2021,
    dominantApproaches: [
      "Real-time one-stage YOLO-family (PP-YOLOv2, YOLOR, YOLOX).",
      "Anchor-free transition begins inside YOLO (YOLOX).",
    ],
    emergingIdeas: [
      "Anchor-free detection in the YOLO family (P3-P5 feature pyramid, decoupled head).",
      "Dynamic top-k label assignment: SimOTA.",
      "End-to-end (NMS-free) training variant via one-to-one label assignment + stop-gradient.",
      "Unified multi-task representation with implicit/explicit knowledge (YOLOR).",
      "Knowledge distillation into industrial detectors (PP-YOLOv2).",
    ],
    decliningIdeas: ["Hand-tuned static anchor matching (IoU threshold rules)."],
    persistentProblems: ["Small objects (incl. thermal imagery).", "Label assignment quality.", "NMS dependence."],
    newProblems: [
      "Dynamic assignment (SimOTA) introduces new hyper-parameters (top-k, cost design).",
      "Domain gap to thermal/low-feature imagery.",
      "End-to-end YOLO training instability (addressed via stop-gradient).",
    ],
    solvedOrReducedProblems: [
      "Anchor prior design removed by anchor-free heads (fewer priors).",
      "Small-object accuracy improved by higher-res multi-scale features (ongoing).",
    ],
    unsolvedProblems: ["Small/thermal objects.", "Removing NMS without accuracy loss at scale."],
    mathematicalDevelopments: [
      "SimOTA / OTA optimal-transport-style assignment.",
      "IoU-aware classification weighting.",
      "Decoupled head with separate classification/regression branches.",
      "Kernel-space unified representation (YOLOR).",
    ],
    architecturalDevelopments: [
      "CSP/v6-style backbones, FPN+PAN necks, decoupled heads.",
      "PP-YOLOv2 ResNet50-vd + PAN with EMA training.",
      "YOLOX anchor-free P3-P5 + decoupled head.",
      "YOLOR unified one-representation design.",
    ],
    computationalDevelopments: ["Real-time with option of NMS-free inference."],
    applications: ["Thermal/IR inspection imaging (small objects).", "Industrial applications (PP-YOLOv2)."],
    edgeCases: [
      "Thermal imagery, low-feature IR objects.",
      "Small objects in industrial imagery.",
      "Rare/crowded configurations.",
    ],
    researchDirections: [
      "Anchor-free conversion.",
      "Dynamic/learned label assignment.",
      "End-to-end NMS-free detection.",
      "Unified multi-task representations.",
      "Domain-specific adaptation (thermal).",
    ],
    supportingPapers: ["P020", "P021", "P022", "P023"],
  },

  2022: {
    year: 2022,
    dominantApproaches: [
      "Industrial-scale real-time YOLO-family (YOLOv6, YOLOv7, PP-YOLOE, DAMO-YOLO).",
      "Anchor-free baseline with strong label-assignment + head design.",
    ],
    emergingIdeas: [
      "NAS-designed backbones for detection (DAMO-YOLO MAE-NAS).",
      "Re-parameterization for inference-time speed (YOLOv7 RepConv, DAMO RepGFPN).",
      "Auxiliary-head deep supervision (YOLOv7 coarse-for-aux + fine-for-lead).",
      "Task-aligned detection heads (PP-YOLOE ET-Head) with alignment metric.",
      "Distribution-based box regression (DFL) and quality focal / VariFocal losses.",
      "Self-distillation (YOLOv6).",
    ],
    decliningIdeas: ["Plain classification/regression coupling in heads."],
    persistentProblems: [
      "Label assignment quality (TAL/OTA family).",
      "NMS dependence in CNN one-stage.",
      "Small-object accuracy.",
      "Training efficiency (convergence).",
    ],
    newProblems: [
      "Latency budget design across head/neck/backbone (trading parameters).",
      "Distribution regression calibration (DFL).",
      "Quantization/deployment friendliness.",
    ],
    solvedOrReducedProblems: ["Better label assignment (SimOTA/TAL/AOTA) and task alignment reduce misalignment."],
    unsolvedProblems: ["Small objects; NMS dependence; generalization of re-param tricks to all hardware."],
    mathematicalDevelopments: [
      "TAL alignment metric (classification-objectness x IoU).",
      "DFL distribution layers and VariFocal/QFL objectives.",
      "Deep-supervision with auxiliary heads (coarse/fine).",
      "SIoU/GIoU/CIoU selection guidance for regression.",
    ],
    architecturalDevelopments: [
      "E-ELAN (YOLOv7), CSPRepResNet (PP-YOLOE), RepGFPN (DAMO), NAS backbone (DAMO).",
      "Reparameterized convolutions (RepConv) collapsed at inference.",
      "Efficient task-aligned head (PP-YOLOE).",
    ],
    computationalDevelopments: [
      "T4-GPU real-time metrics; quantization-friendly designs; self-distillation without runtime teacher.",
    ],
    applications: [
      "Mass industrial deployment (Meituan YOLOv6).",
      "Embedded/edge GPU real-time detection.",
    ],
    edgeCases: ["Heavily overlapping detections.", "Small objects in dense scenes."],
    researchDirections: [
      "NAS in detection.",
      "Re-parameterization.",
      "Task-aligned heads and assignment.",
      "Distribution-based regression.",
    ],
    supportingPapers: ["P024", "P025", "P026", "P027"],
  },

  2023: {
    year: 2023,
    dominantApproaches: [
      "Real-time CNN one-stage (YOLO family) alongside first real-time end-to-end transformers (RT-DETR).",
      "Transformer open-set detection with text grounding (Grounding DINO).",
    ],
    emergingIdeas: [
      "End-to-end transformer detection in the real-time regime (no NMS).",
      "Boundary/hybrid encoders, uncertainty-minimal query selection (RT-DETR).",
      "Grounded/open-set pretraining with language prompts (Grounding DINO, language-aware queries).",
      "Self-supervised DETR pre-training; identification of one-to-one matching as a convergence bottleneck.",
      "Efficient gather-and-distribute feature fusion (Gold-YOLO) for the CNN line.",
    ],
    decliningIdeas: ["NMS as assumed-necessary post-processing."],
    persistentProblems: [
      "Small objects: RT-DETR itself notes DETRs still inferior on small objects; sparser Hungarian supervision limits training.",
      "Convergence speed of end-to-end matching.",
      "Domain shift / evaluation gaps for open-set models.",
    ],
    newProblems: [
      "Hungarian matching sparsity -> slow convergence (DETR-line).",
      "Open-set benchmarks / long-tail vocabulary coverage.",
      "Pretraining data scale for grounded detection.",
      "Decoder latency vs encoder cost trade-off.",
    ],
    solvedOrReducedProblems: [
      "NMS removed in end-to-end transformers (RT-DETR).",
      "Open-set detection enabled by text-conditioned models (Grounding DINO).",
    ],
    unsolvedProblems: [
      "Small objects in DETRs.",
      "Fast convergence of one-to-one matching.",
      "Robust open-set generalization outside training vocabulary.",
    ],
    mathematicalDevelopments: [
      "Hungarian bipartite matching (Kuhn) for set prediction.",
      "GIoU + L1 box loss; class cost composition; pairwise vs focal classification in DETRs.",
      "Attention: multi-scale / deformable-style (in real-time DETR encoders).",
      "Query selection by uncertainty.",
    ],
    architecturalDevelopments: [
      "Hybrid encoder + decoder with matched queries (RT-DETR).",
      "Language-guided query selection + cross-modality fusion (Grounding DINO).",
      "Gather-and-Distribute neck (Gold-YOLO).",
    ],
    computationalDevelopments: ["Real-time DETR latency (decoder dominates).", "Large-scale grounding pretraining."],
    applications: [
      "Open-vocabulary / prompt-based detection (Grounding DINO).",
      "Real-time end-to-end deployment (NMS-free).",
    ],
    edgeCases: [
      "Open-set / unknown categories.",
      "Domain-shift evaluation sets (aerial, thermal, drone, maritime in Grounding DINO).",
      "Small objects in transformer decoders.",
    ],
    researchDirections: [
      "End-to-end transformer detectors for real-time.",
      "Open-vocabulary / grounded detection.",
      "DETR pre-training and convergence.",
      "Efficient feature exchange in CNN necks.",
    ],
    supportingPapers: ["P028", "P029", "P030", "P031"],
  },

  2024: {
    year: 2024,
    dominantApproaches: [
      "NMS-free everywhere: YOLO goes end-to-end (YOLOv10) via dual assignment; DETR-line matures (RT-DETRv2/v3, D-FINE, DEIM).",
      "Open-vocabulary scales up with larger ViT backbones and pretraining data (Grounding DINO 1.5).",
    ],
    emergingIdeas: [
      "Consistent dual assignment (one-to-many + one-to-one) for NMS-free YOLO (YOLOv10).",
      "Distribution-based fine-grained box regression and dense supervision in DETRs (D-FINE FDR + DDF, GO-LSD).",
      "Hierarchical dense positive supervision (RT-DETRv3) to fix sparse Hungarian supervision.",
      "Group/one-to-many matching (DEIM MatchOR) for fast DETR convergence.",
      "Programmable gradient information + reversible architectures (YOLOv9).",
      "Lightweight & edge open-vocabulary detectors (OVLW-DETR) distilling VLMs.",
      "Vision foundation backbone scaling for open-set edge cases (Grounding DINO 1.5 'edge' domains).",
    ],
    decliningIdeas: ["Static one-to-one sparse supervision as sufficient DETR training."],
    persistentProblems: [
      "Small objects (D-FINE: coarse distribution bins hurt small objects; RT-DETRv3/DEIM: need dense supervision for small objects).",
      "Convergence time of end-to-end matching.",
      "Open-set 'edge' categories, domain shift, adversarial conditions.",
    ],
    newProblems: [
      "Distribution bin resolution limits fine localization (D-FINE).",
      "Pretraining data scale for broad vocabulary (Grounding DINO 1.5 - 20M images).",
      "Matching-algorithm design as an axis of research (DEIM).",
      "Edge deployment of open-vocabulary models (OVLW-DETR).",
    ],
    solvedOrReducedProblems: [
      "NMS removed from the YOLO line (YOLOv10).",
      "DETR convergence substantially improved (RT-DETRv3, DEIM).",
      "Open-vocabulary accuracy & long-tail coverage improved by scaling.",
    ],
    unsolvedProblems: [
      "Small objects still lag (explicitly noted).",
      "Dense supervision cost in transformers.",
      "Truly open set: unseen domains/categories and adversarial cases.",
    ],
    mathematicalDevelopments: [
      "Distribution Focal Loss / distribution regression (DFL, DDF).",
      "Consistent dual assignment losses (one-to-many + one-to-one) with contribution weighting.",
      "Hierarchical dense positive supervision + self-attention perturbation.",
      "Group matching (MatchOR) one-to-many.",
      "GIoU + L1 continued as DETR box losses.",
      "Information bottleneck / gradient programming (YOLOv9 PGI).",
    ],
    architecturalDevelopments: [
      "Dual-head (one-to-many/one-to-one) with rank-guided lightweight head (YOLOv10).",
      "PGI/GELAN reversible branches (YOLOv9).",
      "Vision Conv Attention (VCA) modules (D-FINE).",
      "Hybrid deformable attention encoders + token exchange (LW-DETR).",
      "ViT-based open-set backbone (Grounding DINO 1.5).",
    ],
    computationalDevelopments: [
      "Latency-focused design for NMS-free edge deployment.",
      "Large-scale (20M-image) multimodal pretraining.",
      "Distillation from VLMs to small edge detectors.",
    ],
    applications: [
      "Edge/embedded deployment of detectors incl. open-vocabulary.",
      "Open-set recognizers for diverse domains (aerial, thermal, maritime, potholes...).",
      "Robot/deployment-friendly end-to-end models.",
    ],
    edgeCases: [
      "Edge/'long-tail' open-set categories (Grounding DINO 1.5's explicit 'edge' suite).",
      "Adversarial weather / pothole-type infrastructure objects.",
      "Tiny objects in distribution regression.",
      "Occluded/overlapping objects under NMS-free inference.",
    ],
    researchDirections: [
      "NMS-free unified YOLO/DETR designs.",
      "Fine-grained (distributional) regression.",
      "Matching/label-assignment engineering.",
      "Scaling data + models for open-set.",
      "Efficient open-vocabulary at the edge.",
    ],
    supportingPapers: ["P032", "P033", "P034", "P035", "P036", "P037", "P038", "P039", "P040"],
  },

  2025: {
    year: 2025,
    dominantApproaches: [
      "Hybrid CNN + attention real-time detectors (YOLOv12 area-attention).",
      "Maturing real-time DETRs with foundation-model features (RT-DETRv4, DINOv3-DEIM, LW-DETR lineage).",
      "NAS-designed, domain-adaptable DETRs (RF-DETR).",
      "Survey consolidation of the YOLO line (v5/v8/11/26/27).",
    ],
    emergingIdeas: [
      "Attention-centric CNN backbones with efficient attention (area attention, YOLOv12).",
      "Neural architecture search over DETRs with weight sharing, adapted from VLM pretraining (RF-DETR).",
      "Vision foundation models as training/distillation guidance for compact real-time detectors (RT-DETRv4).",
      "Distribution-guided + decoupled distillation focal losses in the DEIM/D-FINE line.",
      "Conscious reporting that small-object gains lag large-object gains.",
    ],
    decliningIdeas: ["Pure CNN backbones without attention in the accuracy frontier (attention now mainstream)."],
    persistentProblems: [
      "Small objects (explicitly: performance on small objects largely unchanged in some SOTA lines).",
      "Attention/compute cost in real-time settings.",
      "Adaptation to new domains (NAS targeted at domain fine-tuning).",
    ],
    newProblems: [
      "Dependency on foundation-model/VLM pretraining weight.",
      "NAS search cost and transferability.",
      "Quantization/thermal/deployment constraints for attention models (survey).",
    ],
    solvedOrReducedProblems: [
      "Real-time accuracy frontier pushed by attention + matching + distribution regression.",
      "Domain-adaptive detectors via NAS fine-tuning (RF-DETR).",
    ],
    unsolvedProblems: [
      "Small-object plateau.",
      "Open-vocabulary + domain generalization (listed as frontier needs).",
      "Robust deployment under quantization (survey discussion).",
    ],
    mathematicalDevelopments: [
      "Area-attention complexity reduction (YOLOv12).",
      "MatchOR / improved matching (DEIM) and decoupled distillation focal (DDF) losses.",
      "Distribution-based regression refinement (D-FINE).",
    ],
    architecturalDevelopments: [
      "R-ELAN attention-centric blocks (YOLOv12).",
      "VLM pretraining + weight-sharing NAS (RF-DETR).",
      "Foundation-model-guided compact real-time DETRs (RT-DETRv4).",
      "YOLO family unification (survey: v5/v8/11/26/27).",
    ],
    computationalDevelopments: [
      "Pareto accuracy-latency discovery via NAS.",
      "Foundation-model inference used as teacher/guide rather than at runtime.",
    ],
    applications: [
      "Broad real-time ecosystem: detection, tracking, pose (survey).",
      "Aerial, agricultural, medical contexts documented (survey).",
      "Domain-specialized detection (RF-DETR).",
    ],
    edgeCases: [
      "Small objects in SOTA real-time lines.",
      "Domain-specific deployment (thermal/compute constraints).",
      "Long-tail and open-vocabulary categories (survey frontier).",
    ],
    researchDirections: [
      "Attention-centric real-time CNN.",
      "NAS for DETRs.",
      "Foundation-model distillation for real-time.",
      "Convergence of matching + distribution regression.",
    ],
    supportingPapers: ["P041", "P042", "P043", "P044", "P045"],
  },

  2026: {
    year: 2026,
    dominantApproaches: [
      "Unified end-to-end multi-task vision models (YOLO26: detection/segmentation/pose/oriented/tracking/classification).",
      "Occlusion-robust tracking frameworks on anchor-free detectors.",
      "Open-vocabulary detection extended to 3D via promptable segmentation.",
    ],
    emergingIdeas: [
      "NMS-free unified heads with one-to-many/one-to-one dual supervision (only one-to-one at inference).",
      "Small-target-aware label assignment (STAL) guaranteeing positive coverage for tiny objects.",
      "Native quantization and attention/MLP scale tiers in one family.",
      "Open-vocabulary extension inside the YOLO family (YOLOE26).",
      "2D open-vocabulary lifting to 3D boxes without dense 3D annotations (OV-3D).",
      "Prediction + compensation under full/long-term occlusion in tracking (Tracking-the-Unseen).",
    ],
    decliningIdeas: ["NMS required for YOLO-family deployment."],
    persistentProblems: [
      "Tiny objects (STAL motivation: positive coverage).",
      "Occlusion (full, long-term) in detection+tracking.",
      "Open-vocabulary generalization beyond in-domain categories.",
      "3D annotation scarcity.",
    ],
    newProblems: [
      "3D box data annotation scarcity for open-vocab (lifting instead).",
      "Occluder leakage into small objects; truncation of large objects in 3D lifting.",
      "Multi-task head contention / optimization balance across tasks.",
      "Long-term occlusion identity association.",
    ],
    solvedOrReducedProblems: [
      "End-to-end NMS-free multi-task inference (YOLO26).",
      "Occlusion robustness improved in tracking (MOTA/IDF1 gains claimed).",
      "Some open-vocabulary capability in 3D without 3D-annotation-heavy training.",
    ],
    unsolvedProblems: [
      "Tiny-object detection (reduced, not resolved).",
      "Full/long-term occlusion in general tracking.",
      "Open-vocabulary 3D generalization.",
    ],
    mathematicalDevelopments: [
      "Dual-supervision (one-to-many + one-to-one) loss architecture.",
      "STAL coverage guarantees for tiny objects.",
      "Kalman filtering + Hungarian assignment for occlusion-aware association.",
      "Greedy and one-to-one matching for 2D->3D promptable-segmentation lifting.",
      "Oriented (rotatable) box regression for aerial/scene-text (YOLO26).",
    ],
    architecturalDevelopments: [
      "Unified multi-task heads (R-UEL2/R-ETB blocks) (YOLO26).",
      "Promptable segmentation + lift networks (OV-3D).",
      "Kalman+Hungarian occlusion-robust trackers (Tracking-the-Unseen).",
    ],
    computationalDevelopments: [
      "Native NMS-free + quantization-friendly edge deployment.",
      "3D open-vocabulary from 2D foundation segmenters.",
    ],
    applications: [
      "Robotics and autonomous driving (3D open-vocabulary detection).",
      "Multi-object tracking under occlusion (surveillance/AV).",
      "Unified vision APIs (detection+segmentation+pose+tracking).",
    ],
    edgeCases: [
      "Tiny objects and positive-coverage label assignment.",
      "Full and long-term occlusion.",
      "Occluders bleeding into small 3D boxes; range truncation.",
      "Novel/unseen categories in 3D scenes.",
    ],
    researchDirections: [
      "Unified end-to-end multi-task models.",
      "Open-vocabulary beyond 2D.",
      "Occlusion-robust tracking.",
      "Deployment-native design (NMS-free + quantized).",
    ],
    supportingPapers: ["P046", "P047", "P048"],
  },
};

export const YEAR_LIST = Object.keys(YEAR_STATES)
  .map(Number)
  .sort((a, b) => a - b);

export function yearState(year: number): YearResearchState | undefined {
  return YEAR_STATES[year];
}