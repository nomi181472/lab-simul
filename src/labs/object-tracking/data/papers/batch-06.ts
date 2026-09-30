import type { PaperRecord } from "../types";

/* batch-06 — T050..T061 */

export const BATCH_06: PaperRecord[] = [
  {
    id: "T050",
    arxiv: "2009.09669",
    title: "Learning Spatio-Appearance Memory Network for High-Performance Visual Tracking",
    shortTitle: "Spatio-Appearance Memory Network",
    year: 2020,
    authors: ["Fei Xie", "Wankou Yang", "Kaihua Zhang", "Bo Liu", "Wanli Xue", "Wangmeng Zuo"],
    fileName: "2009.09669v5.pdf",
    task: "segmentation-tracking",
    tags: ["siamese", "memory-network", "correlation-filter", "mask-prediction", "appearance-features", "pixel-wise-tracking"],
    difficulty: "intermediate",
    summary:
      "The tracker couples two memories: an Appearance Memory Network (AMN) that stores keys and values of sampled past frames and reads them back with a dense non-local operator, and a DCF-based Spatial Memory Network (SMN) that predicts a target location map and curates the samples fed back into the model. Together they form a spatio-appearance memory network that outputs a pixel mask per frame (the reported box is derived from that mask), reaching EAO 0.535 / 0.506 / 0.356 / 0.453 on VOT2016 / 2018 / 2019 / 2020 and J&F 77.3 on DAVIS16.",
    problem:
      "Segmentation-based trackers must both localize and delineate the target, yet they typically match against a single first-frame template: as appearance changes the fixed template drifts, and every per-frame mask prediction fed back as a new sample injects noise that compounds over time. Correlation-filter trackers avoid a fixed template but update blindly each frame, accumulating the same noise, so neither family could keep tracking and segmentation simultaneously accurate.",
    background: [
      "siamese",
      "memory-network",
      "correlation-filter",
      "appearance-features",
      "bounding-box",
      "deep-detection",
    ],
    previousWork: [
      {
        name: "Correlation-filter trackers (KCF, CCOT, UpdateNet)",
        limitation:
          "A single shallow template/filter is updated every frame from whatever the tracker just predicted, so one bad frame corrupts the model permanently; they also emit boxes only, not masks.",
        whyThisPaper:
          "The DCF objective is kept (fast Fourier-domain ridge regression) but its training samples are gated by a learned uncertainty queue, and the appearance evidence comes from a temporal memory rather than one template.",
      },
      {
        name: "Siamese template matchers (SiamFC, SiamRPN++)",
        limitation:
          "The template is taken from the first frame and never revised, so it cannot represent appearance change or remember which past observations were reliable.",
        whyThisPaper:
          "The AMN stores a key/value pair for every sampled past frame and reads them back by non-local attention, so the query matches against a growing appearance dictionary instead of one crop.",
      },
      {
        name: "Mask trackers (D3S, SiamMask)",
        limitation:
          "They predict a mask from one template with no temporal memory, so shape variation and occlusion break the mask; D3S reaches EAO 0.493 on VOT2016 and J&F 74.0 on DAVIS16.",
        whyThisPaper:
          "The SMN supplies background suppression and sample filtering while the AMN supplies appearance values; their interaction localizes the target and segments it in one pass.",
      },
    ],
    researchGap:
      "Before this paper no tracker jointly used a long-term appearance memory and a discriminative spatial memory that update each other, so mask-based trackers could neither remember reliable appearance nor reject unreliable samples.",
    contribution: [
      "An Appearance Memory Network (AMN): past-frame features are mapped to keys and values and read out by a dense non-local operator with positional encoding, capturing stable appearance over time.",
      "A DCF-based Spatial Memory Network (SMN) that predicts a target location map from target/background masks and yields the background mask the AMN needs.",
      "An uncertainty-based memory update that keeps a fixed-length queue of past uncertainty values and removes a stored sample whenever the current uncertainty exceeds the queue average.",
      "Box-to-segmentation training and testing so one tracker answers both the box protocol (VOT) and the mask protocol (VOS), with state-of-the-art results on VOT2016–2020, GOT-10k, TrackingNet and DAVIS16/17.",
    ],
    method: {
      pipeline: ["encode", "read-memory", "localize", "segment", "update"],
      architecture:
        "A shared ResNet-50 encoder (block Conv4e) produces feature map fM for the memory branch and fQ for the query branch. The AMN maps fM to keys KA and values VA and fQ to query Qt and query value VQ, retrieves appearance values by non-local attention and concatenates them with the query value. The SMN trains a correlation filter on target/background masks to get a location map; a decoder then predicts a pixel mask, from which the reported bounding box is obtained.",
      appearanceModel:
        "AMN: non-local read over keys/values of sampled past frames, with a summed spatial matrix added to the read-out features as positional encoding.",
      detectionDependency:
        "None: a single object is given by its first-frame box (or mask when available).",
      trackManagement:
        "Frames are stored at a fixed sampling interval (best EAO at 5). A fixed-length queue (max length L) of past uncertainty values is kept; a stored prediction is preserved if Ut < Uavg,t and removed otherwise, so low-quality samples never enter the memory.",
      optimization:
        "The SMN filter minimizes the ridge-regularized correlation objective (Eq. 6); pseudo-masks for initialization are segmented from the first-frame box.",
    },
    equations: [
      {
        id: "samn-memory-encode",
        label: "Memory key/value and query encoding",
        formula: "KA = KeyA(fM), VA = ValA(fM);  Qt = QueA(fQ), VQ,t = ValQ(fQ)",
        variables: [
          { symbol: "fM", meaning: "feature map of the memory (past/target) branch" },
          { symbol: "fQ", meaning: "feature map of the query (current frame) branch" },
          { symbol: "KA, VA", meaning: "keys and values stored in the appearance memory" },
          { symbol: "Qt, VQ,t", meaning: "query vector and query value for frame t" },
        ],
        intuition:
          "Each frame is written into the memory as a key (where to look) and a value (what it looked like), and the current frame is asked as a query.",
        why:
          "It turns tracking from 'compare with one template' into 'retrieve from a dictionary of past appearances', which is what lets the model survive appearance change.",
        where: "Section 3, Eqs. (1)–(2): memory encoding for the AMN.",
        params: "The memory is refreshed only every 'Interv.' frames (best EAO at interval 5) and only for samples that pass the uncertainty gate.",
        paperIds: ["T050"],
      },
      {
        id: "samn-read-out",
        label: "Read-out value",
        formula: "Rt = concat[VA,t, VQ,t]",
        variables: [
          { symbol: "VA,t", meaning: "appearance value retrieved from the stored memory for frame t" },
          { symbol: "VQ,t", meaning: "query value of the current frame" },
        ],
        intuition:
          "Fuse what the memory says the target looked like with what the current frame actually shows.",
        why:
          "The retrieved history alone can drift and the current frame alone has no history; concatenating them gives the decoder both.",
        where: "Section 3, Eq. (5): output of the non-local read operation.",
        paperIds: ["T050"],
      },
      {
        id: "samn-dcf-objective",
        label: "Correlation filter training objective (SMN)",
        formula: "f* = arg min_f Σ_k ||<x_k^p, f> − y_k||² + λ||f||²₂",
        variables: [
          { symbol: "x_k^p", meaning: "p-th training sample feature (target or background patch)" },
          { symbol: "f", meaning: "correlation filter to learn" },
          { symbol: "y_k", meaning: "desired Gaussian response for sample k" },
          { symbol: "λ", meaning: "ridge regularization weight" },
        ],
        intuition:
          "Find the filter whose correlation with each training patch is as close as possible to a Gaussian peak at the target, with a penalty on large filters.",
        why:
          "This is the classical fast discriminative-localization term: it gives a dense location map in one FFT instead of a sliding-window classifier.",
        where: "Section 3, Eq. (6): training of the spatial memory network.",
        params: "Larger λ smooths the filter and reduces over-fitting to background clutter.",
        paperIds: ["T050"],
      },
      {
        id: "samn-uncertainty-gate",
        label: "Uncertainty-gated memory update",
        formula:
          "Uavg,t = Σ_k Uk;  Dt = preserved if Ut < Uavg,t,  removed if Ut > Uavg,t",
        variables: [
          { symbol: "Ut", meaning: "uncertainty of the current frame's prediction" },
          { symbol: "Uavg,t", meaning: "average uncertainty over the stored queue of length L" },
          { symbol: "Dt", meaning: "decision to keep or discard the stored sample" },
        ],
        intuition:
          "A frame is only worth remembering if the tracker was less unsure about it than it usually is.",
        why:
          "Unfiltered model update was the main failure mode of previous mask trackers — one occluded frame poisoned every later prediction.",
        where: "Section 3, Eqs. (10)–(11): sample filtering before the memory write.",
        params: "Queue length L and the sampling interval control how much history is considered.",
        paperIds: ["T050"],
      },
    ],
    datasets: ["vot", "got10k", "trackingnet", "others"],
    metrics: ["eao", "success-auc", "precision", "norm-precision"],
    baselines: ["D3S", "SiamMask", "ATOM", "DiMP-50", "SiamRPN++", "Ocean"],
    results: [
      "VOT2016: EAO 0.535, accuracy 0.684, robustness 0.121 — beats D3S (0.493), SiamMask-opt (0.442) and ATOM (0.430); the paper claims the first published tracker past 0.53 EAO and 0.68 accuracy.",
      "VOT2018: EAO 0.506, accuracy 0.652, robustness 0.145, ahead of D3S (0.489), Ocean-offline (0.467) and DiMP-50 (0.440).",
      "VOT2019: EAO 0.356, accuracy 0.649, robustness 0.326 — +2.9% EAO over Ocean-offline (0.327).",
      "VOT2020 (mask protocol): EAO 0.453, accuracy 0.711, robustness 0.776 — +1.2% over DET50 (0.441) and +14.5% over the VOS method STM (0.308).",
      "GOT-10k: AO 61.5 and SR75 52.2 versus DiMP-50 (61.1 / 49.2), Ocean-online (61.1 / 47.3) and D3S (59.7 / 46.2); TrackingNet: precision 69.7, normalized precision 79.4, success 74.2 — +0.2% success over DiMP-50 (74.0) and +3.3% accuracy over D3S.",
      "DAVIS16/17: J&F 77.3 (J 79.0, F 75.5) and 66.3 (J 64.8, F 67.7) versus D3S 74.0 / 60.8 and SiamMask 69.8 / 56.4 — +3.3 and +5.5 J&F over D3S.",
    ],
    ablations: [
      "Memory sampling interval (Fig. 14): storing only the first frame (interval 0) costs 6.2% EAO, 8.6% accuracy and 4.2% robustness relative to storing every sample; interval 1 gives the top accuracy (0.663), interval 5 the top EAO (0.506) and always appending the last frame improves EAO by 3.9% and robustness by 2.8%, while interval 30 lowers EAO by 7.9% versus interval 5.",
      "Box-to-segmentation strategy: EAO improves by 1.4% (0.492 → 0.506) and accuracy rises from 0.635 to 0.652.",
      "Removing the SMN's uncertainty-based sample filtering (interval 5) drops EAO from 0.506 to 0.421 and DAVIS16 J&F from 77.3 to 69.1.",
      "Positional encoding: adding a spatial matrix to the read-out beats concatenating coordinate channels by 2% EAO (0.506 vs 0.486).",
    ],
    limitations: {
      authorStated: [
        "Future work: improve the memory architecture, especially 'efficient memory management', and make the AMN and SMN 'more collaborative and unified'.",
        "The authors want a single model that is state-of-the-art on both VOT and VOS 'while keeping real-time inference speed' — implying that this is not yet achieved together.",
      ],
      evident: [
        "No frame rate is printed anywhere in the paper, so the 'fast inference speed' claim is unquantified; the reported box is derived from the predicted mask, so box accuracy is capped by mask quality with no independent box-regression refinement.",
        "The uncertainty gate compares Ut only with the running average of a fixed-length queue (hand-set L and sampling interval), so update behaviour is controlled by two hand-tuned hyper-parameters rather than learned.",
        "Localization is purely appearance-driven — there is no motion model, so a fully occluded target is recovered only when appearance memory re-matches it.",
      ],
    },
    assumptions: [
      "The target is a single object given by its first-frame box (a ground-truth mask is used when the protocol provides one).",
      "Past frames are worth remembering only if their uncertainty is below the running average — appearance change is treated as noise rather than as an explicit state to model.",
    ],
    computation:
      "PyTorch implementation on a shared ResNet-50 (block Conv4e) encoder; the paper states fast inference but reports no FPS or FLOP numbers.",
    relations: [
      {
        to: "T027",
        type: "uses-as-baseline",
        note:
          "SiamMask is compared on VOT (EAO) and on DAVIS16/17, where it scores J&F 69.8 / 56.4 against this tracker's 77.3 / 66.3.",
      },
      {
        to: "T026",
        type: "uses-as-baseline",
        note: "ATOM is the strong filter baseline: on VOT2016 this paper beats it by 10.2% EAO (0.535 vs 0.430).",
      },
      {
        to: "T031",
        type: "uses-as-baseline",
        note:
          "DiMP-50 is the closest competitor: VOT2018 EAO 0.440 vs 0.506, TrackingNet success 74.0 vs 74.2, GOT-10k AO 61.1 vs 61.5.",
      },
      {
        to: "T046",
        type: "uses-as-baseline",
        note: "Ocean-offline is compared on VOT2018/2019/2020 and GOT-10k (AO 61.1 online vs 61.5 here).",
      },
    ],
    concepts: ["memory-network", "correlation-filter", "appearance-features", "siamese"],
    impact:
      "The explicit two-memory decomposition (appearance memory + DCF spatial memory) with an uncertainty-gated write made 'what should the tracker remember?' a first-class design question, and its box-to-segmentation bridge is the standard trick for comparing box-based VOT and mask-based VOS results with one model.",
  },
  {
    id: "T051",
    arxiv: "2011.10875",
    title: "Transparent Object Tracking Benchmark",
    shortTitle: "TOTB",
    year: 2020,
    authors: [
      "Heng Fan",
      "Halady Akhilesha Miththanthaya",
      "Harshit",
      "Siranjiv Ramana Rajan",
      "Xiaoqiong Liu",
      "Zhilin Zou",
      "a.o.",
    ],
    fileName: "2011.10875v2.pdf",
    task: "benchmark",
    tags: ["benchmark", "transparent-object", "dataset", "appearance-features", "correlation-filter", "transferable-feature"],
    difficulty: "intermediate",
    summary:
      "TOTB is the first large-scale single-object tracking benchmark for transparent objects: 225 videos (86K frames) spanning 15 everyday categories annotated with 12 challenge attributes, on which 25 trackers are evaluated with precision, normalized precision and success. The paper also proposes TransATOM, which fuses a transferable transparency feature (an FCN with ResNet-18 trained on 2,844 static images) into ATOM, lifting ATOM's TOTB success from 0.614 to 0.641 at 26 FPS.",
    problem:
      "Transparent objects (bottles, cups, jars, bulbs) show the background through their bodies, so their colour, texture and boundary statistics are literally those of the background; correlation filters and discriminative classifiers trained on opaque objects drift immediately. No benchmark measured how badly transparent targets fail, or which of the twelve difficulty factors (occlusion, scale change, clutter, motion blur, …) hurt them most.",
    background: [
      "benchmark-design",
      "appearance-features",
      "correlation-filter",
      "deep-detection",
      "iou",
      "sot",
    ],
    previousWork: [
      {
        name: "Generic SOT benchmarks (OTB, TrackingNet, LaSOT)",
        limitation:
          "Objects are opaque and transparency is not an annotated attribute, so failures specific to see-through targets were neither standardized nor diagnosed.",
        whyThisPaper:
          "TOTB annotates 12 attributes over 225 transparent sequences and defines the one-pass protocol, making transparent tracking a comparable, reproducible task.",
      },
      {
        name: "ATOM / DiMP / PrDiMP (IoU-target classification)",
        limitation:
          "Their classifier is trained on features in which a transparent target's interior contains background pixels, so the classification map has no distinct foreground signature to learn.",
        whyThisPaper:
          "TransATOM appends a dedicated transparency feature channel, trained from segmentation masks, where transparent regions become foreground instead of background.",
      },
      {
        name: "Colour-fusion trackers (Staple, ECO, SRDCF)",
        limitation:
          "Colour histograms mix the object's colour with whatever is seen through it, so the colour cue actively misleads on transparent targets.",
        whyThisPaper:
          "The transparency feature comes from a segmentation FCN over shape/opacity cues rather than raw colour, so it survives background change.",
      },
    ],
    researchGap:
      "Before this paper there was no standardized transparent-object tracking benchmark, and no tracker explicitly modelled the transparency attribute.",
    contribution: [
      "TOTB: 225 transparent-object sequences, 86K frames, 15 object categories, 12 tracking attributes, one-pass evaluation with precision / normalized precision / success.",
      "First comprehensive evaluation of 25 trackers (correlation-filter, Siamese and deep families) on transparent objects, with full per-attribute results.",
      "TransATOM: an FCN transparency feature (ResNet-18, trained on 2,844 static images) fused with ATOM's classification feature, running at 26 FPS.",
      "Transferability study: the same feature plugged into DiMP and KYS improves them by 1.9% and 2.2% success.",
    ],
    method: {
      pipeline: ["segment", "concatenate-features", "classify", "estimate-iou", "update"],
      architecture:
        "TransATOM = ATOM with one extra branch. An FCN with a ResNet-18 backbone segments transparent pixels and emits feature xtrs; this is concatenated with ATOM's classification feature xcls (X = xcls ⊕ xtrs) and fed to the existing classification network. Target scale is handled by ATOM's IoU-Net unchanged; everything else (classification feature branch, IoU-Net) is directly borrowed from the ATOM baseline.",
      appearanceModel:
        "Two-channel appearance evidence: the generic classification feature plus the transparency feature that highlights see-through regions.",
      detectionDependency: "None: single object given by the first-frame bounding box.",
      optimization:
        "The classifier is learned with an L2-regularized regression onto Gaussian labels (Eq. 3); sample weighting γj and per-layer regularization λk follow ATOM's optimization, and the IoU-Net is trained as in ATOM.",
    },
    equations: [
      {
        id: "totb-fcn",
        label: "Transparency segmentation network",
        formula: "f(X; w) = φ2(w2 ∗ φ1(w1 ∗ X))",
        variables: [
          { symbol: "X", meaning: "input image (or image patch) fed to the FCN" },
          { symbol: "w1, w2", meaning: "weights of the two convolutional stages" },
          { symbol: "φ1, φ2", meaning: "non-linear activations after each stage" },
        ],
        intuition: "A two-stage fully convolutional net labels every pixel as transparent or not.",
        why:
          "Transparency has to be detected as a spatial property of the image, which a global image-classification feature cannot express.",
        where: "Section 4, Eq. (1): the transparency feature extractor (ResNet-18 backbone).",
        params: "Trained on 2,844 static images containing only small, movable transparent objects.",
        paperIds: ["T051"],
      },
      {
        id: "totb-feature-concat",
        label: "Feature fusion",
        formula: "X = xcls ⊕ xtrs",
        variables: [
          { symbol: "xcls", meaning: "pre-trained image classification feature used by ATOM" },
          { symbol: "xtrs", meaning: "transparency feature from the segmentation FCN" },
          { symbol: "⊕", meaning: "channel-wise concatenation" },
        ],
        intuition: "Give the classifier two opinions per location: what the object looks like, and where the see-through region is.",
        why:
          "Neither cue alone separates a transparent target from its background; concatenating keeps both available to the same correlation classifier.",
        where: "Section 4, Eq. (2): input construction for the tracking classifier.",
        paperIds: ["T051"],
      },
      {
        id: "totb-classifier-loss",
        label: "Classifier training loss",
        formula: "ℓw = Σ_j γj ||f(Xj; w) − Yj||₂ + Σ_k λk ||wk||₂",
        variables: [
          { symbol: "Xj, Yj", meaning: "j-th training sample and its Gaussian label centred on the target" },
          { symbol: "γj", meaning: "weight of sample j" },
          { symbol: "λk", meaning: "regularization amount for layer k" },
        ],
        intuition:
          "Fit the classifier so each sample produces a Gaussian peak at the target, weighting samples and regularizing each layer.",
        why:
          "It is the same discriminative-correlation learning used by ATOM, so the only novelty credited to TransATOM is the input feature, not the loss.",
        where: "Section 4, Eq. (3): learning and updating the tracking classifier.",
        params: "γj controls foreground/background balance; λk controls over-fitting per layer.",
        paperIds: ["T051"],
      },
    ],
    datasets: ["transparent-otb"],
    metrics: ["precision", "norm-precision", "success-auc", "fps"],
    baselines: ["ATOM", "DiMP", "PrDiMP", "SiamRPN++", "SiamMask", "KYS"],
    results: [
      "TransATOM on TOTB: precision 0.668, normalized precision 0.747, success 0.641 — best on all three metrics among 25 trackers.",
      "Against its baseline ATOM (0.641 / 0.717 / 0.641 PRE / NPRE / SUC) TransATOM gains 4.1%, 3.0% and 2.7% absolute.",
      "Runner-ups: SiamRPN++ has the second-best precision (0.647), SiamMask the second-best normalized precision (0.724), PrDiMP the second-best success (0.633) — leads of 2.1%, 2.3% and 0.8%.",
      "Attribute results: success 0.635 on partial occlusion and 0.604 on scale variation (both ahead of PrDiMP's 0.621 / 0.598); on rotation PrDiMP wins with 0.592 versus TransATOM's 0.591.",
      "Speed: TransATOM runs at 26 FPS while its baseline ATOM runs at 37 FPS.",
    ],
    ablations: [
      "Backbone depth (Table 3, TOTB success): switching ResNet-18 → ResNet-50 *lowers* ATOM 0.614 → 0.608, DiMP 0.605 → 0.594, PrDiMP 0.639 → 0.633 and TransATOM 0.641 → 0.632, while SiamRPN++ improves 0.585 → 0.617.",
      "Transparency feature alone (Table 4): visual-only TransATOM-V reaches 0.625 versus ATOM's 0.614 (+1.1%), and adding the transparency feature pushes it to 0.641 at 26 FPS.",
      "Transferability (Table 5): DiMP 0.594 → TransDiMP 0.613 (+1.9%) and KYS 0.597 → TransKYS 0.619 (+2.2%) with the same feature.",
    ],
    limitations: {
      authorStated: [
        "The transparency branch is trained on only 2,844 static images and only segments 'small and movable transparent objects' — the stated annotation scope of the feature.",
        "The authors explicitly report that deeper backbones do not help: on transparent targets ResNet-50 degrades ATOM, DiMP, PrDiMP and TransATOM (only the Siamese SiamRPN++ benefits).",
      ],
      evident: [
        "Scores remain low overall (best success 0.641 of 1.0), so transparent tracking is shown to be far from solved by any of the 25 methods.",
        "TOTB's success is defined as the percentage of frames with IoU > 0.5, not the OTB-style AUC, so its success values are not directly comparable with OTB/LaSOT success curves.",
        "The transparency FCN is trained once offline and never updated online, so a target whose transparency appearance shifts (new background seen through it) is handled only through ATOM's slowly-updated classifier; average sequence duration is 12.7 s, so long-term re-identification after full occlusion is barely exercised.",
      ],
    },
    assumptions: [
      "Transparency is a learnable, transferable attribute shared across object categories, so one segmentation net can serve all 15 categories.",
      "Only small, movable transparent objects need to be segmented for the feature to be useful.",
    ],
    computation:
      "TransATOM runs at 26 FPS (ATOM baseline 37 FPS); the added branch is an FCN with a ResNet-18 backbone trained once offline.",
    relations: [
      {
        to: "T026",
        type: "builds-on",
        note:
          "TransATOM keeps ATOM's classification and IoU-Net branches unchanged and only adds the transparency feature input (0.614 → 0.641 success on TOTB).",
      },
      {
        to: "T031",
        type: "extends",
        note:
          "The transferability study plugs the transparency feature into DiMP, raising its TOTB success from 0.594 to 0.613.",
      },
    ],
    concepts: ["benchmark-design", "appearance-features", "deep-detection", "iou"],
    impact:
      "TOTB became the reference benchmark for transparency-aware SOT, and its counter-intuitive finding — deeper backbones hurt on transparent targets while a cheap segmentation feature helps — made attribute-specific feature design a standard move; the plug-one-feature-into-several-trackers transferability protocol is reused by later domain-specific benchmarks.",
  },
  {
    id: "T052",
    arxiv: "2011.11204",
    title: "Graph Attention Tracking",
    shortTitle: "SiamGAT",
    year: 2020,
    authors: ["Dongyan Guo", "Yanyan Shao", "Ying Cui", "Zhenhua Wang", "Liyan Zhang", "Chunhua Shen"],
    fileName: "2011.11204v1.pdf",
    task: "single-object",
    tags: ["siamese", "graph-attention", "anchor-free", "correlation", "appearance-features", "real-time"],
    difficulty: "intermediate",
    summary:
      "SiamGAT replaces the single global cross-correlation of Siamese trackers with a Graph Attention Module: every 1×1 cell of the template and search feature maps becomes a node of a complete bipartite graph, and learned pairwise attention propagates target information part-to-part into the search region. A target-aware area selection zeroes template cells outside the projected target box, and an anchor-free SiamCAR head predicts the box — giving GOT-10k AO 62.7, UAV123 success 0.646 at 70 FPS, and second place on LaSOT (success 0.539).",
    problem:
      "Global cross-correlation treats the template as one undifferentiated blob, so under pose and shape variation the matching is imprecise and similar-looking distractors win. Fixed-crop templates also waste most of their cells on background when the target has an extreme aspect ratio, and anchor-based heads add dataset-specific anchor hyper-parameters on top of that.",
    background: [
      "siamese",
      "attention",
      "anchor-free-head",
      "appearance-features",
      "deep-detection",
      "bounding-box",
    ],
    previousWork: [
      {
        name: "SiamFC / SiamRPN++ global correlation",
        limitation:
          "A single inner product between template and search features cannot express which part of the template corresponds to which part of the search region, so shape variation and distractors degrade the response map.",
        whyThisPaper:
          "GAM builds a complete bipartite graph over all 1×1 cells and learns pairwise similarity scores with softmax attention, making the match part-to-part instead of blob-to-blob.",
      },
      {
        name: "Fixed template crops (SiamFC-style 127×127 patch)",
        limitation:
          "The template always covers the full centre crop; for thin or elongated targets most template cells are background and flood the response map with background evidence.",
        whyThisPaper:
          "Target-aware area selection keeps only cells inside the ground-truth box projected onto the feature map (Eq. 6–7), so the template adapts to the target's scale and aspect ratio.",
      },
      {
        name: "Anchor-based heads (SiamRPN family)",
        limitation:
          "Anchor scales and ratios are hand-tuned, dataset-specific hyper-parameters, and classification runs at anchor centres while regression moves the box — an inconsistency between the two tasks.",
        whyThisPaper:
          "The anchor-free SiamCAR classification–regression head predicts foreground scores and four side distances at the same location, removing all anchor choices.",
      },
    ],
    researchGap:
      "No Siamese tracker before this one modelled template–search correspondence as a learned part-to-part attention graph, nor selected the template feature area according to the target's shape.",
    contribution: [
      "Graph Attention Module (GAM): a complete bipartite graph over template and search nodes with learned similarity (Eq. 1–2), softmax attention (Eq. 3), and aggregate-and-fuse updates (Eq. 4–5).",
      "Target-aware template feature-area selection driven by the projected ground-truth box (Eq. 6–7), making the template adaptive to object scale and aspect ratio.",
      "A simple Siamese graph attention tracker (GoogLeNet backbone + SiamCAR head, 127×127 template / 287×287 search) that is state-of-the-art on GOT-10k and UAV123 and second on LaSOT.",
    ],
    method: {
      pipeline: [
        "extract-features",
        "select-template-area",
        "compute-attention",
        "aggregate-fuse",
        "classify-regress",
      ],
      architecture:
        "Shared-weight GoogLeNet (Inception v3) backbone → GAM over 1×1 cells of template map Ft and search map Fs → SiamCAR anchor-free head (category branch + 4-side offset regression) reading the fused response map. Linear transformations are implemented as 1×1 convolutions so all correlation scores are one matrix multiplication.",
      appearanceModel:
        "Part-to-part graph attention: search node i aggregates target nodes j weighted by learned attention aij, then fuses the aggregate with its own feature.",
      detectionDependency: "None: first-frame object only; the template is fixed for the whole sequence.",
      optimization:
        "SGD, mini-batch 76, 20 epochs; learning rate linearly warmed up 0.005 → 0.01 over the first 5 epochs then exponentially decayed to 0.0005 over the last 15. Backbone frozen for the first 10 epochs (training graph attention + head), then stages 1–2 of GoogLeNet frozen while stages 3–4 fine-tune. Training data: COCO, ImageNet DET/VID, YouTube-BB, GOT-10k (GOT-10k and LaSOT experiments use only their official training splits).",
    },
    equations: [
      {
        id: "siamgat-pair-score",
        label: "Pairwise correlation score",
        formula: "eij = f(hi^s, hj^t) = (Ws hi^s)ᵀ(Wt hj^t)",
        variables: [
          { symbol: "hi^s", meaning: "feature vector of search node i (a 1×1 cell of the search map)" },
          { symbol: "hj^t", meaning: "feature vector of template node j" },
          { symbol: "Ws, Wt", meaning: "learned linear transformations applied to search and template nodes" },
          { symbol: "eij", meaning: "correlation score of the pair (i, j)" },
        ],
        intuition:
          "Transform both sides into a learned space and take their inner product to ask 'how much does this template part explain this search part?'.",
        why:
          "A learned bilinear score is strictly more expressive than the fixed inner product of SiamFC, and it is what the softmax attention normalizes.",
        where: "Section 3.1, Eqs. (1)–(2): scores on the complete bipartite graph before normalization.",
        params: "Batch normalization after each 1×1 convolution is reported to help, and is used in practice.",
        paperIds: ["T052"],
      },
      {
        id: "siamgat-attention",
        label: "Attention normalization and aggregation",
        formula: "aij = exp(eij) / Σ_{k∈Vt} exp(eik);  vi = Σ_{j∈Vt} aij · Wv hj^t;  ĥi^s = ReLU(vi ∥ (Wv hi^s))",
        variables: [
          { symbol: "aij", meaning: "attention that search node i pays to template node j (sums to 1 over j)" },
          { symbol: "vi", meaning: "target information aggregated into search node i" },
          { symbol: "Wv", meaning: "linear transformation applied to template features before aggregation" },
          { symbol: "∥", meaning: "vector concatenation" },
        ],
        intuition:
          "Each search location reads a weighted average of all template parts, then keeps its own feature alongside the read summary.",
        why:
          "It lets every search location see the whole target while weighting only the parts that resemble it, which is what handles pose and shape variation.",
        where: "Section 3.1, Eqs. (3)–(5): softmax over template nodes, aggregation, and the fuse step.",
        params: "The graph is complete (all |Vs|×|Vt| edges), so cost grows with the number of feature cells.",
        paperIds: ["T052"],
      },
      {
        id: "siamgat-target-aware",
        label: "Target-aware template area selection",
        formula: "F̂t(i, j, :) = Ft(i, j, :) if (i, j) ∈ Rt,  0 otherwise",
        variables: [
          { symbol: "Rt", meaning: "region of interest: the labelled template bounding box projected onto the feature map" },
          { symbol: "Ft", meaning: "template feature map" },
          { symbol: "F̂t", meaning: "template feature map with background cells zeroed" },
        ],
        intuition: "Blank out every template cell that lies outside the target, so the graph only sees target parts.",
        why:
          "For extreme aspect ratios the fixed crop is mostly background; zeroing keeps the node set scale-invariant and lets batch normalization be applied to the varying-size tensor.",
        where: "Section 3.2, Eqs. (6)–(7): applied before the graph is built.",
        paperIds: ["T052"],
      },
    ],
    datasets: ["got10k", "uav123", "otb", "lasot"],
    metrics: ["success-auc", "precision", "norm-precision", "fps"],
    baselines: ["SiamCAR", "Ocean", "SiamFC++", "SiamRPN++", "ECO", "ATOM"],
    results: [
      "GOT-10k (protocol-compliant training): AO 62.7, SR0.5 74.3, SR0.75 48.8 — best in the table, +4.8 / +6.6 / +5.1 over the baseline SiamCAR (57.9 / 67.7 / 43.7) and +1.6 / +2.2 / +1.5 over Ocean-online (61.1 / 72.1 / 47.3).",
      "UAV123: success 0.646, precision 0.843 — ahead of SiamBAN (0.631 / 0.833) and SiamCAR (0.623 / 0.813).",
      "OTB-100: success 71.0%, ahead of the baseline SiamCAR by 2.3 points in success and 3.0 points in precision.",
      "LaSOT: success 0.539, precision 0.530, normalized precision 0.633 — second best overall, behind only the online-update tracker.",
      "Speed: 70 FPS with GoogLeNet (165 FPS with AlexNet).",
    ],
    ablations: [
      "Backbone (UAV123): AlexNet → GoogLeNet raises success 0.592 → 0.646 and precision 0.779 → 0.843, while speed drops 165 → 70 FPS.",
      "Target-aware template selection (GoogLeNet + GAM): adding it lifts success 0.626 → 0.646 and precision 0.822 → 0.843 (71 → 70 FPS).",
      "Embedding type without target-aware selection: graph attention (GAM) gives 0.626 / 0.822 versus depthwise cross-correlation 0.615 / 0.815 — +1.1 success and +0.7 precision.",
    ],
    limitations: {
      authorStated: [
        "The template is taken from the initial frame and 'fixed for the whole tracking period', so the model never refreshes its target model.",
        "The ablation table shows the speed cost explicitly: moving from AlexNet to GoogLeNet cuts FPS from 165 to 70.",
      ],
      evident: [
        "The complete bipartite graph has |Vs|×|Vt| edges, so memory and compute grow quadratically with feature-map size — the reason the method caps inputs at 127×127 / 287×287.",
        "LaSOT success 0.539 still trails the best online-update trackers, confirming that one-shot matching without model update remains behind adaptive classifiers on long sequences.",
        "Attention is computed only between template and search of the same pair of frames — there is no temporal memory or motion model, so occlusion recovery relies purely on the frozen first-frame template.",
      ],
    },
    assumptions: [
      "The first-frame crop contains the whole target, so the projected box Rt reliably marks target cells.",
      "Part-to-part correspondence learned from static image pairs transfers to video, since no temporal term enters the graph.",
    ],
    computation:
      "70 FPS with GoogLeNet, 165 FPS with AlexNet; 127×127 template and 287×287 search inputs; implemented in PyTorch on 4×RTX-2080Ti.",
    relations: [
      {
        to: "T009",
        type: "addresses-limitation",
        note:
          "Replaces SiamFC's single global cross-correlation with a learned part-to-part attention graph, fixing the local-linear matching assumption.",
      },
      {
        to: "T046",
        type: "uses-as-baseline",
        note:
          "Ocean-online is the strongest GOT-10k competitor here (61.1 / 72.1 / 47.3 versus 62.7 / 74.3 / 48.8).",
      },
      {
        to: "T038",
        type: "uses-as-baseline",
        note: "SiamBAN is compared on UAV123 (success 0.631 versus 0.646) and other benchmarks.",
      },
    ],
    concepts: ["attention", "siamese", "anchor-free-head", "appearance-features"],
    impact:
      "SiamGAT showed that the Siamese matching step itself could be replaced by a learned attention mechanism rather than a fixed correlation operator — an argument later transformer-based trackers (TransT, STARK) would make at full scale — while its target-aware template cropping became a common trick for extreme-aspect-ratio targets.",
  },
  {
    id: "T053",
    arxiv: "2012.15460",
    title: "TransTrack: Multiple Object Tracking with Transformer",
    shortTitle: "TransTrack",
    year: 2020,
    authors: ["Peize Sun", "Jinkun Cao", "Yi Jiang", "Rufeng Zhang", "Enze Xie", "Zehuan Yuan", "a.o."],
    fileName: "2012.15460v2.pdf",
    task: "multi-object",
    tags: ["transformer", "end-to-end-mot", "joint-detection-and-tracking", "query-based", "pedestrian", "online"],
    difficulty: "intermediate",
    summary:
      "TransTrack is the first to pose MOT as a two-frame query prediction problem: a Deformable Transformer encoder processes the stacked previous and current frames, one set of learned object queries detects the current frame while a second set of track queries (features carried over from the previous frame) predicts where last frame's objects went. Matching the two predicted box sets yields the trajectories, giving 74.5 MOTA / 63.9 IDF1 on the MOT17 test set and 64.5 MOTA on MOT20, without any Kalman filter, Hungarian assignment or hand-tuned appearance ReID.",
    problem:
      "Tracking-by-detection pipelines glue together separate components — a detector, a motion predictor (Kalman), an assignment solver (Hungarian) and often an appearance network — each with its own loss and hand-tuned thresholds. The seams are where identities break: motion and appearance are never learned jointly with detection, and the association stage never sees image features.",
    background: [
      "mot",
      "tracking-by-detection",
      "end-to-end-mot",
      "data-association",
      "bounding-box",
      "attention",
      "kalman",
      "hungarian",
    ],
    previousWork: [
      {
        name: "SORT / DeepSORT (detector + Kalman + Hungarian)",
        limitation:
          "Association is a hand-written cost matrix solved by Hungarian matching; motion is a constant-velocity Kalman filter and appearance is an external ReID net, none of which are trained end-to-end with the detector.",
        whyThisPaper:
          "Track queries carry object features across frames and are matched to detections inside the network, so association becomes a learned prediction rather than an external optimization.",
      },
      {
        name: "CenterTrack (keypoint-based joint detection and tracking)",
        limitation:
          "It links objects with a learned offset to the previous frame, but the offset is a 2-D displacement regression that degrades under occlusion, and the pipeline still depends on previous-frame detections being correct.",
        whyThisPaper:
          "The track query is a full feature embedding of the previous object, so the network can match by appearance and geometry jointly, and object queries can still detect objects that were never propagated.",
      },
      {
        name: "DETR-style detectors (DETR, Deformable DETR)",
        limitation:
          "They solve detection as set prediction but have no mechanism to keep an identity across frames.",
        whyThisPaper:
          "Adding a second query set sourced from the previous frame's object features turns the same set-prediction machinery into a tracker, with Hungarian matching reused for box association.",
      },
    ],
    researchGap:
      "Before this paper no tracker propagated object identity by feeding previous-frame object features back into a Transformer decoder as a second query set, jointly detecting and associating in one forward pass.",
    contribution: [
      "Two decoder blocks (object detection and object propagation) over a shared Deformable Transformer, where the previous frame's object features become the track queries for the current frame.",
      "A simple post-processing step: Hungarian or IoU-based matching between the predicted detection-box set and tracking-box set, with unmatched detection boxes starting new tracklets.",
      "Track rebirth: an unmatched track can stay inactive for K = 32 frames and be re-matched later, improving robustness to occlusion.",
      "State-of-the-art-quality results on MOT17 (74.5 MOTA / 63.9 IDF1 / 80.6 MOTP) and MOT20 (64.5 MOTA) test sets under the private protocol.",
    ],
    method: {
      pipeline: ["extract-features", "detect", "propagate-queries", "predict-track-boxes", "associate", "rebirth"],
      architecture:
        "ResNet-50 backbone → Deformable Transformer encoder (6 layers, 8 heads) over the stacked previous+current frames with temporal encoding → two decoder blocks: the left one takes 500 learned object queries and predicts detection boxes for the current frame, the right one takes the previous frame's object features as track queries and predicts tracking boxes. The two output box sets are merged by matching.",
      association:
        "Post-decoding box matching: Hungarian assignment (default) or NMS merging between tracking boxes and detection boxes — the ablation shows both give identical 65.0 MOTA on the MOT17 validation split.",
      motionModel:
        "Implicit: the track query itself encodes motion, because it is the feature of the object predicted in the previous frame (beats a Kalman filter, see ablations).",
      reid:
        "No external ReID branch; identity information lives in the track-query features. An optional shared/independent ReID cross-attention variant was tested and did not beat the default.",
      detectionDependency:
        "Self-detected: the network produces its own boxes each frame (private protocol), pre-trained on CrowdHuman then fine-tuned on MOT17.",
      trackManagement:
        "A track that fails to match becomes 'inactive' for up to K = 32 frames, during which it may still be re-matched to detection boxes; unmatched detection boxes always start new tracklets.",
      optimization:
        "AdamW, batch 16, initial learning rate 2e-4 for the transformer and 2e-5 for the backbone, weight decay 1e-4; ImageNet-pretrained frozen-BN ResNet-50, Xavier-initialized transformer weights; 150 epochs with the learning rate dropped by 10× at epoch 100; augmentations: random horizontal flip, crop, scale, short side 480–800 px, long side ≤ 1333 px.",
    },
    equations: [
      {
        id: "transtrack-set-loss",
        label: "Set prediction loss over detections and tracks",
        formula: "L = λcls · L_cls + λL1 · L1 + λgiou · L_giou",
        variables: [
          { symbol: "L_cls", meaning: "classification loss over the object classes plus background" },
          { symbol: "L1", meaning: "L1 distance between predicted and ground-truth box coordinates" },
          { symbol: "L_giou", meaning: "generalized-IoU loss between predicted and ground-truth boxes" },
          { symbol: "λcls, λL1, λgiou", meaning: "loss weights balancing the three terms" },
        ],
        intuition:
          "Train the decoder outputs as a set: every predicted box must either cover a real object (class + box losses) or be confidently background.",
        why:
          "It is the DETR-style objective that lets the network output a variable number of boxes without NMS or anchor priors — the same loss supervises both detection boxes and tracking boxes.",
        where: "Section 3.1; applied to both decoder outputs (object queries and track queries).",
        paperIds: ["T053"],
      },
      {
        id: "transtrack-association",
        label: "Box-set association",
        formula: "matching(detection boxes, tracking boxes) → matched pairs ∪ unmatched detections → new tracklets",
        variables: [
          { symbol: "detection boxes", meaning: "decoder-1 outputs for the current frame" },
          { symbol: "tracking boxes", meaning: "decoder-2 outputs propagated from the previous frame" },
          { symbol: "unmatched detection", meaning: "a detection with no corresponding track — starts a new identity" },
        ],
        intuition:
          "Two overlapping predictions of the same person are two views of one track; leftover detections are new arrivals.",
        why:
          "Identity association is reduced to comparing two sets the network itself produced, instead of solving a Kalman+IoU+appearance cost matrix.",
        where: "Section 3.3 Inference: Hungarian assignment (default) or NMS merging of the two box sets.",
        params: "Rebirth keeps unmatched tracks inactive for K = 32 frames.",
        paperIds: ["T053"],
      },
    ],
    datasets: ["mot17", "mot20"],
    metrics: ["mota", "idf1", "motp", "mt-ml", "fp", "fn", "idsw", "fps"],
    baselines: ["FairMOT", "CenterTrack", "CSTrack", "TransCenter", "GSDT", "CorrTracker"],
    results: [
      "MOT17 test (private): MOTA 74.5, IDF1 63.9, MOTP 80.6, MT 46.8, ML 11.3, FP 28,323, FN 112,137, ID switches 3,663.",
      "MOT20 test (private): MOTA 64.5, IDF1 59.2, MOTP 80.0, MT 49.1, ML 13.6, FP 28,566, FN 151,377, ID switches 3,565 — ahead of FairMOT (61.8 / 67.3), CSTrack (66.6 / 68.6) and TransCenter (58.3 / 46.8) in MOTA.",
      "Context on MOT17: TransTrack's MOTA (74.5) is just below CSTrack (74.9) but ahead of FairMOT (73.7) and CenterTrack (67.8); its ID switches (3,663) are far above FairMOT's and CSTrack's, while the paper states MOTP and FN are its strengths.",
      "Speed: 6 decoder layers run at 10 FPS (1 decoder = 15 FPS, 3 decoders = 12 FPS).",
    ],
    ablations: [
      "Transformer architecture and queries (MOT17 val): plain Transformer 55.4 MOTA → DC5 59.0 → P3 59.3 → Deformable Transformer 65.0; object queries only 58.3 (association by output index), track queries only cannot act as a detector (FN 93.8%), both together 65.0 with FP 4.3% / FN 30.3% / IDs 0.4%.",
      "Matching and motion: matching tracking boxes to the previous frame 64.8 MOTA / 0.6% IDs versus matching to current detections 65.0 / 0.4%; Hungarian and NMS merging are identical (65.0); motion model none (IoU) 64.4 → Kalman 64.9 → track query 65.0, and at 4× subsampling the track query halves ID switches (0.5% vs 1.0%).",
      "ReID features: shared cross-attention 61.1 MOTA, independent cross-attention 64.7, none (default) 65.0 — an explicit ReID head hurts overall performance.",
      "Training and capacity: CrowdHuman-only 53.8 → MOT17-only 61.6 → CrowdHuman pre-train + MOT17 fine-tune 65.0; decoder layers 1 → 47.0 (15 FPS), 3 → 64.3 (12 FPS), 6 → 65.0 (10 FPS); input short side 540 px → 62.4, 800 px → 65.0, 1080 px → 59.2.",
    ],
    limitations: {
      authorStated: [
        "The authors explicitly note that the ID-switch score is 'inferior to SOTA methods' and call improving it a promising direction.",
        "The conclusion frames TransTrack as providing 'a novel perspective' rather than a leaderboard-sweeping method.",
      ],
      evident: [
        "3,663 ID switches on MOT17 test — roughly 3× the switches of ByteTrack-class trackers — showing that query propagation alone does not resolve long-term identity.",
        "Only 10 FPS at the default 6-decoder setting, below real-time for 30 FPS video.",
        "Trained and evaluated on pedestrian datasets only (MOT17/MOT20 + CrowdHuman), with no general-object MOT evidence; association happens after decoding, on raw boxes, so the network never supervises the matching itself.",
      ],
    },
    assumptions: [
      "Objects visible in frame t−1 can be found again in frame t by their feature embedding alone (no explicit appearance ReID or motion prior).",
      "Frames arrive at high enough rate that consecutive-frame feature similarity is a reliable cue (the paper's own 1×-subsampling ablation weakens the advantage).",
    ],
    computation:
      "ResNet-50 + Deformable Transformer (6 encoder / 6 decoder layers, 8 heads, 500 object queries); 10 FPS with 6 decoders, 15 FPS with 1; AdamW batch 16, 150 epochs.",
    relations: [
      {
        to: "T042",
        type: "builds-on",
        note:
          "Keeps CenterTrack's joint-detection-and-tracking framing and its track rebirth (K = 32 frames), but replaces keypoint-offset linking with track queries.",
      },
      {
        to: "T043",
        type: "uses-as-baseline",
        note: "FairMOT is compared on MOT17 (73.7 MOTA / 72.3 IDF1) and MOT20 (61.8 / 67.3).",
      },
      {
        to: "T005",
        type: "addresses-limitation",
        note:
          "Replaces SORT's Kalman filter and Hungarian cost matrix with learned query propagation, keeping only box-set matching as post-processing.",
      },
      {
        to: "T039",
        type: "uses-as-baseline",
        note: "Evaluated on MOT20, the crowded-scene benchmark, reaching 64.5 MOTA / 59.2 IDF1.",
      },
    ],
    concepts: ["end-to-end-mot", "attention", "data-association", "mot", "track-management"],
    impact:
      "TransTrack established the 'two query sets, two decoder blocks' pattern for transformer MOT — later query-based trackers (TrackFormer, MOTR, TransTrack successors) all build on the idea that identity should be carried by a query embedding rather than by a post-hoc cost matrix.",
  },
  {
    id: "T054",
    arxiv: "2101.02702",
    title: "TrackFormer: Multi-Object Tracking with Transformers",
    shortTitle: "TrackFormer",
    year: 2021,
    authors: ["Tim Meinhardt", "Alexander Kirillov", "Laura Leal-Taixé", "Christoph Feichtenhofer"],
    fileName: "2101.02702v3.pdf",
    task: "multi-object",
    tags: ["transformer", "end-to-end-mot", "tracking-by-attention", "query-based", "joint-detection-and-tracking", "pedestrian"],
    difficulty: "advanced",
    summary:
      "TrackFormer formulates MOT as tracking-by-attention: a Deformable-DETR encoder-decoder processes two adjacent frames, where learned object queries detect objects and autoregressive track queries — the temporally adapted embeddings of objects from frame t−1 — follow them into frame t. A single set-prediction loss jointly trains detection, identity preservation and trajectory forming, reaching 74.1 MOTA / 68.0 IDF1 on the MOT17 test set (private) and 68.6 MOTA / 65.7 IDF1 on MOT20 at 7.4 FPS.",
    problem:
      "MOT requires simultaneously deciding what objects exist (initialization), which box belongs to which identity (association) and how trajectories start and end — yet classic pipelines treat these as separate stages with separate heuristics: detection thresholds, IoU gating, Hungarian costs, ReID nets. Every interface is a place where identity is lost, and none of the stages are trained together.",
    background: [
      "mot",
      "tracking-by-detection",
      "end-to-end-mot",
      "attention",
      "data-association",
      "track-management",
      "bounding-box",
    ],
    previousWork: [
      {
        name: "Tracking-by-detection pipelines (SORT, CenterTrack)",
        limitation:
          "Track initialization, association and trajectory forming are separate steps (thresholds + matching), trained or tuned independently; identity decisions never see image features end-to-end.",
        whyThisPaper:
          "One set-prediction loss over all output embeddings trains initialization, identity and trajectory together — replacing heuristic matching with attention.",
      },
      {
        name: "Offline global-association methods",
        limitation:
          "They get strong identity scores by processing the whole sequence at once, but cannot run online.",
        whyThisPaper:
          "Track queries are autoregressive: identity at frame t depends only on frames ≤ t, so the model is strictly online while still matching offline identity scores.",
      },
      {
        name: "Detector + external ReID (DeepSORT, Tracktor)",
        limitation:
          "ReID features and detection features conflict (the paper's own ablation shows a shared ReID head lowers MOTA), and ReID nets need dedicated training.",
        whyThisPaper:
          "Track queries reuse the detector's cross-attention mechanism itself for short-term re-identification, with no separate ReID branch.",
      },
    ],
    researchGap:
      "Before this paper no tracker used autoregressive query embeddings to represent object identity across frames, so detection, association and trajectory logic were never optimized by a single objective.",
    contribution: [
      "Tracking-by-attention: object queries initialize tracks, track queries carry identities into the next frame, and both are decoded jointly by a Transformer decoder.",
      "A unified set-prediction loss (classification + box terms) applied to the joint Nobject + Ntrack output set, with hard assignment by track identity and Hungarian matching for new objects.",
      "Track-query re-identification: removed track queries stay decoded for a patience window (Ttrack-reid = 5) and can re-trigger if their score exceeds σtrack-reid = 0.4.",
      "Track augmentations (false-positive and frame-range perturbations) that teach the model recovery from missed detections and long gaps — worth +5.2 MOTA and +11.2 IDF1 together.",
      "State-of-the-art private MOT17 results (74.1 MOTA / 68.0 IDF1) and a unified box + mask formulation that also wins MOTS20 (54.9 MOTSA vs Track R-CNN's 40.6).",
    ],
    method: {
      pipeline: ["encode-pair", "detect-previous", "propagate-queries", "decode-outputs", "set-match", "manage-tracks"],
      architecture:
        "ResNet-50 → Deformable-DETR encoder and decoder, both 6 layers with 8 attention heads; Nobject = 500 learned object queries plus up to Ntrack track queries (previous-frame object embeddings adapted by cross-attention to the current frame) form the decoder input over two stacked adjacent frames. MLP heads predict class logits and boxes; deformable reference-point refinement included.",
      association:
        "Inside the loss: ground truth is assigned to track queries by track identity (hard assignment) and to the remaining queries by minimum-cost Hungarian matching (Eqs. 1–3). At inference no external matching is needed — each output embedding already carries its identity.",
      motionModel: "None; motion is implicit in the attention between the track query and the two-frame feature maps.",
      reid:
        "Attention-based, inside the decoder: track queries re-identify through their cross-attention to the current frame; a separate ReID head is deliberately not used (the ablation shows shared ReID hurts).",
      detectionDependency:
        "None: it is the detector, pre-trained on COCO (public model) or trained from scratch on CrowdHuman with simulated adjacent frames (private model).",
      trackManagement:
        "A track is initialized when an object-query output's score exceeds σdetection = 0.4; it is removed when its score falls below σtrack = 0.4 or when it duplicates another box with IoU > σNMS = 0.9. Removed track queries remain decoded for Ttrack-reid = 5 frames and re-enter the trajectory if their score exceeds σtrack-reid = 0.4.",
      optimization:
        "Two training stages, batch 2, initial learning rates 2e-4. Public model: fine-tune COCO-pretrained Deformable-DETR weights on MOT17 for 50 epochs (lr drop after 10). Private model: train 85 epochs from scratch on CrowdHuman with simulated adjacent frames (lr drop after 50), then fine-tune 40 epochs on CrowdHuman + MOT17 (lr drop after 10); ~2 days on 7 × 32 GB GPUs. Loss weights λcls = 2, λℓ1 = 5, λiou = 2; augmentations pFN = 0.4, pFP = 0.1, 1% image jitter.",
    },
    equations: [
      {
        id: "trackformer-matching",
        label: "Injective minimum-cost assignment",
        formula: "σ̂ = argmin_σ Σ_{k_i ∈ K_object} C_match(y_i, ŷ_{σ(i)});  C_match = −λ_cls · p̂_{σ(i)}(c_i) + C_box(b_i, b̂_{σ(i)});  C_box = λℓ1·||b_i − b̂_{σ(i)}||₁ + λiou·C_iou(b_i, b̂_{σ(i)})",
        variables: [
          { symbol: "y_i, ŷ_j", meaning: "ground-truth object i and prediction j (class, box, identity)" },
          { symbol: "σ", meaning: "permutation assigning ground truth to predictions" },
          { symbol: "K_object", meaning: "ground-truth objects not covered by an existing track identity" },
          { symbol: "p̂(c_i)", meaning: "predicted probability of class c_i (no log, following DETR)" },
          { symbol: "λcls, λℓ1, λiou", meaning: "cost weights, set to 2, 5 and 2" },
        ],
        intuition:
          "New objects are matched like DETR detections: pick the assignment that maximizes class confidence while minimizing box error.",
        why:
          "Track identities are assigned for free by identity, but objects that were not tracked yet still need a principled assignment — this is that step.",
        where: "Section 3.3, Eqs. (1)–(3): set prediction loss, first step.",
        params: "λcls = 2, λℓ1 = 5, λiou = 2; the IoU term gives size-invariant box errors.",
        paperIds: ["T054"],
      },
      {
        id: "trackformer-query-loss",
        label: "Per-query set prediction loss",
        formula:
          "L_query = −λcls·log p̂_i(c_{π=i}) + L_box(b_{π=i}, b̂_i)   if i ∈ π;   −λcls·log p̂_i(0)   if i ∉ π",
        variables: [
          { symbol: "i", meaning: "index of an output embedding (object query or track query)" },
          { symbol: "π(i)", meaning: "ground-truth object assigned to prediction i (if any)" },
          { symbol: "p̂_i(0)", meaning: "predicted probability of the background class for unmatched predictions" },
          { symbol: "L_box", meaning: "box loss computed like C_box but differentiable" },
        ],
        intuition:
          "Every output — whether it came from an object query or a track query — is scored: matched ones must hit their object, unmatched ones must say 'background'.",
        why:
          "This single loss is what jointly trains track initialization, identity preservation and trajectory forming, instead of three separate stages.",
        where: "Section 3.3, Eq. (4): loss for frame t over all N = Nobject + Ntrack outputs.",
        params: "λcls = 2 weights classification against the box terms.",
        paperIds: ["T054"],
      },
    ],
    datasets: ["mot17", "mot20", "others"],
    metrics: ["mota", "idf1", "fp", "fn", "idsw", "fps"],
    baselines: ["CenterTrack", "FairMOT", "Tracktor++", "TraDeS", "GSDT", "CTracker"],
    results: [
      "MOT17 test, public detections: MOTA 62.3, IDF1 57.6, FP 16,591, FN 192,123, ID switches 4,018 at 7.4 FPS — on par with state of the art without CrowdHuman pre-training.",
      "MOT17 test, private detections: MOTA 74.1, IDF1 68.0, FP 34,602, FN 108,777, ID switches 2,829 — a new state of the art (+5.0 MOTA, +1.7 IDF1) among methods trained only on CrowdHuman.",
      "MOT20 test, private: MOTA 68.6, IDF1 65.7, FP 20,348, FN 140,373, ID switches 1,532.",
      "MOTS20 test: MOTSA 54.9, IDF1 63.6 with 278 ID switches, versus Track R-CNN's 40.6 / 42.4 / 567; on the MOTS20 training split (4-fold) 58.7 vs Track R-CNN 52.7.",
      "Speed: 7.4 FPS (self-measured).",
    ],
    ablations: [
      "Component removal (MOT17 val, 50-50 split): full model 71.3 MOTA / 73.4 IDF1; without CrowdHuman pre-training 69.3 / 71.8; without track-query re-identification 69.2 / 70.4; without false-positive track augmentation 68.4 / 70.0; without frame-range track augmentation 64.0 / 59.2; without track queries (greedy center-distance matching à la previous heuristics) 61.0 / 45.1.",
      "Joint mask training (MOTS20 val, box metrics): +1.2 IDF1 (54.8 → 56.0) with unchanged MOTA 61.9 — mask supervision resolves ambiguous occlusions during training even when only boxes are evaluated.",
      "Track-query re-identification alone contributes almost nothing to MOTA (−0.1) but +1.4 IDF1.",
    ],
    limitations: {
      authorStated: [
        "Track queries embed spatial information, which the authors say 'prevents their application for long-term occlusions with large object movement' — the reason a patience window (Ttrack-reid = 5) exists at all.",
        "They expect a heavier or higher-resolution backbone (e.g. DC5) to improve accuracy and leave it to future work.",
        "The public-detection protocol requires a new track-initialization filtering step because the model does not consume external detections directly.",
      ],
      evident: [
        "MOT17 test ID switches (2,829) remain high relative to association-specialized methods; identity still relies on short-window re-identification rather than a global appearance model.",
        "7.4 FPS is below real-time, and the private model needs ~2 days on 7 × 32 GB GPUs plus CrowdHuman pseudo-track simulation.",
        "Trained and evaluated on pedestrians, with two-frame inputs and simulated adjacent frames that assume a specific frame rate and scene type; removing either track augmentation costs 5–11 IDF1 points, so performance depends heavily on synthetic training scenarios.",
      ],
    },
    assumptions: [
      "Objects that leave and re-enter the scene are recoverable within a short patience window (5 frames) by score alone.",
      "A single set-prediction loss with Hungarian matching for new objects is enough to teach both detection and identity — no separate association loss is needed.",
    ],
    computation:
      "Deformable-DETR (6 encoder + 6 decoder layers, 8 heads, 500 object queries) on a ResNet-50 backbone; 7.4 FPS; batch 2; private model training ≈ 2 days on 7 × 32 GB GPUs.",
    relations: [
      {
        to: "T042",
        type: "uses-as-baseline",
        note: "CenterTrack is the key joint-detection-and-tracking baseline (67.8 MOTA / 64.7 IDF1 on MOT17 test, private, CrowdHuman).",
      },
      {
        to: "T043",
        type: "uses-as-baseline",
        note: "FairMOT is compared on MOT17 test (61.8 MOTA / 67.3 IDF1 with 5 extra datasets).",
      },
      {
        to: "T053",
        type: "conceptual-successor",
        note:
          "Shares TransTrack's two-query-set decoder idea but replaces its post-hoc box matching with identity-based hard assignment inside the loss, and drops TransTrack's explicit Hungarian/NMS association.",
      },
      {
        to: "T056",
        type: "uses-as-baseline",
        note:
          "STARK ablates a TrackFormer-style query-update design against its template-update design, reporting 64.8 vs 66.4 LaSOT success.",
      },
    ],
    concepts: ["end-to-end-mot", "attention", "track-management", "data-association", "mot"],
    impact:
      "TrackFormer turned 'identity as an autoregressive query embedding' into a template for transformer MOT, and its unified loss over detections + propagated identities — plus its demonstration that heuristic matching costs 10+ IDF1 points — directly shaped follow-up work such as MOTR, which removes the external association step entirely.",
  },
  {
    id: "T055",
    arxiv: "2103.15436",
    title: "Transformer Tracking",
    shortTitle: "TransT",
    year: 2021,
    authors: ["Xin Chen", "Bin Yan", "Jiawen Zhu", "Dong Wang", "Xiaoyun Yang", "Huchuan Lu"],
    fileName: "2103.15436v1.pdf",
    task: "single-object",
    tags: ["transformer", "siamese", "attention", "feature-fusion", "anchor-free", "real-time"],
    difficulty: "intermediate",
    summary:
      "TransT keeps the Siamese template/search split but replaces the correlation operator with an attention-based feature fusion network: alternating ego-context (self-attention) and cross-feature (cross-attention) modules exchange information between the template and search branches N = 4 times, then an anchor-free three-layer-perceptron head classifies and regresses boxes directly. It runs at ~50 FPS and sets new records on LaSOT (AUC 64.9), TrackingNet (81.4) and GOT-10k (AO 72.3).",
    problem:
      "Correlation (depthwise cross-correlation) is a fixed, local, linear operation: it can only measure template–search similarity, cannot let the two feature maps talk to each other repeatedly, and its local-linear assumption breaks down for large appearance change. The field needed a learnable fusion that reasons about both branches jointly while staying fast enough for real-time tracking.",
    background: [
      "siamese",
      "attention",
      "appearance-features",
      "deep-detection",
      "correlation-filter",
      "bounding-box",
      "anchor-free-head",
    ],
    previousWork: [
      {
        name: "SiamFC / SiamRPN++ correlation",
        limitation:
          "Depthwise cross-correlation is a one-shot linear similarity: the template cannot be updated by what the search region reveals, and matching degrades under large deformation or similar distractors.",
        whyThisPaper:
          "Cross-Feature Augment modules let search features query template features (and vice versa) four times in a stack, so fusion is iterative and learned.",
      },
      {
        name: "Original Transformer (encoder-decoder) applied to tracking",
        limitation:
          "A vanilla Transformer's output size equals its input, forcing an awkward template-encoder / search-decoder split and giving weaker results in the paper's own ablation (LaSOT AUC 62.3 vs 64.9).",
        whyThisPaper:
          "The custom ECA/CFA fusion stack fuses two branches into a search-shaped output, which is what the detection head expects.",
      },
      {
        name: "Elaborate post-processing (scale penalty, cosine window, box smoothing as in SiamRPN++, Ocean)",
        limitation:
          "Three hand-tuned, dataset-sensitive hyper-parameters are needed to clean up the raw response.",
        whyThisPaper:
          "TransT keeps only a single Hanning-window penalty with one fixed weight (w = 0.49) used unchanged on every benchmark — the fused features already produce clean scores.",
      },
    ],
    researchGap:
      "Before this paper, no Siamese tracker replaced the correlation operator itself with a learned attention-based fusion network operating over both branches.",
    contribution: [
      "Attention-based feature fusion network built from two modules: Ego-Context Augment (multi-head self-attention, residual, with spatial positional encoding) and Cross-Feature Augment (multi-head cross-attention + FFN, residual), stacked as fusion layers repeated N = 4 times plus a final CFA.",
      "Anchor-free prediction head: two three-layer perceptrons (classification and regression) output foreground scores and normalized box coordinates directly — no anchors, no response-map tricks.",
      "A single post-processing parameter (Hanning window weight w = 0.49) shared across all test sets.",
      "State-of-the-art on LaSOT (64.9 AUC), TrackingNet (81.4 AUC / 86.7 Pnorm) and GOT-10k (72.3 AO / 82.4 SR0.5) at ~50 FPS.",
    ],
    method: {
      pipeline: ["extract", "project-256d", "fuse-x4", "decode", "penalize-window"],
      architecture:
        "Modified ResNet-50 backbone (stride-2 stage-4 changed to dilation to keep resolution) extracts template (128×128) and search (256×256) features; a 1×1 convolution projects both to d = 256; the fusion network runs N = 4 fusion layers of (2×ECA, 2×CFA) followed by one extra CFA to produce a search-sized feature map; a three-layer perceptron head per branch predicts class scores and normalized box coordinates.",
      appearanceModel:
        "Iterative mutual attention: ECA enriches each branch internally, CFA lets one branch read the other via cross-attention — 8 heads, dm = 256, dk = dv = 32, sine positional encodings.",
      detectionDependency: "None: single object from the first-frame box.",
      optimization:
        "AdamW trained on COCO + TrackingNet + LaSOT + GOT-10k image pairs (video pairs sampled from one sequence), backbone learning rate 1e-5 and 1e-4 for other parameters, batch 38 on 2 × Titan RTX, 1000 epochs × 1000 iterations with the learning rate divided by 10 after 500 epochs; ImageNet-pretrained ResNet-50, Xavier init elsewhere.",
    },
    equations: [
      {
        id: "transt-attention",
        label: "Scaled dot-product attention",
        formula: "Attention(Q, K, V) = softmax(QKᵀ / √dk) · V",
        variables: [
          { symbol: "Q, K, V", meaning: "queries, keys and values of the multi-head attention" },
          { symbol: "dk", meaning: "key dimensionality (32 in this work)" },
        ],
        intuition:
          "Each query scores every key by dot product, turns the scores into weights, and returns the weighted average of the values.",
        why:
          "It is the operation that replaces correlation: similarity is now learned and can be computed in either direction between the branches.",
        where: "Section 3.2, Eq. (1); used inside both ECA and CFA modules.",
        params: "8 heads with dm = 256 and dk = dv = dm/nh = 32.",
        paperIds: ["T055"],
      },
      {
        id: "transt-eca-cfa",
        label: "Ego-Context and Cross-Feature Augment",
        formula:
          "X_EC = X + MultiHead(X + Px, X + Px, X);  X̃_CF = Xq + MultiHead(Xq + Pq, Xkv + Pkv, Xkv);  X_CF = X̃_CF + FFN(X̃_CF)",
        variables: [
          { symbol: "X", meaning: "feature of the branch the module sits in" },
          { symbol: "Xq, Xkv", meaning: "query-side branch and key/value-side branch of the CFA" },
          { symbol: "Px, Pq, Pkv", meaning: "sine spatial positional encodings" },
          { symbol: "FFN", meaning: "two linear layers with a ReLU in between" },
        ],
        intuition:
          "ECA lets a branch refine itself; CFA lets one branch read the other and then passes the result through a feed-forward block, both residual.",
        why:
          "This is the actual replacement for correlation: fusion becomes iterative, learned, and bidirectional instead of a single fixed similarity map.",
        where: "Section 3.2, Eqs. (4)–(6); stacked as 2×ECA + 2×CFA per fusion layer, repeated N = 4 times, plus one final CFA.",
        params: "N = 4 fusion layers; spatial positional encodings added to every attention input.",
        paperIds: ["T055"],
      },
      {
        id: "transt-losses",
        label: "Classification and regression losses",
        formula:
          "L_cls = − Σ_j [y_j·log(p_j) + (1 − y_j)·log(1 − p_j)];  L_reg = Σ_j 1{y_j=1} [λG·L_GIoU(b_j, b̂) + λ1·L1(b_j, b̂)]",
        variables: [
          { symbol: "p_j, y_j", meaning: "predicted foreground probability and ground-truth label at location j" },
          { symbol: "b_j, b̂", meaning: "ground-truth and predicted normalized box coordinates" },
          { symbol: "λG, λ1", meaning: "weights of the GIoU and L1 terms" },
        ],
        intuition:
          "Binary cross-entropy scores every location; only positive locations are regressed, by how far the box is (L1) and how much it overlaps (GIoU).",
        why:
          "An anchor-free head needs an explicit regression objective since there are no anchor boxes to match to.",
        where: "Section 3.3, Eqs. (7)–(8): head training objective.",
        paperIds: ["T055"],
      },
      {
        id: "transt-window-penalty",
        label: "Hanning window score penalty",
        formula: "score_w = (1 − w)·score + w·score_h",
        variables: [
          { symbol: "score", meaning: "raw classification score predicted by the head" },
          { symbol: "score_h", meaning: "value of the 32×32 Hanning window at that location" },
          { symbol: "w", meaning: "penalty weight, fixed at 0.49" },
        ],
        intuition:
          "Discount scores far from the previous target position, then take the argmax box.",
        why:
          "It replaces the usual three-parameter post-processing (cosine window + scale penalty + smoothing) with one fixed constant used on every dataset.",
        where: "Section 4.1, Eq. (9): the only post-processing step.",
        params: "w = 0.49, Hanning window 32×32, identical default for all test sets.",
        paperIds: ["T055"],
      },
    ],
    datasets: ["lasot", "trackingnet", "got10k", "otb", "uav123", "others"],
    metrics: ["success-auc", "precision", "norm-precision", "fps"],
    baselines: ["SiamR-CNN", "Ocean", "PrDiMP", "DiMP", "ATOM", "SiamRPN++"],
    results: [
      "LaSOT test: AUC 64.9, Pnorm 73.8, precision 69.0 — best in the table, just ahead of SiamR-CNN (64.8 / 72.2 / 64.9).",
      "TrackingNet test: AUC 81.4, Pnorm 86.7, precision 80.3 — best among all compared methods (SiamR-CNN: 81.2 / 85.4 / 80.0).",
      "GOT-10k test: AO 72.3, SR0.5 82.4, SR0.75 68.2; the GOT-10k-protocol-trained variant (TransT-GOT) is stated to be 2.2% higher in AO than SiamR-CNN.",
      "NFS / OTB / UAV123 AUC: 65.7 / 69.4 / 69.1 versus PrDiMP 63.5 / 69.6 / 68.0 and SiamRPN++ 50.2 / 69.6 / 61.3.",
      "Speed: ~50 FPS on GPU, versus SiamR-CNN which runs at less than 5 FPS on the same machine.",
    ],
    ablations: [
      "Post-processing (Table 2): the Hanning penalty adds +2.0 LaSOT AUC (62.9 → 64.9) and +0.3 TrackingNet AUC (81.1 → 81.4); TransT-np without it is already state of the art.",
      "Proposed fusion vs original Transformer: TransT 64.9 vs TransT(ori) 62.3 LaSOT AUC; without post-processing 62.9 vs 60.9; TrackingNet AUC 81.4 vs 81.3.",
      "Correlation as fusion (Table 3): keeping only correlation drops LaSOT AUC to 57.7 (TrackingNet 77.5); correlation + ECA gives 62.9 / 81.1; with no fusion at all the tracker collapses to LaSOT AUC 35.3 and TrackingNet 46.5.",
      "The full ECA+CFA stack gives 64.9 / 81.4 — the 9.6-point gap over correlation-only (64.9 vs 57.7 LaSOT AUC) is the paper's core evidence.",
    ],
    limitations: {
      authorStated: [
        "SiamR-CNN is more accurate on some metrics but 'merely runs less than 5 fps', while TransT runs at 50 fps — the authors frame speed as a design constraint they accept.",
        "The fusion network takes only a pair of patches (template + search), so there is no explicit long-term memory or motion reasoning.",
      ],
      evident: [
        "OTB AUC 69.4 is slightly below PrDiMP and SiamRPN++ (both 69.6) — on small, classic benchmarks the transformer fusion buys little.",
        "All hyper-parameters (w = 0.49, N = 4, d = 256) are global constants, so there is no per-sequence adaptation; failure on a target unseen at match time cannot be corrected online.",
        "Template is fixed after the first frame (no update mechanism), so long-term occlusion recovery depends entirely on the frozen first-frame embedding; training requires four large datasets (COCO, TrackingNet, LaSOT, GOT-10k) with 1000 × 1000 iterations — heavy compared with earlier Siamese trackers.",
      ],
    },
    assumptions: [
      "A fixed first-frame template plus iterative cross-attention is sufficient to handle appearance change (no template update).",
      "The target stays roughly centred in the 256×256 search region, so only a mild window penalty is needed.",
    ],
    computation:
      "~50 FPS on GPU; 128×128 template, 256×256 search; modified ResNet-50 + N = 4 fusion layers (8 heads, d = 256); batch 38 on 2 × Titan RTX.",
    relations: [
      {
        to: "T009",
        type: "addresses-limitation",
        note:
          "Replaces SiamFC's depthwise cross-correlation — the fixed linear similarity — with a learned attention fusion stack.",
      },
      {
        to: "T028",
        type: "addresses-limitation",
        note:
          "Overcomes the local-linear assumption of SiamRPN++-style correlation; also discards its anchor boxes for an anchor-free head.",
      },
      {
        to: "T031",
        type: "uses-as-baseline",
        note: "PrDiMP (and DiMP-50) are the main filter-family competitors on LaSOT, NFS, OTB and UAV123.",
      },
      {
        to: "T046",
        type: "uses-as-baseline",
        note: "Ocean is compared on LaSOT, TrackingNet and GOT-10k.",
      },
    ],
    concepts: ["attention", "siamese", "appearance-features", "correlation-filter", "anchor-free-head"],
    impact:
      "TransT replaced the correlation operator with an iterative attention fusion stack and proved a single fixed post-processing constant (w = 0.49) could stand in for three hand-tuned penalties — making learned fusion, not depthwise correlation, the reference design for Siamese transformer trackers.",
  },
  {
    id: "T056",
    arxiv: "2103.17154",
    title: "Learning Spatio-Temporal Transformer for Visual Tracking",
    shortTitle: "STARK",
    year: 2021,
    authors: ["Bin Yan", "Houwen Peng", "Jianlong Fu", "Dong Wang", "Huchuan Lu"],
    fileName: "2103.17154v1.pdf",
    task: "single-object",
    tags: ["transformer", "attention", "template-update", "anchor-free", "corner-regression", "real-time"],
    difficulty: "intermediate",
    summary:
      "STARK feeds a transformer encoder-decoder a triplet — the search region, the initial template and a dynamically updated template — and decodes a single target query into two corner heatmaps, so the box comes from a fully-convolutional keypoint head instead of anchors or a correlation map. A second score head decides whether the dynamic template is reliable enough to be refreshed, and the whole pipeline runs with no post-processing (no cosine window, no box smoothing) at over 40 FPS. It reaches 68.8 AO on GOT-10k, 82.0 AUC on TrackingNet, 67.1 success on LaSOT and 0.308 EAO on VOT2020.",
    problem:
      "Correlation-based Siamese trackers only measure local similarity against a fixed template, while the spatio-temporal trackers that fix this either back-propagate gradients at test time (MDNet, DiMP) or hand identity over by propagating queries between frames (TransTrack, TrackFormer) without ever refreshing an explicit target model. DETR-style heads, meanwhile, regress boxes with a three-layer perceptron and train localization and classification jointly, which the paper shows is sub-optimal. What was missing was an end-to-end way to update the target appearance under a learned confidence gate.",
    background: ["siamese", "attention", "appearance-features", "track-management", "bounding-box", "iou"],
    previousWork: [
      {
        name: "Correlation-based spatial-only Siamese trackers (SiamFC, SiamRPN++, Ocean) and gradient-based spatio-temporal trackers (MDNet, DiMP)",
        limitation:
          "Correlation in its naive, depthwise or point-wise forms 'merely captures local similarity, ignoring global information', and the gradient-based trackers that do model change need back-propagation during inference, which many deployment devices do not support.",
        whyThisPaper:
          "Self-attention models long-range template–search relationships in one forward pass, and a gradient-free dynamic template adds temporal information without any test-time gradients.",
      },
      {
        name: "TransTrack and TrackFormer (T053, T054)",
        limitation:
          "Both inject temporal information by propagating or updating query embeddings across frames — TransTrack matches two predicted box groups with Hungarian IoU, TrackFormer associates purely by attention — so appearance lives in the query embeddings and there is no explicit target model to refresh or reject.",
        whyThisPaper:
          "STARK keeps a single target query and updates an explicit dynamic template instead, refreshed only when a learned score passes τ, so temporal change is handled in the input rather than the query.",
      },
      {
        name: "DETR detection head",
        limitation:
          "A three-layer perceptron predicting box coordinates is 'equivalent to fitting a Dirac delta distribution' and gives lower-quality localization; jointly training localization and classification leads to sub-optimal solutions for both tasks.",
        whyThisPaper:
          "A 5-layer convolutional corner head estimates the top-left and bottom-right corners from heatmaps, and training is split into a localization stage and a score-head stage.",
      },
    ],
    researchGap:
      "Before this paper, no tracker combined a dynamically updated template — refreshed only when a learned confidence score passes a threshold — with an end-to-end transformer that predicts boxes directly, without anchors, proposals or post-processing.",
    contribution: [
      "A transformer architecture dedicated to tracking: the encoder captures global spatio-temporal dependencies over a triplet (search 320×320, initial template 128×128, dynamic template) and the decoder maps one target query to the box.",
      "Direct keypoint-style prediction: a fully-convolutional corner head estimates the two object corners end-to-end — no proposals, anchors, cosine window or bounding-box smoothing.",
      "A score head (three-layer MLP + sigmoid) with update interval Tu = 200 frames and threshold τ = 0.5 governing when the dynamic template is refreshed, adding temporal information at nearly no extra cost.",
      "Two-stage training that decouples localization (500 epochs) from classification (50 epochs), with state of the art on five short- and long-term benchmarks at real-time speed (6× faster than Siam R-CNN).",
    ],
    method: {
      pipeline: [
        "extract-triplet-features",
        "encode",
        "decode-target-query",
        "predict-corner-heatmaps",
        "score-and-update-template",
      ],
      architecture:
        "ImageNet-pretrained ResNet-50/101 with frozen BatchNorm, features pooled from the stride-16 fourth stage → the three patches are flattened, concatenated and passed through 6 encoder layers → 6 decoder layers carrying a single target query (8 heads, width 256, FFN 2048, dropout 0.1) → a corner head of 5 Conv-BN-ReLU blocks producing the heatmaps Ptl and Pbr, plus a three-layer MLP score head with 256 hidden units.",
      appearanceModel:
        "Global self-attention over the concatenated triplet: the encoder mixes template and search elements in both spatial and temporal directions, and the dynamic template carries appearance change from previous frames.",
      association:
        "None: one query yields one box, so there is no Hungarian matching, no multi-hypothesis output and no frame-to-frame association — the contrast the paper draws with DETR, TransTrack and TrackFormer.",
      trackManagement:
        "Both templates are initialized in the first frame; the dynamic template is replaced only when the update interval Tu = 200 frames has been reached and the predicted confidence is above τ = 0.5, otherwise the old template is kept.",
      optimization:
        "Two stages: (1) everything except the score head trains end-to-end on the localization losses for 500 epochs, with every search image guaranteed to contain the target; (2) all other parameters are frozen and only the score head trains with binary cross-entropy for 50 epochs on balanced samples. AdamW with weight decay 1e-4, learning rate 1e-5 for the backbone and 1e-4 for the rest, 6 × 10⁴ triplets per epoch; 128×128 templates (22× the box area) and 320×320 search regions (52× the box area) with horizontal-flip and brightness-jitter augmentation.",
      loss: "Stage 1: L = λiou Liou + λL1 L1 with λL1 = 5, λiou = 2 (Eq. 2). Stage 2: binary cross-entropy score loss (Eq. 3).",
    },
    equations: [
      {
        id: "stark-corner-expectation",
        label: "Corner heatmap expectation",
        formula:
          "(xc_tl, yc_tl) = (Σ_{y=0..H} Σ_{x=0..W} x · Ptl(x, y),  Σ_{y=0..H} Σ_{x=0..W} y · Ptl(x, y));\n(xc_br, yc_br) = (Σ Σ x · Pbr(x, y),  Σ Σ y · Pbr(x, y))",
        variables: [
          { symbol: "Ptl, Pbr", meaning: "predicted top-left and bottom-right corner heatmaps over the H×W search grid" },
          { symbol: "(xc_tl, yc_tl), (xc_br, yc_br)", meaning: "the decoded corner coordinates of the bounding box" },
          { symbol: "x, y", meaning: "grid positions of the search feature map" },
        ],
        intuition:
          "Each corner is read off as the centre of mass of its heatmap, so the box is a soft vote over all locations instead of an arg-max or a regressed offset.",
        why:
          "Corner heatmaps model the uncertainty in coordinate estimation, which the paper argues gives more accurate and robust predictions for tracking than a perceptron that fits a single point.",
        where: "Section 3.1, Eq. (1): decoding of the fully-convolutional box head.",
        paperIds: ["T056"],
      },
      {
        id: "stark-box-loss",
        label: "Localization loss",
        formula: "L = λiou Liou(bi, b̂i) + λL1 L1(bi, b̂i)",
        variables: [
          { symbol: "bi, b̂i", meaning: "predicted and ground-truth boxes" },
          { symbol: "λiou, λL1", meaning: "loss weights, set to 2 and 5 in the experiments" },
          { symbol: "Liou", meaning: "generalized IoU loss" },
          { symbol: "L1", meaning: "element-wise L1 box loss" },
        ],
        intuition: "Penalize the predicted box both by how far each coordinate is off and by how much the boxes fail to overlap.",
        why: "This is the only supervision of stage 1, so localization quality is trained independently of any classification or score signal.",
        where: "Section 3.1, Eq. (2); the full stage-1 objective.",
        params: "λL1 = 5 and λiou = 2 by default (as in DETR).",
        paperIds: ["T056"],
      },
      {
        id: "stark-score-bce",
        label: "Score-head binary cross-entropy",
        formula: "Lce = yi log(Pi) + (1 − yi) log(1 − Pi)",
        variables: [
          { symbol: "yi", meaning: "ground-truth label for whether the predicted state is reliable" },
          { symbol: "Pi", meaning: "confidence predicted by the score head" },
        ],
        intuition: "Train a yes/no judge: was the tracker on target this frame, and therefore may the template be refreshed?",
        why: "The update gate must be learned rather than hand-tuned, since a wrong update (occlusion, drift) poisons every later frame.",
        where: "Section 3.2, Eq. (3); stage-2 training with all other parameters frozen.",
        params: "Inference uses the gate at τ = 0.5 together with the interval Tu = 200 frames.",
        paperIds: ["T056"],
      },
    ],
    datasets: ["got10k", "trackingnet", "lasot", "vot", "others"],
    metrics: ["success-auc", "norm-precision", "precision", "eao", "fps"],
    baselines: ["Siam R-CNN", "PrDiMP50", "DiMP-50", "Ocean", "SiamRPN++", "SiamMask"],
    results: [
      "GOT-10k test (Table 1): AO 67.2 / 68.0 / 68.8 and SR0.5 76.1 / 77.7 / 78.1 for STARK-S50 / ST50 / ST101 — ST50 is +4.6 AO over PrDiMP50 (63.4), and ST101 surpasses Siam R-CNN (64.9) by 3.9 with the same ResNet-101 backbone.",
      "TrackingNet test (Table 2): AUC 80.3 / 81.3 / 82.0 and Pnorm 85.1 / 86.1 / 86.9 — STARK-ST101 beats Siam R-CNN (81.2) by 0.8 AUC.",
      "LaSOT: STARK-ST101 reaches success 67.1 and normalized precision 77.0 versus Siam R-CNN 64.8 / 72.2 and PrDiMP50 59.8 / 68.8; the paper reports its two spatio-temporal variants gaining 6.0% and 6.6% over PrDiMP50.",
      "VOT2020 box-only (Table 3): EAO 0.280 / 0.308 / 0.303 (accuracy 0.477 / 0.478 / 0.481, robustness 0.728 / 0.799 / 0.775) — STARK-ST50's 0.308 beats the best previous box tracker SuperDiMP (0.305); with Alpha-Refine the EAOs become 0.462 / 0.505 / 0.497, ST50+AR ahead of OceanPlus+AR (0.491).",
      "VOT2020-LT (Table 4): F-score 70.2 (ST50) and 70.1 (ST101) with precision 71.0 / 70.2 and recall 69.5 / 70.1 — best in the table, ahead of LT DSE (69.5).",
      "Efficiency (Table 5): STARK-S50 / ST50 / ST101 run at 42.2 / 41.8 / 31.7 FPS with 10.5 / 10.9 / 18.5 GFLOPs and 23.3 / 23.5 / 42.4M parameters, versus SiamRPN++ at 35.0 FPS, 48.9 GFLOPs and 54.0M parameters; ST101 runs 6× faster than Siam R-CNN (5 fps).",
    ],
    ablations: [
      "Component removal on LaSOT (Table 6, base STARK-ST50 success 66.4): removing the encoder costs 5.3 points (61.1), the decoder 1.9 (64.5), positional encoding only 0.2 (66.2); replacing the corner head with a three-layer perceptron costs 2.7 (63.7); removing the score head gives 64.5.",
      "Alternative frameworks on LaSOT (Table 7): using template images as queries 61.2, Hungarian matching with K = 10 queries 63.7 (the paper observes a 'Matthew effect' where only one or two queries learn to predict well), TrackFormer-style query update 64.8, joint localization–classification 62.5, STARK 66.4.",
      "Two-stage training matters: joint one-stage learning is 3.9 points lower, because the score head interferes with the box head and the tasks want different data (search regions that always contain the target for localization, balanced ones for classification).",
      "The score gate is what makes temporal information safe: without it the result (64.5) falls below the spatial-only STARK-S50, so the authors conclude that improper use of temporal information hurts and unreliable templates must be filtered out.",
    ],
    limitations: {
      authorStated: [
        "STARK-S does not predict a confidence score, so the authors state they do not report its result on VOT2020-LT — the spatial-only variant cannot be evaluated in the long-term protocol at all.",
        "The authors note that improper use of temporal information may hurt performance (removing the score head drops below the spatial-only variant), which is why updates are gated rather than applied every frame.",
      ],
      evident: [
        "The spatial-only STARK-S50 is not competitive with the best box-only trackers on VOT2020 (EAO 0.280 versus SuperDiMP 0.305); the gains come from the temporal template and the refinement stack.",
        "The template refresh is fixed at Tu = 200 frames, so a target whose appearance changes sooner keeps a stale dynamic template for up to ~7 seconds of video — the gate only vetoes updates, it never makes them more frequent.",
        "Corner heatmaps live on a stride-16 grid, so box precision is limited by that resolution and only refined by taking the centre of mass of a coarse heatmap.",
      ],
    },
    assumptions: [
      "The first-frame template plus at most one refreshed template every 200 frames covers realistic appearance change; IoU-based reliability labels are accurate enough to train the update gate.",
      "A single query is sufficient — the tracker never needs multiple hypotheses or frame-to-frame association because each frame is localized independently.",
    ],
    computation:
      "42.2 / 41.8 / 31.7 FPS for S50 / ST50 / ST101 with 10.5 / 10.9 / 18.5 GFLOPs and 23.3 / 23.5 / 42.4M parameters; trained on 8 × 16 GB Tesla V100 with Python 3.6 and PyTorch 1.5.1, 500 + 50 epochs at 6 × 10⁴ triplets per epoch.",
    relations: [
      {
        to: "T054",
        type: "uses-as-baseline",
        note: "Table 7 rebuilds TrackFormer's query-update idea on STARK-S50 and compares it with the dynamic-template design (64.8 vs 66.4 LaSOT success).",
      },
      {
        to: "T053",
        type: "addresses-limitation",
        note: "Contrasts TransTrack's two-decoder, frame-to-frame query propagation and Hungarian box matching with an explicit, score-gated template update.",
      },
      {
        to: "T031",
        type: "uses-as-baseline",
        note: "PrDiMP50 and DiMP-50 are the main competitors on GOT-10k, TrackingNet, LaSOT and VOT2020.",
      },
      {
        to: "T046",
        type: "uses-as-baseline",
        note: "Ocean (and OceanPlus with refinement) is compared on LaSOT and in the VOT2020 tables.",
      },
    ],
    concepts: ["attention", "siamese", "appearance-features", "track-management", "anchor-free-head"],
    impact:
      "STARK made template update a learned, gated operation instead of a heuristic — one target query, a corner head and a score gate with no matching and no post-processing — and its ablation of query-update versus template-update designs became the standard reference for how transformer trackers should carry information across frames.",
  },
  {
    id: "T057",
    arxiv: "2104.13202",
    title: "LasHeR: A Large-scale High-diversity Benchmark for RGBT Tracking",
    shortTitle: "LasHeR",
    year: 2021,
    venue: "IEEE Transactions on Image Processing",
    authors: ["Chenglong Li", "Wanlin Xue", "Yaqing Jia", "Zhichen Qu", "Bin Luo", "Jin Tang", "a.o."],
    fileName: "2104.13202v2.pdf",
    task: "benchmark",
    tags: ["benchmark", "rgbt-tracking", "multimodal", "thermal", "large-scale", "dataset"],
    difficulty: "intro",
    summary:
      "LasHeR is a large-scale benchmark for short-term RGB-thermal (RGBT) tracking: 1224 aligned, densely annotated visible/infrared video pairs totalling 734.8K frame pairs, 32 object categories and 19 challenge attributes of which 7 are new, captured across multiple platforms, seasons, weathers and lighting. The paper evaluates 12 RGBT trackers with precision, normalized-precision and success plots, releases an unaligned version for alignment-free tracking, and provides a train/test split — retraining mfDiMP on it lifts its success on the test split from 0.364 to 0.467, past the best non-retrained tracker.",
    problem:
      "Existing real RGBT datasets are small: GTOT has 50 sequences, RGBT210 and RGBT234 share an inclusive relationship so no more than 234 independent sequences exist between them, none offers a training split, and synthetic thermal data generated from RGB (pix2pix) 'have a great gap with real ones'. Deep RGBT trackers therefore had nowhere to learn from at scale, and evaluations could not measure whether a tracker generalizes across categories and real-world challenges.",
    background: ["benchmark-design", "multimodal", "appearance-features", "success-plot", "sot"],
    previousWork: [
      {
        name: "GTOT, RGBT210, RGBT234, VOT-RGBTIR2019",
        limitation:
          "50 to 234 sequences with 9–22 classes and no training dataset — limited size, diversity and bias; the three most-used sets overlap so the total pool of independent sequences does not exceed 234.",
        whyThisPaper:
          "LasHeR supplies 1224 pairs, 32 classes, 19 attributes, multi-platform imaging and a training subset, so deep trackers can be trained and tested at scale.",
      },
      {
        name: "Synthetic RGBT data (pix2pix-generated thermal images)",
        limitation:
          "Thermal frames are produced from visible images by a model trained on 8,335 videos, so the paper argues it is unclear whether the generated modality carries the complementary information that motivates RGBT tracking in the first place.",
        whyThisPaper:
          "LasHeR is captured with real thermal cameras on multiple platforms, and its retraining experiments show real data beats synthetic data (MANet retrained on LasHeR: RGBT234 precision 0.810 vs 0.785 with synthetic data).",
      },
      {
        name: "RGB trackers trained on large RGB datasets (TransT, DiMP-50)",
        limitation:
          "They ignore the thermal modality entirely, yet still outperform most RGBT trackers on LasHeR precisely because they were trained on large-scale data — exposing that RGBT methods are data-starved rather than architecturally superior.",
        whyThisPaper:
          "The benchmark quantifies both effects at once: it re-trains RGBT trackers on real large-scale data (mfDiMP 0.599 / 0.467 PR / SR on the test split) and beats those RGB trackers, isolating training data as the missing ingredient.",
      },
    ],
    researchGap:
      "Before this paper, RGBT tracking had no large-scale, high-diversity benchmark with real thermal data, challenge attributes and a training split — deep trackers had no real data to learn from and no protocol to measure them fairly.",
    contribution: [
      "LasHeR: 1224 RGBT video pairs, 734.8K frame pairs, average 600 and maximum 12862 frames per sequence, 32 categories and 19 attributes, captured on multiple platforms across seasons, weathers, day and night; every frame pair is spatially aligned and manually annotated with a bounding box.",
      "Seven challenge attributes new to RGBT tracking — hyaline occlusion (HO), high illumination (HI), abrupt illumination variation (AIV), similar appearance (SA), aspect ratio change (ARC), out-of-view (OV) and frame lost (FL) — labelled at sequence level for challenge-based evaluation.",
      "An unaligned release whose ground-truth boxes are generated automatically from the aligned annotations via SIFT homography, opening the alignment-free RGBT tracking task.",
      "Two evaluation protocols: all 1224 pairs as one test set, and a split by target-class distribution for training deep trackers, plus a retraining study (MANet, mfDiMP) demonstrating the value of large-scale real training data.",
    ],
    method: {
      pipeline: ["capture", "align", "annotate", "split", "evaluate"],
      architecture:
        "Not a tracker but a corpus and protocol: multi-platform capture (a turnable platform modelled on RGBT234 using a DLS-H37DM-A thermal imager and a SONY EXView HAD CCD camera, plus other platforms), local homography alignment around the target that keeps the thermal image fixed and warps the visible one, sequence-level challenge labelling with the ViTBAT annotation tool, and two evaluation protocols over precision / normalized-precision / success plots.",
      detectionDependency:
        "Evaluation runs the 12 released RGBT trackers as-is (no modification) under protocol 1, and re-trains MANet and mfDiMP on the training subset under protocol 2.",
      optimization:
        "Retraining follows each tracker's own recipe (mfDiMP uses offline training inside the DiMP framework, MANet uses MDNet-style adapters); alignment uses SIFT feature matching and RANSAC-free homography estimation on annotated matching points.",
    },
    equations: [
      {
        id: "lasher-pr",
        label: "Precision rate",
        formula: "PR = fraction of frames whose distance between predicted position and ground truth is ≤ 20 pixels",
        variables: [
          { symbol: "PR", meaning: "precision rate, reported as the representative precision score" },
          { symbol: "20 pixels", meaning: "the distance threshold set by the paper for the representative score" },
          { symbol: "predicted position / ground truth", meaning: "tracker output centre versus annotated box centre in each frame" },
        ],
        intuition: "Count how often the tracker's box lands close enough to the annotated box in raw pixels.",
        why: "It is the simplest location-accuracy measure and is comparable across trackers, though it depends on image resolution — hence the normalized variant.",
        where: "Section IV.G (evaluation metrics), stated in prose; no numbered equation in the paper.",
        paperIds: ["T057"],
      },
      {
        id: "lasher-sr",
        label: "Success rate",
        formula: "SR = area under the curve of the ratio of frames where the overlap of predicted and ground-truth boxes exceeds a threshold, swept over thresholds",
        variables: [
          { symbol: "SR", meaning: "success rate, reported as the area-under-curve score" },
          { symbol: "overlap", meaning: "IoU-style overlap between the predicted and ground-truth bounding boxes" },
          { symbol: "threshold", meaning: "the overlap threshold swept to produce the curve" },
        ],
        intuition: "Sweep how strict 'enough overlap' is, record the pass rate at each strictness, and report the average — one number for the whole curve.",
        why: "It scores localization quality continuously instead of at a single hand-picked overlap, and is the metric on which LasHeR separates the hardest trackers.",
        where: "Section IV.G (evaluation metrics), stated in prose; no numbered equation in the paper.",
        paperIds: ["T057"],
      },
    ],
    datasets: ["lasheR", "lasot", "trackingnet", "others"],
    metrics: ["precision", "norm-precision", "success-auc", "fps"],
    baselines: ["DMCNet", "mfDiMP", "MANet", "DAFNet", "FANet", "TransT"],
    results: [
      "Entire LasHeR (1224 pairs, Table IV): DMCNet has the best precision 0.557 and normalized precision 0.494, mfDiMP the best success 0.399; all 12 trackers score far lower than on smaller RGBT sets — DMCNet holds only 0.839 / 0.593 PR / SR on RGBT234 and mfDiMP reaches 0.785 / 0.559 on RGBT210.",
      "Speed: mfDiMP 38.242 FPS, DAFNet 22.208, FANet 17.476, MANet++ 16.153, CAT 8.938, DMCNet only 2.244 (online training) and MANet 0.904.",
      "Testing-subset protocol: every tracker drops significantly; DMCNet still leads with precision 0.490, normalized precision 0.431 and success 0.355.",
      "Retraining (Table V): mfDiMP trained on the LasHeR training subset reaches precision 0.599 and success 0.467 on the LasHeR test split versus 0.474 / 0.364 trained on GTOT — both scores exceed the best non-retrained tracker (DMCNet 0.490 / 0.355).",
      "MANet retraining: RGBT234 precision 0.777 → 0.810 and success 0.539 → 0.569, LasHeR test split 0.455 → 0.508 and 0.326 → 0.369; the synthetic-data-trained MANet is worse than the LasHeR-trained one on both sets (0.785 / 0.559 and 0.447 / 0.343).",
      "RGB trackers on LasHeR (Table VI/VII): TransT is the best RGB tracker at 0.592 / 0.548 / 0.442 (PR / NPR / SR), ahead of DiMP-50 0.521 / 0.467 / 0.395, but all of them fall far below their own LaSOT and TrackingNet scores (TransT 0.690 / 0.738 / 0.649 and 0.803 / 0.867 / 0.814), and retrained mfDiMP (0.599 / 0.467) beats them.",
    ],
    ablations: [
      "Training data (Table V): for both MANet and mfDiMP, training on the real LasHeR subset beats training on GTOT and on the large synthetic RGBT set — mfDiMP success 0.364 → 0.467 on the test split, MANet precision on RGBT234 0.777 → 0.810 versus 0.785 with synthetic data, so scale and realism of real thermal data both matter.",
      "Protocol effect: the same 12 trackers scored on the whole set versus the class-split test subset all drop substantially (e.g. DMCNet precision 0.557 → 0.490), showing how much single-protocol numbers depend on which sequences are held out.",
      "Challenge analysis (Table IV): the seven new attributes are where trackers fail — performance is almost worst on hyaline occlusion, frame lost and aspect ratio change (e.g. mfDiMP success 0.247 on HO and 0.226 on FL against 0.399 overall), while no-occlusion is easiest (mfDiMP 0.738 / 0.572).",
      "Modality test: RGB trackers retrained on LasHeR (TransT 0.529 / 0.395, DiMP-50 0.442 / 0.336 PR / SR on the test split) still lose to retrained RGBT mfDiMP (0.599 / 0.467), which the paper reads as evidence for the complementary value of thermal data.",
    ],
    limitations: {
      authorStated: [
        "The authors plan to 'expand the data of categories to further enhance the diversity of the data set'.",
        "They state they 'will complete the annotations with other information including target masks and semantic descriptions', and conclude that there is still a lot of room for research and development of RGBT tracking.",
      ],
      evident: [
        "Even the best success rate is only 0.399 on the full set, so the benchmark documents an unsolved problem rather than a saturated one.",
        "The retraining study covers just two architectures (MANet and mfDiMP), so the value of the training split is demonstrated on a narrow sample of the tracker families it targets.",
        "Unaligned ground truth is derived from the aligned annotations through SIFT homography, so its quality is bounded by feature matching — it inherits the aligned set's boxes rather than being independently annotated.",
      ],
    },
    assumptions: [
      "Visible and thermal frames are synchronized in time, and a homography computed locally around the target is enough to align the two modalities (radial distortion is explicitly not corrected).",
      "Sequence-level challenge labels are an adequate proxy for per-frame difficulty in the attribute-based evaluation.",
    ],
    computation:
      "Annotation with the ViTBAT video tracking and behaviour annotation tool; alignment by SIFT feature matching and homography estimation; tracker speeds measured by the authors range from 0.891 (MaCNet) to 38.242 FPS (mfDiMP); datasets and evaluation code released at github.com/mmic-lcl/Datasets-and-benchmark-code.",
    relations: [
      {
        to: "T055",
        type: "uses-as-baseline",
        note: "TransT is the best-performing RGB tracker in Tables VI and VII, evaluated as-is and retrained on the LasHeR training split.",
      },
      {
        to: "T031",
        type: "uses-as-baseline",
        note: "DiMP-50 and the RGBT-adapted mfDiMP (a DiMP variant) are the main baselines; mfDiMP is one of the two retrained trackers.",
      },
      {
        to: "T026",
        type: "uses-as-baseline",
        note: "ATOM appears in the RGB-tracker comparison table, and mfDiMP borrows its IoU-Net architecture.",
      },
    ],
    concepts: ["benchmark-design", "multimodal", "appearance-features", "success-plot"],
    impact:
      "LasHeR became the standard large-scale training and evaluation set for RGBT tracking, and its retraining results made the field's central methodological point: real, diverse thermal training data — not synthetic pix2pix thermal or bigger architectures — is what deep RGBT trackers were missing.",
  },
  {
    id: "T058",
    arxiv: "2104.14545",
    title:
      "LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search",
    shortTitle: "LightTrack",
    year: 2021,
    authors: ["Bin Yan", "Houwen Peng", "Kan Wu", "Dong Wang", "Jianlong Fu", "Huchuan Lu"],
    fileName: "2104.14545v1.pdf",
    task: "single-object",
    tags: ["neural-architecture-search", "siamese", "efficiency", "lightweight", "mobile", "one-shot"],
    difficulty: "advanced",
    summary:
      "LightTrack reformulates tracker design as a one-shot neural architecture search: a backbone supernet (pre-trained on ImageNet) and a tracking-head supernet are trained once, then an evolutionary search picks the best architecture under FLOPs and parameter budgets. The result is a family of trackers — Mobile (≤600M FLOPs, ≤2M params), LargeA and LargeB — that reaches VOT2019 EAO 0.333 / 0.340 / 0.357 and LaSOT success 0.555, while running 12× faster than Ocean on a Snapdragon 845 Adreno GPU (38.4 vs 3.2 FPS).",
    problem:
      "State-of-the-art trackers are too heavy to deploy: SiamRPN++ needs 48.9 GFLOPs and 54M parameters and Ocean 20.3 GFLOPs and 25.9M, roughly 80× the 600M-FLOP mobile budget that deep learning on phones normally requires. Compression (quantization) degrades accuracy and manual compact design depends on human expertise and experiments, so nobody had found the accuracy-versus-complexity trade-off for tracking automatically.",
    background: ["siamese", "correlation-filter", "deep-detection", "appearance-features", "success-plot"],
    previousWork: [
      {
        name: "Hand-designed Siamese trackers (SiamRPN++, Ocean, SiamFC)",
        limitation:
          "SiamRPN++ with ResNet-50 has 48.9 GFLOPs and 54M parameters and Ocean 20.3 GFLOPs — orders of magnitude above the 600M mobile setting, blocking deployment on drones, robotics and driving assistance; even SiamFC's AlexNet model exceeds embedded budgets.",
        whyThisPaper:
          "The search space contains only architectures of 208M–1.4G FLOPs and 0.2M–5.4M parameters, so any tracker the search returns is deployable by construction while still matching or beating these baselines.",
      },
      {
        name: "Model compression and manual compact design",
        limitation:
          "Quantization and pruning bring non-negligible performance degradation, while designing compact models 'heavily relies on human expertise and experiment' — neither searches the trade-off.",
        whyThisPaper:
          "One-shot NAS automates the design, using tracking accuracy and model complexity together as the supervision for the search.",
      },
      {
        name: "DetNAS and one-shot NAS for detection/backbones",
        limitation:
          "DetNAS searches only a backbone, biased towards classification and detection objectives, while tracking needs the correlation head and the backbone designed jointly under tracking supervision.",
        whyThisPaper:
          "LightTrack formulates one-shot NAS over both a backbone supernet and a head supernet (Eq. 3), ranked on tracking validation, with evolutionary search under explicit FLOPs/parameter constraints (Eq. 4).",
      },
    ],
    researchGap:
      "Before this paper, no method had automatically designed a tracking network — trackers were hand-built or compressed, and no study had found the accuracy/complexity trade-off for object tracking.",
    contribution: [
      "The first effort at automating the design of neural architectures for object tracking: a new formulation of one-shot NAS specialized for tracking, in which supernets are trained once and candidate architectures inherit their weights.",
      "A lightweight search space (depthwise-separable and inverted-residual blocks, 14 searchable backbone layers ≈ 6¹⁴ ≈ 7.8×10¹⁰ backbones, ≈ 3.9×10⁸ heads, all within 208M–1.4G FLOPs) plus a dedicated three-phase search pipeline with evolutionary search, architecture constraints and BatchNorm recalibration.",
      "Three budgeted trackers — Mobile (≤600M FLOPs, ≤2M params), LargeA (≤800M, ≤3M) and LargeB (≤800M, ≤4M) — with state-of-the-art results on VOT2019, GOT-10k, TrackingNet and LaSOT that run on resource-limited phones.",
    ],
    method: {
      pipeline: [
        "pretrain-backbone-supernet",
        "train-tracking-supernet",
        "evolutionary-search",
        "recalibrate-bn",
        "retrain-backbone",
        "fine-tune-tracker",
      ],
      architecture:
        "One-shot NAS over two supernets N = {Nb, Nh}. Nb stacks depthwise-separable convolutions (DSConv) and mobile inverted bottlenecks with squeeze-excitation (MBConv, kernels {3, 5, 7}, expansion {4, 6}) in four stride-16 stages with 14 searchable layers; Nh encodes all correlation-based heads — a classification branch and a regression branch with up to 8 searchable layers each. A candidate tracker is a path through both supernets, inheriting weights without extra training.",
      appearanceModel:
        "Template and search features are cross-correlated to produce correlation volumes, which the searched head classifies and regresses from — a Siamese correlation tracker whose backbone and head shapes are both discovered.",
      detectionDependency: "None: single object given by the first-frame box; inference is one forward pass per frame.",
      optimization:
        "Phase 1: backbone supernet 120 epochs on ImageNet, SGD momentum 0.9, weight decay 4e-5, learning rate 0.5 with linear annealing. Phase 2: joint supernet training on Youtube-BB, ImageNet VID/DET, COCO and GOT-10k train (30 epochs, 6×10⁵ pairs each), SGD momentum 0.9, weight decay 1e-4, batch 256, learning rate 1e-2 → 3e-2 over 5 epochs then log-decayed to 1e-4, backbone frozen for the first 10 epochs and at 10× lower rate afterwards, evaluated on the GOT-10k validation split. After search: backbone retrained 500 epochs on ImageNet (MSProp, learning rate 0.064 with 3-epoch warmup and cosine annealing, dropout 0.2, AutoAugment, exponential moving average) and the tracker fine-tuned 50 epochs (learning rate 2e-2 → 1e-1 → 2e-4).",
      loss: "Binary cross-entropy for foreground/background classification plus IoU loss for box regression (Eq. 3's Ltrk train).",
    },
    equations: [
      {
        id: "lighttrack-nas-objective",
        label: "One-shot NAS objective",
        formula: "α* = arg max Accval(N(α, W*(α)))",
        variables: [
          { symbol: "α", meaning: "an architecture (choice of layers/operators) in the supernet" },
          { symbol: "W*(α)", meaning: "weights inherited by that architecture from the trained supernet" },
          { symbol: "Accval", meaning: "validation accuracy used to rank candidates (tracking performance)" },
        ],
        intuition: "Rank every candidate architecture by how well it already performs inside the trained supernet — no candidate ever needs its own training run.",
        why: "It is what makes the search affordable: instead of training thousands of trackers, evaluation is pure inference.",
        where: "Section 3, Eq. (1): the basic one-shot formulation LightTrack builds on.",
        paperIds: ["T058"],
      },
      {
        id: "lighttrack-joint-search",
        label: "Joint backbone-and-head search for tracking",
        formula:
          "αb*, αh* = arg max Acctrk_val(N(αb, Wb*(αb); αh, Wh*(αh)))\ns.t. Wb*, Wh* = arg min Ltrk_train(N(Ab, Wb; Ah, Wh)),  Wb ← Wbp",
        variables: [
          { symbol: "αb, αh", meaning: "backbone and head architectures being searched jointly" },
          { symbol: "Wbp", meaning: "ImageNet-pretrained backbone weights used to initialize the supernet" },
          { symbol: "Acctrk_val, Ltrk_train", meaning: "tracking validation accuracy and tracking training loss" },
        ],
        intuition: "Pre-train the backbone supernet on ImageNet, fine-tune it on tracking data, then pick the backbone–head pair that scores best on held-out tracking data using only inference.",
        why: "Tracking needs both parts designed together — a backbone good for classification alone gives a sub-optimal tracker, which is the paper's critique of DetNAS-style backbone-only search.",
        where: "Section 4.1, Eq. (3): the reformulated one-shot search problem.",
        paperIds: ["T058"],
      },
      {
        id: "lighttrack-budget",
        label: "Deployment constraints",
        formula: "FLOps(αb*) + FLOps(αh*) ≤ FLOpsmax;  Params(αb*) + Params(αh*) ≤ Paramsmax",
        variables: [
          { symbol: "FLOpsmax, Paramsmax", meaning: "the deployment budgets, e.g. 600M FLOPs and 2M parameters for the Mobile variant" },
          { symbol: "αb*, αh*", meaning: "the selected backbone and head architectures" },
        ],
        intuition: "Only count a candidate if the whole tracker fits inside the memory and compute budget of the target device.",
        why: "The constraints steer the evolutionary search: mutation and crossover generate candidates until they satisfy the budget, which is how the three variants are produced.",
        where: "Section 4.1, Eq. (4): architecture constraints on the search.",
        params: "FLOpsmax = 600M / 800M and Paramsmax = 2M / 3M / 4M for Mobile / LargeA / LargeB.",
        paperIds: ["T058"],
      },
    ],
    datasets: ["got10k", "trackingnet", "lasot", "others"],
    metrics: ["eao", "success-auc", "precision", "norm-precision", "fps"],
    baselines: ["SiamRPN++", "Ocean", "ATOM", "DiMP-50", "SiamFC++", "SiamMask"],
    results: [
      "VOT2019 (Table 2): EAO 0.333 / 0.340 / 0.357 for Mobile / LargeA / LargeB (accuracy 0.536 / 0.540 / 0.552, robustness 0.321 / 0.315 / 0.310) — ahead of Ocean(off) 0.327, DiMPr 0.321, TKU 0.314, ATOM 0.301, SiamRPN++(M) 0.292, SiamFC++(G) 0.288 and SiamMask 0.287, with Mobile using 0.53 GFLOPs and 1.97M parameters against Ocean's 20.3 GFLOPs and 25.9M.",
      "GOT-10k (Table 3): AO 0.611 / 0.615 / 0.623 and SR0.5 0.710 / 0.723 / 0.726 — LightTrack-Mobile beats SiamFC++(G) (0.595) by 1.6 and Ocean-offline (0.592) by 1.9 AO, and LargeB beats DiMP-50 (0.611) by 1.2 with 8× fewer parameters (3.1M vs 26.1M).",
      "TrackingNet (Table 4): precision 69.5 / 70.0 / 70.8 (Mobile is +0.8 over DiMP-50's 68.7), Pnorm 77.9 / 78.8 / 78.9, AUC 72.5 / 73.6 / 73.3 — Mobile matches SiamRPN++ and DiMP-50 on Pnorm/AUC while using 96% and 92% fewer parameters.",
      "LaSOT (Fig. 3): success 0.538 / 0.550 / 0.555 and precision 0.537 / 0.552 / 0.561 — LargeB surpasses SiamFC++(G) (0.543) by 1.2 and Ocean-offline (0.526) by 2.9 in success, and beats DiMP-18 (0.534) by 2.1 while using 12× fewer parameters.",
      "Mobile deployment: on Snapdragon 845 Adreno 630 the discovered tracker runs 12× faster than Ocean (38.4 vs 3.2 FPS) with 13× fewer parameters (1.97M vs 25.9M) and 38× fewer FLOPs (530M vs 20,300M); tests cover Apple iPhone 7 PLUS, Huawei Nova 7 5G and Xiaomi Mi 8.",
      "Abstract headline: the 530M-FLOP tracker reaches EAO 0.33 on VOT2019, surpassing SiamRPN++ by 4.6 while cutting its 48.9G FLOPs by 98.9%.",
    ],
    ablations: [
      "Component search (Table 5, VOT2019 EAO): handcrafted baseline 0.268 → searched backbone 0.292 → adding a searched output layer 0.307 → searched head alone 0.297 → fully searched 0.333, so backbone and output-layer search contribute most and head search adds the rest.",
      "ImageNet pre-training of the backbone supernet (Table 6): evaluating at epoch 0 / 200 / 500 gives top-1 accuracy – / 72.4 / 77.6 and VOT2019 EAO 21.3 / 31.2 / 33.3 — without pre-training the searched tracker loses more than 12 EAO points.",
      "What the search learned: about half of the backbone blocks are 7×7 MBConv (large receptive fields help localization), the second-last block is chosen as the feature output (tracking prefers mid-level features), and the classification branch ends up shallower than the regression branch, which the authors attribute to coarse localization being easier than precise box regression.",
    ],
    limitations: {
      authorStated: [
        "The authors state that a search space 'defines which neural architectures a NAS approach might discover in principle', so every tracker reported is confined to their lightweight DSConv/MBConv space of 208M–1.4G FLOPs and 0.2M–5.4M parameters.",
        "They describe applying NAS to tracking as non-trivial because trackers typically need ImageNet pre-training while NAS algorithms require performance feedback from the target task as supervision signals.",
      ],
      evident: [
        "The FLOPs budget costs raw accuracy: LightTrack-Mobile's VOT2019 accuracy (0.536) is well below Ocean's (0.590) and its TrackingNet AUC (72.5) trails DiMP-50 (74.0), so the trade-off is real even when EAO wins.",
        "Supernet ranking is only a proxy — candidates are ranked by inherited weights on the GOT-10k validation split and then fully retrained (500 + 50 epochs) to realize the reported numbers, so search results do not transfer without an expensive training run.",
        "Test-time hyper-parameters are chosen by an automated parameter-tuning toolkit per benchmark, so the reported scores include per-benchmark tuning rather than one fixed configuration.",
      ],
    },
    assumptions: [
      "Weight inheritance inside a trained supernet ranks architectures reliably without training each candidate, and accuracy on the GOT-10k validation split predicts performance on VOT, TrackingNet and LaSOT.",
      "The tracking supervision signal (BCE classification plus IoU regression on image pairs) is enough to guide the search, since no temporal or online-update module is part of the search space.",
    ],
    computation:
      "Search and training on 8 × Tesla V100 with a Xeon E5-2690 2.60 GHz CPU, Python 3.7 and PyTorch 1.1.0; Mobile uses 0.53 GFLOPs / 1.97M parameters, LargeB 0.79 GFLOPs / 3.13M; 38.4 FPS on Snapdragon 845 Adreno 630 versus Ocean's 3.2 FPS.",
    relations: [
      {
        to: "T046",
        type: "uses-as-baseline",
        note: "Ocean (offline and online) is the main baseline on VOT2019, GOT-10k and LaSOT, and the 12× mobile speed comparison is against it.",
      },
      {
        to: "T028",
        type: "uses-as-baseline",
        note: "SiamRPN++ anchors the complexity argument (48.9 GFLOPs) and is compared on VOT2019 and TrackingNet.",
      },
      {
        to: "T026",
        type: "uses-as-baseline",
        note: "ATOM appears in the VOT2019 and GOT-10k comparison tables.",
      },
      {
        to: "T031",
        type: "uses-as-baseline",
        note: "DiMP-50 / DiMP-18 are the accuracy references on GOT-10k, TrackingNet and LaSOT that LightTrack matches with far fewer parameters.",
      },
    ],
    concepts: ["siamese", "correlation-filter", "appearance-features", "success-plot"],
    impact:
      "LightTrack was the first to treat tracker design itself as a search problem, proving that supernet-ranked architectures can beat hand-tuned state of the art below 1 GFLOP — establishing the efficiency-for-tracking thread (budgeted NAS for mobile deployment) alongside the accuracy race.",
  },
];
