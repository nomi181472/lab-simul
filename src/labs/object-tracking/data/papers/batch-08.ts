import type { PaperRecord } from "../types";

/* batch-08 — T064..T068 */

export const BATCH_08: PaperRecord[] = [
  {
    id: "T064",
    arxiv: "2112.00995",
    title: "SwinTrack: A Simple and Strong Baseline for Transformer Tracking",
    shortTitle: "SwinTrack",
    year: 2021,
    authors: ["Liting Lin", "Heng Fan", "Zhipeng Zhang", "Yong Xu", "Haibin Ling"],
    venue: "NeurIPS 2022",
    fileName: "2112.00995v3.pdf",
    task: "single-object",
    tags: ["transformer", "swin-transformer", "siamese", "fully-attentional", "motion-token", "baseline"],
    difficulty: "advanced",
    summary:
      "SwinTrack is a fully attentional Siamese tracker: both feature representation (Swin Transformer backbone) and template–search fusion (concatenation-based Transformer encoder plus a single decoder layer) use attention, with no CNN backbone. A lightweight motion token embeds the recent target trajectory and is fused in the decoder to add temporal context. It sets a new LaSOT record of 0.713 success at ~45 FPS (Base-384) and offers a 96 FPS tiny variant at 0.672.",
    problem:
      "Transformer trackers used attention only to fuse or enhance CNN (ResNet) features, leaving representation learning itself convolutional; the power of Transformer backbones such as ViT/Swin was unexplored in tracking. Tracking is also temporal, yet Siamese pipelines ignored historical trajectory, hurting robustness to distractors and fast motion.",
    background: ["sot", "bounding-box", "siamese", "attention", "appearance-features", "success-plot"],
    previousWork: [
      {
        name: "Hybrid CNN-Transformer trackers (TransT, STARK, TrDiMP)",
        limitation:
          "They apply Transformer only on top of ResNet features for fusion/enhancement, so representation learning never benefits from attention and stays limited by the CNN backbone.",
        whyThisPaper:
          "SwinTrack replaces the backbone itself with Swin Transformer, gaining +2.5% LaSOT and +5.1% LaSOText success over a ResNet-50 backbone in the same pipeline (Table 3).",
      },
      {
        name: "Pure CNN Siamese trackers (SiamFC, SiamRPN++)",
        limitation:
          "Matching relies on convolutional correlation with fixed local receptive fields and no temporal context, capping robustness on long challenging sequences such as LaSOT.",
        whyThisPaper:
          "Fully attentional representation plus concatenation fusion and a motion token push LaSOT past the 0.70 barrier for the first time (0.713).",
      },
      {
        name: "Query-based Transformer decoders (DETR-style object/target queries)",
        limitation:
          "A learnable general target query converges slowly and performs worse, since the generative decoder formulation suits detection but not foreground–background classification with background context.",
        whyThisPaper:
          "SwinTrack drops the query decoder for an encoder-only plus single cross-attention decoder design, which ablates better on all four benchmarks (Table 3).",
      },
    ],
    researchGap:
      "Before this paper, no Siamese tracker performed both representation learning and feature fusion with pure Transformer attention, nor injected trajectory history as a token.",
    contribution: [
      "Fully attentional Siamese framework: Swin Transformer backbone (stage-3 output, stride 16) plus a concatenation-based Transformer encoder that jointly self-/cross-attends template and search tokens with weight sharing.",
      "Single-layer cross-attention decoder fusing encoder outputs with a novel motion token: the recent trajectory (n=16 boxes, interval 15, frame-rate adjusted) is coordinate-quantized, embedded via four embedding matrices, and concatenated into one token at negligible FLOPs.",
      "IoU-aware classification (IACS) with varifocal loss plus GIoU regression loss, three-layer MLP heads, Hanning-window position prior at inference, and a generalized multi-dimensional untied positional encoding.",
      "Two variants (T-224 at ~98 FPS, B-384 at ~45 FPS) reaching state of the art on LaSOT (0.713), LaSOText (0.491), TrackingNet (0.840), GOT-10k (0.724 AO) and TNL2k (0.559).",
    ],
    method: {
      pipeline: ["extract", "concatenate", "encode", "decode-with-motion", "classify", "regress", "smooth"],
      architecture:
        "Siamese Swin Transformer (Tiny for T-224 with C=384/N=4; Base for B-384 with C=512/N=8, ImageNet-1k/22k pretrained), stage-3 tokens concatenated and passed through N encoder blocks (MSA+FFN, pre-LN, residual), then one decoder cross-attention layer over Concat(motion token, template, search) tokens; two 3-layer MLP heads emit a classification map and a box regression map.",
      motionModel:
        "No explicit filter: temporal context enters as an embedded trajectory token (sampled every ∆=15 frames, n=16, granularity g matched to search feature size) fused by decoder cross-attention; lost frames (score below θconf 0.4 LaSOT / 0.3 elsewhere) are recorded as invalid (−∞).",
      appearanceModel:
        "First-frame template tokens fused with search tokens through shared encoder attention; no online template update — robustness comes from attention interactions, motion token, and IoU-aware scoring.",
      detectionDependency: "None — class-agnostic single-object tracker with no external detector.",
      trackManagement:
        "Template cropped once from frame one (background factor 2); search region cropped around the previous prediction (factor 4); Hanning penalty r=(1−γ)r+γh with argmax localization.",
      loss:
        "Lcls = Varifocal(p, IoU(b,b̂)) on the IoU-aware classification score; Lreg = GIoU loss over positive boxes weighted by classification score p, negatives ignored.",
      optimization:
        "AdamW, lr 5e-4 (backbone 5e-5), weight decay 1e-4, 300 epochs × 131,072 pairs on 8 V100s, ×0.1 decay after 210 epochs, 3-epoch warmup, DropPath 0.1; GOT-10k-only protocol trains 150 epochs with decay at 120.",
    },
    equations: [
      {
        id: "swintrack-encoder",
        label: "Concatenation-based encoder fusion",
        formula: "fm = Concat(φ(z), φ(x)); fˡ⁺¹m = fˡm + FFN(LN(fˡm + MSA(LN(fˡm)))); fzᴸ, fxᴸ = DeConcat(fᴸm)",
        variables: [
          { symbol: "φ(z), φ(x)", meaning: "Swin backbone tokens of template z and search region x" },
          { symbol: "MSA / FFN", meaning: "multi-head self-attention and two-layer MLP with GELU" },
          { symbol: "LN", meaning: "layer normalization applied before each module" },
          { symbol: "L", meaning: "number of encoder blocks (4 for T-224, 8 for B-384)" },
        ],
        intuition:
          "Glue template and search tokens into one sequence so one self-attention simultaneously does template self-attention, search self-attention, and cross-attention between them.",
        why:
          "Concatenation shares weights (symmetric metric), saves compute versus separate self+cross attention, and lets representation and fusion both benefit from attention.",
        where: "Section 3.2, Eq. 1; runs every frame on the fused token sequence.",
        params: "More blocks (N=8 vs 4) raise accuracy at higher MACs; stride is fixed at 16 from Swin stage 3.",
        paperIds: ["T064"],
      },
      {
        id: "swintrack-trajectory",
        label: "Motion trajectory sampling and quantization",
        formula: "T = {o_s(1)..o_s(n)}, s(i) = max(t − i·∆, 1); ô = quantize(n(o,l)), n(o,l) = round(g·o/l) if valid else pad",
        variables: [
          { symbol: "o_t", meaning: "target box (x1,y1,x2,y2) at frame t, cropped-transform invariant" },
          { symbol: "∆", meaning: "sampling interval (15, frame-rate adjusted; 8 for GOT-10k)" },
          { symbol: "n / g", meaning: "number of sampled boxes (16, 8 for GOT-10k) and embedding granularity (14/24)" },
          { symbol: "pad", meaning: "last embedding-matrix entry marking absent/out-of-region boxes" },
        ],
        intuition:
          "Summarize where the target has recently been as a short list of boxes, snapped to a coarse grid so the network sees motion, not noise.",
        why:
          "Fixed-length, recency-focused, redundancy-reduced history gives the decoder temporal context for distractors and short-term stability at embedding-lookup cost.",
        where: "Section 3.2, Eqs. 2–3; motion token built per frame during training and inference.",
        params: "Larger ∆/n covers longer history; coarser g acts as trajectory augmentation.",
        paperIds: ["T064"],
      },
      {
        id: "swintrack-decoder",
        label: "Vision-motion decoder",
        formula: "fvm = fxᴸ + MCA(LN(fxᴸ), LN(Concat(Emotion, fzᴸ, fxᴸ))); fvm += FFN(LN(fvm))",
        variables: [
          { symbol: "Emotion", meaning: "single motion token from concatenated box-coordinate embeddings" },
          { symbol: "MCA", meaning: "multi-head cross-attention querying search tokens against motion+template+search memory" },
          { symbol: "fzᴸ, fxᴸ", meaning: "encoder outputs for template and search tokens" },
        ],
        intuition:
          "Let each search location ask the target's appearance and its recent path whether it is the right continuation, then sharpen the features.",
        why:
          "Template–search correlation is dropped for the template side (no need to update it), keeping one cheap layer that measurably helps on LaSOText and GOT-10k.",
        where: "Section 3.2, Eq. 4; feeds the classification/regression heads.",
        params: "A learnable embedding in place of the trajectory token hurts (Table 4), confirming the gain is motion content, not capacity.",
        paperIds: ["T064"],
      },
      {
        id: "swintrack-clsloss",
        label: "IoU-aware classification loss",
        formula: "Lcls = VFL(p, IoU(b, b̂))",
        variables: [
          { symbol: "p", meaning: "predicted IoU-aware classification score (IACS)" },
          { symbol: "b / b̂", meaning: "predicted and ground-truth boxes" },
          { symbol: "VFL", meaning: "varifocal loss weighting positives by IoU quality" },
        ],
        intuition:
          "Teach the score map to predict box quality, not just foregroundness, so the argmax picks the best-localized candidate.",
        why:
          "Removes the classification–regression gap and beats plain binary cross-entropy on every benchmark in Table 3.",
        where: "Section 3.4, Eq. 5; classification head supervision.",
        paperIds: ["T064"],
      },
      {
        id: "swintrack-regloss",
        label: "Score-weighted GIoU regression loss",
        formula: "Lreg = Σ_j 1{IoU(bj,b̂)>0} · p · GIoU(bj, b̂)",
        variables: [
          { symbol: "p", meaning: "classification score weighting high-confidence samples" },
          { symbol: "GIoU", meaning: "generalized IoU box loss" },
          { symbol: "1{·}", meaning: "indicator ignoring negative samples" },
        ],
        intuition:
          "Spend regression effort where the tracker is confident, and ignore background positions entirely.",
        why:
          "Couples the two heads so confident predictions are also geometrically accurate.",
        where: "Section 3.4, Eq. 6; regression head supervision.",
        paperIds: ["T064"],
      },
    ],
    datasets: ["lasot", "trackingnet", "got10k", "others"],
    metrics: ["success-auc", "precision", "fps"],
    baselines: ["SiamRPN++", "Ocean", "DiMP", "TrDiMP", "TransT", "STARK", "KeepTrack", "SiamR-CNN", "AutoMatch"],
    results: [
      "LaSOT: SwinTrack-B-384 success 71.3 / precision 76.5 — first tracker over 0.70, +4.2 points over STARK/KeepTrack (67.1); T-224 reaches 67.2 / 70.8 at ~98 FPS.",
      "LaSOText: B-384 49.1 / 55.6, T-224 47.6 / 53.9 vs KeepTrack 48.2 at under 20 FPS.",
      "TrackingNet: B-384 84.0 / 82.8 (best), T-224 81.1 / 78.4 vs STARK 82.0 and TransT 81.4.",
      "GOT-10k (train-only protocol): B-384 AO 72.4 (SR0.5 80.5, SR0.75 67.8), T-224 AO 71.3 vs STARK 68.8 and TransT 67.1.",
      "TNL2k: B-384 55.9 / 57.1, T-224 53.0 / 53.2 — best in Table 1.",
      "Efficiency (Table 2): T-224 98 FPS / 6.4 GMACs / 23M params; B-384 45 FPS / 69.7 GMACs / 91M — both faster than STARK-ST50 (42 FPS) and ST101 (32 FPS).",
    ],
    ablations: [
      "Backbone (Table 3, T-224 w/o motion): Swin vs ResNet-50 — LaSOT 66.7 vs 64.2, LaSOText 46.9 vs 41.8, TrackingNet 80.8 vs 79.5, GOT-10k 70.9 vs 68.2.",
      "Fusion: concatenation vs cross-attention fusion — concatenation wins on all datasets with fewer params (22.7M vs 34.6M) and higher speed (98 vs 72 FPS).",
      "Decoder: target-query decoder degrades all benchmarks versus the encoder-only design even with 2× training pairs.",
      "Loss/post-processing: varifocal over BCE helps everywhere; removing the Hanning window costs 1.0 LaSOT point and 1.3 GOT-10k AO points.",
      "Motion token (Table 4): T-224 66.7→67.2 LaSOT and 70.0→71.3 GOT-10k AO; B-384 70.2→71.3 LaSOT; a learnable dummy token instead scores worse (66.3 LaSOT).",
    ],
    limitations: {
      authorStated: [
        "The framework is presented as a baseline without complex designs such as multi-scale features or temporal template updating.",
        "Context modeling is short-term (fixed local trajectory window); very long-term temporal modeling is left as future possibility rather than claimed.",
      ],
      evident: [
        "The motion token depends on past predictions, so drift can feed back into the decoder with no correction mechanism shown.",
        "Search-region cropping still assumes the target stays within a 4× background window of its previous position.",
      ],
    },
    assumptions: [
      "First-frame ground-truth box is given; one target per sequence.",
      "Target motion is smooth enough for a Hanning-window position prior and fixed-interval trajectory sampling.",
    ],
    computation:
      "B-384 ~45 FPS / T-224 ~98 FPS inference; training on 8×V100 for 300 epochs × 131,072 pairs (≈12h-scale budget implied by schedule); DropPath 0.1.",
    relations: [
      { to: "T009", type: "builds-on", note: "Keeps the Siamese template/search matching formulation of SiamFC, replacing both stages with attention." },
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ compared on all five benchmarks (e.g. LaSOT 49.6 vs 71.3)." },
      { to: "T055", type: "uses-as-baseline", note: "TransT is the hybrid CNN-Transformer competitor beaten on LaSOT (64.9 vs 71.3) and TrackingNet." },
      { to: "T056", type: "uses-as-baseline", note: "STARK-ST101 beaten by 4.2 LaSOT points; its query-decoder design is ablated against." },
      { to: "T031", type: "uses-as-baseline", note: "DiMP/TrDiMP family compared in Table 1 (best 63.9 LaSOT vs 71.3)." },
      { to: "T024", type: "uses-as-baseline", note: "Evaluated and trained on the LaSOT benchmark; headline record is a LaSOT number." },
      { to: "T025", type: "uses-as-baseline", note: "Evaluated under the GOT-10k train-only protocol (AO 72.4)." },
    ],
    concepts: ["sot", "attention", "siamese", "bounding-box", "success-plot", "appearance-features"],
    impact:
      "Established the fully attentional Siamese baseline that later one-stream Transformer trackers (MixFormer, OSTrack family) extend, and introduced trajectory-as-token conditioning for tracking.",
  },
  {
    id: "T065",
    arxiv: "2201.07425",
    title: "WebUAV-3M: A Benchmark for Unveiling the Power of Million-Scale Deep UAV Tracking",
    shortTitle: "WebUAV-3M",
    year: 2022,
    authors: ["Chunhui Zhang", "Guanjie Huang", "Li Liu", "Shan Huang", "Yinan Yang", "Xiang Wan", "Shiming Ge", "Dacheng Tao"],
    venue: "IEEE TPAMI",
    fileName: "2201.07425v4.pdf",
    task: "benchmark",
    tags: ["benchmark", "uav-tracking", "million-scale", "multimodal-tracking", "language-annotation", "audio-annotation", "semi-automatic-labeling", "scenario-evaluation"],
    difficulty: "intro",
    summary:
      "WebUAV-3M is the largest public UAV tracking benchmark: 4,500 videos, ~3.3M densely annotated frames, 223 target classes and 63 motion classes, each video with a natural-language specification plus audio descriptions. It adds a semi-automatic annotation (SATA) pipeline, a fine-grained UAV-tracking-under-scenario-constraint (UTUSC) protocol with seven scenario subtest sets, a new complete-overlap (cAUC) metric, and baselines for 43 trackers.",
    problem:
      "UAV tracking benchmarks were small (seldom over 500 videos), lacked language/audio modalities, covered few target classes, and evaluated only coarse global attributes — hiding what deep trackers can do and preventing rigorous scenario-specific diagnosis and large-scale training.",
    background: ["sot", "bounding-box", "benchmark-design", "success-plot", "multimodal", "iou"],
    previousWork: [
      {
        name: "Small UAV benchmarks (UAV123, DTB70, UAVDT, VisDrone, UAVDark135)",
        limitation:
          "At most a few hundred videos and ~10 target classes with binary global attributes, so evaluations saturate and cannot reveal tracker strengths per scenario.",
        whyThisPaper:
          "4,500 videos / 3.3M frames / 223 classes with continuous framewise scenario indicators and seven 100-video subtest sets enable fine-grained UTUSC diagnosis.",
      },
      {
        name: "Large GOT benchmarks (GOT-10k, LaSOT, TrackingNet)",
        limitation:
          "Ground-camera viewpoint without UAV-specific motion blur, viewpoint change, night scenes, or adversarial UAV scenarios; no language/audio for UAV targets.",
        whyThisPaper:
          "A UAV-viewpoint million-scale dataset with per-video English sentence plus male/female TTS audio, 12 superclasses, and UAV-specific scenarios (dual-dynamic disturbance, adversarial examples).",
      },
      {
        name: "Language/audio tracking datasets (TNL2K, lingual OTB99, Auditory Vehicle Tracking)",
        limitation:
          "Built for generic ground-camera tracking, ignoring severe viewpoint change, continuous camera motion and dark-night UAV conditions.",
        whyThisPaper:
          "First UAV benchmark coupling dense boxes with language and audio, supporting vision-language-audio tracking research.",
      },
    ],
    researchGap:
      "Before this paper, no million-scale, densely labeled, multi-modal UAV tracking benchmark with rigorous scenario-constrained evaluation existed.",
    contribution: [
      "WebUAV-3M dataset: 4,500 videos, ~3.3M frames (30 fps, 28.9 hours), 223 target classes + 63 motion classes in 12 superclasses, 17 global attributes, train/val/test split (3520/200/780).",
      "SATA semi-automatic labeling pipeline (manual grounding → model running → checking/error fixing) labeling the full dataset in three months at 2.99 s/box mean.",
      "UTUSC protocol: seven continuous framewise difficulty indicators (low light, long-term occlusion, small target, high-speed motion, target distortion, dual-dynamic disturbance, adversarial degree) plus seven 100-video scenario subtest sets.",
      "New complete-overlap metric (cAUC) combining overlap, center distance and aspect-ratio consistency; per-video language sentence and dual-voice audio descriptions.",
      "43-tracker baselines with overall, attribute-based, UTUSC, retraining, domain-generalization and cross-dataset transfer experiments.",
    ],
    method: {
      pipeline: ["collect", "clean", "ground", "auto-label", "verify", "attribute-annotate", "language-audio-annotate", "split", "evaluate"],
      architecture:
        "Benchmark construction: 28k+ YouTube videos filtered to 3,617 plus 616 Stanford-Drone and 267 Okutama-Action videos re-annotated; SATA loop alternates a tracking model with human checking; 10-stage quality control with triple verification by authors; annotation rules require tight boxes, absent labels on full occlusion/out-of-view, and annotated vanish/reappear transitions.",
      motionModel: "Not a tracker — scenario indicators quantify motion instead (Eqs. 1, 3 below).",
      appearanceModel: "Not a tracker — appearance difficulty is quantified via the MetaIQA distortion indicator (Eq. 2 below).",
      association: "UTUSC evaluation uses one-pass evaluation (OPE) with Pre, nPre, AUC, mAcc and the new cAUC; no tracker association is proposed.",
      detectionDependency: "None — the benchmark is class-agnostic and tracker-agnostic.",
      trackManagement: "Absent/occluded frames carry explicit absent labels enabling long-term occlusion and re-catch evaluation.",
      optimization: "No model training proposed; retraining studies reuse original authors' architectures and hyperparameters on the WebUAV-3M train split.",
    },
    equations: [
      {
        id: "webuav-speed",
        label: "High-speed motion indicator",
        formula: "∆t = ||pt − pt−1||₂ / sqrt(s_t−1 · s_t · (Tt − Tt−1))",
        variables: [
          { symbol: "pt", meaning: "target center in frame t" },
          { symbol: "s_t", meaning: "target size w_t·h_t in frame t" },
          { symbol: "Tt", meaning: "timestamp of frame t" },
        ],
        intuition:
          "How far the target jumped relative to its own size and the elapsed time — a scale-normalized speedometer.",
        why:
          "Anchor-based trackers assume small inter-frame displacement; the indicator isolates sequences that violate it.",
        where: "Section 4.1; defines the high-speed motion scenario and subtest set.",
        paperIds: ["T065"],
      },
      {
        id: "webuav-distortion",
        label: "Target distortion indicator",
        formula: "Ψt = f_θ(x_t; θ)",
        variables: [
          { symbol: "x_t", meaning: "image crop (4× extension around target) in frame t" },
          { symbol: "f_θ", meaning: "MetaIQA no-reference image-quality model" },
          { symbol: "Ψt", meaning: "predicted distortion score (blur, noise, brightening)" },
        ],
        intuition:
          "Ask a learned image-quality judge how degraded the target looks right now.",
        why:
          "Distorted appearance breaks feature extraction; the indicator selects frames where discriminative power collapses.",
        where: "Section 4.1, Eq. 2; defines the target-distortion scenario.",
        paperIds: ["T065"],
      },
      {
        id: "webuav-dual",
        label: "Dual-dynamic disturbance indicator",
        formula: "Φt = 1 if d ≥ sqrt(s_t−1), else d/sqrt(s_t−1) (ct=1); 0 if no abrupt camera motion",
        variables: [
          { symbol: "d", meaning: "Euclidean center displacement between frames" },
          { symbol: "ct", meaning: "binary abrupt-camera-motion flag" },
          { symbol: "s_t−1", meaning: "previous-frame target size" },
        ],
        intuition:
          "Flags frames where a moving target and a jerking camera coincide — the UAV-specific double trouble.",
        why:
          "Joint target/camera motion causes the worst model drift; global attributes cannot isolate it.",
        where: "Section 4.1, Eq. 3; defines the dual-dynamic scenario.",
        paperIds: ["T065"],
      },
      {
        id: "webuav-adv",
        label: "Adversarial example generation",
        formula: "I^j_{k+1} = ς(I^j_k, η^j) + ε·ψ(I^j_k, η^j), s.t. ρ(I_{k+1} − I_0) < M; Λt = M",
        variables: [
          { symbol: "ς / ψ", meaning: "tangential / normal perturbations of the IoU black-box attack" },
          { symbol: "ρ", meaning: "l2 perturbation magnitude bounded by M" },
          { symbol: "Λt", meaning: "adversarial degree of frame t (0..10000)" },
        ],
        intuition:
          "Add invisible noise that bends the tracker's boxes, with a knob M controlling attack strength.",
        why:
          "Provides the first UAV adversarial-robustness subtest set; TransT and MDNet degrade visibly under it.",
        where: "Section 4.1, Eq. 4; defines the adversarial-example scenario.",
        paperIds: ["T065"],
      },
      {
        id: "webuav-cauc",
        label: "Complete overlap score and cAUC",
        formula: "Sc = |B_G ∩ B_P|/|B_G ∪ B_P| − d²(bc_G,bc_P)/c² − α·v; v = (4/π²)(arctan(w_G/h_G) − arctan(w_P/h_P))²",
        variables: [
          { symbol: "B_G / B_P", meaning: "ground-truth and predicted boxes" },
          { symbol: "d / c", meaning: "center distance and diagonal of the enclosing box" },
          { symbol: "v / α", meaning: "aspect-ratio consistency and its balance weight" },
        ],
        intuition:
          "Overlap minus penalties for being off-center and for having the wrong shape — one number for position, area and shape.",
        why:
          "Pre/AUC measure only center distance or overlap; cAUC reflects all three geometric factors the paper argues a box metric needs.",
        where: "Section 5.3; proposed metric computed by the released toolkit.",
        params: "α > 0 balances the shape penalty against overlap and distance terms.",
        simulator: "metrics",
        paperIds: ["T065"],
      },
    ],
    datasets: ["webuav-3m", "got10k", "lasot", "trackingnet", "uav123", "others"],
    metrics: ["success-auc", "precision", "norm-precision", "fps"],
    baselines: ["KCF", "ECO", "SiamFC", "SiamRPN", "DaSiamRPN", "SiamRPN++", "ATOM", "DiMP", "PrDiMP", "TransT", "TrDiMP", "HiFT", "KeepTrack", "AlphaRefine", "SiamMask", "SiamBAN", "Ocean", "MDNet"],
    results: [
      "Scale: 4,500 videos, ~3.3M frames at 30 fps (28.9 h), 223 target + 63 motion classes, 12 superclasses; split 3520 train / 200 val / 780 test; long-tail class distribution (person 1,305 vs balloon 4).",
      "Overall test (Table 5, 43 trackers): AlphaRefine best (Pre 0.753, nPre 0.643, AUC 0.593, cAUC 0.562, mAcc 0.602), then KeepTrack (0.710/0.543) and PrDiMP (0.674/0.514); fastest are SiamRPN 142.8, KCF 132.9 (CPU), HiFT 122.6 FPS.",
      "Retraining (Table 6): WebUAV-3M training lifts SiamFC 0.351→0.454 AUC, SiamRPN 0.345→0.453, ATOM 0.291→0.390, MDNet 0.390→0.445, GOTURN 0.215→0.318.",
      "Annotation cost (Table 3): SATA 2.99 s/box mean (0.11 easy) vs CVAT 3.86, ViTBAT 4.68, VoTT 8.05, Labelme 15.81.",
      "Transfer (Table 11): WebUAV-3M-trained SiamFC/SiamRPN generalize best to VisDrone and second-best to GOT-10k; vehicle superclass transfers best cross-superclass (0.502 cAUC).",
      "UTUSC: all trackers collapse below low-light indicator 25 and under long occlusion; TransT most occlusion-robust; PrDiMP ranks 1st in high-speed motion; CNN and Transformer trackers both vulnerable to adversarial noise.",
    ],
    ablations: [
      "Data volume (Fig. 13): SiamRPN jumps steeply from 1% to 10% of training videos while SiamFC rises moderately; SiamFC plateaus/overfits at 100% whereas SiamRPN peaks with all data.",
      "Intraclass generalization (Tables 7–8): SUV/sedan-trained and biking/walking-trained SiamFC generalize best to unseen classes; sedan is simultaneously the best source and hardest target (0.434 mean).",
      "Cross-superclass (Table 10): same-superclass training always wins; vehicle (0.502) and person (0.497) transfer best overall.",
      "Scenario ablations (Figs. 11–12): mAcc flat above size 150 px but collapses below 30 px; dual-dynamic disturbance drives all trackers down from indicator 0 to 1.0; adversarial strength shows non-monotonic but revealing degradations.",
    ],
    limitations: {
      authorStated: [
        "Evaluation uses fixed default parameters and weights with no per-sequence tuning, so reported numbers are a lower bound of tracking performance.",
        "Comparing FPS fairly is difficult since speed depends on devices and whether detection time is included.",
        "Future work listed: nighttime tracking, adversarial robustness, multi-modal fusion, and long-tail data imbalance remain unsolved.",
      ],
      evident: [
        "Language/audio annotations are single-sentence TTS conversions, not diverse human utterances, limiting multi-modal training realism.",
        "Ranking-stability analysis and SATA quality rest on the authors' own verification loop rather than independent annotation audit.",
      ],
    },
    assumptions: [
      "Creative-Commons YouTube clips plus re-annotated drone datasets represent real UAV operating conditions.",
      "Continuous indicators computed from boxes/metadata are valid proxies for scenario difficulty.",
    ],
    computation:
      "Evaluation server: Xeon Gold 6230R, 3× RTX A5000, 64 GB RAM; SATA labels 3.3M boxes in three months with ~10 annotators.",
    relations: [
      { to: "T009", type: "uses-as-baseline", note: "SiamFC evaluated (0.351 AUC) and retrained (0.454) on WebUAV-3M." },
      { to: "T026", type: "uses-as-baseline", note: "ATOM evaluated and retrained (0.291→0.390 AUC); also a pretraining-data precedent." },
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ evaluated (0.433 AUC) as a representative deep Siamese baseline." },
      { to: "T025", type: "uses-as-baseline", note: "GOT-10k used for cross-dataset transfer comparison against WebUAV-3M training." },
      { to: "T024", type: "uses-as-baseline", note: "LaSOT cited as the large-scale GOT precedent and retraining-data source." },
    ],
    concepts: ["benchmark-design", "sot", "bounding-box", "success-plot", "multimodal", "mota"],
    impact:
      "Gave UAV tracking its first million-scale multi-modal training resource and the UTUSC fine-grained protocol; retraining and transfer results established WebUAV-3M as a generalization-improving pretraining source.",
  },
  {
    id: "T066",
    arxiv: "2201.13066",
    title: "Single Object Tracking: A Survey of Methods, Datasets, and Evaluation Metrics",
    shortTitle: "SOT survey 2022",
    year: 2022,
    authors: ["Zahra Soleimanitaleb", "Mohammad Ali Keyvanrad"],
    fileName: "2201.13066v1.pdf",
    task: "survey",
    tags: ["survey", "single-object-tracking", "taxonomy", "datasets", "evaluation-metrics", "deep-learning"],
    difficulty: "intro",
    summary:
      "A taxonomy-first survey of single-object tracking organizing methods into feature-based, segmentation-based (bottom-up/joint), estimation-based (Kalman/particle), and learning-based families, with depth on discriminative deep trackers (feature-extraction, Siamese, patch-learning, graph-based), generative methods and reinforcement learning. It catalogs 16 datasets from OTB to TrackingNet and formalizes center error, overlap, tracking length, failure rate, success plots and OPE/TRE/SRE protocols, closing with an AOS table favoring deep trackers.",
    problem:
      "Tracking literature spans decades of fragmented families (filters, segmentation, Bayesian estimation, shallow and deep learning) with incompatible datasets and metrics; newcomers lack a single map from challenges (occlusion, illumination, fast motion) to method choice and evaluation practice.",
    background: ["sot", "bounding-box", "iou", "detection", "siamese", "correlation-filter", "success-plot"],
    previousWork: [
      {
        name: "Correlation-filter review of Fiaz et al. (CF vs non-CF split)",
        limitation:
          "Covers only the correlation-filter axis, leaving segmentation, estimation and the deep-learning zoo outside one taxonomy.",
        whyThisPaper:
          "Adopts the five-family grouping of Verma and expands it into a full SOT tree with learning-based methods at the center.",
      },
      {
        name: "Deep visual tracking review of Li et al.",
        limitation:
          "Restricted to deep trackers, so classical feature/segmentation/estimation baselines that still matter for speed are missing.",
        whyThisPaper:
          "Places deep methods (Siamese, patch, graph, RL) alongside color/texture/flow, segmentation and Kalman/particle foundations with worked examples (GOTURN, MDNet, TCNN).",
      },
      {
        name: "VOT/OTB evaluation methodology of Kristan/Cehovin et al.",
        limitation:
          "Metric definitions live scattered across challenge papers, mixing center error, overlap, re-initialization and robustness plots.",
        whyThisPaper:
          "Collects center error, IoU, tracking length, failure rate, performance plots and OPE/TRE/SRE into one formal section with equations.",
      },
    ],
    researchGap:
      "Before this paper, no single survey jointly taxonomized classical and learning-based SOT methods while also cataloging modern large-scale datasets and their evaluation protocols.",
    contribution: [
      "Four-family taxonomy (feature / segmentation / estimation / learning) with sub-trees: color-texture-flow features; bottom-up vs joint segmentation; Kalman vs particle filters; generative vs discriminative vs reinforcement learning.",
      "Deep-tracker systematization: feature-extraction hybrids (Faster-RCNN/KCF/Siamese detectors) vs end-to-end Siamese, patch-learning (MDNet sampling IoU≥0.7/≤0.3) and graph-based (TCNN tree) designs.",
      "Dataset catalog of 16 benchmarks (OTB50/100, VOT2013–2019, DTB70, NfS, UAV123, TempleColor128, ALOV300++, NUS-PRO, OxUvA, LaSOT, GOT-10k, TrackingNet, YouTube-BB) with sizes, classes and attribute lists.",
      "Metric formalization: center error, IoU overlap with TP/FP/FN reading, AOS, tracking length, failure rate, threshold-free AUC plots, and OPE/TRE/SRE robustness protocols.",
      "AOS comparison (Table 3) on GOT-10k/OTB100/TrackingNet across DSST, KCF, ECO, GOTURN, MDNet and SiamFC showing deep methods on top but dataset-dependent.",
    ],
    method: {
      pipeline: ["categorize", "exemplify", "catalog-datasets", "formalize-metrics", "tabulate"],
      architecture:
        "Literature synthesis, not a tracker: each family is defined, its pipeline drawn (e.g. histogram tracking flowchart, Kalman predict–update loop, particle-filter sampling, GOTURN two-stream conv plus fully connected box head, TCNN tree of shared-conv CNNs, RL matching-plus-policy networks), and representative trackers are worked through.",
      motionModel:
        "Surveyed, not proposed: Kalman prediction–correction for linear-Gaussian motion and particle-filter Monte-Carlo sampling for nonlinear/non-Gaussian motion are exposited as the estimation family.",
      appearanceModel:
        "Surveyed across families: color histograms, Gabor/LBP texture, optical-flow vectors, and learned CNN/Siamese embeddings, with their invariances and failure modes compared.",
      association: "Not applicable — single-object survey; identity association is out of scope.",
      detectionDependency: "Varies by family; feature-extraction hybrids explicitly pair detectors (Faster-RCNN, SSD, YOLO) with KCF, Kalman or LSTM trackers.",
      trackManagement: "Failure handling is surveyed via metrics (tracking length, failure rate with human re-initialization) rather than proposed.",
      optimization: "No new training; MDNet-style online sampling and RL trial-and-error policy learning are described as found in the literature.",
    },
    equations: [
      {
        id: "sotsurvey-iou",
        label: "Region overlap (IoU) score",
        formula: "IoU = OS = |R_G ∩ R_P| / |R_G ∪ R_P|",
        variables: [
          { symbol: "R_G", meaning: "ground-truth box area in frame t" },
          { symbol: "R_P", meaning: "predicted box area in frame t" },
          { symbol: "t", meaning: "frame index" },
        ],
        intuition:
          "What fraction of the union of the two boxes they agree on — 1 is perfect, 0 means the tracker lost the target.",
        why:
          "Unlike center error it couples position and size, and it saturates at zero on failure instead of exploding the average.",
        where: "Section 4.2, Eq. 1; also averaged as AOS and thresholded for success plots.",
        simulator: "iou-track",
        paperIds: ["T066"],
      },
      {
        id: "sotsurvey-tpfpfn",
        label: "Overlap as pixel classification",
        formula: "|R_G ∩ R_P| / |R_G ∪ R_P| = TP / (TP + FN + FP)",
        variables: [
          { symbol: "TP", meaning: "correctly predicted target pixels" },
          { symbol: "FP", meaning: "background pixels predicted as target" },
          { symbol: "FN", meaning: "target pixels missed" },
        ],
        intuition:
          "A box overlap is just pixel-wise precision/recall arithmetic: hits over hits plus both kinds of mistakes.",
        why:
          "Connects tracking accuracy to the detection vocabulary readers already know.",
        where: "Section 4.2, Eq. 2.",
        paperIds: ["T066"],
      },
    ],
    datasets: ["otb", "vot", "got10k", "lasot", "trackingnet", "uav123", "others"],
    metrics: ["success-auc", "precision", "eao"],
    baselines: ["DSST", "KCF", "ECO", "GOTURN", "MDNet", "SiamFC"],
    results: [
      "Table 3 AOS: ECO leads OTB100 (0.687) and is reported at 0.395 GOT-10k / 0.554 TrackingNet; MDNet 0.352 / 0.660 / 0.606; SiamFC 0.392 / 0.569 / 0.571.",
      "Classical baselines trail deep ones on large data: DSST 0.317 / 0.470 / 0.464 and KCF 0.279 / 0.477 / 0.447 across GOT-10k/OTB100/TrackingNet.",
      "Qualitative finding: rankings flip between datasets and criteria, so no family dominates everywhere; deep networks win accuracy but lose speed to shallow ones.",
    ],
    ablations: [
      "No ablations — survey paper; the Table 3 cross of 6 trackers × 3 datasets serves as its comparative analysis.",
      "Protocol comparison: OPE is initial-value sensitive with no re-initialization; TRE averages over start frames; SRE averages 12 spatial perturbations (±10% shifts/scales).",
    ],
    limitations: {
      authorStated: [
        "Deep learning cannot be said to work best in all cases; method choice must weigh the advantages and disadvantages of each family.",
        "Deep networks are slower than shallow ones because of model complexity despite higher accuracy.",
      ],
      evident: [
        "Transformer trackers, discriminative online learners (ATOM/DiMP) and MOT are absent — the deep section stops at Siamese/patch/graph/RL circa 2021.",
        "Dataset statistics contain errors (e.g. LaSOT listed as 123 videos/70 classes; OTB table inconsistencies), so figures must be cross-checked with source papers.",
      ],
    },
    assumptions: [
      "Single target specified in frame one; multi-object identity tracking is explicitly out of scope.",
      "Standard axis-aligned box ground truth underlies all metrics discussed.",
    ],
    computation: "No experiments run; computational claims are qualitative (deep slower than shallow).",
    relations: [
      { to: "T001", type: "uses-as-baseline", note: "KCF tabulated in Table 3 (0.279/0.477/0.447 AOS) as the CF baseline." },
      { to: "T003", type: "uses-as-baseline", note: "MDNet worked through as the patch-learning exemplar and tabulated (0.352/0.660/0.606)." },
      { to: "T009", type: "uses-as-baseline", note: "SiamFC presented as the fully-convolutional Siamese baseline and tabulated." },
      { to: "T011", type: "uses-as-baseline", note: "ECO tabulated as the strongest classical entry (0.687 OTB100)." },
      { to: "T024", type: "uses-as-baseline", note: "LaSOT cataloged with its 14 attributes as a large-scale dataset reference." },
      { to: "T025", type: "uses-as-baseline", note: "GOT-10k cataloged (10k videos/563 classes) and used as a Table 3 evaluation axis." },
    ],
    concepts: ["sot", "bounding-box", "iou", "siamese", "correlation-filter", "success-plot", "benchmark-design"],
    impact:
      "A compact teaching entry point mapping challenges to families, datasets and OPE/TRE/SRE practice; its AOS table documents the pre-Transformer changing of the guard from CF to deep Siamese/patch trackers.",
  },
  {
    id: "T067",
    arxiv: "2202.13514",
    title: "StrongSORT: Make DeepSORT Great Again",
    shortTitle: "StrongSORT",
    year: 2022,
    authors: ["Yunhao Du", "Zhicheng Zhao", "Yang Song", "Yanyun Zhao", "Fei Su", "Tao Gong", "Hongying Meng"],
    fileName: "2202.13514v2.pdf",
    task: "multi-object",
    tags: ["tracking-by-detection", "deepsort", "reid", "kalman-filter", "baseline", "aflink", "gaussian-interpolation", "global-link"],
    difficulty: "intermediate",
    summary:
      "StrongSORT revisits DeepSORT and upgrades it with a YOLOX-X detector, BoT ReID embeddings, EMA feature updating, ECC camera compensation, NSA Kalman, joint motion–appearance cost and vanilla matching — a strong fair baseline. Two plug-and-play modules address MOT's 'missing' problems: AFLink (appearance-free tracklet linking) and GSI (Gaussian-process-smoothed interpolation). StrongSORT++ sets state of the art on MOT17, MOT20, DanceTrack and KITTI.",
    problem:
      "MOT progress was confounded by everyone using different detectors, embeddings, external data and inference tricks, so no fair baseline existed; meanwhile online trackers fragment trajectories (missing association) and suffer gaps from occlusions/low resolution (missing detection) that linear interpolation fixes only crudely.",
    background: ["mot", "tracking-by-detection", "kalman", "reid", "data-association", "hungarian", "cost-matrix", "mota", "idf1", "hota"],
    previousWork: [
      {
        name: "DeepSORT (Faster R-CNN + simple CNN + feature bank + cascade matching)",
        limitation:
          "Outdated detector/embeddings, noise-sensitive feature bank, motion used only as a gate, and cascade priors that throttle strong trackers — misread as a paradigm failure rather than a component gap.",
        whyThisPaper:
          "Component-wise upgrades alone (Table I: 77.3→82.3 IDF1 on MOT17 val) prove the TBD paradigm is still competitive and yield the fair baseline.",
      },
      {
        name: "Appearance-based global linkers (TNT, TPM, ReMOT, GIAOTracker)",
        limitation:
          "They re-encode tracklet appearance with heavy 3D/TP models plus many thresholds, costing large compute and staying fragile under occlusion.",
        whyThisPaper:
          "AFLink links tracklets from spatiotemporal cues only (frame id + x,y over 30 frames), training in ~10 s and adding 1.7 ms/image.",
      },
      {
        name: "Linear interpolation and MAT/MAATrack gap-filling",
        limitation:
          "Linear interpolation ignores motion and jitters; MAT-style fixes need extra single-object trackers, CMC models or Kalman passes.",
        whyThisPaper:
          "GSI models each trajectory with Gaussian-process regression and an adaptive length-based smoothness, smoothing noise and interpolating in one 7.1 ms/image step.",
      },
    ],
    researchGap:
      "Before this paper, the community lacked a strong, fairly equipped DeepSORT baseline, and lightweight appearance-free global linking plus motion-aware interpolation for the two 'missing' problems.",
    contribution: [
      "StrongSORT: DeepSORT equipped with YOLOX-X, BoT embeddings, EMA updating (α=0.9), ECC compensation, NSA Kalman, λ=0.98 appearance–motion cost and vanilla Hungarian matching.",
      "AFLink: two-branch temporal (7×1 convs) + fusion (1×3 conv) network predicting tracklet-pair connectivity from motion only, solved as linear assignment under 30-frame/75-px gates.",
      "GSI: Gaussian-process regression (RBF kernel) interpolation with adaptive λ=τ·log(τ³/l), τ=10, max gap 20 frames — simultaneously a detection-noise filter.",
      "StrongSORT++ (StrongSORT+AFLink+GSI) achieving SOTA on MOT17 (64.4 HOTA/79.5 IDF1), MOT20 (62.6/77.0), DanceTrack (55.6/55.2) and KITTI.",
      "Demonstration that cascade matching helps weak trackers but harms strong ones (up to +1.8 IDF1 by removing it), plus plug-in gains on CenterTrack, TransTrack and FairMOT.",
    ],
    method: {
      pipeline: ["detect", "embed", "predict", "compensate", "associate", "link", "interpolate"],
      architecture:
        "Tracking-by-detection: YOLOX-X detections (NMS 0.8, conf 0.6) → BoT appearance embeddings with EMA state → NSA-Kalman motion prediction on ECC-compensated frames → weighted appearance+motion cost → vanilla global Hungarian matching (threshold 0.45); offline AFLink stitches tracklets and GSI smooths/interpolates trajectories.",
      motionModel:
        "Kalman filter with ECC (Euclidean warp) camera-motion compensation and noise-scale-adaptive covariance R^e=(1−c)R; GSI refines trajectories afterwards with length-adaptive GP smoothing.",
      appearanceModel:
        "BoT ReID embeddings (DukeMTMC-pretrained) maintained by exponential moving average e_t=αe_{t−1}+(1−α)f_t, replacing DeepSORT's 100-frame feature bank; AFLink deliberately uses no appearance.",
      association:
        "Single-stage global assignment on C=λA_a+(1−λ)A_m (λ=0.98, Mahalanobis motion gated), replacing the matching cascade; AFLink resolves residual fragmentation as a second linear-assignment pass (score>0.95).",
      reid:
        "BoT with batch-norm neck trained for person ReID; cosine distance between EMA state and detection embedding is the appearance cost.",
      detectionDependency:
        "Entirely detector-bound: YOLOX-X trained on CrowdHuman plus MOT17-half (plus CityPersons/ETHZ for test); high score threshold causes the FN-heavy MOTA profile.",
      trackManagement:
        "Standard TBD birth/death on match status; AFLink merges short tracklets into full trajectories offline; GSI fills gaps up to 20 frames.",
      loss:
        "AFLink trained as binary classification with cross-entropy on tracklet pairs cut with spatiotemporal noise (1:3 positive:negative), Adam with cosine schedule for 20 epochs.",
    },
    equations: [
      {
        id: "strongsort-appearance",
        label: "Appearance matching cost",
        formula: "d(i,j) = min{1 − f_jᵀ f_k | f_k ∈ B_i}",
        variables: [
          { symbol: "f_j", meaning: "appearance embedding of detection j" },
          { symbol: "B_i", meaning: "feature bank (later EMA state) of tracklet i" },
          { symbol: "d(i,j)", meaning: "minimum cosine distance used as assignment cost" },
        ],
        intuition:
          "A track remembers what its object looked like; a new detection pays the distance to the closest memory.",
        why:
          "Appearance disambiguates crossings where motion alone is symmetric; BoT+EMA make the memory discriminative yet noise-robust.",
        where: "Section III-A, Eq. 1; appearance branch of DeepSORT/StrongSORT.",
        simulator: "reid",
        paperIds: ["T067"],
      },
      {
        id: "strongsort-kalman",
        label: "Kalman predict–update with NSA noise",
        formula: "x̂'=Fx̂; P'=FPFᵀ+Q; K=P'Hᵀ(HP'Hᵀ+R^e)⁻¹; x=x̂'+K(z−Hx̂'); R^e=(1−c)R",
        variables: [
          { symbol: "x̂ / P", meaning: "track state mean and covariance" },
          { symbol: "F / H", meaning: "state-transition and observation models" },
          { symbol: "K", meaning: "Kalman gain blending prediction with measurement z" },
          { symbol: "c", meaning: "detection confidence lowering noise R^e for clean boxes" },
        ],
        intuition:
          "Predict where each track should be, then correct with the detection — trusting confident detections more and shaky ones less.",
        why:
          "Vanilla Kalman trusts all detections equally; NSA weighting plus ECC compensation fixes camera-motion and low-quality-box errors.",
        where: "Section III, Eqs. 2–6 and 9; motion branch every frame.",
        params: "ECC warp mode MOTION_EUCLIDEAN; NSA needs calibrated confidence scores to work.",
        simulator: "kalman",
        paperIds: ["T067"],
      },
      {
        id: "strongsort-ema",
        label: "EMA appearance update",
        formula: "e_t = α·e_{t−1} + (1−α)·f_t, α = 0.9",
        variables: [
          { symbol: "e_t", meaning: "track appearance state at frame t" },
          { symbol: "f_t", meaning: "embedding of the newly matched detection" },
          { symbol: "α", meaning: "momentum retaining history versus new evidence" },
        ],
        intuition:
          "Blend the new look into a running portrait instead of keeping a gallery of 100 old photos.",
        why:
          "Suppresses single-frame detection noise, improves IDF1 (+0.4) and is faster (+1.2 FPS) than the feature bank.",
        where: "Section III-B, Eq. 7; applied on every successful match.",
        simulator: "reid",
        paperIds: ["T067"],
      },
      {
        id: "strongsort-cost",
        label: "Joint appearance–motion assignment cost",
        formula: "C = λ·A_a + (1−λ)·A_m, λ = 0.98",
        variables: [
          { symbol: "A_a", meaning: "appearance (cosine-distance) cost matrix" },
          { symbol: "A_m", meaning: "Mahalanobis motion-distance cost matrix" },
          { symbol: "C", meaning: "fused cost solved by vanilla Hungarian matching" },
        ],
        intuition:
          "Rank candidate matches mostly by looks, slightly by motion plausibility, then solve the whole puzzle at once.",
        why:
          "DeepSORT gated on motion and matched appearance-only in cascade order; joint cost plus global assignment adds +0.8 IDF1 and removes cascade harm (+1.4).",
        where: "Section III-B, Eq. 10; first association stage.",
        simulator: "assoc-cost",
        paperIds: ["T067"],
      },
      {
        id: "strongsort-aflink",
        label: "AFLink association classifier",
        formula: "s = MLP(Concat(pool(φ(T_i)), pool(φ(T_j)))); L = −[y·log σ(s) + (1−y)·log(1−σ(s))]",
        variables: [
          { symbol: "T_*", meaning: "tracklet of (frame id, x, y) over the last N=30 frames, zero-padded" },
          { symbol: "φ", meaning: "temporal 7×1 convs plus 1×3 fusion branch (unshared weights)" },
          { symbol: "s / y", meaning: "predicted link score and binary same-ID label" },
        ],
        intuition:
          "Two short trails either continue each other smoothly in space-time or they do not — no faces needed.",
        why:
          "Global stitching without appearance is cheap, occlusion-proof, and transfers to CenterTrack/TransTrack/FairMOT (+1.3 to +3.8 IDF1).",
        where: "Section IV-A, Eq. 11; offline linking under 30-frame/75-px gates, threshold 0.95.",
        params: "Unshared branch weights; 1:3 positive:negative sampling with random spatiotemporal noise.",
        paperIds: ["T067"],
      },
      {
        id: "strongsort-gsi",
        label: "Gaussian-smoothed interpolation",
        formula: "p = f(t)+ε; f ∼ GP(0, k), k(x,x') = exp(−||x−x'||²/2λ²); P* = K(F*,F)(K(F,F)+σ²I)⁻¹P; λ = τ·log(τ³/l), τ = 10",
        variables: [
          { symbol: "p / t", meaning: "box coordinate (x,y,w,h) at frame t" },
          { symbol: "k / λ", meaning: "RBF kernel and length-adaptive smoothness" },
          { symbol: "l", meaning: "trajectory length driving λ" },
          { symbol: "P*", meaning: "smoothed/interpolated positions on new frame set F*" },
        ],
        intuition:
          "Fit a smooth flexible curve through the noisy observed boxes and read off the missing frames — denoising and gap-filling in one step.",
        why:
          "Linear interpolation kinks at gaps and copies jitter; GSI stabilizes velocity (Fig. 5) and adds +1.4–2.0 MOTA wherever applied.",
        where: "Section IV-B, Eqs. 12–15; applied to gaps up to 20 frames.",
        params: "τ=10; longer tracks get smaller λ (stiffer); max interpolation gap 20 frames (5 on DanceTrack).",
        simulator: "motion",
        paperIds: ["T067"],
      },
    ],
    datasets: ["mot17", "mot20", "dancetrack", "kitti-tracking", "others"],
    metrics: ["mota", "motp", "idf1", "hota", "deta", "assa", "idsw", "fps"],
    baselines: ["SORT", "DeepSORT", "ByteTrack", "OC-SORT", "FairMOT", "CenterTrack", "TransTrack", "JDE", "CSTrack", "RelationTrack", "MOTR"],
    results: [
      "MOT17 test: StrongSORT++ HOTA 64.4 / IDF1 79.5 / MOTA 79.6 / AssA 64.4 / DetA 64.6 / 1,194 IDs — 1st in HOTA/IDF1/AssA/DetA, +2.1 IDF1 over ByteTrack; StrongSORT alone 63.5/78.5/78.3.",
      "MOT20 test: StrongSORT++ HOTA 62.6 / IDF1 77.0 / MOTA 73.8 / 770 IDs — 1st in HOTA/IDF1/AssA with fewest IDs; FN 117,920 vs ByteTrack 87,594 explains the MOTA gap.",
      "DanceTrack: StrongSORT++ HOTA 55.6 / IDF1 55.2 / MOTA 91.1 — best across the table using ByteTrack detections with appearance modules disabled.",
      "KITTI: car HOTA 77.75 / MOTA 90.35 / AssA 78.20; pedestrian HOTA 54.48 / MOTA 67.38 with fewest pedestrian IDs (178).",
      "Ablation (Table I, MOT17 val): BoT +2.2 IDF1; ECC +0.2/+0.3; NSA +0.4 HOTA; EMA +0.4 IDF1 and +1.2 FPS; motion cost +0.8; vanilla matching +1.4 (77.3→82.3 total).",
      "Plug-in (Table II): AFLink+GSI improve all six hosts, e.g. CenterTrack +3.8 IDF1/+2.3 HOTA, FairMOT +1.5/+1.7; extra cost 1.7 ms (AFLink) + 7.1 ms (GSI) per image.",
    ],
    ablations: [
      "Cascade vs vanilla (Table III): cascade helps DeepSORT (+1.1) but harms StrongSORTv4/v5 (−1.8/−1.4 IDF1) — priors throttle strong trackers.",
      "AFLink helps weak trackers most (CenterTrack +3.7 IDF1) while GSI helps strong ones more, being confused by false associations in weak hosts.",
      "NSA Kalman raises HOTA without moving MOTA/IDF1 — a localization-accuracy effect, consistent with its noise-weighting design.",
      "GSI velocity analysis (Fig. 5): linear interpolation velocity jitters wildly from detection noise; GSI trajectories hold stable velocity.",
    ],
    limitations: {
      authorStated: [
        "Relatively low running speed versus joint and appearance-free trackers because of the extra detector plus appearance model (AFLink/GSI themselves are lightweight).",
        "Slightly lower MOTA on MOT17/MOT20 from many missing detections under the high detection-score threshold; a better threshold or association strategy would help.",
        "AFLink cannot fix false associations — it cannot split mixed-up ID trajectories back into accurate tracklets.",
      ],
      evident: [
        "Single fixed hyperparameter set across datasets aids fairness claims but leaves detection-threshold tuning (the known MOTA weakness) unaddressed.",
        "Offline AFLink+GSI passes make StrongSORT++ non-causal despite the online StrongSORT core.",
      ],
    },
    assumptions: [
      "Private-detection protocol with a strong detector available; tracking quality inherits detector recall.",
      "Camera motion is globally parametric (ECC Euclidean warp suffices); ReID embeddings transfer from DukeMTMC pretraining.",
    ],
    computation:
      "StrongSORT 7.5 FPS / StrongSORT++ 7.1 FPS on MOT17 (single-GPU setting); AFLink trains in ~10 s and tests MOT17 in ~10 s; GSI adds 7.1 ms/image.",
    relations: [
      { to: "T013", type: "builds-on", note: "Directly revisits and upgrades the DeepSORT two-branch (appearance+Kalman) framework." },
      { to: "T005", type: "uses-as-baseline", note: "SORT compared on MOT17 (43.1 MOTA) and MOT20 as the motion-only ancestor." },
      { to: "T034", type: "builds-on", note: "Adopts the JDE-style EMA feature update and λ=0.98 appearance–motion cost weighting." },
      { to: "T043", type: "uses-as-baseline", note: "FairMOT compared on MOT17/MOT20 and used as an AFLink+GSI plug-in host." },
      { to: "T061", type: "uses-as-baseline", note: "ByteTrack is the closest SOTA beaten on IDF1/HOTA (MOT17 77.2→79.5, MOT20 74.9→77.0)." },
      { to: "T073", type: "uses-as-baseline", note: "OC-SORT compared with and without linear interpolation on MOT17/MOT20/KITTI/DanceTrack." },
    ],
    concepts: ["tracking-by-detection", "kalman", "reid", "data-association", "cost-matrix", "hungarian", "track-management", "idf1", "hota", "mota"],
    impact:
      "Reset the MOT baseline bar: subsequent papers must beat a properly equipped DeepSORT, and AFLink/GSI became standard plug-in boosters; the cascade-harms-strong-trackers finding changed matching practice.",
  },
  {
    id: "T068",
    arxiv: "2203.01885",
    title: "TCTrack: Temporal Contexts for Aerial Tracking",
    shortTitle: "TCTrack",
    year: 2022,
    authors: ["Ziang Cao", "Ziyuan Huang", "Liang Pan", "Shiwei Zhang", "Ziwei Liu", "Changhong Fu"],
    fileName: "2203.01885v3.pdf",
    task: "single-object",
    tags: ["aerial-tracking", "uav", "temporal-context", "adaptive-convolution", "temporal-transformer", "siamese", "real-time", "edge-deployment"],
    difficulty: "intermediate",
    summary:
      "TCTrack exploits temporal context at two Siamese pipeline stages: TAdaCNN (online temporally adaptive convolution recalibrating AlexNet weights from a queue of L past frame descriptors) for feature extraction, and AT-Trans (encoder–decoder temporal transformer with an information filter and online-updated prior) for similarity-map refinement. With a lightweight AlexNet backbone it tops four aerial benchmarks at 125.6 FPS on PC and over 27 FPS on a Jetson AGX Xavier in real UAV flights.",
    problem:
      "Aerial tracking faces motion blur, camera motion and occlusion on power-limited platforms; Siamese and DCF trackers process frames independently (or touch only the template), ignoring the strong correlations across consecutive frames, while deeper backbones that could compensate are too heavy for UAVs.",
    background: ["sot", "bounding-box", "siamese", "attention", "motion-model", "success-plot", "appearance-features"],
    previousWork: [
      {
        name: "Siamese trackers with static templates (SiamFC, SiamRPN++, SiamAPN)",
        limitation:
          "Each frame is matched independently against a fixed first-frame template, so appearance change from fast motion or occlusion breaks the correlation peak.",
        whyThisPaper:
          "TAdaCNN adapts features per frame from recent history and AT-Trans refines the similarity map with accumulated temporal prior — TCTrack beats SiamRPN++ by ~4.3% AUC on UAV123.",
      },
      {
        name: "Single-stage temporal methods (dynamic templates, memory/graph/transformer template updates)",
        limitation:
          "Temporal information enters at only one stage (the template feature), leaving feature extraction and similarity-map reasoning temporally blind.",
        whyThisPaper:
          "Two-level design covers contexts before correlation (TAdaCNN) and after it (AT-Trans); ablations show each level adds several AUC points and they stack.",
      },
      {
        name: "DCF trackers with temporal penalties (STRCF, AutoTrack, ARCF)",
        limitation:
          "Efficient on UAVs via Fourier tricks but representationally weak, failing fast motion and severe appearance variation (Fig. 6: best DCF far below TCTrack).",
        whyThisPaper:
          "A temporally enriched lightweight Siamese net keeps edge-friendly speed (125.6 FPS) while exceeding both DCF and deeper Siamese competitors.",
      },
    ],
    researchGap:
      "Before this paper, no Siamese aerial tracker integrated temporal context into both online feature extraction and similarity-map refinement with a memory-efficient online update.",
    contribution: [
      "Online TAdaConv: per-frame weight/bias calibration Wt=Wb·αw_t, bt=bb·αb_t from a queue of L frame descriptors via 1D convolutions, turning AlexNet into TAdaCNN at negligible frame-rate cost.",
      "AT-Trans: adaptive temporal transformer whose encoder integrates previous prior Fm_{t−1} with current map Ft (querying the current map), filters with α=FFN(GAP(·)), and whose decoder cross-attends temporal prior into the similarity map — memory-efficient via online prior update.",
      "Full TCTrack pipeline (TAdaCNN → depthwise correlation → AT-Trans → classification/regression) plus TCTrack-L variant, trained on 4-frame clips from VID/LaSOT/GOT-10k.",
      "State of the art on UAV123 (0.604), DTB70 (0.622), UAV123@10fps (0.588) and UAVTrack112_L (0.582) with real-world Jetson deployment over 27 FPS.",
    ],
    method: {
      pipeline: ["extract-adaptive", "correlate", "encode-temporal", "filter", "decode-refine", "classify-regress"],
      architecture:
        "AlexNet backbone (ImageNet-pretrained, last two convs replaced by online TAdaConv) → depthwise correlation with the template → 1-conv feature Ft → AT-Trans (6-head attention encoder with temporal filter + 2-layer decoder) → classification/regression heads; template 127², search 287².",
      motionModel:
        "Implicit: the encoder's running temporal prior accumulates target-location history from frame one, relocating occluded/fast targets; no Kalman or explicit dynamics.",
      appearanceModel:
        "Template fixed from frame one; per-frame appearance variation is absorbed by TAdaConv weight calibration and the filtered temporal prior rather than template updates.",
      detectionDependency: "None — class-agnostic Siamese tracking with no external detector.",
      trackManagement: "No re-detection module; occlusion survival comes from accumulated prior (documented +11.4% occlusion success).",
      loss: "Standard Siamese classification plus box regression losses (details deferred to supplementary material).",
      optimization:
        "100 epochs on 2× TITAN RTX, batch 124 pairs of 4-frame clips; backbone frozen 10 epochs then lr 0.005→0.0005 log-decay, SGD momentum 0.9; AT-Trans randomly initialized, TAdaConv as in [55].",
    },
    equations: [
      {
        id: "tctrack-tadaconv",
        label: "Online temporally adaptive convolution",
        formula: "X̃t = Wt ∗ Xt + bt; Wt = Wb·αw_t; bt = bb·αb_t; αw_t = Fw(X̂)+1; αb_t = Fb(X̂)+1",
        variables: [
          { symbol: "Xt / X̃t", meaning: "input/output features of the layer at frame t" },
          { symbol: "Wb / bb", meaning: "learnable base weight and bias shared across frames" },
          { symbol: "X̂", meaning: "queue of L frame descriptors (GAP-pooled) including the current frame" },
          { symbol: "Fw / Fb", meaning: "1D temporal convolutions (kernel L, zero-initialized) producing calibration factors" },
        ],
        intuition:
          "Nudge each convolution's knobs per frame based on what the last few frames looked like — same network, slightly retuned every instant.",
        why:
          "Injects temporal context into feature extraction itself (a first for tracking) with only descriptor-level overhead; L=3 chosen on the efficiency knee.",
        where: "Section 3.1, Eq. 1; replaces the last two AlexNet convolutions, online one frame at a time.",
        params: "Longer queue L raises precision/AUC (Table 4: L=1→3, 0.561→0.580) at small cost; short histories pad with the first frame.",
        paperIds: ["T068"],
      },
      {
        id: "tctrack-queue",
        label: "Temporal context queue",
        formula: "X̂ = Cat(X̂t, X̂t−1, …, X̂t−L+1); X̂t = GAP(Xt)",
        variables: [
          { symbol: "X̂t", meaning: "global-average-pooled descriptor of frame t" },
          { symbol: "L", meaning: "queue length (3 in the final tracker)" },
          { symbol: "Cat", meaning: "concatenation along the temporal axis" },
        ],
        intuition:
          "A short rolling summary of recent scene appearance that the calibration convolutions read.",
        why:
          "Past-only context matches real tracking causality while keeping per-frame compute constant.",
        where: "Section 3.1, Eq. 2; feeds both TAdaConv calibration branches.",
        paperIds: ["T068"],
      },
      {
        id: "tctrack-correlation",
        label: "Depthwise-correlation similarity map",
        formula: "Rt = φ_tada(Z) ★ φ_tada(Xt); Ft = F(Rt)",
        variables: [
          { symbol: "φ_tada", meaning: "TAdaCNN backbone with online calibrated weights" },
          { symbol: "Z / Xt", meaning: "template and t-th frame search region" },
          { symbol: "★", meaning: "depthwise correlation operator" },
          { symbol: "F", meaning: "post-correlation convolution producing Ft" },
        ],
        intuition:
          "Slide the temporally enriched template over the enriched search image; bright spots are candidate target locations.",
        why:
          "Keeps the proven Siamese matching core while both sides now carry temporal context.",
        where: "Section 3.1, Eq. 3; bridge between TAdaCNN and AT-Trans.",
        simulator: "siamese",
        paperIds: ["T068"],
      },
      {
        id: "tctrack-encoder",
        label: "Adaptive temporal encoder with information filter",
        formula: "F1t = Norm(Ft + MH(Fm_{t−1}, Ft, Ft)); F2t = Norm(F1t + MH(F1t,F1t,F1t)); Ff_t = F2t + F(Cat(F2t,F1t))·α; Fm_t = Norm(Ff_t + MH(Ff_t,Ff_t,Ff_t))",
        variables: [
          { symbol: "Fm_{t−1} / Fm_t", meaning: "previous/current temporal prior knowledge" },
          { symbol: "Ft", meaning: "current similarity-map feature" },
          { symbol: "MH", meaning: "6-head multi-head attention (Eq. 4)" },
          { symbol: "α", meaning: "filter gate α = FFN(GAP(F(F1t))) suppressing blur/occlusion context" },
        ],
        intuition:
          "Ask the current map what past knowledge is still relevant, throw away the corrupted parts, and roll the cleaned remainder into a running memory.",
        why:
          "Unfiltered accumulation confuses the tracker (Table 3: multi-frame without filter collapses); the filter plus current-as-query choice gives +9.8% overall AUC over plain Transformer.",
        where: "Section 3.2, Eqs. 5–7; prior updated per frame, no history stored.",
        params: "Convolutional (first-frame) prior init beats random by ~6% under occlusion; querying Ft beats querying Fm_{t−1} by >10% on motion.",
        paperIds: ["T068"],
      },
      {
        id: "tctrack-decoder",
        label: "Adaptive temporal decoder refinement",
        formula: "F3t = Norm(Ft + MH(Ft,Ft,Ft)); F4t = Norm(F3t + MH(F3t, Fm_t, Ft)); F*t = Norm(F4t + FFN(F4t))",
        variables: [
          { symbol: "Fm_t", meaning: "current temporal prior from the encoder" },
          { symbol: "F*t", meaning: "refined similarity map fed to the heads" },
          { symbol: "MH / FFN", meaning: "multi-head attention and feed-forward refinement" },
        ],
        intuition:
          "Polish the current evidence map by cross-examining it with everything remembered — the map keeps its voice, memory only advises.",
        why:
          "Decoding (rather than overwriting) preserves spatial detail while injecting temporal disambiguation, visibly cleaning maps under camera motion and occlusion (Fig. 5).",
        where: "Section 3.2, Eq. 8; output goes to classification/regression.",
        paperIds: ["T068"],
      },
    ],
    datasets: ["uav123", "others"],
    metrics: ["success-auc", "precision", "fps"],
    baselines: ["SiamFC", "SiamRPN++", "DaSiamRPN", "SiamAPN", "SiamAPN++", "HiFT", "ECO", "CCOT", "STRCF", "AutoTrack", "ARCF", "TransT"],
    results: [
      "UAV123: precision 0.800 / success 0.604 — best of 30+ trackers, +3.0% AUC over HiFT and +4.3% over SiamRPN++.",
      "DTB70: precision 0.813 / success 0.622 — 1st, +5% AUC over the next best, proving motion robustness.",
      "UAV123@10fps: precision 0.774 / success 0.588 — best under abrupt 10-FPS motion.",
      "UAVTrack112_L: success 0.582 / precision 0.786 — best long-term aerial result in Table 2.",
      "Attributes (Fig. 7): best on fast camera motion (0.630), background clutter (0.403), partial occlusion (0.518) and deformation (0.647).",
      "Speed: 125.6 FPS on PC (50.4 for TCTrack-L in Fig. 8 scale) and over 27 FPS on Jetson AGX Xavier without TensorRT (15.29% RAM, 3% VRAM, 46% GPU).",
      "Deep-tracker comparison (Fig. 8): competitive with TransT (best deep tracker) while 2.49× faster on the same GPU.",
    ],
    ablations: [
      "AT-Trans (Table 3, UAV123): plain Transformer 0.750/0.550 → +filter 0.765/0.573; multi-frame unfiltered training collapses (0.732/0.508); full model 0.800/0.604, with +12.0%/+15.1% on motion and +11.4% occlusion success.",
      "TAdaConv length (Table 4): L=1 gives 0.749/0.561, L=2 0.774/0.573, L=3 0.776/0.580 — diminishing returns motivate L=3.",
      "Prior init and query: first-frame convolutional init beats random, especially in occlusion (~+6%); querying the current map beats querying the prior, especially in motion (>+10%).",
    ],
    limitations: {
      authorStated: [
        "Short-term training clips (length 4) leave very-long-term temporal modeling and long occlusion potential not fully explored.",
        "TensorRT/ONNX deployment versions are future work, not included.",
        "Efficiency plus effectiveness make misuse for unauthorized UAV surveillance easier (negative-impact note).",
      ],
      evident: [
        "Fixed AlexNet + fixed template means extreme appearance change beyond the prior's reach has no re-detection fallback.",
        "Loss/evaluation details are offloaded to supplementary material, so head supervision is not reproducible from the main text alone.",
      ],
    },
    assumptions: [
      "Recent past (L=3 descriptors, 4-frame clips) carries sufficient motion/appearance context for aerial targets.",
      "Onboard Jetson-class compute favors AlexNet latency over mobile-network alternatives given memory-access costs.",
    ],
    computation:
      "125.6 FPS full / real-time TCTrack-L on PC; 27+ FPS on Jetson AGX Xavier; training 100 epochs, batch 124 pairs on 2× TITAN RTX.",
    relations: [
      { to: "T009", type: "builds-on", note: "Siamese template/search framework inherited from SiamFC; SiameseFC compared in all figures." },
      { to: "T028", type: "builds-on", note: "Uses the SiamRPN++ depthwise-correlation operator and evaluates it as a same-backbone competitor." },
      { to: "T011", type: "uses-as-baseline", note: "ECO compared on UAV123/DTB70 figures as the representative strong DCF tracker." },
      { to: "T055", type: "uses-as-baseline", note: "TransT is the best deep tracker in Fig. 8 that TCTrack matches at 2.49× the speed." },
    ],
    concepts: ["siamese", "sot", "attention", "motion-model", "bounding-box", "success-plot", "appearance-features"],
    impact:
      "Showed temporal context beats deeper backbones for aerial tracking — a lightweight AlexNet with two-level temporal reasoning matched TransT-class accuracy at UAV-deployable speed, steering later efficient aerial trackers toward temporal design.",
  },
];
