import type { PaperRecord } from "../types";

/* batch-09 — T069..T078 */

export const BATCH_09: PaperRecord[] = [
{
    id: "T069",
    arxiv: "2203.05328",
    title: "Backbone is All Your Need: A Simplified Architecture for Visual Object Tracking",
    shortTitle: "SimTrack",
    year: 2022,
    authors: ["Boyu Chen", "Peixia Li", "Lei Bai", "Lei Qiao", "Qiuhong Shen", "Bo Li", "Weihao Gan", "Wei Wu", "Wanli Ouyang"],
    fileName: "2203.05328v2.pdf",
    task: "single-object",
    tags: ["transformer", "vision-transformer", "one-stream", "siamese-free", "foveal-window", "simplified-architecture"],
    difficulty: "advanced",
    summary:
      "SimTrack replaces the Siamese backbone plus transformer interaction head with a single one-branch Vision Transformer backbone: exemplar and search images are serialized, concatenated, and fed jointly through the backbone so feature learning and interaction happen in every layer. The search-side output goes directly to a STARK-S corner predictor with no interaction module. A foveal window crops the exemplar center for extra diverse target patches to counter ViT down-sampling loss. Sim-L/14 reaches 70.5% AUC on LaSOT and 55.6% on TNL2K.",
    problem:
      "Transformer trackers only used attention as a customized interaction head on top of a Siamese (CNN or transformer) backbone, keeping task-specific sub-modules, prior-heavy architecture selection, and a randomly initialized head that needs long training. Representation learning itself never benefited from target-aware interaction.",
    background: ["sot", "bounding-box", "siamese", "attention", "appearance-features", "success-plot"],
    previousWork: [
      {
        name: "Transformer-head Siamese trackers (TransT, STARK)",
        limitation:
          "Interaction lives only in a hand-designed transformer head after independent Siamese feature extraction; the head is randomly initialized and needs ~500 epochs, and the backbone features stay target-agnostic.",
        whyThisPaper:
          "SimTrack moves interaction into the pre-trained backbone itself (Eqs. 4–6), matching baseline accuracy in 200 vs 500 epochs and gaining +2.5/+2.6 LaSOT/TNL2K AUC over the same-backbone STARK-SV.",
      },
      {
        name: "CNN Siamese trackers (SiamFC, SiamRPN++, DiMP)",
        limitation:
          "Cross-correlation or online optimization on CNN features gives only shallow, late interaction with fixed local receptive fields.",
        whyThisPaper:
          "Joint self-attention over concatenated exemplar+search tokens gives bidirectional, multi-level interaction in all 12 blocks (Table 10: halving interaction blocks costs 2.5 AUC points).",
      },
      {
        name: "In-backbone interaction with hand-designed modules (TransInMo)",
        limitation:
          "Inserts separate interaction modellers at the end of a few backbone blocks with exemplar-to-search-only flow, keeping Siamese structure and extra design.",
        whyThisPaper:
          "SimTrack uses no Siamese branches and no added modules — interaction is the backbone's own attention in every block, bidirectional, initialized from CLIP/MAE pre-training.",
      },
    ],
    researchGap:
      "Before this paper, no tracker performed joint feature learning and exemplar–search interaction purely with a generic one-branch transformer backbone and no interaction head.",
    contribution: [
      "SimTrack: serialize exemplar Z and search X into tokens, concatenate, and pass through one ViT backbone (B/32, B/16, L/14 variants, CLIP-initialized) for joint learning + interaction; search output straight to corner predictor.",
      "Distinguishable position embedding for the exemplar (Eq. 7): two FC layers over patch position (i, j) plus target-area ratio Rij, since the smaller exemplar cannot reuse the search positional embedding.",
      "Foveal window strategy: crop a smaller central exemplar region Z* with offset partitioning lines, serialize to extra tokens e0* for denser target detail at modest cost (+0.8 TNL2K AUC, Table 8).",
      "State-of-the-art results: 70.5% LaSOT / 55.6% TNL2K AUC (Sim-L/14), 69.8% GOT-10k AO, 83.4% TrackingNet, 71.2% UAV123, with fewer FLOPs than the baseline and >40 FPS for Sim-B/16.",
    ],
    method: {
      pipeline: ["serialize", "concatenate", "joint-encode", "predict-corners"],
      architecture:
        "One-branch ViT backbone (Sim-B/32, Sim-B/16, Sim-L/14; CLIP vision-branch init; Sim-B/16* drops last 4 layers) over concatenated exemplar + foveal + search tokens; output search tokens reshaped and fed to the STARK-S corner predictor (two probability maps for top-left/bottom-right corners). No transformer head, no decoder, no post-processing.",
      appearanceModel:
        "Target relevance emerges from bidirectional exemplar–search attention in every backbone layer (Eq. 6): search features become exemplar-sensitive and vice versa; foveal tokens add central-target detail.",
      detectionDependency: "None — class-agnostic single-object tracker with no external detector.",
      trackManagement:
        "Exemplar cropped from frame one (2x of target box; 1.5x for L/14); search region cropped around previous prediction (4x); inference is a single forward pass plus coordinate transform.",
      loss:
        "L = λiou·L_iou(bi, bi*) + λL1·L1(bi, bi*) with λiou=2, λL1=5 (Eq. 3): generalized IoU plus L1 on the predicted box.",
      optimization:
        "AdamW, weight decay 1e-4, 500 epochs x 6e4 pairs/epoch, batch 256; backbone lr 1e-5, head lr 1e-4, x0.1 decay after 400 epochs; train on LaSOT+GOT-10k+COCO2017+TrackingNet splits (GOT-10k test: train split only); 8x16GB V100.",
    },
    equations: [
      {
        id: "simtrack-selfattn",
        label: "Baseline per-image self-attention (no interaction)",
        formula: "Att(e^l) = softmax((e^l W_Q)(e^l W_K)^T / sqrt(d)) (e^l W_V); e* = e^l + Att(LN(e^l)); e^{l+1} = e* + FFN(LN(e*))",
        variables: [
          { symbol: "e^l / s^l", meaning: "exemplar / search token sequences into layer l+1" },
          { symbol: "W_Q, W_K, W_V", meaning: "query, key, value projection matrices" },
          { symbol: "d", meaning: "key dimension (scaling factor)" },
          { symbol: "FFN / LN", meaning: "feed-forward network and layer normalization" },
        ],
        intuition:
          "Each image attends only to itself: the backbone learns generic features with no knowledge of the designated target.",
        why:
          "Formalizes the baseline (STARK-SV) whose independent branches the paper removes.",
        where: "Section 3.1, Eqs. 1–2; baseline backbone forward pass.",
        paperIds: ["T069"],
      },
      {
        id: "simtrack-jointattn",
        label: "Joint exemplar–search attention in the one-branch backbone",
        formula: "Att([e^l; s^l]) = softmax([[a(e^l,e^l), a(e^l,s^l)]; [a(s^l,e^l), a(s^l,s^l)]]) [e^l W_V; s^l W_V], a(x,y) = (xW_Q)(yW_K)^T / sqrt(d)",
        variables: [
          { symbol: "a(x, y)", meaning: "scaled query–key affinity between token sets x and y" },
          { symbol: "a(e^l,s^l) / a(s^l,e^l)", meaning: "cross-image interaction terms absent from the baseline" },
          { symbol: "[e^l; s^l]", meaning: "concatenated exemplar and search sequences fed jointly" },
        ],
        intuition:
          "Every layer mixes four affinities: each image's self-attention plus both cross directions, so search features are learned already conditioned on the target.",
        why:
          "This is the core simplification: interaction in every block removes the need for any transformer head or decoder (Table 9: added decoders change nothing).",
        where: "Section 3.2, Eqs. 4–6; every backbone layer, both training and inference.",
        params: "Interaction in 100% of blocks scores 54.8 TNL2K AUC vs 52.3 at 50% and 49.8 at 25% (Table 10).",
        paperIds: ["T069"],
      },
      {
        id: "simtrack-posemb",
        label: "Distinguishable exemplar position embedding",
        formula: "p_e = FCs(i, j, R_ij)",
        variables: [
          { symbol: "p_e", meaning: "position embedding added to exemplar tokens (ps = p0 reused for search)" },
          { symbol: "(i, j)", meaning: "spatial position of the exemplar patch" },
          { symbol: "R_ij", meaning: "ratio of target area inside patch (i, j)" },
          { symbol: "FCs", meaning: "two fully connected layers" },
        ],
        intuition:
          "Tell the backbone which tokens are the (smaller) exemplar and how much target each patch holds, so shared attention can distinguish the two images.",
        why:
          "Reusing the search positional embedding for both images gives no image identity and wrong geometry; the learned embedding adds +0.5 TNL2K AUC (Table 8, 53.5→54.0).",
        where: "Section 3.2, Eq. 7; added to token embeddings before the backbone.",
        paperIds: ["T069"],
      },
      {
        id: "simtrack-loss",
        label: "Box training loss",
        formula: "L = λ_iou · L_iou(b_i, b_i*) + λ_L1 · L_1(b_i, b_i*)",
        variables: [
          { symbol: "b_i / b_i*", meaning: "predicted and ground-truth boxes on the search frame" },
          { symbol: "L_iou / L_1", meaning: "generalized IoU loss and L1 loss" },
          { symbol: "λ_iou = 2, λ_L1 = 5", meaning: "loss weights" },
        ],
        intuition:
          "Penalize both absolute coordinate error (L1) and geometric overlap quality (GIoU).",
        why:
          "Same recipe as STARK-S, keeping the comparison to architecture rather than supervision.",
        where: "Section 3.1, Eq. 3; offline training on sampled exemplar–search pairs.",
        paperIds: ["T069"],
      },
    ],
    datasets: ["lasot", "trackingnet", "got10k", "uav123", "others"],
    metrics: ["success-auc", "norm-precision", "precision", "fps"],
    baselines: ["SiamFC", "ATOM", "DiMP", "SiamRPN++", "SiamFC++", "Ocean", "TransT", "TrDiMP", "KeepTrack", "AutoMatch", "STARK-S", "STARK-ST", "TransInMo"],
    results: [
      "LaSOT: Sim-L/14 70.5 AUC / 79.7 Pnorm; Sim-B/16 69.3 / 78.5; Sim-B/16* 68.7 / 77.5 — all above STARK-ST101 (67.1) and KeepTrack (67.1).",
      "TNL2K: Sim-L/14 55.6 AUC / 55.7 P vs best compared 52.0 (TransInMo); Sim-B/16 54.8 / 53.8.",
      "TrackingNet: Sim-L/14 83.4 / 87.4; Sim-B/16 82.3 / 86.5 vs STARK-ST 82.0 / 86.9.",
      "GOT-10k (train-only): Sim-L/14 AO 69.8 (SR0.5 78.8, SR0.75 66.0); Sim-B/16 AO 68.6 vs STARK-S 67.2.",
      "UAV123: Sim-L/14 71.2, Sim-B/16 69.8 vs TransT 68.1 (Table 2).",
      "Efficiency: Sim-B/16* 14.7G FLOPs beats all compared trackers; Sim-B/16 runs >40 FPS; Sim-B/32 needs only 11.5G FLOPs for 66.2 LaSOT.",
    ],
    ablations: [
      "Framework vs STARK-SV, same backbone (Table 4): ViT-B/32 +3.7/+3.1, ViT-B/16 +2.5/+2.6, ViT-L/14 +1.3/+1.6 LaSOT/TNL2K AUC with similar or fewer FLOPs.",
      "Training speed (Fig. 6/Table 6): SimTrack at 200 epochs matches STARK-SV at 500 epochs (66.8% LaSOT each); same-epoch loss is lower without the randomly initialized head.",
      "Components on TNL2K (Table 8): +Sim framework 52.2→53.5; +PosEmb →54.0; +foveal window →54.8 AUC.",
      "Decoders are redundant (Table 9): 0/1/3/6 extra decoder layers give 54.8/54.8/54.6/54.7 AUC.",
      "Interaction density (Table 10): 100%→50%→25% of blocks interacting gives 54.8→52.3→49.8 AUC.",
      "Pre-training (Table 5, ViT-B/16): MAE 70.3 LaSOT best, then CLIP 69.3, SLIP 67.6, MoCoV3 66.4, DeiT 66.9 — all competitive with SOTA.",
      "Backbone generality (Table 7): Swin-B SimTrack 68.3 LaSOT / 69.4 UAV123 at 15.0G FLOPs; PVT-M 66.6 / 68.5 at 8.9G.",
    ],
    limitations: {
      authorStated: [
        "Both architecture and training techniques can still be optimized for further performance improvements (Conclusion).",
      ],
      evident: [
        "No online template update or multi-scale features — robustness rests entirely on first-frame tokens plus attention.",
        "Foveal window adds exemplar tokens, so larger variants (L/14, 95.4G FLOPs) trade the claimed simplicity for heavy compute.",
      ],
    },
    assumptions: [
      "First-frame ground-truth box is given; one target per sequence.",
      "Exemplar (112x112) and search (224x224) input sizes fixed between training and inference.",
    ],
    computation:
      "Sim-B/16 >40 FPS inference; training 500 epochs x 6e4 pairs on 8x16GB V100; FLOPs 11.5G (B/32) / 25.0G (B/16) / 95.4G (L/14).",
    relations: [
      { to: "T055", type: "uses-as-baseline", note: "TransT compared on all benchmarks (LaSOT 64.9 vs 69.3) as the transformer-head exemplar." },
      { to: "T056", type: "builds-on", note: "Uses the STARK-S corner predictor and training recipe; STARK-SV is the direct baseline beaten by +2.5 LaSOT." },
      { to: "T009", type: "uses-as-baseline", note: "SiamFC compared in Tables 1–3 (e.g. LaSOT 33.6 vs 69.3)." },
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ compared in Tables 1–2 (LaSOT 49.6 vs 69.3)." },
      { to: "T026", type: "uses-as-baseline", note: "ATOM compared in Table 1 (LaSOT 51.5 vs 69.3)." },
      { to: "T031", type: "uses-as-baseline", note: "DiMP/TrDiMP compared in Tables 1–3 (best 63.9 LaSOT vs 69.3)." },
    ],
    concepts: ["sot", "attention", "siamese", "bounding-box", "success-plot", "appearance-features"],
    impact:
      "Showed a plain one-branch ViT with serialized concatenation can replace Siamese backbones plus interaction heads, prefiguring the one-stream OSTrack/MixFormer line and the value of pre-trained backbones for tracking.",
  },
  {
    id: "T070",
    arxiv: "2203.11082",
    title: "MixFormer: End-to-End Tracking with Iterative Mixed Attention",
    shortTitle: "MixFormer",
    year: 2022,
    authors: ["Yutao Cui", "Cheng Jiang", "Limin Wang", "Gangshan Wu"],
    fileName: "2203.11082v2.pdf",
    task: "single-object",
    tags: ["transformer", "mixed-attention", "end-to-end", "online-template-update", "score-prediction", "multi-stage-backbone"],
    difficulty: "advanced",
    summary:
      "MixFormer unifies feature extraction and target–search integration in a Mixed Attention Module (MAM) that applies self-attention within each of the target-template and search token streams plus cross-attention between them, stackable as a progressive multi-stage backbone with depth-wise convolutional projections. An asymmetric MAM variant prunes target-to-search cross-attention so templates stay fixed and cheap, enabling multiple online templates filtered by a score prediction module (SPM). It sets SOTA on LaSOT (79.9% NP), TrackingNet (88.9%), GOT-10k and VOT2020 (0.555 EAO).",
    problem:
      "Prevailing trackers decouple generic CNN feature extraction (pre-trained for recognition, local kernels) from a separate integration module, so features are target-agnostic and fine structure needed for tracking is lost; transformer integration applied only late on abstract features cannot recover it.",
    background: ["sot", "bounding-box", "siamese", "attention", "memory-network", "success-plot"],
    previousWork: [
      {
        name: "Siamese correlation trackers (SiamFC, SiamRPN, SiamBAN, Ocean)",
        limitation:
          "Correlation over separately extracted CNN features gives only global matching with no target-specific feature learning.",
        whyThisPaper:
          "MAM extracts target-conditioned features from the first layer via mixed self+cross attention, beating correlation baselines by ~8 AUC points on LaSOT (Table 4: 59.8→68.4).",
      },
      {
        name: "Online discriminative trackers (ATOM, DiMP, FCOT)",
        limitation:
          "Learned target models are separate optimization loops on top of generic features, adding complexity without end-to-end target-aware extraction.",
        whyThisPaper:
          "Online adaptation is folded into the backbone via multiple templates plus SPM selection instead of a separate optimizer.",
      },
      {
        name: "Transformer-fusion trackers (TransT, STARK, TREG, STMTrack)",
        limitation:
          "Transformers are used only as a late fusion module over frozen-style CNN features, with TransT's progressive ECA+CFA and STARK's encoder–decoder adding parameters for less gain.",
        whyThisPaper:
          "MixFormer-Base (MAM x21, corner head) beats the TransT-style ECA+CFA(4) design 68.4 vs 66.9 LaSOT with fewer params/FLOPs (Table 4 #3 vs #8).",
      },
    ],
    researchGap:
      "Before this paper, no tracker coupled generic feature extraction and target information integration inside one iterative transformer backbone without an explicit integration module.",
    contribution: [
      "Mixed Attention Module (MAM): concatenated target+search tokens with depth-wise conv projections; dual self-attention (own features) and cross-attention (mutual communication) in one operation (Eq. 1).",
      "Asymmetric MAM (Eq. 2): prunes target-to-search cross-attention so template tokens stay unchanged during tracking — 24% faster (19→25 FPS) at equal accuracy, enabling multiple templates.",
      "Progressive multi-stage MAM backbone (Table 2: MixFormer 35.6M MACs-scale / MixFormer-L 183.9M) with overlapped convolutional token embedding per stage, plus a fully-convolutional corner head or a query-based head.",
      "Score Prediction Module (SPM): learnable score token attends search ROI then the first template; MLP+sigmoid predicts template quality, threshold 0.5, updating every 200 frames.",
      "SOTA on five benchmarks: VOT2020 EAO 0.555, LaSOT NP 79.9, TrackingNet 88.9, GOT-10k AO 75.6, UAV123 70.4.",
    ],
    method: {
      pipeline: ["embed", "mixed-attend", "localize", "score", "update-templates"],
      architecture:
        "Backbone: 3 stages of overlapped patch embedding (7x7 stride 4, then 3x3 stride 2) + Ni MAM+MLP blocks (MixFormer: dims 64/192/384; MixFormer-L: 192/768/1024, CVT-21/CVT24-W ImageNet init); search tokens reshaped to Hs/16 x Ws/16 x 6C; head is a fully-convolutional corner head (top-left/bottom-right expectation) or a DETR-style query head with a regression token + 3-layer FFN. No post-processing, no positional embedding, no multi-layer aggregation.",
      appearanceModel:
        "Target-conditioned features via iterative mixed attention; temporal appearance handled by 1 static + N dynamic online templates selected by SPM score.",
      detectionDependency: "None — single-object tracker; search region cropped from previous state.",
      trackManagement:
        "Templates 128x128, search 320x320; dynamic templates refreshed at 200-frame intervals keeping the highest-SPM-score sample; SPM trained 40 extra epochs with backbone frozen.",
      loss:
        "L_loc = 5·L1 + 2·L_GIoU on the box (Eq. 3); L_score is binary cross-entropy on template quality (Eq. 4).",
      optimization:
        "ADAM, weight decay 1e-4, 500 epochs for backbone+head then 40 for SPM; lr 1e-4 → 1e-5 at epoch 400; gradient clip 0.1; BN frozen; train on TrackingNet+LaSOT+GOT-10k+COCO (GOT-10k test: train split only); 8x Tesla V100, batch 32.",
    },
    equations: [
      {
        id: "mixformer-mam",
        label: "Mixed attention (symmetric)",
        formula: "k_m = Concat(k_t, k_s); v_m = Concat(v_t, v_s); Attention_t = Softmax(q_t k_m^T / sqrt(d)) v_m; Attention_s = Softmax(q_s k_m^T / sqrt(d)) v_m",
        variables: [
          { symbol: "q_t, k_t, v_t", meaning: "queries, keys, values of target-template tokens (depth-wise-conv projected)" },
          { symbol: "q_s, k_s, v_s", meaning: "queries, keys, values of search tokens" },
          { symbol: "k_m / v_m", meaning: "concatenated keys / values over target + search" },
          { symbol: "d", meaning: "key dimension" },
        ],
        intuition:
          "Each side queries the union of both sides: self-attention extracts own features while cross-attention mixes target↔search information in the same softmax.",
        why:
          "One operation does what Siamese pipelines split into backbone + fusion module, letting extraction be target-specific from layer one.",
        where: "Section 3.1, Eq. 1; every MAM block of the backbone.",
        paperIds: ["T070"],
      },
      {
        id: "mixformer-asymmam",
        label: "Asymmetric mixed attention",
        formula: "Attention_t = Softmax(q_t k_t^T / sqrt(d)) v_t; Attention_s = Softmax(q_s k_m^T / sqrt(d)) v_m",
        variables: [
          { symbol: "Attention_t", meaning: "target attention restricted to target keys/values only (no search influence)" },
          { symbol: "Attention_s", meaning: "search attention still over the concatenated memory" },
        ],
        intuition:
          "Freeze what the template remembers (ignore possibly distractor-filled search) while still letting the search consult the template.",
        why:
          "Template tokens stay constant during tracking, saving compute (+24% FPS) and avoiding distractor contamination; accuracy unchanged (68.4→68.1, Table 5).",
        where: "Section 3.1, Eq. 2; used in all online-tracking experiments with multiple templates.",
        params: "Required for multi-template use; symmetric version recomputes templates per frame.",
        paperIds: ["T070"],
      },
      {
        id: "mixformer-locloss",
        label: "Localization loss",
        formula: "L_loc = λ_L1 · L_1(B_i, B̂_i) + λ_giou · L_giou(B_i, B̂_i), λ_L1 = 5, λ_giou = 2",
        variables: [
          { symbol: "B_i / B̂_i", meaning: "ground-truth and predicted bounding boxes" },
          { symbol: "L_1 / L_giou", meaning: "L1 and generalized-IoU box losses" },
        ],
        intuition:
          "Supervise the box for both coordinate closeness and overlap geometry.",
        why:
          "Standard tracking box supervision; shared with STARK/TransT practice for fair comparison.",
        where: "Section 3.3, Eq. 3; stage-1 training (500 epochs).",
        paperIds: ["T070"],
      },
      {
        id: "mixformer-scoreloss",
        label: "Score prediction loss",
        formula: "L_score = y_i · log(p_i) + (1 − y_i) · log(1 − p_i)",
        variables: [
          { symbol: "p_i", meaning: "SPM predicted confidence (sigmoid of MLP over the score token)" },
          { symbol: "y_i", meaning: "binary label: reliable vs poor template" },
        ],
        intuition:
          "Teach a judge token to say whether a candidate online template is worth keeping, by comparing mined search content against the first template.",
        why:
          "Naive fixed-interval template update hurts (69.2→66.6 without SPM, Table 6); SPM-gated update gives the best 69.2 LaSOT.",
        where: "Section 3.3, Eq. 4; stage-2 training (40 epochs, backbone frozen).",
        paperIds: ["T070"],
      },
    ],
    datasets: ["lasot", "trackingnet", "got10k", "uav123", "otb", "vot", "others"],
    metrics: ["success-auc", "precision", "norm-precision", "eao", "fps"],
    baselines: ["SiamFC", "SiamRPN++", "ATOM", "DiMP", "TransT", "STARK", "TrDiMP", "KeepTrack", "TREG", "DualTFR", "SiamR-CNN", "PrDiMP", "STMTracker", "Ocean", "FCOT"],
    results: [
      "VOT2020: MixFormer-L EAO 0.555 / Accuracy 0.762 / Robustness 0.855 — top rank, +5.0 EAO over STARK (0.505).",
      "LaSOT: MixFormer-L 70.1 AUC / 79.9 NP / 76.3 P; MixFormer-22k 69.2 / 78.7; MixFormer-1k 67.9 / 77.3 — all above STARK (67.1) and KeepTrack (67.1).",
      "TrackingNet: MixFormer-L 83.9 AUC / 88.9 NP; MixFormer-22k 83.1 / 88.1 vs STARK 82.0 / 86.9.",
      "GOT-10k (train-only): MixFormer-1k* AO 71.2, MixFormer-22k* 70.7; full-data MixFormer-L AO 75.6 / SR0.5 85.7 / SR0.75 72.8.",
      "UAV123: MixFormer-22k 70.4 AUC / 91.8 P; MixFormer-L 69.5 / 91.0 — best in Table 3.",
      "Speed: MixFormer 25 FPS / 23.0G FLOPs; MixFormer-L 18 FPS / 127.8G FLOPs on GTX 1080Ti.",
    ],
    ablations: [
      "Simultaneous vs separate (Table 4, LaSOT): MixFormer-Base 68.4 beats SAM(21)+CAM(1) 59.8, +CAM(3) 60.5, and CVT+ECA/CFA(4) 66.9 with fewer params/FLOPs.",
      "MAM depth (#4–#8): 1→6→11→16→21 MAMs give 65.8→66.2→67.4→68.1→68.4 AUC — hierarchical integration matters.",
      "Heads (#8 vs #9): corner head 68.4 vs query head 66.0; query-only still matches STARK query variant (63.7).",
      "Asymmetric MAM (Table 5): 25 vs 19 FPS at 68.1 vs 68.4 AUC.",
      "Online templates (Table 6): first-only 68.1; +naive online 66.6; +online with SPM 69.2.",
      "Pre-training (Table 7): ImageNet-22k 69.2 vs 1k 67.9; GOT-10k-only training still reaches 62.1, beating most full-data trackers.",
      "OTB100: MixFormer-L tops TransT by +1.3% AUC (Fig. 7).",
    ],
    limitations: {
      authorStated: [
        "Future work: extending MixFormer to multiple object tracking (Conclusion).",
      ],
      evident: [
        "MixFormer-L costs 127.8G FLOPs / 18 FPS — the accuracy leader is not real-time on the reported GPU.",
        "SPM threshold (0.5) and update interval (200) are fixed heuristics with no adaptivity analysis.",
      ],
    },
    assumptions: [
      "First-frame box given; single target; search crop from previous state contains the target.",
      "Motion is locally linear enough for a 320x320 search window; no re-detection after total loss.",
    ],
    computation:
      "MixFormer 25 FPS (23.0G FLOPs), MixFormer-L 18 FPS (127.8G) on GTX 1080Ti; training on 8x Tesla V100 (also fits 8x2080Ti/11GB at batch 8/GPU).",
    relations: [
      { to: "T055", type: "uses-as-baseline", note: "TransT is the separate-fusion competitor beaten on all five benchmarks (LaSOT 64.9 vs 70.1) and ablated as ECA+CFA." },
      { to: "T056", type: "uses-as-baseline", note: "STARK beaten by 5.0 EAO on VOT2020 and 2.9 NP on LaSOT; corner head inspired by STARK." },
      { to: "T009", type: "uses-as-baseline", note: "SiamFC tabulated on all benchmarks (LaSOT 33.6 vs 70.1)." },
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ tabulated (LaSOT 49.6 vs 70.1)." },
      { to: "T026", type: "uses-as-baseline", note: "ATOM tabulated (LaSOT 51.5 vs 70.1)." },
      { to: "T031", type: "uses-as-baseline", note: "DiMP/TrDiMP tabulated (best 63.9 LaSOT vs 70.1)." },
    ],
    concepts: ["sot", "attention", "siamese", "memory-network", "bounding-box", "success-plot"],
    impact:
      "Established the iterative mixed-attention backbone as the compact end-to-end SOT standard of 2022; asymmetric attention plus SPM became the template-update pattern reused by efficient successors (MixFormerV2).",
  },
  {
    id: "T071",
    arxiv: "2203.11991",
    title: "Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework",
    shortTitle: "OSTrack",
    year: 2022,
    authors: ["Botao Ye", "Hong Chang", "Bingpeng Ma", "Shiguang Shan", "Xilin Chen"],
    fileName: "2203.11991v4.pdf",
    task: "single-object",
    tags: ["transformer", "one-stream", "vision-transformer", "early-elimination", "one-shot-tracking", "mae-pretraining"],
    difficulty: "advanced",
    summary:
      "OSTrack concatenates template and search patch embeddings and feeds them through a vanilla MAE-pre-trained ViT, so stacked self-attention simultaneously extracts features and models template–search relations with bidirectional flow — no Siamese branches, no heavy fusion module. A free by-product is an early target–candidate similarity prior, exploited by an in-network early candidate elimination module (keep top-k by center-template attention at layers 4/7/10) that cuts MACs ~26% while slightly helping accuracy. OSTrack-384 hits 73.7% AO on GOT-10k one-shot (+4.3 over SwinTrack) and 71.1% on LaSOT.",
    problem:
      "Two-stream two-stage pipelines extract template and search features independently (target-unaware, weak on unseen classes) and then fuse with either a weak linear operator (information loss) or a heavy transformer module (slow: STARK-S50 spends 7.5ms extracting + 14.1ms fusing per search image).",
    background: ["sot", "bounding-box", "siamese", "attention", "success-plot", "appearance-features"],
    previousWork: [
      {
        name: "Light-fusion two-stream trackers (SiamFC, SiamRPN++, DiMP/ATOM correlation filters)",
        limitation:
          "Single cross-correlation/DCF fusion is a shallow unidirectional linear operation; template features cannot adapt to the search region, losing discriminative information.",
        whyThisPaper:
          "Early concatenation plus iterative self-attention gives mutual guidance from layer one; aligned OSTrack beats aligned two-stream variants on GOT-10k one-shot 71.0 vs 68.8/69.5 (Table 5).",
      },
      {
        name: "Heavy-fusion transformer trackers (TransT, STARK, SwinTrack)",
        limitation:
          "Stacked self/cross-attention fusion after separate CNN/ViT extraction is effective but slow and leaves randomly initialized fusion layers that converge slowly.",
        whyThisPaper:
          "Fusion IS the backbone: same-ViT-aligned OSTrack is 40.2 FPS faster than aligned STARK and 25.6 faster than aligned SwinTrack while scoring higher (Table 5).",
      },
      {
        name: "Adaptive ViT inference (DynamicViT, EViT)",
        limitation:
          "Token sparsification is built for classification with learned gates/fusion and does not transfer to tracking's candidate structure.",
        whyThisPaper:
          "Elimination reuses the tracker's own free similarity score (center-template attention, no extra gates) to drop background candidates inside the network.",
      },
    ],
    researchGap:
      "Before this paper, no tracking framework seamlessly unified feature extraction and relation modeling in a single stream from the raw image pair, nor exploited the resulting early similarity for in-network candidate pruning.",
    contribution: [
      "One-stream one-stage framework: concatenate template+search embeddings into a shared ViT (MAE-pre-trained Base), whose self-attention provably splits into intra-image (extraction) and inter-image (modeling) terms (Eqs. 3–5).",
      "Early candidate elimination: keep ratio ρ=0.7 at layers 4/7/10 using averaged multi-head similarity of the center template token to each search token; restore order + zero-pad for the FCN head.",
      "Lightweight FCN head (4x Conv-BN-ReLU per output): classification score map P, offset O, size S; argmax with Hanning penalty gives the box (Eq. 7); weighted focal + L1 + GIoU loss (Eq. 8).",
      "SOTA on seven benchmarks with a speed/accuracy trade-off: OSTrack-256 105.4 FPS at 69.1 LaSOT; OSTrack-384 58.1 FPS at 71.1 LaSOT / 73.7 GOT-10k AO.",
    ],
    method: {
      pipeline: ["patchify", "concatenate", "joint-encode", "eliminate", "restore", "classify", "regress"],
      architecture:
        "ViT-Base backbone (MAE init) over concatenated template (128x128 / 192x192) + search (256x256 / 384x384) tokens, 2D-interpolated positional embeddings, 3 elimination modules; FCN head reshapes padded search tokens to a 2D map with three outputs (P, O, S).",
      appearanceModel:
        "No separate template model: target-oriented features emerge from mutual template–search guidance; no online update — first-frame template only.",
      detectionDependency: "None — single-object tracker without a detector.",
      trackManagement:
        "Template/search area are 2x/4x (256) or 2x/5x (384) of target box area; Hanning window multiplies the score map at inference; eliminated candidates zero-padded as placeholders.",
      loss:
        "L_track = L_cls + 2·L_iou + 5·L_1 (Eq. 8); L_cls is Gaussian-weighted focal loss (α=2, β=4, Eq. 9); box terms are L1 + generalized IoU.",
      optimization:
        "AdamW, weight decay 1e-4; backbone lr 4e-5, rest 4e-4, x0.1 after 240/300 epochs, 60k pairs/epoch, batch 128; train on COCO+LaSOT+GOT-10k (minus 1k forbidden)+TrackingNet with flip+brightness jitter; GOT-10k-only: 100 epochs, decay at 80; 4x A100; inference on RTX2080Ti.",
    },
    equations: [
      {
        id: "ostrack-attn",
        label: "One-stream self-attention over concatenated pair",
        formula: "A = Softmax([Q_z; Q_x][K_z; K_x]^T / sqrt(d_k)) · [V_z; V_x]",
        variables: [
          { symbol: "Q_z, K_z, V_z", meaning: "queries, keys, values of template tokens" },
          { symbol: "Q_x, K_x, V_x", meaning: "queries, keys, values of search tokens" },
          { symbol: "d_k", meaning: "key dimension" },
          { symbol: "A", meaning: "attention output over the concatenated sequence" },
        ],
        intuition:
          "One softmax over the joint pair lets every token gather from both images at once.",
        why:
          "Concatenation makes extraction and fusion the same highly parallel operation instead of two sequential stages.",
        where: "Section 3.1, Eq. 3; every ViT encoder layer.",
        paperIds: ["T071"],
      },
      {
        id: "ostrack-attnexpand",
        label: "Attention expansion into extraction + modeling terms",
        formula: "Softmax(·) = [W_zz, W_zx; W_xz, W_xx]; A = [W_zz V_z + W_zx V_x; W_xz V_z + W_xx V_x]",
        variables: [
          { symbol: "W_xx / W_zz", meaning: "intra-image weights: feature extraction" },
          { symbol: "W_xz / W_zx", meaning: "inter-image weights: relation modeling (template↔search similarity)" },
        ],
        intuition:
          "The joint attention algebraically separates into 'look at yourself' and 'look at the other image' parts — the proof that one operation does both jobs.",
        why:
          "Justifies dropping the fusion module: W_xz V_z aggregates target information into search features every layer, bidirectionally.",
        where: "Section 3.1, Eqs. 4–5; analysis, visualized in Fig. 4 (early layers already highlight the target).",
        paperIds: ["T071"],
      },
      {
        id: "ostrack-eliminate",
        label: "Early candidate elimination score",
        formula: "w̄_ϕ^x = (1/M) Σ_m w_ϕ^x(m); keep top-k candidates, ρ = k/n = 0.7",
        variables: [
          { symbol: "w_ϕ^x(m)", meaning: "head-m attention from center template token ϕ to each search candidate" },
          { symbol: "ϕ", meaning: "index of the center template token (target representative)" },
          { symbol: "M", meaning: "number of attention heads" },
          { symbol: "ρ", meaning: "token keeping ratio per elimination module" },
        ],
        intuition:
          "Ask the middle of the template how much it likes each search patch, average over heads, and throw away the 30% least-liked patches mid-network.",
        why:
          "Background tokens are identified free from attention (Figs. 4–5); pruning cuts 256-model MACs 25.9% and boosts speed 13.2% while LaSOT rises +0.4 (Table 3).",
        where: "Section 3.2, Eq. 6 context; inserted after MHA at ViT layers 4, 7, 10.",
        params: "ρ=0.7 x3 modules; center-token choice ablated in supplement; larger inputs gain more (384: +40.3% FPS).",
        paperIds: ["T071"],
      },
      {
        id: "ostrack-bbox",
        label: "Box decoding from FCN outputs",
        formula: "(x_d, y_d) = argmax P_xy; (x, y, w, h) = (x_d + O(0,x_d,y_d), y_d + O(1,x_d,y_d), S(0,x_d,y_d), S(1,x_d,y_d))",
        variables: [
          { symbol: "P", meaning: "classification score map" },
          { symbol: "O", meaning: "local offset compensating stride discretization" },
          { symbol: "S", meaning: "normalized width/height map" },
        ],
        intuition:
          "Pick the hottest score location, nudge it by the learned sub-cell offset, and read off the box size stored there.",
        why:
          "Anchor-free direct decoding with no matching step — possible only because search features are already target-aware.",
        where: "Section 3.3, Eq. 7; inference with Hanning-windowed P.",
        paperIds: ["T071"],
      },
      {
        id: "ostrack-loss",
        label: "Overall tracking loss",
        formula: "L_track = L_cls + λ_iou · L_iou + λ_L1 · L_1, λ_iou = 2, λ_L1 = 5",
        variables: [
          { symbol: "L_cls", meaning: "Gaussian-weighted focal classification loss (Eq. 9, α=2, β=4)" },
          { symbol: "L_iou / L_1", meaning: "generalized IoU and L1 box losses" },
        ],
        intuition:
          "Learn where the target is (focal loss on a Gaussian heatmap) and exactly how big (box losses) jointly.",
        why:
          "Balances hard-example classification against precise regression, following STARK practice.",
        where: "Section 3.3, Eq. 8; training on sampled pairs.",
        paperIds: ["T071"],
      },
    ],
    datasets: ["lasot", "trackingnet", "got10k", "uav123", "others"],
    metrics: ["success-auc", "precision", "norm-precision", "fps"],
    baselines: ["SiamFC", "MDNet", "ECO", "SiamRPN++", "DiMP", "SiamR-CNN", "Ocean", "TrDiMP", "TransT", "AutoMatch", "STARK", "KeepTrack", "SwinTrack"],
    results: [
      "GOT-10k one-shot: OSTrack-384 AO 73.7 / SR0.5 83.2 / SR0.75 70.8; OSTrack-256 AO 71.0 — vs SwinTrack-B 69.4 (+4.3 for the 256 model over prior best).",
      "LaSOT: OSTrack-384 71.1 / 81.1 / 77.6; OSTrack-256 69.1 / 78.7 / 75.2 at 105.4 FPS (2x SwinTrack-B's 52 FPS).",
      "TrackingNet: OSTrack-384 83.9 / 88.5 / 83.2; OSTrack-256 83.1 / 87.8 / 82.0 — both above SwinTrack-B (82.5).",
      "LaSOText: OSTrack-384 50.5 AUC (+2.3 over KeepTrack 48.2 at 3x speed); OSTrack-256 47.4 at 105.4 FPS.",
      "NFS / UAV123 / TNL2K: OSTrack-384 66.5 / 70.7 / 55.9 — best on all three (Table 2).",
      "Elimination (Table 3): 256-model MACs 29.0→21.5G (−25.9%), 93.1→105.4 FPS; 384-model 65.3→48.3G, 41.4→58.1 FPS (+40.3%), accuracy neutral-to-positive.",
    ],
    ablations: [
      "Aligned two-stream comparison (Table 5, same ViT/head/loss): OSTrack 68.7 LaSOT / 71.0 GOT-10k AO vs STARK-aligned 67.6/68.8 and SwinTrack-aligned 68.0/69.5; OSTrack trains on 18M pairs vs 30M/39.3M and runs 93.1 vs 52.9/67.5 FPS.",
      "Pre-training (Table 4): MAE 68.7 LaSOT / 73.6 GOT-10k AO beats ImageNet-21k (66.9/70.2), ImageNet-1k (66.1/69.7), and none (60.4/62.7).",
      "Identity/relative position embeddings: no significant gain, omitted (Sec. 3.1).",
      "Template-token choice for elimination and keeping-ratio studies in the supplement.",
    ],
    limitations: {
      authorStated: [],
      evident: [
        "First-frame template only with no online update — long-term drift handling is delegated to the Hanning prior.",
        "Elimination irreversibly discards tokens mid-network; an early similarity mistake cannot be recovered downstream.",
      ],
    },
    assumptions: [
      "First-frame box given; single target; template center token is a valid target representative.",
      "Discarded (zero-padded) candidates are irrelevant background that will not affect classification/regression.",
    ],
    computation:
      "OSTrack-256 105.4 FPS / 21.5G MACs; OSTrack-384 58.1 FPS / 48.3G MACs (RTX2080Ti); training 300 epochs x 60k pairs, batch 128 on 4x A100.",
    relations: [
      { to: "T064", type: "uses-as-baseline", note: "SwinTrack-B is the headline competitor beaten by 1.6–4.3 GOT-10k AO points and re-implemented as SwinTrack-aligned." },
      { to: "T055", type: "uses-as-baseline", note: "TransT compared in Tables 1–2 (LaSOT 64.9 vs 71.1) as the heavy-fusion exemplar." },
      { to: "T056", type: "uses-as-baseline", note: "STARK compared (LaSOT 67.1, GOT-10k 68.8) and re-implemented as STARK-aligned." },
      { to: "T009", type: "uses-as-baseline", note: "SiamFC tabulated (LaSOT 33.6 vs 71.1)." },
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ tabulated (GOT-10k 51.7 vs 73.7)." },
      { to: "T026", type: "uses-as-baseline", note: "ATOM tabulated (LaSOT 51.5 vs 71.1)." },
      { to: "T031", type: "uses-as-baseline", note: "DiMP tabulated (GOT-10k 61.1 vs 73.7)." },
      { to: "T024", type: "uses-as-baseline", note: "LaSOT is the headline long-term benchmark (71.1 AUC)." },
      { to: "T025", type: "uses-as-baseline", note: "GOT-10k one-shot protocol is the headline result (73.7 AO)." },
    ],
    concepts: ["sot", "attention", "siamese", "bounding-box", "success-plot", "appearance-features"],
    impact:
      "Made one-stream ViT tracking the default 2022 recipe (best one-shot generalization: +6.5 SR0.75 on GOT-10k) and introduced in-network candidate elimination, widely reused for efficient trackers.",
  },
  {
    id: "T072",
    arxiv: "2203.13437",
    title: "BCOT: A Markerless High-Precision 3D Object Tracking Benchmark",
    shortTitle: "BCOT",
    year: 2022,
    authors: ["Jiachen Li", "Bin Wang", "Shiqiang Zhu", "Xin Cao", "Fan Zhong", "Wenxuan Chen", "Te Li", "Jason Gu", "Xueying Qin"],
    fileName: "2203.13437v1.pdf",
    task: "3d-tracking",
    tags: ["benchmark", "3d-tracking", "markerless", "textureless", "pose-annotation", "multi-view-optimization", "binocular"],
    difficulty: "intermediate",
    summary:
      "BCOT is a markerless high-precision 3D object tracking benchmark: an object-centered multi-view joint optimization annotates real binocular video (two ~90° synchronized MER-131-210U3 cameras) by minimizing shape re-projection error across views, reaching <2mm annotation error. The dataset has 20 textureless 3D-printed objects, 22 scenes (static/movable rigs, indoor/outdoor), 404 sequences and 126K images. Seven monocular textureless trackers are re-evaluated; ACCV2020 leads at 5°/5cm (89.0) while TVCG2021 leads rotation at 2°/2cm.",
    problem:
      "Template-based 3D tracking had no high-precision real-scene benchmark: synthetic RBOT lacks authenticity, OPT needs invasive markers and static objects, YCB-Video has static objects plus depth-misalignment error — and monocular RGB pose is fundamentally uncertain along the viewing (Z) direction.",
    background: ["3d-tracking", "benchmark-design", "bounding-box", "motion-model"],
    previousWork: [
      {
        name: "RBOT (semi-synthetic tracking dataset)",
        limitation:
          "Rendered moving objects over real backgrounds differ from real video in camera effects and motion, with pre-set shared trajectories limiting diversity.",
        whyThisPaper:
          "BCOT captures fully real moving objects and cameras with free motion in 22 designed scenes.",
      },
      {
        name: "OPT (marker-based real tracking dataset)",
        limitation:
          "Artificial markers around objects invade the background and objects may not move, breaking naturalness.",
        whyThisPaper:
          "Markerless joint optimization from two views needs no scene invasion and allows free object/camera motion.",
      },
      {
        name: "YCB-Video / pose-estimation datasets (Linemod, T-LESS, TOD, StereOBJ-1M)",
        limitation:
          "Static objects and/or depth-sensor error (≥17mm random) and keypoint RMSE (2.3–3.4mm) prevent high-precision tracking evaluation; single-frame sets cannot evaluate trackers.",
        whyThisPaper:
          "Binocular joint optimization guarantees <2mm error (Table 2) on dynamic sequences purpose-built for tracking.",
      },
    ],
    researchGap:
      "Before this paper, no markerless method could annotate accurate 3D poses of real moving textureless objects, so no high-precision real-scene 3D tracking benchmark existed.",
    contribution: [
      "Object-centered multi-view pose model (Eqs. 1–3): reformulates the camera projection and Gauss-Newton update in the object frame Oo so all views' sample points join one likelihood (Eqs. 7–8).",
      "Joint shape re-projection optimization over N views using a region-based energy (Eqs. 4–6), cutting Z-axis error from 22.09mm (mono) to 0.28mm at 90° (Table 3).",
      "BCOT benchmark: 20 textureless objects (91.7–229.5mm), 22 scenes, 404 sequences, 126K images at 640x512/30FPS annotated from 1280x1024/60FPS binocular capture; error <2mm by theory and validation.",
      "Re-evaluation of 7 SOTA monocular textureless trackers under n°/ncm and ADD metrics (Table 6, Fig. 7) plus multi-view precision studies over angle, camera count, resolution and free motion.",
    ],
    method: {
      pipeline: ["capture-binocular", "calibrate", "annotate-frame-by-frame", "downsample", "evaluate-monocular"],
      architecture:
        "Annotation engine: two orthogonal (90°) synchronized high-resolution high-speed cameras (5ms exposure, Cam1 complex scene + Cam2 clean background); object-centered Gauss-Newton joint optimization of region-based shape energy; first frame roughly manual, then 60FPS frame-by-frame tracking; sequences with >2px re-projection error discarded.",
      motionModel:
        "Template-based 3D tracker with constant inter-frame pose prior at 60FPS capture; relative camera pose fixed on a rigid set (tripod or movable bracket) and pre-calibrated; object motion modes: translation (toy car), suspension (fishing line), handheld; slow movement to preserve precision.",
      appearanceModel:
        "Shape-based (contour/region) features for textureless single-color reflective objects; no point correspondences between views.",
      trackManagement:
        "Lost defined as rotation error >5° or translation error >5cm (reset to GT); max annotation error enforced at 2px per view → 2mm spatial.",
      optimization:
        "Gauss-Newton on the joint energy (Eqs. 7–8); more iterations allowed at annotation time than at tracking time.",
    },
    equations: [
      {
        id: "bcot-projection",
        label: "Object-centered projection model",
        formula: "x = π(K(cTt X̃_t)_3x1) = π(K(oTc^-1 Xt̃_o)_3x1); x = fx·Xc/Zc + cx, y = fy·Yc/Zc + cy",
        variables: [
          { symbol: "x", meaning: "2D image point" },
          { symbol: "K", meaning: "pre-calibrated camera intrinsics" },
          { symbol: "cTt / oTc", meaning: "object pose in camera frame / camera pose in object-centered frame Oo" },
          { symbol: "π", meaning: "perspective division [X/Z, Y/Z]" },
        ],
        intuition:
          "Rewrite projection so every camera speaks the same object-centered language — the prerequisite for pooling views.",
        why:
          "Camera-centered models cannot associate cameras; the Oo frame lets all views constrain one pose (Fig. 2).",
        where: "Section 3.1, Eqs. 1–3 and Eq. 9; used for every sample point in every view.",
        paperIds: ["T072"],
      },
      {
        id: "bcot-jointenergy",
        label: "Joint multi-view energy",
        formula: "Δξ_o = argmin Σ_i=1..N Σ_x∈Ωi F_i(x, ξ'_ci, oT_ci); F = −log(He(Φ(x(ξ_o)))·Pf(x) + (1−He(Φ(x(ξ_o))))·Pb(x))",
        variables: [
          { symbol: "Δξ_o", meaning: "pose increment solved in the object frame" },
          { symbol: "F_i", meaning: "region-based shape re-projection energy of view i (from Tjaden et al.)" },
          { symbol: "Φ / Pf, Pb", meaning: "rendered shape template / foreground-background color posteriors" },
          { symbol: "Ω", meaning: "union of all views' sample points (independent after log)" },
        ],
        intuition:
          "Pull the 3D pose until its rendered silhouette explains foreground vs background colors in ALL cameras at once.",
        why:
          "One view leaves Z-depth ambiguous (Fig. 2a); the orthogonal view's X/Y pins down the first view's Z (Fig. 2b).",
        where: "Section 3.2, Eqs. 4–8; annotation optimization, solved with Gauss-Newton.",
        params: "90° included angle is the operating point; small angles (<30°) still lose tracks (Table 3: 21 lost mono → 0 at 45°+).",
        paperIds: ["T072"],
      },
      {
        id: "bcot-errormetrics",
        label: "Translation, rotation and ADD errors",
        formula: "e(t) = ||t̂ − t||_2; e(R) = cos^-1((trace(R̂^T R) − 1)/2); ADD = (1/M) Σ_i ||(R̂X_i + t̂) − (RX_i + t)||",
        variables: [
          { symbol: "e(t) / e(R)", meaning: "translation (Euclidean) and rotation (geodesic) errors" },
          { symbol: "ADD", meaning: "mean model-point distance between predicted and GT poses" },
          { symbol: "M", meaning: "number of model points" },
        ],
        intuition:
          "Score a 3D tracker by how far the object slid, how much it twisted, and on average how far its surface ended up — in millimetres, not pixels.",
        why:
          "n°/ncm gate correctness (5°/5cm, tightened to 2°/2cm) while ADD curves (Fig. 7) rank fine precision; ADD-S unused since the previous frame disambiguates symmetry.",
        where: "Section 4.7, Eqs. 10–12; benchmark evaluation protocol.",
        simulator: "metrics",
        paperIds: ["T072"],
      },
    ],
    datasets: ["bcot"],
    metrics: ["3d-error"],
    baselines: ["MTAP2019", "TPAMI2019 (Tjaden region-based)", "CGF2020", "ACCV2020 (sparse Gaussian)", "C&G2021", "JCST2021", "TVCG2021"],
    results: [
      "Scale: 20 textureless objects, 22 scenes, 404 valid sequences, 126K images; object sizes 91.7–229.5mm; annotation error <2mm (vs RGB-D ≥17mm, TOD 3.4mm, StereOBJ-1M 2.3mm, Table 2).",
      "Binocular vs mono at 90° (Table 3, 4 objects avg): rotation 1.62→0.76°, tx 4.36→0.28mm, ty 2.39→0.23mm, tz 22.09→0.28mm; lost 21→0.",
      "Free-moving cameras (Table 4): two views give r 0.59–0.80°, tz 0.33–0.78mm vs mono 1.47°/12.96mm; three views marginally better.",
      "Resolution (Table 5, Cat): 320→2560px width improves r 1.01→0.34° and per-axis error to <0.1mm (sub-mm at 2560px).",
      "Monocular benchmark (Table 6): ACCV2020 best at 5°/5cm 89.0 (ADD-0.1d 76.9, 3.5ms); TVCG2021 best rotation at 2°/2cm (59.0° acc., 51.4 overall); JCST2021 best ADD-0.02d 14.4.",
    ],
    ablations: [
      "Included angle sweep (Table 3): errors fall monotonically 5°→90° (tz 16.67→0.28mm); 120° slightly worse (0.37mm); cone-type off-plane pairs (C-0/C-9..12) behave like in-plane pairs.",
      "Camera count (Fig. 5): small-to-large appending reduces error; large-to-small appending can increase it — angle dominates count; 2 orthogonal cameras already suffice.",
      "Motion modes and resolution studies (Tables 4–5) confirm free camera motion is trackable given GT inter-camera pose.",
    ],
    limitations: {
      authorStated: [
        "Cameras must stay relatively fixed to each other, limiting application scenarios; freely moving cameras need markers/SLAM with extra calibration error.",
        "Object motion is kept relatively slow to guarantee precision.",
        "No proper method yet to evaluate the rotation error of the annotation itself (Section 5.3).",
      ],
      evident: [
        "Occlusion sequences are deliberately few since occlusion hurts the annotation optimizer — the hardest tracking cases are under-represented.",
        "Clean-background second view and controlled lighting aid annotation but differ from in-the-wild monocular use.",
      ],
    },
    assumptions: [
      "Inter-camera transform is pre-calibrated and fixed during capture; system-clock synchronization holds.",
      "CAD models exact; single-color paint equals textureless; 2px re-projection bound converts to ≤2mm via fx/Zc, fy/Zc in [1,2].",
    ],
    computation:
      "Annotation-time Gauss-Newton with extra iterations (no efficiency constraint); evaluated tracker runtimes 3.5–38.5ms on a i7-8565U/MX250 laptop (Table 6).",
    relations: [],
    concepts: ["3d-tracking", "benchmark-design", "motion-model", "bounding-box"],
    impact:
      "First real-scene markerless sub-mm-capable 3D tracking benchmark; its <2mm protocol and 90°-binocular finding set the annotation standard and gave deep 3D trackers real training data.",
  },
  {
    id: "T073",
    arxiv: "2203.14360",
    title: "Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking",
    shortTitle: "OC-SORT",
    year: 2022,
    authors: ["Jinkun Cao", "Jiangmiao Pang", "Xinshuo Weng", "Rawal Khirodkar", "Kris Kitani"],
    fileName: "2203.14360v3.pdf",
    task: "multi-object",
    tags: ["tracking-by-detection", "kalman-filter", "sort", "observation-centric", "occlusion-robust", "nonlinear-motion", "realtime"],
    difficulty: "advanced",
    summary:
      "OC-SORT diagnoses SORT as estimation-centric: Kalman dummy-updates without observations accumulate error quadratically in time (Eq. 5), and velocity noise explodes at small Δt (Eq. 4). It adds three observation-centric fixes on pure motion: ORU backchecks and re-updates KF along a virtual constant-velocity trajectory between the last-seen and re-associating observations; OCM adds an observation-based direction-consistency cost (Δt=3, λ=0.2) to association; OCR recovers unmatched tracks from their last observation. Result: 63.2 HOTA on MOT17, 62.1 on MOT20, 54.6 on DanceTrack at 793 FPS.",
    problem:
      "SORT trusts linear KF estimates through occlusions (dummy update, Eq. 3), so error grows ∝T² and re-associated tracks point the wrong way and get lost again; direction cues are unusable because KF velocity estimates are noise-dominated (a 1px shift matters against 1.93/0.65px average MOT17 displacement).",
    background: ["mot", "tracking-by-detection", "kalman", "motion-model", "data-association", "cost-matrix", "hungarian", "mota", "idf1", "hota"],
    previousWork: [
      {
        name: "SORT (KF + Hungarian on IoU)",
        limitation:
          "Estimation-centric dummy updates accumulate error with no observations; linear constant-velocity prior plus noisy velocity estimates fail joint occlusion + non-linear motion.",
        whyThisPaper:
          "Keeps SORT's skeleton but makes every stage observation-centric: ORU/OCM/OCR lift DanceTrack 47.9→54.6 HOTA over SORT with identical detections (Table 3).",
      },
      {
        name: "DeepSORT and appearance/ReID association",
        limitation:
          "Appearance embeddings fail when scenes are crowded, boxes are coarse, or clothing looks alike — exactly DanceTrack's regime.",
        whyThisPaper:
          "Pure-motion OC-SORT beats appearance-based FairMOT/QDTrack on DanceTrack (54.6 vs 39.7/45.7 HOTA); appearance is left as an optional plug-in (Deep OC-SORT).",
      },
      {
        name: "Transformer trackers (TransTrack, TrackFormer, MOTR, TransMOT)",
        limitation:
          "Learned trajectory+visual representations trail tracking-by-detection in both accuracy and speed on MOT17/MOT20.",
        whyThisPaper:
          "A fixed KF with three small fixes matches or beats them (MOT17 63.2 vs TransMOT 61.7; MOT20 62.1 vs 61.9) at 793 FPS association speed.",
      },
    ],
    researchGap:
      "Before this paper, no work showed a basic Kalman filter — with only observation-anchored corrections and no appearance or learned motion — could reach SOTA under prolonged occlusion and non-linear motion.",
    contribution: [
      "Theoretical diagnosis: velocity-noise amplification 2σ²/Δt (Eq. 4), T² temporal error magnification under dummy updates (Eq. 5), and estimation-centricity (σ′<σ from modern detectors) as SORT's three linked failure causes.",
      "Observation-centric Re-Update (ORU, Eqs. 6–7): on re-association after t1..t2 untracked, replay predict + re-update along virtual trajectory z̃_t = z_t1 + (t−t1)/(t2−t1)(z_t2 − z_t1).",
      "Observation-Centric Momentum (OCM, Eq. 8): association cost C = C_IoU + 0.2·C_v with direction difference Δθ = |θ_track − θ_intention| from observations Δt=3 apart, with closed-form noise analysis (Appendix A).",
      "Observation-Centric Recovery (OCR): second association attempt between unmatched tracks' last observations and unmatched detections for stopped/briefly-occluded objects.",
      "SOTA pure-motion results with shared parameters: MOT17 63.2, MOT20 62.1, DanceTrack 54.6 (55.1 +interp), KITTI-pedestrian 52.95 HOTA, all online real-time.",
    ],
    method: {
      pipeline: ["detect", "predict", "associate-with-ocm", "recover-with-ocr", "reupdate-with-oru", "interpolate"],
      architecture:
        "SORT skeleton: 7-dim KF state [u,v,s,r,u̇,v̇,ṡ], YOLOX (ByteTrack weights; PermaTrack dets on KITTI) detections, IoU + OCM cost, Hungarian matching (IoU threshold 0.3), constant-velocity virtual trajectory; linear interpolation for MOT17/MOT20 reporting; head padding (HP) for KITTI cars.",
      motionModel:
        "Linear KF with Δt=1 transitions (Eq. 2); ORU corrects KF history on re-activation; process noise tuned small on velocity (0.01) per Appendix G convention.",
      appearanceModel: "None — pure motion; no ReID embeddings.",
      association:
        "C(X̂,Z) = C_IoU + λC_v (λ=0.2); θ = arctan((v1−v2)/(u1−u2)) from observations 3 steps apart; second OCR pass; detection confidence thresholds 0.6 (0.4 MOT20).",
      detectionDependency:
        "Fully detector-bound: shares ByteTrack YOLOX weights for fair comparison; no adaptive thresholds (unlike ByteTrack) and high-score detections only, hence lower MOTA but higher HOTA/AssA.",
      trackManagement:
        "Standard SORT birth/death; re-activation triggers ORU over the untracked span; offline linear interpolation (+GPR studied and rejected) fills gaps.",
      optimization: "No learning — fixed filter and hyperparameters shared across all datasets.",
    },
    equations: [
      {
        id: "ocsort-kf",
        label: "Kalman predict–update",
        formula: "x̂_{t|t−1} = F_t x̂_{t−1|t−1}; P_{t|t−1} = F_t P_{t−1|t−1} F_t^T + Q_t; K_t = P_{t|t−1} H_t^T (H_t P_{t|t−1} H_t^T + R_t)^−1; x̂_{t|t} = x̂_{t|t−1} + K_t (z_t − H_t x̂_{t|t−1})",
        variables: [
          { symbol: "x̂ / P", meaning: "posteriori state estimate and covariance" },
          { symbol: "F_t / H_t", meaning: "linear transition and observation models" },
          { symbol: "Q_t / R_t", meaning: "process and observation noise covariances" },
          { symbol: "K_t", meaning: "Kalman gain" },
          { symbol: "z_t", meaning: "detection observation (also called measurement)" },
        ],
        intuition:
          "Guess where each box goes, then correct the guess with the detector's box, weighted by who is more trustworthy.",
        why:
          "The substrate SORT builds on — and the source of dummy-update failure when z_t is missing.",
        where: "Section 3.1, Eq. 1; every frame per track.",
        simulator: "kalman",
        paperIds: ["T073"],
      },
      {
        id: "ocsort-dummy",
        label: "Dummy update without observations and its error growth",
        formula: "x̂_{t|t} = x̂_{t|t−1}, P_{t|t} = P_{t|t−1}; u_{t+T} = u_t + T·u̇_t, noise δu_{t+T} ∼ N(0, 2T²σ_u²)",
        variables: [
          { symbol: "T", meaning: "number of consecutive occluded steps" },
          { symbol: "σ_u", meaning: "per-step position noise (≈ inter-frame displacement)" },
          { symbol: "u̇_t", meaning: "frozen velocity estimate propagated blindly" },
        ],
        intuition:
          "With no detections, SORT photocopies its own guess forward — and the photocopy error grows with the square of the blackout length.",
        why:
          "Quantifies estimation-centric failure: 10 occluded frames × 1px noise ≈ a full pedestrian-size shift (Sec. 3.2.2).",
        where: "Section 3.2.2, Eqs. 3 and 5; the gap ORU later backchecks.",
        simulator: "kalman",
        paperIds: ["T073"],
      },
      {
        id: "ocsort-velnoise",
        label: "Velocity estimation noise amplification",
        formula: "u̇ = (u_{t+Δt} − u_t)/Δt; δu̇ ∼ N(0, 2σ_u²/(Δt)²)",
        variables: [
          { symbol: "Δt", meaning: "time gap between the two states used for velocity" },
          { symbol: "δu̇", meaning: "velocity noise from position noise σ_u" },
        ],
        intuition:
          "Estimating speed from two nearby frames divides pixel noise by a tiny number — a 1px wobble looks like a sprint.",
        why:
          "Explains why SORT cannot use direction cues raw, and why OCM uses Δt=3 observations instead of adjacent KF estimates.",
        where: "Section 3.2.1, Eq. 4 (Appendix G generalizes with process noise on velocity).",
        simulator: "motion",
        paperIds: ["T073"],
      },
      {
        id: "ocsort-oru",
        label: "Observation-centric Re-Update",
        formula: "z̃_t = z_{t1} + (t−t1)/(t2−t1)·(z_{t2} − z_{t1}), t1<t<t2; x̂_{t|t} = x̂_{t|t−1} + K_t (z̃_t − H_t x̂_{t|t−1})",
        variables: [
          { symbol: "z_{t1} / z_{t2}", meaning: "last-seen observation before loss / re-associating observation" },
          { symbol: "z̃_t", meaning: "virtual constant-velocity observation replayed over the gap" },
          { symbol: "K_t update", meaning: "standard KF update applied to virtual (re-update) instead of real observations" },
        ],
        intuition:
          "When a lost target reappears, draw a straight line between where you last saw it and where it is now, and re-teach the filter along that line so its speed points the right way.",
        why:
          "Fixes the post-occlusion wrong-direction estimates (Fig. 7) that make SORT lose the track again immediately; ablates +1.4 MOT17 / +0.7 DanceTrack HOTA (Table 5).",
        where: "Section 4.1, Eqs. 6–7; triggered only on re-activation, replaying the untracked span.",
        params: "Constant-velocity hypothesis beats GPR/LR/constant-acceleration for online replay (Table 6).",
        simulator: "kalman",
        paperIds: ["T073"],
      },
      {
        id: "ocsort-ocm",
        label: "Observation-centric momentum association cost",
        formula: "C(X̂, Z) = C_IoU(X̂, Z) + λ·C_v(Z, Z), λ = 0.2; Δθ = |θ_track − θ_intention|, θ = arctan((v1−v2)/(u1−u2))",
        variables: [
          { symbol: "C_IoU", meaning: "negative pairwise IoU cost" },
          { symbol: "C_v", meaning: "direction-consistency cost over all track↔detection pairs" },
          { symbol: "θ_track / θ_intention", meaning: "motion direction of the existing track vs of track-history-to-candidate link (Fig. 4)" },
        ],
        intuition:
          "Prefer candidates that continue the target's observed heading, not just overlap its predicted box.",
        why:
          "Observation-based headings avoid KF noise and error magnification; OCM adds the DanceTrack jump 48.5→52.1 HOTA where motion is non-linear (Table 5).",
        where: "Section 4.2, Eq. 8; Hungarian association each frame.",
        params: "Δt=3 optimal (Table 7: 66.1→66.5→66.0 MOT17 for Δt=1→3→6); λ=0.2 fixed everywhere.",
        simulator: "assoc-cost",
        paperIds: ["T073"],
      },
    ],
    datasets: ["mot17", "mot20", "dancetrack", "kitti-tracking", "others"],
    metrics: ["hota", "deta", "assa", "mota", "idf1", "idsw", "fp", "fn", "fragments"],
    baselines: ["SORT", "DeepSORT", "ByteTrack", "FairMOT", "QDTrack", "TransTrack", "TrackFormer", "MOTR", "CenterTrack", "TraDes", "GTR", "MeMOT", "PermaTrack", "TransMOT"],
    results: [
      "MOT17-test (shared ByteTrack dets): OC-SORT 63.2 HOTA / 78.0 MOTA / 77.5 IDF1 / 1950 IDs vs ByteTrack 63.1/80.3/77.3/2196; AssA 63.2 vs 62.0.",
      "MOT20-test (shared dets): OC-SORT 62.1 HOTA / 75.9 IDF1 / 913 IDs — SOTA, vs ByteTrack 61.3/75.2/1223 and TransMOT 61.9/75.2/1615.",
      "DanceTrack-test: OC-SORT 54.6 HOTA / 40.2 AssA / 54.6 IDF1 (55.1/40.4/54.9 +interp) vs SORT 47.9, ByteTrack 47.3, MOTR 54.2 — best, same detections.",
      "KITTI-test (PermaTrack dets): pedestrian 52.95 HOTA / 57.81 AssA / 181 IDs — SOTA (PermaTrack 47.43/43.66/483); car 74.64 (76.54 +HP) limited by IoU matching at 10 FPS.",
      "Public detections: OC-SORT still SOTA on MOT17 (52.4 HOTA) and MOT20 (54.3 HOTA) (Tables 12–13).",
      "Speed: 793 FPS association on i9-9980XE given detections — online real-time with one shared parameter set.",
    ],
    ablations: [
      "Components (Table 5): base 64.9/47.8 → +ORU 66.3/48.5 → +OCM 66.4/52.1 → +OCR 66.5/52.1 HOTA (MOT17-val/DanceTrack-val); OCM matters where motion is non-linear.",
      "Virtual trajectory (Table 6): constant speed 66.5/52.1 beats constant acceleration 66.2/51.3, LR 64.3/49.3, GPR 63.1/49.5 (HOTA MOT17/DanceTrack).",
      "OCM Δt (Table 7): Δt=1→2→3→6 gives 66.1→66.3→66.5→66.0 MOT17 and 51.3→52.2→52.1→52.1 DanceTrack HOTA.",
      "Interpolation: linear +2.9–3.0 MOTA on val (Table 8: 74.9→77.9 MOT17, 87.3→89.8 DanceTrack); GPR interpolation or GPR smoothing rejected (App. B).",
      "MOT17-trained detector on DanceTrack (Table 11): OC-SORT 48.6 vs SORT 47.9 HOTA — association alone generalizes.",
    ],
    limitations: {
      authorStated: [
        "IoU matching struggles for fast objects / low frame-rate (KITTI cars) where consecutive boxes barely overlap — needs extra cues.",
        "Non-linear trajectory fitting (GPR/LR) with limited online points does not beat constant-velocity; fitting non-linear motion remains open (Appendix B).",
      ],
      evident: [
        "No appearance model: uniform-appearance DanceTrack is ideal, but ReID-friendly sequences leave discriminative signal unused (delegated to Deep OC-SORT).",
        "High-score-only detections trade MOTA/FN against HOTA/AssA (MOT17 FN 10.8 vs ByteTrack 8.37).",
      ],
    },
    assumptions: [
      "Linear motion holds over ~3-frame windows for OCM and over occlusion gaps for ORU's virtual trajectory.",
      "Detector observations have i.i.d. noise with σ′<σ (more accurate than propagated estimates) on every frame.",
      "Constant frame rate (Δt=1) and calibrated detection confidences for thresholds.",
    ],
    computation:
      "793 FPS on Intel i9-9980XE CPU given detections; no training; shared thresholds/weights across all datasets.",
    relations: [
      { to: "T005", type: "builds-on", note: "Keeps SORT's KF+Hungarian skeleton; fixes its dummy-update and velocity-noise failures." },
      { to: "T061", type: "uses-as-baseline", note: "Shares ByteTrack YOLOX detections for fair comparison; beats it on HOTA/AssA/IDs on MOT17/MOT20/DanceTrack." },
      { to: "T013", type: "uses-as-baseline", note: "DeepSORT beaten on DanceTrack (45.6 vs 54.6 HOTA) as the appearance-based reference." },
      { to: "T043", type: "uses-as-baseline", note: "FairMOT beaten on MOT17/MOT20/DanceTrack tables with same-detector blocks." },
      { to: "T053", type: "uses-as-baseline", note: "TransTrack beaten on MOT17 (54.1 vs 63.2) and DanceTrack (45.5 vs 54.6)." },
      { to: "T054", type: "uses-as-baseline", note: "TrackFormer in MOT17 public table, beaten (62.5 MOTA vs OC-SORT 58.2–59.4 with far fewer IDs)." },
      { to: "T059", type: "uses-as-baseline", note: "MOTR beaten on MOT17 (57.2 vs 63.2) and DanceTrack (54.2 vs 54.6)." },
      { to: "T063", type: "uses-as-baseline", note: "DanceTrack is the headline non-linear-motion benchmark (54.6 HOTA SOTA)." },
      { to: "T006", type: "uses-as-baseline", note: "MOT16-family public-detection protocol roots; MOT17/MOT20 public tables reported." },
    ],
    concepts: ["mot", "tracking-by-detection", "kalman", "motion-model", "data-association", "cost-matrix", "hungarian", "track-management", "occlusion", "hota"],
    impact:
      "Rehabilitated pure-motion tracking-by-detection: three tiny observation-centric fixes beat learned and appearance-heavy trackers on occlusion/non-linear regimes, and DanceTrack became the association benchmark because of it.",
  },
{
    id: "T074",
    arxiv: "2203.16761",
    title: "MeMOT: Multi-Object Tracking with Memory",
    shortTitle: "MeMOT",
    year: 2022,
    authors: ["Jiarui Cai", "Mingze Xu", "Wei Li", "Yuanjun Xiong", "Wei Xia", "Zhuowen Tu", "Stefano Soatto"],
    fileName: "2203.16761v1.pdf",
    task: "multi-object",
    tags: ["end-to-end-tracking", "transformer", "spatio-temporal-memory", "in-network-association", "tracking-by-detection-alternative", "occlusion-recovery"],
    difficulty: "advanced",
    summary:
      "MeMOT is an online Transformer tracker that performs detection and data association under one framework by maintaining a large spatio-temporal memory (up to 300–600 tracks × 20–22 frames) of identity embeddings. A memory encoder splits each tracklet into short-term (noise smoothing) and long-term (re-identification context) branches fused with attention, and a memory decoder outputs tracked, new, and suppressed objects in a single pass with no post-processing. It is the best in-network-association method on MOT16/17/20, reaching 72.5 MOTA / 69.0 IDF1 on MOT17 with only 1.9× training data.",
    problem:
      "Online MOT systems either separate detection from association (tracking-by-detection, which recovers poorly after occlusion) or unify them but simplify association to adjacent-frame propagation (TrackFormer/MOTR-style query passing), so objects lost for many frames cannot be re-linked. Optimal state estimation is intractable for non-linear, non-Gaussian events like occlusion with a finite-dimensional state.",
    background: ["mot", "tracking-by-detection", "data-association", "attention", "hungarian", "memory-network"],
    previousWork: [
      {
        name: "Tracking-by-detection (Tracktor++, CenterTrack, JDE/FairMOT)",
        limitation:
          "Ready-made detectors plus hand-crafted linking recover poorly after occlusion and cannot reconnect long-term missing objects; joint detection+ReID variants trade runtime for fragile embeddings.",
        whyThisPaper:
          "MeMOT replaces the two-stage link with a learned memory decoder that re-links after long gaps (MOT16 IDsw only 845 vs 3303 for FairMOT).",
      },
      {
        name: "Transformer query-propagation trackers (TrackFormer, MOTR)",
        limitation:
          "Track queries are passed only between adjacent frames, so long-range temporal context is never aggregated and association degrades under crossing/occlusion.",
        whyThisPaper:
          "A FIFO spatio-temporal memory with short/long-term cross-attention aggregation lifts MOT17 AssA to 55.2 vs 47.9 (TransTrack) with the same in-network-association setting.",
      },
      {
        name: "Single-object memory networks (MemTrack, STMTrack)",
        limitation:
          "They store temporal templates for one target and never solve inter-object association, so the design does not transfer to MOT.",
        whyThisPaper:
          "MeMOT stores per-object tracklet states for all N active objects with a fusion module that resolves identity competition across objects.",
      },
    ],
    researchGap:
      "Before this paper, no online MOT method combined joint detection–association with a large, adaptively aggregated spatio-temporal memory for long-span re-linking.",
    contribution: [
      "Spatiotemporal memory buffer X ∈ R^{N×T×d} (FIFO, Nmax 300/600, Tmax 22/20) storing history states of every active tracklet, updated online.",
      "Memory aggregator with three attentions: short-term cross-attention fshort (latest state queries past Ts=3 frames), long-term cross-attention flong (learnable DMAT queries past Tl=24 frames), and a self-attention fusion block.",
      "Memory decoder solving detection and association simultaneously via objectness × uniqueness scores, emitting tracked / new / suppressed entries with identities and no post-processing.",
      "Clip-centric end-to-end training (tracking loss + auxiliary Deformable-DETR detection loss, MOTR-style normalization) reaching IAS state of the art on MOT16/17/20.",
    ],
    method: {
      pipeline: ["encode-frame", "generate-proposals", "encode-memory", "decode-jointly", "threshold", "update-memory"],
      architecture:
        "ResNet-50 + Deformable DETR encoder–decoder (COCO-pretrained, Transformer layers reduced to 4) as hypothesis generator ΘH producing Npro proposal embeddings; memory encoder ΘE (fshort/flong/ffusion) producing track embeddings Qttck and updated DMAT; stacked Transformer decoder ΘD querying image features with [Qtpro, Qttck] and predicting box offsets w.r.t. learned reference points plus objectness and uniqueness.",
      appearanceModel:
        "Identity embeddings accumulated in the memory buffer; short-term branch smooths recent states, long-term branch retrieves full-body/pre-occlusion features (e.g. frame 124 and 121 re-link occluded object 55 at frame 129).",
      association:
        "Learned in-network solver: unified confidence s=o·u thresholded at εtck=0.6 (tracked) and εpro=0.7 (new births); proposal-to-GT assignment by bipartite matching during training.",
      detectionDependency:
        "Self-contained: Deformable-DETR-based proposal head trained with auxiliary detection loss (discarded after training); no external detector at inference.",
      trackManagement:
        "Confidence-gated update of trajectories and memory; unmatched-above-threshold proposals initialize new tracks; graduated oldest states leave the FIFO memory.",
      loss:
        "Ltck = λcls(Lobj+Luni) focal + λL1 Lbbox + λiou Liou (λ 2/5/2); Ldet same form on auxiliary head; Lclip = λtck·normalized tracking loss + λdet·detection loss with λtck=λdet=1.",
      optimization:
        "AdamW, lr 2e-4 (÷10 at epoch 100), 200 epochs, 1 clip/GPU × 8 A100; clip length from 2 growing by 4 every 20 epochs, frame interval sampled 1–10; input shorter side 800.",
    },
    equations: [
      {
        id: "memot-confidence",
        label: "Unified objectness–uniqueness confidence",
        formula: "sᵏ_t = oᵏ_t · uᵏ_t, with u = 1 for track queries",
        variables: [
          { symbol: "oᵏ_t", meaning: "objectness score: entry depicts a visible object (0–1)" },
          { symbol: "uᵏ_t", meaning: "uniqueness score: entry is unique and should be output (0–1)" },
          { symbol: "sᵏ_t", meaning: "unified confidence thresholded at ε to keep the entry" },
        ],
        intuition:
          "An output must both be a real object and not duplicate an already-tracked one; multiplying the two answers gives one keep/kill score.",
        why:
          "A single classification head is ambiguous (low score = not an object, or not a new object); splitting removes the ambiguity and cuts false positives and ID switches (Table 8).",
        where: "Section 3.5, Eq. 1; applied to every proposal and track query at inference.",
        params: "Thresholds εtck=0.6 for tracked objects, εpro=0.7 for new births.",
        paperIds: ["T074"],
      },
      {
        id: "memot-trackloss",
        label: "Tracking loss on objectness, uniqueness, boxes",
        formula: "L_tck = λcls·(L_obj + L_uni) + λL1·L_bbox + λiou·L_iou",
        variables: [
          { symbol: "L_obj / L_uni", meaning: "focal losses on objectness and uniqueness scores" },
          { symbol: "L_bbox / L_iou", meaning: "L1 and generalized-IoU box regression losses" },
          { symbol: "λcls / λL1 / λiou", meaning: "loss weights 2 / 5 / 2" },
        ],
        intuition:
          "Supervise what the object is, whether it is new or already tracked, and exactly where its box is — all at once.",
        why:
          "Joint supervision is what lets one decoder learn detection and association together instead of two separate stages.",
        where: "Section 3.6, Eq. 2; computed per query after GT assignment each frame.",
        paperIds: ["T074"],
      },
      {
        id: "memot-detloss",
        label: "Auxiliary detection loss on proposals",
        formula: "L_det = λcls·L_obj + λL1·L_bbox + λiou·L_iou",
        variables: [
          { symbol: "L_obj", meaning: "object classification loss on proposal embeddings" },
          { symbol: "L_bbox / L_iou", meaning: "box regression losses as above" },
        ],
        intuition:
          "An extra detection-only exam for the proposal branch so localization stays sharp while the main head learns association.",
        why:
          "Strengthens the hypothesis generator; the auxiliary decoder is discarded after training, costing nothing at inference.",
        where: "Section 3.6, Eq. 3; attached to proposal embeddings during training only.",
        paperIds: ["T074"],
      },
      {
        id: "memot-cliploss",
        label: "Clip-level multi-frame loss",
        formula: "L_clip = λtck·Σ_t (1/N_t)·Σ_i L_tck^(i,t) + λdet·Σ_t (1/N_t)·Σ_j L_det^(j,t), λtck = λdet = 1",
        variables: [
          { symbol: "N_t", meaning: "number of visible objects in frame t (normalizer)" },
          { symbol: "L_tck^(i,t) / L_det^(j,t)", meaning: "per-query tracking / detection losses" },
          { symbol: "T", meaning: "clip length, grown from 2 during training" },
        ],
        intuition:
          "Average the per-frame exam scores over a whole clip so the memory learns to carry identity across time, not just within a frame.",
        why:
          "Clip training with growing length and random frame intervals teaches long-span linking (following the MOTR protocol).",
        where: "Section 3.6, Eq. 4; overall training objective.",
        params: "Clip length starts at 2, +4 every 20 epochs; sampling interval 1–10.",
        paperIds: ["T074"],
      },
    ],
    datasets: ["mot16", "mot17", "mot20"],
    metrics: ["mota", "idf1", "hota", "deta", "assa", "idsw", "mt-ml", "fp", "fn"],
    baselines: ["FairMOT", "TransTrack", "TransCenter", "TrackFormer", "MOTR", "CorrTracker", "TubeTK", "CTracker", "TraDeS", "PermaTrack", "GSDT"],
    results: [
      "MOT16: IDF1 69.7, MOTA 72.6, HOTA 57.4, AssA 55.7, IDsw 845, MT 44.9%, ML 16.6%, FP 14595, FN 34595 — best IAS on every headline metric.",
      "MOT17: IDF1 69.0, MOTA 72.5, HOTA 56.9, AssA 55.2, IDsw 2724, MT 43.8%, ML 18.0%, FP 37221, FN 115248 — beats all Transformer IAS rivals on association.",
      "MOT20 (crowded): IDF1 66.1, MOTA 63.7, HOTA 54.1, AssA 55.0, IDsw 1938 — comparable to FairMOT with ~5–8× less training data (1.0× vs 8.2×).",
      "Data efficiency: MOT16/17 trained on CrowdHuman-val + MOT17-train only (1.9×); MOT20 with no extra data.",
    ],
    ablations: [
      "Short-term length (Table 2, Tl=24): Ts=2 gives 72.52/65.62 IDF1/MOTA, Ts=3 gives 73.15/68.08, Ts=4–5 flat — Ts=3 chosen for accuracy-efficiency.",
      "Long-term length (Table 3): Tl=3 → 71.27 IDF1/136 IDsw; Tl=10 → 71.66/117; Tl=20 → 72.83/96; Tl=24 → 73.15/93 — association improves monotonically.",
      "Query design (Table 4): latest-observation short query + learnable long query best (73.15 IDF1); learnable tokens for both collapses to 41.09 IDF1 / 59.80 MOTA.",
      "Aggregation (Tables 5–6): mean/max pooling catastrophic (25.04/30.72 at T=3 avg); single cross-attention without short/long split loses 0.3–0.5 MOTA and adds ID switches.",
      "DMAT update (Table 7): dynamic update 73.15/68.08 vs frozen 61.03/43.42 IDF1/MOTA; dual confidence (Table 8) 73.15/68.08 vs single-head 69.09/63.51.",
    ],
    limitations: {
      authorStated: [
        "Supervised training requires video datasets with tracking annotations, which are limited in size and diversity because video labeling is expensive.",
        "The spatio-temporal memory increases GPU-memory cost in training, limiting the temporal length (Tmax 22/20) and calling for efficiency improvements.",
      ],
      evident: [
        "FIFO graduation plus fixed thresholds (εpro 0.7, εtck 0.6) govern birth/death with no learned track-management policy shown.",
        "No explicit motion model: fast erratic motion outside the memory window has no Kalman-style fallback.",
      ],
    },
    assumptions: [
      "Causal online processing: only past and current frames inform each decision.",
      "Memory capacities (300/600 tracks) cover the typical object count per video.",
    ],
    computation:
      "8× Tesla A100 training, 200 epochs; ResNet-50 + 4-layer Transformers; input shorter side 800; trimmed 2-layer models for ablations.",
    relations: [
      { to: "T059", type: "builds-on", note: "Adopts MOTR's clip loss normalization and joint detection–association formulation, adding memory aggregation." },
      { to: "T054", type: "uses-as-baseline", note: "TrackFormer beaten on MOT17 (65.0→72.5 MOTA); its adjacent-frame propagation is the cited limitation." },
      { to: "T053", type: "uses-as-baseline", note: "TransTrack beaten on MOT17 IDF1 (63.5 vs 69.0) with 34% more ID switches." },
      { to: "T043", type: "uses-as-baseline", note: "FairMOT matched on IDF1/HOTA with ~5× less training data and far fewer ID switches (845 vs 3303 on MOT16)." },
      { to: "T006", type: "uses-as-baseline", note: "Evaluated under the MOT16/17/20 challenge protocols with CLEAR and HOTA metrics." },
      { to: "T013", type: "conceptual-successor", note: "Replaces DeepSORT-style separate association with a learned in-network solver over memory." },
    ],
    concepts: ["mot", "online-vs-offline", "end-to-end-mot", "attention", "memory-network", "data-association", "occlusion", "mota", "idf1", "hota"],
    impact:
      "Showed that explicit long-range memory with short/long-term attention aggregation beats adjacent-frame query passing, making learned in-network association competitive with heavily supervised tracking-by-detection pipelines.",
  },
  {
    id: "T075",
    arxiv: "2206.14651",
    title: "BoT-SORT: Robust Associations Multi-Pedestrian Tracking",
    shortTitle: "BoT-SORT",
    year: 2022,
    authors: ["Nir Aharon", "Roy Orfaig", "Ben-Zion Bobrovsky"],
    fileName: "2206.14651v2.pdf",
    task: "multi-object",
    tags: ["tracking-by-detection", "kalman-filter", "camera-motion-compensation", "reid", "iou-reid-fusion", "bytetrack", "baseline"],
    difficulty: "intermediate",
    summary:
      "BoT-SORT is a SORT-like tracker that fixes three weaknesses of its predecessors — a width/height-direct Kalman state vector, ECC image-registration camera-motion compensation, and a min-fusion of IoU with ReID cosine distance — integrated into ByteTrack's two-stage association. With a FastReID SBS-50 appearance branch it becomes BoT-SORT-ReID, the first tracker over 80 IDF1 on MOT17 (80.5 MOTA / 80.2 IDF1 / 65.0 HOTA) and rank-1 on MOT20 (77.8 / 77.5 / 63.3).",
    problem:
      "SORT-like trackers used an aspect-ratio Kalman state that mis-estimated box width, ignored camera motion (breaking IoU association under rotation/vibration), and fused motion with appearance through a weighted sum that forced a MOTA–IDF1 trade-off; ByteTrack used motion alone and left appearance gains on the table.",
    background: ["mot", "tracking-by-detection", "kalman", "data-association", "hungarian", "reid", "iou", "cost-matrix"],
    previousWork: [
      {
        name: "SORT / DeepSORT (aspect-ratio Kalman state)",
        limitation:
          "Estimating aspect ratio instead of width yields inaccurate box widths and sub-optimal output boxes versus detector boxes.",
        whyThisPaper:
          "Direct (xc, yc, w, h, velocities) state with size-scaled Q/R fits widths accurately (Fig. 3) and raises HOTA in ablations.",
      },
      {
        name: "ByteTrack (motion-only two-stage association)",
        limitation:
          "IoU-only matching discards appearance, failing under large displacement and long occlusion despite rescuing low-score boxes.",
        whyThisPaper:
          "ByteTrack is kept as the backbone and extended with CMC, a better KF, track-shape prediction output, and an IoU–ReID min-fusion branch.",
      },
      {
        name: "Appearance trackers (DeepSORT/JDE weighted-sum fusion C = λAa + (1−λ)Am, λ=0.98)",
        limitation:
          "A fixed weighted sum forces a detect-vs-identify trade-off and lets one bad modality corrupt the cost.",
        whyThisPaper:
          "Gated min-fusion (reject far/low-similarity pairs, take the min) beats IoU-only, cosine-only, JDE-style and masked-cosine variants on all three metrics.",
      },
    ],
    researchGap:
      "Before this paper, no SORT-like tracker combined a corrected Kalman state, camera-motion compensation, and a robust IoU–appearance fusion inside the ByteTrack association framework.",
    contribution: [
      "Width/height-direct 8-D Kalman state with size-scaled time-dependent Qk/Rk (σp=0.05, σv=0.00625, σm=0.05) and full constant-velocity F/H matrices.",
      "ECC-based global motion compensation: affine A from keypoints + sparse optical flow + RANSAC, correcting predicted state and covariance (Eqs. 5–9).",
      "IoU–ReID min-fusion with θiou=0.5 proximity gate and θemb=0.25 appearance gate (histogram-grounded), EMA appearance update (α=0.9) on high-confidence detections only.",
      "BoT-SORT / BoT-SORT-ReID rank-1 on MOT17 and MOT20 across MOTA, IDF1 and HOTA; plus the frame-wise cMOTA diagnostic metric.",
    ],
    method: {
      pipeline: ["detect", "extract-reid", "estimate-camera-motion", "predict-kalman", "compensate", "fuse-costs", "associate-high", "associate-low", "manage-tracks", "interpolate"],
      architecture:
        "YOLOX-X detections (τ=0.6) → FastReID SBS-50/ResNeSt50 embeddings for high-score boxes → KF prediction → ECC affine compensation → min(IoU, gated-cosine) cost via Hungarian (reject <0.2) → second IoU-only low-score association → 30-frame lost buffer → offline linear interpolation (gap ≤20).",
      motionModel:
        "Discrete constant-velocity Kalman filter on (xc,yc,w,h) with ECC rigid-motion correction of both state and covariance; full velocity-term correction in fast-camera scenes; track-shape prediction step used as output.",
      appearanceModel:
        "EMA tracklet state eᵏ_i = 0.9·eᵏ⁻¹_i + 0.1·fᵏ_i updated only on high-confidence matches; cosine distance to new detections, gated by θemb=0.25.",
      association:
        "First stage: C = min(dIoU, d̂cos) after gating; Hungarian assignment. Second stage: IoU-only matching of remaining tracks to low-score detections (ByteTrack).",
      reid:
        "FastReID SBS-50 trained 60 epochs on first halves of MOT17/MOT20; ReID extraction restricted to high-confidence detections to bound cost.",
      detectionDependency:
        "Fully detector-bound: public ByteTrack YOLOX-X weights (CrowdHuman + MOT17-half for ablations); private-detection protocol on test sets.",
      trackManagement:
        "New-track score threshold η, 30-frame rebirth buffer, linear interpolation post-processing; cMOTA(t) cumulative metric for failure diagnosis.",
      optimization:
        "No tracker training: hand-set gates and covariances; ReID backbone trained with FastReID default 60-epoch schedule.",
    },
    equations: [
      {
        id: "botsort-state",
        label: "Corrected Kalman state and measurement vectors",
        formula: "x_k = [xc, yc, w, h, vxc, vyc, vw, vh]ᵀ; z_k = [zxc, zyc, zw, zh]ᵀ",
        variables: [
          { symbol: "xc / yc", meaning: "box center coordinates in the image plane" },
          { symbol: "w / h", meaning: "box width and height estimated directly" },
          { symbol: "vxc..vh", meaning: "constant-velocity terms for center, width, height" },
          { symbol: "z_k", meaning: "detector measurement of center, width, height" },
        ],
        intuition:
          "Track the box's center plus its actual width and height and their speeds, instead of a scale/aspect-ratio proxy.",
        why:
          "Aspect-ratio estimation deforms widths (Fig. 3, legs clipped); direct estimation fits detector boxes and improves HOTA.",
        where: "Section 3.1, Eqs. 1–2; state definition used in every predict/update step.",
        simulator: "kalman",
        paperIds: ["T075"],
      },
      {
        id: "botsort-noise",
        label: "Size-scaled process and measurement noise",
        formula: "Q_k = diag((σp·ŵ)², (σp·ĥ)², (σp·ŵ)², (σp·ĥ)², (σv·ŵ)², (σv·ĥ)², (σv·ŵ)², (σv·ĥ)²); R_k = diag((σm·ŵ)², (σm·ĥ)², (σm·ŵ)², (σm·ĥ)²)",
        variables: [
          { symbol: "ŵ / ĥ", meaning: "posterior (Q) or prior (R) width/height estimate" },
          { symbol: "σp / σv / σm", meaning: "position/velocity/measurement factors 0.05/0.00625/0.05 at 30 FPS" },
        ],
        intuition:
          "Let big boxes tolerate big pixel noise and small boxes demand small noise — uncertainty in proportion to size.",
        why:
          "Time-dependent, size-relative covariances (DeepSORT-style, adapted to the new state) keep the gain calibrated across scales.",
        where: "Section 3.1, Eqs. 3–4; rebuilt each step; long-loss deformation guarded by ByteTrack-style logic.",
        simulator: "kalman",
        paperIds: ["T075"],
      },
      {
        id: "botsort-cmc",
        label: "Camera-motion compensation of prediction",
        formula: "A = [M|T] (2×3 affine); x̂' = M̃·x̂ + T̃; P' = M̃·P·M̃ᵀ; then standard K, x, P update",
        variables: [
          { symbol: "A / M / T", meaning: "ECC+RANSAC affine from keypoints + sparse optical flow; its rotation-scale and translation parts" },
          { symbol: "M̃ / T̃", meaning: "8×8 / 8×1 expansions applying M to center, size and velocity blocks" },
          { symbol: "x̂ / P", meaning: "KF predicted state and covariance before compensation; primed versions after" },
        ],
        intuition:
          "Before matching, shift every predicted box by the camera's own motion so predictions land where the detector sees them.",
        why:
          "Dynamic cameras (MOT17-13 turn, MOT20 vibration) destroy IoU overlap; compensation restores association (Figs. 4, 6, +1.2 HOTA).",
        where: "Section 3.2, Eqs. 5–9; between KF prediction and update each frame.",
        params: "Sparse registration ignores dynamic objects via detections; slow cameras may skip the covariance correction.",
        simulator: "motion",
        paperIds: ["T075"],
      },
      {
        id: "botsort-ema",
        label: "EMA appearance-state update",
        formula: "eᵏ_i = α·eᵏ⁻¹_i + (1−α)·fᵏ_i, α = 0.9",
        variables: [
          { symbol: "eᵏ_i", meaning: "averaged appearance state of tracklet i at frame k" },
          { symbol: "fᵏ_i", meaning: "embedding of the currently matched high-confidence detection" },
          { symbol: "α", meaning: "momentum retaining history (0.9)" },
        ],
        intuition:
          "Maintain a running portrait of each person, refreshed mostly from history and slightly from the newest sighting.",
        why:
          "Averaging resists single-frame occlusion/blur corruption; restricting updates to confident detections keeps the portrait clean.",
        where: "Section 3.3, Eq. 10; on every matched high-confidence detection.",
        simulator: "reid",
        paperIds: ["T075"],
      },
      {
        id: "botsort-fusion",
        label: "Gated IoU–ReID min-fusion cost",
        formula: "d̂cos = 0.5·dcos if (dcos < 0.25 and dIoU < 0.5) else 1; C = min(dIoU, d̂cos)",
        variables: [
          { symbol: "dIoU", meaning: "IoU distance between predicted tracklet box and detection (motion cost)" },
          { symbol: "dcos", meaning: "cosine distance between EMA state and detection embedding" },
          { symbol: "θemb = 0.25 / θiou = 0.5", meaning: "appearance-positivity and proximity gates" },
          { symbol: "C", meaning: "final assignment cost solved by the Hungarian algorithm" },
        ],
        intuition:
          "Throw out pairs that are far apart or look like different people, then score survivors by whichever cue — position or looks — agrees more.",
        why:
          "Min-fusion lets each modality veto but never average-corrupt the other, beating weighted sums and either cue alone (Table 2: 78.5/82.1/69.2).",
        where: "Section 3.3, Eqs. 12–13; first (high-confidence) association stage.",
        simulator: "assoc-cost",
        paperIds: ["T075"],
      },
    ],
    datasets: ["mot17", "mot20"],
    metrics: ["mota", "idf1", "hota", "idsw", "fp", "fn", "fps"],
    baselines: ["ByteTrack", "SORT", "DeepSORT", "StrongSORT++", "OC-SORT", "FairMOT", "QDTrack (QuasiDense)", "TransTrack", "TransCenter", "MOTR", "TrackFormer", "CenterTrack", "CorrTracker"],
    results: [
      "MOT17 test: BoT-SORT 80.6 MOTA / 79.5 IDF1 / 64.6 HOTA (FP 22524, FN 85398, IDs 1257); BoT-SORT-ReID 80.5 / 80.2 / 65.0 (FP 22521, FN 86037, IDs 1212) — rank 1, first tracker above 80 IDF1.",
      "MOT20 test: BoT-SORT 77.7 / 76.3 / 62.6 (IDs 1212); BoT-SORT-ReID 77.8 / 77.5 / 63.3 (IDs 1257) — rank 1 on all three headline metrics in crowded scenes.",
      "Speed: 6.6 FPS (BoT-SORT) / 4.5 FPS (ReID) on RTX 3060 incl. YOLOX-X; GMC cost negligible vs detection and multi-threadable.",
    ],
    ablations: [
      "Component path on MOT17 val (Table 1): ByteTrack baseline 77.66/79.77/67.88 → +KF 77.67/79.89/68.12 → +CMC 78.31/81.51/69.06 → +Pred output 78.39/81.53/69.11 → +ReID 78.46/82.07/69.17.",
      "Similarity strategies (Table 2): IoU-only 78.4/81.5/69.1 beats cosine-only 73.7/70.0/62.4 and JDE-style 77.7/80.1/68.2; masked-cosine 78.3/81.0/68.7; proposed min-fusion best at 78.5/82.1/69.2.",
      "Appearance gate θemb=0.25 grounded in the positive/negative cosine histogram (Fig. 5); assignment rejection threshold 0.2, first-stage matching 0.8.",
      "cMOTA diagnosis (Fig. 6): MOT17-13 rotation collapses cMOTA frames 400–470 without CMC and stays high with it under identical detections.",
    ],
    limitations: {
      authorStated: [
        "In scenes with a high density of dynamic objects, camera-motion estimation may fail for lack of background keypoints, and wrong motion can cause unexpected behavior.",
        "Global camera-motion computation is time-consuming on large images, though negligible beside detection and parallelizable with multi-threading.",
      ],
      evident: [
        "Entirely detector-bound: gains assume the ByteTrack YOLOX-X operating point (τ=0.6); detector errors flow directly into MOTA.",
        "Offline linear interpolation (gap ≤20) and fixed gates/thresholds aid test scores but are not online decisions.",
      ],
    },
    assumptions: [
      "Adjacent-frame registration approximates rigid camera motion projected onto the image plane.",
      "High-frame-rate smoothness: object displacement between frames is small after ego-motion removal.",
    ],
    computation:
      "PyTorch on i9-11900F + RTX 3060; YOLOX-X detection dominates runtime; ReID extracted for high-confidence boxes only; 6.6/4.5 FPS reported.",
    relations: [
      { to: "T061", type: "builds-on", note: "Keeps ByteTrack's high/low two-stage association and adds KF, CMC and ReID fusion on top." },
      { to: "T005", type: "improves", note: "Fixes SORT's state vector and adds camera compensation and appearance to the SORT association loop." },
      { to: "T013", type: "improves", note: "Replaces DeepSORT's aspect-ratio state and weighted-sum fusion with direct w/h state and gated min-fusion." },
      { to: "T067", type: "uses-as-baseline", note: "StrongSORT++ beaten on MOT17 (79.6/79.5/64.4) and MOT20 (73.8/77.0/62.6)." },
      { to: "T073", type: "uses-as-baseline", note: "OC-SORT compared on both leaderboards (MOT17 78.0/77.5/63.2; MOT20 75.7/76.3/62.4)." },
      { to: "T043", type: "uses-as-baseline", note: "FairMOT beaten decisively (MOT17 73.7/72.3/59.3; MOT20 61.8/67.3/54.6)." },
      { to: "T045", type: "uses-as-baseline", note: "QuasiDense similarity-learning tracker compared (MOT17 68.7/66.3/53.9)." },
      { to: "T053", type: "uses-as-baseline", note: "TransTrack compared (MOT17 75.2/63.5/54.1)." },
      { to: "T059", type: "uses-as-baseline", note: "MOTR compared (MOT17 65.1/66.4)." },
    ],
    concepts: ["mot", "tracking-by-detection", "kalman", "motion-model", "data-association", "cost-matrix", "hungarian", "reid", "dual-threshold", "mota", "idf1", "hota"],
    impact:
      "Re-established tuned tracking-by-detection as the MOT leaderboard leader and provided a reusable bag-of-tricks (corrected KF, CMC equations, min-fusion) plus the cMOTA failure-diagnosis tool.",
  },
  {
    id: "T076",
    arxiv: "2207.07078",
    title: "Towards Grand Unification of Object Tracking",
    shortTitle: "Unicorn",
    year: 2022,
    authors: ["Bin Yan", "Yi Jiang", "Peize Sun", "Dong Wang", "Zehuan Yuan", "Ping Luo", "Huchuan Lu"],
    fileName: "2207.07078v4.pdf",
    task: "other",
    tags: ["unified-tracking", "multi-task", "single-object-tracking", "multi-object-tracking", "video-object-segmentation", "pixel-wise-correspondence", "target-prior"],
    difficulty: "advanced",
    summary:
      "Unicorn solves four tracking tasks — SOT, MOT, VOS and MOTS — with one network and one set of parameters: a shared backbone, a deformable-attention embedding module building pixel-wise correspondence, a propagated target prior that switches the same detection head between class-agnostic (SOT/VOS) and class-specific (MOT/MOTS) modes, and full-image inputs for all tasks. It matches or beats task-specific SOTAs on 8 benchmarks: LaSOT 68.5, TrackingNet 83.0, MOT17 77.2 MOTA, BDD100K 41.2 mMOTA, DAVIS16 87.4 J&F, MOTS20 65.3 sMOTSA.",
    problem:
      "Tracking is fragmented into SOT/MOT/VOS/MOTS with incompatible inputs (search region vs full image), correspondence types (target-vs-background vs instance matching) and heads, causing over-specialization, redundant parameters, and no shared learning; SOT-vs-MOT unification had barely progressed beyond adding mask branches or sharing only an appearance model.",
    background: ["sot", "mot", "detection", "siamese", "attention", "benchmark-design"],
    previousWork: [
      {
        name: "Search-region SOT trackers (SiamFC, SiamRPN++, DiMP, TransT)",
        limitation:
          "Small search windows save compute but trap the tracker after failure, cannot scale to many targets, and need heavy detector surgery to go global.",
        whyThisPaper:
          "Full-image SOT inputs plus a propagated target prior beat Siam R-CNN 64.8→68.5 on LaSOT with a simpler top-1 strategy.",
      },
      {
        name: "Tracking-by-detection MOT (FairMOT, CenterTrack, TransTrack, MOTR, TrackFormer) with independent SOT helpers",
        limitation:
          "SOT assistants share no weights with the MOT network; detection plus association are task-siloed.",
        whyThisPaper:
          "Instance correspondence is formulated as a sub-matrix of pixel correspondence, so one embedding serves both MOT matching and SOT propagation.",
      },
      {
        name: "Partial unifiers (SiamMask, TraDeS, UniTrack)",
        limitation:
          "They unify only SOT&VOS or MOT&MOTS, or share just an appearance model while heads and detectors stay separate — UniTrack lags SOTA far behind.",
        whyThisPaper:
          "One backbone, one embedding, one head with the same parameters address all four tasks at SOTA level with a single-task-parity ablation.",
      },
    ],
    researchGap:
      "Before this paper, no single network with shared architecture, parameters and learning paradigm solved SOT, MOT, VOS and MOTS together at task-specific SOTA level.",
    contribution: [
      "Target prior: propagated reference-target map fed to the head for SOT/VOS, zeros for MOT/MOTS — a switch degenerating the same head between class-agnostic and class-specific detection.",
      "Pixel-wise correspondence Cpix with instance correspondence Cinst as its sub-matrix; deformable-attention interaction plus 2× upsampling to stride-8 embeddings.",
      "Unified full-image inputs for SOT and MOT (no search region), giving re-detection after disappearance and N-target efficiency (head run once, not N times).",
      "Two-stage joint training (SOT–MOT, then VOS–MOTS mask branch) with shared parameters achieving SOTA on 8 benchmarks across 4 tasks.",
    ],
    method: {
      pipeline: ["encode-both-frames", "interact-features", "build-correspondence", "propagate-prior", "fuse-prior", "detect-unified", "associate"],
      architecture:
        "Weight-shared ConvNeXt-Large backbone (Tiny/ResNet-50 in ablations) → FPN stride-16 features → deformable-attention interaction → 2× upsampled stride-8 embeddings Eref/Ecur → Cpix propagation → broadcast-sum fusion of FPN features with target prior P → YOLOX-style unified box head (+ mask branch in stage 2).",
      appearanceModel:
        "Discriminative embeddings {Eref, Ecur} learned for both propagation (Dice on propagated mask) and association (contrastive cross-entropy on Cinst).",
      association:
        "MOT/MOTS: instance embeddings from E at centers matched across frames (embeddings + motion model at inference); SOT/VOS: top-1 box/mask pick, no cosine window or other post-processing.",
      detectionDependency:
        "Internal unified detector (YOLOX lineage) reusing pretrained detector weights; no external detector at inference.",
      trackManagement:
        "SOT/VOS: reference target map fixed from frame one; MOT/MOTS: detect-all plus embedding/motion association across frames.",
      loss:
        "Lcorr = Dice(T̃cur, Tcur) for SOT/VOS or CrossEntropy(Cinst, G) for MOT/MOTS, plus detection loss (stage 1) and mask loss with other params fixed (stage 2).",
      optimization:
        "AdamW, batch 32 on 16× A100, 2×15 epochs × 200k frame pairs; lr 2.5e-4 with 1-epoch warmup + cosine annealing, weight decay 5e-4; 800×1280 inputs (736–864 multiscale); GroupNorm replaces BatchNorm.",
    },
    equations: [
      {
        id: "unicorn-correspondence",
        label: "Pixel-wise and instance-level correspondence",
        formula: "Cpix = softmax(Ecur·Erefᵀ) ∈ R^{hw×hw}; Cinst = softmax(ecur·erefᵀ) ∈ R^{N×M}",
        variables: [
          { symbol: "Eref / Ecur", meaning: "flattened stride-8 embeddings of reference and current frames (hw×c)" },
          { symbol: "Cpix", meaning: "all-pairs pixel similarity between the two frames" },
          { symbol: "eref / ecur", meaning: "instance embeddings sampled at object centers (M reference, N current)" },
          { symbol: "Cinst", meaning: "instance matching matrix, a sub-matrix of Cpix" },
        ],
        intuition:
          "Every pixel of now shakes hands with every pixel of before; object-to-object matches are just the handshakes at object centers.",
        why:
          "One similarity object serves SOT propagation and MOT association, which is the mathematical core of the unification.",
        where: "Section 3.2, Eq. 1; computed from interacted embeddings every frame pair.",
        simulator: "siamese",
        paperIds: ["T076"],
      },
      {
        id: "unicorn-interaction",
        label: "Deformable-attention embedding interaction",
        formula: "{Eref, Ecur} = Upsample(Attention(Fref, Fcur))",
        variables: [
          { symbol: "Fref / Fcur", meaning: "stride-16 shared-backbone features of the two frames" },
          { symbol: "Attention", meaning: "deformable attention exchanging information across frames" },
          { symbol: "Upsample", meaning: "2× upsampling to stride-8 embeddings" },
        ],
        intuition:
          "Let the two frames look at each other before matching, so repetitive patterns become distinguishable, then work at higher resolution.",
        why:
          "Independent embeddings cause ambiguous matches; interaction beats full attention on accuracy with far less memory, and conv (no interaction) is worst.",
        where: "Section 3.2, Eq. 2; embedding module.",
        params: "Deformable over full attention: better scores at lower memory (Table 7).",
        paperIds: ["T076"],
      },
      {
        id: "unicorn-propagation",
        label: "Reference-target propagation",
        formula: "T̃cur(i,j) = Σ_k Cpix(i,k)·Tref(k,j)",
        variables: [
          { symbol: "Tref", meaning: "binary reference target map (hw×1), 1 on the tracked target" },
          { symbol: "Cpix", meaning: "pixel-wise correspondence transporting mass across frames" },
          { symbol: "T̃cur", meaning: "estimated current-frame target map (the target prior source)" },
        ],
        intuition:
          "Paint the old target mask through the correspondence field to predict where the target should be now.",
        why:
          "Provides dense, unsupervised-by-flow prior information that turns a plain detector into a target-conditioned tracker.",
        where: "Section 3.2, Eq. 3; supervised by Dice against the true current map.",
        paperIds: ["T076"],
      },
      {
        id: "unicorn-corrloss",
        label: "Unified correspondence loss",
        formula: "Lcorr = Dice(T̃cur, Tcur) for SOT/VOS; CrossEntropy(Cinst, G) for MOT/MOTS",
        variables: [
          { symbol: "Tcur", meaning: "ground-truth current-frame target map" },
          { symbol: "G", meaning: "one-hot instance identity matrix (G_i,j=1 iff matched)" },
          { symbol: "Dice / CrossEntropy", meaning: "mask-overlap loss vs contrastive identity loss" },
        ],
        intuition:
          "SOT asks 'did the paint land on the target', MOT asks 'did each detection pick its true identity' — one loss switch covers both.",
        why:
          "A single embedding optimized end-to-end for propagation and association is the learning-paradigm half of the unification.",
        where: "Section 3.2, Eqs. 4–5; stage-1 training objective alongside detection loss.",
        paperIds: ["T076"],
      },
      {
        id: "unicorn-prior",
        label: "Target prior switch",
        formula: "P = T̃cur^reshape (h×w×1) for SOT/VOS; P = 0 for MOT/MOTS; F' = F + P (broadcast sum)",
        variables: [
          { symbol: "P", meaning: "extra head input carrying target location belief, or zeros" },
          { symbol: "F / F'", meaning: "FPN features before/after prior fusion" },
        ],
        intuition:
          "Whisper the target's likely location into the detector's ear for SOT; stay silent for MOT so it detects everything of each class.",
        why:
          "Removing the prior collapses LaSOT 67.7→50.9 and DAVIS17 68.0→29.2 while MOT barely moves — the switch is what unifies the head with minimal surgery.",
        where: "Section 3.3, Eq. 6; fusion before the shared detection head.",
        params: "Broadcast sum beats concatenation (Table 7) and preserves pretrained detector weights.",
        paperIds: ["T076"],
      },
    ],
    datasets: ["lasot", "trackingnet", "mot17", "others"],
    metrics: ["success-auc", "precision", "norm-precision", "mota", "idf1", "hota", "mt-ml", "idsw", "fp", "fn"],
    baselines: ["SiamFC", "SiamRPN++", "DiMP", "SiamFC++", "TransT", "STARK", "Siam R-CNN", "KeepTrack", "FairMOT", "TransTrack", "TrackFormer", "MOTR", "QDTrack", "CenterTrack", "UniTrack", "SiamMask"],
    results: [
      "LaSOT: Success 68.5 / Pnorm 76.6 / P 74.1 — SOTA, +3.7 over Siam R-CNN with simpler top-1 tracking.",
      "TrackingNet: Success 83.0 / Pnorm 86.4 / P 82.2 — best on all three measures.",
      "MOT17 test: MOTA 77.2 / IDF1 75.5 / HOTA 61.7, MT 58.7%, ML 11.2%, FP 50087, FN 73349, IDs 5379 — best MOTA/IDF1 under private detection.",
      "BDD100K val: mMOTA 41.2 / mIDF1 54.0 (MOTA 66.6, IDF1 71.3, mAP 41.4) — +4.6/+3.2 over QDTrack.",
      "DAVIS16/17 (box init): 87.4 (J 86.5, F 88.2) / 69.2 (J 65.2, F 73.2) — best box-initialized, +17.6 over SiamMask on DAVIS16.",
      "MOTS20: sMOTSA 65.3 / IDF1 65.9 (+3.0 over PointTrackV2); BDD MOTS: mMOTSA 29.6 / mMOTSP 67.7 / mAP 32.1 (+2.2/+5.5 over PCAN).",
    ],
    ablations: [
      "Backbone: ConvNeXt-Tiny 67.7/39.9/68.0/29.7 vs ResNet-50 65.3/35.1/66.2/30.8 across LaSOT/BDD/DAVIS17/BDD-MOTS.",
      "Interaction: deformable 67.7/39.9 beats full attention 67.1/38.5 and conv 66.8/37.6 with less memory.",
      "Fusion: broadcast sum 67.7/68.0 vs concat 66.8/66.7 vs no prior 50.9/29.2 on LaSOT/DAVIS17 — prior is critical for SOT/VOS only.",
      "Single-task parity: unified 67.7/39.9/68.0/29.7 vs SOT-only 67.5, MOT-only 39.6, VOS-only 68.4, MOTS-only 28.1 — unification costs nothing.",
      "Real-time variant (640×1024): 67.1/37.5/66.8/26.2 at 23 FPS vs 14 FPS full resolution.",
    ],
    limitations: {
      authorStated: [
        "Authors note a remaining gap to mask-initialized SOTA VOS methods (STCN, HMMN) on DAVIS, while leading among box-initialized ones.",
      ],
      evident: [
        "MOT association still leans on embeddings plus a motion model at inference rather than a fully unified learned linker.",
        "Two-stage training (SOT–MOT then frozen VOS–MOTS) and 16×A100 joint training show unification is parameter- but not compute-cheap.",
      ],
    },
    assumptions: [
      "Reference frame (first for SOT/VOS, previous for MOT/MOTS) always provides valid targets or trajectories.",
      "One embedding family can be simultaneously optimal for dense propagation and instance discrimination.",
    ],
    computation:
      "16× NVIDIA A100, global batch 32, 2×15 epochs × 200k pairs; ConvNeXt-Large for SOTA numbers; 14 FPS full / 23 FPS real-time variant.",
    relations: [
      { to: "T009", type: "uses-as-baseline", note: "SiamFC beaten on LaSOT (33.6 vs 68.5); the Siamese lineage Unicorn generalizes to full-image." },
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ beaten on LaSOT (49.6 vs 68.5) and TrackingNet." },
      { to: "T031", type: "uses-as-baseline", note: "DiMP beaten on LaSOT (56.9 vs 68.5) and TrackingNet." },
      { to: "T035", type: "uses-as-baseline", note: "SiamFC++ beaten on LaSOT (54.4 vs 68.5) and TrackingNet (75.4 vs 83.0)." },
      { to: "T045", type: "uses-as-baseline", note: "QDTrack beaten on BDD100K tracking (36.6→41.2 mMOTA) and MOTS." },
      { to: "T053", type: "uses-as-baseline", note: "TransTrack beaten on MOT17 (75.2/63.5 vs 77.2/75.5 MOTA/IDF1)." },
      { to: "T054", type: "uses-as-baseline", note: "TrackFormer beaten on MOT17 (74.1/68.0) and MOTS20 (54.9 sMOTSA vs 65.3)." },
      { to: "T043", type: "uses-as-baseline", note: "FairMOT beaten on MOT17 (73.7/72.3 vs 77.2/75.5)." },
      { to: "T024", type: "uses-as-baseline", note: "LaSOT is the headline SOT benchmark (68.5 Success SOTA)." },
      { to: "T025", type: "uses-as-baseline", note: "GOT-10k used among joint-training SOT data sources." },
    ],
    concepts: ["sot", "mot", "attention", "end-to-end-mot", "tracking-by-detection", "learned-association", "benchmark-design"],
    impact:
      "First SOTA-level proof that SOT, MOT, VOS and MOTS can share one architecture, one parameter set and one correspondence-learning paradigm — the reference design for generalist tracking models.",
  },
  {
    id: "T077",
    arxiv: "2207.09603",
    title: "AiATrack: Attention in Attention for Transformer Visual Tracking",
    shortTitle: "AiATrack",
    year: 2022,
    authors: ["Shenyuan Gao", "Chunluan Zhou", "Chao Ma", "Xinggang Wang", "Junsong Yuan"],
    fileName: "2207.09603v2.pdf",
    task: "single-object",
    tags: ["transformer", "attention-refinement", "attention-in-attention", "siamese", "temporal-references", "target-background-embedding", "anchor-free"],
    difficulty: "advanced",
    summary:
      "AiATrack observes that each query-key correlation in standard attention is computed independently, producing noisy ambiguous weights under clutter, and inserts an inner attention that refines the correlation map by seeking consensus among all correlation vectors. Built into a streamlined Transformer (3-layer encoder, 1-layer two-branch long/short-term decoder) with cached feature reuse and learnable target-background embeddings plus an IoU head for reference update, it sets SOTAs of 69.0 LaSOT AUC, 82.7 TrackingNet AUC and 69.6 GOT-10k AO at 38 FPS.",
    problem:
      "Transformer trackers inherit conventional attention where every query-key dot product is independent, so imperfect features or background distractors create erroneous correlations that corrupt both self-attention aggregation and cross-attention propagation; meanwhile long/short-term reference handling either re-encodes frames wastefully or needs costly online optimization.",
    background: ["sot", "bounding-box", "siamese", "attention", "appearance-features", "success-plot"],
    previousWork: [
      {
        name: "Siamese trackers (SiamFC, SiamRPN++)",
        limitation:
          "One-shot matching with fixed correlation lacks global context and temporal adaptation, plateauing far below Transformer trackers (49.6 LaSOT).",
        whyThisPaper:
          "Full Transformer aggregation + propagation with refined correlations reaches 69.0 LaSOT AUC.",
      },
      {
        name: "Early Transformer trackers (TransT, TrDiMP/TrSiam, STARK)",
        limitation:
          "They use vanilla attention, so noisy correlations persist; reference updates re-crop and re-encode frames, wasting compute.",
        whyThisPaper:
          "AiA consensus refinement adds +1.7 LaSOT on the same backbone; cached feature reuse with IoU-gated update keeps 38 FPS.",
      },
      {
        name: "Online-update trackers (DiMP/PrDiMP, KYS, STMTrack-style ensembles)",
        limitation:
          "Discriminative optimization or generative re-encoding during inference needs sparse-update tricks to stay real-time.",
        whyThisPaper:
          "Short-term ensemble (default 3) sampled from a memory cache with target-background embeddings adapts without any re-encoding or optimization.",
      },
    ],
    researchGap:
      "Before this paper, no module refined attention correlations themselves inside both self- and cross-attention for tracking, nor a Transformer tracker combining such refinement with encoding-free temporal reference reuse.",
    contribution: [
      "Attention-in-attention (AiA) module: inner attention over correlation-vector columns producing a residual correlation map, shared across heads, with reduced dim D=64, LayerNorm, 2-D sinusoidal positions and identical connection.",
      "Plug-in gains in both encoder self-attention (feature aggregation) and decoder cross-attention (reference propagation), validated separately and jointly.",
      "Streamlined AiATrack: two-branch long/short-term cross-attention decoder (no decoder self-attention), learnable target/background embeddings attached to reused reference features, corner-distribution + IoU prediction heads.",
      "Efficient update loop: IoU-gated caching of encoded features + embedding maps, uniform short-term sampling with latest-always-kept, SOTA on 6 benchmarks plus VOT2020 at 38 FPS.",
    ],
    method: {
      pipeline: ["crop", "encode", "self-attend-aia", "cross-attend-aia", "predict-corners", "predict-iou", "cache-update"],
      architecture:
        "ResNet-50 (ImageNet-1k, stride 16, 320×320 crops of 25× target area) → 3-layer Transformer encoder with AiA self-attention (4 heads, C=256, FFN 1024) applied separately to search and reference features → 1-layer two-branch cross-attention decoder (long-term initial frame + short-term ensemble) with AiA → 5-layer Conv-BN-ReLU corner heads + 3-layer Conv + PrPool(3×3) + 2-FC IoU head.",
      appearanceModel:
        "Reference values combine reused features with target-background embedding maps E(p) ∈ {Etgt, Ebg} assigned by box membership — context preserved, target distinguished.",
      association:
        "Not applicable (single object); temporal linkage via long/short-term cross-attention propagation and IoU-gated cache refresh.",
      detectionDependency: "None — class-agnostic box tracker with no external detector.",
      trackManagement:
        "Init caches long+short references; per frame, if predicted IoU exceeds threshold, cache current encoded features + embedding map, uniformly sample short-term ensemble (latest always kept, oldest popped at capacity).",
      loss:
        "GIoU + L1 box loss combined with MSE IoU loss: L = λgiou·Lgiou + λl1·L1 + λmse·(i−î)²; IoU head trained on boxes sampled around GT.",
      optimization:
        "AdamW on LaSOT + TrackingNet + GOT-10k + COCO-synthetic clips; backbone lr 1e-5, rest 1e-4 with ×0.1 decay; first conv + stage-1 frozen; random affine jitter on short-term/search frames; one sampled short-term frame in training.",
    },
    equations: [
      {
        id: "aiatrack-convenattn",
        label: "Conventional attention with independent correlations",
        formula: "ConvenAttn(Q,K,V) = (softmax(Q̄·K̄ᵀ / √C)·V̄)·Wo; M = Q̄·K̄ᵀ / √C ∈ R^{HW×HW}",
        variables: [
          { symbol: "Q̄ / K̄ / V̄", meaning: "linearly projected queries, keys, values (QWq, KWk, VWv)" },
          { symbol: "M", meaning: "raw correlation map with each query-key pair scored independently" },
          { symbol: "Wo", meaning: "output projection" },
          { symbol: "C", meaning: "channel width (256)" },
        ],
        intuition:
          "Each query picks its favorite keys by dot product and averages their values — but every pick is made in isolation.",
        why:
          "Naming the flaw precisely (independent M entries → noise under clutter, Fig. 4) motivates refining M before softmax.",
        where: "Section 3.1, Eq. 1; baseline block replaced by AiA everywhere.",
        paperIds: ["T077"],
      },
      {
        id: "aiatrack-inner",
        label: "Inner attention consensus over correlations",
        formula: "InnerAttn(M) = (softmax(Q̄'·K̄'ᵀ / √D)·V̄')·(1 + Wo')",
        variables: [
          { symbol: "Q' / K' / V'", meaning: "columns of M treated as correlation-vector queries, keys, values" },
          { symbol: "Q̄' / K̄'", meaning: "dim-reduced (HW×64), normalized, position-encoded, re-projected queries/keys" },
          { symbol: "V̄'", meaning: "LayerNorm-normalized correlation vectors" },
          { symbol: "D / Wo'", meaning: "inner dim 64 (≪ HW) and residual adjustment with identical connection" },
        ],
        intuition:
          "Let every correlation vector poll all other vectors: widely supported matches get boosted, lonely spikes get voted down.",
        why:
          "Neighboring keys of a true match correlate jointly; global dynamic weighting beats fixed local smoothing (conv bottleneck).",
        where: "Section 3.1, Eq. 2; Fig. 2b; parameters shared across heads.",
        params: "Positional encoding required (variant h drops without it); D=64 bounds cost.",
        paperIds: ["T077"],
      },
      {
        id: "aiatrack-attninattn",
        label: "Attention-in-attention output",
        formula: "AttnInAttn(Q,K,V) = (softmax(M + InnerAttn(M))·V̄)·Wo",
        variables: [
          { symbol: "M + InnerAttn(M)", meaning: "raw correlations plus learned consensus residual" },
          { symbol: "V̄ / Wo", meaning: "projected values and output projection as in conventional attention" },
        ],
        intuition:
          "Correct the vote counts before voting: add the consensus adjustment to the raw scores, then average values as usual.",
        why:
          "Drop-in replacement improving both self-attention aggregation and cross-attention propagation (+1.7 LaSOT jointly, Table 3).",
        where: "Section 3.1, Eq. 3; used in all encoder and decoder attention blocks.",
        paperIds: ["T077"],
      },
      {
        id: "aiatrack-embed",
        label: "Target-background embedding assignment",
        formula: "E(p) = Etgt if p in target region else Ebg",
        variables: [
          { symbol: "Etgt / Ebg", meaning: "learnable C-dim target and background embeddings" },
          { symbol: "E(p)", meaning: "embedding map entry at grid position p (H×W)" },
        ],
        intuition:
          "Stamp every reference pixel as 'target' or 'background' with a learned ink, keeping full context while marking identity.",
        why:
          "Masking out background loses context (variant b collapses); stamping keeps context and adds discrimination (+1.2 LaSOT over no-embedding).",
        where: "Section 3.2, Eq. 4; maps attached to reference features as cross-attention values.",
        paperIds: ["T077"],
      },
      {
        id: "aiatrack-boxloss",
        label: "Corner-expectation box and joint training loss",
        formula: "x̂tl = Σ x·Ptl(x,y) (likewise y, br); L = λgiou·Lgiou + λl1·‖b−b̂‖₁ + λmse·(i−î)²",
        variables: [
          { symbol: "Ptl / Pbr", meaning: "predicted top-left / bottom-right corner probability maps" },
          { symbol: "x̂tl..ŷbr", meaning: "expected corner coordinates (anchor-free, no post-processing)" },
          { symbol: "Lgiou / L1 / MSE", meaning: "box overlap, coordinate, and IoU-prediction losses" },
        ],
        intuition:
          "Predict a heat map per corner and take its center of mass as the corner; simultaneously learn to judge your own box quality.",
        why:
          "End-to-end anchor-free localization plus a self-assessment score that drives the reference-update gate.",
        where: "Supplementary Eqs. 6–7; target and IoU heads trained jointly.",
        paperIds: ["T077"],
      },
    ],
    datasets: ["lasot", "trackingnet", "got10k", "otb", "uav123", "others"],
    metrics: ["success-auc", "precision", "norm-precision", "eao"],
    baselines: ["SiamRPN++", "DiMP", "PrDiMP", "TransT", "TrDiMP", "TrSiam", "STARK-ST50", "KeepTrack", "DTT", "KYS", "Ocean", "SiamAttn"],
    results: [
      "LaSOT: AUC 69.0 / PNorm 79.4 / P 73.8 — +1.9 AUC over KeepTrack at 2× its speed; best on all 14 attribute splits.",
      "TrackingNet: AUC 82.7 / PNorm 87.8 / P 80.4 — best published; GOT-10k (one-shot): AO 69.6 / SR0.75 63.2 / SR0.5 80.0.",
      "Small benchmarks AUC: NfS30 67.9 (best), UAV123 70.6 (best), OTB100 69.6 (comparable, saturated).",
      "VOT2020 (suppl., anchor protocol + Alpha-Refine masks): EAO 0.530 / Accuracy 0.764 / Robustness 0.827 — beats STARK 0.505.",
      "Speed 38 FPS on RTX 2080 Ti (vs 44 w/o AiA, 42 STARK, 18 KeepTrack).",
    ],
    ablations: [
      "Embeddings (Table 3, LaSOT/LaSOTExt AUC): none 65.8/44.5 → mask-background 64.3/42.8 (context matters) → target-background 67.0/44.7.",
      "Branches: w/o short-term 66.5/44.5; single merged branch 63.8/42.7 — unreliable short-term cues must not disturb the long-term branch.",
      "AiA placement: self-only 68.6/46.2, cross-only 67.5/46.2, both 68.7/46.8 vs 67.0/44.7 base; removing inner positions (h) costs 0.7 LaSOT.",
      "Overhead control (Table 4): cascaded attention 67.1/44.6 (no gain), conv bottleneck 67.9/46.0 (helps locally), AiA 68.7/46.8 — gain is consensus, not parameters.",
      "Ensemble size (Table 5): 1→66.8, 2→68.1, 3→68.7, ≥4 flat; suppl. AiAv3 structure reaches 69.2/48.4 but adds a value transform.",
    ],
    limitations: {
      authorStated: [
        "Authors note how to introduce long- and short-term references remains an open problem; their IoU-gated ensemble is one practical answer.",
      ],
      evident: [
        "AiA costs 6 FPS (44→38) and adds inner-attention hyperparameters (D, positions, sharing) with no failure-mode analysis shown.",
        "Update gate inherits IoU-head calibration error: a confident-but-wrong box pollutes the cache with no recovery mechanism described.",
      ],
    },
    assumptions: [
      "Reliable first-frame box; short-term references stay valid between IoU-gated refreshes.",
      "High correlation of a key implies high correlations of its neighbors (the consensus prior).",
    ],
    computation:
      "RTX 2080 Ti inference at 38 FPS; ResNet-50 backbone; training on LaSOT/TrackingNet/GOT-10k/COCO clips with AdamW.",
    relations: [
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ beaten on all large benchmarks (LaSOT 49.6 vs 69.0; GOT-10k AO 51.7 vs 69.6)." },
      { to: "T031", type: "uses-as-baseline", note: "DiMP/PrDiMP beaten (LaSOT 56.9/59.8 vs 69.0); their online optimization is the efficiency contrast." },
      { to: "T009", type: "conceptual-successor", note: "Continues the Siamese matching lineage of SiamFC with Transformer aggregation and propagation." },
      { to: "T024", type: "uses-as-baseline", note: "LaSOT headline SOTA plus LaSOTExt ablations and 14-attribute analysis." },
      { to: "T025", type: "uses-as-baseline", note: "GOT-10k one-shot protocol shows unseen-class generalization (AO 69.6)." },
    ],
    concepts: ["sot", "attention", "siamese", "appearance-features", "bounding-box", "success-plot"],
    impact:
      "Introduced correlation-consensus refinement as a portable attention upgrade for trackers and showed encoding-free temporal caching can match heavy update schemes at real-time speed.",
  },
  {
    id: "T078",
    arxiv: "2209.12010",
    title: "Spiking SiamFC++: Deep Spiking Neural Network for Object Tracking",
    shortTitle: "Spiking SiamFC++",
    year: 2022,
    authors: ["Shuiying Xiang", "Tao Zhang", "Shuqing Jiang", "Yanan Han", "Yahui Zhang", "Chenyang Du", "Xingxing Guo", "Licun Yu", "Yuechun Shi", "Yue Hao"],
    fileName: "2209.12010v1.pdf",
    task: "single-object",
    tags: ["siamese", "spiking-neural-network", "neuromorphic", "surrogate-gradient", "anchor-free", "energy-efficient"],
    difficulty: "advanced",
    summary:
      "Spiking SiamFC++ ports the SiamFC++ tracker to deep spiking neural networks: a time-unrolled Spiking AlexNet (IF neurons, T=6) encodes template and search into spike trains, firing rates collapse the time axis, and the original cls/reg cross-correlation heads with PSS/IoU quality assessment are kept. Trained end-to-end on GOT-10k with softsign surrogate gradients, it reaches 85.24% precision / 64.37% success on OTB100 — near ANN parity (−4/−4 points) and far above the prior SNN tracker SiamSNN (52.78/44.32) — at ~67 FPS.",
    problem:
      "ANN trackers (SiamFC++) are accurate but power-hungry von-Neumann workloads; SNNs promise event-driven low-power tracking on neuromorphic chips, yet spike non-differentiability blocked direct training of deep SNN trackers — conversion needs many time steps and accumulates layer-wise error, and the existing SiamSNN lagged badly (52.78% precision).",
    background: ["sot", "bounding-box", "siamese", "iou", "success-plot"],
    previousWork: [
      {
        name: "ANN Siamese trackers (SiamFC, SiamRPN/RPN++, SiamFC++)",
        limitation:
          "Continuous-valued CNNs demand heavy multiply-accumulate compute and memory traffic, blocking deployment on resource/energy-constrained chips.",
        whyThisPaper:
          "Keeps the SiamFC++ head design but replaces the backbone with a directly trained spiking AlexNet at small ANN-parity loss.",
      },
      {
        name: "ANN-to-SNN conversion (incl. SiamSNN-style converted trackers)",
        limitation:
          "Rate coding needs many time steps (high latency) and conversion error accumulates layer by layer, capping depth and accuracy (SiamSNN 44.32 success).",
        whyThisPaper:
          "Surrogate-gradient direct training with T=6 steps cuts latency and reaches 64.37 success — best reported SNN tracking.",
      },
      {
        name: "Shallow/unsupervised SNN training (STDP, tempotron, ReSuMe)",
        limitation:
          "Biologically flavored rules train shallow nets well but do not scale to the deep classification+localization stack tracking needs.",
        whyThisPaper:
          "Spatio-temporal backprop with a softsign surrogate scales supervision through 5 spiking conv layers plus two head branches.",
      },
    ],
    researchGap:
      "Before this paper, no directly trained deep SNN tracker matched ANN-level accuracy; SNN tracking trailed SiamFC++ by ~37 precision points.",
    contribution: [
      "Spiking SiamFC++ framework: spike encoder (pixel → length-96 spike sequence, T=6) + time-domain Spiking AlexNet + rate-code collapse + per-branch 3×3 convs + cross-correlation cls/reg heads.",
      "End-to-end surrogate-gradient supervised training (softsign, α-controlled) with the full SiamFC++ loss (focal cls + BCE quality + IoU reg) in SpikingJelly/PyTorch.",
      "PSS and IoU quality-assessment variants; IoU variant best (85.24/64.37), both above SiamFC and far above SiamSNN on OTB100.",
      "Cross-benchmark validation (OTB2015, VOT2016/2018, UAV123) with per-scenario robustness demos (occlusion, deformation, clutter, darkness) and ~67 FPS inference.",
    ],
    method: {
      pipeline: ["crop", "spike-encode", "spike-extract", "rate-collapse", "branch-conv", "cross-correlate", "score-quality", "track"],
      architecture:
        "Template 3×127×127 / search 3×303×303 → spike encoder (T=6) → Spiking AlexNet (5 conv + IF nodes + 2 max-pools; template T×256×6×6, search T×256×28×28) → firing-rate collapse → three 3×3 convs per branch → cross-correlation → 1×17×17 cls and 4×17×17 reg maps; score = cls × quality (PSS or IoU).",
      appearanceModel:
        "Spike-train features with binary transmission/compute; IF neurons stand in for ReLU nonlinearity; final representation is the T-step firing rate.",
      association: "Not applicable — single-object Siamese matching by cross-correlation response.",
      detectionDependency: "None — template-initialized SOT with no external detector.",
      trackManagement:
        "Per-frame argmax over quality-weighted score map (SiamFC++ protocol, stride s=8 back-projection); positive if mapped center inside GT box.",
      loss:
        "L = mean focal Lcls + mean quality BCE (positive-only) + mean IoU Lreg (positive-only); PSS from (l,t,r,b) vector or box-IoU as quality target.",
      optimization:
        "SGD lr 0.01, batch 16, 20 epochs on GOT-10k (9935 train videos); T=6 ms window, dt=1 ms, Vth=1 mV; IoU/cls/ctr/reg curves monitored to convergence.",
    },
    equations: [
      {
        id: "spikingsiam-firingrate",
        label: "Rate-code collapse of the time axis",
        formula: "ψ = N / T (max 1 when a spike fires every step)",
        variables: [
          { symbol: "N", meaning: "spike count of the neuron in the observation window" },
          { symbol: "T", meaning: "observation time steps (6)" },
          { symbol: "ψ", meaning: "firing rate replacing the spike train for the heads" },
        ],
        intuition:
          "Count how often each neuron fired and use the fraction as an ordinary activation number.",
        why:
          "Removes the extra time dimension so standard conv/correlation heads operate on static feature maps.",
        where: "Section II-A, Eq. 1; final layer of Spiking AlexNet.",
        paperIds: ["T078"],
      },
      {
        id: "spikingsiam-xcorr",
        label: "Branch-wise template–search cross-correlation",
        formula: "f_i(z,x) = φ_i(ϕ(ζ(z))) ∗ φ_i(ϕ(ζ(x))), i ∈ {cls, reg}",
        variables: [
          { symbol: "ζ", meaning: "spike encoder (RGB → spike train)" },
          { symbol: "ϕ", meaning: "shared Spiking AlexNet feature extractor + rate collapse" },
          { symbol: "φ_i", meaning: "three 3×3 task convs for the cls or reg branch" },
          { symbol: "∗", meaning: "cross-correlation (convolution) of template over search features" },
        ],
        intuition:
          "Slide the spiking template features across the search features per branch; peaks mark the target for classification and box refinement.",
        why:
          "Preserves the proven SiamFC++ two-task matching structure while all features stay spiking up to the rate collapse.",
        where: "Section II-A, Eq. 2; head of both branches.",
        simulator: "siamese",
        paperIds: ["T078"],
      },
      {
        id: "spikingsiam-pss",
        label: "Prior spatial score from box geometry",
        formula: "PSS = √(min(l,r)/max(l,r) · min(t,b)/max(t,b)); d* = (l,t,r,b) from GT corners",
        variables: [
          { symbol: "l / t / r / b", meaning: "distances from mapped center to GT left/top/right/bottom edges" },
          { symbol: "d*", meaning: "4-D regression target vector" },
          { symbol: "PSS", meaning: "centerness-like quality: 1 at box center, →0 at edges" },
        ],
        intuition:
          "Positions near the box center get trusted more than positions near its rim.",
        why:
          "Down-weights ambiguous edge predictions at inference (cls × PSS) without extra parameters.",
        where: "Section II-A, Eq. 3; cls-branch quality for the PSS variant.",
        paperIds: ["T078"],
      },
      {
        id: "spikingsiam-iou",
        label: "IoU quality assessment",
        formula: "IoU = Intersection(B, B*) / Union(B, B*)",
        variables: [
          { symbol: "B", meaning: "predicted bounding box" },
          { symbol: "B*", meaning: "ground-truth bounding box" },
        ],
        intuition:
          "The fraction of agreed area over total covered area — the standard box-quality ruler.",
        why:
          "IoU as the quality target beats PSS (64.37 vs 62.49 success) and defines the success-rate metric itself.",
        where: "Section II-A, Eq. 4; quality branch target and evaluation overlap.",
        simulator: "iou-track",
        paperIds: ["T078"],
      },
      {
        id: "spikingsiam-loss",
        label: "Joint classification–quality–regression loss",
        formula: "L = (1/Npos)·Σ Lcls + (1/Npos)·Σ c*·Lquality + (1/Npos)·Σ c*·Lreg",
        variables: [
          { symbol: "Lcls", meaning: "focal classification loss" },
          { symbol: "Lquality", meaning: "binary cross-entropy on PSS/IoU quality" },
          { symbol: "Lreg", meaning: "IoU box-regression loss" },
          { symbol: "c* / Npos", meaning: "positive-sample indicator and positive count" },
        ],
        intuition:
          "Grade the tracker on finding the target, judging box quality, and tightening boxes — averaged over real target positions only.",
        why:
          "Identical three-way supervision to SiamFC++ makes the ANN→SNN comparison apples-to-apples.",
        where: "Section II-B, Eq. 8; training objective.",
        paperIds: ["T078"],
      },
      {
        id: "spikingsiam-surrogate",
        label: "Softsign surrogate gradient",
        formula: "forward: step spikes; backward: g(x) = (1/2)·(αx / (1 + |αx|) + 1)",
        variables: [
          { symbol: "g(x)", meaning: "differentiable surrogate replacing the spike derivative" },
          { symbol: "α", meaning: "steepness control of the softsign" },
        ],
        intuition:
          "Fire hard 0/1 spikes going forward, but pretend the threshold was a smooth S-curve when sending gradients back.",
        why:
          "The core trick enabling direct gradient training of the deep spiking stack in 6 time steps.",
        where: "Section II-B, Eq. 9; backprop through IF nodes.",
        paperIds: ["T078"],
      },
      {
        id: "spikingsiam-metrics",
        label: "Precision, success rate and frame error",
        formula: "precision = N(dist<thr)/Ntotal, dist = √((x0−x1)²+(y0−y1)²); success = N(IoU>thr)/Ntotal; error = (1/Nf)·Σ √((xM−xgt)²+(yM−ygt)²)",
        variables: [
          { symbol: "dist", meaning: "center Euclidean distance between predicted and GT boxes" },
          { symbol: "IoU", meaning: "predicted–GT box overlap" },
          { symbol: "error", meaning: "mean center error of tracker M over Nf frames" },
        ],
        intuition:
          "How often the center is close, how often boxes overlap enough, and the average miss distance per frame.",
        why:
          "OTB-style quantification behind every claim: precision/success curves plus per-frame error traces (Car4, DragonBaby, Basketball, Bird2).",
        where: "Section III-C, Eqs. 10–13; evaluation protocol.",
        simulator: "metrics",
        paperIds: ["T078"],
      },
    ],
    datasets: ["otb", "vot", "got10k", "uav123"],
    metrics: ["precision", "success-auc", "eao"],
    baselines: ["SiamFC", "SiamFC++", "SiamRPN", "SiamRPN++", "SiamSNN"],
    results: [
      "OTB100: Spiking SiamFC++ (IoU) 85.24 precision / 64.37 success; (PSS) 82.09/62.49 — vs SiamFC++ 89.30/68.50 and SiamFC 76.06/56.29; vs SiamSNN 52.78/44.32.",
      "OTB2015: precision 0.854 / success 0.644 (−0.030/−0.038 vs SiamFC++ 0.884/0.682; above SiamFC 0.761/0.563 and SiamRPN 0.851/0.637).",
      "VOT2016: A 0.600 / R 0.359 / EAO 0.302 (SiamFC++ 0.626/0.144/0.460); VOT2018: A 0.556 (= SiamFC++) / R 0.445 / EAO 0.255 (SiamFC++ 0.400).",
      "UAV123: AUC 0.578 (= SiamRPN++) / DP 0.744 (SiamFC++ 0.623/0.781; SiamRPN++ 0.769).",
      "Head-to-head over OTB100 videos: better than SiamFC++ on 32 sequences, worse on 36, comparable otherwise; ~67 FPS inference.",
    ],
    ablations: [
      "Quality assessment: IoU target (64.37/85.24) beats PSS (62.49/82.09) on OTB100 success/precision.",
      "Threshold sweeps (Fig. 7): at small center-error thresholds all three models tie; gaps open only at relaxed thresholds — near-parity on easy frames.",
      "Per-frame error traces (Fig. 8): wins on Car4/DragonBaby over time; slightly worse on Basketball/Bird2; scenario demos hold IoU >0.7–0.9 under occlusion, deformation, clutter and darkness.",
    ],
    limitations: {
      authorStated: [
        "Authors note SNNs still trail ANNs in accuracy on most learning tasks — the field-level gap this work narrows but does not close (−4 points vs SiamFC++).",
      ],
      evident: [
        "Robustness gap concentrates in VOT R/EAO (0.359/0.302 vs 0.144/0.460): more tracking failures, not just looser boxes.",
        "Fixed T=6 rate code with 1 mV threshold is unevaluated outside the chosen operating point; energy claims are architectural, with no chip power measurement.",
      ],
    },
    assumptions: [
      "Rate coding over 6 steps carries sufficient information for classification and localization.",
      "GOT-10k training (563 classes) transfers to OTB/VOT/UAV evaluation domains.",
    ],
    computation:
      "2× Xeon E5-2620 + 2× RTX 2070 Super (8 GB); PyTorch 1.8 + SpikingJelly; SGD batch 16; ~67 FPS inference reported.",
    relations: [
      { to: "T035", type: "builds-on", note: "Ports the SiamFC++ head, PSS/IoU quality and loss design directly into a spiking backbone." },
      { to: "T009", type: "builds-on", note: "Keeps the SiamFC fully-convolutional Siamese matching formulation (SiamSNN comparison is also SiamFC-derived)." },
      { to: "T028", type: "uses-as-baseline", note: "SiamRPN++ compared on OTB2015/VOT/UAV123 (e.g. UAV123 AUC tied at 0.578)." },
      { to: "T025", type: "uses-as-baseline", note: "Trained on the GOT-10k train split; OTB100-only evaluation protocol mirrors its one-shot spirit." },
    ],
    concepts: ["sot", "siamese", "bounding-box", "success-plot", "appearance-features"],
    impact:
      "Best-reported SNN tracking to date, cutting the ANN gap from ~37 to ~4 precision points and giving neuromorphic tracking a directly trained deep baseline for energy-efficient chips.",
  },
];
