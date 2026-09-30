import type { PaperRecord } from "../types";

/* batch-02 — T001, T002, T003, T004, T006, T007, T008, T009, T010, T011, T012, T013 */

export const BATCH_02: PaperRecord[] = [
  {
    id: "T001",
    arxiv: "1404.7584",
    title: "High-Speed Tracking with Kernelized Correlation Filters",
    shortTitle: "KCF",
    year: 2014,
    authors: ["João F. Henriques", "Rui Caseiro", "Pedro Martins", "Jorge Batista"],
    venue: "IEEE TPAMI",
    fileName: "1404.7584v3.pdf",
    task: "single-object",
    tags: [
      "correlation-filter",
      "kernel-methods",
      "circulant-matrix",
      "fourier-domain",
      "real-time",
      "online-learning",
    ],
    difficulty: "intermediate",
    summary:
      "KCF shows that the thousands of translated training patches a tracker wants are not independent data: stacked as a matrix they are circulant, so a DFT diagonalizes them and ridge regression collapses into a few element-wise operations. With a Gaussian kernel the same trick works exactly (the kernel matrix of circulant samples is also circulant), giving a nonlinear classifier as fast as the linear correlation filter — hundreds of frames per second while beating Struck and TLD on the 50-video benchmark.",
    problem:
      "Discriminative trackers train on many translated copies of the target patch so the classifier sees the object and its shifted surroundings, but extracting and solving over those samples naively costs O(n^3) per update — too slow for video. Exhaustive sliding-window detectors have the same flaw: every candidate window must be scored explicitly. There was no way to exploit a huge, highly redundant set of translated samples without paying for each of them.",
    background: [
      "correlation-filter",
      "bounding-box",
      "detection",
      "appearance-features",
      "online-vs-offline",
      "sot",
      "iou",
    ],
    previousWork: [
      {
        name: "MOSSE (Bolme et al.)",
        limitation:
          "Adaptive correlation filters proved that online least-squares filters are fast and robust, but they use only grayscale intensity and a single linear model with limited features.",
        whyThisPaper:
          "KCF keeps the filter framework but trains on dense cyclic shifts inside a kernel feature space, lifting performance with HOG features while preserving MOSSE-like speed.",
      },
      {
        name: "Exhaustive sampling plus ridge regression (Struck-style patch classification)",
        limitation:
          "The data matrix built from all translated patches is deeply redundant — overlapping pixels are constrained to be identical — and inverting it costs O(n^3).",
        whyThisPaper:
          "Proves the sample matrix is circulant and therefore diagonalized by the DFT, turning the exact same regression into O(n log n) element-wise work (Eq. 12).",
      },
      {
        name: "Struck and TLD",
        limitation:
          "Top accuracy on OTB but at only about 20-28 FPS, with TLD relying on explicit failure detection and Struck on expensive structured-output optimization.",
        whyThisPaper:
          "KCF and DCF outperform both on the 50-video benchmark while running at 172-292 FPS in a few lines of code.",
      },
    ],
    researchGap:
      "Before this paper, nobody had shown that dense translated samples can be handled exactly and cheaply through circulant structure, nor derived a kernel correlation filter with the same complexity as its linear counterpart.",
    contribution: [
      "Analytic model of thousands of translated patches: the resulting data matrix X is circulant, so X = F diag(xhat) F^H and ridge regression closes in the Fourier domain (Eq. 12).",
      "KCF: kernelized ridge regression on cyclic shifts whose dual solution alphahat = yhat/(khat_xx + lambda) has exactly the same O(n log n) complexity as the linear filter.",
      "DCF: a fast multi-channel linear extension via a linear kernel, equivalent to a correlation filter but with arbitrary feature channels.",
      "End-to-end tracker (Algorithm 1) on the 50-video benchmark: KCF with HOG reaches 73.2% precision at 172 FPS; all 50 videos processed in under 2 minutes on 4 cores.",
      "Open-source implementation; the circulant-matrix analysis tools are reused by the whole subsequent CF literature.",
    ],
    method: {
      pipeline: ["crop", "features", "train", "detect", "update"],
      architecture:
        "Single-object online tracker: extract the target patch plus a cyclic-shift-padded context region (padding factor s = sqrt(mn)/10), build the augmented sample matrix once per frame, solve ridge regression in the Fourier domain, then correlate the learned filter with the new patch and take the argmax response.",
      motionModel:
        "None — the paper states the framework includes no heuristics for failure detection or motion modeling; the search is centred on the previous target position.",
      appearanceModel:
        "HOG feature channels (or raw grayscale pixels), embedded by a Gaussian kernel k(x, x') = exp(-||x - x'||^2 / sigma^2) for KCF, or a linear kernel for DCF.",
      loss:
        "Ridge regression (regularized least squares) over all cyclic shifts: min_w ||X^H w - y||^2 + lambda ||w||^2, solved in closed form.",
      optimization:
        "DFT diagonalization of circulant matrices; dual alphahat = yhat/(khat_xx + lambda); detection response fhat(z) = khat_xz .* alphahat evaluated entirely with element-wise products and one inverse FFT.",
    },
    equations: [
      {
        id: "kcf-ridge",
        label: "Ridge regression closed form",
        formula: "w = (X^H X + lambda I)^-1 X^H y",
        variables: [
          { symbol: "X", meaning: "data matrix whose rows are all translated training patches" },
          { symbol: "y", meaning: "label vector peaked at the target centre (Gaussian ridge)" },
          { symbol: "lambda", meaning: "regularization weight (1e-4 in the experiments)" },
          { symbol: "w", meaning: "filter weights to learn" },
        ],
        intuition:
          "The classifier is just a least-squares fit to shifted copies of the target — but the matrix inversion is the expensive part.",
        why:
          "This is the objective KCF actually solves; everything in the paper exists to avoid paying for this inversion directly.",
        where: "Section 4 (ridge regression formulation), Eq. 3.",
        params: "lambda = 1e-4; label ridge sigma = 0.2 (KCF) / 0.5 (DCF); learning rate 0.075 (KCF) / 0.02 (DCF).",
        paperIds: ["T001"],
      },
      {
        id: "kcf-fourier",
        label: "Fourier solution of the linear filter",
        formula: "what = (xhat* .* yhat) / (xhat* .* xhat + lambda)",
        variables: [
          { symbol: "xhat, yhat", meaning: "DFT of the base patch and of the label signal" },
          { symbol: "xhat*", meaning: "complex conjugate" },
          { symbol: ".*, /", meaning: "element-wise multiplication and division" },
          { symbol: "what", meaning: "DFT of the learned filter" },
        ],
        intuition:
          "Because X is circulant, inverting it in pixel space becomes dividing spectra point-by-point in Fourier space — no matrix inversion at all.",
        why:
          "This is the exact ridge solution, not an approximation, and it cuts storage and computation from O(n^3) to a nearly-linear O(n log n).",
        where: "Section 4.4-4.5, Eq. 12; recovered to the spatial domain by one inverse DFT.",
        params: "Adding samples costs almost nothing: each new cyclic shift only adds another Fourier coefficient.",
        simulator: "corr-filter",
        paperIds: ["T001"],
      },
      {
        id: "kcf-dual-kernel",
        label: "Kernelized dual solution",
        formula: "alphahat = yhat / (khat_xx + lambda)",
        variables: [
          { symbol: "alphahat", meaning: "DFT of the dual coefficients (representer weights)" },
          { symbol: "khat_xx", meaning: "DFT of the kernel correlation of the patch with itself" },
          { symbol: "yhat", meaning: "DFT of the labels" },
        ],
        intuition:
          "The kernel trick replaces the feature-space filter with coefficients over the samples, and because the kernel matrix of cyclic shifts is also circulant, the same divide-in-Fourier trick applies.",
        why:
          "It is what makes KCF nonlinear yet exactly as fast as the linear DCF — the headline claim of the paper.",
        where: "Sections 5-6 (kernel regression): dual form of the representer solution applied to circulant K.",
        params:
          "Gaussian kernel width sigma controls nonlinearity: larger sigma gives a smoother, more linear-like decision surface.",
        simulator: "corr-filter",
        paperIds: ["T001"],
      },
      {
        id: "kcf-detect",
        label: "Detection response of a candidate patch",
        formula: "fhat(z) = khat_xz .* alphahat ,  z* = argmax_z IFFT(fhat(z))",
        variables: [
          { symbol: "z", meaning: "candidate patch (a cyclic shift of the new frame region)" },
          { symbol: "khat_xz", meaning: "kernel cross-correlation between training patch and candidate" },
          { symbol: "alphahat", meaning: "learned dual coefficients" },
        ],
        intuition:
          "Score every shifted position of the new frame at once by correlating spectra — the peak of the response map is the new target location.",
        why:
          "This is detection without an exhaustive sliding window: one inverse FFT yields the score of all positions simultaneously.",
        where: "Detection step of Algorithm 1, applied each frame after cropping around the previous position.",
        params:
          "Learning rate eta blends old and new alphahat so the model adapts (0.075 for KCF); no explicit failure or scale handling.",
        simulator: "corr-filter",
        paperIds: ["T001"],
      },
    ],
    datasets: ["otb"],
    metrics: ["precision", "fps"],
    baselines: ["Struck", "TLD", "MOSSE", "CSK", "DSST"],
    results: [
      "50-video benchmark, precision at the 20 px threshold: KCF (HOG) 73.2% at 172 FPS; DCF (HOG) 72.8% at 292 FPS; Struck 65.6% at 20 FPS; TLD 60.8% at 28 FPS; MOSSE 43.1% at 615 FPS.",
      "With raw grayscale pixels instead of HOG: KCF 56.0% at 154 FPS, DCF 45.1% at 278 FPS — most of the accuracy gain comes from the feature, not the kernel.",
      "Abstract claim: KCF and DCF outperform top-ranking trackers such as Struck or TLD on the 50-video benchmark while running at hundreds of frames per second.",
      "All 50 videos (about 29,000 frames) take less than 2 minutes on 4 CPU cores; the tracker is a few lines of code (Algorithm 1).",
    ],
    ablations: [
      "Features: HOG vs raw pixels at fixed hyper-parameters (KCF 73.2 vs 56.0 precision; DCF 72.8 vs 45.1).",
      "Model: linear DCF vs Gaussian-kernel KCF with the same features (72.8 vs 73.2) — the kernel adds little on OTB but comes at identical asymptotic cost.",
      "Hyper-parameter grid reported: sigma in {0.2, 0.5}, learning rate {0.075, 0.02}, lambda = 1e-4, context padding s = sqrt(mn)/10.",
    ],
    limitations: {
      authorStated: [
        "The framework includes no heuristics for failure detection and no motion modeling; the model is trained on the first-frame patch and updated at the position found in the previous frame.",
        "Detection runs over the patch at the previous position only — scale estimation is not part of the algorithm.",
      ],
      evident: [
        "The circulant assumption treats wrapped-around border pixels as valid negatives, introducing the periodic boundary effects that SRDCF (T010) later attacks with spatial regularization.",
        "Evaluation reports precision plots on the 50-video OTB benchmark only — no VOT-style reset-based robustness protocol.",
      ],
    },
    assumptions: [
      "Translated copies of the target patch are valid training samples (the cyclic-shift data model).",
      "The target stays inside the padded search window between frames (no explicit motion model).",
    ],
    computation:
      "All training and detection in the Fourier domain: O(n log n) per frame, O(n) memory. 172 FPS (KCF-HOG) / 292 FPS (DCF-HOG) on a 4-core desktop; full OTB-50 in under 2 minutes.",
    relations: [],
    concepts: ["correlation-filter", "appearance-features", "online-vs-offline", "sot", "success-plot"],
    impact:
      "KCF became the canonical fast tracker and the default baseline of the whole CF line: the CF survey (T002), MDNet (T003), Staple (T004), GOTURN (T007), SiamFC (T009), SRDCF (T010), ECO (T011) and BACF (T012) all evaluate against it, and its Fourier-domain recipe is still the skeleton of every correlation-filter tracker.",
  },
  {
    id: "T002",
    arxiv: "1509.05520",
    title: "An Experimental Survey on Correlation Filter-based Tracking",
    shortTitle: "CF Tracking Survey",
    year: 2015,
    authors: ["Zhe Chen", "Zhibin Hong", "Dacheng Tao"],
    fileName: "1509.05520v1.pdf",
    task: "survey",
    tags: [
      "correlation-filter",
      "survey",
      "experimental-comparison",
      "feature-analysis",
      "long-term-tracking",
    ],
    difficulty: "intermediate",
    summary:
      "This paper surveys eleven correlation-filter trackers (CFTs), extracts the general train-detect-update framework they all share, and re-evaluates them in one controlled experiment on OTB (OPE/TRE/SRE plus attributes). It concludes that recent CFTs such as MUSTer and SAMF reach state-of-the-art robustness and speed, and it maps the open problems of the family: scale estimation, part-based models and long-term tracking.",
    problem:
      "Correlation filters exploded after MOSSE and KCF, but each new CFT was published with its own features, implementation, training schedule and evaluation protocol, so their numbers were not comparable and their differences were unclear. The field had no single account of what the shared framework actually is, which design choices matter, and where the family still fails.",
    background: [
      "correlation-filter",
      "appearance-features",
      "online-vs-offline",
      "sot",
      "success-plot",
      "bounding-box",
    ],
    previousWork: [
      {
        name: "MOSSE (Bolme et al.)",
        limitation:
          "Introduced efficient adaptive filtering for tracking, but uses grayscale intensity only and has no mechanism for scale change or long-term recovery.",
        whyThisPaper:
          "The survey traces every later CFT (kernelized, color-attribute, scale-adaptive, long-term) as an improvement over this single starting point.",
      },
      {
        name: "KCF / DCF (T001) and related kernel CFTs",
        limitation:
          "Kernelized filters improved discrimination but still assume a fixed-scale, single-model template trained only from shifted crops.",
        whyThisPaper:
          "Summarizes the ridge-regression/FFT training schemes in one place (Eqs. 8-17) and evaluates KCF alongside ten newer CFTs under one protocol.",
      },
      {
        name: "Published benchmark results of individual CFT papers",
        limitation:
          "Results were scattered over different implementations, feature sets and evaluation settings — not directly comparable.",
        whyThisPaper:
          "Re-runs 11 CFTs plus four indicative trackers with common OPE/TRE/SRE code and reports center location error, overlap score and FPS in a single table.",
      },
    ],
    researchGap:
      "Before this survey, correlation-filter trackers had no shared framework description and no controlled head-to-head comparison across accuracy, robustness and speed.",
    contribution: [
      "Surveys 11 CFTs and abstracts the general framework: train a filter on the cropped patch at the first (or previous) position, detect by correlating the new crop, update the filter online.",
      "Unified mathematical treatment of training schemes: regularized least squares objective, circulant-matrix Fourier solution, kernelization, and the adaptive update rule.",
      "Controlled experiments on OTB (OPE, TRE, SRE and per-attribute success/precision plots) covering 10 CFTs plus four indicative non-CFT trackers, with average center location error, overlap score and speed.",
      "Identifies future directions for the family: estimating scale, adopting part-based strategies, and cooperating with long-term tracking methods.",
    ],
    method: {
      pipeline: ["framework", "classify-trainers", "implement", "protocol", "evaluate", "conclude"],
      architecture:
        "Not a new tracker: a survey pipeline that (a) formalizes the shared CFT loop — crop, extract features, apply cosine window, learn the filter in the Fourier domain, correlate at the previous position, adaptively update — and (b) re-implements the surveyed trackers under one evaluation harness.",
      appearanceModel:
        "Compares the feature choices across surveyed CFTs: raw grayscale, HOG, color names, color attributes, and hybrids, plus the CNN/long-term modules added by MUSTer and LCT.",
      optimization:
        "Training schemes classified into closed-form frequency-domain solutions (MOSSE/KCF style ridge solved with FFT) versus iterative or spatial-domain learning; adaptive learning-rate updates of the filter each frame.",
    },
    equations: [
      {
        id: "survey-design",
        label: "Filter design in the Fourier domain",
        formula: "y = F^-1(xhat' .* conj(hhat)) ,  conj(hhat) = yhat / xhat'",
        variables: [
          { symbol: "xhat'", meaning: "DFT of the training patch (signal x')" },
          { symbol: "yhat", meaning: "DFT of the desired correlation output (peak at the target)" },
          { symbol: "hhat", meaning: "DFT of the filter h" },
          { symbol: "F^-1", meaning: "inverse discrete Fourier transform" },
        ],
        intuition:
          "Correlation is division of spectra: the filter that turns the input patch into a peak at the target is just yhat divided by xhat in Fourier space.",
        why:
          "It is the shared mechanism behind every surveyed CFT and the reason they are fast — one FFT pair replaces an exhaustive convolution.",
        where: "Section II, correlation filter design equations (2)-(3).",
        simulator: "corr-filter",
        paperIds: ["T002"],
      },
      {
        id: "survey-objective",
        label: "Regularized training objective",
        formula: "min_w sum_i L(f(w, x_i), y_i) + lambda ||w||^2",
        variables: [
          { symbol: "w", meaning: "filter parameters" },
          { symbol: "L", meaning: "loss — hinge (SVM), quadratic (RLS/ridge), etc." },
          { symbol: "lambda", meaning: "regularization preventing overfitting" },
          { symbol: "x_i, y_i", meaning: "training patch and label pair" },
        ],
        intuition:
          "Every discriminative filter is a classifier fit: make the scores match the labels, with a penalty on large weights.",
        why:
          "Unifies SVM, ridge regression and correlation filters under one objective so the survey can explain which CFT uses which loss.",
        where: "Section III, Eq. 8 (closed-form ridge solution in Eq. 9).",
        paperIds: ["T002"],
      },
      {
        id: "survey-fourier-solution",
        label: "Frequency-domain ridge solution",
        formula: "what = (xhat* .* yhat) / (xhat* .* xhat + lambda)",
        variables: [
          { symbol: "what", meaning: "DFT of the learned filter" },
          { symbol: "xhat*, yhat", meaning: "conjugate DFT of the patch and DFT of the labels" },
          { symbol: "lambda", meaning: "ridge regularization" },
        ],
        intuition:
          "With circulant samples the ridge solution collapses to an element-wise formula in the Fourier domain, and the kernel version follows as alphahat = yhat/(khat + lambda).",
        why:
          "This equation is the mathematical core the survey attributes to Henriques et al. and reuses to explain all kernel CFTs.",
        where: "Section III, Eqs. 15-17 (Fourier solution and kernel dual).",
        simulator: "corr-filter",
        paperIds: ["T002", "T001"],
      },
    ],
    datasets: ["otb"],
    metrics: ["precision", "success-auc", "fps"],
    baselines: ["KCF", "DSST", "SAMF", "MUSTer", "MEEM", "MOSSE"],
    results: [
      "OTB (OOTB) one-pass evaluation, average results: MUSTer CLE 17.3 px / overlap 65.0% / 3.85 FPS; KCF CLE 35.5 / overlap 51.9% / 191 FPS; MEEM overlap 57.9%; RPT 58.2%; DSST 56.1%.",
      "Efficiency range among surveyed trackers: STC 580 FPS, CSK 288 FPS, MOSSE 281 FPS, KCF 191 FPS, down to MUSTer 3.85 FPS.",
      "Qualitative finding: six improved CFTs (including MUSTer, RPT, DSST, SAMF) consistently deliver top-10 robustness, and CFTs dominate the illumination-variation, occlusion and background-clutter attributes.",
      "Future directions stated by the authors: scale estimation, part-based tracking strategies, and cooperating CFTs with long-term tracking methods.",
    ],
    ablations: [
      "Three evaluation protocols run for every tracker (OPE, TRE, SRE) plus per-attribute success and precision plots — separating true robustness from sensitivity to initialization and temporal shifts.",
      "Head-to-head of training schemes and features across the 11 CFTs (raw vs HOG vs color features, linear vs kernel, with/without scale estimation) under identical code.",
    ],
    limitations: {
      authorStated: [
        "CFT weaknesses named by the authors: sensitivity to scale change, rigid templates that break under deformation, and weak long-term capability — the three areas they list for future improvement.",
        "Reported speeds and scores are the survey's own measurements of its implementations, not the numbers printed in the original papers.",
      ],
      evident: [
        "Only OTB-family evaluation is used — no MOT or VOT2015/16 protocol — so conclusions transfer mainly to short-term single-object tracking on OTB.",
        "Cross-tracker FPS differences partly reflect implementation quality, since the trackers are the survey's re-implementations in a common harness.",
      ],
    },
    computation:
      "Survey runs eleven trackers over OTB with OPE/TRE/SRE; reported speeds span 3.85-580 FPS depending on the tracker.",
    assumptions: [
      "The eleven surveyed CFTs can be re-implemented faithfully enough that their re-run numbers are comparable to the original papers.",
      "OTB's OPE/TRE/SRE protocols are sufficient to characterize both accuracy and robustness of correlation-filter trackers.",
    ],
    relations: [
      {
        to: "T001",
        type: "uses-as-baseline",
        note: "Evaluates Henriques et al.'s KCF among the eleven surveyed CFTs under its own protocol (overlap 51.9%, CLE 35.5, 191 FPS) and derives its kernel-training section from that work.",
      },
    ],
    concepts: ["correlation-filter", "appearance-features", "online-vs-offline", "success-plot", "sot"],
    impact:
      "It fixed the vocabulary of the CF literature — the shared train/detect/update framework, the training-scheme taxonomy and the list of open problems (scale, parts, long-term) that the next CF papers (T004, T010, T011, T012) each attacked one by one.",
  },
  {
    id: "T003",
    arxiv: "1510.07945",
    title: "Learning Multi-Domain Convolutional Neural Networks for Visual Tracking",
    shortTitle: "MDNet",
    year: 2015,
    authors: ["Hyeonseob Nam", "Bohyung Han"],
    fileName: "1510.07945v2.pdf",
    task: "single-object",
    tags: [
      "cnn",
      "multi-domain-learning",
      "online-learning",
      "hard-negative-mining",
      "bounding-box-regression",
    ],
    difficulty: "intermediate",
    summary:
      "MDNet pretrains a small CNN on many tracking videos with one classification branch per video (multi-domain learning), so the shared layers learn domain-independent target/background features while each branch absorbs its sequence's idiosyncrasies. At test time a single new branch and the fully connected layers are fine-tuned online with hard negative mining, giving state-of-the-art accuracy on OTB and VOT2014 — at about 1 FPS.",
    problem:
      "CNNs trained for classification do not transfer directly to tracking: a fixed class label is inconsistent with locating an arbitrary single target, and a single softmax over all videos forces the network to merge sequences with completely different targets, backgrounds and motion. Training a per-video detector from scratch online was the only alternative, and it drifts without a strong generic prior.",
    background: [
      "deep-detection",
      "appearance-features",
      "online-vs-offline",
      "bounding-box",
      "sot",
      "occlusion",
      "detection",
    ],
    previousWork: [
      {
        name: "Image-classification CNNs (AlexNet, VGG) used as fixed features",
        limitation:
          "Shallow methods on CNN features do not take advantage of end-to-end learning, while per-sequence SGD fine-tuning of large networks is slow and prone to overfitting.",
        whyThisPaper:
          "Trains the network itself for tracking, but offline and multi-domain, so only a tiny classifier is adapted per sequence.",
      },
      {
        name: "Correlation-filter trackers (KCF/T001 and friends)",
        limitation:
          "Fast, but the hand-crafted shallow model cannot represent the semantic appearance changes (deformation, occlusion, illumination) that distinguish target from background.",
        whyThisPaper:
          "Replaces the filter with a discriminatively learned CNN representation and compares directly against KCF, DSST and SAMF on OTB and VOT2014.",
      },
      {
        name: "Single-domain or per-sequence training",
        limitation:
          "One branch trained on all sequences forces unrelated domains into one classifier; purely per-sequence learning lacks generic features and drifts.",
        whyThisPaper:
          "Multi-domain learning: domain-specific branches fc6_1...fc6_K during pretraining, shared layers absorbing what all sequences have in common.",
      },
    ],
    researchGap:
      "Before MDNet, no CNN was trained end-to-end specifically for tracking in a way that separates domain-independent features from sequence-specific classification.",
    contribution: [
      "Multi-domain learning framework: K domain-specific branches (one per training sequence) plus shared conv/fc layers, trained by SGD where each iteration updates only the (k mod K)-th branch.",
      "Online tracking with a single new domain branch: long-term updates every tau_l = 100 frames and short-term updates every tau_s = 20 frames, plus hard negative mining that keeps the top-scoring negatives of each minibatch.",
      "Bounding-box regression on conv3 features trained once on the first frame to refine candidate locations.",
      "State-of-the-art accuracy: OTB100 precision 0.909 / success 0.678, OTB50 precision 0.948 / success 0.708; first in accuracy on VOT2014 (combined rank 2.29).",
      "A deliberately small network (5 conv + 3 fc layers) with only the fully connected layers w4:6 updated online.",
    ],
    method: {
      pipeline: ["pretrain", "swap-branch", "sample", "classify", "hard-mine", "update", "regress"],
      architecture:
        "VGG-like small CNN: conv1-conv3 (shared, frozen online), fc4-fc5 (shared, fine-tuned online), plus K domain-specific branches fc6_1...fc6_K used only during multi-domain pretraining and replaced by a single fc6 at test time. Each branch is a binary softmax classifier target vs background.",
      motionModel:
        "None — N candidates are randomly sampled around the previous target state each frame; tracking control relies on update scheduling, not motion prediction.",
      appearanceModel:
        "CNN features from the shared layers, adapted online per sequence; positive samples near the current estimate and hard negatives with high background scores.",
      detectionDependency:
        "None — a sampling-based single-object tracker that classifies its own candidate windows.",
      trackManagement:
        "Failure-driven updates: when the estimated target scores as background (f+(x*) < 0.5), short-term retraining kicks in; long-term updates run at fixed intervals using a longer positive buffer.",
      optimization:
        "SGD with minibatches of M+ positives and Mh- hard negatives selected from the tested negatives; only w4:6 updated online (conv layers fixed to avoid overfitting and keep speed).",
      loss: "Softmax cross-entropy binary classification (target vs background) per domain branch.",
    },
    equations: [
      {
        id: "mdnet-select",
        label: "Target state selection",
        formula: "x* = argmax_{i in 1..N} f+(x_i)",
        variables: [
          { symbol: "x_i", meaning: "candidate windows sampled around the previous state" },
          { symbol: "f+(x_i)", meaning: "positive-class (target) score from the network" },
          { symbol: "x*", meaning: "estimated target state for the current frame" },
        ],
        intuition:
          "Track by classification: draw many boxes near the last known position and keep the one the network is most confident is still the target.",
        why:
          "It is the tracking rule of MDNet — everything else (sampling, mining, updates) exists to make this argmax reliable.",
        where: "Section 4.1, Eq. 1; bounding-box regression then adjusts x* when f+(x*) > 0.5.",
        params:
          "N candidates per frame; the 0.5 score threshold also gates both bbox refinement and failure-triggered short-term updates.",
        paperIds: ["T003"],
      },
      {
        id: "mdnet-hard-mine",
        label: "Hard negative minibatch selection",
        formula: "keep top Mh- negatives ranked by f+ among the tested negatives of the minibatch",
        variables: [
          { symbol: "Mh-", meaning: "number of hard negatives kept per minibatch" },
          { symbol: "f+", meaning: "current network positive score used to rank candidates" },
        ],
        intuition:
          "Retrain on the hardest impostors — the background patches the network currently mistakes for the target — instead of easy random negatives.",
        why:
          "Ordinary SGD over evenly distributed samples suffers drift because distractors are underweighted; mining concentrates capacity on the failures.",
        where: "Section 4.2 (hard minibatch mining), integrated into the online update step.",
        params:
          "The mined negatives become harder as training proceeds (Figure 2), automatically curriculum-ing the update.",
        paperIds: ["T003"],
      },
    ],
    datasets: ["otb", "vot"],
    metrics: ["precision", "success-auc", "fps"],
    baselines: ["KCF", "DSST", "SAMF", "MEEM", "MUSTer", "CNN-SVM"],
    results: [
      "OTB50 (OPE): precision 0.948, success 0.708 — ahead of MUSTer (0.865/0.641), CNN-SVM (0.852/0.597), MEEM (0.840/0.572), KCF (0.740).",
      "OTB100 (OPE): precision 0.909, success 0.678 — ahead of CNN-SVM (0.814/0.555), MUSTer (0.774/0.575), MEEM (0.786/0.533).",
      "VOT2014 baseline experiment: accuracy 0.63 (rank 2.50), robustness 0.16 (rank 2.08), combined rank 2.29 — best overall; under the region-noise protocol MDNet still leads with combined rank 3.45 (accuracy 0.60, rank 3.31).",
      "Runtime: around 1 FPS with eight cores of a 2.20 GHz Intel Xeon E5-2660 plus an NVIDIA Tesla K20m GPU.",
      "Offline pretraining uses 58 sequences from VOT2013/2014 (VOT evaluation excludes OTB-overlap videos); 89 OTB100 sequences are used for the OTB-specific pretraining run.",
    ],
    ablations: [
      "Multi-domain vs single-domain pretraining (precision/success): MDNet 0.909/0.678 vs SDNet 0.865/0.645 — separating domains is worth about 4 points of precision.",
      "Bounding-box regression: MDNet-BB drops to 0.891/0.650.",
      "Hard negative mining: without it (MDNet-BB-HM) precision/success fall to 0.816/0.602.",
      "Robustness to imprecise re-initialization (VOT region noise): MDNet combined rank 3.45 vs baseline 2.29, still first among compared trackers.",
    ],
    limitations: {
      authorStated: [
        "Failure cases Coupon and Jump: appearance change causes drift when the target blends with similar-looking content (Figure 7).",
        "The conclusion states MDNet can be combined with a re-detection module to achieve long-term tracking — i.e., the current tracker has no re-detection after long occlusion.",
        "About 1 FPS: online fine-tuning of the fully connected layers makes the tracker impractical for real-time use.",
      ],
      evident: [
        "Pretraining and online updates are both per-sequence binary classification — the tracker cannot handle multiple targets and restarts from scratch in each video.",
        "Relies on random candidate sampling rather than a motion model, so large inter-frame motion must be covered by the sampling radius.",
      ],
    },
    assumptions: [
      "Training sequences (VOT/OTB) share enough structure with the test video that domain-independent features transfer.",
      "The target remains classifiable as foreground against background from current appearance alone.",
    ],
    computation:
      "Pretraining is offline; online tracking evaluates N candidates and fine-tunes 3 fc layers per frame — around 1 FPS on an 8-core Xeon E5-2660 with a Tesla K20m.",
    relations: [
      {
        to: "T001",
        type: "uses-as-baseline",
        note: "Reports KCF (precision 0.740 on OTB50, plus a KCF row in the VOT2014 accuracy/robustness table) among the trackers it compares against on both benchmarks.",
      },
    ],
    concepts: ["deep-detection", "appearance-features", "online-vs-offline", "sot", "occlusion", "bounding-box"],
    impact:
      "MDNet defined the online-fine-tuning CNN tracker and became the deep baseline everyone reported against (SiamFC T009, ECO T011 and BACF T012 all list its scores); its multi-domain idea reappeared in later domain-adaptive trackers, and its 1 FPS cost is exactly the gap the offline-trained siamese trackers (T007-T009) were built to close.",
  },
  {
    id: "T004",
    arxiv: "1512.01355",
    title: "Staple: Complementary Learners for Real-Time Tracking",
    shortTitle: "Staple",
    year: 2015,
    authors: ["Luca Bertinetto", "Jack Valmadre", "Stuart Golodetz", "Ondrej Miksik", "Philip H.S. Torr"],
    fileName: "1512.01355v2.pdf",
    task: "single-object",
    tags: [
      "correlation-filter",
      "color-features",
      "complementary-learners",
      "real-time",
      "deformation",
      "bounding-box",
    ],
    difficulty: "intermediate",
    summary:
      "Staple combines two complementary learners in a single response map: a HOG correlation filter that is robust to deformation but sensitive to color, and a color-histogram template that is cheap, deformation-tolerant and independent of shape. Their scores are fused with a fixed mixing factor, giving real-time speed (over 80 FPS) with top accuracy on OTB and VOT2014 — and no online update of the appearance histogram, which keeps it drift-resistant and fast.",
    problem:
      "Correlation filters and template matching each fail in opposite regimes: filters track shape but are thrown off by illumination and background color changes, while pixel-intensity templates handle color and deformation but drift when the shape changes and are fragile to lighting. Combining learners had been tried, but usually at the cost of speed or with components that still under-perform on deformable objects.",
    background: [
      "correlation-filter",
      "appearance-features",
      "bounding-box",
      "sot",
      "iou",
      "occlusion",
    ],
    previousWork: [
      {
        name: "HOG correlation filters (KCF/T001, DSST)",
        limitation:
          "Invariant to illumination but blind to color; degrade when the object deforms or the background color shifts.",
        whyThisPaper:
          "Keeps the filter as the shape-aware learner and complements it with a color learner rather than replacing features.",
      },
      {
        name: "Pixel-intensity template matching / conventional trackers",
        limitation:
          "The template is tightly tied to the original shape and appearance, so it drifts when the object deforms.",
        whyThisPaper:
          "Staple's histogram learner ignores spatial structure entirely — it matches average colors, which survive deformation.",
      },
      {
        name: "SRDCF, MUSTer, LCT (prior fusion/scale works)",
        limitation:
          "SRDCF's spatial regularization gives excellent results at the cost of real-time performance; multi-component trackers get complex and slow.",
        whyThisPaper:
          "Staple keeps every component cheap (a filter plus a fixed histogram) so fusion remains above 80 FPS.",
      },
    ],
    researchGap:
      "No tracker before Staple showed that a single response map could fuse a discriminative filter and a deformation-tolerant color model without sacrificing real-time speed.",
    contribution: [
      "Two complementary learners combined by Eq. 3: a correlation filter on HOG features and a color-histogram template on raw pixels, added into one response map.",
      "The histogram template is estimated once from the first frame and never updated — avoiding the drift that plagues pixel templates (the authors call this the key to its robustness).",
      "End-to-end real-time tracker at over 80 FPS with accuracy competitive with the best (deep) trackers of the time.",
      "Empirical study on OTB-13 (OPE/TRE/SRE) and VOT2014 showing the fusion beats each learner alone and matches SRDCF-level robustness at CF speed.",
    ],
    method: {
      pipeline: ["crop", "filter-response", "histogram-response", "fuse", "update-filter", "estimate"],
      architecture:
        "Single-object tracker with two parallel scoring branches: (1) a learned correlation filter over HOG patches, (2) a fixed color histogram of the target window compared against candidate windows by a linear function of the average feature vector. The two response maps are normalized and added.",
      motionModel:
        "No explicit motion model; candidates come from the previous state and a small translation search (the response map argmax).",
      appearanceModel:
        "HOG shape features (filter, updated every frame with learning rate 0.075) plus a raw-pixel color histogram in Lab/RGB (template estimated from frame 1, not updated).",
      loss:
        "Filter branch: ridge-regression correlation filter. Histogram branch: a linear function that scores how well a candidate's average color matches the target histogram.",
      optimization:
        "Fourier-domain filter training (closed form) each frame; histogram branch needs no optimization after initialization; fusion factor alpha fixed by validation (alpha = 0.3).",
    },
    equations: [
      {
        id: "staple-estimate",
        label: "Target state estimate",
        formula: "p_t = argmax_{p in S_t} f(T(x_t, p); theta_{t-1})",
        variables: [
          { symbol: "p", meaning: "candidate target state (box position) in the search set" },
          { symbol: "T(x_t, p)", meaning: "patch of frame t extracted at candidate p" },
          { symbol: "theta_{t-1}", meaning: "model parameters carried from the previous frame" },
          { symbol: "f", meaning: "combined score function (Eq. 3)" },
        ],
        intuition:
          "Pick the box where the fused response — filter plus color — is highest.",
        why:
          "This is the tracking rule; both learners exist only to make this argmax well-founded.",
        where: "Section 3 (tracking problem formulation), Eq. 1.",
        paperIds: ["T004"],
      },
      {
        id: "staple-fuse",
        label: "Complementary learners fusion",
        formula: "f(x_t, p) = gamma_t * f_tmpl(x_t, p) + gamma_h * f_hist(x_t, p)",
        variables: [
          { symbol: "f_tmpl", meaning: "correlation-filter response (shape/HOG)" },
          { symbol: "f_hist", meaning: "color-histogram response" },
          { symbol: "gamma_t, gamma_h", meaning: "branch weights summing to one (equivalently alpha)" },
        ],
        intuition:
          "Add the two response maps: where one learner is uncertain the other fills in — color survives deformation, shape survives lighting.",
        why:
          "The paper's central claim: two learners with opposite failure modes, cheap enough to combine in real time.",
        where: "Section 4, Eq. 3; per-branch scores defined in Eqs. 4-5.",
        params:
          "Fusion weight alpha = 0.3 (filter branch weight), chosen from a parameter sweep; parameters listed in the implementation table.",
        paperIds: ["T004", "T001"],
      },
      {
        id: "staple-filter",
        label: "Correlation-filter template score",
        formula: "f_tmpl(x, p) = sum_u h[u]^T phi_x[u]",
        variables: [
          { symbol: "h", meaning: "learned filter (HOG channel weights)" },
          { symbol: "phi_x", meaning: "feature map of the candidate patch at offset u" },
          { symbol: "u", meaning: "spatial location in the patch" },
        ],
        intuition:
          "Slide the learned filter over the candidate patch and sum the responses — standard correlation-filter scoring.",
        why:
          "Carries the shape/edge information; updated online so the filter adapts to recent appearances.",
        where: "Section 4.1, Eq. 4 (filter score).",
        params: "Updated each frame with the usual Fourier-domain filter learning.",
        simulator: "corr-filter",
        paperIds: ["T004"],
      },
      {
        id: "staple-hist",
        label: "Color-histogram template score",
        formula: "f_hist(x, p) = w^T mu(T(x, p)),  mu_i = (1/|p|) sum_{j in p} phi_i(x_j)",
        variables: [
          { symbol: "mu", meaning: "average feature (color) vector of the candidate window" },
          { symbol: "phi_i", meaning: "per-bin color feature of pixel j" },
          { symbol: "w", meaning: "linear weights mapping average colors to a match score" },
        ],
        intuition:
          "Ignore layout, keep the distribution: score how similar the candidate's average colors are to the target's — tolerant to rotation and deformation.",
        why:
          "This is the complementary learner: no spatial structure, hence no drift when the object deforms, and it is nearly free to compute.",
        where: "Section 4.2, Eq. 5; the histogram is estimated from the first frame only.",
        params:
          "No online update of the histogram template by design — stated as the reason the appearance model does not drift.",
        paperIds: ["T004"],
      },
    ],
    datasets: ["otb", "vot"],
    metrics: ["precision", "success-auc", "fps", "eao"],
    baselines: ["KCF", "SRDCF", "DSST", "SAMF", "DAT", "MDNet"],
    results: [
      "VOT2014 (EOT): Staple accuracy 0.644, 9.38 expected failures, combined rank 4.37 — vs KCF 0.613/19.79/6.58, SRDCF 0.600/15.90, SAMF 0.603/19.23, DAT 0.519/15.87.",
      "VOT2015 (60 sequences): accuracy 0.538 with 80 expected failures, against DSST 0.491 with 152 failures (Table).",
      "OTB-13: OPE success 0.600, TRE 0.617, SRE 0.545 — matching or beating SRDCF-class trackers with CF-style speed.",
      "Runs at over 80 FPS — the speed advantage over SRDCF-style regularization.",
    ],
    ablations: [
      "Each learner alone vs the fusion: filter-only and histogram-only scores compared with the combined response on OTB and VOT (fused model wins).",
      "Fusion weight sweep: performance as a function of alpha, with alpha = 0.3 as the chosen operating point.",
      "Cross-protocol robustness: VOT2014 vs VOT2015 rankings show the tracker's stability under different initialization/noise regimes.",
    ],
    limitations: {
      authorStated: [
        "Staple has no re-detection: after a long occlusion or leave-frame event, the paper states tracking would be greatly improved by an ability to re-detect the target.",
        "The color-histogram learner is explicitly noted to be sensitive when illumination is inconsistent (colors shift), since histograms assume stable color statistics.",
        "Drift risk comes from the updated filter branch — the histogram is frozen precisely to avoid this.",
      ],
      evident: [
        "Scale is not explicitly estimated; the fixed two-branch search inherits the usual CF fixed-scale assumption.",
        "No motion model: after occlusion, recovery depends on the response map alone.",
      ],
    },
    assumptions: [
      "HOG features are invariant to illumination, so shape remains a reliable cue across frames.",
      "Color statistics of the target are consistent after initialization (justifies the never-updated histogram).",
      "Candidate states near the previous estimate contain the true target (single-frame search window).",
    ],
    computation:
      "Real-time: over 80 FPS. The filter branch runs in the Fourier domain; the histogram branch is a single dot product per candidate, so fusion adds negligible cost.",
    relations: [
      {
        to: "T001",
        type: "uses-as-baseline",
        note: "KCF is a direct comparison on OTB/VOT; Staple's filter branch uses the same Fourier-domain correlation-filter machinery introduced in KCF.",
      },
      {
        to: "T010",
        type: "uses-as-baseline",
        note: "SRDCF is compared as the strongest CF competitor (VOT2014 accuracy 0.600/15.90 fails) and is credited with excellent results at the cost of real-time speed — the gap Staple targets.",
      },
    ],
    concepts: ["correlation-filter", "appearance-features", "bounding-box", "sot", "success-plot"],
    impact:
      "Staple showed that learner diversity (shape + color) could be had almost for free, making it a standard CF-era baseline in ECO (T011) and BACF (T012); its frozen-histogram argument became the reference answer to template drift, and its response-fusion pattern reappears in later multi-cue trackers.",
  },
  {
    id: "T006",
    arxiv: "1603.00831",
    title: "MOT16: A Benchmark for Multi-Object Tracking",
    shortTitle: "MOT16",
    year: 2016,
    authors: [
      "Anton Milan",
      "Laura Leal-Taixé",
      "Ian Reid",
      "Stephan Roth",
      "Konrad Schindler",
    ],
    fileName: "1603.00831v2.pdf",
    task: "multi-object",
    tags: [
      "benchmark-design",
      "multi-object-tracking",
      "evaluation-protocol",
      "detection",
      "data-association",
      "ground-truth",
    ],
    difficulty: "intro",
    summary:
      "MOT16 is the successor benchmark to MOT15: 14 video sequences (7 train / 7 test) with pixel-accurate pedestrian annotations, providing a standardized protocol and metrics (MOTA, MOTP, ID measures, MT/ML) for multi-object tracking. It introduces more challenging sequences with denser crowds and harsher occlusions, and runs baseline tracking-by-detection systems showing large gaps between detectors (SDP vs DPM) — making detector quality the dominant factor in every ranking.",
    problem:
      "Multi-object tracking benchmarks were fragmented: datasets differed in label density, protocols and metrics, so methods could not be compared reliably. Existing benchmarks also had few difficult occlusion/crowd cases, and it was unclear how much of reported progress came from better trackers versus better detectors.",
    background: [
      "mot",
      "detection",
      "tracking-by-detection",
      "data-association",
      "occlusion",
      "bounding-box",
    ],
    previousWork: [
      {
        name: "MOT15 (Leal-Taixé et al.)",
        limitation:
          "Fewer sequences and sparser annotations; also lacked the density of occlusion and crowding needed to expose failure modes of modern trackers.",
        whyThisPaper:
          "MOT16 keeps the protocol but adds sequences with denser crowds, stronger occlusion and lower frame rates.",
      },
      {
        name: "TUD-Stadtmitte / ETH / Pettorsely-style earlier datasets",
        limitation:
          "Small, curated and with limited viewpoint/lighting variation; metrics were inconsistent across papers.",
        whyThisPaper:
          "A single standardized benchmark with clear annotation guidelines and a fixed evaluation script (CLEARMOT-style).",
      },
      {
        name: "Published tracker tables using different detectors",
        limitation:
          "Detector choice (DPM vs SDP vs R-CNN) dominates MOTA, making cross-paper comparisons meaningless.",
        whyThisPaper:
          "Runs every baseline with a common protocol and reports detector quality explicitly, so the tracker contribution is visible.",
      },
    ],
    researchGap:
      "There was no community benchmark that combined dense annotation, a fixed protocol, and publicly shared evaluation code with enough occlusion-heavy sequences to stress modern tracking-by-detection systems.",
    contribution: [
      "14 new sequences (7 train, 7 test) with pixel-accurate annotations: 476,532 total annotations across all sequences (the training split alone contains 206,938 person instances).",
      "Public train/test split, annotation guidelines and evaluation server with the standard metric set (MOTA, MOTP, ID measures, FAR, MT/ML, fragments).",
      "Five tracking-by-detection baselines re-evaluated under one protocol, isolating detector quality: SDP-based baselines roughly double MOTA over DPM-based ones.",
      "Analysis of annotation challenges: distractors (standing/occluded people near the true target) and relevance rules that tell systems which boxes count — explicitly acknowledging that distractor boxes lower IDSW but are not trivially removed.",
      "Discussion of metric shortcomings (MOTA alone is highly debatable; MOTP reflects detector quality more than tracker quality).",
    ],
    method: {
      pipeline: ["collect", "annotate", "protocol", "evaluate", "baseline"],
      architecture:
        "Benchmark pipeline rather than a tracker: sequence curation, dense pixel-level annotation with occlusion flags, train/test split, and an evaluation script computing CLEARMOT-derived metrics against a detection set.",
      detectionDependency:
        "All baselines are tracking-by-detection: the detector (DPM, SDP or a combination) supplies candidate boxes; the tracker associates them across frames.",
      optimization:
        "Evaluation matches predicted boxes to ground truth by IoU-based Hungarian assignment and accumulates TP/FP/FN, IDSW and fragments per sequence.",
    },
    equations: [
      {
        id: "mot16-mota",
        label: "Multiple Object Tracking Accuracy",
        formula: "MOTA = 1 - (sum(FN + FP + IDSW)) / sum(GT)",
        variables: [
          { symbol: "FN", meaning: "false negatives (missed ground-truth detections)" },
          { symbol: "FP", meaning: "false positives (spurious detections)" },
          { symbol: "IDSW", meaning: "identity switches relative to the ground truth" },
          { symbol: "GT", meaning: "number of ground-truth objects" },
        ],
        intuition:
          "Penalize every detection and identity error against the total number of true objects, then read accuracy as one minus that rate.",
        why:
          "MOTA is the headline metric of the benchmark; all baseline rankings are quoted with it.",
        where: "Section 4 (metrics), CLEARMOT formulation, standard MOT definition.",
        params: "Computed per sequence and averaged over the split.",
        simulator: "metrics",
        paperIds: ["T006"],
      },
      {
        id: "mot16-motp",
        label: "Multiple Object Tracking Precision",
        formula: "MOTP = sum_{i,j} d_{i,j} / sum_j c_j",
        variables: [
          { symbol: "d_{i,j}", meaning: "position distance (1 - IoU) between matched boxes" },
          { symbol: "c_j", meaning: "number of matched pairs for ground-truth object j" },
        ],
        intuition:
          "Average localization error over all matched detections — how tightly boxes sit on the objects, independently of coverage.",
        why:
          "Separates localization quality from recall; MOT16's discussion notes MOTP mostly reflects detector localization.",
        where: "Section 4, MOTP definition alongside MOTA.",
        simulator: "metrics",
        paperIds: ["T006"],
      },
      {
        id: "mot16-id-measures",
        label: "Normalized ID measures",
        formula: "rel-IDSW = IDSW / Recall ,  rel-FM = FM / Recall",
        variables: [
          { symbol: "IDSW", meaning: "identity switches" },
          { symbol: "FM", meaning: "fragmentation events (track broken then resumed)" },
          { symbol: "Recall", meaning: "recall of matched detections" },
        ],
        intuition:
          "Raw IDSW counts punish trackers that find more objects; normalizing by recall makes identity stability comparable across recall levels.",
        why:
          "Addresses the known criticism that absolute IDSW is not expressive when recalls differ between methods.",
        where: "Section 4, alongside the MT/ML (mostly tracked 80%+, mostly lost 20%-) definitions.",
        simulator: "metrics",
        paperIds: ["T006"],
      },
    ],
    datasets: ["mot16", "mot15"],
    metrics: ["mota", "motp", "idf1", "idsw", "fp", "fn", "mt-ml", "faf", "fps"],
    baselines: [
      "TBD (tracking-by-detection)",
      "CEM",
      "DP NMS",
      "SMOT",
      "JPDA M",
    ],
    results: [
      "476,532 annotations across the 14 sequences; the training split has 206,938 person instances, giving dense occlusion and crowding cases.",
      "Test-set baselines (MOTA / MOTP / FAR / MT / ML / speed): TBD 33.7 / 76.5 / 1.0 / 7.2% / 54.2% / 1.3 Hz; CEM 33.2 / 76.0 / 8.0 / 5.7% / 54.7% / 1.7 Hz; DP NMS 32.2 / 77.7 / 6.7 / 5.4% / 48.0% / 212.6 Hz; SMOT 29.7 / 75.5 / 9.3 / 4.2% / 52.4% / 16.6 Hz; JPDA M 26.2 / 74.2 / 14.6 / 2.9% / 53.6% / 6.5 Hz.",
      "Detector effect: replacing DPM detections with SDP roughly doubles MOTA (33.7 -> 68.9 for the TBD system) — detector quality dominates the ranking.",
      "47 methods and over 180 registered users had used the earlier benchmark as of October 2014, establishing the community protocol MOT16 extends.",
    ],
    ablations: [
      "Baselines evaluated with three detection qualities (DPM, SDP, and combination) to quantify how much of MOTA is detector-driven.",
      "Train vs test split results reported separately for each baseline to expose overfitting to the public split.",
      "Per-sequence analysis of distractor/relevance handling: with distractor boxes removed, IDSW falls (TBD 781 -> 684) while MOTA barely moves (49.3 -> 49.7), showing IDSW is sensitive to annotation policy.",
    ],
    limitations: {
      authorStated: [
        "MOTA is 'highly debatable' as a single summary metric; MOTP mainly reflects the detector rather than the tracker.",
        "Absolute IDSW is 'not expressive' without normalization — hence rel-IDSW and rel-FM are also defined.",
        "Annotation of crowded, occluded pedestrians required judgment calls (relevance/distractor rules), so some ground-truth ambiguity remains.",
      ],
      evident: [
        "The benchmark evaluates pedestrian tracking only; other object classes require new annotation.",
        "Since all baselines are tracking-by-detection, the benchmark says little about detection-free/online single-camera trackers.",
      ],
    },
    assumptions: [
      "Every sequence is a fixed camera view of pedestrians with consistent annotation conventions.",
      "IoU-based matching at 50% threshold plus CLEARMOT accumulation is a fair basis for comparison.",
      "Distractor boxes (occluded/non-relevant people) may be present in detections and must be handled by the protocol, not by hand-cleaning.",
    ],
    computation:
      "Evaluation is cheap (Hungarian matching per frame); the fastest baseline (DP NMS) runs at 212.6 Hz, the slowest (TBD) at 1.3 Hz — tracking-by-detection cost is dominated by the detector.",
    relations: [],
    concepts: [
      "mot",
      "benchmark-design",
      "tracking-by-detection",
      "data-association",
      "detection",
      "mota",
      "occlusion",
    ],
    impact:
      "MOT16 became the standard MOT benchmark of the deep-tracking era — its protocol and metric set (MOTA/MOTP/IDF1-adjacent measures) are what DeepSORT (T013) and most later MOT papers report, and its detector-dominates-MOTA finding reframed how the community reads tracking leaderboards.",
  },
  {
    id: "T007",
    arxiv: "1604.01802",
    title: "Learning to Track at 100 FPS with Deep Regression Networks",
    shortTitle: "GOTURN",
    year: 2016,
    authors: ["David Held", "Sebastian Thrun", "Silvio Savarese"],
    fileName: "1604.01802v2.pdf",
    task: "single-object",
    tags: [
      "regression-network",
      "offline-training",
      "tracking-by-detection",
      "motion-model",
      "real-time",
    ],
    difficulty: "intermediate",
    summary:
      "GOTURN is the first tracker that learns to track purely offline: a deep regression network takes the previous-frame crop and a search region of the current frame and regresses the new bounding box directly. Because nothing is updated online (no filter learning, no hyper-parameter tuning per video), inference is a single forward pass — 100 FPS on a GPU (2.7 FPS on CPU) — while reaching the best overall ranking on VOT2014 among 39 methods.",
    problem:
      "Existing trackers adapt their model online, which costs speed and forces per-sequence hyper-parameter choices; training tracking systems online from scratch is hard, and it was unclear whether a purely offline-learned model could generalize across arbitrary objects without any test-time adaptation.",
    background: [
      "siamese",
      "appearance-features",
      "online-vs-offline",
      "bounding-box",
      "sot",
      "motion-model",
    ],
    previousWork: [
      {
        name: "Struck (Hare et al.)",
        limitation:
          "Structural-output SVM with online updates: only about 10 FPS and needs careful per-sequence settings; tracking-by-detection approaches make strong assumptions about objectness.",
        whyThisPaper:
          "Replaces online learning with an offline-trained regression so no per-sequence optimization happens at test time.",
      },
      {
        name: "Correlation-filter trackers (KCF/T001, SRDCF/T010)",
        limitation:
          "The CF family was 'unable to cope with scale changes and extreme object deformations'; their template-based object model is fitted frame-to-frame rather than learned for tracking.",
        whyThisPaper:
          "Learns features for detection in a generic way and regresses the box end-to-end, trained only offline.",
      },
      {
        name: "MDNet (T003)",
        limitation:
          "State-of-the-art accuracy but online fine-tuning makes it run at only about 1 FPS.",
        whyThisPaper:
          "GOTURN targets the speed end of the accuracy-speed trade-off: same offline data advantage, no online updates.",
      },
    ],
    researchGap:
      "No prior work had shown that a deep network could learn generic tracking ability from offline video alone and then track at frame rate without any online model update.",
    contribution: [
      "A regression formulation of single-object tracking: given a fixed-size search region centred on the previous position, regress the target's new bounding box — tracking becomes a regression problem trained offline.",
      "Fully offline training pipeline combining tracking pairs from ImageNet Video with single-frame localization supervision from ImageNet detection, avoiding online hyper-parameter fitting.",
      "A motion-smoothing augmentation (Laplace-sampled box jitter) that models target motion and cuts combined tracking errors by about 20%.",
      "100 FPS (9.98 ms per frame) on GPU and 2.7 FPS on CPU — the first tracker faster than real time among deep trackers.",
      "Best overall VOT2014 rank (8.21 among 39) with best accuracy rank (5.54) while running two orders of magnitude faster than MDNet-class trackers.",
    ],
    method: {
      pipeline: ["crop-prev", "crop-search", "encode", "fuse", "regress"],
      architecture:
        "Two input branches (target region from the previous frame, search region of the current frame, both resized to a fixed grid) whose feature maps are concatenated along channels and fed to fully connected layers that regress (dx, dy) center displacement and scale change (dw, dh); crops are rescaled by a constant factor k1=k2=2 around the previous box.",
      motionModel:
        "Learned motion prior: training examples are perturbed with Laplace-distributed box displacements, and a constant-position model c'=c is used at test time around the previous center — the network itself learns where targets tend to go.",
      appearanceModel:
        "Generic deep features learned offline on ImageNet Video + ImageNet detection; the network compares previous-frame appearance against the current search region.",
      detectionDependency:
        "Not a detection-pipeline tracker, but trained with ImageNet detection crops as auxiliary localization data.",
      trackManagement:
        "No explicit track state: a single box is carried frame-to-frame; the paper lists occlusion and fast motion as out of scope for the learned regressor.",
      loss: "L1 regression loss between predicted and ground-truth box coordinates (ablated against L2).",
      optimization:
        "Offline SGD with momentum and step-wise learning-rate decay; no test-time optimization of any kind.",
    },
    equations: [
      {
        id: "goturn-box",
        label: "Bounding-box motion model",
        formula: "c'_x = c_x + w * dx ,  c'_y = c_y + h * dy",
        variables: [
          { symbol: "c_x, c_y", meaning: "center of the bounding box in the previous frame" },
          { symbol: "w, h", meaning: "width and height of the previous box" },
          { symbol: "dx, dy", meaning: "regressed displacements from the deep network (Laplace-distributed in training)" },
          { symbol: "c'_x, c'_y", meaning: "predicted center in the current frame" },
        ],
        intuition:
          "Express where the object moves as a fraction of its own size — the network predicts a scale-normalized offset, so absolute pixel speed does not matter.",
        why:
          "It is the tracker's motion model: displacements sampled from a Laplace distribution during training teach the network typical target motion without any explicit motion filter.",
        where: "Bounding-box motion model section, Eqs. 1-2 (identical form for height/scale).",
        params:
          "Crop scale k1 = k2 = 2 (target and search crops are twice the previous box); Laplace sampling used only during training.",
        paperIds: ["T007"],
      },
      {
        id: "goturn-loss",
        label: "Regression loss",
        formula: "L = sum over box coordinates of |pred - truth|   (L1)   or squared error (L2)",
        variables: [
          { symbol: "pred", meaning: "regressed box parameters" },
          { symbol: "truth", meaning: "ground-truth box of the current frame" },
        ],
        intuition:
          "Supervise the network to land exactly on the annotated box; L1 penalizes errors linearly and is far more robust than L2 here.",
        why:
          "Ablation shows L2 loss degrades combined accuracy/robustness from 0.43 to 0.24 overall (0.69 to 0.39 accuracy), so the L1 form is the paper's chosen objective.",
        where: "Training objective section (loss formulation) and the loss ablation table.",
        params: "Single-frame regression labels come from ImageNet detection boxes; video pairs from ImageNet Video.",
        paperIds: ["T007"],
      },
      {
        id: "goturn-smooth",
        label: "Motion-smoothing augmentation",
        formula: "dx, dy ~ Laplace(0, b) applied to ground-truth boxes during training",
        variables: [
          { symbol: "b", meaning: "Laplace scale controlling jitter magnitude" },
          { symbol: "dx, dy", meaning: "synthetic displacements added to the training box" },
        ],
        intuition:
          "Randomly jitter where the target 'was' so the network learns a distribution of plausible motions instead of memorizing a static crop.",
        why:
          "The ablation shows training with motion-smoothing reduces combined errors substantially (about 20% relative) versus image-only or video-only training without jitter.",
        where: "Motion-smoothing subsection of the training pipeline.",
        paperIds: ["T007"],
      },
    ],
    datasets: ["otb", "vot"],
    metrics: ["success-auc", "precision", "eao", "fps"],
    baselines: ["KCF", "Struck", "DSST", "MDNet", "SINT", "LCT", "TLD"],
    results: [
      "VOT2014 overall accuracy/robustness rank 8.21 of 39 methods — best overall; accuracy rank 5.54 (best), robustness rank 10.87.",
      "VOT2014 comparisons (accuracy/robustness ranks): KCF 10.37, DSST 9.19, Struck 21.04, TLD present, along with ThunderStruck, NCC, IVT and FoT in the same table.",
      "Speed: 9.98 ms per frame = 100 FPS on GPU; 2.7 FPS on CPU.",
      "Training data study: increasing ImageNet Video sequences from 14 to 37 to 157 to 307 steadily improves tracking; combined with 478,807 ImageNet detection objects for localization supervision.",
      "Robustness: motion-smoothness regularization lowers combined errors from about 0.43/0.39/0.17-style breakdowns to the reported 0.30 with both image and video supervision.",
    ],
    ablations: [
      "Loss: L2 vs L1 — overall 0.43 -> 0.24 (accuracy 0.69 -> 0.39, robustness 0.17 -> 0.10).",
      "Training source: image-only (0.35), video-only (0.29), video with motion smoothing (0.30), best combined image+video configuration (0.24 overall).",
      "Dataset size: 14 / 37 / 157 / 307 ImageNet Video sequences — accuracy improves monotonically with more pretraining videos.",
    ],
    limitations: {
      authorStated: [
        "Occlusion and fast motion are explicitly out of scope; combining the regressor with a detection module and an attention mechanism is listed as future work for long-term tracking.",
        "At test time a constant-position assumption (c' = c) is used — no explicit motion filter or failure detection.",
        "Comparison with Struck is described as not entirely fair because GOTURN cannot (yet) cope with the full difficulty of arbitrary-video tracking as that work defines it.",
      ],
      evident: [
        "Trained on ImageNet/ILSVRC categories, so domain shift to unusual target types is untested in the paper.",
        "No scale search or re-detection: after a full miss the crop centre simply follows the previous box.",
      ],
    },
    assumptions: [
      "The target remains inside the search region centred on its previous position (c' = c constant-position model).",
      "Offline supervision from detection images transfers to the general tracking task.",
    ],
    computation:
      "One forward pass per frame: 9.98 ms (100 FPS) on GPU, about 0.37 s (2.7 FPS) on CPU; training is fully offline.",
    relations: [
      {
        to: "T001",
        type: "uses-as-baseline",
        note: "KCF appears as a compared method in the VOT2014 accuracy/robustness table, representing the fast correlation-filter trackers GOTURN trades against.",
      },
      {
        to: "T003",
        type: "addresses-limitation",
        note: "Cites MDNet (reference [30]) as the accuracy leader whose online fine-tuning costs about 1 FPS; GOTURN removes test-time learning entirely to reach 100 FPS.",
      },
    ],
    concepts: ["siamese", "appearance-features", "online-vs-offline", "bounding-box", "motion-model", "sot"],
    impact:
      "GOTURN established that tracking knowledge can be learned entirely offline and distilled into one forward pass — the template for the siamese wave that followed (SINT T008, SiamFC T009), and its accuracy-speed curve (100 FPS at VOT2014-comparable accuracy) became the reference plot for real-time deep tracking.",
  },
  {
    id: "T008",
    arxiv: "1605.05863",
    title: "Siamese Instance Search for Tracking",
    shortTitle: "SINT",
    year: 2016,
    venue: "CVPR 2016",
    authors: ["Ran Tao", "Efstratios Gavves", "Arnold W.M. Smeulders"],
    fileName: "1605.05863v1.pdf",
    task: "single-object",
    tags: [
      "siamese",
      "instance-search",
      "offline-training",
      "ranking-loss",
      "appearance-features",
    ],
    difficulty: "intermediate",
    summary:
      "SINT casts tracking as instance search: a Siamese network trained offline with a ranking loss compares the first-frame target patch against a dense grid of candidates in every new frame and returns the most similar one. The model never adapts after initialization, is trained on ALOV300 with strong scale and displacement sampling, and matches or beats MUSTer/MEEM on OTB (AUC 62.5 OPE) while running at about 2-3 FPS — notable for being trained with an explicit discriminative ranking objective rather than classification or regression.",
    problem:
      "Tracking needs to compare the appearance of the target seen at initialization with appearances in later frames across shifts, scales and deformations. Hand-crafted similarity measures drift, and it was unclear whether a purely offline-learned deep similarity function — trained on thousands of displaced patch pairs — could generalize to unseen target instances without any online update.",
    background: [
      "siamese",
      "appearance-features",
      "online-vs-offline",
      "iou",
      "bounding-box",
      "sot",
      "detection",
    ],
    previousWork: [
      {
        name: "Online trackers (Struck, TLD, MDNet-style fine-tuning)",
        limitation:
          "Adapt online per sequence: flexible but slow and prone to drift or to careful per-dataset tuning.",
        whyThisPaper:
          "SINT fixes the model after offline training — no hyper-parameter or model fitting per video.",
      },
      {
        name: "KCF-style correlation filters",
        limitation:
          "Fast but trained on cyclic-shift crops of a single template; their similarity measure is shallow and degrades with large appearance/scale change.",
        whyThisPaper:
          "Replaces the filter similarity with a learned deep feature space in which cosine similarity between patches identifies instances.",
      },
      {
        name: "Previous siamese/instance-search formulations",
        limitation:
          "Either trained for binary similarity without scale handling, or not evaluated at the dense, multi-scale search required by tracking.",
        whyThisPaper:
          "Trains on randomly displaced and scaled patch pairs and performs dense search over a grid of 10 radial x 10 angular candidate locations at three scales.",
      },
    ],
    researchGap:
      "Before SINT nobody had demonstrated an offline-trained deep similarity function for tracking that performs dense instance search across translation and scale with no test-time learning.",
    contribution: [
      "Formulates tracking as instance search with a Siamese network: score every candidate patch of the current frame against the first-frame patch by cosine similarity of learned features, take the argmax.",
      "Trains with an extended ranking loss (positive pairs forced above negative pairs by a margin, smooth-L2 distance term) on tens of thousands of patch pairs from ALOV300.",
      "Dense multi-scale candidate generation (10 radial x 10 angular displacements at 3 scaled versions per location) with box regressors fitted on the first frame to refine width/height.",
      "State-of-the-art OTB performance at the time: OPE AUC 62.5 / precision 84.8, TRE 64.3/84.9, SRE 57.9/80.6 — beating MEEM and comparable with MUSTer.",
      "Cross-dataset evidence on YouTube: mean AUC 58.1 vs MUSTer 37.8 and MEEM 29.4 across six hard sequences.",
    ],
    method: {
      pipeline: ["init", "sample", "encode", "dense-score", "select", "refine"],
      architecture:
        "Two weight-shared branches: the first-frame target patch and a candidate patch are each passed through an AlexNet/VGG-style CNN to a global feature vector; the cosine similarity of the two vectors is the score. Four ridge regressors (x, y, w, h) trained from the first frame refine the winning box.",
      motionModel:
        "None: candidates are laid on a fixed displacement grid around the previous position, with an optional fixed center-constant bias.",
      appearanceModel:
        "Offline-learned deep feature space (ranking loss), compared by cosine similarity — no online template update at all.",
      detectionDependency:
        "None — an appearance-only instance-search tracker.",
      trackManagement:
        "No explicit state machine; occlusion handling is by design absent — the tracker always picks the globally most similar patch.",
      loss:
        "Ranking loss: for a positive pair (x_j) and negatives (x_k), minimize half*y*D^2 + half*(1-y)*max(0, epsilon - D^2) with D the feature distance, enforcing a margin between matched and unmatched instances.",
      optimization:
        "Offline SGD initialized from an ImageNet classification net; learning rate 0.001 with step decay and batch-schedule decay; trained on 60,000 frame pairs x 128 box pairs per epoch with an internal validation split of 2,000 x 128.",
    },
    equations: [
      {
        id: "sint-ranking",
        label: "Pairwise ranking loss",
        formula: "L = 1/2 * y * D^2 + 1/2 * (1 - y) * max(0, eps - D^2),  D = ||f(x_j) - f(x_k)||_2",
        variables: [
          { symbol: "f", meaning: "shared feature extractor of the Siamese network" },
          { symbol: "y", meaning: "label: 1 for a matched (same object) patch pair, 0 otherwise" },
          { symbol: "D", meaning: "Euclidean distance between the two feature vectors" },
          { symbol: "eps", meaning: "margin — negatives must stay eps away from positives" },
        ],
        intuition:
          "Pull features of the same instance together and push different instances apart by at least a margin, so cosine similarity at test time separates target from background.",
        why:
          "The whole tracker depends on this learned similarity; classification losses would only separate target/background, not rank candidate instances by likeness.",
        where: "Training objective section, pairwise loss formulation.",
        params:
          "60,000 frame pairs, each with 128 positive box pairs (IoU > 0.7) and negatives with IoU < 0.5.",
        paperIds: ["T008"],
      },
      {
        id: "sint-search",
        label: "Instance-search scoring",
        formula: "x* = argmax_{x in candidates} m(x_target, x),  m(a, b) = f(a)^T f(b)",
        variables: [
          { symbol: "x_target", meaning: "first-frame target patch (never replaced)" },
          { symbol: "candidates", meaning: "dense grid of displaced and scaled patches of the current frame" },
          { symbol: "m", meaning: "cosine similarity of the learned features (vectors normalized)" },
          { symbol: "x*", meaning: "selected target location for the current frame" },
        ],
        intuition:
          "Score every plausible crop in the current frame against the initial target and keep the most similar one — tracking as retrieval.",
        why:
          "This is the tracker itself: no filter update, no regression head, just a global similarity argmax over a grid.",
        where: "Method section, instance-search formulation; candidates sampled at 10 radial x 10 angular offsets x 3 scales.",
        params:
          "Four ridge regressors fit on the first frame map the winning box to width/height; center held constant as a default prior.",
        paperIds: ["T008"],
      },
    ],
    datasets: ["otb", "others"],
    metrics: ["success-auc", "precision"],
    baselines: ["MUSTer", "MEEM", "KCF", "Struck", "TLD", "Staple"],
    results: [
      "OTB (one-pass): OPE AUC 62.5 / precision@20 84.8; TRE 64.3/84.9; SRE 57.9/80.6 — ahead of MEEM (57.2/84.0) and comparable to MUSTer (62.1/83.6).",
      "Ablation of features (Table): fixed pretrained features give 44.0 AUC / 67.2 precision, while the fine-tuned Siamese feature set reaches 59.2/83.6 (with box regressors and no max-pooling).",
      "YouTube evaluation (six sequences): mean AUC SINT 58.1, MUSTer 37.8, MEEM 29.4; per-sequence examples: Fishing 53.7, Rally 53.4, Soccer 72.5.",
      "Attribute analysis: SINT outperforms in 6 of 11 attributes by AUC and 7 of 11 by precision.",
      "Training corpus: ALOV300 minus 12 videos overlapping with OTB; first-frame supervision from the same videos with scale/translation augmentation.",
    ],
    ablations: [
      "Feature training: off-the-shelf vs fine-tuned Siamese features (44.0 -> 59.2 AUC), showing the ranking objective is what creates the tracking similarity.",
      "Box regressors on/off and max-pooling variants — the final configuration (Siamese + regressors + fc without max-pool) is the best in the table.",
      "Positive/negative sampling thresholds (IoU > 0.7 vs < 0.5) and scale augmentation controls the difficulty of the learned similarity.",
    ],
    limitations: {
      authorStated: [
        "Failure cases (Figure): jumps to a similar object (Bolt — another runner in the same uniform) and breaks under large occlusion (Lemming — hidden by a lighter), both because there is no occlusion detection by design.",
        "The model does not re-detect after losing the target: once the argmax leaves the true object, nothing pulls it back.",
      ],
      evident: [
        "Runs at only about 2-3 FPS: dense multi-scale grid scoring with a deep network is expensive (no speed figure is printed for SINT itself; timings come from the OTB runs).",
        "Fixed first-frame template: long-term appearance change cannot be absorbed without online adaptation.",
      ],
    },
    assumptions: [
      "The first-frame patch is the definitive appearance of the target (single reference instance).",
      "The true target is inside the candidate grid each frame (no re-detection outside the search region).",
    ],
    computation:
      "One deep forward pass per candidate crop over a dense grid each frame — offline training only, but dense multi-scale search keeps test-time cost at a few FPS.",
    relations: [],
    concepts: ["siamese", "appearance-features", "online-vs-offline", "sot", "bounding-box", "iou"],
    impact:
      "SINT anticipated the siamese line one step before SiamFC (T009): it proved offline-learned deep similarity alone can track competitively on OTB and YouTube, and its ranking-loss instance-search framing directly influenced siamese trackers that followed, while its failure cases (distractor instances, no occlusion handling) became the standard cautionary examples.",
  },
  {
    id: "T009",
    arxiv: "1606.09549",
    title: "Fully-Convolutional Siamese Networks for Object Tracking",
    shortTitle: "SiamFC",
    year: 2016,
    authors: [
      "Luca Bertinetto",
      "Jack Valmadre",
      "João F. Henriques",
      "Andrea Vedaldi",
      "Philip H.S. Torr",
    ],
    fileName: "1606.09549v3.pdf",
    task: "single-object",
    tags: [
      "siamese",
      "fully-convolutional",
      "offline-training",
      "correlation-layer",
      "real-time",
      "detection",
    ],
    difficulty: "intro",
    summary:
      "SiamFC replaces the recurrent online learning of classic trackers with a single similarity measure learned offline: a fully-convolutional Siamese network embeds the 127x127 exemplar and the 255x255 search region of the current frame, and their cross-correlation gives a score map whose argmax localizes the target. The model never updates, runs at 58-86 FPS, and matched the best AUC on OTB-13 while introducing the recipe (dense binary labels, ImageNet Video training, three-scale search) that essentially all real-time siamese trackers copy.",
    problem:
      "Tracking-by-detection and template-update trackers spend enormous effort learning online, with per-dataset hyper-parameters and constant risk of drift. It was open whether a fixed similarity function, trained once offline on thousands of video pairs, could compare an initial target template with later frames accurately enough to track arbitrary objects — without any online learning or dataset-specific tuning.",
    background: [
      "siamese",
      "correlation-filter",
      "appearance-features",
      "online-vs-offline",
      "bounding-box",
      "iou",
      "sot",
    ],
    previousWork: [
      {
        name: "TLD, Struck, KCF (T001)",
        limitation:
          "The paper opens by noting the deficiency of trackers that only learn online — KCF and friends adapt a template during the video, which is what causes drift and per-sequence tuning.",
        whyThisPaper:
          "Learns everything offline; no update rule exists at test time, removing drift by construction.",
      },
      {
        name: "GOTURN (T007)",
        limitation:
          "A regression approach requiring paired crops as input and a fully connected head sized to a fixed box layout.",
        whyThisPaper:
          "SiamFC is fully convolutional: a single convolution yields the whole score map, so search regions can be arbitrary size and boxes come from a correlation peak instead of regression.",
      },
      {
        name: "Classic correlation filters",
        limitation:
          "Filters are trained online on small hand-crafted patches (HOG/pixels) with limited semantic power.",
        whyThisPaper:
          "Keeps the correlation machinery (cross-correlation of a learned filter) but moves the filter learning entirely offline into a deep network.",
      },
    ],
    researchGap:
      "No one had yet shown that a single offline-learned convolutional similarity, evaluated as cross-correlation, is sufficient for tracking at frame rate across arbitrary object types and datasets.",
    contribution: [
      "Fully-convolutional Siamese architecture: exemplar and search region are embedded separately and scored by a cross-correlation layer plus learned bias, producing a dense score map in one forward pass.",
      "Training on dense labels: for every candidate location in the search region a binary label (inside radius R of the true center = positive) supervises the score map with logistic loss — turning tracking into dense prediction.",
      "Training corpus: 4,417 ImageNet Video sequences, more than 2 million tracked objects, 50 epochs over 50,000 pairs per epoch.",
      "Three-scale search with a temporally damped scale factor (base 1.025, damping 0.35) and low-resolution score maps upsampled bicubically — cheap scale handling without extra training.",
      "State-of-the-art real-time results: OTB-13 OPE AUC 61.2 (58 FPS), VOT2015 EAO 0.524 (SiamFC) / 0.534 (SiamFC-3s at 86 FPS), VOT2016 EAO 0.388/0.405.",
    ],
    method: {
      pipeline: ["embed-exemplar", "embed-search", "correlate", "upsample", "select"],
      architecture:
        "AlexNet-like backbone with the final max-pool removed: two weight-shared branches produce feature maps; a correlation layer computes all pairwise inner products between 6x6 exemplar features and 22x22 search features, followed by a bias and a 22x22 output score map.",
      motionModel:
        "Center-prior only: the search region is a square around the previous position; scale is handled by re-scoring at 5 scale factors each frame and damping the change.",
      appearanceModel:
        "Fixed offline-learned deep similarity — the exemplar from frame 1 (or previous frame) is never re-estimated from model updates.",
      detectionDependency:
        "None — a detection-free single-object tracker; candidates are all locations of the search region.",
      trackManagement:
        "No explicit state management; the paper flags as limitation that the model is never updated, making it sensitive to scenes with confusion.",
      loss:
        "Logistic loss on dense score-map labels: ell(y, v) = log(1 + exp(-y*v)) summed over all candidate locations with labels determined by a Gaussian radius R.",
      optimization:
        "Plain SGD: momentum 0.9, learning rate 1e-2 decaying geometrically to 1e-5, batch of 8 pairs, 50 epochs x 50,000 pairs, initialized from ImageNet classification weights.",
    },
    equations: [
      {
        id: "siamfc-equivariance",
        label: "Translation equivariance of the network",
        formula: "h(L_T^k * x) = L_T^k * h(x)",
        variables: [
          { symbol: "h", meaning: "the convolutional embedding (feature map)" },
          { symbol: "L_T^k", meaning: "translation operator that shifts image x by vector k" },
          { symbol: "*", meaning: "applies the shift to the feature map" },
        ],
        intuition:
          "Shifting the input shifts the features identically, so a template filter correlates with the search map consistently everywhere.",
        why:
          "It justifies computing the response for all positions at once by cross-correlation rather than running the network on every cropped window.",
        where: "Section 2, property of fully-convolutional networks, Eq. 1.",
        paperIds: ["T009"],
      },
      {
        id: "siamfc-score",
        label: "Siamese score function",
        formula: "f(z, x) = phi(z) (*) phi(x) + b1",
        variables: [
          { symbol: "z", meaning: "exemplar (target template) patch, 127x127" },
          { symbol: "x", meaning: "search region of the current frame, 255x255" },
          { symbol: "phi", meaning: "shared convolutional embedding" },
          { symbol: "(*)", meaning: "cross-correlation operator" },
          { symbol: "b1", meaning: "learned per-location bias" },
        ],
        intuition:
          "One correlation of the two embeddings yields a score map: high values where the template is found.",
        why:
          "This single expression is the entire tracker — cheap enough for 58 FPS yet learned end-to-end offline.",
        where: "Section 2, model formulation, Eq. 2.",
        params:
          "Exemplar 127x127 from the previous target, search 255x255 (about 4x the exemplar context), output 17x17 score map upsampled to 272x272.",
        simulator: "corr-filter",
        paperIds: ["T009"],
      },
      {
        id: "siamfc-loss",
        label: "Dense logistic score-map loss",
        formula: "L = (1/|D|) * sum_{u in D} log(1 + exp(-y_u * v_u))",
        variables: [
          { symbol: "v_u", meaning: "predicted score at location u of the score map" },
          { symbol: "y_u", meaning: "binary label (+1 if within radius R of the true target center, else -1)" },
          { symbol: "D", meaning: "set of all score-map locations" },
        ],
        intuition:
          "Supervise every location of the output map at once: locations near the true box should be positive, all others negative.",
        why:
          "Dense supervision is what lets a single offline training session learn localization — there is no separate bounding-box regression or online fitting.",
        where: "Section 3, Eqs. 3-4; label radius R given by Eq. 6.",
        params:
          "Labels from a Gaussian of radius R around the true center; balanced positive/negative implicitly through the dense map.",
        paperIds: ["T009"],
      },
    ],
    datasets: ["otb", "vot"],
    metrics: ["success-auc", "precision", "eao", "fps"],
    baselines: ["Staple", "MDNet", "KCF", "LCT", "CCT", "SRDCF", "GOTURN", "MEEM"],
    results: [
      "OTB-13 OPE AUC: SiamFC 61.2, SiamFC-3s 60.8, Staple 60.0, CCT 60.5, LCT 61.2, KCFDP 58.1; TRE SiamFC 62.1 vs Staple 61.7; SRE SiamFC 55.4 vs Staple 54.5.",
      "VOT2015: SiamFC-3s accuracy 0.5335 / 84 failures / overlap 0.2889 at 86 FPS; SiamFC 0.5240 / 87 failures / 0.2743 at 58 FPS; versus MDNet 0.5620/46/0.3575 at 1 FPS, SRDCF 0.5260/71, DeepSRDCF 0.5350/60.",
      "VOT2016 EAO: SiamFC 0.3876, SiamFC-3s 0.4051.",
      "Dataset-size ablation (VOT2015): training on 2% (88 videos, ~60k objects) gives EAO 0.168 with 183 failures, rising to 0.274 with all 4,417 videos (~2M objects, 87 failures).",
      "Training throughput: 50 epochs x 50,000 pairs; ImageNet Video frames plus ImageNet detection objects as the source pool.",
    ],
    ablations: [
      "Training-set size: 2%, 10%, 25%, 50%, 100% subsets — VOT2015 EAO climbs from 0.168 (88 videos) to 0.274 (4,417 videos), showing data volume drives siamese tracking accuracy.",
      "Three-scale search vs single-scale: SiamFC-3s vs SiamFC on VOT2015 (0.5335 vs 0.5240 accuracy) and VOT2016 (0.4051 vs 0.3876 EAO).",
      "Ablation of context: performance with and without the enlarged search region padding and with/without the final max-pool removal (design choices discussed in the appendix).",
    ],
    limitations: {
      authorStated: [
        "The model is never updated — unlike online trackers it cannot adapt to scenes with confusing similar objects (explicitly listed as limitation).",
        "No bounding-box regression or detection refinement: localization granularity is set by the score-map resolution.",
        "Scale handling is a search-time heuristic (5 discrete scale factors with damping), not learned by the network.",
      ],
      evident: [
        "Struggles with fast motion/abrupt scale change because the search region is anchored to the previous frame.",
        "Failure on similar-looking distractors shares the instance-confusion weakness SINT documented.",
      ],
    },
    assumptions: [
      "A similarity learned from ImageNet Video transfers to arbitrary unseen object categories.",
      "The target stays within the search region between consecutive frames (no global re-detection).",
    ],
    computation:
      "One forward pass with a correlation layer per frame: 58 FPS (SiamFC) and 86 FPS (SiamFC-3s, three scales) on GPU; training is offline SGD over 50 x 50,000 pairs.",
    relations: [
      {
        to: "T001",
        type: "addresses-limitation",
        note: "The paper opens by citing the deficiency of online-only learning (TLD, Struck, KCF/T001) — SiamFC removes online updates entirely, at the cost of never adapting.",
      },
      {
        to: "T003",
        type: "uses-as-baseline",
        note: "MDNet is compared on OTB-13 and VOT2015 (accuracy 0.5620, 46 failures, 0.3575 overlap at ~1 FPS) as the accuracy-leading online tracker.",
      },
      {
        to: "T004",
        type: "uses-as-baseline",
        note: "Staple is the principal CF-era competitor on OTB-13 (OPE AUC 60.0 vs SiamFC 61.2) and appears throughout the comparison tables.",
      },
      {
        to: "T007",
        type: "uses-as-baseline",
        note: "GOTURN (reference [28]) is compared as the prior offline-learned tracker; SiamFC's fully-convolutional correlation head is the alternative to GOTURN's regression head.",
      },
      {
        to: "T010",
        type: "uses-as-baseline",
        note: "SRDCF and DeepSRDCF are listed among the state-of-the-art trackers in the VOT2015/OTB comparisons (DeepSRDCF 0.5350 accuracy).",
      },
    ],
    concepts: ["siamese", "appearance-features", "correlation-filter", "online-vs-offline", "sot", "bounding-box"],
    impact:
      "SiamFC is the ur-siamese tracker: its cross-correlation head, dense-label training, ImageNet Video pipeline and three-scale search were copied by nearly every real-time tracker that followed (SiamRPN++ T028, SiamCAR T036, SiamFC++ T035 and beyond), and it made 'learn a similarity offline, never update online' the dominant paradigm of real-time tracking.",
  },
  {
    id: "T010",
    arxiv: "1608.05571",
    title: "Learning Spatially Regularized Correlation Filters for Visual Tracking",
    shortTitle: "SRDCF",
    year: 2016,
    authors: [
      "Martin Danelljan",
      "Gustav Häger",
      "Fahad Shahbaz Khan",
      "Michael Felsberg",
    ],
    fileName: "1608.05571v1.pdf",
    task: "single-object",
    tags: [
      "correlation-filter",
      "spatial-regularization",
      "background-modeling",
      "adaptive-sampling",
      "deformation",
      "hough",
    ],
    difficulty: "advanced",
    summary:
      "SRDCF fixes the central weakness of correlation filters — the circular-shift assumption that treats background border pixels as part of the target — by weighting regularization spatially: a target-shaped weight map penalizes filter coefficients where background dominates and keeps them free over the object. The resulting Fourier-domain objective is optimized by a small number of Gauss-Newton/gradient iterations, lifting the CF family to the top of OTB-2013/2015, ALOV++ and VOT2014 while running at about 5 FPS.",
    problem:
      "Correlation filters learn from cyclic shifts of a small patch, so filter coefficients at background locations are contaminated by the wrap-around of the patch border. Discriminative filters must therefore suppress background responses, but that also shrinks the usable training region and hurts robustness to deformation — filters degraded when trained only on the object itself, while expanding the region without care floods training with background.",
    background: [
      "correlation-filter",
      "bounding-box",
      "appearance-features",
      "online-vs-offline",
      "iou",
      "sot",
      "occlusion",
    ],
    previousWork: [
      {
        name: "KCF/DCF (T001) and MIL/Spotlight-style cyclic training",
        limitation:
          "The circular-shift model corrupts filter learning with wrapped background; discarding non-target regions (as some variants do) loses the very background context that makes filters discriminative.",
        whyThisPaper:
          "Keeps the enlarged region but regularizes it with a spatial weight: free coefficients over the target, shrunk coefficients over background.",
      },
      {
        name: "Hough-style correlation trackers",
        limitation:
          "The correlation filter tends to degrade rapidly if trained only on the object itself, and background modeling was missing.",
        whyThisPaper:
          "SRDCF explicitly models background through spatially dependent regularization rather than a hand-crafted background feature.",
      },
      {
        name: "Prior CF datasets limited to 1-2 video pairs (benchmark work)",
        limitation:
          "Training on only one or two image pairs does not cover the range of appearances a filter must handle.",
        whyThisPaper:
          "Introduces a unified framework trained on more than 100 image pairs sampled across sequences (spatial + temporal sampling).",
      },
    ],
    researchGap:
      "Before SRDCF, no correlation-filter formulation allowed different regularization strength at different spatial locations, so neither the background-contamination nor the scale-degradation problem had a principled solution.",
    contribution: [
      "Spatially dependent regularization of the filter: a target-overlapping weight map w scales per-pixel regularization, protecting coefficients over the object and damping those over background (objective in Eq. 4).",
      "Learning over many sequences: a sampling scheme drawing more than 100 image pairs per training run, combining spatial sampling within sequences and temporal sampling across sequences.",
      "Optimization in the Fourier domain using Gauss-Newton iterations with a fixed number of gradient steps — practical at about 5 FPS in MATLAB.",
      "State-of-the-art at publication: OTB-2013 mean overlap 78.1 vs MEEM 70.1, OTB-2015 72.9 (AUC 60.5), ALOV++ mean F 0.787, VOT2014 accuracy 0.63 / 15.90 failures.",
      "Ablation proving spatially dependent regularization matters: uniform-weight baseline with expanded samples scores 72.2 mean OP versus 78.1 for SRDCF's spatial weights (50.1 when both samples and weights are mishandled).",
    ],
    method: {
      pipeline: ["sample", "crop", "weight", "optimize-filter", "detect", "update"],
      architecture:
        "Single correlation-filter tracker with HOG features (4x4 cells) over a region about 42 pixels larger than the target; the filter is learned once over many training pairs and adapted by re-learning with the previous estimate as center each frame.",
      motionModel:
        "None — detection crops around the previous state; no explicit motion filter.",
      appearanceModel:
        "HOG feature channels of an enlarged window, with background explicitly down-weighted by the spatial regularizer instead of being discarded.",
      detectionDependency:
        "None — a detection-free filter tracker (bounding-box regression used only to build training boxes from annotations).",
      loss:
        "Regularized least squares where the penalty term is spatially weighted: sum of squared errors over training pairs plus sum over pixels of ||w * f_l||^2 (Eq. 4).",
      optimization:
        "Fourier-domain optimization with Gauss-Newton iterations (parameter mu initialized 1, updated by factor beta = 10, capped at 10^3) plus a small fixed number of gradient steps; 2 outer iterations in practice.",
    },
    equations: [
      {
        id: "srddf-objective",
        label: "Plain regularized filter objective",
        formula: "E_t(f) = sum_k (alpha_k * S_f(x_k) - y_k)^2 + lambda * sum_l ||f_l||^2",
        variables: [
          { symbol: "x_k", meaning: "training sample k (a cropped window)" },
          { symbol: "y_k", meaning: "desired correlation output (Gaussian peak)" },
          { symbol: "S_f", meaning: "correlation response of filter f on sample x_k" },
          { symbol: "alpha_k", meaning: "sample weight" },
          { symbol: "lambda", meaning: "uniform ridge regularization" },
        ],
        intuition:
          "Standard CF training: fit the responses with uniform weight everywhere in the filter — exactly the formulation whose background coefficients get contaminated.",
        why:
          "This is the baseline SRDCF modifies; uniform lambda cannot distinguish object from background pixels.",
        where: "Section 2, Eq. 2 (spatial-regularization-free formulation).",
        paperIds: ["T010"],
      },
      {
        id: "srddf-regularized",
        label: "Spatially regularized objective",
        formula: "E(f) = sum_k (alpha_k * S_f(x_k) - y_k)^2 + sum_l ||w .* f_l||^2",
        variables: [
          { symbol: "w", meaning: "spatial weight map: near 0 over the target center, growing toward the background border" },
          { symbol: "f_l", meaning: "filter coefficients of channel l at each pixel" },
          { symbol: ".*", meaning: "element-wise multiplication by the weight map" },
        ],
        intuition:
          "Free the filter over the object (w small), squeeze it over the background (w large): background suppression happens through regularization instead of data corruption.",
        why:
          "It is the paper's central idea and what the name Spatially Regularized CF refers to; the ablation (78.1 vs 72.2 mean OP) measures its contribution.",
        where: "Section 3, Eq. 4; weight map derived from target overlap with the filter support.",
        params:
          "Regularization grows from the target interior to the crop border; mu controls the ADMM/GN penalty parameter (init 1, xbeta 10, max 10^3).",
        simulator: "corr-filter",
        paperIds: ["T010"],
      },
      {
        id: "srddf-fourier",
        label: "Fourier-domain optimization form",
        formula: "E(f) = sum_k (alpha_k * S_f(x_k) - y_k)^2 + sum_l ||(w_hat / (M*N)) * conv(f_hat_l)||^2",
        variables: [
          { symbol: "w_hat, f_hat", meaning: "DFT of the weight map and filter channels" },
          { symbol: "M*N", meaning: "size of the filter region" },
          { symbol: "conv", meaning: "circular convolution (pointwise multiplication after Fourier transform)" },
        ],
        intuition:
          "Because correlation is multiplication in Fourier space, the weighted penalty becomes a per-frequency product — but the weight's spectrum couples frequencies, so an iterative solver is needed.",
        why:
          "Makes the spatially regularized problem tractable: the whole optimization runs spectrally at a few FPS instead of in pixel space.",
        where: "Section 3-4, Fourier representation of Eq. 4 and its solution algorithm.",
        params: "Training region ~42 px larger than target; HOG 4x4; ~5 FPS in MATLAB.",
        simulator: "corr-filter",
        paperIds: ["T010"],
      },
    ],
    datasets: ["otb", "vot", "others"],
    metrics: ["success-auc", "precision", "fps"],
    baselines: ["KCF", "DSST", "SAMF", "MEEM", "Staple", "MOSSE"],
    results: [
      "OTB-2013: mean overlap 78.1 (SRDCF) vs 70.1 (MEEM) with AUC 63.3; OTB-2015 mean overlap 72.9 vs 64.7 (SAMF), AUC 60.5 (+5.7% over SAMF).",
      "ALOV++: mean F-measure 0.787 — the best among compared filters.",
      "VOT2014 (EOT): accuracy 0.63, 15.90 expected failures, accuracy rank 6.43, robustness rank 10.08, combined rank 8.26.",
      "Training-sample ablation (mean OP): standard samples 71.1 (uniform) / 72.2 (spatial); expanded samples 50.1 (uniform weights) / 78.1 (spatial weights).",
      "Runtime: about 5 FPS in the paper's MATLAB implementation.",
    ],
    ablations: [
      "Uniform vs spatial regularization weights with the same expanded samples: 72.2 -> 78.1 mean OP (and 50.1 for uniform+expanded showing the failure mode the weight fixes).",
      "Standard vs expanded training samples: more data only helps when paired with spatial regularization.",
      "Comparison against CFLB-style background learning (78.1 vs 48.6 baseline variants in Table 1).",
    ],
    limitations: {
      authorStated: [
        "Optimization needs iterative numerical methods (Gauss-Newton/gradient with penalty parameter mu), unlike the closed-form DCF — noted as the cost of spatial regularization.",
        "At about 5 FPS in MATLAB it does not reach real-time, which the follow-up works (ECO, BACF) target.",
      ],
      evident: [
        "The weight map is derived from the target crop, so incorrect box estimates feed back into regularization.",
        "No appearance-model update semantics beyond re-centering: occlusion robustness comes only from the regularizer, not explicit handling.",
      ],
    },
    assumptions: [
      "The region around the previous box mostly contains background — hence penalizing filter weights there is safe.",
      "Training samples drawn from many sequences capture the background statistics the filter must reject.",
    ],
    computation:
      "Iterative Fourier-domain optimization per frame (2 GN iterations, 4 gradient steps, mu schedule 1 -> 10 -> capped 10^3) — about 5 FPS in MATLAB.",
    relations: [
      {
        to: "T001",
        type: "builds-on",
        note: "Keeps KCF's circulant/Fourier machinery and ridge objective but replaces uniform regularization with the spatially weighted penalty — the direct CF successor to T001.",
      },
    ],
    concepts: ["correlation-filter", "appearance-features", "online-vs-offline", "sot", "bounding-box", "occlusion"],
    impact:
      "SRDCF's spatial regularization became the standard fix for CF background contamination and the foundation for ECO (T011), which reports SRDCF as its strongest prior CF baseline, and its expanded-sampling idea (100+ image pairs) shifted CF training from per-video patches to pooled multi-sequence datasets.",
  },
  {
    id: "T011",
    arxiv: "1611.09224",
    title: "ECO: Efficient Convolution Operators for Tracking",
    shortTitle: "ECO",
    year: 2016,
    authors: [
      "Martin Danelljan",
      "Goutam Bhat",
      "Fahad Shahbaz Khan",
      "Michael Felsberg",
    ],
    fileName: "1611.09224v2.pdf",
    task: "single-object",
    tags: [
      "correlation-filter",
      "generative-model",
      "dimensionality-reduction",
      "feature-compactness",
      "efficiency",
      "real-time",
    ],
    difficulty: "advanced",
    summary:
      "ECO attacks the two practical bottlenecks of C-COT-style convolutional filters — the huge per-frame training cost of densely sampled redundant data and the over-complex model that overfits — with three changes: a compact generative model of the training distribution (GMM instead of storing every sample), a smaller learned projection between feature channels, and a simplified optimization schedule. The result is a 20x speedup over C-COT with better accuracy: EAO 0.374 on VOT2016, 60 FPS for the compact version, and top results on OTB, TempleColor and UAV123.",
    problem:
      "The best correlation trackers (C-COT) learn per-frame from hundreds of thousands of densely sampled patches with large feature maps — each update is expensive (0.51 FPS, EFO 0.51) and the huge number of nearly identical samples causes overfitting, so accuracy saturates while speed collapses. Storing and re-optimizing the entire training history every frame is fundamentally wasteful.",
    background: [
      "correlation-filter",
      "appearance-features",
      "online-vs-offline",
      "deep-detection",
      "sot",
      "bounding-box",
      "iou",
    ],
    previousWork: [
      {
        name: "C-COT (Danelljan et al.)",
        limitation:
          "State-of-the-art accuracy but trains on about 512 feature channels with every stored sample each frame — running at 0.51 FPS and prone to overfitting on redundant data.",
        whyThisPaper:
          "ECO keeps C-COT's continuous-resolution formulation but reduces feature dimension, training samples and update cost.",
      },
      {
        name: "SRDCF (T010) and prior CFs",
        limitation:
          "Spatial regularization helped but fixed-scale, huge-sample training remained; SRDCF still ran slowly with hand-tuned hyper-parameters.",
        whyThisPaper:
          "ECO inherits the regularization/continuous-resolution idea and adds generative sample compression, cutting samples by 90% while improving accuracy.",
      },
      {
        name: "Deep trackers (MDNet T003, SiamFC T009) and shallow trackers",
        limitation:
          "Deep trackers are accurate but slow (MDNet 0.8 FPS); shallow real-time trackers lack accuracy; neither could be the default.",
        whyThisPaper:
          "ECO explicitly targets the accuracy-speed Pareto: better than C-COT with 20x speedup, and ECO-HC reaches 15 FPS (60 FPS) on CPU-quality features.",
      },
    ],
    researchGap:
      "No prior CF work had identified that sample redundancy (rather than sample count) causes overfitting, nor provided a principled way to compress the training history while preserving the learned filter.",
    contribution: [
      "Compact generative model: replace the stored sample set with a Gaussian mixture model whose components are merged whenever two are close (Eq. 11), reducing stored samples by ~90% and computation by ~80%.",
      "Factorized interpolation operator with a learned projection matrix P that reduces hundreds of feature channels to a compact set before convolution (Eq. 7), cutting model size ~80%.",
      "Simplified optimization schedule: far fewer Gauss-Newton iterations, with a deliberately coarse early schedule, cutting iterations ~80%.",
      "Unified result: 20x faster than C-COT with higher accuracy — VOT2016 EAO 0.374 (C-COT 0.331, +13.0%), EFO 4.53 (C-COT 0.51), ~4-6 FPS for ECO, 60 FPS for ECO-HC.",
      "Extensive evaluation showing ECO leads on OTB-2015 (AUC 70.0), UAV123 (53.7), TempleColor (60.5) against C-COT, MDNet, SiamFC and SRDCF.",
    ],
    method: {
      pipeline: ["embed", "project", "fit-gmm", "optimize", "detect"],
      architecture:
        "C-COT-style correlation tracker: deep or compact feature channels (VGG-M conv layers or hand-crafted HOG+color for ECO-HC), a learned projection P reducing channel count, and a continuous-resolution filter learned from a Gaussian-mixture summary of all past training patches.",
      motionModel:
        "None — crop around previous state; scale handled by the filter's continuous-resolution interpolation.",
      appearanceModel:
        "Multi-channel convolutional filter over learned projections of deep features (ECO) or HOG+color-names (ECO-HC), summarized by a GMM over the patch distribution.",
      detectionDependency:
        "None — a filter tracker.",
      trackManagement:
        "No explicit track state; robustness comes from the compact model reducing drift (the paper links model complexity to overfitting rather than occlusion handling).",
      loss:
        "Regularized expected squared error between the filter response and a Gaussian label, plus channel-wise quadratic regularization on the filter (Eqs. 7, 10).",
      optimization:
        "Approximate Gauss-Newton with an aggressive schedule (many iterations early at coarse scale, few at fine scale) plus the GMM-based data term; components initialized with pi = gamma, mu = sample.",
    },
    equations: [
      {
        id: "eco-objective",
        label: "Compact-model training objective",
        formula: "E(f, P) = ||z_hat^T P f_hat - y_hat||^2_l2 + sum_c ||w_hat * f_hat_c||^2_l2 + lambda * ||P||^2_F",
        variables: [
          { symbol: "P", meaning: "learned projection matrix reducing feature channels" },
          { symbol: "f", meaning: "multi-channel correlation filter" },
          { symbol: "z", meaning: "feature maps of the training sample" },
          { symbol: "y_hat", meaning: "Gaussian label in Fourier domain" },
          { symbol: "w", meaning: "regularization weights (spatial/feature)" },
        ],
        intuition:
          "Fit responses through a compressed channel representation, penalizing filter complexity and projection size — accuracy with a small model.",
        why:
          "The projection is one of ECO's three compression levers; it directly reduces parameters and computation versus C-COT.",
        where: "Section 3, Eq. 7 (compact model with learned projection).",
        params: "Projection initialized from PCA of samples, then learned; lambda controls regularization.",
        simulator: "corr-filter",
        paperIds: ["T011", "T010"],
      },
      {
        id: "eco-gmm",
        label: "Generative-model merging",
        formula: "pi_n = pi_k + pi_l ,  mu_n = (pi_k * mu_k + pi_l * mu_l) / (pi_k + pi_l)",
        variables: [
          { symbol: "pi, mu", meaning: "mixture weight (sample proportion) and mean feature of a component" },
          { symbol: "k, l, n", meaning: "two merged components and the new merged one" },
        ],
        intuition:
          "Summarize thousands of stored patches as a handful of Gaussians; when two components are similar, merge them into their weighted average.",
        why:
          "It is the core efficiency idea: ~90% fewer samples to store and train on each frame, while preserving the distribution instead of dropping data arbitrarily.",
        where: "Section 4, GMM formulation and merge rule, Eq. 11 (loss over mixture in Eq. 12).",
        params:
          "Component count kept below a cap (L ≈ M/8 in practice); pi initialized to gamma, mu initialized to the incoming sample x_j.",
        paperIds: ["T011"],
      },
      {
        id: "eco-interp",
        label: "Factorized interpolation operator",
        formula: "S_P f = P^T (f (*) x)",
        variables: [
          { symbol: "P", meaning: "projection applied to feature channels" },
          { symbol: "f (* x)", meaning: "correlation of filter with feature map x" },
          { symbol: "S_P f", meaning: "response after dimensionality reduction" },
        ],
        intuition:
          "Compute correlations on the compressed channels: the projection sits before the expensive convolution work.",
        why:
          "It reduces both the filter size and per-frame correlation cost — part of the 80% parameter reduction the paper reports.",
        where: "Section 3, Eq. 6 (interpolation with projection).",
        paperIds: ["T011"],
      },
    ],
    datasets: ["otb", "vot", "uav123", "others"],
    metrics: ["eao", "success-auc", "fps"],
    baselines: ["C-COT", "SRDCF", "MDNet", "SiamFC", "Staple", "DeepSRDCF"],
    results: [
      "VOT2016: ECO EAO 0.374 vs C-COT 0.331 (+13.0%), with EFO 4.53 vs 0.51 — comparable accuracy at ~9x better runtime; ECO-HC reaches 15.13 EFO and EAO 0.322.",
      "Ablation of ECO components (EAO/FPS): 0.331/0.3 -> 0.342/1.1 -> 0.352/2.6 -> 0.374/6.0 as compact-model features are added.",
      "Projection ablation: PCA initialization vs learned projection 0.319 -> 0.342 EAO; generative model with L = 50 samples 0.338 -> 0.351 with GMM summary.",
      "OTB-2015 AUC: ECO 70.0, C-COT 69.0, ECO-HC 65.0, MDNet 68.5, SRDCF 60.5, SiamFC 57.5, Staple variants 45.3-58.4.",
      "UAV123: ECO 53.7 vs C-COT 51.7 and SRDCF-class baselines; TempleColor: ECO 60.5 vs C-COT 59.7.",
      "Efficiency: ~80% parameter reduction (512 -> 64 filter channels in Figure 2), 90% fewer samples, 80% fewer iterations; ECO-HC runs at 60 FPS.",
    ],
    ablations: [
      "Component-by-component build-up: each of projection, generative model and simplified optimization adds both EAO and FPS (0.331/0.3 -> 0.374/6.0).",
      "PCA vs learned projection (0.319 vs 0.342) shows learning the projection matters beyond initialization.",
      "Fixed sample count vs generative summary (0.338 vs 0.351) shows the GMM helps accuracy, not just speed.",
      "Deep (VGG-M) vs hand-crafted (HOG+color) features: ECO vs ECO-HC trade accuracy for 60 FPS CPU-friendly speed.",
    ],
    limitations: {
      authorStated: [
        "ECO's optimization is approximate Gauss-Newton with a simplified schedule — accuracy could be recovered by more iterations at extra cost.",
        "ECO-HC trades accuracy for speed (EAO 0.322 vs 0.374); the hand-crafted variant is the fast operating point.",
      ],
      evident: [
        "GMM component merging is a lossy compression: tracking sequences with rapidly changing appearance may lose old modes.",
        "No explicit occlusion/failure handling — like its CF predecessors, robustness is statistical rather than stateful.",
      ],
    },
    assumptions: [
      "The training-patch distribution is well approximated by a small Gaussian mixture over frames.",
      "Feature channels are redundant and a learned projection retains what the filter needs.",
      "Correlation responses alone suffice — no motion or track-state model.",
    ],
    computation:
      "Per-frame cost cut ~80% on parameters, ~90% samples, ~80% iterations vs C-COT: 6.0 FPS (ECO) and 60 FPS (ECO-HC), vs C-COT 0.3-0.5 FPS.",
    relations: [
      {
        to: "T010",
        type: "builds-on",
        note: "Carries SRDCF's spatial/regularized filter training into the continuous-resolution setting and reports SRDCF/SRDCFad/DeepSRDCF as its primary CF baselines (OTB-2015 SRDCF 60.5 vs ECO 70.0).",
      },
      {
        to: "T001",
        type: "uses-as-baseline",
        note: "KCF is listed among the compared shallow trackers in the OTB-2015 evaluation, representing the fast-but-weak end of the spectrum ECO dominates.",
      },
      {
        to: "T003",
        type: "uses-as-baseline",
        note: "MDNet (68.5 AUC on OTB-2015) is the deep accuracy baseline ECO beats while running far faster.",
      },
      {
        to: "T004",
        type: "uses-as-baseline",
        note: "Staple is compared in all OTB/TempleColor tables (45.3-58.4 AUC range depending on protocol).",
      },
      {
        to: "T009",
        type: "uses-as-baseline",
        note: "SiamFC (57.5 AUC on OTB-2015) appears as the representative fast siamese tracker in the comparisons.",
      },
    ],
    concepts: [
      "correlation-filter",
      "appearance-features",
      "online-vs-offline",
      "deep-detection",
      "sot",
      "occlusion",
    ],
    impact:
      "ECO defined the practical accuracy-speed frontier for CF trackers and stood as the state-of-the-art shallow tracker for years; its GMM sample compression and learned projection influenced subsequent efficient-filter designs, and its OTB/UAV/VOT numbers became the benchmark bar that deep siamese trackers (SiamRPN++ and later) had to clear.",
  },
  {
    id: "T012",
    arxiv: "1703.04590",
    title: "Learning Background-Aware Correlation Filters for Visual Tracking",
    shortTitle: "BACF",
    year: 2017,
    authors: ["Hamed Kiani Galoogahi", "Ashton Fagg", "Simon Lucey"],
    fileName: "1703.04590v2.pdf",
    task: "single-object",
    tags: [
      "correlation-filter",
      "background-aware",
      "real-time",
      "tracking-by-detection",
      "admm",
      "hough",
    ],
    difficulty: "advanced",
    summary:
      "BACF keeps the entire background surrounding the target as negative training data — instead of discarding it (as SRDCF effectively does by regularizing it away) — by framing the filter learning as a constrained optimization where a cropping operator extracts the small filter support from a large background-rich training sample. Solved with ADMM entirely in the Fourier domain, this gives a filter trained on all available context at about 35 FPS, beating SRDCF, LCT, Staple and KCF on OTB-50/100 and TempleColor while matching deep trackers at 170x their speed.",
    problem:
      "Correlation filters want large training windows full of background for discrimination, but the filter must stay small to avoid overfitting and keep detection cheap. Prior works either shrank the window (losing background) or added spatial regularization penalties (costly to optimize with hyper-parameters needing careful tuning); the useful background pixels were never fully exploited as negatives.",
    background: [
      "correlation-filter",
      "appearance-features",
      "bounding-box",
      "online-vs-offline",
      "iou",
      "sot",
    ],
    previousWork: [
      {
        name: "SRDCF (T010) spatial regularization",
        limitation:
          "Penalizes filter coefficients over the background rather than training on it — costly to optimize even in the Fourier domain and its hyper-parameters must be carefully tuned.",
        whyThisPaper:
          "BACF instead includes the background as positive training data through a hard cropping constraint, with only one hyper-parameter (ADMM mu) that adapts automatically.",
      },
      {
        name: "KCF/DCF (T001) small-window training",
        limitation:
          "Trains on the target crop with cyclic shifts, effectively discarding the surrounding background context that would make the filter discriminative.",
        whyThisPaper:
          "The cropping operator P lets the filter train on a much larger sample region while the learned filter remains small.",
      },
      {
        name: "Deep trackers (MDNet T003, SINT, SiamFC T009)",
        limitation:
          "Accurate but slow (MDNet 0.8 FPS, SINT 2.5 FPS, SiamFC 2 FPS) or less accurate (SiamFC 79.2 success on the deep-tracker OTB50 table).",
        whyThisPaper:
          "BACF reaches 85.4 success at 38.6 FPS — beating MDNet and SINT accuracy at roughly 48x/15x their speed.",
      },
    ],
    researchGap:
      "No CF formulation had simultaneously used the full background of a large training window as negatives and kept optimization simple enough for real-time execution without per-dataset tuning.",
    contribution: [
      "Background-aware formulation: a constrained objective where a cropping/selection operator P extracts the small filter region from a large, background-rich sample (large T >> D), so all context is trained on rather than regularized away.",
      "ADMM solution entirely in the Fourier domain with closed-form updates — no spatial-domain optimization, one adaptive penalty parameter.",
      "Real-time speed: about 35-40 FPS average (35.3 FPS reported in the deep-tracker comparison) versus SRDCF 3.8 FPS — a 9x speedup with higher accuracy.",
      "State-of-the-art shallow results: OTB50 AUC 67.78 (vs LCT 62.48, SRDCF 62.35, KCF 51.86), OTB100 62.98, TempleColor 51.97; VOT2015 accuracy 0.59 (best in its table).",
      "Deep-tracker comparison: BACF 85.4 success at 38.6 FPS vs MDNet 87.3 at 0.8, SINT 85.3 at 2.5, SiamFC 79.2 at 2.0 — accuracy parity at orders-of-magnitude speed.",
    ],
    method: {
      pipeline: ["sample-large", "features", "constrain", "admm", "detect", "update"],
      architecture:
        "Single correlation filter with HOG (31 channels, 4x4 cells) plus a Hann window; the training sample is several times larger than the filter support, and the objective's P operator selects the central filter-sized region each iteration.",
      motionModel:
        "None — search centered on the previous state.",
      appearanceModel:
        "Target-centered filter trained with the full surrounding background of each sample as negatives (the 'background-aware' part).",
      detectionDependency:
        "None — filter-based, though the paper also builds deep-tracker comparisons using external deep detections/regressions for the combined variant.",
      loss:
        "Constrained least-squares correlation objective: minimize sample response error subject to g = sqrt(T)(P^T tensor I)h (Eq. 3-4), enforced by ADMM.",
      optimization:
        "ADMM with Fourier-domain updates: 2 iterations, penalty parameter mu initialized to 1 and increased multiplicatively (beta = 10, mu_max = 10^3); the closed-form h update uses a Sherman-Morrison identity.",
    },
    equations: [
      {
        id: "bacf-objective",
        label: "Background-aware constrained objective",
        formula: "min_h E_h = ||sqrt(T)(P^T tensor I_K) h - y||^2_F + lambda ||h||^2_F   s.t.  g = sqrt(T)(P^T tensor I_K) h",
        variables: [
          { symbol: "h", meaning: "the small correlation filter (support D x D)" },
          { symbol: "P", meaning: "cropping operator selecting the central filter region from the large sample (T x T, T much larger than D)" },
          { symbol: "g", meaning: "auxiliary variable equal to the cropped filter response" },
          { symbol: "y", meaning: "Gaussian label peaked at the sample center" },
        ],
        intuition:
          "Learn a small filter but evaluate it as if it covered a huge background-rich window: the constraint ties the compact filter to the large sample, so background becomes training data.",
        why:
          "This is the paper's core idea — background exploitation via constraint instead of via regularization penalties.",
        where: "Eqs. 3-4 (frequency-domain formulation and constrained form).",
        params: "lambda small (no per-dataset grid search); T set so the crop includes generous background.",
        simulator: "corr-filter",
        paperIds: ["T012", "T010"],
      },
      {
        id: "bacf-admm",
        label: "ADMM filter update",
        formula: "h* = (mu + lambda / sqrt(T))^-1 (mu g + zeta)",
        variables: [
          { symbol: "mu", meaning: "ADMM penalty parameter (init 1, multiplied by 10 each stage, capped 10^3)" },
          { symbol: "g", meaning: "auxiliary variable from the previous ADMM step" },
          { symbol: "zeta", meaning: "scaled dual variable" },
          { symbol: "lambda", meaning: "ridge regularization" },
        ],
        intuition:
          "Each ADMM sweep solves a simple diagonal system in Fourier space: blend the auxiliary signal and dual correction, divided by a scalar — one element-wise formula.",
        why:
          "It is why BACF is fast: every update is a pointwise spectral operation (via a Sherman-Morrison identity), so the background-aware objective trains in real time.",
        where: "ADMM solver section, closed-form h update (with the Sherman-Morrison identity).",
        params:
          "2 ADMM iterations per frame, mu schedule 1 -> 10 -> ... -> 10^3, only one meaningful hyper-parameter.",
        simulator: "corr-filter",
        paperIds: ["T012"],
      },
    ],
    datasets: ["otb", "vot", "others"],
    metrics: ["success-auc", "precision", "fps"],
    baselines: ["SRDCF", "LCT", "KCF", "Staple", "DSST", "MDNet", "SiamFC"],
    results: [
      "OTB50 OPE success: BACF 67.78 vs LCT 62.48, SRDCF 62.35, KCF 51.86; OTB100 62.98 vs SRDCF 60.13, Staple 58.03; TempleColor 51.97 vs SRDCF 51.66.",
      "Speed in the deep-tracker table: BACF 35.3 FPS average vs SRDCF 3.8, KCF 173.4, LCT 44.5; BACF wins on 41 videos (most) in per-video comparison against CCOT.",
      "Deep-tracker OTB50 success: BACF 85.4 at 38.6 FPS vs MDNet 87.3 at 0.8 FPS, SINT 85.3 at 2.5, SiamFC 79.2 at 2.0; combined deep-variant BACF vs CCOT 77.8 at 0.2 FPS (about 170x speed).",
      "VOT2015: BACF accuracy 0.59 (best in table); CCOT has better robustness (0.82 failures vs BACF 1.56).",
      "Average success across trackers: BACF 76.0 vs SRDCF 70.0.",
    ],
    ablations: [
      "Background exploitation: BACF vs SRDCF's penalty approach at matched features (67.78 vs 62.35 on OTB50) isolates the value of training on background rather than regularizing it.",
      "Shallow BACF vs its deep-feature variant vs CCOT: accuracy slightly below CCOT (85.4 vs 77.8-table comparison) but 170x faster — the accuracy-speed trade-off quantified.",
      "Hyper-parameter sensitivity: single ADMM mu schedule versus SRDCF's tuned lambda/regularization grid (paper argues no grid search needed).",
    ],
    limitations: {
      authorStated: [
        "On VOT2015 the robustness (failure count 1.56) is worse than CCOT's 0.82 — the fast shallow model still loses under reset-based occlusion stress.",
        "Per-video wins show BACF beats CCOT on 41 videos but not all — some sequences still favor the deep model.",
      ],
      evident: [
        "Fixed-scale correlation search with no explicit scale estimation or motion model.",
        "HOG-only features: the accuracy ceiling versus learned deep features is visible in the MDNet/CCOT comparisons.",
      ],
    },
    assumptions: [
      "The background around the previous box is representative of the background to reject (context-rich crops are valid negatives).",
      "The filter support can stay small while the sample region is large — the cropping constraint keeps them consistent.",
    ],
    computation:
      "ADMM in the Fourier domain with 2 iterations per frame, all element-wise: 35.3 FPS average (38.6 FPS in the deep table), versus SRDCF 3.8 FPS.",
    relations: [
      {
        to: "T010",
        type: "addresses-limitation",
        note: "Explicitly targets SRDCF's drawbacks — 'costly to optimize even in the Fourier domain' with hyper-parameters 'carefully tuned' — replacing its background-damping regularizer with a background-using constraint.",
      },
      {
        to: "T001",
        type: "uses-as-baseline",
        note: "KCF is the canonical fast CF baseline (173.4 FPS, 51.86 OTB50 success) against which BACF shows it gains accuracy without giving up real-time speed.",
      },
      {
        to: "T003",
        type: "uses-as-baseline",
        note: "MDNet (87.3 success at 0.8 FPS) anchors the deep-accuracy end of the comparison BACF nearly matches at 38.6 FPS.",
      },
      {
        to: "T004",
        type: "uses-as-baseline",
        note: "Staple appears throughout the OTB tables (58.03 on OTB100) as the fast complementary-learner tracker BACF outperforms.",
      },
      {
        to: "T009",
        type: "uses-as-baseline",
        note: "SiamFC (79.2 success at 2 FPS) is the siamese baseline in the deep-tracker comparison table.",
      },
    ],
    concepts: ["correlation-filter", "appearance-features", "online-vs-offline", "sot", "bounding-box", "iou"],
    impact:
      "BACF showed the background itself is the resource CFs were wasting, and its real-time-plus-accuracy result became the practical bar for shallow trackers (later ECO-style works cite it); its ADMM-in-Fourier solution also demonstrated that non-diagonal CF constraints need not cost real-time speed.",
  },
  {
    id: "T013",
    arxiv: "1703.07402",
    title: "Simple Online and Realtime Tracking with a Deep Association Metric",
    shortTitle: "Deep SORT",
    year: 2017,
    authors: ["Nicolai Wojke", "Alex Bewley", "Dietrich Paulus"],
    fileName: "1703.07402v1.pdf",
    task: "multi-object",
    tags: [
      "tracking-by-detection",
      "deep-association",
      "appearance-features",
      "kalman-filter",
      "online",
      "real-time",
      "reid",
    ],
    difficulty: "intermediate",
    summary:
      "Deep SORT upgrades SORT's IoU-only data association with a deep appearance descriptor: a small CNN (trained on the MARS re-identification dataset) produces 128-D embeddings of each detection, and the assignment cost combines Mahalanobis motion gating with cosine appearance distance. Along with cascaded (camera-motion-compensated) matching, a larger track-gallery and a track-confirmation state machine, this cuts identity switches on MOT16 by 45% (1423 -> 781) while staying near real time (about 20 Hz) — turning SORT's speed into usable online MOT.",
    problem:
      "SORT is fast and simple but relies on the IoU of bounding boxes alone for association: when detections are sparse, objects deform, or targets cross, boxes of different identities overlap enough to swap, producing 1423 identity switches on MOT16. Purely motion-based association cannot distinguish nearby similar-looking pedestrians — a learned appearance metric was needed without losing online real-time operation.",
    background: [
      "mot",
      "tracking-by-detection",
      "data-association",
      "kalman",
      "appearance-features",
      "reid",
      "occlusion",
      "online-vs-offline",
    ],
    previousWork: [
      {
        name: "SORT (T005)",
        limitation:
          "Associates detections with tracks by IoU distance only, yielding high identity switching (1423 IDSW on MOT16 test) because motion alone cannot resolve crossings and sparse detections.",
        whyThisPaper:
          "Adds a deep appearance metric and cascade matching on top of SORT's Kalman/Hungarian skeleton while keeping it online.",
      },
      {
        name: "MOT16 baselines (TBD, CEM, DP NMS, JPDA M)",
        limitation:
          "Batch/offline systems with limited identity preservation, or too slow for online use (1.3-6.5 Hz), and ID measures unaddressed.",
        whyThisPaper:
          "Deep SORT runs at about 20 Hz and directly optimizes the identity metrics (IDSW/FM) the baseline table reports.",
      },
      {
        name: "Classical re-identification metrics / appearance models",
        limitation:
          "Either trained per-scene online or too costly; full pairwise re-ID over all tracks conflicts with real-time MOT.",
        whyThisPaper:
          "A fixed pre-trained CNN descriptor (MARS-trained) gives a cosine metric that gates association without per-scene training.",
      },
    ],
    researchGap:
      "Before Deep SORT, no online real-time MOT system combined a learned appearance descriptor with motion gating in the assignment problem while remaining simple enough to reproduce SORT's speed.",
    contribution: [
      "Deep appearance feature: a CNN (2,800,864 parameters, trained on MARS with 1.1M images of 1261 pedestrians) maps detections to 128-D embeddings; association cost uses cosine distance within a Mahalanobis-gated region.",
      "Cascaded matching (Listing 1): priority tiers by track age — confirmed/long-lived tracks matched first with the stricter metric, then unconfirmed tracks by IoU — reducing switches caused by late matches.",
      "Larger track management: gallery size R_k = 100 appearance vectors per track with running cosine-update, plus a max-age A_max of 30 frames allowing long-term re-association after occlusion.",
      "MOT16 test results: MOTA 61.4 (vs SORT 59.8), IDSW 781 vs SORT's 1423 (45% fewer), FM 2008, MOTP 79.1, 32.8% MT, 18.2% ML at about 20 Hz (table reports 40 Hz on the evaluation hardware).",
      "Ablations over lambda (motion/appearance trade-off), A_max and detection threshold quantify how appearance weight reduces identity switches at a small MOTA cost.",
    ],
    method: {
      pipeline: ["detect", "predict", "cascade-match", "unmatched", "manage", "update"],
      architecture:
        "Detection-based online tracker: an off-the-shelf pedestrian detector supplies boxes; a constant-velocity Kalman filter predicts each track state (bounding box + velocity); an assignment problem pairs predicted states to detections using the combined metric; track lifecycle states are tentative -> confirmed -> deleted.",
      motionModel:
        "Kalman filter with constant-velocity model in image coordinates (bounding box center, aspect ratio, height with derived velocities); prediction used both for gating and for bridging missed detections.",
      appearanceModel:
        "Pre-trained CNN descriptor (MARS re-ID dataset), cosine similarity over a gallery of the last R_k = 100 embeddings per track; running update f_i <- gamma f_i + (1-gamma) normalized average.",
      detectionDependency:
        "Fully detection-driven (generic pedestrian detector, confidence threshold 0.3 in experiments) — no motion-only detection.",
      trackManagement:
        "Tentative -> confirmed after enough hits; track deletion on T_lost; unmatched confirmed tracks keep predicting up to A_max = 30 frames for post-occlusion recovery; unmatched detections spawn tentative tracks.",
      loss: "Re-ID cross-entropy/triplet-style training on MARS (the descriptor is pre-trained offline; no online loss).",
      optimization: "Descriptor training offline only; per-frame assignment solved exactly with the Hungarian algorithm on the gated cost matrix.",
    },
    equations: [
      {
        id: "deepsort-mahalanobis",
        label: "Mahalanobis motion distance",
        formula: "d((i, j), (r_k)) = (y_i - H x_k)^T S_k^-1 (y_i - H x_k)",
        variables: [
          { symbol: "y_i", meaning: "measurement (detection) i" },
          { symbol: "x_k, S_k", meaning: "predicted state and covariance of track k" },
          { symbol: "H", meaning: "measurement projection matrix" },
        ],
        intuition:
          "Distance in units of the track's own uncertainty: a sloppy prediction gates loosely, a confident one gates tightly.",
        why:
          "It provides the motion gate that limits which (track, detection) pairs are allowed to be associated at all.",
        where: "Section 3.1, Eq. 1 (with the gating threshold chi-squared table).",
        paperIds: ["T013"],
      },
      {
        id: "deepsort-appearance",
        label: "Cosine appearance distance",
        formula: "d^((2))((i, j), (r_k)) = min(1 - r_j^T r_k^i, 1 - r_j^T r_k^i),  r_k^i = normalized gallery embedding",
        variables: [
          { symbol: "r_j", meaning: "128-D normalized descriptor of detection j" },
          { symbol: "r_k^i", meaning: "i-th gallery descriptor of track k" },
          { symbol: "min over gallery", meaning: "smallest distance to any recent appearance of the track" },
        ],
        intuition:
          "How different does this detection look from anything this track looked like recently? Minimum over the 100-entry gallery tolerates pose/lighting variation.",
        why:
          "This is the 'deep' part of Deep SORT — the cue that separates identities when motion alone cannot.",
        where: "Section 3.1, Eq. 2-3 (appearance feature distance, gallery of size R_k).",
        params: "R_k = 100; descriptor trained on MARS (1.1M images, 1261 pedestrians), 30 ms per 32 boxes on a GTX 1050.",
        paperIds: ["T013"],
      },
      {
        id: "deepsort-cost",
        label: "Combined assignment cost",
        formula: "c_{i,j} = lambda * d^((1))_{i,j} + (1 - lambda) * d^((2))_{i,j},  gated by d^((1)) <= tau",
        variables: [
          { symbol: "lambda", meaning: "weight trading motion against appearance (0 = pure appearance in the reported settings)" },
          { symbol: "d^((1))", meaning: "Mahalanobis distance (also the gating gate)" },
          { symbol: "d^((2))", meaning: "cosine appearance distance" },
        ],
        intuition:
          "Only pairs that pass the motion gate enter the cost; among them, appearance decides who matches whom.",
        why:
          "It is the association formulation the Hungarian algorithm solves each frame — and the knob (lambda) the paper sweeps in its ablations.",
        where: "Section 3.1, Eqs. 4-5 (combined metric and gating region).",
        params: "Reported runs use lambda = 0 (appearance-dominated) with gating on the Mahalanobis distance; A_max = 30.",
        simulator: "kalman",
        paperIds: ["T013"],
      },
      {
        id: "deepsort-cascade",
        label: "Cascade matching priority",
        formula: "match confirmed/long tracks first (small cascade depth), then unconfirmed by IoU; unmatched -> track maintenance",
        variables: [
          { symbol: "cascade levels", meaning: "ordered from tracks with the most history to newest tentative tracks" },
          { symbol: "unmatched detections", meaning: "spawn new tentative tracks" },
        ],
        intuition:
          "Give mature tracks first pick of detections so younger tracks cannot steal their appearances — older tracks have better models.",
        why:
          "It directly reduces IDSW: the paper's Listing 1 cascade plus gallery is what separates Deep SORT from one-shot Hungarian matching.",
        where: "Section 3.2 and Listing 1 (cascade matching), followed by track management.",
        params: "Confirmed-track IoU recovery matching and A_max = 30 handle long occlusions.",
        simulator: "track-mgmt",
        paperIds: ["T013", "T005"],
      },
    ],
    datasets: ["mot16"],
    metrics: ["mota", "motp", "idf1", "idsw", "fp", "fn", "fragments", "mt-ml", "fps"],
    baselines: [
      "SORT",
      "KDNT",
      "LMP",
      "MCMOT",
      "NOMTwSDP16",
      "EAMTT",
      "POI",
    ],
    results: [
      "MOT16 test (table): Deep SORT MOTA 61.4, MOTP 79.1, MT 32.8%, ML 18.2%, IDSW 781, FM 2008, FP 12852, FN 56668, 40 Hz (table).",
      "Versus SORT on the same table: SORT MOTA 59.8, IDSW 1423, FM 2453, FP 13141, FN 57952, 14.5 Hz — Deep SORT cuts identity switches by 45% and fragments by 18% for +1.6 MOTA.",
      "Comparison rows: KDNT 68.2 (batch), LMP 71.0 (batch), MCMOT 62.4, NOMTwSDP16 62.2, EAMTT 52.5, POI 66.1.",
      "Descriptor cost: CNN with 2,800,864 parameters takes 30 ms to process 32 boxes on an NVIDIA GTX 1050; the abstract/section text states tracking runs at approximately 20 Hz.",
      "Descriptor training: MARS dataset, 1.1M images of 1261 pedestrians; detections thresholded at 0.3 confidence.",
    ],
    ablations: [
      "lambda sweep (motion vs appearance weight): increasing appearance weight reduces IDSW substantially with only small MOTA degradation — identity gains come from appearance, not motion.",
      "A_max (max track age): raising the allowed miss count recovers tracks after longer occlusions, trading some FP for fewer IDSW/FM.",
      "Detection threshold and gallery size R_k variations show descriptor freshness matters: a 100-entry gallery outperforms short histories under occlusion.",
      "MOTP with and without appearance gating: 79.6 (SORT-style) to 79.1 (Deep SORT) — the appearance cost slightly lowers localization precision while improving identity.",
    ],
    limitations: {
      authorStated: [
        "False positives increase as A_max grows — keeping lost tracks alive longer admits background detections.",
        "MOTP drops (79.6 -> 79.1) with the appearance term — identity gains cost a little localization precision.",
        "The descriptor is trained specifically for pedestrians and evaluated only on MOT16; other object classes need new training.",
        "GPU dependence for the descriptor (30 ms per batch on a GTX 1050) — the original SORT's CPU-only claim no longer holds.",
      ],
      evident: [
        "Purely online: no batch smoothing or global optimization, so MOTA trails offline methods (LMP 71.0, KDNT 68.2) on MOT16.",
        "Kalman constant-velocity assumption fails under abrupt camera motion; recovery relies entirely on appearance after A_max frames.",
      ],
    },
    assumptions: [
      "A pedestrian detector provides boxes every frame (tracking quality bounded by detector recall/precision).",
      "A pre-trained re-ID descriptor transfers to the MOT16 scenes without fine-tuning.",
      "Near-constant image-plane velocity between frames (Kalman model) with appearance bridging the gaps.",
    ],
    computation:
      "Kalman prediction + one Hungarian solve per frame plus a small CNN pass (30 ms per 32 detections on GTX 1050): about 20 Hz end-to-end (40 Hz reported in the results table), vs SORT 14.5 Hz.",
    relations: [
      {
        to: "T005",
        type: "extends",
        note: "Deep SORT keeps SORT's Kalman-filter + Hungarian online skeleton and replaces its IoU cost with the deep appearance metric; the paper is explicitly the extended version of SORT.",
      },
      {
        to: "T006",
        type: "uses-as-baseline",
        note: "Evaluates on the MOT16 protocol and compares against the benchmark's baseline family (SORT row) and leaderboard methods with the standard MOTA/MOTP/ID measures.",
      },
    ],
    concepts: [
      "tracking-by-detection",
      "data-association",
      "appearance-features",
      "reid",
      "kalman",
      "track-management",
      "learned-association",
    ],
    impact:
      "Deep SORT became the default online MOT baseline: its appearance-augmented assignment is the skeleton reused by FairMOT, CenterTrack-era trackers and most production MOT systems, and its 45% IDSW reduction established appearance metrics (later IDF1/HOTA) as the axis that matters once MOTA is near-saturated.",
  },
];
