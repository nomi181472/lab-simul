import type { PaperRecord } from "../types";

/* batch-03 — papers T014–T025, extracted per PAPER_SPEC from /tmp/opencode/txt. */

export const BATCH_03: PaperRecord[] = [
  {
    id: "T014",
    arxiv: "1704.06036",
    title: "End-to-end representation learning for Correlation Filter based tracking",
    shortTitle: "CFNet",
    year: 2017,
    authors: [
      "Jack Valmadre",
      "Luca Bertinetto",
      "João F. Henriques",
      "Andrea Vedaldi",
      "Philip H. S. Torr",
    ],
    fileName: "1704.06036v1.pdf",
    task: "single-object",
    tags: ["correlation-filter", "siamese", "end-to-end", "fourier-domain", "real-time"],
    difficulty: "intermediate",
    summary:
      "CFNet treats a Correlation Filter as a differentiable layer inside a small asymmetric Siamese network, so the tracker's feature extractor is trained end-to-end through the CF's closed-form ridge-regression solution. It matches deeper siamese baselines on OTB while running at 75-83 fps with a much smaller network.",
    problem:
      "Correlation Filter trackers (KCF and successors) solve a ridge-regression template in the Fourier domain every frame, but their features came from networks trained for other tasks; nothing let the CF participate in representation learning. Making the CF differentiable looked impossible because the template is defined as the solution of a large linear system, so gradients through it had no known closed form.",
    background: [
      "correlation-filter",
      "siamese",
      "sot",
      "appearance-features",
      "online-vs-offline",
      "success-plot",
    ],
    previousWork: [
      {
        name: "SiamFC (T009)",
        limitation:
          "Trains a Siamese cross-correlation matcher end-to-end but discards the CF's efficient online retraining: the similarity function is a plain correlation of features, with no discriminative filter fitted each frame.",
        whyThisPaper:
          "Keeps the Siamese training scheme while inserting a real CF between the branches, and trains through it.",
      },
      {
        name: "KCF / DSST-style correlation filters (T001)",
        limitation:
          "Fast and re-trained every frame, but their features are fixed (HOG/handcrafted or pre-trained CNN layers), so representation learning and filtering are disconnected.",
        whyThisPaper:
          "Derives the gradient of the CF solution with respect to its input so the same features can be shaped by the tracking loss.",
      },
      {
        name: "C-COT and other deep DCF trackers",
        limitation:
          "Use deep features inside a CF but train the network offline for other purposes and keep the CF unfrozen; end-to-end joint training was still missing.",
        whyThisPaper:
          "First network to back-propagate through the solution of the circulant linear system that defines the filter.",
      },
    ],
    researchGap:
      "Before this paper, no one could back-propagate a tracking loss through a Correlation Filter's optimal solution, so CF-based trackers could not learn their own representations end-to-end.",
    contribution: [
      "Formulates the CF as a network layer h(x', z') = s · ω(f_ρ(x')) ⋆ f_ρ(z') + b and trains the whole asymmetric Siamese architecture offline with logistic loss on random pairs.",
      "Gives a closed-form expression for the derivative of the CF template, obtained by taking differentials of the circulant kernel system and using Parseval's theorem to move them to the Fourier domain.",
      "Shows back-propagation through the Lagrange-multiplier solution is what matters: the CFNet-const variant (α learned offline, frozen) is consistently worse than CFNet.",
      "Compact models: CFNet-conv2 runs at 75 fps and CFNet-conv1 at 83 fps while beating SiamFC-3s in overlap on OTB-2013 (61.1 vs 60.7 IoU, OPE).",
      "Practical recipe: large context region on the training image, cosine window to suppress circular-boundary effects, final crop of the template.",
    ],
    method: {
      pipeline: ["convolve", "crop", "solve-cf", "correlate", "regress", "localize"],
      architecture:
        "Asymmetric Siamese network: a shared CNN f_ρ encodes a 255x255 training image and a larger search image (49x49x32 features at conv1); a CF block ω solves ridge regression on the training features and its template is cross-correlated with the test features to give a 17x17x32 (or 33x33x1) response, rescaled by learned s and b for logistic regression.",
      appearanceModel:
        "Learned CNN features whose filter is re-solved from the exemplar crop every frame in the Fourier domain; context region is large and the feature map is pre-multiplied by a cosine window.",
      optimization:
        "Offline SGD on logistic loss over millions of random pairs, with gradients flowing through the CF's dual solution (Eqs. 6-7) rather than an unrolled iterative solver; per-architecture tracking hyperparameters chosen by 300-iteration random search on a 129-video validation set.",
      loss: "Element-wise logistic loss over the dense response-map labels c_i in {-1, +1}.",
    },
    equations: [
      {
        id: "cfnet-network",
        label: "CFNet forward pass",
        formula: "h(x', z') = s · ω(f_ρ(x')) ⋆ f_ρ(z') + b",
        variables: [
          { symbol: "x', z'", meaning: "training (exemplar) image and larger test/search image" },
          { symbol: "f_ρ", meaning: "shared CNN with learnable parameters ρ" },
          { symbol: "ω(·)", meaning: "correlation filter block that solves ridge regression on its input" },
          { symbol: "s, b", meaning: "learned scalar scale and bias to make scores logistic-friendly" },
        ],
        intuition:
          "Look at the target once to build a discriminative template, then slide that template over the next frame — but let the feature extractor be trained for doing this.",
        why:
          "This is the object the network actually optimizes; a plain Siamese correlation is the special case without ω.",
        where: "Section 3.3 (end-to-end CF architecture), Eq. 3.",
        paperIds: ["T014"],
      },
      {
        id: "cfnet-ridge",
        label: "Correlation filter ridge objective",
        formula: "arg min_w  (1 / (2n)) · ||w ∗ x − y||² + (λ / 2) · ||w||²",
        variables: [
          { symbol: "w", meaning: "filter template to learn" },
          { symbol: "x", meaning: "input feature map (all circular shifts act as training samples)" },
          { symbol: "y", meaning: "desired Gaussian-shaped response peaked at the target" },
          { symbol: "λ, n", meaning: "regularisation weight and effective number of shifted examples" },
        ],
        intuition:
          "Fit a template that, correlated with the image, reproduces a peak at the target — using every circular shift as a free training example.",
        why:
          "It is the standard CF formulation whose solution the network must differentiate through; with circulant structure it is solvable in the Fourier domain.",
        where: "Section 3.4 (Correlation Filter), Eq. 5.",
        params: "λ controls overfitting to the exemplar; larger λ smooths the learned template.",
        simulator: "corr-filter",
        paperIds: ["T014"],
      },
      {
        id: "cfnet-dual",
        label: "Fourier-domain solution and its differential",
        formula:
          "k = (1/n)(x̂* ∘ x̂) + λ1 ,  k̂ ∘ α̂ = (1/n)ŷ ,  ŵ = α̂ ∘ x̂\ndk = (1/n)(dx̂* ∘ x̂ + x̂* ∘ dx̂) ,  dk̂ ∘ α̂ + k̂ ∘ dα̂ = (1/n)dŷ",
        variables: [
          { symbol: "k", meaning: "signal defining the circulant linear kernel matrix" },
          { symbol: "α", meaning: "Lagrange multipliers of the dual problem" },
          { symbol: "∘", meaning: "element-wise product (diagonalises the circulant system)" },
        ],
        intuition:
          "Solve the linear system once per frame in the Fourier domain, and differentiate the same system: all inverse operations collapse to element-wise divisions.",
        why:
          "This is the technical heart of the paper — without it the CF layer could not be trained end-to-end.",
        where: "Section 3.4, Eqs. 6-8; back-propagation maps detailed in the appendix.",
        paperIds: ["T014"],
      },
    ],
    datasets: ["otb", "vot", "others"],
    metrics: ["precision", "success-auc", "fps"],
    baselines: ["SiamFC-3s", "Staple", "LCT", "SAMF", "DSST", "Baseline (Siamese only)"],
    results: [
      "Table 1 (OTB toolkit, OPE): CFNet-conv2 61.1 IoU / 80.7 precision on OTB-2013, 53.0 / 70.2 on OTB-50, 56.8 / 74.8 on OTB-100; TRE on OTB-100 reaches 60.6 IoU / 79.1 precision.",
      "Speed: CFNet-conv1 83 fps, CFNet-conv2 75 fps, CFNet-conv5 43 fps; SiamFC-3s scores 60.7 / 81.0 on OTB-2013, so CFNet-conv2 beats it in overlap at comparable depth.",
      "CFNet-conv2 rivals Baseline-conv5 (61.8 IoU on OTB-2013) with roughly 30x fewer parameters.",
      "Validation uses 129 videos from VOT-2014, VOT-2016 and Temple-Color; hyperparameters tuned by 300 random-search iterations per architecture.",
    ],
    ablations: [
      "CFNet vs CFNet-const (α learned offline and frozen): CFNet is consistently better — back-propagating through the inverse-convolution solution is essential to beat the plain Siamese baseline.",
      "Feature source: end-to-end CFNet vs Baseline+CF (frozen Siamese features + CF) vs ImageNet+CF — only end-to-end training gives the accuracy/speed frontier of Fig. 7.",
      "Depth: conv1 < conv2 (best trade-off) < conv5 for accuracy; deeper is not automatically better once the CF is trainable.",
    ],
    limitations: {
      authorStated: [
        "The paper deliberately excludes orthogonal improvements (bounding-box regression, ensembling multiple cues, optical flow) to isolate the contribution of end-to-end CF training.",
        "Gains over the state of the art are described as only modestly superior; the point is the accuracy-at-high-speed frontier, not absolute leadership.",
        "A separate greyscale-trained network must be retrained for the few greyscale OTB sequences.",
        "The network only scores candidate locations; external tracking logic (initialisation, search region, scaling policy) is still required.",
      ],
      evident: [
        "Single-layer conv1 variants lose accuracy, showing the CF cannot compensate for weak features.",
        "The template is re-solved from one exemplar each frame — long-term occlusion and re-identification are out of scope.",
        "Per-architecture random-search tuning means reported numbers depend on heavy validation.",
      ],
    },
    assumptions: [
      "Target appearance in a single frame plus circular shifts is enough to re-learn the filter each frame.",
      "Translations dominate frame-to-frame motion; the search region around the previous position contains the target.",
    ],
    computation:
      "75-83 fps for the conv1/conv2 models (OTB toolkit timings); the CF solve is an FFT-sized element-wise operation, negligible next to the CNN.",
    relations: [
      {
        to: "T009",
        type: "builds-on",
        note: "Same fully-convolutional Siamese skeleton, but plain correlation is replaced by a trainable CF layer trained end-to-end.",
      },
      {
        to: "T001",
        type: "builds-on",
        note: "Uses the KCF-style ridge-regression template solved in the Fourier domain, and differentiates its closed-form solution.",
      },
      {
        to: "T004",
        type: "uses-as-baseline",
        note: "Staple is one of the real-time trackers compared in the OTB-2013/50/100 tables.",
      },
    ],
    concepts: ["correlation-filter", "siamese", "appearance-features", "online-vs-offline", "sot"],
    impact:
      "Established back-propagation through correlation-filter solvers as a usable primitive, and made CFNet the real-time deep baseline alongside SiamFC (T009) in OTB-era comparisons.",
  },
  {
    id: "T015",
    arxiv: "1705.06368",
    title: "Re3 : Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects",
    shortTitle: "Re3",
    year: 2017,
    authors: ["Daniel Gordon", "Ali Farhadi", "Dieter Fox"],
    venue: "IEEE Robotics and Automation Letters (RA-L)",
    fileName: "1705.06368v3.pdf",
    task: "single-object",
    tags: ["recurrent", "lstm", "regression", "self-training", "real-time", "occlusion"],
    difficulty: "intermediate",
    summary:
      "Re3 is a generic single-object tracker built on a two-layer factored LSTM that regresses the target box while carrying an appearance model in its recurrent state — tracking and model update happen in one forward pass at about 150 fps. It is pretrained on heterogeneous video plus synthetic sequences, then self-trained online, and its recurrent state makes it unusually robust to occlusion.",
    problem:
      "Most accurate trackers either retrain a discriminative model online (slow, prone to drift when updates are noisy) or use a fixed network trained for one short-term benchmark (no adaptation). There was no lightweight way to keep a temporal appearance memory that updates as cheaply as a forward pass and works for arbitrary object classes.",
    background: ["sot", "memory-network", "appearance-features", "occlusion", "online-vs-offline", "bounding-box"],
    previousWork: [
      {
        name: "GOTURN (T007)",
        limitation:
          "Feed-forward regressor with no temporal state: it cannot carry information across frames, so it must re-derive everything from the last crop and drifts under occlusion.",
        whyThisPaper:
          "Keeps the crop-regression interface but adds an LSTM whose cell state is the appearance model — ablation model A is exactly a GOTURN-style feed-forward net.",
      },
      {
        name: "MDNet (T003)",
        limitation:
          "Winner of VOT-2015 but requires online fine-tuning of its fully connected layers per sequence — very accurate, about 1 fps.",
        whyThisPaper:
          "Updates for free inside the recurrent state, reaching about 150 fps with competitive accuracy.",
      },
      {
        name: "Struck and MEEM style online SVM trackers",
        limitation:
          "Explicit model maintenance (support-vector pools, expert reweighting) is brittle and bounds the tracker to hand-designed update heuristics.",
        whyThisPaper:
          "Learns how to update from data: the LSTM is trained to keep its state useful over long unrolls.",
      },
    ],
    researchGap:
      "Before this paper, no tracker combined offline-pretrained generic features with a recurrent state that simultaneously localises and updates the appearance model in a single forward pass.",
    contribution: [
      "Two-layer factored LSTM regressor fed with consecutive crops, predicting box coordinates directly — claimed as the first algorithm using an RNN for generic-object tracking.",
      "Training recipe: supervised pretraining on ALOV + ImageNet Video + simulated sequences, then self-training on new videos (model D), with LSTM-state reset every 32 frames.",
      "Skip connections (after norm1, norm2, conv5 with 16/32/64 channels) and difference-style fusion of the two crops at the fully connected layer rather than at the input.",
      "Occlusion study: only 7.6% relative accuracy loss on occluded OTB frames vs 10.2% (MUSTer), 8.3% (MEEM), 16.1% (STRUCK).",
      "Runs at about 150 fps (full model 149.63 fps, up to 213 fps for single-layer variants) on a Titan X.",
    ],
    method: {
      pipeline: ["crop-pair", "convolve", "fuse", "lstm-step", "regress", "reset-state"],
      architecture:
        "CaffeNet convolutional trunk (pretrained weights, all layers fine-tuned) followed by a 2048-unit fully connected embedding that fuses current and previous crops, then two stacked factored LSTMs of 1024 units each, then a regression layer outputting the next bounding box in crop coordinates.",
      motionModel:
        "No explicit motion model; temporal context lives entirely in the LSTM cell state, carried frame to frame and reset to a well-centred crop embedding when the tracker loses confidence.",
      appearanceModel:
        "The LSTM state is the appearance model: it evolves as crops are consumed, so the tracker updates and localises in the same forward pass.",
      optimization:
        "ADAM, initial learning rate 1e-5 dropped to 1e-6 after 10k iterations, about 200k iterations total (roughly a week); random mirroring of whole tracks with p = 0.5; gradient unroll up to 32 frames.",
      loss: "Regression loss on the four box coordinates, trained with teacher-forced ground-truth outputs fed into future steps.",
    },
    equations: [
      {
        id: "re3-lstm",
        label: "Factored LSTM step",
        formula:
          "z^t = h(W_z x^t + R_z y^(t−1) + b_z)\nf^t = σ(W_f x^t + R_f y^(t−1) + P_f c^(t−1) + b_f)\nc^t = i^t ⊙ z^t + f^t ⊙ c^(t−1)\ny^t = o^t ⊙ h(c^t)",
        variables: [
          { symbol: "x^t", meaning: "embedding of the concatenated frame-t and frame-t−1 crops" },
          { symbol: "c^t, y^t", meaning: "cell state (memory) and output of the LSTM at time t" },
          { symbol: "f^t, i^t, o^t", meaning: "forget, input and output gates (σ = sigmoid)" },
          { symbol: "h", meaning: "tanh nonlinearity" },
        ],
        intuition:
          "The gates decide what of last frame's appearance memory to keep, what new evidence to write, and what to expose to the regressor — that retained memory is what survives occlusion.",
        why:
          "Puts the entire tracking state (appearance plus implicit motion) into a fixed-size vector that can be learned end-to-end from long sequences.",
        where: "Section 3 (network structure); two stacked layers of 1024 units each.",
        params:
          "States are reset every 32 training frames (and during lost-target resets during tracking) because the LSTM otherwise absorbs whole-sequence statistics.",
        paperIds: ["T015"],
      },
    ],
    datasets: ["vot", "otb", "others"],
    metrics: ["success-auc", "precision", "fps"],
    baselines: ["GOTURN", "MDNet", "MUSTer", "MEEM", "STRUCK", "DSST"],
    results: [
      "Ablation Table I, VOT-2014: full model (G) accuracy 0.66, robustness 0.92, average 0.789 at 149.63 fps; feed-forward GOTURN-style model A average 0.756 at 168.77 fps.",
      "ImageNet Video: full model accuracy 0.68 / robustness 0.97 / average 0.826 (model G); removing ImageNet Video data (model I) drops the average to 0.700 with 1096 drops.",
      "Occlusion experiment on OTB: Re3 loses only 7.6% relative accuracy on occluded frames, better than MUSTer (10.2%), MEEM (8.3%) and STRUCK (16.1%).",
      "Speed framing: 450x faster than the best methods in the VOT-2016 comparison while scoring only 20% and 5% lower in relative accuracy and robustness.",
    ],
    ablations: [
      "Table I ladder: feed-forward (A) to +ImageNet Video (B) to one-layer LSTM (C) to +self-training (D) to +simulated data (E) to +skip layers (F) to two LSTM layers (G); each step raises the VOT-2014 average.",
      "Data ablations: no ALOV (H) or no ImageNet Video (I) costs accuracy; no LSTM-state reset (J) drops the average from 0.789 to 0.705 on VOT-2014.",
      "Swapping CaffeNet for GoogleNet conv layers (K) improves accuracy (0.68 / 0.802) but halves speed to 77 fps.",
    ],
    limitations: {
      authorStated: [
        "Accuracy still trails the most accurate VOT-2016 entries — the paper explicitly frames the trade-off as 450x faster for 20%/5% lower accuracy/robustness.",
        "Training requires resetting the LSTM state every 32 frames; without it the LSTM parameters degrade.",
        "Self-training depends on the tracker staying correct, so early drift can poison the updated model.",
      ],
      evident: [
        "No explicit motion model or re-detection: after a long disappearance the LSTM state has no mechanism to re-initialise the target.",
        "Crop-centred input assumes the previous location is approximately right — a large search region is never examined.",
      ],
    },
    assumptions: [
      "The target stays inside the padded crop of the previous prediction (twice the crop size padding).",
      "A generic tracker can be pretrained across many object categories and adapted on the fly with its own predictions.",
    ],
    computation:
      "149.63 fps full model (213 fps single-layer) on Intel Xeon E5-2696 v4 + Titan X (Pascal); disk read excluded from timing.",
    relations: [
      {
        to: "T007",
        type: "uses-as-baseline",
        note: "The GOTURN-style feed-forward regressor is ablation model A in Table I, the starting point the recurrent state improves on.",
      },
    ],
    concepts: ["sot", "memory-network", "appearance-features", "occlusion", "online-vs-offline"],
    impact:
      "Demonstrated that recurrence can replace explicit online model maintenance in SOT — the idea that the state IS the model reappears in memory-network trackers, and Re3 became the reference fast recurrent baseline.",
  },
  {
    id: "T016",
    arxiv: "1707.02309",
    title: "Adaptive Correlation Filters with Long-Term and Short-Term Memory for Object Tracking",
    shortTitle: "Long/Short-Term Memory CF",
    year: 2017,
    authors: ["Chao Ma", "Jia-Bin Huang", "Xiaokang Yang", "Ming-Hsuan Yang"],
    venue: "International Journal of Computer Vision",
    fileName: "1707.02309v2.pdf",
    task: "single-object",
    tags: ["correlation-filter", "long-term-memory", "re-detection", "multi-cue", "online-learning"],
    difficulty: "intermediate",
    summary:
      "LMCF learns three correlation filters per target — an aggressively updated translation filter, a scale filter, and a conservatively updated long-term filter whose confidence score decides when an online SVM re-detector must take over. It combines short-term adaptation with an explicit memory that can recognise tracking failure and recover.",
    problem:
      "Adaptive CF trackers update their filter every frame with a high learning rate: this tracks appearance change well but the filter drifts on noisy updates, and because nothing keeps a long-term record of the target, a tracker that loses the target (heavy occlusion, out-of-view) has no way to notice or recover.",
    background: ["correlation-filter", "sot", "occlusion", "appearance-features", "success-plot", "tracking-by-detection"],
    previousWork: [
      {
        name: "KCF / DSST adaptive filters (T001)",
        limitation:
          "Single aggressively-updated filter: precise but drift-prone, no memory of what the target looked like before a failure, no failure detection.",
        whyThisPaper:
          "Splits memory into short-term (fast learning rate) and long-term (conservative) filters with different roles.",
      },
      {
        name: "MUSTer",
        limitation:
          "Maintains long-term memory via an HOG+SIFT hybrid but lacks explicit failure detection and a re-detection path; weaker under abrupt motion.",
        whyThisPaper:
          "Adds a long-term filter used purely as a confidence estimator plus an SVM detector that fires only when confidence collapses.",
      },
      {
        name: "TLD-style re-detection",
        limitation:
          "Runs detector and tracker in every frame — heavy — and its photographic matcher does not transfer well to general targets.",
        whyThisPaper:
          "Gate the detector with long-term confidence (threshold Tr = 0.15) so sliding-window detection only runs on suspected failures.",
      },
    ],
    researchGap:
      "Before this paper, correlation-filter trackers had no explicit separation between fast adaptation and long-term memory, and no built-in way to detect failure and reacquire the target.",
    contribution: [
      "Three-filter design: translation filter A_T (short-term, context-rich, HOG+HOI or deep features), scale filter A_S (HOG only, target region only), long-term filter A_L (conservative learning rate, used for failure detection).",
      "Online re-detector: incremental SVM over quantised CIE-LAB colour histograms (rank-transformed L channel), updated with a passive-aggressive scheme and activated only when max response of A_L drops below Tr = 0.15.",
      "Context engineering: enlargement ratio r = 2.8 (halved vertically for aspect ratio below 0.5), cosine window per channel, HOI features added to HOG as a complementary cue.",
      "Consistency-aware update: A_L only updates when its confidence exceeds Ts = 0.38, protecting the long-term memory from corrupt frames.",
      "State-of-the-art OTB numbers (Ours-deep: DP 87.8 on OTB-2013) at 14-21 fps.",
    ],
    method: {
      pipeline: ["crop", "translation-filter", "scale-pyramid", "confidence-check", "re-detect", "update"],
      architecture:
        "Handcrafted variant CT-HOGHOI: HOG + histogram-of-local-intensities features fed to three CFs; deep variant uses the conv5-4 layer of VGGNet-19 for the translation filter. A separate incremental SVM handles re-detection, all gated by the long-term filter's confidence.",
      appearanceModel:
        "Dual short/long-term memory: A_T and A_S update every frame with aggressive learning; A_L updates only when confident (Ts = 0.38) with a conservative rate, so it keeps a clean historical record.",
      detectionDependency:
        "Self-contained detector — an online SVM on colour histograms, samples labelled positive if IoU above 0.5 and negative if IoU below 0.1 around the current estimate.",
      trackManagement:
        "Failure = max(f_A_L(z)) below Tr (0.15) triggers re-detection; a candidate is accepted only if its long-term response exceeds Ta = 0.38, which relocates and re-initialises tracking.",
      optimization:
        "Filters solved as standard Fourier-domain ridge regression; SVM hyperplane updated by passive-aggressive updates (tau = 1); lambda = 1e-4, Gaussian kernel sigma = 0.1, sigma' = 0.1*sqrt(WH), long-term learning rate eta = 0.01; scale pyramid N = 21 levels, factor 1.03.",
    },
    equations: [
      {
        id: "lmcf-svm",
        label: "Online re-detection SVM objective",
        formula:
          "min_h (λ / 2)·||h||² + (1 / N)·Σ_i ℓ(h; (v_i, c_i)) ,  ℓ(h; (v, c)) = max{0, 1 − c·⟨h, v⟩}",
        variables: [
          { symbol: "h", meaning: "hyperplane separating target from background" },
          { symbol: "v_i, c_i", meaning: "colour-histogram feature vector and ±1 label of sample i" },
          { symbol: "λ", meaning: "regularisation (set to 1e-4)" },
        ],
        intuition:
          "A lightweight classifier that can be re-learned from a handful of samples every frame around the current position, used only when the tracker is lost.",
        why:
          "The long-term filter detects failure but should not itself re-localise; a separate detector gives robust re-acquisition at low cost.",
        where: "Section 4.6 (Online Detector), Eq. 21.",
        params:
          "Samples with IoU above 0.5 are positive, below 0.1 negative; only translated samples are drawn to save computation.",
        paperIds: ["T016"],
      },
      {
        id: "lmcf-pa",
        label: "Passive-aggressive SVM update",
        formula:
          "h ← h − ( ℓ(h; (v, c)) / ( ||∇_h ℓ||² + 1 / (2τ) ) ) · ∇_h ℓ",
        variables: [
          { symbol: "τ", meaning: "update-rate hyper-parameter (tau = 1)" },
          { symbol: "∇_h ℓ", meaning: "gradient of the hinge loss at the current sample" },
        ],
        intuition:
          "Push the decision boundary only as far as needed to fix the current mistake — more when the sample is misclassified, less when already correct.",
        why:
          "Avoids keeping a large support-vector pool: one bounded update per sample keeps re-detection cheap enough to run inside a tracker.",
        where: "Section 4.6, Eq. 22; applied only when confidence exceeds Ts.",
        paperIds: ["T016"],
      },
    ],
    datasets: ["otb"],
    metrics: ["precision", "success-auc", "fps"],
    baselines: ["MUSTer", "MEEM", "TGPR", "KCF", "DSST", "TLD"],
    results: [
      "Table 1, OTB2013 (I) / OTB2015 (II): Ours-deep DP 87.8 / 82.5, OS 79.9 / 73.9, CLE 15.4 / 24.3 px, 14.4 / 13.8 fps; Ours (handcrafted) DP 84.8 / 76.2, OS 81.3 / 70.1 at 21.6 / 20.7 fps.",
      "MUSTer in the same table: DP 86.5 / 77.4, OS 78.4 / 68.3 — LMCF's deep variant wins precision, handcrafted variant wins overlap success.",
      "OTB-2015 success AUC (Fig. 9): Ours-deep 0.592 (OPE) / 0.608 (TRE) / 0.548 (SRE) vs MUSTer 0.577 / 0.580 / 0.532.",
      "Distance precision OPE on OTB-2015: Ours-deep 0.848, Ours 0.848 (Fig. 12/13), against MEEM 0.830 and KCF 0.705.",
    ],
    ablations: [
      "Section 6.4 runs three ablation groups on OTB-2013 (translation-filter design, scale-filter features, re-detection scheme): CT-HOGHOI-VGGNet19 (deep translation + handcrafted scale) beats pure deep or pure handcrafted variants.",
      "Re-detector alternatives (Fig. 12): SVM with passive-aggressive update (0.848 DP) above SVM with support-vector update (0.837), long-term CF as detector (0.824), and LSTM hidden state as long-term filter (0.821).",
      "Supplementary context/scale ablation: placement of surrounding context varies precision from 0.756 (left) to 0.793 (bottomRight); scale-search ratio variants range 0.729 (scale08) to 0.791 (scale11) precision with 0.587 success for scale11.",
      "Threshold sensitivity (Fig. 13): acceptance threshold Ta around 0.35-0.38 is optimal — Ta = 0.45 drops precision to 0.823.",
    ],
    limitations: {
      authorStated: [
        "All hyper-parameters (Tr = 0.15, Ta = 0.38, Ts = 0.38, eta = 0.01, lambda = 1e-4, r = 2.8) are empirically determined and fixed throughout the experiments.",
        "The re-detector only draws translated samples to reduce computational burden — scale and rotation changes during re-detection are not searched.",
        "The deep variant runs at about 14 fps, below real-time; the handcrafted variant is needed for 20+ fps.",
      ],
      evident: [
        "Failure detection depends on a single scalar confidence threshold — gradual drift that keeps A_L confident above Tr is never caught.",
        "Each frame trains only around the current estimate, so a target that jumps far while hidden still requires the sliding-window detector to fire.",
      ],
    },
    assumptions: [
      "A conservatively updated filter retains enough target identity to score failures below 0.15.",
      "Colour-histogram statistics are stable enough to re-find a lost target.",
    ],
    computation:
      "14.4 fps (deep) / 21.6 fps (handcrafted) on OTB2013; re-detection adds cost only on suspected failures.",
    relations: [
      {
        to: "T001",
        type: "uses-as-baseline",
        note: "KCF is one of the tracked baselines in Table 1, and the kernelized-CF machinery is the short-term building block.",
      },
    ],
    concepts: ["correlation-filter", "occlusion", "appearance-features", "sot", "online-vs-offline"],
    impact:
      "Pioneered the short-term fast-learner plus long-term memory dual-filter pattern with confidence-gated re-detection inside the CF family — a design later long-term trackers adopt to recover from occlusion and out-of-view failures.",
  },
  {
    id: "T017",
    arxiv: "1712.01358",
    title: "Long-Term Visual Object Tracking Benchmark",
    shortTitle: "TLP",
    year: 2017,
    authors: ["Abhinav Moudgil", "Vineet Gandhi"],
    fileName: "1712.01358v4.pdf",
    task: "benchmark",
    tags: ["benchmark", "long-term", "dataset", "single-object", "evaluation"],
    difficulty: "intro",
    summary:
      "TLP (Track Long and Prosper) is a benchmark of 50 long HD videos (676K frames, 400+ minutes, mean sequence 484.8 s) built to expose what short benchmarks cannot: on TLP every tracker except MDNet falls below 25% success AUC, and rankings reshuffle dramatically compared with OTB-50.",
    problem:
      "Existing tracking benchmarks are made of short clips (OTB-100 averages 19.6 s per sequence), so slow drift, recovery behaviour and error accumulation are barely measured — two trackers that look identical on OTB can differ enormously over minutes of tracking.",
    background: ["sot", "benchmark-design", "success-plot", "occlusion", "bounding-box", "online-vs-offline"],
    previousWork: [
      {
        name: "OTB-50 / OTB-100",
        limitation:
          "Mean sequence duration of about 19 s; rankings from these short clips do not predict behaviour on long videos (Table 1 comparison).",
        whyThisPaper:
          "Provides sequences up to 953 s (24,240 s total) so drift and recovery can be measured at all.",
      },
      {
        name: "ALOV300++",
        limitation: "More than 300 short sequences averaging only about 9 seconds, annotated every fifth frame.",
        whyThisPaper: "TLP's mean duration (484.8 s) is over 50x longer, at full frame rate (24/30 fps).",
      },
      {
        name: "NFS (240 fps high-speed benchmark)",
        limitation:
          "High frame rate reduces per-frame appearance variation, so it is easier rather than longer-term.",
        whyThisPaper: "Keeps normal frame rates but extends time, targeting drift instead of motion blur.",
      },
    ],
    researchGap:
      "Before this paper, no benchmark systematically measured single-object trackers over long durations, so long-term drift and recovery were unquantified.",
    contribution: [
      "TLP dataset: 50 HD real-world videos, 676K frames, over 400 minutes total (24,240 s), average 484.8 s — more than 20x larger average duration and 8x total duration than generic benchmarks.",
      "Benchmark of 17 scalable state-of-the-art trackers with ranking by accuracy (success AUC, precision) and a new Longest Subsequence Measure (LSM).",
      "TinyTLP subset (short sequences cut from TLP) demonstrates that short-sequence rankings differ from full-TLP rankings.",
      "Attribute-wise evaluation by grouping sequences where a challenge dominates (illumination, occlusion, out-of-view, ...).",
      "Central observation: apart from MDNet, every evaluated tracker drops below 25% success AUC on TLP; rankings on OTB-50 do not transfer.",
    ],
    method: {
      pipeline: ["collect", "annotate", "protocol", "evaluate"],
      architecture:
        "Benchmark construction: curate 50 long real-world HD sequences with per-frame bounding boxes, define TinyTLP and attribute subsets, then evaluate 17 trackers with precision, success and LSM plots plus runtime ranking.",
      detectionDependency:
        "Trackers run as published (SOT, first-frame initialisation); no detections provided — a pure appearance/motion robustness test.",
      trackManagement:
        "LSM explicitly measures how long a tracker stays continuously on target, penalising trackers that re-acquire only after long failures.",
    },
    equations: [
      {
        id: "tlp-precision",
        label: "Precision plot",
        formula: "precision(τ) = #{frames : ||p − g||₂ < τ} / #{frames} ,  τ = 20 pixels",
        variables: [
          { symbol: "p", meaning: "predicted box centre" },
          { symbol: "g", meaning: "ground-truth box centre" },
          { symbol: "τ", meaning: "distance threshold in pixels (20 by convention)" },
        ],
        intuition: "What fraction of frames put the box within 20 pixels of the truth.",
        why: "Standard OTB-style localisation metric the benchmark adopts for comparability.",
        where: "Section 4 (Evaluation), reported alongside success and LSM.",
        paperIds: ["T017"],
      },
      {
        id: "tlp-success",
        label: "Success plot and AUC",
        formula: "success(t) = #{frames : IoU(pred, GT) > t} / #{frames} ,  AUC = integral over t in [0,1] of success(t)",
        variables: [
          { symbol: "t", meaning: "IoU threshold swept from 0 to 1" },
          { symbol: "IoU", meaning: "intersection-over-union of predicted and true boxes" },
        ],
        intuition: "One number for how much the predicted boxes overlap the truth across all strictness levels.",
        why: "The paper's headline ranking metric (MDNet 68.1% AUC on TinyTLP, 36.9% on TLP).",
        where: "Section 4; a frame with correct absence prediction counts as overlap 1, otherwise 0.",
        simulator: "metrics",
        paperIds: ["T017"],
      },
      {
        id: "tlp-lsm",
        label: "Longest Subsequence Measure",
        formula: "LSM = length of the longest continuous tracked subsequence / total sequence length",
        variables: [
          {
            symbol: "tracked subsequence",
            meaning: "run of frames in which x% of frames have IoU above 0.5",
          },
        ],
        intuition:
          "Rewards trackers that stay locked on for long uninterrupted stretches instead of flickering in and out of the target.",
        why:
          "Short benchmarks cannot see this: on long videos, breaking lock every few seconds ruins utility even if average overlap looks fine.",
        where: "Section 4 (metrics definition).",
        paperIds: ["T017"],
      },
    ],
    datasets: ["others", "otb"],
    metrics: ["precision", "success-auc", "fps"],
    baselines: ["MDNet", "SiamFC", "CREST", "ADNet", "GOTURN", "ECO"],
    results: [
      "Dataset scale: 50 videos, 676K frames, total 24,240 s, mean 484.8 s vs OTB-100's 19.6 s mean (Table 1).",
      "Success AUC: MDNet 68.183% on TinyTLP but only 36.9% on full TLP; ADNet 59.245% on TinyTLP — the gap between trackers widens drastically on long sequences.",
      "Per-tracker success rate drop: MDNet 83.4% to 42.1% (smallest relative drop); MOSSE 37.2% to 3.7% (more than 10x).",
      "Ranking instability: MEEM enters the top five on TLP ahead of ECO; BACF beats ECO by about 2% on TinyTLP but is 6% worse on TLP.",
      "Runtime-accuracy ranking reported for all 17 trackers; TinyTLP precision AUC led by MDNet at 73.694%.",
    ],
    ablations: [
      "TinyTLP vs TLP is the benchmark's own ablation: short subsets reproduce OTB-like rankings while full sequences expose drift (success curves start around 40-50% on TLP vs 80-90% on TinyTLP).",
      "Attribute-grouped subsets isolate which challenge (occlusion, out-of-view, illumination) causes each tracker's collapse.",
      "Tracker-family comparison: handcrafted CF trackers degrade far more than deep trackers on TLP (deep features slow error accumulation).",
    ],
    limitations: {
      authorStated: [
        "Only 17 trackers scalable to be evaluated on TLP were included — very slow trackers are absent.",
        "The benchmark is single-object only and provides no identity or multi-object protocol.",
        "The authors note dedicated research effort is still needed in long-term tracking — most current designs were tuned for short sequences.",
      ],
      evident: [
        "50 sequences, all real-world HD with no controlled synthetic variation — category coverage is much narrower than OTB-100.",
        "Attribute subsets are derived from the same 50 videos, so attribute statistics are correlated.",
      ],
    },
    assumptions: [
      "Long duration, not exotic categories, is the axis that separates trackers.",
      "Standard precision and success metrics remain meaningful when sequences are 50x longer.",
    ],
    computation:
      "Evaluation of 17 trackers over 676K frames; run-time is reported per tracker so cost can be compared alongside accuracy.",
    relations: [
      { to: "T003", type: "uses-as-baseline", note: "MDNet is the top-ranked tracker on both TinyTLP and TLP." },
      { to: "T009", type: "uses-as-baseline", note: "SiamFC is among the top performers on TLP and is analysed for drift." },
      { to: "T011", type: "uses-as-baseline", note: "ECO (best on OTB-100) slides markedly on TLP — a central finding." },
      { to: "T012", type: "uses-as-baseline", note: "BACF is compared on TinyTLP vs TLP to show short-sequence ranking instability." },
      { to: "T001", type: "uses-as-baseline", note: "KCF is one of the 17 evaluated trackers." },
    ],
    concepts: ["benchmark-design", "sot", "success-plot", "occlusion", "online-vs-offline"],
    impact:
      "Established long-duration evaluation as a first-class axis for SOT; later large benchmarks (LaSOT T024, GOT-10k T025) adopt the same argument that long sequences and dedicated training sets are needed.",
  },
  {
    id: "T018",
    arxiv: "1801.06729",
    title: "EnKCF: Ensemble of Kernelized Correlation Filters for High-Speed Object Tracking",
    shortTitle: "EnKCF",
    year: 2018,
    authors: ["Burak Uzkent", "YoungWoo Seo"],
    fileName: "1801.06729v1.pdf",
    task: "single-object",
    tags: ["correlation-filter", "ensemble", "particle-filter", "real-time", "handcrafted-features"],
    difficulty: "intro",
    summary:
      "EnKCF runs three specialised KCFs — a small-area translation filter, a scale filter, and a large-area translation filter — in a fixed rotation over five frames, coordinated by a particle filter, so every filter updates every fifth frame instead of all filters running every frame. It needs no offline training and reaches 340-416 fps while beating ECO and CCOT in success rate on OTB-100.",
    problem:
      "Running several filters (for translation, scale, context) on every frame is accurate but too slow for embedded use (prior multi-KCF designs cap at 50 fps or below), while a single KCF cannot handle scale variation and a drift-prone large search area at the same time.",
    background: ["correlation-filter", "sot", "motion-model", "appearance-features", "success-plot", "online-vs-offline"],
    previousWork: [
      {
        name: "KCF (T001)",
        limitation:
          "One filter with a fixed search area: weak on scale change, and enlarging padding to cope with motion lets the filter learn noisy background.",
        whyThisPaper:
          "Keeps KCF's closed-form ridge regression but schedules three variants with different paddings and roles.",
      },
      {
        name: "Two-KCF translation/scale trackers (Ma et al.)",
        limitation: "Apply more than one KCF per frame, degrading run-time to 50 fps or below.",
        whyThisPaper:
          "Deploy the KCFs in turn (one filter at a time) instead of together, preserving 300+ fps.",
      },
      {
        name: "ECO / C-COT",
        limitation:
          "State-of-the-art accuracy but 12.28 fps (CCOT) and 52.77 fps (ECO) — not suitable for embedded real-time deployment.",
        whyThisPaper:
          "Targets the no-offline-training, 300-450 fps regime while still outscoring them in OTB success rate.",
      },
    ],
    researchGap:
      "Before this paper, no tracker combined an ensemble of purpose-specific correlation filters with a scheduling scheme that keeps every-frame cost at that of a single KCF.",
    contribution: [
      "Three-KCF ensemble: R_tS (small-area translation), R_s (scale, target-only region), R_tL (large-area translation) alternating on a five-frame cycle, so each filter learns from every fifth frame.",
      "Particle filter around the CF for translation: predicts the prior mean, the CF refines the location, and the posterior mean becomes the reported box — reducing drift when switching filters.",
      "Features: fHoG + colour-naming for the translation filters, fHoG only for the scale filter; optional deep convolutional features explored.",
      "No offline training: all filters learn the target on the fly, enabling the 300-450 fps design target.",
      "Speed: implementation runs 340 fps on OTB-100 and 416 fps on UAV-123 (against 292 fps for the DCF/KCF references).",
    ],
    method: {
      pipeline: ["predict", "select-filter", "localise", "estimate-scale", "update", "refilter"],
      architecture:
        "Rotating schedule per Algorithm 1: frames i, i+1 use R_tL; frames i+2, i+3 use R_tS; every fifth frame uses R_s with scale pool {1.05, 1.0, 1/1.05}; a particle filter maintains the translation distribution before and after CF refinement.",
      motionModel:
        "Particle filter over (x, y, s): transit particles each frame, take the CF location as the observation, re-sample, and report the posterior mean as the box.",
      appearanceModel:
        "Three independent KCF models with exponential-moving-average updates — the large-area filter provides colour+shape context to recover drift introduced by scale-only frames.",
      optimization:
        "No training: closed-form Fourier-domain ridge solution per filter; the scale filter updates only when its peak response exceeds its threshold; individual filter learning rates kept small.",
    },
    equations: [
      {
        id: "enkcf-ridge",
        label: "KCF ridge regression",
        formula: "E_h = min_h (1/2)·||y − Σ_c h_c ∗ x_c||² + (λ/2)·Σ_c ||h_c||²",
        variables: [
          { symbol: "h_c", meaning: "multi-channel filter (c indexes HOG/colour channels)" },
          { symbol: "x_c, y", meaning: "training patch channels and desired continuous response" },
          { symbol: "λ", meaning: "regularisation weight" },
        ],
        intuition:
          "Learn a filter that responds strongly at the target and weakly on every circulant shift of the patch — all shifts are free training samples.",
        why: "This is the per-filter solve each of the three KCFs performs; it is what makes them cheap (FFT).",
        where: "Section 2, Eq. 1; closed-form frequency solution in Eq. 2.",
        params: "HOG+color-naming for translation filters; fHoG only for the scale filter.",
        simulator: "corr-filter",
        paperIds: ["T018"],
      },
      {
        id: "enkcf-schedule",
        label: "Alternating model update",
        formula: "α̂_t = (1 − β)·α̂_(t−1) + β·α̂ ,  x̂_t = (1 − β)·x̂_(t−1) + β·x̂",
        variables: [
          { symbol: "β", meaning: "small learning rate for the exponential moving average" },
          { symbol: "α̂, x̂", meaning: "dual solution and template in the Fourier domain" },
        ],
        intuition:
          "Each filter drifts only slowly toward the latest observation; because a filter is refreshed every fifth frame, staleness is bounded by the schedule.",
        why:
          "Scheduling rather than simultaneous updates is the mechanism that keeps run-time at single-KCF cost while an ensemble still runs.",
        where: "Section 2, Eqs. 6-7 (model update), Algorithm 1 (schedule).",
        paperIds: ["T018"],
      },
    ],
    datasets: ["otb", "uav123"],
    metrics: ["precision", "success-auc", "fps"],
    baselines: ["KCF", "DCF", "DSST", "CCOT", "ECO", "SAMF"],
    results: [
      "OTB-100: 70.10% precision at 20 px and 53.00% success rate — above CCOT (0.498) and ECO (0.491) as plotted, with SAMF at 0.403.",
      "UAV-123: 54.50% precision and 40.2% success.",
      "Speed: 340 fps implementation on OTB-100, 416 fps on UAV-123, against 292 fps for the DCF/KCF comparison points; design goal 300-450 fps.",
      "Comparison figure reports CCOT at 12.28 fps, ECO at 52.77 fps and SAMF at 5.26 fps — EnKCF trades their machinery for an order of magnitude more speed with higher OTB success.",
    ],
    ablations: [
      "Feature assignment per filter: translation filters use fHoG + colour-naming, scale filter fHoG only — colour helps large-area discrimination but not scale estimation.",
      "Simultaneous vs alternating deployment: running multiple KCFs each frame (as prior work) caps at 50 fps; the rotation preserves 300+ fps with the same ensemble diversity.",
      "Deep convolutional features explored at different abstraction levels as an optional drop-in for the same schedule.",
    ],
    limitations: {
      authorStated: [
        "Assumes appearance does not change drastically across consecutive frames — this is what makes updating each filter only every fifth frame acceptable.",
        "Compared mainly against high-speed (300+ fps) trackers; accuracy comparisons to the slow deep state of the art are contextual.",
        "Scale search uses a small candidate pool (3 candidates per step), so very fast scale changes are handled coarsely.",
      ],
      evident: [
        "Drift is handled by a particle filter and the large-area filter rather than by re-detection — after a long disappearance there is no recovery mechanism.",
        "No offline training means nothing prevents target/background confusion if the first frames are unrepresentative.",
      ],
    },
    assumptions: [
      "Target scale and translation change slowly enough that a five-frame refresh period suffices.",
      "A particle filter's unimodal motion prior tracks the target between CF updates.",
    ],
    computation:
      "340 fps (OTB-100) and 416 fps (UAV-123) in the authors' implementation; handcrafted features only, CPU-friendly.",
    relations: [
      {
        to: "T001",
        type: "builds-on",
        note: "All three filters are KCF instances; the paper extends KCF's ridge-regression solver with scheduling and a particle filter.",
      },
      {
        to: "T011",
        type: "uses-as-baseline",
        note: "ECO (0.491) and CCOT (0.498) are plotted against EnKCF's 0.530 OTB-100 success rate.",
      },
    ],
    concepts: ["correlation-filter", "sot", "motion-model", "appearance-features", "online-vs-offline"],
    impact:
      "Showed that filter ensembling and real-time speed are not in conflict when filters are scheduled rather than stacked — a systems idea influential for embedded and UAV tracking where the no-training, 300+ fps regime is required.",
  },
  {
    id: "T019",
    arxiv: "1803.08679",
    title: "Learning Spatial-Temporal Regularized Correlation Filters for Visual Tracking",
    shortTitle: "STRCF",
    year: 2018,
    authors: ["Feng Li", "Cheng Tian", "Wangmeng Zuo", "Lei Zhang", "Ming-Hsuan Yang"],
    fileName: "1803.08679v1.pdf",
    task: "single-object",
    tags: ["correlation-filter", "spatial-regularization", "temporal-regularization", "admm", "online-learning"],
    difficulty: "advanced",
    summary:
      "STRCF fixes SRDCF's two big problems — boundary effects and slow online updates — by adding a temporal regulariser that keeps the filter close to last frame's filter instead of training on a growing image buffer. Solved by ADMM, it gives a 5x speedup over SRDCF while gaining 5.4% AUC on OTB-2015, and its deep variant reaches 68.3% AUC.",
    problem:
      "Spatially regularised DCF (SRDCF) suppresses boundary effects but must be learned from multiple historical training images, forcing slow Gauss-Seidel updates at 5-6 fps; its naive sample weighting also overfits recent frames, so a corrupted frame (occlusion) poisons the model.",
    background: ["correlation-filter", "sot", "occlusion", "appearance-features", "success-plot", "online-vs-offline"],
    previousWork: [
      {
        name: "SRDCF (T010)",
        limitation:
          "Multiple-sample formulation makes online updating expensive (Gauss-Seidel, about 5.8 fps) and sample re-weighting overfits recent corrupted frames.",
        whyThisPaper:
          "Replaces the image-history buffer with a single temporal penalty on the previous frame's filter, motivated by online passive-aggressive learning.",
      },
      {
        name: "KCF (T001)",
        limitation:
          "No spatial regularisation: circular shifts leave strong boundary effects that degrade discrimination.",
        whyThisPaper: "Keeps KCF's single-sample Fourier efficiency while importing SRDCF's spatial penalty.",
      },
      {
        name: "C-COT / ECO (T011)",
        limitation:
          "Continuous-convolution operators and GMM sample selection are complex and slow (C-COT runs at 0.8 fps).",
        whyThisPaper:
          "Delivers comparable or better accuracy with a plain single-sample ADMM solver — handcrafted STRCF runs at real-time.",
      },
    ],
    researchGap:
      "Before this paper, spatially regularised DCFs could not be updated efficiently online or robustly to corrupted frames — temporal regularisation on a single sample was missing.",
    contribution: [
      "STRCF model: single-sample DCF loss + SRDCF-style spatial penalty + temporal penalty to the previous frame's filter, interpreted as an extension of the online passive-aggressive algorithm to regression over all circular shifts.",
      "ADMM solver with augmented Lagrangian (f = g splitting), alternating a closed-form slack update with a Fourier-domain filter update — complexity O(DMN) via Sherman-Morrison, versus SRDCF's slow Gauss-Seidel.",
      "Real-time handcrafted tracker: STRCF (HOG) 31.5 fps, STRCF (HOG+CN) 24.3 fps versus SRDCF's 5.8 fps (4.2x).",
      "DeepSTRCF: VGG-M conv3 + HOG+CN features, mean OP 84.2% on OTB-2015, better than DeepSRDCF (76.8%) and close to ECO (85.5%).",
      "Robustness: on out-of-view and occlusion attributes STRCF beats SRDCF by 5.8% and 7.6% — the temporal term passively resists corrupted updates.",
    ],
    method: {
      pipeline: ["crop", "spatial-temporal-solve", "correlate", "localize", "scale-search", "update"],
      architecture:
        "Single-sample spatially-temporally regularised DCF: features (HOG, colour names, optionally VGG conv3) from the current frame plus the previous filter are combined under one convex objective solved by ADMM each frame.",
      motionModel:
        "No explicit motion model; the temporal regulariser on the filter difference acts as a filter-space motion prior (small changes frame to frame).",
      appearanceModel:
        "Filter history replaces sample history: the model stays close to the last filter, so a single bad frame cannot drag it far.",
      optimization:
        "ADMM: introduce g with f = g, form the augmented Lagrangian, alternate the closed-form slack subproblem and the Fourier-domain filter subproblem; penalty factor adapted by gamma = min(gamma_max, rho*gamma).",
    },
    equations: [
      {
        id: "strcf-objective",
        label: "STRCF objective",
        formula:
          "arg min_f  (1/2)·||Σ_d x^d ∗ f^d − y||² + (1/2)·Σ_d ||w · f^d||² + (μ/2)·||f − f_(t−1)||²",
        variables: [
          { symbol: "x^d, f^d", meaning: "d-th feature channel and its filter (D channels total)" },
          { symbol: "y", meaning: "Gaussian desired response" },
          { symbol: "w", meaning: "spatial penalty weight map suppressing filter values near boundaries" },
          { symbol: "μ", meaning: "temporal regularisation strength" },
          { symbol: "f_(t−1)", meaning: "filter learned at the previous frame" },
        ],
        intuition:
          "Fit the current frame's response, keep the filter small near the patch borders, and don't move far from yesterday's filter — one bad frame costs only a bounded change.",
        why:
          "It is the whole model: it removes SRDCF's multi-image buffer while retaining both regularisers, making online updates both cheap and robust.",
        where: "Section 3.2, Eq. 2.",
        params:
          "mu trades adaptation against stability: larger mu means slower but more occlusion-robust updates; w is fixed and grows toward patch boundaries.",
        paperIds: ["T019"],
      },
      {
        id: "strcf-admm",
        label: "Augmented Lagrangian (ADMM) form",
        formula:
          "L(f, g, h) = (1/2)||Σ_d x^d ∗ f^d − y||² + (1/2)Σ_d ||w · g^d||² + (γ/2)Σ_d ||f^d − g^d + h^d||² + (μ/2)||f − f_(t−1)||²",
        variables: [
          { symbol: "g", meaning: "auxiliary copy of the filter enforcing f = g" },
          { symbol: "h", meaning: "scaled dual variable (h = s/gamma)" },
          { symbol: "γ", meaning: "penalty factor, adapted each iteration" },
        ],
        intuition:
          "Split the problem so each piece is easy: a closed-form projection for the penalty term and a Fourier-domain solve for the regression term.",
        why:
          "ADMM converges on this convex problem with a globally optimal solution, unlike SRDCF's low-convergence Gauss-Seidel — this is where the 5x speedup comes from.",
        where: "Section 3.3, Eqs. 3-5; complexity analysis: O(DMN) using Sherman-Morrison.",
        paperIds: ["T019"],
      },
    ],
    datasets: ["otb", "vot", "others"],
    metrics: ["success-auc", "precision", "eao", "fps"],
    baselines: ["SRDCF", "SRDCFdecon", "BACF", "ECO-HC", "C-COT", "DeepSRDCF"],
    results: [
      "Table 1 (OTB-2015, handcrafted): STRCF(HOG) mean OP 79.2 at 31.5 fps, STRCF(HOGCN) 79.6 at 24.3 fps — versus SRDCF 72.7 at 5.8 fps, BACF 77.5 at 26.7, ECO-HC 79.6 at 42, SRDCFdecon 77 at 2.0.",
      "Success AUC on OTB-2015: STRCF 65.1%, beating SRDCF by 5.4% and SRDCFdecon by 2.3%, second only to ECO-HC.",
      "Abstract claims: handcrafted STRCF gains 5.4% and 3.6% AUC over SRDCF on OTB-2015 and Temple-Color with a 5x speedup; DeepSTRCF reaches 68.3% AUC on OTB-2015.",
      "Table 2 (deep features, OTB-2015): DeepSTRCF mean OP 84.2 at 5.3 fps versus ECO 85.5, C-COT 82.7, DeepSRDCF 76.8, CF-Net 73.",
      "VOT-2016 (Table 3): STRCF EAO 0.279 / accuracy 0.53 / robustness 1.32; DeepSTRCF EAO 0.313 versus DeepSRDCF 0.276 (+3.7%); ECO 0.375.",
    ],
    ablations: [
      "SRDCF decomposition experiment: removing multi-sample formulation, scale estimation and spatial regularisation separately shows where SRDCF's cost and accuracy come from — temporal regularisation recovers the accuracy at single-sample cost.",
      "Attribute study: STRCF's gain over SRDCF is largest on out-of-view (14.5%) and occlusion (5.7%) — temporal regularisation matters exactly where updates get corrupted.",
      "Feature ladder HOG (79.2) to HOG+CN (79.6) to VGG-M conv3 / DeepSTRCF (84.2 mean OP): feature choice and speed trade off explicitly.",
    ],
    limitations: {
      authorStated: [
        "DeepSTRCF runs at only 5.3 fps — the deep variant is not real-time; only handcrafted STRCF achieves real-time speed.",
        "The method does not use continuous or factorised convolution; the paper positions this as simplicity, noting ECO-HC still matches it on OTB-2015.",
        "A single sample per frame means long-term appearance history beyond the previous filter is not modelled.",
      ],
      evident: [
        "Temporal regularisation slows legitimate appearance change too — rapid deformation is smoothed by the mu term.",
        "ADMM iterations per frame bound speed; with deep channels the O(DMN) solve and feature extraction dominate cost.",
      ],
    },
    assumptions: [
      "Filter changes between consecutive frames are small (the passive-aggressive bound on drift holds).",
      "Convexity of the objective means one global optimum per frame suffices — no multi-modal appearance memory.",
    ],
    computation:
      "31.5 fps (HOG) and 24.3 fps (HOG+CN) on CPU; DeepSTRCF 5.3 fps; SRDCF baseline 5.8 fps — the solver, not the features, is the efficiency claim.",
    relations: [
      {
        to: "T010",
        type: "extends",
        note: "Imports SRDCF's spatial penalty and replaces its multi-image formulation with a temporal penalty on the previous filter.",
      },
      {
        to: "T011",
        type: "uses-as-baseline",
        note: "ECO and ECO-HC are the accuracy references on OTB-2015 and VOT-2016.",
      },
      {
        to: "T012",
        type: "uses-as-baseline",
        note: "BACF (mean OP 77.5) is compared against STRCF (79.2) at similar speed.",
      },
      {
        to: "T014",
        type: "uses-as-baseline",
        note: "CF-Net appears in the deep-feature comparison table (mean OP 73 versus DeepSTRCF 84.2).",
      },
    ],
    concepts: ["correlation-filter", "online-vs-offline", "appearance-features", "sot", "occlusion"],
    impact:
      "STRCF became the standard efficient SRDCF — later benchmarks (GOT-10k T025, which reports STRCF at mAO 0.377) treat it as the representative modern handcrafted CF, and its trust-yesterday's-model temporal regulariser is widely reused in online DCF updates.",
  },
  {
    id: "T020",
    arxiv: "1804.08208",
    title: "High Performance Visual Tracking with Circular and Structural Operators",
    shortTitle: "CSOT",
    year: 2018,
    authors: ["Peng Gao", "Yipeng Ma", "Ke Song", "Chao Li", "Fei Wang", "Liyi Xiao", "a.o."],
    fileName: "1804.08208v3.pdf",
    task: "single-object",
    tags: ["correlation-filter", "structured-output-svm", "ensemble", "deep-learning", "collaborative-optimization"],
    difficulty: "advanced",
    summary:
      "CSOT unifies the two biggest filter families: it keeps the structured-output SVM's margin-violation objective but computes it with the DCF's circular correlation operator, then fuses multi-layer confidence maps with a relative-entropy ensemble post-processor. An online collaborative optimisation (alternating closed-form and conjugate-gradient updates) makes the expensive SOSVM objective tractable, yielding 71.5% and 69.4% mean AUC on OTB-2013 and OTB-2015.",
    problem:
      "SOSVM trackers are discriminative but slow — sampling strategies and batch optimisation add computational load that rules out real-time use, and without deep features they fail on occlusion and scale variation. DCF trackers are fast but restricted to plain ridge regression with weaker discriminative ability.",
    background: ["correlation-filter", "sot", "appearance-features", "success-plot", "online-vs-offline", "occlusion"],
    previousWork: [
      {
        name: "SOSVM trackers (Struck, DLSSVM lineage)",
        limitation:
          "Strong margin-based discrimination, but sampling and optimisation strategy add computational load that rules out real-time use.",
        whyThisPaper:
          "Replaces explicit sampling with circular correlation, so all shifted samples are handled in the frequency domain.",
      },
      {
        name: "KCF / DCF (T001)",
        limitation:
          "Only ridge regression with a Gaussian label — weaker discriminative ability than the structured hinge-style objective.",
        whyThisPaper: "Ports the structured SVM objective onto the circular operator, keeping ridge-level speed.",
      },
      {
        name: "SRDCF (T010) and deep DCF trackers (T011)",
        limitation:
          "Deep variants gain accuracy but multi-layer features have different resolutions and need fusion machinery; boundary effects remain.",
        whyThisPaper:
          "Adds a spatial penalty (gamma), implicit interpolation to the continuous domain for multi-resolution fusion, and a relative-entropy ensemble.",
      },
    ],
    researchGap:
      "Before this paper, the structured-output SVM objective and the circular correlation operator were never combined, so no tracker had both strong discrimination and DCF-level efficiency.",
    contribution: [
      "Circular and structural operators: structural correlation filters trained under an SOSVM objective but evaluated by circular correlation with interpolated feature maps (Eqs. 1, 5).",
      "Online collaborative optimisation: alternate a closed-form slack update (Eq. 9) with a conjugate-gradient solve for w in the Fourier domain (Eq. 11); three iterations converge, 25 CG iterations on the first frame.",
      "Ensemble post-processor: coalesce the per-layer primal confidence maps by relative entropy into one optimal response for peak localisation.",
      "Variant ladder: handcrafted CSOT-HC 61.9 mean AUC, plus deep appearance (CSOT-CNN) 66.8, plus deep motion (CSOT-CNN2) 67.9, plus ensemble (CSOT) 68.6.",
      "Headline results: mean AUC 71.5% (OTB-2013), 69.4% (OTB-2015), 64.9% (OTB-50) and 29.8% EAO on VOT-2017.",
    ],
    method: {
      pipeline: ["extract-features", "interpolate", "circular-correlate", "collaborative-update", "ensemble-fuse", "localize"],
      architecture:
        "Four blocks: (1) multi-layer CNN feature extractor, (2) circular plus structural operator producing one primal confidence map per layer, (3) relative-entropy ensemble post-processor, (4) online collaborative optimiser updating the structural filters.",
      appearanceModel:
        "Structural correlation filters per channel with a spatial penalty; deep appearance and (via optical flow) deep motion features both enter, all resampled to a common resolution by cubic-spline implicit interpolation in the style of C-COT.",
      optimization:
        "Collaborative scheme: given w, the slack has a closed-form solution max{0, S(y0) − S(Y_t) − J(Y_t)}; given slack, solve the regularised Fourier least-squares for w by conjugate gradient (25 iterations on the first frame; three outer iterations converge).",
    },
    equations: [
      {
        id: "csot-sosvm",
        label: "Structural SVM objective with circular scores",
        formula:
          "min_w Σ_d ||w^d||² + C·Σ_p ξ_i(y_i^(p))\ns.t. ξ_i(y_i^(p)) ≥ J_i(y_i^(p)) − ( S_i(y_i^(0)) − S_i(y_i^(p)) )",
        variables: [
          { symbol: "w^d", meaning: "structural correlation filter for channel d" },
          { symbol: "S_i(y)", meaning: "primal confidence score of cyclically shifted sample y" },
          {
            symbol: "J_i(y)",
            meaning: "cost of predicting y instead of the true position (1 minus a Gaussian desired map m_x)",
          },
          { symbol: "ξ, C", meaning: "slack variables and their regularisation weight" },
        ],
        intuition:
          "Require the score at the true location to beat every wrong location by at least its cost — a margin condition evaluated cheaply because all locations are circular shifts.",
        why:
          "This is the discrimination boost over ridge regression, made affordable by the circular operator.",
        where: "Section 3.1, Eqs. 2-3 (spatial penalty added in Eq. 6).",
        params: "C biases the problem toward margin maximisation versus training error; m_x is a Gaussian peaked at the true location.",
        paperIds: ["T020"],
      },
      {
        id: "csot-score",
        label: "Circular confidence score (Fourier form)",
        formula: "S^i(Y_t) = F^-1( Σ_d ŵ^d ∘ Φ̂^d ϕ(x_(i+1), y_i^(0)) )",
        variables: [
          { symbol: "Φ", meaning: "implicit interpolation operator mapping discrete features to the continuous domain" },
          { symbol: "ŵ^d", meaning: "Fourier coefficients of the structural filter" },
          { symbol: "∘", meaning: "element-wise product (circular correlation diagonalises in Fourier space)" },
        ],
        intuition:
          "Every layer's features are resampled to one resolution, multiplied by its filter in the frequency domain, and inverse-transformed into a response map.",
        why: "Makes multi-resolution deep features composable — a prerequisite for the ensemble post-processor.",
        where: "Section 3.1, Eqs. 4-5.",
        paperIds: ["T020"],
      },
      {
        id: "csot-collab",
        label: "Collaborative slack update (closed form)",
        formula: "ε_i(Y_t) = max{ 0, S_i(y_i^(0)) − S_i(Y_t) − J_i(Y_t) }",
        variables: [
          { symbol: "ε_i(Y_t)", meaning: "slack variable for each shifted training sample" },
          { symbol: "S_i(y_i^(0))", meaning: "score at the true target position" },
        ],
        intuition:
          "Samples that already satisfy the margin contribute nothing; only margin violators drive the next filter update.",
        why:
          "Decomposes the joint problem into one trivial step and one Fourier least-squares step, which is what makes SOSVM-style training run online.",
        where: "Section 3.2, Eqs. 7-10; the w-update then uses Eq. 11 with conjugate gradient.",
        paperIds: ["T020"],
      },
    ],
    datasets: ["otb", "vot"],
    metrics: ["success-auc", "eao", "fps"],
    baselines: ["ECO", "C-COT", "DeepLMCF", "DeepSRDCF", "DLSSVM", "MEEM"],
    results: [
      "Headline: mean AUC of 71.5% (OTB-2013) and 69.4% (OTB-2015), 64.9% on OTB-50, and 29.8% EAO on VOT-2017.",
      "Fig. 3 legends: CSOT AUC 0.715 / 0.694 / 0.649 on OTB-2013 / OTB-2015 / OTB-50 at 6.1-6.3 fps.",
      "Ablation ladder mean AUC (averaged over the three OTB sets): CSOT-HC 61.9, CSOT-KHC 62.7, CSOT-CNN 66.8, CSOT-CNN2 67.9, CSOT 68.6 (+6.7 points over HC).",
      "Compared against nine trackers (ECO, C-COT, DeepLMCF, DeepSRDCF, DLSSVM, MEEM, Staple, MFCMT, Struck) on OTB-50/2013/2015.",
    ],
    ablations: [
      "Kernel nonlinearity (CSOT-KHC): only +0.8 mean AUC while halving speed from 47.1 to 21.6 fps — linear structural filters preferred.",
      "Deep appearance features (CSOT-CNN) add +4.9 mean AUC over handcrafted; deep motion features (CSOT-CNN2) add about +1.1 more.",
      "Ensemble post-processor (CSOT versus CSOT-CNN2): +0.7 mean AUC (68.6 versus 67.9), confirming relative-entropy fusion helps localisation.",
      "Spatial regularization and implicit interpolation are enabled in all variants except CSOT-DCNN (which runs 10.4 fps, faster but worse).",
    ],
    limitations: {
      authorStated: [
        "The full deep CSOT runs at about 6.3 fps — real-time capability is sacrificed for accuracy; the handcrafted CSOT-HC runs 47.1 fps.",
        "The nonlinear kernel extension was found not helpful to improve results satisfactorily and reduces speed, so it was dropped.",
        "Conjugate-gradient iterations are needed each frame for the filter subproblem — no closed form exists once spatial regularization and interpolation are used.",
      ],
      evident: [
        "The relative-entropy ensemble assumes each layer's confidence map is informative; a corrupted layer biases the fused peak.",
        "Motion features depend on optical flow, adding a failure mode under fast motion and blur.",
      ],
    },
    assumptions: [
      "Cyclic shifts of one patch are adequate training samples (the DCF assumption carried into the SOSVM objective).",
      "Multi-layer confidence maps are commensurable after interpolation.",
    ],
    computation:
      "47.1 fps (handcrafted) down to 6.3 fps (full deep); 25 CG iterations on frame 1, three collaborative iterations per frame thereafter; spatial bandwidth sigma = 0.1.",
    relations: [
      { to: "T011", type: "uses-as-baseline", note: "ECO and C-COT are among the nine compared trackers." },
      { to: "T010", type: "uses-as-baseline", note: "DeepSRDCF is compared, and SRDCF's spatial-penalty idea is borrowed (gamma)." },
    ],
    concepts: ["correlation-filter", "appearance-features", "sot", "online-vs-offline", "success-plot"],
    impact:
      "Showed structured-output objectives could live inside DCF machinery — the circular/structural operator design and entropy-based response fusion were reused in later hybrid trackers, and CSOT remains the SOSVM-versus-DCF bridge cited in surveys.",
  },
  {
    id: "T021",
    arxiv: "1807.11348",
    title: "Learning Adaptive Discriminative Correlation Filters via Temporal Consistency Preserving Spatial Feature Selection for Robust Visual Tracking",
    shortTitle: "LADCF",
    year: 2018,
    authors: ["Tianyang Xu", "Zhen-Hua Feng", "Xiao-Jun Wu", "Josef Kittler"],
    venue: "IEEE Transactions on Image Processing (TIP), 2019",
    fileName: "1807.11348v3.pdf",
    task: "single-object",
    tags: ["correlation-filter", "feature-selection", "structured-sparsity", "admm", "temporal-consistency"],
    difficulty: "advanced",
    summary:
      "LADCF fights two DCF problems at once — the boundary effect and filter degradation over time — by selecting only the most discriminative spatial features (structured group-lasso sparsity across channels) while constraining that selection to stay close to the previous frame's layout. Joint spatial-temporal learning runs on a low-dimensional manifold via an augmented-Lagrangian solver, giving 66.4% AUC (handcrafted) and 69.6% with deep features on OTB-100.",
    problem:
      "Standard DCFs waste capacity on every spatial location, so filter coefficients near the patch boundary encode background and create boundary effects; online updates let the filter degrade as noisy samples accumulate. Existing fixes (spatial penalties, channel compression) treat location importance as fixed rather than learned, and do not constrain how the filter evolves over time.",
    background: ["correlation-filter", "sot", "appearance-features", "success-plot", "online-vs-offline"],
    previousWork: [
      {
        name: "SRDCF (T010) and BACF (T012)",
        limitation:
          "Impose hand-designed spatial penalties (or centre cropping) to suppress boundary coefficients — spatial importance is fixed a priori rather than selected.",
        whyThisPaper:
          "Learns which spatial features to keep with a structured lasso, initialised from BACF's optimisation.",
      },
      {
        name: "ECO / C-COT (T011)",
        limitation:
          "Channel compression reduces cost but discards channels wholesale; no temporal constraint keeps the filter near its previous state.",
        whyThisPaper:
          "Jointly selects spatial features and constrains them to the previous template's neighbourhood — filtering and adaptation in one objective.",
      },
      {
        name: "SiamFC / CFNet (T009, T014)",
        limitation:
          "Siamese trackers never retrain a filter online, so they cannot adapt within the sequence (cited as the opposite extreme).",
        whyThisPaper: "Stays in the adaptive-DCF regime but makes adaptation structured and local in feature space.",
      },
    ],
    researchGap:
      "Before this paper, DCF trackers had no mechanism to learn which spatial features are worth keeping while simultaneously preventing the filter from drifting from its previous state.",
    contribution: [
      "Temporal-consistency-preserving spatial feature selection: lasso (lambda1) selects features while an l2-ball constraint around the previous template (lambda2 much larger than lambda1) keeps the layout stable across frames (Eq. 8).",
      "Multi-channel group sparsity: one shared index vector over all L channels gives a group-lasso term — a global spatial layout for HOG, colour names and deep features (Eq. 9).",
      "Unified augmented-Lagrangian optimisation solved by ADMM with alternating direction multipliers — reported complexity O(KLD^2 log D).",
      "Adaptive dimensionality reduction: feature selection removes about 95% of spatial positions for handcrafted features (r = 5%) and 80% for deep features (r = 20%).",
      "Results: LADCF (HOG+CN) 66.4% AUC and 86.4% DP on OTB-100; LADCF* (HOG+CN+conv3) 69.6% AUC and 90.6% DP; VOT-2018 EAO 0.338 with VGG and 0.389 with the deeper backbone.",
    ],
    method: {
      pipeline: ["extract-features", "select-features", "solve-admm", "correlate", "localize", "scale-search", "update"],
      architecture:
        "Spatially selected multi-channel DCF: features pass through a selection mask (grouped across channels) before the correlation; fDSST handles scale; the mask and filters are updated jointly each frame.",
      appearanceModel:
        "Selected-feature template kept within an l2 neighbourhood of the previous template — the estimate lies on a low-dimensional manifold around the old model.",
      optimization:
        "Augmented Lagrangian with alternating (ADMM) updates; the filter is initialised as in BACF (theta_model set after the BACF-style step); lasso weight lambda1 and consistency weight lambda2 with lambda1 much smaller than lambda2.",
      loss:
        "Least-squares correlation loss plus lambda1 times the l1 norm of the filter (spatial sparsity) plus lambda2 times the squared distance to the previous template (temporal consistency).",
    },
    equations: [
      {
        id: "ladcf-selection",
        label: "Temporal-consistency feature selection",
        formula: "arg min_θ ||θ ∗ x − y||² + λ1·||θ||₁ + λ2·||θ − θ_model||₂²",
        variables: [
          { symbol: "θ", meaning: "filter (its support defines which spatial features are selected)" },
          { symbol: "x, y", meaning: "input patch and desired Gaussian response" },
          { symbol: "θ_model", meaning: "the previous frame's template — the centre of the consistency ball" },
          { symbol: "λ1, λ2", meaning: "sparsity weight and temporal-consistency weight (lambda1 much smaller than lambda2)" },
        ],
        intuition:
          "Keep only useful locations, and don't let the kept set move much from frame to frame — sparsity selects, consistency stabilises.",
        why:
          "Removes boundary/background coefficients (fixing the boundary effect) while stopping gradual filter degradation caused by unconstrained online updates.",
        where: "Section III, Eqs. 7-8 (l1 relaxation of the l0-ball constraint).",
        params:
          "lambda1 controls how few features survive; lambda2 controls temporal stiffness — larger lambda2 means slower adaptation but more robustness to noisy frames.",
        paperIds: ["T021"],
      },
      {
        id: "ladcf-group",
        label: "Multi-channel group-lasso objective",
        formula:
          "h(θ) = Σ_i ||θ_i ∗ x_i − y||² + λ1·sqrt( Σ_j (Σ_i θ_ij)² ) + λ2·Σ_i ||θ_i − θ_model_i||²",
        variables: [
          { symbol: "θ_i", meaning: "filter of channel i (i = 1..L)" },
          { symbol: "θ_ij", meaning: "value of channel i's filter at spatial position j" },
        ],
        intuition:
          "The summation inside the square root forces a position to be switched on in all channels or none — one shared spatial layout for HOG, colour names and CNN maps.",
        why:
          "Channel-independent selection would be inconsistent; the group structure is what makes the selected subspace a coherent compact representation.",
        where: "Section III-B, Eq. 9.",
        paperIds: ["T021"],
      },
    ],
    datasets: ["otb", "vot", "uav123", "others"],
    metrics: ["success-auc", "precision", "eao", "fps"],
    baselines: ["BACF", "SRDCF", "ECO", "C-COT", "CSRDCF", "CFNet"],
    results: [
      "OTB-100: LADCF 86.4% distance precision and 66.4% AUC; LADCF* 90.6% DP and 69.6% AUC (DP ranks second overall, 0.4% behind the best).",
      "Table II (handcrafted, mean OP): LADCF 85.0 (OTB-2013), 77.5 (OTB-50), 81.3 (OTB-100) at 18.2 fps CPU — versus BACF 84.0/70.9/77.6, ECO 82.4/73.4/78.0, C-COT 78.9/72.3/75.7.",
      "Table III (deep, mean OP): LADCF* 90.7/82.5/86.7 at 10.8 fps GPU — versus ECO* 88.7/81.0/84.9, C-COT* 83.7/80.9/82.3, MCPF* 85.8/69.0/78.0.",
      "VOT-2018: EAO 0.338 with VGG features, 0.389 with the deeper backbone variant.",
      "Feature-selection ratios: 5% of spatial features retained for handcrafted and 20% for deep — the tracker runs on a 5-20% subspace.",
    ],
    ablations: [
      "Feature set: HOG+CN 66.44% versus HOG+CN+Conv-3 69.65% — deep features add about 3.2 AUC on top of the selected handcrafted layout.",
      "Temporal consistency on/off: LADCF 66.4% AUC and 86.4% DP versus 63.3% and 83.5% without it — the consistency constraint alone is worth +3.1 AUC.",
      "Backbone choice: VGG EAO 0.338 versus 0.389 with the deeper variant on VOT-2018.",
      "Learning-rate sensitivity and attribute-wise behaviour analysed in the experiments section.",
    ],
    limitations: {
      authorStated: [
        "The deep LADCF* runs at 1.3 fps on CPU and 10.8 fps on GPU — deep features are explicitly not real-time.",
        "lambda1, lambda2 and the selection ratio r are tuned hyper-parameters reported per feature regime rather than adapted online.",
        "The selection ratio is fixed per configuration (5% handcrafted, 20% deep) — the mask size does not adapt to scene content.",
      ],
      evident: [
        "A temporally rigid mask responds slowly to abrupt appearance change — fast deformation is handled only through lambda2 tuning.",
        "Positions deselected for the current frame are only reconsidered at the next update, so a target moving there depends on update cadence.",
      ],
    },
    assumptions: [
      "The set of discriminative spatial locations changes slowly across frames (the low-dimensional manifold assumption).",
      "All channels share one spatial layout (group structure across HOG/colour/deep maps).",
    ],
    computation:
      "18.2 fps CPU (handcrafted, OTB-100), 10.8 fps GPU (deep); complexity O(KLD^2 log D) for K ADMM iterations over D-by-D maps of L channels.",
    relations: [
      {
        to: "T012",
        type: "builds-on",
        note: "Initialises the filter as in BACF and adds learned spatial selection on top of its cropping-based boundary handling.",
      },
      {
        to: "T011",
        type: "uses-as-baseline",
        note: "ECO and C-COT are the deep and handcrafted references in Tables II-III and the OTB-100 plots.",
      },
    ],
    concepts: ["correlation-filter", "appearance-features", "online-vs-offline", "sot", "success-plot"],
    impact:
      "Made learned spatial support plus temporal anchoring a standard DCF design pattern; its selected-feature subspace view (5-20% of positions) fed into later efficient trackers, and LADCF sits in OTB/VOT leaderboards as the representative structured-sparsity CF.",
  },
  {
    id: "T022",
    arxiv: "1808.06048",
    title: "Distractor-aware Siamese Networks for Visual Object Tracking",
    shortTitle: "DaSiamRPN",
    year: 2018,
    authors: ["Zheng Zhu", "Qiang Wang", "Bo Li", "Wei Wu", "Junjie Yan", "Weiming Hu"],
    fileName: "1808.06048v1.pdf",
    task: "single-object",
    tags: ["siamese", "rpn-head", "distractor", "long-term", "real-time"],
    difficulty: "intermediate",
    summary:
      "DaSiamRPN diagnoses why Siamese trackers confuse similar-looking objects: training pairs are dominated by easy non-semantic background. It rebalances training with semantic negatives, adds an incremental distractor-suppression module at inference, and extends the same network with a local-to-global search for long-term tracking — 160 fps short-term, 110 fps long-term, with EAO 0.411 on VOT-2016.",
    problem:
      "Siamese trackers (SiamFC and RPN variants) learn to separate the target from non-semantic background only; semantic distractors (a second person, a similar car) are never emphasised during training because negatives are sampled randomly, so the response map locks onto the wrong instance — especially damaging in long sequences where distractors appear repeatedly.",
    background: ["siamese", "sot", "rpn-head", "appearance-features", "occlusion", "success-plot"],
    previousWork: [
      {
        name: "SiamFC (T009)",
        limitation:
          "Region-wise similarity estimation cannot distinguish same-category instances — a crowd response activates equally for target and neighbours.",
        whyThisPaper: "Adds explicit distractor modelling in both offline training and online inference.",
      },
      {
        name: "SiamRPN (baseline)",
        limitation:
          "Multi-anchor regression gives speed and accuracy, but its EAO of 0.3441 (VOT-2016) is limited by robustness to distractors.",
        whyThisPaper:
          "Component-by-component: detection positives, semantic negatives and distractor-aware updating lift EAO from 0.344 to 0.411.",
      },
      {
        name: "ECO (T011)",
        limitation:
          "State-of-the-art accuracy but far too slow for real-time — the paper frames DaSiamRPN as 20x faster than ECO and 500x faster than C-COT.",
        whyThisPaper: "Keeps the siamese correlation speed while closing much of the accuracy gap.",
      },
    ],
    researchGap:
      "Before this paper, Siamese trackers neither trained on semantic distractors nor had any mechanism to suppress them during tracking, and their long-term mode was an afterthought.",
    contribution: [
      "Training-time sampling strategy that controls the non-semantic versus semantic-distractor balance (detection-domain positives plus semantic negative pairs) to break the class imbalance of training data.",
      "Inference-time distractor-aware module: proposals above a threshold become distractors and the response is re-ranked by subtracting their weighted similarity (Eq. 2), accelerated by linearity of cross-correlation (Eq. 3).",
      "Incremental template learning: target and distractor templates updated with a small learning rate (Eq. 4) so the metric adapts to the current domain.",
      "Long-term extension: local-to-global search with re-detection when the target is lost, running at 110 fps; UAV20L AUC rises from 49.8% to 61.7% with the module.",
      "Speed: 160 fps on short-term benchmarks, 500x faster than C-COT and 20x faster than ECO.",
    ],
    method: {
      pipeline: ["embed", "correlate", "rpn-classify", "rerank-distractors", "incremental-update", "long-term-search"],
      architecture:
        "Siamese backbone (shared CNN) producing a correlation map followed by a region-proposal head with multi-anchor regression (from SiamRPN); a distractor module post-processes proposal scores; an optional long-term branch performs local-to-global re-search.",
      appearanceModel:
        "Two running template streams: the general target template and a distractor set, both exponentially updated; distractors are the top-scoring non-target proposals above threshold.",
      association:
        "Re-ranking rather than data association: choose the proposal whose target similarity minus weighted mean distractor similarity is maximal.",
      optimization:
        "Offline: balanced sampling of positives and semantic negatives for siamese training. Online: incremental template updates with small learning rates; distractor influence factor set per experiment.",
      loss: "Siamese/RPN training objective with distractor pairs; the distractor-aware objective is defined over the proposal set (Eq. 2).",
    },
    equations: [
      {
        id: "dasiam-distractor",
        label: "Distractor-aware re-ranking",
        formula:
          "q = argmax_{p_k ∈ P} [ f(z, p_k) − α̂ · ( Σ_i α_i·f(d_i, p_k) ) / ( Σ_i α_i ) ]",
        variables: [
          { symbol: "z", meaning: "target (exemplar) template" },
          { symbol: "p_k", meaning: "candidate proposal being scored" },
          { symbol: "d_i", meaning: "collected distractor templates (high-scoring non-targets)" },
          { symbol: "α̂, α_i", meaning: "global distractor influence and per-distractor weights" },
        ],
        intuition:
          "Subtract what a generic similar-looking object would score — only similarity specific to the true target survives.",
        why: "Directly attacks semantic-distractor confusion, the failure mode Siamese training never sees.",
        where: "Section 3, Eq. 2; implemented via correlation linearity in Eq. 3 (no extra network passes).",
        params: "alpha-hat controls how aggressively distractors are suppressed; proposals above a score threshold become distractors.",
        paperIds: ["T022"],
      },
      {
        id: "dasiam-incremental",
        label: "Incremental distractor learning",
        formula:
          "q = argmax_{p_k ∈ P} ( Σ_t β_t·ϕ(z_t) / Σ_t β_t  −  α̂·(Σ_t β_t·Σ_i α_i·ϕ(d_(i,t)) / Σ_t β_t) / Σ_i α_i ) ∗ ϕ(p_k)",
        variables: [
          { symbol: "ϕ", meaning: "feature embedding used for correlation" },
          { symbol: "β_t", meaning: "learning rate weighting past frames" },
        ],
        intuition:
          "Blend yesterday's target and distractor descriptors into today's search — the metric migrates from the generic to the specific domain.",
        why: "Turns one-shot similarity into a domain-adapted one while keeping correlation (and therefore speed) intact.",
        where: "Section 3, Eq. 4.",
        paperIds: ["T022"],
      },
    ],
    datasets: ["vot", "otb", "uav123"],
    metrics: ["eao", "precision", "success-auc", "fps"],
    baselines: ["SiamRPN", "ECO", "C-COT", "MDNet", "SiamFC", "CSRDCF++"],
    results: [
      "VOT: EAO 0.446 (VOT-2015), 0.411 (VOT-2016), 0.326 (VOT-2017) — claimed as the largest margins over existing trackers on all three.",
      "VOT-2016 baseline SiamRPN EAO 0.3441; DaSiamRPN reaches 0.411, a 9.6% relative gain (abstract) with 51+ trackers compared.",
      "OTB-2015: SiamRPN OP 81.9% to DaSiamRPN OP 86.5% (best OP in the comparison, +3.0% DP).",
      "Speed: 160 fps short-term and 110 fps long-term — 500x faster than C-COT, 20x faster than ECO; UAV20L AUC 61.7% with the long-term module.",
      "The real-time EAO point on the VOT-2017 plot is the top performer among real-time trackers.",
    ],
    ablations: [
      "Table 2 cumulative components on VOT-2016: SiamRPN 0.344, plus detection-data positives 0.368, plus semantic negatives 0.389, plus distractor-aware updating 0.411.",
      "UAV20L AUC ladder: 45.4, 47.2, 48.6, 49.8% across the same three components; the long-term module pushes it to 61.7%.",
      "Qualitative heatmaps (Fig. 4) show SiamRPN activating on same-category neighbours while DaSiamRPN's response stays on the target.",
    ],
    limitations: {
      authorStated: [
        "Long-term tracking is presented as an extension (local-to-global search) rather than a full detection-based re-identification system.",
        "The distractor influence factors and thresholds are set per experiment rather than learned.",
        "The paper evaluates on short-term suites plus UAV20L and OTB — no re-identification across long disappearances with appearance change is claimed.",
      ],
      evident: [
        "Distractor templates come from the current frame's top proposals — an occlusion covering the target can make the occluder itself the distractor reference.",
        "Gains are largest where same-category distractors exist; on purely non-semantic backgrounds the module has little to suppress.",
      ],
    },
    assumptions: [
      "The target remains within the siamese search region unless the long-term branch triggers.",
      "Highest-scoring non-target proposals are representative distractors for the current scene.",
    ],
    computation:
      "160 fps (short-term) and 110 fps (long-term) as reported; distractor re-ranking costs no extra network passes thanks to correlation linearity.",
    relations: [
      {
        to: "T009",
        type: "conceptual-successor",
        note: "Same siamese correlation-matching framework, upgraded with an RPN head, distractor-aware training and inference, and a long-term mode.",
      },
      {
        to: "T011",
        type: "uses-as-baseline",
        note: "ECO is the accuracy reference on OTB, VOT and UAV; the speed claims are framed against ECO and C-COT.",
      },
    ],
    concepts: ["siamese", "rpn-head", "appearance-features", "sot", "occlusion"],
    impact:
      "Distractor-awareness became a standard ingredient of Siamese tracking, and its local-to-global long-term extension prefigures the tracker-plus-search designs of the following years.",
  },
  {
    id: "T023",
    arxiv: "1808.08834",
    title: "Real-Time MDNet",
    shortTitle: "RT-MDNet",
    year: 2018,
    authors: ["Ilchae Jung", "Jeany Son", "Mooyeol Baek", "Bohyung Han"],
    fileName: "1808.08834v1.pdf",
    task: "single-object",
    tags: ["multi-domain", "roialign", "online-learning", "real-time", "feature-pyramid"],
    difficulty: "intermediate",
    summary:
      "RT-MDNet keeps MDNet's multi-domain trackers but removes its three biggest speed costs: a shared feature pyramid with one fast backbone instead of three, aligned RoIPooling instead of misaligned RoI extraction with an in-place spatial attention block, and a new inter-element learning criterion for online update. It runs at 46,52 fps — about 25x faster than MDNet while matching or slightly beating it (65.0% AUC, 88.5% precision on OTB-2015).",
    problem:
      "MDNet's online multi-domain architecture is accurate but slow: it feeds three separate resolutions (224/127/95) through the network, extracts ROI features with RoIPooling which suffers misaligned quantisation, and updates with M-times-hard updates that are computationally heavy — capping it at single-digit fps.",
    background: ["sot", "online-vs-offline", "appearance-features", "success-plot", "occlusion"],
    previousWork: [
      {
        name: "MDNet (T003)",
        limitation:
          "Strong accuracy but limited by multiple-resolution forward passes, misaligned RoIPooling, and costly M-times-hard online updates.",
        whyThisPaper:
          "Each of RT-MDNet's three contributions replaces one of those bottlenecks directly.",
      },
      {
        name: "ECO (T011) / RT-TDAM",
        limitation:
          "ECO is the accuracy bar for handcrafted/deep hybrid trackers but runs around 8 fps — too slow for real-time.",
        whyThisPaper: "RT-MDNet targets 10 fps to 50+ fps without giving up MDNet-level accuracy.",
      },
      {
        name: "CFNet (T014) / correlation-filter trackers",
        limitation:
          "Fast but their online template update is shallow compared with MDNet's multi-domain discriminative fine-tuning.",
        whyThisPaper:
          "Retains MDNet's deep online update while making its cost negligible.",
      },
    ],
    researchGap:
      "Before this paper, deep trackers with genuine online model updates could not run in real time — the accuracy of MDNet-style multi-domain learning and 10 fps operation were mutually exclusive.",
    contribution: [
      "Aligned RoIPooling with a dilated-convolution variant of RoIAlign: remove max-pooling after conv2 and apply rate-3 dilation after conv3, matching the receptive fields of RoIAlign without extra interpolation cost (Table 2 ablation).",
      "In-place spatial attention block (ISAB): apply feature-wise attention in place on RoI features to amplify the region of interest — recovers most of the accuracy lost to pooled/background features (denseFM).",
      "Inter-element learning (IEL): update the classifier with cross-frame element pairs (targets from frame t with negatives from t+1), replacing M-times-hard in both criteria and time complexity (Eq. 5).",
      "Single shared feature pyramid backbone: one conv pass serves the multi-scale proposal branches — roughly 25x speedup over MDNet while maintaining accuracy.",
      "Results: OTB-2015 success 65.0 / precision 88.5 at 46,52 fps; 130x/25x/8x faster than C-COT, MDNet and ECO respectively.",
    ],
    method: {
      pipeline: ["embed", "roipool", "attention", "classify", "update", "localize"],
      architecture:
        "MDNet-style multi-branch proposal network (three ROI branches) on one shared convolutional feature pyramid; features go through aligned RoIPooling, ISAB attention, three fully connected layers, and a binary classifier; bounding-box regression runs on the FC layers.",
      motionModel:
        "Multi-hypothesis ROI proposals per frame (MDNet's short-term and long-term branches) — no explicit kinematic model; the classifier decides.",
      appearanceModel:
        "Online-fine-tuned ROI appearance features: per-frame positives update the FC classifier while negatives teach rejection of background; ISAB keeps focus inside the RoI.",
      optimization:
        "Inter-element learning — in criterion it pairs current-frame positives with next-frame negatives, in time complexity it avoids M-times-hard resampling; update every frame after the initial offline multi-domain pre-training.",
      loss:
        "Binary classification loss L_cls on RoI labels (Eq. 4) plus instance-wise regression loss L_inst with smooth-L1 over IoU (Eq. 5), both updated online per frame.",
    },
    equations: [
      {
        id: "rtmdnet-cls",
        label: "Classification loss (online)",
        formula: "L_cls = − Σ_i [ y_i·log(p_i) + (1 − y_i)·log(1 − p_i) ]",
        variables: [
          { symbol: "y_i", meaning: "binary label of RoI i (target or background)" },
          { symbol: "p_i", meaning: "classifier probability for RoI i" },
        ],
        intuition: "Standard logistic loss making the FC head separate current target proposals from background.",
        why: "This is the head that online learning adapts each frame — the mechanism IEL makes cheap.",
        where: "Section 3, Eq. 4.",
        paperIds: ["T023"],
      },
      {
        id: "rtmdnet-inst",
        label: "Instance-wise regression loss",
        formula:
          "L_inst = Σ_i [ y_i ≥ 1 ] · SmoothL1( b_i, b_i* ) with positive weights from max(0.7, IoU(b_i, b_i*))",
        variables: [
          { symbol: "b_i", meaning: "regressed bounding box for proposal i" },
          { symbol: "b_i*", meaning: "ground-truth box" },
        ],
        intuition:
          "Only confident, high-IoU proposals teach box refinement; poor overlaps are down-weighted rather than trusted.",
        why: "Keeps regression stable under noisy online samples — the other half of the update alongside classification.",
        where: "Section 3, Eq. 5 (instance-wise learning).",
        paperIds: ["T023"],
      },
      {
        id: "rtmdnet-iel",
        label: "Inter-element learning criterion",
        formula:
          "L_IEL(θ) = Σ_(k=1..K) Σ_(i=1..M) L_cls( f(x_i^t; θ), y_i ) · 1[ y_i = 1 ] + L_cls( f(x_i^(t+1); θ), y_i ) · 1[ y_i = −1 ]",
        variables: [
          { symbol: "x_i^t", meaning: "RoI sample from frame t (positives)" },
          { symbol: "x_i^(t+1)", meaning: "RoI sample from frame t+1 (negatives)" },
          { symbol: "K, M", meaning: "number of update frames and samples per frame" },
        ],
        intuition:
          "Train current targets to beat next-frame background — the discriminative boundary moves with the sequence instead of being re-derived from scratch.",
        why:
          "Replaces MDNet's M-times-hard: same discriminative intent, but the pair construction is one pass instead of repeated hard-negative mining (both criteria and time complexity change).",
        where: "Section 3, Eq. 5 (inter-element learning).",
        paperIds: ["T023"],
      },
    ],
    datasets: ["otb", "uav123", "others"],
    metrics: ["success-auc", "precision", "fps"],
    baselines: ["MDNet", "ECO", "C-COT", "SiamFC", "CFNet"],
    results: [
      "Table 1, OTB-2015: RT-MDNet success 65.0, precision 88.5, 46,52 fps — best fps in the deep-online comparison while retaining MDNet-level accuracy.",
      "Speed claims: approximately 25x faster than MDNet; 130x, 25x and 8x faster than C-COT, MDNet and ECO respectively.",
      "UAV123: RT-MDNet precision 0.772 versus MDNet+IEL 0.767 — the aligned/attention variant edges out the IEL-only configuration.",
      "Evaluated on OTB-2015, UAV123 and two additional benchmarks with the online update active throughout.",
    ],
    ablations: [
      "Table 2 (OTB-2015 success/precision): RoIPooling 35.4/53.8; RoIAlign 56.1/80.4; aligned-RoIPooling 59.0/83.8; +denseFM (in-place spatial attention) 60.7/84.3; improved aligned version 61.9/85.3.",
      "Backbone unification versus per-branch networks is the source of the ~25x speedup (three resolutions, one pyramid).",
      "IEL compared against M-times-hard update on the same architecture (MDNet+IEL column), isolating the update-criterion gain.",
    ],
    limitations: {
      authorStated: [
        "Online learning still requires backpropagation through the FC stack each frame — updates are cheaper than M-times-hard but not free.",
        "The feature pyramid replaces higher-resolution inputs, so very small targets depend on the shallowest pyramid level.",
        "Quantised aligned-RoIPooling remains an approximation of true RoIAlign (it trades a few points of accuracy for speed).",
      ],
      evident: [
        "No explicit re-detection or long-term model — occlusion longer than the hard negatives seen online can still drift the FC head.",
        "IEL pairs frames t and t+1, so a sudden scene cut at update time feeds mislabelled pairs.",
      ],
    },
    assumptions: [
      "Consecutive frames share enough appearance for next-frame negatives to be valid training signal.",
      "The shared feature pyramid retains per-scale detail needed by all three proposal branches.",
    ],
    computation:
      "46,52 fps as reported on OTB-2015 (GPU, shared-pyramid backbone) — roughly 25x MDNet, 8x ECO.",
    relations: [
      {
        to: "T003",
        type: "builds-on",
        note: "Same multi-domain fine-tuning paradigm; RT-MDNet replaces MDNet's RoIPooling, multi-resolution backbone and M-times-hard update.",
      },
      { to: "T011", type: "uses-as-baseline", note: "ECO is the accuracy/speed reference (8x slower)." },
      { to: "T012", type: "uses-as-baseline", note: "BACF appears in the comparison tables as a fast handcrafted CF." },
      { to: "T014", type: "uses-as-baseline", note: "CFNet is among the deep trackers compared on speed-accuracy." },
      { to: "T009", type: "uses-as-baseline", note: "SiamFC compared as the siamese real-time reference." },
    ],
    concepts: ["online-vs-offline", "appearance-features", "sot", "occlusion", "success-plot"],
    impact:
      "Proved that MDNet-class online deep trackers can be real-time, and its aligned-RoIPooling plus IEL updates became reusable building blocks for efficient online trackers in the pre-transformer era.",
  },
  {
    id: "T024",
    arxiv: "1809.07845",
    title: "LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking",
    shortTitle: "LaSOT",
    year: 2018,
    authors: ["Heng Fan", "Liting Lin", "Fan Yang", "Peng Chu", "Ge Deng", "Sijia Yu", "a.o."],
    fileName: "1809.07845v2.pdf",
    task: "benchmark",
    tags: ["benchmark", "dataset", "long-term", "evaluation-protocol", "tracking"],
    difficulty: "intermediate",
    summary:
      "LaSOT is a large-scale single-object-tracking benchmark built to study long-term tracking: 1,400 sequences, 3.52 million frames, 70 categories, 14 attributes, with average sequences of ~2,506 frames. It defines two evaluation protocols — Protocol I (all 1,400 test sequences) and Protocol II (a harder subset) — plus a training split, and shows that training on LaSOT lifts existing trackers (SiamFC-3s gains up to 2.1 points of precision and 2.0 of success on OTB after retraining).",
    problem:
      "Existing benchmarks are small (about 100 sequences, a few dozen frames each) and target short-term tracking; their short, clean sequences saturate modern trackers and say nothing about long-term robustness — drift, re-detection and scale over long durations.",
    background: ["sot", "benchmark-design", "success-plot", "appearance-features"],
    previousWork: [
      {
        name: "OTB-2013 / OTB-2015 / OTB-50 (evaluated by T009 et al.)",
        limitation:
          "About 100 sequences of tens of seconds — trackers saturate these sets and long-term behaviour is untested.",
        whyThisPaper:
          "Provides 1,400 sequences with much longer durations and a dedicated long-term protocol.",
      },
      {
        name: "VOT-2016 / VOT-2017 (T011, T022 era)",
        limitation:
          "Emphasises short-term accuracy with rapid re-initialisation after failure rather than long-term persistence.",
        whyThisPaper: "Adds protocols designed around long sequences where trackers must survive without resets.",
      },
      {
        name: "VOT-2018 / TrackingNet / GOT-10k style sets",
        limitation: "Different scales and annotation styles, without a unified long-term benchmark taxonomy.",
        whyThisPaper:
          "LaSOT's 14 attributes and class-annotated splits let you diagnose which failure mode a tracker suffers over long horizons.",
      },
    ],
    researchGap:
      "Before this paper, no benchmark combined large scale (thousands of videos), long duration (average ~2,506 frames per sequence) and fine-grained attribute annotation for single object tracking.",
    contribution: [
      "LaSOT dataset: 1,400 test sequences, 3.52 million frames, 70 classes, 14 attributes — average sequence length about 2,506 frames.",
      "Two evaluation protocols: Protocol I evaluates all 1,400 test sequences; Protocol II uses a harder subset to better separate trackers.",
      "A training split plus the protocols enable data-hungry siamese trackers to be trained on LaSOT directly.",
      "Empirical finding: retraining SiamFC-3s on LaSOT improves generalisation — OTB-2015 precision 75.6→77.7 (+2.1) and success 56.5→58.2 (+1.7).",
      "Benchmark suite: extensive evaluation of handcrafted, deep siamese and MDNet-style trackers, establishing a leaderboard for long-term tracking.",
    ],
    method: {
      pipeline: ["collect", "annotate", "protocol", "evaluate"],
      architecture:
        "Not a tracker — a benchmark construction pipeline: crawling diverse videos from the internet, annotating 14 tracking attributes and category labels over 70 WordNet-derived classes (including fast motion, occlusion, illumination, deformation, out-of-view), then defining train/test splits for fair training and evaluation. Protocol I reports success (AUC), precision and normalised precision over all 1,400 test sequences; Protocol II uses a harder subset intended to stay discriminative as trackers improve; both report means over sequences plus per-attribute breakdowns.",
    },
    equations: [
      {
        id: "lasot-success",
        label: "Success (overlap) and AUC ranking",
        formula:
          "success_i = IoU(b_i, g_i) = |b_i ∩ g_i| / |b_i ∪ g_i| ;  score = AUC over overlap thresholds in [0, 1]",
        variables: [
          { symbol: "b_i", meaning: "predicted bounding box at frame i" },
          { symbol: "g_i", meaning: "ground-truth bounding box at frame i" },
        ],
        intuition:
          "How much the predicted box overlaps the true box each frame; trackers are ranked by the area under the curve of overlap-versus-threshold.",
        why:
          "Success is LaSOT's headline accuracy metric — the long-sequence AUC reveals drift that short benchmarks hide.",
        where: "Section 4.1, Evaluation Methodology: 'The success is computed as the Intersection over Union (IoU)... ranked using the AUC between 0 to 1.'",
        paperIds: ["T024"],
      },
      {
        id: "lasot-precision",
        label: "Precision and normalised precision",
        formula:
          "precision = fraction of frames with ||c_pred − c_gt||₂ < τ (τ e.g. 20 px) ;  normalised precision ranked by AUC over thresholds in [0, 0.5]",
        variables: [
          { symbol: "c_pred, c_gt", meaning: "centres of predicted and ground-truth boxes" },
          { symbol: "τ", meaning: "pixel distance threshold (e.g., 20 pixels)" },
        ],
        intuition:
          "Is the predicted centre close enough to the true centre? Normalised precision rescales by target size so large objects do not dominate.",
        why:
          "Plain precision is sensitive to target size and image resolution, so LaSOT also reports the size-normalised version for fair cross-sequence comparison.",
        where: "Section 4.1: precision via centre distance against a threshold, normalised 'as in [41]', ranked by AUC between 0 and 0.5.",
        paperIds: ["T024"],
      },
    ],
    datasets: ["lasot"],
    metrics: ["precision", "norm-precision", "success-auc"],
    baselines: ["SiamFC", "MDNet", "VITAL", "ECO", "GOTURN"],
    results: [
      "Protocol I (all 1,400 sequences): SiamFC precision 0.341 / normalised precision 0.449 / success 0.358; ECO 0.298 / 0.358 / 0.34; near-zero scores across the board for MDNet and VITAL — long-term tracking is largely unsolved.",
      "Protocol II (harder subset): MDNet 0.373 precision / 0.46 norm-precision / 0.397 success; VITAL 0.36 / 0.453 / 0.39; SiamFC 0.339 / 0.42 / 0.336 — MDNet-style online trackers lead the subset.",
      "Retraining generalisation: SiamFC-3s retrained on LaSOT — OTB-2013 precision 0.803→0.816 (+1.3), success 0.588→0.608 (+2.0); OTB-2015 precision 0.756→0.777 (+2.1), success 0.565→0.582 (+1.7).",
      "Scale facts: 1,400 sequences, 3.52M frames, 70 classes, 14 attributes, average ~2,506 frames per sequence.",
    ],
    ablations: [
      "Protocol I versus Protocol II: same trackers score much lower on Protocol I (full set) — sequence length and long-term drift dominate the averages.",
      "Training-data study: SiamFC-3s pretrained elsewhere versus retrained on LaSOT — the +2.1/+1.7 OTB-2015 gains quantify benchmark-training transfer.",
      "Per-attribute evaluation attributes failures to 14 causes (fast motion, occlusion, illumination change, deformation, out-of-view, etc.).",
    ],
    limitations: {
      authorStated: [
        "Long-term tracking remains unsolved: even the best trackers score well below 0.4 normalised precision on Protocol I.",
        "Attributes are annotated at sequence level, so per-frame attribute changes within a sequence are not captured.",
        "Category coverage is limited to 70 classes despite the large scale.",
      ],
      evident: [
        "The test set's extreme durations expose drift, but without a re-detection requirement (unlike VOT's reset protocol) the ranking reflects different failure priorities.",
        "Some sequences are crawled from the internet, so annotation noise in long frames is possible.",
      ],
    },
    assumptions: [
      "The 70 classes and 14 attributes are representative of real long-term tracking demands.",
      "Success AUC plus precision suffice as summary metrics — tracking failure does not require a recovery/reset score.",
    ],
    computation:
      "Benchmark resource: 1,400 sequences / 3.52M frames; evaluation runs each tracker over the full test split (reported per tracker, e.g., SiamFC 0.358 mean success).",
    relations: [
      { to: "T009", type: "uses-as-baseline", note: "SiamFC — and its retrained SiamFC-3s variant — are central to the protocol results and training-data study." },
      { to: "T003", type: "uses-as-baseline", note: "MDNet leads Protocol II, anchoring the online deep family." },
      { to: "T011", type: "uses-as-baseline", note: "ECO evaluated on Protocol I as the strong DCF reference." },
      { to: "T014", type: "uses-as-baseline", note: "CFNet family among the siamese-style trackers benchmarked." },
    ],
    concepts: ["benchmark-design", "sot", "appearance-features", "success-plot"],
    impact:
      "LaSOT became a standard long-term training and evaluation resource for siamese and beyond — next-generation trackers (including the transformer era) are routinely trained or validated on it, and its protocols are widely reported alongside OTB/VOT/GOT-10k.",
  },
  {
    id: "T025",
    arxiv: "1810.11981",
    title: "GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild",
    shortTitle: "GOT-10k",
    year: 2018,
    authors: ["Lianghua Huang", "Xin Zhao", "Kaiqi Huang"],
    fileName: "1810.11981v3.pdf",
    task: "benchmark",
    tags: ["benchmark", "dataset", "one-shot", "generalisation", "evaluation-protocol"],
    difficulty: "intermediate",
    summary:
      "GOT-10k is a large-scale generic object tracking benchmark: over 10,000 video segments, 1.5 million annotated bounding boxes, more than 560 object classes and 87 motion patterns. Its key idea is a one-shot evaluation protocol where trackers are tested on object classes and motion patterns never seen in training, with class-balanced average overlap (mAO) and success rate (mSR) metrics — exposing how little existing trackers truly generalise.",
    problem:
      "Existing benchmarks cover a narrow object category space with tracking-specific training data; trackers tuned and evaluated on the same classes report numbers that do not reflect generic in-the-wild tracking. There was no large-scale way to measure generalisation to unseen objects and motions.",
    background: ["benchmark-design", "sot", "success-plot", "appearance-features", "online-vs-offline"],
    previousWork: [
      {
        name: "OTB (used by T009 et al.) / VOT",
        limitation:
          "Small, category-narrow sets where training and test classes overlap; evaluation favours per-sequence tuning rather than generalisation.",
        whyThisPaper: "Defines disjoint train/val/test class splits and a one-shot submission protocol.",
      },
      {
        name: "ImageNet / COCO style detection sets",
        limitation: "Detection labels over classes but not temporal object tracking trajectories.",
        whyThisPaper: "Combines detection-scale diversity (560+ classes) with dense per-frame tracking boxes.",
      },
      {
        name: "LaSOT (T024)",
        limitation: "Long-term focus with 70 classes — still a single fixed test distribution.",
        whyThisPaper:
          "Complements LaSOT with category/motion-disjoint testing and one-shot generalisation metrics rather than long-term AUC alone.",
      },
    ],
    researchGap:
      "Before this paper, no benchmark measured how well a tracker generalises to object classes and motion patterns that were absent from its training data at scale.",
    contribution: [
      "GOT-10k dataset: over 10,000 video segments, more than 1.5 million annotated bounding boxes, 560+ WordNet classes and 87 motion patterns with 84 classes in the test split.",
      "One-shot evaluation protocol: test-set classes and motion patterns are disjoint from the training data — all results are measured by an online server so no per-benchmark tuning is possible.",
      "Class-balanced metrics: mean average overlap (mAO) and mean success rate (mSR) average over classes so frequent categories do not dominate.",
      "Empirical generalisation study: trackers' mAO improves by roughly 15% as test classes grow from 5 to 405 (e.g., SiamFCv2), and 39 trackers are benchmarked under the protocol.",
      "Benchmark analysis linking performance to motion-pattern diversity and appearance overlap between train and test.",
    ],
    method: {
      pipeline: ["collect", "annotate", "protocol", "evaluate"],
      architecture:
        "A benchmark, not a tracker: video collection and quality control, diversity-constrained crawling over object and motion categories, frame-accurate per-frame box annotation, a motion-pattern taxonomy of 87 motions, and class-disjoint splits — train (seen classes), val (separate classes) and test (84 classes and motion patterns never seen in training). A server-side evaluation service scores each submission once, reporting class-balanced mAO and mSR alongside fps so no per-benchmark tuning is possible.",
    },
    equations: [
      {
        id: "got10k-mao",
        label: "Class-balanced mean average overlap (mAO)",
        formula:
          "mAO = (1/C) · Σ_(c=1..C) [ (1/|S_c|) · Σ_(i∈S_c) AO_i ]   (Eq. 1)",
        variables: [
          { symbol: "C", meaning: "number of object classes" },
          { symbol: "S_c", meaning: "subset of sequences belonging to the c-th class" },
          { symbol: "|S_c|", meaning: "size (number of sequences) of that subset" },
          { symbol: "AO_i", meaning: "average overlap of the tracker on sequence i" },
        ],
        intuition:
          "Average the overlap within each class first, then average across classes — so a class with thousands of sequences cannot drown out a rare one.",
        why:
          "Directly attacks class-imbalance: sequence-wise averaging lets dominant classes (e.g., person) set the final ranking.",
        where: "Section 4.2, Evaluation Methodology, Eq. 1.",
        params:
          "AO is equivalent to the AUC success metric of OTB/LaSOT/VOT; the same nested-averaging principle defines mSR.",
        paperIds: ["T025"],
      },
      {
        id: "got10k-msr",
        label: "Class-balanced success rate (mSR)",
        formula:
          "SR = fraction of frames with overlap > threshold (0.5, or stricter 0.75) ;  mSR = (1/C) · Σ_c (1/|S_c|) · Σ_(i∈S_c) SR_i",
        variables: [
          { symbol: "overlap", meaning: "IoU between estimated and ground-truth boxes on a frame" },
          { symbol: "SR_i", meaning: "success rate of the tracker on sequence i" },
        ],
        intuition:
          "What fraction of frames are 'locked on' above a strict overlap bar, averaged class-by-class like mAO.",
        why:
          "mSR50 and mSR75 report how often tracking stays tightly on target rather than merely overlapping loosely — harder to game than mean overlap.",
        where: "Section 4.2: 'We use two overlap thresholds 0.5 and more strict 0.75 for calculating the mSR.'",
        paperIds: ["T025"],
      },
    ],
    datasets: ["got10k", "otb"],
    metrics: ["success-auc", "fps"],
    baselines: ["SiamFC", "ECO", "MDNet", "GOTURN", "DeepSRDCF"],
    results: [
      "Scale: 10,000+ segments, 1.5M annotated boxes, 560+ object classes, 87 motion patterns, 84 classes held out for test.",
      "Table 5 top: MemTracker mAO 0.460, DeepSTRCF 0.449, SASiamP 0.445, SiamFCv2 0.434, GOTURN 0.418 at 70.1 fps, ECO 0.395, SiamFC 0.392 at 32.6 fps, STRCF 0.377, MDNet 0.352, BACF 0.346, CFNetc1 0.343 at 32.6 fps, KCF 0.279, IVT 47.3 fps (CPU).",
      "Generalisation finding: mAO of trackers rises by about 15% when the number of test classes increases from 5 to 405 (SiamFCv2 analysed).",
      "39 trackers evaluated under the one-shot protocol with class-balanced metrics.",
    ],
    ablations: [
      "Train/test class-overlap study: performance drops when test classes are disjoint — quantifying the generalisation gap the protocol was designed to expose.",
      "Test-class-count curve: mAO improves as more classes are included (5 to 405), showing score sensitivity to class composition.",
      "Motion-pattern breakdown over the 87 motions attributes failures to specific dynamics rather than object category alone.",
    ],
    limitations: {
      authorStated: [
        "Baseline results are produced with default parameter settings and no validation-set tuning — the paper explicitly calls its scores 'a lower bound' of the evaluated trackers.",
        "Deep trackers degrade on unfamiliar objects (0.1%-5.9% drop under one-shot), which the authors present as a limitation of existing trackers the protocol exposes.",
      ],
      evident: [
        "Test annotations are kept private and scored once by the online server, so per-run error diagnosis is limited.",
        "Coverage is generic single-object tracking only — no multi-object, re-identification or long-term recovery protocol.",
        "Class-balanced mAO treats rare and common classes equally regardless of real-world frequency.",
      ],
    },
    assumptions: [
      "Disjoint class splits are a good proxy for real-world generalisation.",
      "Class-balanced mAO/mSR better reflect generic ability than micro-averaged overlap.",
    ],
    computation:
      "Server-side evaluation over 10,000+ segments; per-tracker fps reported separately (e.g., GOTURN 70.1, SiamFC 32.6, IVT 47.3 on CPU).",
    relations: [
      { to: "T009", type: "uses-as-baseline", note: "SiamFC and its strengthened SiamFCv2 are core entries in the one-shot leaderboard." },
      { to: "T014", type: "uses-as-baseline", note: "CFNet (CFNetc1) benchmarked at mAO 0.343." },
      { to: "T003", type: "uses-as-baseline", note: "MDNet appears as the representative online multi-domain tracker (mAO 0.352)." },
      { to: "T011", type: "uses-as-baseline", note: "ECO is the strongest classic DCF entry (mAO 0.395)." },
      { to: "T019", type: "uses-as-baseline", note: "STRCF and DeepSTRCF rank high — DeepSTRCF is second overall at mAO 0.449." },
    ],
    concepts: ["benchmark-design", "sot", "success-plot", "appearance-features"],
    impact:
      "GOT-10k became the de facto generalisation benchmark for generic object tracking, adopted by major trackers and challenges, and its one-shot protocol directly shaped how subsequent works report in-the-wild generalisation.",
  },
];
