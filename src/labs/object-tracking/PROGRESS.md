# Object Tracking Lab — progress & resume notes

> Updated each work session so the next agent can resume without re-reading everything.

## Corpus
- Source: `papers/object_trackers/` — 159 PDFs, manifest `papers/object_trackers/manifest.csv`
- Canonical IDs: T001..T159 = manifest order (see `src/labs/object-tracking/data/manifest.ts`)
- Papers are arXiv-ID filenames, e.g. `1602.00763v2.pdf` = SORT.

## Data already done (159/159 — T001..T159 COMPLETE, 2026-09-30)
- `src/labs/object-tracking/data/types.ts` — schema (PaperRecord, Equation, Concept, Dataset, Metric, SECTIONS 13 entries)
- `src/labs/object-tracking/data/manifest.ts` — generated, all 159 entries
- `src/labs/object-tracking/data/concepts.ts` — 35-concept prerequisite DAG
- `src/labs/object-tracking/data/datasets.ts` — registry (mot15/16/17/20, dancetrack, otb, vot, got10k, lasot, trackingnet, uav123, lsotb-tir, lasheR, webuav-3m, bcot, antitrack, ego4d…)
- `src/labs/object-tracking/data/metrics.ts` — 19 metrics with grounded formulas
- `src/labs/object-tracking/data/papers/batch-01.ts` — T005 (SORT exemplar)
- `batch-02.ts` — T001,T002,T003,T004,T006,T007,T008,T009,T010,T011,T012,T013
- `batch-03.ts` — T014..T025 (12)
- `batch-04.ts` — T026..T037 (12)
- `batch-05.ts` — T038..T049 (12)
- `batch-06.ts` — T050..T058 (9)
- `batch-07.ts` — T059 MOTR, T060 SiamMOT, T061 ByteTrack, T062 TAPL, T063 DanceTrack (5, session 2026-09-30)
- `batch-08.ts` — T064 SwinTrack, T065 WebUAV-3M, T066 SOT-survey-2022, T067 StrongSORT, T068 TCTrack (5, session 2026-09-30)
- `batch-09.ts` — T069 SimTrack, T070 MixFormer, T071 OSTrack, T072 BCOT, T073 OC-SORT, T074 MeMOT, T075 BoT-SORT, T076 Unicorn, T077 AiATrack, T078 Spiking-SiamFC++ (10, session 2026-09-30)
- `batch-10.ts` — T079 ProContEXT, T080 MOTRv2, T081 UOSTrack, T082 MotionTrack, T083 ViPT, T084 SCT-night, T085 OVTrack, T086 MOTRv3, T087 MixFormerV2, T088 SparseTrack (10, session 2026-09-30)
- `batch-11.ts` — T089 STTracker-3D, T090 EANet-RGBT, T091 BOTT-3D, T092 DiffusionTrack, T093 LEGO-3D, T094 MITS-unified-seg, T095 LightFC, T096 ZoomTrack, T097 Un-Track-any-modality, T098 Hy-Tracker-HS (10, session 2026-09-30)
- `batch-12.ts` — T099 DiffMOT, T100 AMTTrack/FELT-event, T101 MambaMOT, T102 DepthMOT, T103 survey-beyond-SOT, T104 survey-multimodal, T105 XTrack, T106 eMoE-event, T107 DeepMoveSORT, T108 Diff-Tracker-unsupervised (10, session 2026-09-30)
- `batch-13.ts` — T109 MM-Tracker-Mamba, T110 Best-of-N, T111 RTAT, T112 ConsistencyTrack, T113 FACT-continual, T114 ABBG-attack, T115 DeTrack-denoise, T116 LightFC-X, T117 Deep-LG-Track, T118 DARTer-night (10, session 2026-09-30)
- `batch-14.ts` — T119 survey-generic-2025, T120 Multi-State, T121 DM3T-diffusion, T122 AntiUAV-bench, T123 query-attack, T124 diffusion-unsupervised, T125 AVTrack, T126 HDST-GNN, T127 FEMOT-event, T128 Polycepta (10, session 2026-09-30)
- `batch-15.ts` — T129 APRTrack-event, T130 FalconTrack-UAV, T131 GOT-dissertation, T132 AE-UAV-event-bench, T133 generator-tracker, T134 CMRTrack-UAV, T135 SCDT-RGBT-missing, T136 CD-RMOT-bench, T137 SATATrack-antiUAV, T138 ACTrack-agentic (10, session 2026-09-30)
- `batch-16.ts` — T139 PAFCNet-RGBT, T140 AnyTrack, T141 EgoTrack3D, T142 MSP-hyperspectral, T143 GenTrack3-MOT, T144 MVTrack-bitstream, T145 McByte++-sports, T146 MaST-light, T147 LVTrack-referring, T148 PLANET-world-grounded, T149 YesTrack-MLLM (11, session 2026-09-30)
- `batch-17.ts` — T150 CST-WM-embodied, T151 TLCTrack-tokens, T152 OmniSORT-panorama, T153 TFTrack-3D-templatefree, T154 TBD-survey, T155 MOT-HPO-study, T156 EECTracker-swarm, T157 SAVTrack-3D, T158 MAETrack-3D-prior, T159 occlusion-robust-YOLO (10, session 2026-09-30)
- Batch shape: `export const BATCH_0N: PaperRecord[]`. Provenance policy in types.ts header.

## Lab wiring (done this session — see below)
- Real lab lives at `src/labs/object-tracking/` (data + context/shell/views/sims).
- Root tab id stays `object-tracker` (in `src/root/labs.ts`); `src/labs/object-tracker/index.tsx`
  re-exports the full lab so the existing tab becomes the tracking lab without breaking hashes.
- `src/labs/object-tracking/data/papers/index.ts` aggregates batches → PAPERS, PAPER_BY_ID.
- Views: overview, path, timeline, graph, gaps, foundations, pipeline, math, papers, compare,
  datasets, metrics, audit (13 = SECTIONS in types.ts).
- Sims registry: `src/labs/object-tracking/sims/index.tsx` → SIMULATORS, SimShell.
  Keys used by concepts/equations: iou-track, kalman, hungarian, motion, assoc-cost,
  bytetrack, reid, metrics, corr-filter, siamese, track-mgmt.

## TODO for next agent (in order)
1. **Cross-paper review (DONE 2026-09-30).**
   - 159/159 records, 0 missing vs manifest, 0 duplicates, 0 dangling relations, 0 self-relations.
   - 10 relation-less papers audited and justified: T001 (CF root), T006 (MOT16 bench), T008 (SINT — compares only non-corpus MUSTer/MEEM), T072 (BCOT 3D bench), T089 (STTracker — cites only non-corpus 3D work), T125 (first audio-visual bench), T128 (Polycepta — cites only non-corpus RobMOT/FastTracker line), T130 (FalconTrack new thread), T141 (EgoTrack3D new thread), T150 (CST-WM new thread).
   - 2 equation-less records justified: T030 Tracktor (re-evaluation paper, no novel equations), T104 (multimodal survey).
   - Lineages verified: SORT→DeepSORT→StrongSORT→OC-SORT→BoT-SORT; SiamFC→SiamRPN++→SiamCAR/Ocean→TransT/MixFormer/OSTrack; MOTR→MOTRv2→MOTRv3; ByteTrack as hub.
   - ~600+ equation entries with variable-level explanations; 11 simulators wired.
2. **Remaining future work (NOT papers).**
   - Optional enrichment: dataset `paperIds` back-links for T069+ papers (datasets.ts lists only early paperIds); metric `paperIds` likewise. Views compute counts from records where possible, but DATASETS cards show "no structured paper yet" for sets used by later papers. A backfill script mapping record.datasets → dataset.paperIds would close this.
   - Optional: per-paper PDF spot-checks beyond the audited samples (T062 authors, T073/T069 titles, T089 venue, T090 authors, T122/T123 headers, T130/T131/T133 headers, T150 title/authors all verified against PDFs).

## Validation
4. Detection↔tracking links: `views/pipeline.tsx` already links to `#object-detection` hash;
   extend if new concepts demand it.

## How to read a paper PDF here
- `ls papers/object_trackers/ | head` for filenames; manifest maps arxiv→T-id.
- Text extraction: `pdftotext -layout papers/object_trackers/<f> - | head -n 200`
  (poppler installed? if not, `python3 -c "import pypdf..."` — check first).
- Never invent venue/results/equations. If PDF is unreadable, record summary from readable
  sections only and mark limitations.evident accordingly — do not browse the web for results.

## Validation
- `npx tsc --noEmit` must pass; `npm run build` must pass; detection lab untouched
  (only `src/root/labs.ts` blurb + `src/labs/object-tracker/index.tsx` changed on its side).
- Views degrade on mobile (overflow-x-auto, responsive grids already in place).
