# Research Labs

Interactive, evidence-grounded research labs over paper corpora.

Each lab reads a curated corpus of academic papers and turns it into a browsable, hands-on workspace: concepts are explored with interactive simulations, and every claim is traceable back to the source PDFs it came from.

## Labs

| Lab | Status | Corpus |
| --- | --- | --- |
| **Object Detection** | Active | 48 papers · 2015–2026 |
| Object Tracker | Coming soon | — |
| Evolutionary | Coming soon | — |
| Nature-Inspired Algorithms | Coming soon | — |
| Neuroevolution | Coming soon | — |
| LLMs | Coming soon | — |

## Object Detection lab

The flagship lab covers the modern object detection literature (2015–2026) — from the early one-stage and two-stage detectors (YOLO, SSD, Faster R-CNN) through DETR, deformable, and anchor-free approaches.

### Interactive simulations

Hands-on widgets for the core mechanisms behind the papers:

- **IoU** and **NMS** — how boxes are matched and suppressed
- **Anchors** / anchors-vs-free — anchoring pipelines vs. anchor-free designs
- **Focal loss** and **regression** — loss shaping for class imbalance
- **Hungarian & dynamic assignment** — how DETR-style matchers assign labels
- **Deformable**, **feature pyramid**, **cascade**, **attention**, **denoising**, **receptive field**, **mosaic** augmentation, and **edge scenarios**

### Exploration views

- **Overview** — era-by-era sweep across 2015–2026
- **Year** — what changed each year
- **Explorer** — browse all 48 papers
- **Concepts** — mechanisms and where they appear
- **Problems** — problem lifecycles and how they were solved
- **Applications** — where the detectors are used
- **Directions** — active research directions
- **Math** — the equations that define the field
- **Ask** — query the corpus
- **Audit** — data provenance and coverage checks

### Data integrity

Lab content lives in structured data files under `src/labs/object-detection/data/`. An audit suite (`npm test`) verifies every reference resolves, the corpus is 48 unique papers over a contiguous 2015–2026 range, and there are no hallucinated entries.

Raw paper PDFs are stored under `papers/<topic>/`, one directory per lab.

## Tech stack

- Next.js 16 (App Router, dynamic bundle loading per lab)
- React 19, TypeScript
- Tailwind CSS 4
- Vitest for tests

## Getting started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start the dev server |
| `npm run build` | Production build |
| `npm start` | Serve the production build |
| `npm run lint` | Lint with ESLint |
| `npm test` | Run Vitest (data integrity tests) |