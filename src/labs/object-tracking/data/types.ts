/* Core research data model for the Object Tracking Research Lab.
 *
 * Provenance rules (same policy as the Object Detection lab):
 *  - Every record field must be grounded in the corpus PDF named by `fileName`.
 *  - results[] only contain numbers actually reported in the paper.
 *  - relations[] only connect corpus papers (T001..T159) when the paper itself
 *    builds on / compares against / addresses the cited work.
 *  - Nothing is invented: no invented equations, results, venues, or links.
 */

export type Confidence = "HIGH" | "MEDIUM" | "LOW";

/** How a paper relates to its predecessor inside the corpus. */
export type RelationType =
  | "builds-on" // uses the predecessor's machinery directly
  | "improves" // same problem, better solution
  | "replaces" // supersedes the predecessor's approach
  | "extends" // adds a capability to the predecessor
  | "combines" // fuses ideas from the predecessor with others
  | "uses-as-baseline" // evaluates against the predecessor
  | "addresses-limitation" // exists because of a stated weakness
  | "conceptual-successor"; // continues the predecessor's research thread

export type Relation = {
  /** corpus id of the earlier paper (T001..T159) */
  to: string;
  type: RelationType;
  /** one line: what the current paper takes / fixes / changes */
  note?: string;
};

export type Task =
  | "single-object"
  | "multi-object"
  | "segmentation-tracking"
  | "3d-tracking"
  | "multimodal-tracking"
  | "benchmark"
  | "survey"
  | "other";

export type Difficulty = "intro" | "intermediate" | "advanced";

export type Equation = {
  id: string;
  label: string;
  /** plain-text / unicode math, rendered in a monospace block */
  formula: string;
  variables: { symbol: string; meaning: string }[];
  /** plain-language meaning for a non-CV reader */
  intuition: string;
  /** why the equation is needed at all */
  why: string;
  /** where in the algorithm it is applied */
  where: string;
  /** how changing parameters changes behaviour */
  params?: string;
  /** key into SIMULATOR_INDEX when an interactive sim exists */
  simulator?: string;
  paperIds?: string[];
};

export type PaperMethod = {
  /** ordered pipeline steps, e.g. ["detect", "predict", "associate", "update"] */
  pipeline: string[];
  architecture: string;
  motionModel?: string;
  appearanceModel?: string;
  association?: string;
  reid?: string;
  detectionDependency?: string;
  trackManagement?: string;
  optimization?: string;
  loss?: string;
};

export type Limitations = {
  /** stated by the authors themselves */
  authorStated: string[];
  /** visible from the methodology (lab synthesis, never presented as author claim) */
  evident: string[];
};

export type PaperRecord = {
  id: string; // T001..T159 — must match the manifest entry
  arxiv: string;
  title: string;
  shortTitle: string;
  year: number;
  authors: string[];
  venue?: string;
  fileName: string;
  task: Task;
  /** paradigm / family tags, lowercase kebab, e.g. "tracking-by-detection", "siamese", "correlation-filter", "transformer", "benchmark" */
  tags: string[];
  difficulty: Difficulty;
  /** 2–3 sentence plain explanation */
  summary: string;
  /** what problem existed and why it was hard */
  problem: string;
  /** concept ids a reader needs first (see concepts.ts) */
  background: string[];
  /** preceding methods → their limitation → why this paper was needed */
  previousWork: { name: string; limitation: string; whyThisPaper: string }[];
  /** one line: what was missing before this paper */
  researchGap: string;
  /** concrete contributions, 3–6 bullets */
  contribution: string[];
  method: PaperMethod;
  equations: Equation[];
  /** dataset ids from DATASET_IDS (datasets.ts) */
  datasets: string[];
  /** metric ids from METRIC_IDS (metrics.ts) */
  metrics: string[];
  baselines: string[];
  /** factual, paper-reported outcomes (numbers allowed only if printed in the paper) */
  results: string[];
  ablations: string[];
  limitations: Limitations;
  assumptions?: string[];
  computation?: string;
  /** earlier corpus papers this one relates to */
  relations: Relation[];
  /** concept ids this paper advances */
  concepts: string[];
  /** what idea became important / what later work reused */
  impact: string;
};

export type Concept = {
  id: string;
  name: string;
  category:
    | "basics"
    | "motion"
    | "association"
    | "appearance"
    | "architecture"
    | "evaluation"
    | "paradigm";
  intuition: string;
  formula?: string;
  simulator?: string;
  /** prerequisite concept ids */
  prereqs: string[];
};

export type DatasetRecord = {
  id: string;
  name: string;
  year: number;
  purpose: string;
  domain: string;
  trackingType: "SOT" | "MOT" | "both" | "segmentation" | "3D" | "multimodal";
  videos?: number;
  frames?: number;
  objects?: number;
  annotations?: number;
  characteristics: string[];
  metrics: string[];
  paperIds: string[];
};

export type MetricRecord = {
  id: string;
  name: string;
  family: "detection" | "association" | "localization" | "robustness" | "efficiency";
  formula: string;
  variables?: { symbol: string; meaning: string }[];
  meaning: string;
  intuition: string;
  example: string;
  limitations: string[];
  simulator?: string;
  paperIds: string[];
};

export type SectionId =
  | "overview"
  | "path"
  | "timeline"
  | "graph"
  | "gaps"
  | "foundations"
  | "pipeline"
  | "math"
  | "papers"
  | "compare"
  | "datasets"
  | "metrics"
  | "audit";

export const SECTIONS: { id: SectionId; label: string; blurb: string }[] = [
  { id: "overview", label: "Overview", blurb: "The tracking lab at a glance" },
  { id: "path", label: "Learning Path", blurb: "Beginner → research, in order" },
  { id: "timeline", label: "Research Timeline", blurb: "Evolution of ideas, 2014–2026" },
  { id: "graph", label: "Research Graph", blurb: "Paper-to-paper lineage" },
  { id: "gaps", label: "Research Gaps", blurb: "Why each paper had to exist" },
  { id: "foundations", label: "Foundations", blurb: "Concepts and prerequisites" },
  { id: "pipeline", label: "Tracking Pipelines", blurb: "How a tracker actually works" },
  { id: "math", label: "Math Lab", blurb: "Equations you can operate" },
  { id: "papers", label: "Paper Explorer", blurb: "159 papers, deep records" },
  { id: "compare", label: "Compare", blurb: "Technical differences, not scores" },
  { id: "datasets", label: "Datasets", blurb: "What the field measures on" },
  { id: "metrics", label: "Metrics", blurb: "MOTA, IDF1, HOTA and friends" },
  { id: "audit", label: "Audit", blurb: "Corpus coverage & provenance" },
];
