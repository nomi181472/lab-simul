/* Core research data model for the WiFi Sensing Research Lab.
 *
 * Provenance rules (same policy as the Object Detection / Object Tracking labs):
 *  - Every record field must be grounded in the corpus PDF named by `fileName`.
 *  - results[] only contain numbers actually reported in the paper.
 *  - relations[] only connect corpus papers (W001..W300) when the paper itself
 *    builds on / compares against / addresses the cited work.
 *  - Nothing is invented: no invented equations, results, venues, or links.
 *
 * The distinguishing property of this corpus versus the vision labs: a paper is
 * defined by *what signal it senses with* and *what it infers from that signal*,
 * not by pixels. Hence `modality` (CSI vs RSSI vs radar) and `signalChain`
 * carry the same weight as `architecture` does in the tracking lab.
 */

export type Confidence = "HIGH" | "MEDIUM" | "LOW";

/** How a paper relates to its predecessor inside the corpus. */
export type RelationType =
  | "builds-on"
  | "improves"
  | "replaces"
  | "extends"
  | "combines"
  | "uses-as-baseline"
  | "addresses-limitation"
  | "conceptual-successor";

export type Relation = {
  /** corpus id of the earlier paper (W001..W300) */
  to: string;
  type: RelationType;
  /** one line: what the current paper takes / fixes / changes */
  note?: string;
};

/** The perceptual task a WiFi paper claims to solve. */
export type Task =
  | "activity-recognition"
  | "gesture-recognition"
  | "fall-detection"
  | "localization"
  | "tracking"
  | "vital-signs"
  | "occupancy-presence"
  | "posture-pose"
  | "keystroke-identification"
  | "breathing"
  | "through-wall"
  | "in-baggage"
  | "material-signature"
  | "benchmark"
  | "survey"
  | "other";

/**
 * What the receiver actually measures. This is the single most important
 * axis in the field: RSSI is a single scalar per packet, CSI is a complex
 * value per subcarrier, and the choice constrains every downstream task.
 */
export type SensingModality =
  | "csi-toolkit"
  | "csi-80211n"
  | "rssi"
  | "wifi-radar"
  | "wifi-sonar"
  | "hybrid"
  | "other";

export type Difficulty = "intro" | "intermediate" | "advanced";

export type Bandwidth = "20 MHz" | "40 MHz" | "80 MHz" | "160 MHz" | "unknown";

export type Equation = {
  id: string;
  label: string;
  /** plain-text / unicode math, rendered in a monospace block */
  formula: string;
  variables: { symbol: string; meaning: string }[];
  /** plain-language meaning for a non-engineer */
  intuition: string;
  /** why the equation is needed at all */
  why: string;
  /** where in the pipeline it is applied */
  where: string;
  /** how changing parameters changes behaviour */
  params?: string;
  /** key into SIMULATOR_INDEX when an interactive sim exists */
  simulator?: string;
  paperIds?: string[];
};

/**
 * The WiFi sensing pipeline, in the order a packet actually flows.
 * This mirrors PaperMethod in the tracking lab but names RF-specific stages.
 */
export type SignalChain = {
  /** e.g. "802.11n 3x3 MIMO laptop NIC, 20MHz, 100Hz sampling" */
  hardware: string;
  /** what is captured per packet */
  captured: string;
  /** e.g. "bandpass, detrend, unwrap, sanitize" */
  preprocessing: string[];
  /** e.g. "Doppler spectrogram, RMS, DFR" */
  features: string[];
  /** e.g. "2D CNN + LSTM", "gradient boosting", "DTW" */
  model: string;
  /** e.g. "TX-RX 3m, target 4m, NLOS through wall"; NOT_DESCRIBED if unstated */
  deployment: string;
};

export type Limitations = {
  /** stated by the authors themselves */
  authorStated: string[];
  /** visible from the methodology (lab synthesis, never presented as author claim) */
  evident: string[];
};

export type PaperRecord = {
  id: string; // W001..W300 — must match the manifest entry
  arxiv: string;
  doi?: string;
  title: string;
  shortTitle: string;
  year: number;
  authors: string[];
  venue?: string;
  fileName: string;
  task: Task;
  modality: SensingModality;
  bandwidth: Bandwidth;
  /** Crossref is-referenced-by-count at harvest time */
  citations: number;
  /** paradigm / family tags, lowercase kebab */
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
  /** the RF → inference pipeline this paper builds */
  chain: SignalChain;
  /**
   * ids of the standard equations this paper demonstrably uses, resolved by
   * keyword match over the paper's own text (see scripts/wifi_records.py).
   * The formula itself lives in concepts.ts and is not duplicated here.
   */
  equationIds: string[];
  /** dataset ids from datasets.ts */
  datasets: string[];
  /** metric ids from metrics.ts */
  metrics: string[];
  baselines: string[];
  /** factual, paper-reported outcomes (numbers only if printed in the paper) */
  results: string[];
  ablations: string[];
  limitations: Limitations;
  /** sentences from the paper that state an explicit assumption */
  assumptions?: string[];
  /** sentences describing computational cost / runtime */
  computation?: string[];
  /** earlier corpus papers this one relates to */
  relations: Relation[];
  /** concept ids this paper advances */
  concepts: string[];
  impact?: string;
  /** how this paper obtained ground truth, detected from its own text */
  groundTruth?: "manual-annotation" | "wearable-sensor" | "depth-camera" | "radar" | "simulated";
  /** real page count from pdfinfo */
  pages?: number;
};

export type ConceptCategory =
  | "propagation"
  | "signal"
  | "features"
  | "estimation"
  | "learning"
  | "tasks"
  | "evaluation";

export type Concept = {
  id: string;
  name: string;
  category: ConceptCategory;
  intuition: string;
  formula?: string;
  simulator?: string;
  /** prerequisite concept ids */
  prereqs: string[];
  paperIds: string[];
};

export type DatasetRecord = {
  id: string;
  name: string;
  year: number;
  purpose: string;
  domain: string;
  /** how the ground truth was obtained — the hardest part in WiFi sensing */
  groundTruth: "manual-annotation" | "wearable-sensor" | "depth-camera" | "radar" | "simulated";
  subjects?: number;
  locations?: number;
  samples?: string;
  characteristics: string[];
  metrics: string[];
  paperIds: string[];
};

export type MetricFamily = "classification" | "localization" | "vital-sign" | "detection" | "efficiency";

export type MetricRecord = {
  id: string;
  name: string;
  family: MetricFamily;
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
  | "signal"
  | "math"
  | "papers"
  | "compare"
  | "datasets"
  | "metrics"
  | "audit";

export const SECTIONS: { id: SectionId; label: string; blurb: string }[] = [
  { id: "overview", label: "Overview", blurb: "The WiFi sensing lab at a glance" },
  { id: "path", label: "Learning Path", blurb: "Beginner → research, in order" },
  { id: "timeline", label: "Timeline", blurb: "Evolution of CSI sensing, 2010–2026" },
  { id: "graph", label: "Research Graph", blurb: "Paper-to-paper lineage" },
  { id: "gaps", label: "Research Gaps", blurb: "Why each paper had to exist" },
  { id: "foundations", label: "Foundations", blurb: "RF concepts and prerequisites" },
  { id: "signal", label: "Signal Chain", blurb: "From packet to prediction" },
  { id: "math", label: "Math Lab", blurb: "20 simulators you can operate" },
  { id: "papers", label: "Paper Explorer", blurb: "300 papers, deep records" },
  { id: "compare", label: "Compare", blurb: "Modality and task differences" },
  { id: "datasets", label: "Datasets", blurb: "What the field measures on" },
  { id: "metrics", label: "Metrics", blurb: "Accuracy, MAE, F1 and friends" },
  { id: "audit", label: "Audit", blurb: "Corpus coverage & provenance" },
];

export const TASK_LABEL: Record<Task, string> = {
  "activity-recognition": "Activity recognition",
  "gesture-recognition": "Gesture recognition",
  "fall-detection": "Fall detection",
  localization: "Localization",
  tracking: "Tracking",
  "vital-signs": "Vital signs",
  "occupancy-presence": "Occupancy / presence",
  "posture-pose": "Posture / pose",
  "keystroke-identification": "Keystroke identification",
  breathing: "Breathing",
  "through-wall": "Through-wall",
  "in-baggage": "In-baggage",
  "material-signature": "Material signature",
  benchmark: "Benchmark",
  survey: "Survey",
  other: "Other",
};

export const MODALITY_LABEL: Record<SensingModality, string> = {
  "csi-toolkit": "CSI (toolkit)",
  "csi-80211n": "CSI (802.11n NIC)",
  rssi: "RSSI",
  "wifi-radar": "WiFi radar (FMCW)",
  "wifi-sonar": "WiFi sonar",
  hybrid: "Hybrid",
  other: "Other",
};