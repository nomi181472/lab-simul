/* Core research data model for the Neuroevolution Research Lab.
 *
 * Provenance rules (same policy as the Object Detection / Object Tracking /
 * WiFi Sensing labs):
 *  - Every record field must be grounded in the corpus PDF named by `fileName`.
 *  - results[] only contain numbers actually printed in the paper.
 *  - relations[] only connect corpus papers (N001..Nnnn) when the paper itself
 *    builds on / compares against / addresses the cited work.
 *  - Nothing is invented: no invented equations, results, venues, or links.
 *
 * The one thing this lab adds on top of the shared model is `architecture`.
 * A neuroevolution paper is unintelligible without it: the reader needs to see
 * what shape of network came out of the search, and *what the operators
 * actually modified* to produce it. So `Architecture` is a first-class,
 * required field rather than a free-text note, and every part of it is either
 * quoted from the paper or explicitly flagged as not stated.
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
  | "conceptual-successor"
  /** cited, but this paper's text does not say how it relates */
  | "related";

export type Relation = {
  /** corpus id of the earlier paper (N001..Nnnn) */
  to: string;
  type: RelationType;
  /** one line: what the current paper takes / fixes / changes */
  note?: string;
};

/** What kind of thing the search produces. */
export type Phenotype =
  | "mlp"
  | "cnn"
  | "rnn"
  | "modular"
  | "neurocontroller"
  | "policy"
  | "architecture-search"
  | "program"
  | "ensemble"
  | "other";

/** One layer/step of the evolved network, in the order the paper lists it. */
export type LayerKind =
  | "input"
  | "dense"
  | "conv"
  | "pool"
  | "recurrent"
  | "attention"
  | "skip"
  | "output"
  | "step"
  | "unknown";

export type ArchitectureLayer = {
  kind: LayerKind;
  /** short label, e.g. "Conv 32·3×3" */
  label: string;
  /** node/channel count when the paper states one */
  units?: number;
  /** quoted sentence from the paper that establishes this layer.
   * Required: a layer with no evidence is not a finding, so the generator
   * omits the layer instead. */
  quote: string;
};

/**
 * Which part of a candidate the operators are allowed to touch. This is the
 * "how the modification is being done" axis: two papers can both call
 * themselves neuroevolution yet differ completely here.
 */
export type Locus =
  | "weights"
  | "topology"
  | "activation"
  | "learning-rate"
  | "architecture"
  | "hyperparameters"
  | "rule"
  | "neurons"
  | "connections"
  | "behavior";

export type LocusChange = {
  locus: Locus;
  /** how the operator alters this locus */
  operator: string;
  /** quoted sentence from the paper that establishes it.
   * Required for the same reason as ArchitectureLayer.quote. */
  quote: string;
};

export type Encoding =
  | "direct"
  | "indirect"
  | "developmental"
  | "modular"
  | "graph"
  | "other";

export type AlgorithmFamily =
  | "ga"
  | "gp"
  | "es"
  | "qd"
  | "novelty-search"
  | "neat"
  | "swarm"
  | "other";

/** The evolutionary machinery, as the paper describes it. */
export type EvolutionMethod = {
  family: AlgorithmFamily;
  /** the paper's words for the algorithm, e.g. "truncation selection + SBX crossover" */
  algorithm: string;
  /** what one genome is */
  representation: string;
  encoding: Encoding;
  /** operators that perturb a candidate */
  mutation: string[];
  /** operators that recombine two candidates */
  crossover: string[];
  /** how survivors are chosen */
  selection: string[];
  /** what is actually modified — the core of the lab */
  changes: LocusChange[];
  /** the objective, quoted */
  fitness?: string;
  /** quoted sentence establishing the phenotype */
  phenotypeQuote?: string;
};

export type Architecture = {
  /** one line: what shape of network this paper evolved */
  summary: string;
  phenotype: Phenotype;
  /** ordered layers as stated by the paper; may be empty when unstated */
  layers: ArchitectureLayer[];
  /** free-text details the paper gives that layers alone cannot carry */
  details: string[];
  /** how genotype maps to the network above */
  genotypeToPhenotype?: string;
  /** the paper's quoted architecture sentences, so the claim stays checkable */
  quotes: string[];
};

export type Method = {
  /** the evolutionary search itself */
  evolution: EvolutionMethod;
  /** what the network is for */
  task: string;
  /** how the paper evaluates a candidate */
  evaluation: string;
};

export type Limitations = {
  /** stated by the authors themselves */
  authorStated: string[];
  /** visible from the methodology (lab synthesis, never presented as author claim) */
  evident: string[];
};

export type Difficulty = "intro" | "intermediate" | "advanced";

export type PaperRecord = {
  id: string; // N001..Nnnn — must match the manifest entry
  arxiv: string;
  title: string;
  shortTitle: string;
  year: number;
  authors: string[];
  venue?: string;
  doi?: string;
  fileName: string;
  citations: number;
  pages?: number;
  /** paradigm family, lowercase kebab, e.g. "quality-diversity", "strongly-typed-gp" */
  tags: string[];
  difficulty: Difficulty;
  /** 2–3 sentence plain explanation */
  summary: string;
  /** what problem existed and why it was hard */
  problem: string;
  /** concept ids a reader needs first (see concepts.ts) */
  background: string[];
  /** one line: what was missing before this paper */
  researchGap: string;
  /** concrete contributions, quoted */
  contribution: string[];
  /** REQUIRED: what the search built and what it modified */
  architecture: Architecture;
  method: Method;
  /** dataset ids from datasets.ts */
  datasets: string[];
  /** metric ids from metrics.ts */
  metrics: string[];
  baselines: string[];
  /** factual, paper-reported outcomes (numbers only if printed in the paper) */
  results: string[];
  /** sentences describing sensitivity to search settings */
  ablations: string[];
  limitations: Limitations;
  /** quoted sentences stating explicit assumptions */
  assumptions?: string[];
  /** sentences describing computational cost / population size / generations */
  computation?: string[];
  /** earlier corpus papers this one relates to */
  relations: Relation[];
  /** concept ids this paper advances */
  concepts: string[];
  /** quoted sentences on generalisation / transfer */
  impact?: string[];
  /** how ground truth or the objective reference is obtained, if stated */
  groundTruth?: string;
};

export type Concept = {
  id: string;
  name: string;
  category:
    | "basics"
    | "encoding"
    | "operators"
    | "selection"
    | "diversity"
    | "evaluation"
    | "applications";
  intuition: string;
  formula?: string;
  variables?: { symbol: string; meaning: string }[];
  simulator?: string;
  /** prerequisite concept ids */
  prereqs: string[];
  paperIds?: string[];
};

export type DatasetRecord = {
  id: string;
  name: string;
  year: number;
  purpose: string;
  domain: string;
  task: string;
  observations?: number;
  agents?: number;
  episodes?: number;
  /** quoted sentence from a corpus paper establishing the dataset */
  characteristics: string[];
  metrics: string[];
  paperIds: string[];
};

export type MetricRecord = {
  id: string;
  name: string;
  family: "fitness" | "performance" | "diversity" | "efficiency";
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
  | "architecture"
  | "modification"
  | "path"
  | "timeline"
  | "graph"
  | "foundations"
  | "operators"
  | "papers"
  | "compare"
  | "datasets"
  | "metrics"
  | "audit";

export const SECTIONS: { id: SectionId; label: string; blurb: string }[] = [
  { id: "overview", label: "Overview", blurb: "The neuroevolution lab at a glance" },
  { id: "architecture", label: "Architecture", blurb: "What each search actually built" },
  { id: "modification", label: "Modification", blurb: "How the operators change a candidate" },
  { id: "path", label: "Learning Path", blurb: "Beginner → research, in order" },
  { id: "timeline", label: "Timeline", blurb: "Evolution of the field, 1990s–2026" },
  { id: "graph", label: "Research Graph", blurb: "Paper-to-paper lineage" },
  { id: "foundations", label: "Foundations", blurb: "Concepts and prerequisites" },
  { id: "operators", label: "Operators", blurb: "Mutation, crossover, selection, diversity" },
  { id: "papers", label: "Paper Explorer", blurb: "Every paper with its architecture" },
  { id: "compare", label: "Compare", blurb: "Encoding and operator differences" },
  { id: "datasets", label: "Datasets", blurb: "Environments the field searches in" },
  { id: "metrics", label: "Metrics", blurb: "Fitness, performance, diversity" },
  { id: "audit", label: "Audit", blurb: "Corpus coverage & provenance" },
];

/** Colour per layer kind, shared by the architecture diagrams. */
export const LAYER_COLOR: Record<LayerKind, string> = {
  input: "#38bdf8",
  dense: "#a78bfa",
  conv: "#34d399",
  pool: "#22d3ee",
  recurrent: "#fbbf24",
  attention: "#f472b6",
  skip: "#94a3b8",
  output: "#fb7185",
  step: "#c084fc",
  unknown: "#71717a",
};

/** One-line label per layer kind, for legends and accessibility. */
export const LAYER_LABEL: Record<LayerKind, string> = {
  input: "Input",
  dense: "Fully-connected",
  conv: "Convolution",
  pool: "Pooling",
  recurrent: "Recurrent",
  attention: "Attention",
  skip: "Skip / residual",
  output: "Output",
  step: "Search step",
  unknown: "Unstated",
};

/** Colour per locus, for the modification diagrams. */
export const LOCUS_COLOR: Record<Locus, string> = {
  weights: "#34d399",
  topology: "#f472b6",
  connections: "#fbbf24",
  neurons: "#a78bfa",
  activation: "#22d3ee",
  "learning-rate": "#60a5fa",
  architecture: "#fb7185",
  hyperparameters: "#c084fc",
  rule: "#4ade80",
  behavior: "#f59e0b",
};

export const LOCUS_LABEL: Record<Locus, string> = {
  weights: "Weights",
  topology: "Layer topology",
  connections: "Connections",
  neurons: "Neuron count",
  activation: "Activation",
  "learning-rate": "Learning rate",
  architecture: "Architecture genes",
  hyperparameters: "Hyperparameters",
  rule: "Program rules",
  behavior: "Behaviour policy",
};

export const PHENOTYPE_LABEL: Record<Phenotype, string> = {
  mlp: "Feed-forward MLP",
  cnn: "Convolutional network",
  rnn: "Recurrent network",
  modular: "Modular network",
  neurocontroller: "Neurocontroller",
  policy: "Evolved policy",
  "architecture-search": "Architecture search",
  program: "Evolved program",
  ensemble: "Ensemble",
  other: "Other",
};

export const FAMILY_LABEL: Record<AlgorithmFamily, string> = {
  ga: "Genetic algorithm",
  gp: "Genetic programming",
  es: "Evolution strategy",
  qd: "Quality-diversity",
  "novelty-search": "Novelty search",
  neat: "NEAT",
  swarm: "Swarm / other",
  other: "Other",
};

export const ENCODING_LABEL: Record<Encoding, string> = {
  direct: "Direct encoding",
  indirect: "Indirect encoding",
  developmental: "Developmental",
  modular: "Modular",
  graph: "Graph / syntax",
  other: "Other",
};