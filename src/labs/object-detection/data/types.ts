/* Core research data model for the Object Detection Research Lab.
 *
 * Provenance rules:
 *  - "paperSays" claims are quotes/paraphrases grounded in a specific corpus PDF.
 *  - "crossPaper" claims are consistent statements across two or more corpus papers.
 *  - "labInterpretation" is an analytic synthesis by the lab; never shown as fact.
 *  - If a claim is not supported we mark evidence: "INSUFFICIENT_EVIDENCE".
 */

export type EvidenceKind = "paperSays" | "crossPaper" | "labInterpretation";

export type Confidence = "HIGH" | "MEDIUM" | "LOW";

export type Evidence = {
  paperIds: string[];
  kind: EvidenceKind;
  confidence: Confidence;
  /** e.g. "Abstract", "Sec 3.1", "Fig 2", "Table 1", "Eq 4" */
  location?: string;
  quote?: string;
};

export type Paper = {
  id: string;
  arxiv?: string;
  title: string;
  shortTitle: string;
  year: number;
  authors: string[];
  fileName: string;
  /** original corpus filename */
  venue?: string;
  tags: string[];
  summary: string;
  contributions: string[];
  /** concept/problem ids this paper evidences */
  evidences: string[];
};

export type YearResearchState = {
  year: number;
  dominantApproaches: string[];
  emergingIdeas: string[];
  decliningIdeas: string[];
  persistentProblems: string[];
  newProblems: string[];
  solvedOrReducedProblems: string[];
  unsolvedProblems: string[];
  mathematicalDevelopments: string[];
  architecturalDevelopments: string[];
  computationalDevelopments: string[];
  applications: string[];
  edgeCases: string[];
  researchDirections: string[];
  supportingPapers: string[];
};

export type YearDelta = {
  from: number;
  to: number;
  added: string[];
  removed: string[];
  persisted: string[];
  modified: string[];
  newProblems: string[];
  newArchitectures: string[];
  newLosses: string[];
  newApplications: string[];
  newScenarios: string[];
  summary: string;
  supportingPapers: string[];
};

export type ProblemOccurrence = {
  year: number;
  papers: string[];
  attemptedSolutions: string[];
  evidence: string[];
};

export type ProblemStatus = "resolved" | "reduced" | "persistent" | "transformed" | "uncertain";

export type ProblemLifecycle = {
  id: string;
  problem: string;
  shortLabel: string;
  firstObservedYear: number;
  occurrences: ProblemOccurrence[];
  currentStatus: ProblemStatus;
  statusEvidence: string;
};

export type DetectionScenario = {
  name: string;
  firstObservedYear: number;
  papers: string[];
  difficulty: string[];
  attemptedSolutions: string[];
  remainingProblems: string[];
};

export type Concept = {
  id: string;
  name: string;
  category: "mathematics" | "architecture" | "training" | "postprocessing" | "data";
  intuition: string;
  formula?: string;
  simulator?: string;
  paperIds: string[];
  motivatesAllOf?: string[];
  solves?: string;
  limitation?: string;
};

export type ResearchDirection = {
  id: string;
  name: string;
  firstAppearance: number;
  sourceIn: string[]; // prior paradigm ids
  representativePapers: string[];
  driver: string;
  previousLimitation: string;
  unlockedCapability: string;
  newProblems: string[];
  status: Confidence;
};

export type ResearchEpisode = {
  id: string;
  title: string;
  year: number;
  problem: string;
  pressure: string;
  idea: string;
  mathematics: string[];
  architecture: string;
  capability: string;
  application: string;
  tradeoff: string;
  newProblem: string;
  paperIds: string[];
  /** downstream episode ids (forks / convergence) */
  leadsTo: string[];
  evidence: Evidence;
};

export type ApplicationLink = {
  id: string;
  name: string;
  researchCapability: string;
  whyUseful: string;
  constraints: string;
  detectorEvolution: string;
  edgeCases: string;
  paperIds: string[];
  firstAppearance: number;
  status: Confidence;
};

export type ReportSection = {
  id: string;
  label: string;
  kind: string;
};

export type SectionId =
  | "overview"
  | "year"
  | "evolution"
  | "problems"
  | "directions"
  | "applications"
  | "benchmark"
  | "explorer"
  | "ask"
  | "missing"
  | "papers"
  | "math"
  | "edge"
  | "audit";

export const SECTIONS: { id: SectionId; label: string; blurb: string }[] = [
  { id: "overview", label: "Overview", blurb: "The lab at a glance" },
  { id: "year", label: "Year-by-Year", blurb: "Field state per year" },
  { id: "evolution", label: "Evolution", blurb: "How the field forked and converged" },
  { id: "problems", label: "Problem Matrix", blurb: "Problems across years" },
  { id: "directions", label: "Directions", blurb: "Open threads and oscillations" },
  { id: "applications", label: "Applications", blurb: "Where detectors landed" },
  { id: "benchmark", label: "Benchmark Map", blurb: "Design space of 48 papers" },
  { id: "explorer", label: "Corpus Explorer", blurb: "Cross-filter the evidence" },
  { id: "ask", label: "Ask the Corpus", blurb: "Questions answered from evidence only" },
  { id: "missing", label: "What's Missing", blurb: "Off-the-corpus knowledge" },
  { id: "papers", label: "Paper Explorer", blurb: "48 papers, evidence-tagged" },
  { id: "math", label: "Mathematics Lab", blurb: "Interactive simulators" },
  { id: "edge", label: "Edge Cases", blurb: "Detection scenarios" },
  { id: "audit", label: "Research Audit", blurb: "Groundedness checks" },
];