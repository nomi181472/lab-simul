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

export interface SectionMeta {
  id: SectionId;
  label: string;
  /** One line on what the section actually shows. */
  blurb: string;
}

export const SECTIONS: SectionMeta[] = [
  {
    id: "overview",
    label: "Overview",
    blurb: "The arc from attention to micro-LLM, in numbers.",
  },
  {
    id: "architecture",
    label: "Architecture",
    blurb: "The structural stack of each model, quoted from the paper.",
  },
  {
    id: "modification",
    label: "Modification",
    blurb: "Which transformation each paper applies, and to what.",
  },
  {
    id: "path",
    label: "Scaling Path",
    blurb: "Follow the thread from pretrained base to deployed micro-model.",
  },
  {
    id: "timeline",
    label: "Timeline",
    blurb: "Papers by year across the 2019-2026 window.",
  },
  {
    id: "graph",
    label: "Graph",
    blurb: "Concept and operator co-occurrence across the corpus.",
  },
  {
    id: "foundations",
    label: "Foundations",
    blurb: "The most-cited work, which is what the field leaned on.",
  },
  {
    id: "operators",
    label: "Operators",
    blurb: "Quantize, prune, distill: the compression vocabulary.",
  },
  {
    id: "papers",
    label: "Papers",
    blurb: "All papers, most cited first, with evidence and PDFs.",
  },
  {
    id: "compare",
    label: "Compare",
    blurb: "Side-by-side quality against cost for two papers.",
  },
  {
    id: "datasets",
    label: "Benchmarks",
    blurb: "Which benchmarks the corpus actually reports against.",
  },
  {
    id: "metrics",
    label: "Metrics",
    blurb: "Quality metrics and cost metrics, counted by paper.",
  },
  {
    id: "audit",
    label: "Audit",
    blurb: "Provenance checks over the whole corpus.",
  },
];