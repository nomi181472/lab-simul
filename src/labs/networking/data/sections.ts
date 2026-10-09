export type SectionId =
  | "overview"
  | "corpus"
  | "years"
  | "evolution"
  | "matrix"
  | "clusters"
  | "industry"
  | "future"
  | "benchmarks"
  | "saturation"
  | "gaps"
  | "adoption"
  | "simulators"
  | "analogies"
  | "audit";

export interface SectionMeta {
  id: SectionId;
  label: string;
  blurb: string;
}

export const SECTIONS: SectionMeta[] = [
  { id: "overview", label: "Dashboard", blurb: "Corpus at a glance: domains, citations, evidence depth." },
  { id: "corpus", label: "Corpus", blurb: "All 300 papers, most cited first, quoted evidence and links." },
  { id: "years", label: "Year Explorer", blurb: "Year-by-year comparison across 2011–2023." },
  { id: "evolution", label: "Evolution", blurb: "Problem and solution evolution, era by era." },
  { id: "matrix", label: "Problem Matrix", blurb: "Domain × period incidence grid." },
  { id: "clusters", label: "Clusters", blurb: "Identical-problem clusters: repeated attempts at the same problem." },
  { id: "industry", label: "Industry", blurb: "Industry evidence, deployment quotes, application evolution." },
  { id: "future", label: "Future Tracker", blurb: "Future directions from past papers vs. later realization." },
  { id: "benchmarks", label: "Benchmarks", blurb: "Datasets, metrics and algorithms maps; metric evolution." },
  { id: "saturation", label: "Saturation", blurb: "Attention vs. progress; remaining unsolved gaps." },
  { id: "gaps", label: "Gaps", blurb: "Academic-only and industry-only problem gaps." },
  { id: "adoption", label: "Adoption", blurb: "Academic ↔ industry adoption ratio per domain." },
  { id: "simulators", label: "Simulators", blurb: "Interactive models of the identical-problem clusters." },
  { id: "analogies", label: "Analogies", blurb: "Cross-domain analogies from physics, chemistry, biology." },
  { id: "audit", label: "Audit", blurb: "Provenance: every fact traces back to a paper." },
];
