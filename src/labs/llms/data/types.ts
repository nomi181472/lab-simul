/**
 * Corpus contracts for the Transformers / LLM / micro-LLM lab.
 *
 * Every evidence-bearing field carries the quote it was extracted from. The
 * generator only ever fills a field from a sentence it actually found in the
 * paper, so an empty array means "the text did not say", not "the extractor
 * failed silently".
 */

/** One structural element of a model, with the sentence that evidences it. */
export interface ArchLayer {
  kind:
    | "tokenizer"
    | "embedding"
    | "positional"
    | "attention"
    | "ffn"
    | "norm"
    | "router"
    | "adapter"
    | "cache"
    | "head";
  label: string;
  quote: string;
}

/** A named transformation applied to a model, with the sentence that states it. */
export interface Change {
  operator:
    | "quantization"
    | "pruning"
    | "distillation"
    | "low-rank-adaptation"
    | "mixed-precision"
    | "layer-reduction"
    | "token-reduction"
    | "attention-approximation"
    | "weight-sharing"
    | "architecture-search"
    | "speculative-decoding"
    | "caching"
    | "calibration";
  label: string;
  quote: string;
}

/** The axis a paper is optimising: quality, or cost. */
export interface Objective {
  kind:
    | "accuracy"
    | "perplexity"
    | "latency"
    | "memory"
    | "throughput"
    | "energy"
    | "compression"
    | "size";
  label: string;
  quote: string;
}

/** A benchmark the paper reports against, with the mentioning sentence. */
export interface Result {
  dataset: string;
  label: string;
  quote: string;
}

export interface ArchitectureBlock {
  summary: string;
  layers: ArchLayer[];
}

export interface PaperRecord {
  id: string;
  openalex: string;
  arxiv: string;
  doi: string;
  title: string;
  shortTitle: string;
  year: number | null;
  venue: string;
  authors: string[];
  citations: number;
  pages: number;
  fileName: string;
  pdfUrl: string;
  relevance: "transformers";
  abstractNote: string;
  textChars: number;
  architecture: ArchitectureBlock;
  changes: Change[];
  objectives: Objective[];
  results: Result[];
  referenceCount: number;
}

/** One row of the corpus manifest, ordered by descending citation count. */
export interface CorpusEntry {
  id: string;
  title: string;
  year: number | null;
  citations: number;
  venue: string;
  arxiv: string;
  openalex: string;
  fileName: string;
  pages: number;
  hasArchitecture: boolean;
  changeCount: number;
}

export interface CorpusManifest {
  generatedBy: string;
  window: string;
  ordering: string;
  total: number;
  totalCitations: number;
  stats: Record<string, unknown>;
  papers: CorpusEntry[];
}

export interface ConceptLink {
  key: string;
  label: string;
  papers: string[];
}

export interface MetricLink {
  key: string;
  label: string;
  papers: string[];
}

export interface DatasetLink {
  key: string;
  label: string;
  papers: string[];
}