/**
 * Display vocabulary for the Transformers lab.
 *
 * `ArchKind` and `ChangeOperator` are re-exported from the generated contract so
 * a colour map that forgets a variant is a compile error rather than an
 * undefined at runtime.
 */
import type { ArchLayer, Change } from "./types";

export type ArchKind = ArchLayer["kind"];
export type ChangeOperator = Change["operator"];

/** Coarse -> fine: the emitted stack reads top to bottom. */
export const ARCH_LABEL: Record<ArchKind, string> = {
  tokenizer: "Tokenizer",
  embedding: "Embedding",
  positional: "Positional",
  attention: "Attention",
  ffn: "Feed-forward",
  norm: "Normalization",
  router: "MoE router",
  adapter: "Adapter",
  cache: "KV cache",
  head: "Output head",
};

export const ARCH_COLOR: Record<ArchKind, string> = {
  tokenizer: "#a1a1aa",
  embedding: "#38bdf8",
  positional: "#818cf8",
  attention: "#22d3ee",
  ffn: "#34d399",
  norm: "#fbbf24",
  router: "#f472b6",
  adapter: "#c084fc",
  cache: "#fb923c",
  head: "#4ade80",
};

export const CHANGE_LABEL: Record<ChangeOperator, string> = {
  quantization: "Quantization",
  pruning: "Pruning",
  distillation: "Distillation",
  "low-rank-adaptation": "Low-rank",
  "mixed-precision": "Mixed precision",
  "layer-reduction": "Layer reduction",
  "token-reduction": "Token reduction",
  "attention-approximation": "Attention approx.",
  "weight-sharing": "Weight sharing",
  "architecture-search": "Architecture search",
  "speculative-decoding": "Speculative",
  caching: "Caching",
  calibration: "Calibration",
};

export const CHANGE_COLOR: Record<ChangeOperator, string> = {
  quantization: "#22d3ee",
  pruning: "#a78bfa",
  distillation: "#34d399",
  "low-rank-adaptation": "#c084fc",
  "mixed-precision": "#60a5fa",
  "layer-reduction": "#fb923c",
  "token-reduction": "#f472b6",
  "attention-approximation": "#fbbf24",
  "weight-sharing": "#4ade80",
  "architecture-search": "#f87171",
  "speculative-decoding": "#e879f9",
  caching: "#2dd4bf",
  calibration: "#94a3b8",
};

/** Every declared variant, so the UI can show the full space and mark the gaps. */
export const ALL_ARCH_KINDS: ArchKind[] = Object.keys(ARCH_LABEL) as ArchKind[];
export const ALL_CHANGE_OPERATORS: ChangeOperator[] =
  Object.keys(CHANGE_LABEL) as ChangeOperator[];