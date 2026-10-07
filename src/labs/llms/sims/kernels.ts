/**
 * Deterministic simulator kernels for the Transformers lab.
 *
 * Each kernel is a small, honest model of one mechanism from the corpus:
 * quantization really does trade bits for error, KV-cache really does grow
 * linearly with sequence length, speculative decoding really does trade a draft
 * model for fewer sequential steps. The numbers are illustrative, not measured,
 * and every run is pure so a reader can trace what produced a curve.
 *
 * Nothing here is claimed to be what an individual paper did. The papers each
 * kernel is grounded in are listed in `evidence`, and the UI shows them.
 */
import type { SimSeries, SimStep } from "./types";

/** Deterministic pseudo-noise, so curves are reproducible across renders. */
function wobble(i: number, seed: number): number {
  const x = Math.sin((i + 1) * 12.9898 + seed * 78.233) * 43758.5453;
  return x - Math.floor(x) - 0.5;
}

function series(key: string, label: string, pts: { label: string; value: number }[]): SimSeries {
  return { key, label, points: pts };
}

export interface Kernel {
  run: (config: Record<string, number>) => { series: SimSeries[]; steps: SimStep[] };
}

export const BITS = [16, 8, 6, 4, 3, 2];

export const quantization: Kernel = {
  run: (c) => {
    const bits = c.bits ?? 8;
    const steps = BITS.map((b) => ({ label: `${b}-bit`, value: b }));
    // Perplexity rises as bits fall; the knee is between 4 and 3 bits.
    const ppl = BITS.map((b) => ({
      label: `${b}-bit`,
      value:
        Math.round(
          (7.2 + Math.pow(2, (16 - b) / 2.4) * 0.42 + wobble(b, 1) * 0.35) * 100,
        ) / 100,
    }));
    const size = BITS.map((b) => ({ label: `${b}-bit`, value: (1 / b) * 100 }));
    return {
      series: [
        series("ppl", "perplexity (worse ↑)", ppl),
        series("size", "weight memory (of 16-bit)", size),
      ],
      steps: steps.map((s, i) => ({
        t: i,
        action: `${s.value}-bit grid`,
        detail:
          i === 0
            ? "baseline: no quantization, 16 bits per weight"
            : `halving the grid from ${BITS[i - 1]} to ${s.value} bits; perplexity goes ${ppl[i - 1].value} → ${ppl[i].value} while memory falls to ${Math.round(size[i].value)}%`,
      })),
    };
  },
};

export const KEEP_FRACTIONS = [1, 0.9, 0.75, 0.5, 0.25, 0.1];

export const pruning: Kernel = {
  run: (c) => {
    const keep = c.keep ?? 0.5;
    const steps = KEEP_FRACTIONS.map((k) => ({ label: `${Math.round(k * 100)}% kept`, value: k }));
    // Error grows superlinearly once below half density.
    const ppl = KEEP_FRACTIONS.map((k) => ({
      label: `${Math.round(k * 100)}%`,
      value:
        Math.round((7.2 + Math.pow(1 - k, 2.3) * 26 + wobble(k * 10, 2) * 0.5) * 100) / 100,
    }));
    return {
      series: [
        series("ppl", "perplexity (worse ↑)", ppl),
        series("sparsity", "weights remaining", KEEP_FRACTIONS.map((k) => ({ label: `${Math.round(k * 100)}%`, value: k * 100 }))),
      ],
      steps: steps.map((s, i) => ({
        t: i,
        action: s.label,
        detail:
          i === 0
            ? "dense baseline, nothing removed"
            : `${Math.round((KEEP_FRACTIONS[i - 1] - s.value) * 100)}% of weights zeroed; perplexity ${ppl[i - 1].value} → ${ppl[i].value}`,
      })),
    };
  },
};

export const DEPTHS = [48, 36, 24, 16, 12, 8];

export const layerReduction: Kernel = {
  run: () => {
    const ppl = DEPTHS.map((d) => ({
      label: `${d}L`,
      value: Math.round((9.1 - Math.log2(48 / d) * 1.35 + wobble(d, 3) * 0.3) * 100) / 100,
    }));
    const lat = DEPTHS.map((d) => ({ label: `${d}L`, value: Math.round(d * 0.94) }));
    return {
      series: [
        series("ppl", "perplexity (worse ↑)", ppl),
        series("latency", "decode latency (relative)", lat),
      ],
      steps: DEPTHS.map((d, i) => ({
        t: i,
        action: `drop to ${d} layers`,
        detail:
          i === 0
            ? "full 48-layer stack"
            : `removing ${DEPTHS[i - 1] - d} blocks cuts latency ${lat[i - 1].value} → ${lat[i].value} and raises perplexity ${ppl[i - 1].value} → ${ppl[i].value}`,
      })),
    };
  },
};

export const TEACHER = [70, 13, 7, 3, 1.5, 0.5];

export const distillation: Kernel = {
  run: () => {
    // The student/teacher curve is the interesting shape: small students get most
    // of the quality for a fraction of the cost, but never reach the teacher.
    const quality = TEACHER.map((s) => ({
      label: `${s}B`,
      value: Math.round((30.4 + Math.log2(s) * 2.9 + wobble(s, 4) * 0.4) * 100) / 100,
    }));
    const cost = TEACHER.map((s) => ({ label: `${s}B`, value: Math.round(s / 70 * 100) }));
    return {
      series: [
        series("quality", "distilled quality (of teacher)", quality),
        series("cost", "inference cost (of teacher)", cost),
      ],
      steps: TEACHER.map((s, i) => ({
        t: i,
        action: `${s}B student`,
        detail:
          i === 0
            ? `70B teacher defines the ceiling at ${quality[0].value}`
            : `${s}B student recovers ${Math.round((quality[i].value / quality[0].value) * 100)}% of teacher quality at ${cost[i].value}% of the cost`,
      })),
    };
  },
};

export const RANKS = [64, 32, 16, 8, 4, 2];

export const peft: Kernel = {
  run: () => {
    // Trainable-parameter count collapses; quality barely moves, which is the
    // whole reason adapter tuning shipped.
    const params = RANKS.map((r) => ({ label: `r=${r}`, value: Math.round((0.5 + r * 0.62) * 10) / 10 }));
    const quality = RANKS.map((r) => ({
      label: `r=${r}`,
      value: Math.round((33.8 - (1 - Math.log2(64 / r) / 6) * 1.4 + wobble(r, 5) * 0.2) * 100) / 100,
    }));
    return {
      series: [
        series("params", "trainable params (M)", params),
        series("quality", "quality (of full finetune)", quality),
      ],
      steps: RANKS.map((r, i) => ({
        t: i,
        action: `rank ${r}`,
        detail:
          i === 0
            ? "full fine-tuning: every weight trainable"
            : `low-rank correction of rank ${r} trains ${params[i].value}M parameters for ${quality[i].value} of full-finetune quality`,
      })),
    };
  },
};

export const EXPERTS = [1, 4, 8, 16, 32, 64];

export const moe: Kernel = {
  run: () => {
    // Active parameters stay flat: more experts means more total parameters, not
    // more compute per token.
    const total = EXPERTS.map((e) => ({ label: `${e} experts`, value: Math.round(6.7 * e * 10) / 10 }));
    const active = EXPERTS.map((e) => ({ label: `${e} experts`, value: 6.7 }));
    const quality = EXPERTS.map((e) => ({
      label: `${e} experts`,
      value: Math.round((32.1 + Math.log2(e) * 1.9 + wobble(e, 6) * 0.3) * 100) / 100,
    }));
    return {
      series: [
        series("total", "total parameters (B)", total),
        series("active", "active parameters per token (B)", active),
        series("quality", "quality", quality),
      ],
      steps: EXPERTS.map((e, i) => ({
        t: i,
        action: `${e} expert${e > 1 ? "s" : ""}`,
        detail:
          i === 0
            ? "dense baseline: 6.7B active parameters"
            : `${total[i].value}B total but still ${active[i].value}B active per token; quality ${quality[i].value}`,
      })),
    };
  },
};

export const CACHE_LENGTHS = [512, 2048, 8192, 32768, 131072];

export const kvCache: Kernel = {
  run: () => {
    // Cache memory is linear in sequence length, which is why long context is
    // bounded by memory long before it is bounded by attention.
    const mem = CACHE_LENGTHS.map((L) => ({ label: `${L}`, value: Math.round((L * 0.5 * 32 * 2 * 2) / 1024) / 1024 }));
    return {
      series: [series("kv", "KV cache size (GiB per sequence)", mem)],
      steps: CACHE_LENGTHS.map((L, i) => ({
        t: i,
        action: `${L.toLocaleString()} tokens`,
        detail:
          i === 0
            ? "32 layers x 2 (K and V) x 8 KV heads x 128 dims, half precision"
            : `cache grows linearly to ${mem[i].value} GiB for one sequence at ${L.toLocaleString()} tokens`,
      })),
    };
  },
};

export const DRAFT_LENGTHS = [1, 2, 3, 4, 5, 6, 7];

export const speculative: Kernel = {
  run: () => {
    // Tokens per forward step, given a fixed acceptance rate. Speedup is
    // sublinear in draft length because acceptance falls as drafts get longer.
    const accept = 0.72;
    const speedup = DRAFT_LENGTHS.map((d) => ({
      label: `γ=${d}`,
      value:
        Math.round(
          ((1 - Math.pow(1 - accept, d + 1)) / (d * (1 - accept)) + wobble(d, 7) * 0.02) * 100,
        ) / 100,
    }));
    return {
      series: [series("speedup", "tokens per decode step (relative)", speedup)],
      steps: DRAFT_LENGTHS.map((d, i) => ({
        t: i,
        action: `draft ${d} token${d > 1 ? "s" : ""}`,
        detail:
          i === 0
            ? `no drafting: one token per verify pass at ${accept * 100}% acceptance`
            : `${speedup[i].value}x tokens per pass; output distribution unchanged because every accepted token was verified by the target model`,
      })),
    };
  },
};

export const CONTEXTS = [1, 4, 16, 64, 256];

export const retrieval: Kernel = {
  run: () => {
    // Retrieval adds context but not parameters, and the quality gain is
    // sublinear in retrieved chunks: the last chunk adds little.
    const quality = CONTEXTS.map((c) => ({
      label: `${c} chunks`,
      value: Math.round((31.2 + Math.log2(c + 1) * 3.4 + wobble(c, 8) * 0.5) * 100) / 100,
    }));
    const params = CONTEXTS.map((c) => ({ label: `${c} chunks`, value: 6.7 }));
    const latency = CONTEXTS.map((c) => ({
      label: `${c} chunks`,
      value: Math.round(c * 34 + wobble(c, 9) * 2),
    }));
    return {
      series: [
        series("quality", "quality", quality),
        series("params", "parameters (B)", params),
        series("latency", "time to first token (ms)", latency),
      ],
      steps: CONTEXTS.map((c, i) => ({
        t: i,
        action: `retrieve ${c} chunk${c > 1 ? "s" : ""}`,
        detail:
          i === 0
            ? "no retrieval: the model must already know the answer"
            : `+${(quality[i].value - quality[i - 1].value).toFixed(2)} quality for ${latency[i].value}ms, at unchanged ${params[i].value}B parameters`,
      })),
    };
  },
};

export const REASONING_KINDS = [
  "direct",
  "few-shot",
  "chain-of-thought",
  "self-consistency",
  "verifier",
];

export const reasoning: Kernel = {
  run: () => {
    // Each strategy buys accuracy with more sequential decoding.
    const quality = REASONING_KINDS.map((k, i) => ({
      label: k,
      value: Math.round((18.4 + i * 8.6 + wobble(i, 10) * 0.6) * 100) / 100,
    }));
    const passes = REASONING_KINDS.map((k, i) => ({ label: k, value: 1 + i * i }));
    return {
      series: [
        series("quality", "accuracy", quality),
        series("passes", "sampling passes", passes),
      ],
      steps: REASONING_KINDS.map((k, i) => ({
        t: i,
        action: k,
        detail:
          i === 0
            ? "single greedy decode: the cheapest strategy"
            : `${k} needs ${passes[i]} sampling passes for ${quality[i].value} accuracy`,
      })),
    };
  },
};