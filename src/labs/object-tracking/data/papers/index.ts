import type { PaperRecord } from "../types";
import { BATCH_01 } from "./batch-01";
import { BATCH_02 } from "./batch-02";
import { BATCH_03 } from "./batch-03";
import { BATCH_04 } from "./batch-04";
import { BATCH_05 } from "./batch-05";
import { BATCH_06 } from "./batch-06";
import { BATCH_07 } from "./batch-07";
import { BATCH_08 } from "./batch-08";
import { BATCH_09 } from "./batch-09";
import { BATCH_10 } from "./batch-10";
import { BATCH_11 } from "./batch-11";
import { BATCH_12 } from "./batch-12";
import { BATCH_13 } from "./batch-13";
import { BATCH_14 } from "./batch-14";
import { BATCH_15 } from "./batch-15";
import { BATCH_16 } from "./batch-16";
import { BATCH_17 } from "./batch-17";

/* Aggregated corpus: complete — T001..T159 (17 batches). */

export const BATCHES: PaperRecord[][] = [
  BATCH_01,
  BATCH_02,
  BATCH_03,
  BATCH_04,
  BATCH_05,
  BATCH_06,
  BATCH_07,
  BATCH_08,
  BATCH_09,
  BATCH_10,
  BATCH_11,
  BATCH_12,
  BATCH_13,
  BATCH_14,
  BATCH_15,
  BATCH_16,
  BATCH_17,
];

export const PAPERS: PaperRecord[] = BATCHES.flat().sort((a, b) =>
  a.id.localeCompare(b.id),
);

export const PAPER_BY_ID: Record<string, PaperRecord> = Object.fromEntries(
  PAPERS.map((p) => [p.id, p]),
);

export const PAPER_COUNT = PAPERS.length;

export function papersByYear(): Record<number, PaperRecord[]> {
  const out: Record<number, PaperRecord[]> = {};
  for (const p of PAPERS) {
    (out[p.year] ??= []).push(p);
  }
  return out;
}

export function successorsOf(id: string): PaperRecord[] {
  return PAPERS.filter((p) => p.relations.some((r) => r.to === id)).sort(
    (a, b) => a.year - b.year || a.id.localeCompare(b.id),
  );
}

export function predecessorsOf(p: PaperRecord): PaperRecord[] {
  return p.relations
    .map((r) => PAPER_BY_ID[r.to])
    .filter(Boolean)
    .sort((a, b) => a.year - b.year || a.id.localeCompare(b.id));
}
