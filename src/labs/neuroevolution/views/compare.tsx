"use client";

import { useState } from "react";
import { PAPERS, PAPER_BY_ID } from "@/labs/neuroevolution/data/papers";
import { Card, SectionTitle } from "@/labs/neuroevolution/ui";
import { ArchitectureDiagram } from "@/labs/neuroevolution/architecture";

/* Side-by-side comparison of two papers' evolved architectures and operators. */

/* ------------------------------ compare ------------------------------ */

function Row({ label, x, y }: { label: string; x: React.ReactNode; y: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[130px,1fr,1fr] gap-2 border-t border-zinc-800 py-2 text-[11px]">
      <span className="font-mono text-[10px] text-zinc-600">{label}</span>
      <span className="text-zinc-300">{x ?? "\u2014"}</span>
      <span className="text-zinc-300">{y ?? "\u2014"}</span>
    </div>
  );
}

export function CompareView() {
  const [a, setA] = useState(PAPERS[0]?.id ?? "");
  const [b, setB] = useState(PAPERS[1]?.id ?? "");

  const pa = PAPER_BY_ID[a];
  const pb = PAPER_BY_ID[b];

  return (
    <div className="flex flex-col gap-5">
      <SectionTitle>Compare</SectionTitle>
      <div className="flex flex-wrap gap-3">
        <select
          value={a}
          onChange={(e) => setA(e.target.value)}
          className="max-w-md rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-[11px] text-zinc-300"
        >
          {PAPERS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.id} · {p.shortTitle}
            </option>
          ))}
        </select>
        <select
          value={b}
          onChange={(e) => setB(e.target.value)}
          className="max-w-md rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-[11px] text-zinc-300"
        >
          {PAPERS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.id} · {p.shortTitle}
            </option>
          ))}
        </select>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {[pa, pb].map((p) =>
          p ? (
            <Card key={p.id} tone="accent">
              <ArchitectureDiagram architecture={p.architecture} />
            </Card>
          ) : null,
        )}
      </div>

      <Card>
        <div className="grid grid-cols-[130px,1fr,1fr] gap-2 pb-2 font-mono text-[10px] text-zinc-500">
          <span />
          <span>{pa?.id ?? "—"}</span>
          <span>{pb?.id ?? "—"}</span>
        </div>
        <Row label="year" x={pa?.year} y={pb?.year} />
        <Row label="citations" x={pa?.citations} y={pb?.citations} />
        <Row label="venue" x={pa?.venue || "—"} y={pb?.venue || "—"} />
        <Row label="phenotype" x={pa?.architecture.phenotype} y={pb?.architecture.phenotype} />
        <Row
          label="layers found"
          x={pa?.architecture.layers.length || 0}
          y={pb?.architecture.layers.length || 0}
        />
        <Row label="family" x={pa?.method.evolution.family} y={pb?.method.evolution.family} />
        <Row label="encoding" x={pa?.method.evolution.encoding} y={pb?.method.evolution.encoding} />
        <Row
          label="mutation"
          x={pa?.method.evolution.mutation.join(", ") || "—"}
          y={pb?.method.evolution.mutation.join(", ") || "—"}
        />
        <Row
          label="crossover"
          x={pa?.method.evolution.crossover.join(", ") || "—"}
          y={pb?.method.evolution.crossover.join(", ") || "—"}
        />
        <Row
          label="selection"
          x={pa?.method.evolution.selection.join(", ") || "—"}
          y={pb?.method.evolution.selection.join(", ") || "—"}
        />
        <Row
          label="loci changed"
          x={pa?.method.evolution.changes.map((c) => c.locus).join(", ") || "—"}
          y={pb?.method.evolution.changes.map((c) => c.locus).join(", ") || "—"}
        />
        <Row label="metrics" x={pa?.metrics.join(", ") || "—"} y={pb?.metrics.join(", ") || "—"} />
        <Row label="datasets" x={pa?.datasets.join(", ") || "—"} y={pb?.datasets.join(", ") || "—"} />
      </Card>
    </div>
  );
}
