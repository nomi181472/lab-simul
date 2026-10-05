"use client";

import { useMemo, useState } from "react";
import { PAPERS, PAPER_BY_ID } from "@/labs/neuroevolution/data/papers";
import { Card, SectionTitle, Badge, CiteBadge, NeuroPaperLink, DifficultyBadge, Quote, Unstated } from "@/labs/neuroevolution/ui";
import { ArchitectureDiagram, ModificationDiagram } from "@/labs/neuroevolution/architecture";
import { DATASET_BY_ID } from "@/labs/neuroevolution/data/datasets";
import { useNeuroLab } from "@/labs/neuroevolution/context";

/* ------------------------------ papers ------------------------------ */

function Section({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-600">{title}</div>
      <div className="text-[12px] leading-6 text-zinc-300">{body}</div>
    </div>
  );
}

export function PapersView() {
  const { selectedPaper, setSelectedPaper } = useNeuroLab();
  const [q, setQ] = useState("");
  const [tag, setTag] = useState<string>("all");
  const [diff, setDiff] = useState<string>("all");

  const tags = useMemo(() => {
    const m = new Map<string, number>();
    for (const p of PAPERS) for (const t of p.tags) m.set(t, (m.get(t) ?? 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 18);
  }, []);

  const list = useMemo(() => {
    let out = PAPERS;
    if (tag !== "all") out = out.filter((p) => p.tags.includes(tag));
    if (diff !== "all") out = out.filter((p) => p.difficulty === diff);
    const n = q.trim().toLowerCase();
    if (n)
      out = out.filter(
        (p) =>
          p.title.toLowerCase().includes(n) ||
          p.shortTitle.toLowerCase().includes(n) ||
          p.summary.toLowerCase().includes(n) ||
          (p.venue ?? "").toLowerCase().includes(n),
      );
    return out;
  }, [q, tag, diff]);

  const detail = selectedPaper ? PAPER_BY_ID[selectedPaper] : null;

  return (
    <div className="flex flex-col gap-5">
      <SectionTitle>Papers</SectionTitle>

      <div className="flex flex-wrap items-center gap-3">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="search title, summary, venue…"
          className="min-w-[220px] flex-1 rounded-md border border-zinc-800 bg-zinc-900/70 px-3 py-1.5 text-[12px] text-zinc-200 placeholder-zinc-600 focus:border-rose-800 focus:outline-none"
        />
        <div className="flex gap-2">
          {["all", "intro", "intermediate", "advanced"].map((d) => (
            <button
              key={d}
              onClick={() => setDiff(d)}
              className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${
                diff === d ? "border-rose-700/60 bg-rose-600/20 text-rose-300" : "border-zinc-800 text-zinc-600"
              }`}
            >
              {d}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5">
        <button
          onClick={() => setTag("all")}
          className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${
            tag === "all" ? "border-zinc-600 bg-zinc-800 text-zinc-200" : "border-zinc-800 text-zinc-600"
          }`}
        >
          all tags
        </button>
        {tags.map(([t, n]) => (
          <button
            key={t}
            onClick={() => setTag(tag === t ? "all" : t)}
            className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${
              tag === t ? "border-zinc-600 bg-zinc-800 text-zinc-200" : "border-zinc-800 text-zinc-600"
            }`}
          >
            {t} ({n})
          </button>
        ))}
      </div>

      <div className="font-mono text-[10px] text-zinc-600">{list.length} papers</div>

      <div className="grid gap-3 lg:grid-cols-2">
        {list.map((p) => (
          <Card
            key={p.id}
            className={`flex cursor-pointer flex-col gap-2 transition-colors ${
              selectedPaper === p.id ? "border-rose-700/60" : "hover:border-zinc-700"
            }`}
          >
            <div className="flex flex-wrap items-center gap-2">
              <NeuroPaperLink id={p.id} link />
              <CiteBadge n={p.citations} />
              <Badge tone="zinc">{p.year}</Badge>
              <DifficultyBadge d={p.difficulty} />
            </div>
            <button
              onClick={() => setSelectedPaper(selectedPaper === p.id ? null : p.id)}
              className="text-left text-[13px] font-medium leading-5 text-zinc-100"
            >
              {p.title}
            </button>
            <div className="line-clamp-2 text-[11px] leading-5 text-zinc-400">{p.summary}</div>
            {p.venue ? <div className="font-mono text-[10px] text-zinc-600">{p.venue}</div> : null}
          </Card>
        ))}
      </div>

      {detail ? <PaperDetail id={detail.id} /> : null}
    </div>
  );
}

export function PaperDetail({ id }: { id: string }) {
  const p = PAPER_BY_ID[id];
  if (!p) return null;
  const e = p.method.evolution;

  return (
    <Card tone="accent" className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-base font-semibold text-zinc-100">{p.title}</h3>
        <CiteBadge n={p.citations} />
      </div>
      <div className="font-mono text-[10px] text-zinc-500">
        {p.authors.slice(0, 6).join(", ")}
        {p.authors.length > 6 ? " et al." : ""}
        {p.year} · {p.venue || "venue not recorded"}
        {p.doi ? ` · doi:${p.doi}` : ""}
      </div>

      <Section title="summary" body={p.summary} />
      {p.problem && p.problem !== p.summary ? <Section title="problem" body={p.problem} /> : null}

      <div className="flex flex-col gap-2">
        <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-600">
          architecture
        </div>
        <ArchitectureDiagram architecture={p.architecture} />
        {p.architecture.genotypeToPhenotype ? (
          <Section title="genotype → phenotype" body={p.architecture.genotypeToPhenotype} />
        ) : (
          <Unstated what="genotype-to-phenotype mapping" />
        )}
      </div>

      <div className="flex flex-col gap-2">
        <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-600">
          modification
        </div>
        {e.changes.length > 0 ? (
          <ModificationDiagram changes={e.changes} />
        ) : (
          <Unstated what="locus-level modification" />
        )}
        <div className="flex flex-wrap gap-1.5">
          {e.mutation.map((m) => (
            <Badge key={m} tone="rose">
              mutation: {m}
            </Badge>
          ))}
          {e.crossover.map((c) => (
            <Badge key={c} tone="violet">
              crossover: {c}
            </Badge>
          ))}
          {e.selection.map((s) => (
            <Badge key={s} tone="emerald">
              selection: {s}
            </Badge>
          ))}
        </div>
        {e.fitness ? <Section title="fitness" body={e.fitness} /> : <Unstated what="fitness" />}
      </div>

      {p.contribution.length > 0 ? (
        <div className="flex flex-col gap-1">
          <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-600">
            contributions
          </div>
          <ul className="flex flex-col gap-1">
            {p.contribution.map((c, i) => (
              <li key={i} className="text-[11px] leading-5 text-zinc-400">
                · {c}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {p.results.length > 0 ? (
        <div className="flex flex-col gap-1">
          <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-600">
            results (quoted, numbers only where printed)
          </div>
          {p.results.map((r, i) => (
            <Quote key={i}>{r}</Quote>
          ))}
        </div>
      ) : (
        <Unstated what="results" />
      )}

      {p.datasets.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {p.datasets.map((d) => (
            <Badge key={d} tone="cyan">
              {DATASET_BY_ID[d]?.name ?? d}
            </Badge>
          ))}
        </div>
      ) : null}

      {p.limitations.evident.length > 0 ? (
        <div className="flex flex-col gap-1">
          <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-600">
            evident from the text (lab, not author claims)
          </div>
          {p.limitations.evident.map((l, i) => (
            <div key={i} className="text-[11px] leading-5 text-amber-300/70">
              · {l}
            </div>
          ))}
        </div>
      ) : null}

      <a
        href={`/papers/${p.fileName}`}
        target="_blank"
        rel="noreferrer"
        className="font-mono text-[10px] text-rose-400 underline"
      >
        open the source PDF ({p.fileName})
      </a>
    </Card>
  );
}
