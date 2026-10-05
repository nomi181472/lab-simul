"use client";

import { useMemo, useState } from "react";
import { PAPERS } from "@/labs/wifi-sensing/data/papers";
import { MANIFEST } from "@/labs/wifi-sensing/data/manifest";
import { TASK_LABEL, MODALITY_LABEL } from "@/labs/wifi-sensing/data/types";
import { Badge, Card, CiteBadge, DifficultyBadge, Formula, SectionTitle, TaskBadge } from "@/labs/wifi-sensing/ui";
import { CONCEPT_BY_ID } from "@/labs/wifi-sensing/data/concepts";
import type { PaperRecord } from "@/labs/wifi-sensing/data/types";

type SortKey = "citations" | "year" | "title";

function PaperDetail({ p }: { p: PaperRecord }) {
  return (
    <div className="space-y-4">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="cyan">{p.id}</Badge>
          <CiteBadge n={p.citations} />
          <TaskBadge t={TASK_LABEL[p.task] ?? p.task} />
          <Badge tone="zinc">{MODALITY_LABEL[p.modality] ?? p.modality}</Badge>
          {p.bandwidth !== "unknown" ? <Badge tone="zinc">{p.bandwidth}</Badge> : null}
          <DifficultyBadge d={p.difficulty} />
          {p.groundTruth ? <Badge tone="violet">GT: {p.groundTruth}</Badge> : null}
        </div>
        <h3 className="mt-2 text-base font-semibold leading-snug text-zinc-100">{p.title}</h3>
        <p className="mt-1 font-mono text-[10px] text-zinc-500">
          {p.authors.join(", ")} · {p.venue || "arXiv"} · {p.year} · {p.pages}p ·{" "}
          <a
            href={`/papers/${encodeURIComponent(p.fileName)}`}
            target="_blank"
            className="text-cyan-400 hover:text-cyan-300"
          >
            open PDF ({p.fileName})
          </a>
        </p>
      </div>

      {p.summary ? <p className="text-[13px] leading-6 text-zinc-300">{p.summary}</p> : null}

      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-lg border border-zinc-800 bg-black/20 p-3">
          <div className="font-mono text-[9px] uppercase tracking-wide text-amber-400">
            signal chain (detected in this paper&apos;s text)
          </div>
          <dl className="mt-2 space-y-1.5 text-[11px] leading-5">
            {[
              ["hardware", p.chain.hardware],
              ["captured", p.chain.captured],
              ["preprocessing", p.chain.preprocessing.join(", ")],
              ["features", p.chain.features.join(", ")],
              ["model", p.chain.model],
              ["deployment", p.chain.deployment],
            ].map(([k, v]) => (
              <div key={k} className="flex gap-2">
                <dt className="w-24 shrink-0 font-mono text-[10px] text-zinc-500">{k}</dt>
                <dd className="min-w-0 flex-1 text-zinc-400">{v}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="space-y-3">
          {p.problem ? (
            <div>
              <div className="font-mono text-[9px] uppercase tracking-wide text-rose-400">problem</div>
              <p className="mt-0.5 text-[11px] leading-5 text-zinc-400">{p.problem}</p>
            </div>
          ) : null}
          {p.researchGap ? (
            <div>
              <div className="font-mono text-[9px] uppercase tracking-wide text-violet-400">
                research gap
              </div>
              <p className="mt-0.5 text-[11px] leading-5 text-zinc-400">{p.researchGap}</p>
            </div>
          ) : null}
          {p.contribution.length ? (
            <div>
              <div className="font-mono text-[9px] uppercase tracking-wide text-emerald-400">
                contributions (paper&apos;s own sentences)
              </div>
              <ul className="mt-0.5 space-y-1">
                {p.contribution.map((c, i) => (
                  <li key={i} className="text-[11px] leading-5 text-zinc-400">
                    · {c}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      </div>

      {p.equationIds.length ? (
        <div>
          <div className="font-mono text-[9px] uppercase tracking-wide text-cyan-400">
            standard equations this paper uses
          </div>
          <div className="mt-1 space-y-1">
            {p.equationIds.slice(0, 4).map((id) => {
              const c = CONCEPT_BY_ID[id];
              if (!c?.formula) return null;
              return (
                <div key={id}>
                  <Formula>{c.formula}</Formula>
                  <p className="mt-0.5 text-[10px] text-zinc-500">{c.name}</p>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}

      {p.results.length ? (
        <div>
          <div className="font-mono text-[9px] uppercase tracking-wide text-emerald-400">
            reported outcomes (quoted)
          </div>
          <ul className="mt-0.5 space-y-1">
            {p.results.slice(0, 6).map((r, i) => (
              <li key={i} className="text-[11px] leading-5 text-zinc-400">
                · {r}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {p.limitations.authorStated.length || p.limitations.evident.length ? (
        <div className="grid gap-3 md:grid-cols-2">
          {p.limitations.authorStated.length ? (
            <div className="rounded-lg border border-amber-900/40 bg-amber-950/10 p-3">
              <div className="font-mono text-[9px] uppercase tracking-wide text-amber-400">
                limitations the authors state
              </div>
              <ul className="mt-1 space-y-1">
                {p.limitations.authorStated.map((l, i) => (
                  <li key={i} className="text-[11px] leading-5 text-zinc-400">
                    · {l}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {p.limitations.evident.length ? (
            <div className="rounded-lg border border-zinc-800 bg-black/20 p-3">
              <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                visible from the methodology (lab synthesis)
              </div>
              <ul className="mt-1 space-y-1">
                {p.limitations.evident.map((l, i) => (
                  <li key={i} className="text-[11px] leading-5 text-zinc-400">
                    · {l}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}

      {p.relations.length ? (
        <div>
          <div className="font-mono text-[9px] uppercase tracking-wide text-cyan-400">
            cites in this corpus ({p.relations.length})
          </div>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {p.relations.map((r) => (
              <span
                key={r.to}
                title={`${r.type}${r.note ? ` — ${r.note}` : ""}`}
                className="inline-flex items-center gap-1 rounded-md border border-zinc-800 bg-zinc-900/60 px-1.5 py-0.5 font-mono text-[10px]"
              >
                <span className="text-cyan-400">{r.to}</span>
                <span className="text-zinc-500">{r.type}</span>
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function PapersView() {
  const [q, setQ] = useState("");
  const [task, setTask] = useState("all");
  const [modality, setModality] = useState("all");
  const [sort, setSort] = useState<SortKey>("citations");
  const [open, setOpen] = useState<string | null>(PAPERS[0]?.id ?? null);

  const tasks = useMemo(() => [...new Set(MANIFEST.map((m) => m.task))].sort(), []);
  const modalities = useMemo(() => [...new Set(MANIFEST.map((m) => m.modality))].sort(), []);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const list = PAPERS.filter((p) => {
      if (task !== "all" && p.task !== task) return false;
      if (modality !== "all" && p.modality !== modality) return false;
      if (!needle) return true;
      return (
        p.title.toLowerCase().includes(needle) ||
        p.summary.toLowerCase().includes(needle) ||
        p.tags.some((t) => t.includes(needle)) ||
        p.id.toLowerCase() === needle
      );
    });
    const dir = sort === "title" ? 1 : -1;
    return [...list].sort((a, b) =>
      sort === "title"
        ? a.title.localeCompare(b.title) * dir
        : (b[sort] - a[sort]) * dir,
    );
  }, [q, task, modality, sort]);

  const selected = PAPERS.find((p) => p.id === open) ?? null;

  return (
    <div className="space-y-5">
      <SectionTitle>Paper Explorer</SectionTitle>
      <p className="max-w-3xl text-[12px] leading-6 text-zinc-500">
        {MANIFEST.length} papers in citation order. Each record is built by extracting the
        paper&apos;s own sentences and detecting its own pipeline — the fields are evidence, not
        commentary. {PAPERS.length} records are generated.
      </p>

      <Card>
        <div className="flex flex-wrap items-center gap-3">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="search title, summary, tag or id…"
            className="min-w-48 flex-1 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-[12px] text-zinc-200 placeholder:text-zinc-600"
          />
          <select
            value={task}
            onChange={(e) => setTask(e.target.value)}
            className="rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-[11px] text-zinc-300"
          >
            <option value="all">all tasks</option>
            {tasks.map((t) => (
              <option key={t} value={t}>
                {TASK_LABEL[t as keyof typeof TASK_LABEL] ?? t}
              </option>
            ))}
          </select>
          <select
            value={modality}
            onChange={(e) => setModality(e.target.value)}
            className="rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-[11px] text-zinc-300"
          >
            <option value="all">all modalities</option>
            {modalities.map((m) => (
              <option key={m} value={m}>
                {MODALITY_LABEL[m as keyof typeof MODALITY_LABEL] ?? m}
              </option>
            ))}
          </select>
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as SortKey)}
            className="rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-[11px] text-zinc-300"
          >
            <option value="citations">by citations</option>
            <option value="year">by year</option>
            <option value="title">by title</option>
          </select>
          <span className="font-mono text-[10px] text-zinc-500">
            {filtered.length} shown
          </span>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div className="max-h-[72vh] space-y-1.5 overflow-y-auto pr-1">
          {filtered.slice(0, 400).map((p) => (
            <button
              key={p.id}
              onClick={() => setOpen(p.id)}
              className={`block w-full rounded-lg border p-2.5 text-left transition-colors ${
                open === p.id
                  ? "border-cyan-700/60 bg-cyan-950/20"
                  : "border-zinc-800 bg-zinc-900/40 hover:border-zinc-700"
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="rounded bg-cyan-950 px-1 py-0.5 font-mono text-[10px] text-cyan-300">
                  {p.id}
                </span>
                <span className="font-mono text-[10px] text-zinc-500">
                  {p.citations}c · {p.year}
                </span>
                <span className="ml-auto font-mono text-[10px] text-zinc-600">
                  {TASK_LABEL[p.task] ?? p.task}
                </span>
              </div>
              <div className="mt-1 line-clamp-2 text-[11px] leading-4 text-zinc-300">{p.title}</div>
            </button>
          ))}
          {filtered.length === 0 ? (
            <p className="px-2 py-6 text-center text-[12px] text-zinc-600">no papers match</p>
          ) : null}
        </div>

        <div className="max-h-[72vh] overflow-y-auto">
          {selected ? (
            <PaperDetail p={selected} />
          ) : (
            <p className="py-10 text-center text-[12px] text-zinc-600">select a paper</p>
          )}
        </div>
      </div>
    </div>
  );
}