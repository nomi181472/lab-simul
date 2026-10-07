"use client";

import { useMemo, useState } from "react";
import {
  Card,
  SectionTitle,
  Badge,
  Quote,
  Unstated,
  CiteBadge,
  TfPaperLink,
  ArchChip,
  ChangeChip,
} from "@/labs/llms/ui";
import { PAPERS } from "@/labs/llms/data/papers";
import { MANIFEST } from "@/labs/llms/data/manifest";

export function PapersView() {
  const [query, setQuery] = useState("");
  const [operator, setOperator] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);

  const list = useMemo(() => {
    // The corpus is already emitted in descending citation order; filtering
    // preserves it, which is the ordering the manifest promises.
    let out = [...PAPERS];
    if (operator) out = out.filter((p) => p.changes.some((c) => c.operator === operator));
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      out = out.filter(
        (p) =>
          p.title.toLowerCase().includes(q) ||
          p.venue.toLowerCase().includes(q) ||
          p.authors.some((a) => a.toLowerCase().includes(q)),
      );
    }
    return out;
  }, [query, operator]);

  const allOps = useMemo(() => {
    const s = new Set<string>();
    for (const p of PAPERS) for (const c of p.changes) s.add(c.operator);
    return [...s].sort();
  }, []);

  return (
    <div className="space-y-6">
      <SectionTitle>Papers</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          {MANIFEST.total} papers, ordered by descending citation count. Each row carries
          the structural elements and transformations the paper names, with the sentence
          each came from one click away.
        </p>
      </Card>

      <Card>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="search title, venue or author"
          className="w-full rounded-md border border-zinc-800 bg-black/40 px-3 py-2 font-mono text-[11px] text-zinc-200 placeholder:text-zinc-600"
        />
        <div className="mt-3 flex flex-wrap gap-1">
          <button
            onClick={() => setOperator(null)}
            className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${
              operator
                ? "border-zinc-700 bg-zinc-900 text-zinc-400"
                : "border-cyan-700/50 bg-cyan-600/20 text-cyan-300"
            }`}
          >
            all
          </button>
          {allOps.map((op) => (
            <button
              key={op}
              onClick={() => setOperator(operator === op ? null : op)}
              className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${
                operator === op
                  ? "border-cyan-600 bg-cyan-950 text-cyan-200"
                  : "border-zinc-800 bg-zinc-900/60 text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {op}
            </button>
          ))}
        </div>
        <p className="mt-3 font-mono text-[10px] text-zinc-600">
          {list.length} of {PAPERS.length} papers
        </p>
      </Card>

      <div className="space-y-2">
        {list.map((p) => {
          const open = openId === p.id;
          return (
            <Card key={p.id}>
              <div className="flex flex-wrap items-start justify-between gap-2">
                <button
                  onClick={() => setOpenId(open ? null : p.id)}
                  className="min-w-0 flex-1 text-left"
                >
                  <div className="flex flex-wrap items-center gap-1.5">
                    <Badge tone="cyan">{p.id}</Badge>
                    <CiteBadge n={p.citations} />
                    {p.year && <Badge>{p.year}</Badge>}
                    {p.venue && <Badge>{p.venue.slice(0, 34)}</Badge>}
                    {p.arxiv && <Badge tone="violet">arXiv:{p.arxiv}</Badge>}
                    {p.pages > 0 && <Badge>{p.pages}p</Badge>}
                  </div>
                  <p className="mt-1.5 text-[13px] leading-5 text-zinc-100">{p.title}</p>
                  {p.authors.length > 0 && (
                    <p className="mt-1 truncate text-[11px] text-zinc-500">
                      {p.authors.slice(0, 6).join(", ")}
                      {p.authors.length > 6 ? " et al." : ""}
                    </p>
                  )}
                </button>
                <div className="flex shrink-0 items-center gap-1.5">
                  <TfPaperLink id={p.id} link short />
                  <button
                    onClick={() => setOpenId(open ? null : p.id)}
                    className="rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1 text-[11px] text-zinc-400"
                  >
                    {open ? "less" : "evidence"}
                  </button>
                </div>
              </div>

              <div className="mt-2 flex flex-wrap gap-1">
                {p.architecture.layers.slice(0, 6).map((l) => (
                  <ArchChip key={l.kind} kind={l.kind} />
                ))}
                {p.architecture.layers.length === 0 && <Unstated what="architecture" />}
              </div>
              <div className="mt-1.5 flex flex-wrap gap-1">
                {p.changes.map((c) => (
                  <ChangeChip key={c.operator} operator={c.operator} />
                ))}
              </div>

              {open && (
                <div className="mt-3 grid gap-4 border-t border-zinc-800 pt-3 lg:grid-cols-2">
                  <div>
                    <p className="mb-1.5 text-[12px] font-medium text-zinc-200">
                      Architecture evidence
                    </p>
                    <div className="space-y-2">
                      {p.architecture.layers.length === 0 ? (
                        <Unstated what="architecture" />
                      ) : (
                        p.architecture.layers.map((l) => (
                          <div key={l.kind}>
                            <p className="font-mono text-[10px] text-zinc-500">{l.label}</p>
                            <Quote>{l.quote}</Quote>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                  <div>
                    <p className="mb-1.5 text-[12px] font-medium text-zinc-200">
                      Modification evidence
                    </p>
                    <div className="space-y-2">
                      {p.changes.length === 0 ? (
                        <Unstated what="modification" />
                      ) : (
                        p.changes.map((c) => (
                          <div key={c.operator}>
                            <p className="font-mono text-[10px] text-zinc-500">{c.label}</p>
                            <Quote>{c.quote}</Quote>
                          </div>
                        ))
                      )}
                    </div>
                    {p.objectives.length > 0 && (
                      <>
                        <p className="mb-1.5 mt-3 text-[12px] font-medium text-zinc-200">
                          Objectives named
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {p.objectives.map((o) => (
                            <span
                              key={o.kind}
                              className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-300"
                              title={o.quote}
                            >
                              {o.label}
                            </span>
                          ))}
                        </div>
                      </>
                    )}
                    {p.results.length > 0 && (
                      <>
                        <p className="mb-1.5 mt-3 text-[12px] font-medium text-zinc-200">
                          Benchmarks named
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {p.results.map((r) => (
                            <span
                              key={r.dataset}
                              className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-300"
                              title={r.quote}
                            >
                              {r.label}
                            </span>
                          ))}
                        </div>
                      </>
                    )}
                    <p className="mt-3 font-mono text-[10px] text-zinc-600">
                      {p.textChars.toLocaleString()} chars extracted ·{" "}
                      {p.referenceCount} reference lines
                    </p>
                  </div>
                </div>
              )}
            </Card>
          );
        })}
      </div>

      {list.length === 0 && (
        <p className="font-mono text-[11px] text-zinc-600">no paper matches that filter</p>
      )}
    </div>
  );
}