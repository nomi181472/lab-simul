"use client";

import { useMemo, useState } from "react";
import {
  Card, SectionTitle, Badge, CiteBadge, Quote, Unstated, truncate,
} from "@/labs/networking/ui";
import { PAPERS } from "@/labs/networking/data/records";
import { KB } from "@/labs/networking/data/kb";
import Link from "next/link";

const DOMAIN_LABEL: Record<string, string> = Object.fromEntries(
  KB.domains.map((d) => [d.id, d.label]),
);
const FAMILY_LABEL: Record<string, string> = Object.fromEntries(
  KB.solutions.map((s) => [s.familyId, s.label]),
);

const NOT = "not stated in the retrieved text";

export function CorpusView() {
  const [query, setQuery] = useState("");
  const [domain, setDomain] = useState<string | null>(null);
  const [depth, setDepth] = useState<"all" | "fulltext" | "abstract">("all");
  const [openId, setOpenId] = useState<string | null>(null);

  const allDomains = useMemo(() => {
    const counts = new Map<string, number>();
    for (const p of PAPERS)
      for (const d of p.domains) counts.set(d, (counts.get(d) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 14);
  }, []);

  const list = useMemo(() => {
    let out = [...PAPERS];
    if (domain) out = out.filter((p) => p.domains.includes(domain));
    if (depth === "fulltext") out = out.filter((p) => p.hasFullText);
    if (depth === "abstract") out = out.filter((p) => !p.hasFullText);
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      out = out.filter(
        (p) =>
          p.title.toLowerCase().includes(q) ||
          (p.venue || "").toLowerCase().includes(q) ||
          p.authors.some((a) => a.toLowerCase().includes(q)) ||
          p.domains.some((d) => (DOMAIN_LABEL[d] ?? d).toLowerCase().includes(q)),
      );
    }
    return out;
  }, [query, domain, depth]);

  return (
    <div className="space-y-6">
      <SectionTitle>Corpus explorer</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          {PAPERS.length} papers, descending citation count. Every row carries its quoted
          problem / solution / experiments / limitations with a source-depth badge, plus
          DOI and PDF links where they resolve.
        </p>
      </Card>

      <Card>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="search title, venue, author or domain"
          className="w-full rounded-md border border-zinc-800 bg-black/40 px-3 py-2 font-mono text-[11px] text-zinc-200 placeholder:text-zinc-600"
        />
        <div className="mt-3 flex flex-wrap items-center gap-1">
          <button
            onClick={() => setDomain(null)}
            className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${
              domain === null
                ? "border-sky-700/50 bg-sky-600/20 text-sky-300"
                : "border-zinc-700 bg-zinc-900 text-zinc-400"
            }`}
          >
            all domains
          </button>
          {allDomains.map(([d, n]) => (
            <button
              key={d}
              onClick={() => setDomain(domain === d ? null : d)}
              className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${
                domain === d
                  ? "border-sky-700/50 bg-sky-600/20 text-sky-300"
                  : "border-zinc-700 bg-zinc-900 text-zinc-400"
              }`}
            >
              {DOMAIN_LABEL[d] ?? d} ({n})
            </button>
          ))}
        </div>
        <div className="mt-2 flex gap-1">
          {(["all", "fulltext", "abstract"] as const).map((d) => (
            <button
              key={d}
              onClick={() => setDepth(d)}
              className={`rounded-md border px-2 py-0.5 font-mono text-[10px] ${
                depth === d
                  ? "border-sky-700/50 bg-sky-600/20 text-sky-300"
                  : "border-zinc-700 bg-zinc-900 text-zinc-400"
              }`}
            >
              {d === "all" ? "all sources" : d === "fulltext" ? "full-text only" : "abstract only"}
            </button>
          ))}
        </div>
        <p className="mt-2 font-mono text-[10px] text-zinc-600">
          {list.length} shown
        </p>
      </Card>

      <div className="space-y-2">
        {list.map((p) => {
          const open = openId === p.id;
          return (
            <Card key={p.id} className={open ? "border-sky-800/60" : ""}>
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <button
                    onClick={() => setOpenId(open ? null : p.id)}
                    className="text-left"
                  >
                    <span className="font-mono text-[10px] text-sky-400">{p.id}</span>
                    <span className="ml-2 text-[13px] text-zinc-100">{p.title}</span>
                  </button>
                  <p className="mt-0.5 font-mono text-[10px] text-zinc-500">
                    {p.year} · {truncate(p.venue || "venue not recorded", 50)} ·{" "}
                    {p.authors.slice(0, 3).join(", ")}
                    {p.authors.length > 3 ? " et al." : ""}
                  </p>
                </div>
                <div className="flex shrink-0 flex-col items-end gap-1">
                  <CiteBadge n={p.citations} />
                  <Badge tone={p.hasFullText ? "emerald" : "amber"}>
                    {p.source}
                  </Badge>
                </div>
              </div>

              <div className="mt-2 flex flex-wrap gap-1">
                {p.domains.map((d) => (
                  <Badge key={d}>{DOMAIN_LABEL[d] ?? d}</Badge>
                ))}
                {p.solutionFamilies.map((f) => (
                  <Badge key={f} tone="violet">
                    {FAMILY_LABEL[f] ?? f}
                  </Badge>
                ))}
              </div>

              {open && (
                <div className="mt-3 space-y-3 border-t border-zinc-800 pt-3">
                  <Field label="Problem" text={p.problem} />
                  <Field label="Proposed solution" text={p.solution} />
                  <Field label="Experiments" text={p.experiments} />
                  <Field label="Limitations" text={p.limitations} />
                  <Field label="Future directions" text={p.futureDirections} />
                  <Field label="Industry evidence" text={p.industryEvidence} />
                  {(p.datasets.length > 0 || p.metrics.length > 0 || p.algorithms.length > 0) && (
                    <div className="flex flex-wrap gap-1">
                      {[...p.datasets, ...p.metrics, ...p.algorithms].map((x, i) => (
                        <Badge key={`${x}-${i}`} tone="cyan">
                          {x}
                        </Badge>
                      ))}
                    </div>
                  )}
                  <div className="flex flex-wrap gap-3 font-mono text-[10px]">
                    {p.links.doi && (
                      <a
                        href={p.links.doi}
                        target="_blank"
                        rel="noreferrer"
                        className="text-sky-400 hover:text-sky-300"
                      >
                        DOI ↗
                      </a>
                    )}
                    {p.fileName && (
                      <Link
                        href={`/papers/${encodeURIComponent(p.fileName)}`}
                        target="_blank"
                        className="text-sky-400 hover:text-sky-300"
                      >
                        PDF ↗ ({p.fileName})
                      </Link>
                    )}
                    {!p.links.doi && !p.fileName && (
                      <span className="text-zinc-600">no resolvable link in harvest</span>
                    )}
                  </div>
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}

function Field({ label, text }: { label: string; text: string }) {
  if (!text || text === NOT) return <Unstated what={label.toLowerCase()} />;
  return (
    <div>
      <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">{label}</p>
      <Quote>{text}</Quote>
    </div>
  );
}
