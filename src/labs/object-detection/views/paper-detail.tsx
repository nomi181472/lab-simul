"use client";

import { useState } from "react";
import { Badge, Formula } from "@/labs/object-detection/ui";
import { HARVEST_RECORDS } from "@/labs/object-detection/data/harvest-records";
import type { Paper, QuotedText } from "@/labs/object-detection/data/types";

function Label({ children, tone = "text-zinc-500" }: { children: React.ReactNode; tone?: string }) {
  return (
    <div className={`font-mono text-[10px] uppercase tracking-wide ${tone}`}>{children}</div>
  );
}

function QuoteList({ quotes }: { quotes: QuotedText[] }) {
  return (
    <ul className="mt-1.5 space-y-1.5">
      {quotes.map((e, i) => (
        <li key={i} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5">
          <p className="text-[11px] leading-4 text-zinc-400">
            <span className="text-zinc-600">“</span>
            {e.quote}
            <span className="text-zinc-600">”</span>
          </p>
          {e.location && (
            <p className="mt-1 font-mono text-[10px] text-zinc-600">{e.location}</p>
          )}
        </li>
      ))}
    </ul>
  );
}

const EVIDENCE_SECTIONS = [
  ["problem", "problem evidence"],
  ["background", "background evidence"],
  ["solution", "solution evidence"],
  ["experiments", "experiment evidence"],
  ["limitations", "limitation evidence"],
  ["industry", "industry evidence"],
] as const;

export default function PaperDetail({ p }: { p: Paper }) {
  const [showDirections, setShowDirections] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);
  const rec = HARVEST_RECORDS[p.id];

  return (
    <div className="mt-3 space-y-4 border-t border-zinc-800 pt-3">
      <div>
        <h3 className="text-[13px] font-semibold leading-5 text-zinc-100">{p.title}</h3>
        <p className="mt-1 font-mono text-[10px] text-zinc-500">
          {p.authors.slice(0, 6).join(", ")}
          {p.authors.length > 6 ? " et al." : ""}
        </p>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
          <Label>problem summary</Label>
          <p className="mt-1 text-[12px] leading-5 text-zinc-400">{p.problemSummary}</p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
          <Label>method summary</Label>
          <p className="mt-1 text-[12px] leading-5 text-zinc-300">{p.summary}</p>
        </div>
      </div>

      {rec && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
          <Label>background</Label>
          <p className="mt-1 text-[12px] leading-5 text-zinc-400">{rec.background.summary}</p>
          {rec.background.relatedWork.length > 0 && (
            <div className="mt-2 space-y-1.5">
              <Label tone="text-zinc-600">related work cited</Label>
              {rec.background.relatedWork.map((r, i) => (
                <div key={i} className="text-[11px] leading-4">
                  <span className="font-medium text-zinc-300">{r.name}</span>
                  <span className="text-zinc-500"> — {r.relation}</span>
                  {r.quote && (
                    <p className="mt-0.5 text-zinc-600">
                      <span className="text-zinc-700">“</span>
                      {r.quote}
                      <span className="text-zinc-700">”</span>
                      {r.location && <span className="ml-1 font-mono text-[10px]">{r.location}</span>}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <Label tone="text-emerald-400">contributions ({(rec?.solution.components.length ?? p.contributions.length)})</Label>
          <ul className="mt-1 list-disc space-y-1 pl-4 text-[12px] leading-5 text-zinc-400">
            {(rec ? [rec.solution.approach, ...rec.solution.components] : p.contributions).map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
        <div>
          <Label tone="text-amber-400">limitations ({rec?.limitations.items.length ?? p.limitations.length})</Label>
          <ul className="mt-1 list-disc space-y-1 pl-4 text-[12px] leading-5 text-amber-200/80">
            {(rec?.limitations.items ?? p.limitations).map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </div>
      </div>

      {rec && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
          <Label>technical</Label>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {rec.technical.models.map((m) => (
              <Badge key={`mo-${m}`} tone="violet">{m}</Badge>
            ))}
            {rec.technical.backbones.map((b) => (
              <Badge key={`b-${b}`} tone="sky">{b}</Badge>
            ))}
            {rec.technical.algorithms.map((a) => (
              <Badge key={`a-${a}`} tone="zinc">{a}</Badge>
            ))}
          </div>
          {rec.technical.equations.length > 0 && (
            <div className="mt-2 space-y-2">
              <Label tone="text-amber-400">equations ({rec.technical.equations.length})</Label>
              {rec.technical.equations.map((e, i) => (
                <div key={i}>
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <span className="font-mono text-[10px] text-zinc-400">{e.name ?? `eq ${i + 1}`}</span>
                    <span className="font-mono text-[10px] text-zinc-600">{e.location}</span>
                  </div>
                  <Formula>{e.expr}</Formula>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {rec && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
          <Label>experiments</Label>
          <p className="mt-1 text-[12px] leading-5 text-zinc-400">{rec.experiments.summary}</p>
          {rec.experiments.baselines.length > 0 && (
            <div className="mt-2">
              <Label tone="text-zinc-600">baselines ({rec.experiments.baselines.length})</Label>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {rec.experiments.baselines.map((b, i) => (
                  <Badge key={i} tone="zinc">{b}</Badge>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {(p.datasets.length > 0 || p.metrics.length > 0) && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
          <Label>datasets / metrics</Label>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {p.datasets.map((d) => (
              <Badge key={`d-${d}`} tone="sky">{d}</Badge>
            ))}
            {p.metrics.map((m) => (
              <Badge key={`m-${m}`} tone="zinc">{m}</Badge>
            ))}
          </div>
        </div>
      )}

      {p.benchmarks.length > 0 && (
        <div>
          <Label>reported benchmarks ({p.benchmarks.length})</Label>
          <div className="mt-1 overflow-x-auto rounded-lg border border-zinc-800">
            <table className="w-full border-collapse text-[11px]">
              <thead>
                <tr className="bg-zinc-900/80 font-mono text-[10px] uppercase text-zinc-500">
                  <th className="p-1.5 text-left">dataset</th>
                  <th className="p-1.5 text-left">metric</th>
                  <th className="p-1.5 text-right">value</th>
                  <th className="p-1.5 text-left">split</th>
                  <th className="p-1.5 text-left">location</th>
                </tr>
              </thead>
              <tbody>
                {p.benchmarks.map((b, i) => (
                  <tr key={i} className="border-t border-zinc-800/70">
                    <td className="p-1.5 text-zinc-300">{b.dataset}</td>
                    <td className="p-1.5 font-mono text-zinc-400">{b.metric}</td>
                    <td className="p-1.5 text-right font-mono text-emerald-300">{b.value}</td>
                    <td className="p-1.5 text-zinc-500">{b.split ?? "—"}</td>
                    <td className="p-1.5 font-mono text-zinc-500">{b.location ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div>
        <div className="flex flex-wrap items-center gap-2">
          <Label tone="text-violet-400">
            future directions ({p.futureDirections.length})
          </Label>
          {p.futureDirections.length > 0 && (
            <button
              onClick={() => setShowDirections((v) => !v)}
              className="rounded-md border border-zinc-700 px-1.5 py-0.5 font-mono text-[10px] uppercase text-zinc-300 hover:border-zinc-500"
            >
              {showDirections ? "hide" : "show"}
            </button>
          )}
        </div>
        {showDirections && (
          <ul className="mt-1.5 space-y-1.5">
            {p.futureDirections.map((d, i) => (
              <li key={i} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5">
                <p className="text-[12px] leading-5 text-zinc-300">{d.direction}</p>
                <p className="mt-1 text-[11px] leading-4 text-zinc-500">
                  <span className="text-zinc-600">“</span>
                  {d.quote}
                  <span className="text-zinc-600">”</span>
                  {d.location ? (
                    <span className="ml-1 font-mono text-[10px] text-zinc-600">{d.location}</span>
                  ) : null}
                </p>
              </li>
            ))}
          </ul>
        )}
      </div>

      {p.industryApplications.length > 0 && (
        <div>
          <Label tone="text-sky-400">industry applications ({p.industryApplications.length})</Label>
          <ul className="mt-1.5 space-y-1.5">
            {p.industryApplications.map((a, i) => (
              <li key={i} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="sky">{a.domain}</Badge>
                  {a.location && (
                    <span className="font-mono text-[10px] text-zinc-600">{a.location}</span>
                  )}
                </div>
                <p className="mt-1 text-[12px] leading-5 text-zinc-400">{a.detail}</p>
                {a.quote && (
                  <p className="mt-1 text-[11px] leading-4 text-zinc-500">
                    <span className="text-zinc-600">“</span>
                    {a.quote}
                    <span className="text-zinc-600">”</span>
                  </p>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {p.deploymentClaims.length > 0 && (
        <div>
          <Label tone="text-emerald-400">deployment claims ({p.deploymentClaims.length})</Label>
          <ul className="mt-1 list-disc space-y-1 pl-4 text-[12px] leading-5 text-zinc-400">
            {p.deploymentClaims.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      )}

      {rec && (
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <Label tone="text-emerald-400">
              harvest evidence ({EVIDENCE_SECTIONS.reduce(
                (n, [k]) => n + rec[k].evidence.length, 0,)})
            </Label>
            <button
              onClick={() => setShowEvidence((v) => !v)}
              className="rounded-md border border-zinc-700 px-1.5 py-0.5 font-mono text-[10px] uppercase text-zinc-300 hover:border-zinc-500"
            >
              {showEvidence ? "hide" : "show"}
            </button>
          </div>
          {showEvidence && (
            <div className="mt-2 space-y-3">
              {EVIDENCE_SECTIONS.map(([key, label]) =>
                rec[key].evidence.length > 0 ? (
                  <div key={key}>
                    <Label tone="text-zinc-500">{label} ({rec[key].evidence.length})</Label>
                    <QuoteList quotes={rec[key].evidence} />
                  </div>
                ) : null,
              )}
            </div>
          )}
        </div>
      )}

      {p.evidences.length > 0 && (
        <div>
          <Label>evidence keys (problem / solution tags)</Label>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {p.evidences.map((e) => (
              <Badge key={e} tone={p.problemTags.includes(e) ? "amber" : "sky"}>
                {e}
              </Badge>
            ))}
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-zinc-800 bg-black/30 p-2.5">
        <span className="font-mono text-[10px] text-zinc-600">{p.harvestFile}</span>
        <span className="flex flex-wrap gap-2">
          {p.doi && (
            <a
              href={`https://doi.org/${p.doi}`}
              target="_blank"
              rel="noreferrer"
              className="rounded border border-zinc-700 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300 hover:border-emerald-600"
            >
              doi:{p.doi}
            </a>
          )}
          {p.arxiv && (
            <a
              href={`https://arxiv.org/abs/${p.arxiv}`}
              target="_blank"
              rel="noreferrer"
              className="rounded border border-zinc-700 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300 hover:border-emerald-600"
            >
              arXiv:{p.arxiv}
            </a>
          )}
        </span>
      </div>
    </div>
  );
}
