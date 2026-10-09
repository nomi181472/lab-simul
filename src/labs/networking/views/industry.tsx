"use client";
import { useState } from "react";
import { Card, SectionTitle, Badge, Quote, PaperLink } from "@/labs/networking/ui";
import { ANALYSIS } from "@/labs/networking/data/analysis";
import { KB } from "@/labs/networking/data/kb";

const DOMAIN_LABEL: Record<string, string> = Object.fromEntries(
  KB.domains.map((d) => [d.id, d.label]),
);

export function IndustryView() {
  const ai = ANALYSIS.academicIndustry;
  const [openId, setOpenId] = useState<string | null>(null);

  return (
    <div className="space-y-6">
      <SectionTitle>Industry solution explorer</SectionTitle>

      <div className="grid gap-3 sm:grid-cols-3">
        <Card tone="accent">
          <p className="font-mono text-[10px] uppercase tracking-wide text-sky-400">
            industry-evidence papers
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">
            {ai.industryEvidenceCount}
          </p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            deployment / operator / RFC quotes found
          </p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            academic-only papers
          </p>
          <p className="mt-1 text-2xl font-semibold text-zinc-100">{ai.academicOnlyCount}</p>
          <p className="mt-1 font-mono text-[10px] text-zinc-500">
            no industry sentence in retrieved text
          </p>
        </Card>
        <Card>
          <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            application evolution
          </p>
          <p className="mt-1 text-[13px] leading-5 text-zinc-400">
            Deployment quotes are ranked by citation; each is the paper&apos;s own
            sentence about operating networks.
          </p>
        </Card>
      </div>

      <Card>
        <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          strongest industry evidence (citations desc)
        </p>
        <div className="space-y-2">
          {ai.industryExamples.map((ex) => {
            const open = openId === ex.id;
            return (
              <div key={ex.id} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                <button
                  className="flex w-full items-start justify-between gap-3 text-left"
                  onClick={() => setOpenId(open ? null : ex.id)}
                >
                  <div className="min-w-0">
                    <PaperLink id={ex.id} />
                    <p className="mt-0.5 font-mono text-[10px] text-zinc-500">
                      {ex.year} · {ex.domains.map((d) => DOMAIN_LABEL[d] ?? d).join(", ")}
                    </p>
                  </div>
                  <Badge tone={open ? "cyan" : "zinc"}>{open ? "hide" : "evidence"}</Badge>
                </button>
                {open && (
                  <div className="mt-2 space-y-2 border-t border-zinc-800 pt-2">
                    <Quote>{ex.evidence}</Quote>
                    {ex.solutionFamilies.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {ex.solutionFamilies.map((f) => (
                          <Badge key={f} tone="violet">
                            {KB.solutions.find((s) => s.familyId === f)?.label ?? f}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </Card>

      <Card>
        <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          academic vs industry comparison by domain
        </p>
        <div className="space-y-1">
          {ai.adoptionByDomain.slice(0, 14).map((c) => (
            <div key={c.domainId} className="flex items-center gap-3">
              <span className="w-44 shrink-0 truncate text-[11px] text-zinc-300">
                {DOMAIN_LABEL[c.domainId] ?? c.domainId}
              </span>
              <div className="flex h-4 flex-1 overflow-hidden rounded bg-zinc-800">
                <div
                  className="h-4 bg-emerald-700/70"
                  style={{
                    width: `${(c.adoptionRatio * 100).toFixed(1)}%`,
                  }}
                  title="industry-evidence share"
                />
              </div>
              <span className="w-32 shrink-0 text-right font-mono text-[10px] text-zinc-500">
                {c.industryEvidencePapers} ind / {c.academicOnlyPapers} aca
              </span>
              <Badge
                tone={
                  c.verdict === "high-adoption"
                    ? "emerald"
                    : c.verdict === "low-adoption"
                      ? "rose"
                      : "amber"
                }
              >
                {c.verdict}
              </Badge>
            </div>
          ))}
        </div>
        <p className="mt-2 font-mono text-[9px] text-zinc-600">
          green segment = share of the domain&apos;s papers carrying an industry-evidence
          quote · verdict thresholds: high ≥ 0.5, low ≤ 0.25
        </p>
      </Card>
    </div>
  );
}
