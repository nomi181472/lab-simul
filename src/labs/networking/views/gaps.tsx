"use client";
import { Card, SectionTitle, Badge, Quote, PaperLink } from "@/labs/networking/ui";
import { ANALYSIS } from "@/labs/networking/data/analysis";
import { PAPERS } from "@/labs/networking/data/records";

const NOT = "not stated in the retrieved text";

export function GapsView() {
  const ai = ANALYSIS.academicIndustry;
  const industryOnly = ai.industryExamples;
  const academicOnly = PAPERS.filter((p) => p.industryEvidence === NOT).slice(0, 14);

  return (
    <div className="space-y-6">
      <SectionTitle>Academic ↔ industry gap explorer</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          Three gap classes: solutions the literature works on but no industry quote in
          the corpus supports (academic-only), problems industry papers raise that the
          academic subset does not restate (industry-leaning), and the combined picture —
          domains where neither side quotes the other.
        </p>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card tone="violet">
          <p className="mb-2 font-mono text-[10px] uppercase tracking-wide text-violet-400">
            academic-only papers (no industry evidence in retrieved text)
          </p>
          <div className="space-y-1">
            {academicOnly.map((p) => (
              <div key={p.id} className="flex items-baseline justify-between gap-2">
                <span className="min-w-0 truncate text-[11px] text-zinc-300">
                  <PaperLink id={p.id} />
                </span>
                <span className="shrink-0 font-mono text-[10px] text-zinc-500">
                  {p.citations.toLocaleString()}
                </span>
              </div>
            ))}
          </div>
          <p className="mt-2 font-mono text-[9px] text-zinc-600">
            absence of an industry quote is not absence of adoption — it is absence of
            evidence in the text we hold
          </p>
        </Card>

        <Card tone="ok">
          <p className="mb-2 font-mono text-[10px] uppercase tracking-wide text-emerald-400">
            industry-leaning papers (strongest quotes)
          </p>
          <div className="space-y-2">
            {industryOnly.slice(0, 8).map((ex) => (
              <div key={ex.id}>
                <PaperLink id={ex.id} />
                <Quote>{ex.evidence.slice(0, 220)}</Quote>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card tone="warn">
        <p className="mb-3 font-mono text-[10px] uppercase tracking-wide text-amber-400">
          low-adoption domains and the barrier each states
        </p>
        <div className="space-y-3">
          {ai.lowAdoption.map((c) => (
            <div key={c.domainId} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="text-[12px] font-medium text-zinc-200">
                  {c.domainId.replace(/-/g, " ")}
                </span>
                <Badge tone="rose">
                  adoption {(c.adoptionRatio * 100).toFixed(0)}% ({c.industryEvidencePapers} of{" "}
                  {c.industryEvidencePapers + c.academicOnlyPapers})
                </Badge>
              </div>
              <p className="mt-1 text-[11px] leading-5 text-zinc-500">{c.whyLow}</p>
            </div>
          ))}
          {ai.lowAdoption.length === 0 && (
            <p className="text-[12px] text-zinc-500">no domain fell below the 25% bar</p>
          )}
        </div>
      </Card>
    </div>
  );
}
