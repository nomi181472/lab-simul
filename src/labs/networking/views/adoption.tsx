"use client";
import { Card, SectionTitle, Badge } from "@/labs/networking/ui";
import { ANALYSIS } from "@/labs/networking/data/analysis";
import { KB } from "@/labs/networking/data/kb";
import { useNetworking } from "@/labs/networking/context";

const DOMAIN_LABEL: Record<string, string> = Object.fromEntries(
  KB.domains.map((d) => [d.id, d.label]),
);

export function AdoptionView() {
  const { setSection } = useNetworking();
  const ai = ANALYSIS.academicIndustry;

  const sorted = [...ai.adoptionByDomain].sort((a, b) => b.adoptionRatio - a.adoptionRatio);

  return (
    <div className="space-y-6">
      <SectionTitle>Academic ↔ industry adoption explorer</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          Adoption ratio per domain = share of that domain&apos;s papers whose retrieved
          text contains an industry sentence (deployment, operator, RFC, commercial
          product). High ratio means the topic is discussed alongside real networks in the
          corpus; low ratio means it stays on the academic side.
        </p>
      </Card>

      <Card>
        <div className="space-y-2">
          {sorted.map((c) => (
            <div key={c.domainId} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="text-[12px] font-medium text-zinc-200">
                  {DOMAIN_LABEL[c.domainId] ?? c.domainId}
                </span>
                <div className="flex items-center gap-2">
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
                  <span className="font-mono text-[10px] text-zinc-500">
                    {(c.adoptionRatio * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="mt-2 flex h-3 overflow-hidden rounded bg-zinc-800">
                <div
                  className="h-3 bg-emerald-700/70"
                  style={{ width: `${(c.adoptionRatio * 100).toFixed(1)}%` }}
                />
                <div className="h-3 flex-1 bg-violet-800/50" />
              </div>
              <div className="mt-1 flex flex-wrap gap-3 font-mono text-[10px] text-zinc-500">
                <span className="text-emerald-500">
                  {c.industryEvidencePapers} industry-evidence
                </span>
                <span className="text-violet-400">{c.academicOnlyPapers} academic-only</span>
                {c.verdict === "low-adoption" && (
                  <span className="text-amber-500">why low: {c.whyLow}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <p className="text-[12px] leading-6 text-zinc-500">
          Green = industry-evidence share, violet = academic-only share.{" "}
          <button
            onClick={() => setSection("industry")}
            className="font-mono text-[11px] text-sky-400 hover:text-sky-300"
          >
            → industry explorer for the quotes behind the green bars
          </button>
        </p>
      </Card>
    </div>
  );
}
