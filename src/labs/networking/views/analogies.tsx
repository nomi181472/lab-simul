"use client";

import { Card, SectionTitle, Badge, PaperLink } from "@/labs/networking/ui";
import { ANALYSIS } from "@/labs/networking/data/analysis";
import { KB } from "@/labs/networking/data/kb";
import { useNetworking } from "@/labs/networking/context";

const DOMAIN_LABEL: Record<string, string> = Object.fromEntries(
  KB.domains.map((d) => [d.id, d.label]),
);

export function AnalogiesView() {
  const { setSection } = useNetworking();
  const rows = ANALYSIS.crossDomain;

  return (
    <div className="space-y-6">
      <SectionTitle>Cross-domain analogies</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          Where the corpus itself reaches outside networking: papers that map a networking
          problem onto a physical, mathematical or biological system. The mapping and
          shared-principle text is quoted from the papers; the field label and target
          domain are the lab&apos;s classification of that quote.
        </p>
      </Card>

      <div className="space-y-3">
        {rows.map((a) => (
          <Card key={a.id}>
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="text-[13px] font-medium text-zinc-100">{a.analogy}</span>
              <div className="flex items-center gap-2">
                <Badge tone="violet">{a.physicalField}</Badge>
                <button
                  onClick={() => setSection("clusters")}
                  className="rounded-md border border-zinc-700 bg-zinc-900 px-2 py-0.5 font-mono text-[10px] text-zinc-400 hover:bg-zinc-800"
                >
                  → target: {DOMAIN_LABEL[a.target] ?? a.target}
                </button>
              </div>
            </div>

            <div className="mt-2 grid gap-3 lg:grid-cols-2">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                  mapping
                </p>
                <p className="mt-1 text-[12px] leading-5 text-zinc-300">{a.mapping}</p>
              </div>
              <div>
                <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                  shared principle
                </p>
                <p className="mt-1 text-[12px] leading-5 text-zinc-300">
                  {a.sharedPrinciple}
                </p>
              </div>
            </div>

            <div className="mt-3">
              <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                {a.paperIds.length} papers making this analogy
              </p>
              <div className="mt-1 flex flex-wrap gap-1">
                {a.paperIds.map((id) => (
                  <PaperLink key={id} id={id} short />
                ))}
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
