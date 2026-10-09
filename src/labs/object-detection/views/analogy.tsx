"use client";

import { useMemo, useState } from "react";
import { ANALOGIES } from "@/labs/object-detection/data/crossdomain";
import { Card, Badge, PaperLink, SectionTitle } from "@/labs/object-detection/ui";

export function AnalogyView() {
  const domains = useMemo(
    () => [...new Set(ANALOGIES.map((a) => a.domain))],
    [],
  );
  const [domain, setDomain] = useState<string | null>(null);
  const shown = domain ? ANALOGIES.filter((a) => a.domain === domain) : ANALOGIES;

  return (
    <div className="space-y-6">
      <div>
        <SectionTitle>Cross-domain analogies</SectionTitle>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-zinc-400">
          {ANALOGIES.length} analogies across {domains.length} source domains. Policy: a mapping
          is shown only when a corpus paper itself states the connection — each card lists the
          papers that make it. Wording the lab adds beyond those papers is labeled{" "}
          <span className="font-mono text-violet-300">lab synthesis</span>, never presented as a
          paper&apos;s claim. Nothing here is a metaphor the corpus did not draw.
        </p>
      </div>

      <div className="flex flex-wrap gap-1.5">
        <button
          onClick={() => setDomain(null)}
          className={`rounded-md border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wide transition-colors ${
            domain === null
              ? "border-violet-700/60 bg-violet-600/20 text-violet-300"
              : "border-zinc-800 bg-zinc-900/40 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200"
          }`}
        >
          all ({ANALOGIES.length})
        </button>
        {domains.map((d) => (
          <button
            key={d}
            onClick={() => setDomain(d)}
            className={`rounded-md border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wide transition-colors ${
              domain === d
                ? "border-violet-700/60 bg-violet-600/20 text-violet-300"
                : "border-zinc-800 bg-zinc-900/40 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200"
            }`}
          >
            {d}
          </button>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        {shown.map((a) => (
          <Card key={a.id}>
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone="violet">{a.domain}</Badge>
              <Badge tone="emerald">paper-stated connection</Badge>
            </div>

            <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-stretch">
              <div className="flex-1 rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
                <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                  source concept
                </div>
                <p className="mt-1 text-[12px] leading-5 text-zinc-300">{a.sourceConcept}</p>
              </div>
              <div className="flex items-center justify-center font-mono text-lg text-emerald-500 rotate-90 sm:rotate-0">
                →
              </div>
              <div className="flex-1 rounded-lg border border-emerald-800/50 bg-emerald-950/20 p-3">
                <div className="font-mono text-[9px] uppercase tracking-wide text-emerald-500">
                  detection concept
                </div>
                <p className="mt-1 text-[12px] leading-5 text-emerald-100/90">
                  {a.detectionConcept}
                </p>
              </div>
            </div>

            <p className="mt-3 text-[12px] leading-6 text-zinc-400">{a.mapping}</p>

            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                evidence
              </span>
              {a.paperIds.map((pid) => (
                <PaperLink key={pid} id={pid} />
              ))}
            </div>
          </Card>
        ))}
      </div>

      <Card tone="accent">
        <div className="flex flex-wrap items-start gap-2">
          <Badge tone="violet">policy</Badge>
          <p className="text-[11px] leading-5 text-zinc-400">
            Cross-domain borrowing is only credited to a domain when a corpus paper states it in
            its own text; the paper links on each card are that statement. The lab never invents
            an analogy to make a mechanism sound intuitive — where a mapping needs interpretive
            wording, it carries the{" "}
            <span className="font-mono text-violet-300">lab synthesis</span> label and stands as
            analysis, not evidence.
          </p>
        </div>
      </Card>
    </div>
  );
}
