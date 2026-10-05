"use client";

import { useMemo } from "react";
import { PAPERS } from "@/labs/wifi-sensing/data/papers";
import { CONCEPTS } from "@/labs/wifi-sensing/data/concepts";
import { useWifiLab } from "@/labs/wifi-sensing/context";
import { Badge, Card, Formula, SectionTitle, WifiPaperLink } from "@/labs/wifi-sensing/ui";

/* The path is defined by prerequisite chains between concepts, not by paper
 * order: you cannot understand a paper that uses an equation you have not met.
 * Each stage then lists the corpus papers that instantiate its concepts. */

const STAGES = [
  {
    id: "radio",
    title: "1. The link already exists",
    blurb:
      "Before sensing, a normal WiFi link: transmitter, receiver, multipath, and a channel that changes for free with every reflection. Nothing here is sensing-specific.",
    concepts: ["csi", "multipath", "rssi", "path-loss", "fresnel-zone"],
  },
  {
    id: "motion",
    title: "2. Motion shows up as phase",
    blurb:
      "A moving reflector advances the phase of the echo it returns. That single fact is the whole basis of WiFi motion sensing — and the source of every ambiguity that follows.",
    concepts: ["doppler", "packet-rate", "phase-wrap", "cfo-sfo", "quantization"],
  },
  {
    id: "features",
    title: "3. Turning the stream into evidence",
    blurb:
      "Raw CSI is unusable directly. Windows, transforms and statistics convert a packet stream into something that separates classes.",
    concepts: ["stft", "periodogram", "feature-families", "receptive-field"],
  },
  {
    id: "estimation",
    title: "4. From signal to position",
    blurb:
      "Two families of solution: measure the map and look it up (fingerprinting), or model the physics and solve (model-based). Their failure modes differ.",
    concepts: ["fingerprinting", "model-based-localization", "kalman", "through-wall"],
  },
  {
    id: "learning",
    title: "5. Learning the decision",
    blurb:
      "Where the accuracy numbers come from — and why most of them are decided by feature separability before the architecture is chosen.",
    concepts: ["separation-ceiling", "snr", "receptive-field"],
  },
  {
    id: "limits",
    title: "6. Knowing when to distrust it",
    blurb:
      "The part of the literature that matters most and is reported least: SNR ceilings, domain shift, and the ground-truth problem.",
    concepts: ["snr", "domain-shift", "ground-truth", "privacy"],
  },
] as const;

export function PathView() {
  const { setSection, setMathSim } = useWifiLab();

  const byId = useMemo(() => {
    const m = new Map<string, (typeof CONCEPTS)[number]>();
    for (const c of CONCEPTS) m.set(c.id, c);
    return m;
  }, []);

  return (
    <div className="space-y-6">
      <SectionTitle>Learning path</SectionTitle>
      <p className="max-w-3xl text-[12px] leading-6 text-zinc-500">
        Ordered by conceptual prerequisite, not by citation count. Each stage names the concepts
        you need, then points at the corpus papers that instantiate them — so you can read a real
        paper at each stage instead of only reading about it.
      </p>

      <div className="space-y-4">
        {STAGES.map((s, i) => (
          <div key={s.id} className="relative pl-6">
            {i < STAGES.length - 1 ? (
              <div className="absolute left-2 top-6 h-full w-px bg-zinc-800" />
            ) : null}
            <div className="absolute left-0 top-1.5 h-3 w-3 rounded-full border-2 border-cyan-600 bg-zinc-950" />
            <Card tone={i === 0 ? "accent" : "default"}>
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-sm font-semibold text-zinc-100">{s.title}</h3>
                <Badge tone="cyan">{s.concepts.length} concepts</Badge>
              </div>
              <p className="mt-1 text-[12px] leading-5 text-zinc-400">{s.blurb}</p>

              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                {s.concepts.map((cid) => {
                  const c = byId.get(cid);
                  if (!c) return null;
                  return (
                    <div key={cid} className="rounded-lg border border-zinc-800 bg-black/20 p-2.5">
                      <div className="flex items-center gap-2">
                        <span className="text-[12px] font-medium text-zinc-200">{c.name}</span>
                        {c.simulator ? (
                          <button
                            onClick={() => {
                              setMathSim(c.simulator!);
                              setSection("math");
                            }}
                            className="ml-auto rounded border border-cyan-800 px-1.5 py-0.5 font-mono text-[9px] text-cyan-300 hover:bg-cyan-950"
                          >
                            simulate
                          </button>
                        ) : null}
                      </div>
                      {c.formula ? (
                        <div className="mt-1.5">
                          <Formula>{c.formula}</Formula>
                        </div>
                      ) : null}
                      <p className="mt-1.5 text-[11px] leading-5 text-zinc-500">{c.intuition}</p>
                      {c.paperIds.length ? (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {c.paperIds.slice(0, 8).map((pid) => (
                            <WifiPaperLink key={pid} id={pid} short link />
                          ))}
                          {c.paperIds.length > 8 ? (
                            <span className="font-mono text-[10px] text-zinc-600">
                              +{c.paperIds.length - 8}
                            </span>
                          ) : null}
                        </div>
                      ) : (
                        <div className="mt-2 font-mono text-[10px] text-zinc-600">
                          no corpus paper matched this concept
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              <div className="mt-3">
                <div className="font-mono text-[9px] uppercase tracking-wide text-zinc-500">
                  read at this stage
                </div>
                <div className="mt-1 space-y-1">
                  {papersFor(s.concepts).slice(0, 4).map((p) => (
                    <div key={p.id} className="flex items-start gap-2">
                      <WifiPaperLink id={p.id} short link />
                      <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-500">
                        {p.title}
                      </span>
                      <span className="shrink-0 font-mono text-[10px] text-zinc-600">
                        {p.citations}c
                      </span>
                    </div>
                  ))}
                  {papersFor(s.concepts).length === 0 ? (
                    <div className="font-mono text-[10px] text-zinc-600">no papers matched</div>
                  ) : null}
                </div>
              </div>
            </Card>
          </div>
        ))}
      </div>
    </div>
  );
}

function papersFor(conceptIds: readonly string[]) {
  const set = new Set(conceptIds);
  const hits: { id: string; title: string; citations: number }[] = [];
  for (const p of PAPERS) {
    if (p.equationIds.some((c) => set.has(c)) || p.concepts.some((c) => set.has(c))) {
      hits.push({ id: p.id, title: p.title, citations: p.citations });
    }
  }
  return hits.sort((a, b) => b.citations - a.citations);
}