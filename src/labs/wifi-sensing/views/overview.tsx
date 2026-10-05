"use client";

import { MANIFEST, MANIFEST_YEAR_MAX, MANIFEST_YEAR_MIN, MANIFEST_PAGES } from "@/labs/wifi-sensing/data/manifest";
import { PAPERS, PAPER_COUNT } from "@/labs/wifi-sensing/data/papers";
import { CONCEPTS } from "@/labs/wifi-sensing/data/concepts";
import { METRICS } from "@/labs/wifi-sensing/data/metrics";
import { DATASETS } from "@/labs/wifi-sensing/data/datasets";
import { SECTIONS, TASK_LABEL, MODALITY_LABEL } from "@/labs/wifi-sensing/data/types";
import { useWifiLab } from "@/labs/wifi-sensing/context";
import { Badge, Card, CiteBadge, SectionTitle, WifiPaperLink } from "@/labs/wifi-sensing/ui";

function countBy<T extends string>(rows: { [k: string]: T }[], field: string) {
  const m = new Map<string, number>();
  for (const r of rows) {
    const k = String((r as Record<string, unknown>)[field] ?? "other");
    m.set(k, (m.get(k) ?? 0) + 1);
  }
  return [...m.entries()].sort((a, b) => b[1] - a[1]);
}

export function OverviewView() {
  const { setSection } = useWifiLab();

  const byTask = countBy(MANIFEST as unknown as Record<string, string>[], "task");
  const byModality = countBy(MANIFEST as unknown as Record<string, string>[], "modality");
  const totalCites = MANIFEST.reduce((s, p) => s + p.citations, 0);
  const withRelations = PAPERS.filter((p) => p.relations.length > 0).length;
  const totalRelations = PAPERS.reduce((s, p) => s + p.relations.length, 0);
  const withResults = PAPERS.filter((p) => p.results.length > 0).length;

  return (
    <div className="space-y-8">
      <section className="rounded-2xl border border-cyan-800/40 bg-gradient-to-b from-cyan-950/30 to-zinc-950 p-8">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="cyan">evidence-grounded</Badge>
          <Badge tone="zinc">
            {PAPER_COUNT}/{MANIFEST.length} papers structured
          </Badge>
          <Badge tone="zinc">
            {MANIFEST_YEAR_MIN}–{MANIFEST_YEAR_MAX}
          </Badge>
          <Badge tone="emerald">device-free</Badge>
        </div>
        <h1 className="mt-4 max-w-2xl text-3xl font-semibold leading-tight tracking-tight text-zinc-50">
          WiFi Sensing Research Lab
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-zinc-400">
          Inference from radio. A WiFi link already exists between an access point and a device;{" "}
          <span className="text-zinc-200">a person walking through that link perturbs it</span>, and
          the perturbation can be decoded into what the person is doing, where they are, or how
          they are breathing. Every record (W001…W
          {MANIFEST.length}) is extracted from its corpus PDF — no invented results, no invented
          equations, no invented relations. Papers appear here as a lineage: each exists because of
          previous ideas, and each leaves a gap the next one attacks.
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          <button
            onClick={() => setSection("path")}
            className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-zinc-950 hover:bg-cyan-500"
          >
            Start the learning path
          </button>
          <button
            onClick={() => setSection("math")}
            className="rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-300 hover:bg-zinc-900"
          >
            Operate the 20 simulators
          </button>
          <button
            onClick={() => setSection("papers")}
            className="rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-300 hover:bg-zinc-900"
          >
            Browse {MANIFEST.length} papers
          </button>
        </div>
      </section>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { k: "papers", v: String(MANIFEST.length), s: `${totalCites.toLocaleString()} citations` },
          { k: "pages read", v: MANIFEST_PAGES.toLocaleString(), s: "extracted with pdftotext" },
          { k: "citation edges", v: String(totalRelations), s: `across ${withRelations} papers` },
          {
            k: "papers with numbers",
            v: `${withResults}/${PAPER_COUNT}`,
            s: "reported outcomes extracted",
          },
        ].map((s) => (
          <Card key={s.k}>
            <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">{s.k}</div>
            <div className="mt-1 font-mono text-2xl text-cyan-300">{s.v}</div>
            <div className="mt-1 text-[11px] text-zinc-500">{s.s}</div>
          </Card>
        ))}
      </div>

      <section className="space-y-3">
        <SectionTitle>What the corpus is made of</SectionTitle>
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
              by task (classified from each paper’s own text)
            </div>
            <div className="mt-2 space-y-1">
              {byTask.slice(0, 9).map(([k, v]) => (
                <div key={k} className="flex items-center gap-2">
                  <span className="w-40 shrink-0 truncate text-[11px] text-zinc-400">
                    {TASK_LABEL[k as keyof typeof TASK_LABEL] ?? k}
                  </span>
                  <div className="h-2 flex-1 overflow-hidden rounded bg-zinc-800">
                    <div
                      className="h-full bg-cyan-600/70"
                      style={{ width: `${(v / byTask[0][1]) * 100}%` }}
                    />
                  </div>
                  <span className="w-6 text-right font-mono text-[10px] text-zinc-500">{v}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
              by measurement modality
            </div>
            <div className="mt-2 space-y-1">
              {byModality.map(([k, v]) => (
                <div key={k} className="flex items-center gap-2">
                  <span className="w-40 shrink-0 truncate text-[11px] text-zinc-400">
                    {MODALITY_LABEL[k as keyof typeof MODALITY_LABEL] ?? k}
                  </span>
                  <div className="h-2 flex-1 overflow-hidden rounded bg-zinc-800">
                    <div
                      className="h-full bg-violet-600/70"
                      style={{ width: `${(v / byModality[0][1]) * 100}%` }}
                    />
                  </div>
                  <span className="w-6 text-right font-mono text-[10px] text-zinc-500">{v}</span>
                </div>
              ))}
            </div>
            <p className="mt-3 text-[11px] leading-5 text-zinc-500">
              The split that matters is RSSI (one scalar per packet) versus CSI (a complex value
              per subcarrier). Everything fine-grained in this field is downstream of having
              per-subcarrier phase.
            </p>
          </Card>
        </div>
      </section>

      <section className="space-y-3">
        <SectionTitle>Most-cited work in the corpus</SectionTitle>
        <div className="grid gap-3 md:grid-cols-2">
          {MANIFEST.slice(0, 8).map((p) => {
            const rec = PAPERS.find((r) => r.id === p.id);
            return (
              <Card key={p.id}>
                <div className="flex items-start justify-between gap-2">
                  <WifiPaperLink id={p.id} link />
                  <CiteBadge n={p.citations} />
                </div>
                <p className="mt-1.5 text-[12px] leading-5 text-zinc-300">{p.title}</p>
                <p className="mt-1 font-mono text-[10px] text-zinc-500">
                  {p.year} · {p.venue || "arXiv"} · {p.authors.slice(0, 3).join(", ")}
                  {p.authors.length > 3 ? " et al." : ""}
                </p>
                {rec?.summary ? (
                  <p className="mt-2 line-clamp-3 text-[11px] leading-5 text-zinc-500">
                    {rec.summary}
                  </p>
                ) : null}
              </Card>
            );
          })}
        </div>
      </section>

      <section className="space-y-3">
        <SectionTitle>How to use this lab</SectionTitle>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {SECTIONS.map((s) => (
            <button
              key={s.id}
              onClick={() => setSection(s.id)}
              className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 text-left transition-colors hover:border-cyan-800 hover:bg-zinc-900"
            >
              <div className="text-[12px] font-medium text-zinc-200">{s.label}</div>
              <div className="mt-0.5 text-[11px] leading-4 text-zinc-500">{s.blurb}</div>
            </button>
          ))}
        </div>
      </section>

      <Card tone="accent">
        <div className="font-mono text-[10px] uppercase tracking-wide text-cyan-400">
          evidence policy
        </div>
        <p className="mt-1 text-[12px] leading-6 text-zinc-300">
          {CONCEPTS.length} concepts, {METRICS.length} metrics and {DATASETS.length} dataset records
          are linked to the papers that actually contain them. Every prose field in a paper record is
          a sentence taken from that paper&apos;s own text; every relation is a real citation edge
          detected in a reference block; every pipeline field is a keyword detection over the paper
          itself. Where a paper&apos;s text does not contain something, the field is empty rather
          than filled in — and the Audit tab reports those gaps honestly.
        </p>
      </Card>
    </div>
  );
}