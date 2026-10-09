"use client";

import { YEARS, YEAR_STATES } from "@/labs/object-detection/data/years";
import { YEAR_DELTAS } from "@/labs/object-detection/data/year-deltas";
import { useLab } from "@/labs/object-detection/context";
import {
  Card,
  Badge,
  PaperLink,
  SectionTitle,
  type BadgeTone,
} from "@/labs/object-detection/ui";

function ChipBlock({
  label,
  items,
  tone,
}: {
  label: string;
  items: string[];
  tone: BadgeTone;
}) {
  return (
    <div>
      <div className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
        {label}
      </div>
      {items.length === 0 ? (
        <div className="mt-1 font-mono text-[11px] text-zinc-600">—</div>
      ) : (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {items.map((it, i) => (
            <Badge key={`${it}-${i}`} tone={tone}>
              {it}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

function ListBlock({
  label,
  items,
  tone = "zinc",
}: {
  label: string;
  items: string[];
  tone?: BadgeTone;
}) {
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={tone}>{label}</Badge>
        <span className="font-mono text-[10px] text-zinc-600">{items.length}</span>
      </div>
      {items.length === 0 ? (
        <div className="mt-2 font-mono text-[11px] text-zinc-600">—</div>
      ) : (
        <ul className="mt-2 space-y-1">
          {items.map((it, i) => (
            <li key={`${it}-${i}`} className="flex gap-2 text-[12px] leading-5 text-zinc-300">
              <span className="text-zinc-600">›</span>
              <span>{it}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function YearView() {
  const { year, setYear } = useLab();
  const state = YEAR_STATES[year];
  const delta = YEAR_DELTAS.find((d) => d.from === year);

  const idx = YEARS.indexOf(year);
  const prev = idx > 0 ? YEARS[idx - 1] : null;
  const next = idx >= 0 && idx < YEARS.length - 1 ? YEARS[idx + 1] : null;

  if (!state) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">{year}</h1>
        <p className="text-sm text-zinc-400">No field-state record for {year}.</p>
        <div className="flex items-center justify-between">
          {prev ? (
            <button
              onClick={() => setYear(prev)}
              className="rounded-lg border border-zinc-800 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-900"
            >
              ← {prev}
            </button>
          ) : (
            <span />
          )}
          {next ? (
            <button
              onClick={() => setYear(next)}
              className="rounded-lg border border-zinc-800 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-900"
            >
              {next} →
            </button>
          ) : (
            <span />
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* header */}
      <div className="rounded-2xl border border-emerald-800/40 bg-gradient-to-b from-emerald-950/30 to-zinc-950 p-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="font-mono text-5xl font-semibold text-emerald-300">{state.year}</div>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Badge tone="zinc">{state.paperCount} papers</Badge>
              <Badge tone="zinc">field state</Badge>
              <Badge tone="zinc">{state.supportingPapers.length} supporting</Badge>
            </div>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {state.supportingPapers.slice(0, 14).map((id) => (
              <PaperLink key={id} id={id} short />
            ))}
            {state.supportingPapers.length > 14 && (
              <span className="font-mono text-[10px] text-zinc-500">
                +{state.supportingPapers.length - 14} more
              </span>
            )}
          </div>
        </div>
      </div>

      {/* year navigation */}
      <div className="flex items-center justify-between">
        {prev ? (
          <button
            onClick={() => setYear(prev)}
            className="rounded-lg border border-zinc-800 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-900"
          >
            ← {prev}
          </button>
        ) : (
          <span className="font-mono text-[10px] text-zinc-600">first year of the corpus</span>
        )}
        <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-600">
          year-by-year
        </span>
        {next ? (
          <button
            onClick={() => setYear(next)}
            className="rounded-lg border border-zinc-800 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-900"
          >
            {next} →
          </button>
        ) : (
          <span className="font-mono text-[10px] text-zinc-600">latest year of the corpus</span>
        )}
      </div>

      {/* vs previous year */}
      {delta ? (
        <Card tone="accent">
          <div className="flex flex-wrap items-center gap-2">
            <SectionTitle>
              {delta.from} → {delta.to}
            </SectionTitle>
            <Badge tone="emerald">vs previous year</Badge>
          </div>
          <p className="mt-2 text-xs leading-5 text-zinc-400">{delta.summary}</p>
          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            <ChipBlock label="added" items={delta.added} tone="emerald" />
            <ChipBlock label="removed" items={delta.removed} tone="rose" />
            <ChipBlock label="persisted" items={delta.persisted} tone="zinc" />
          </div>
          <div className="mt-3">
            <ChipBlock label="new problems" items={delta.newProblems} tone="violet" />
          </div>
          {delta.supportingPapers.length > 0 && (
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
                evidence
              </span>
              {delta.supportingPapers.map((id) => (
                <PaperLink key={id} id={id} short />
              ))}
            </div>
          )}
        </Card>
      ) : (
        <Card>
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="zinc">baseline year</Badge>
            <span className="text-[11px] leading-5 text-zinc-500">
              {state.year} opens the corpus — there is no previous year to diff against.
            </span>
          </div>
        </Card>
      )}

      {/* approaches & ideas */}
      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <ChipBlock label="dominant approaches" items={state.dominantApproaches} tone="emerald" />
        </Card>
        <Card>
          <ChipBlock label="emerging ideas" items={state.emergingIdeas} tone="sky" />
          <div className="mt-4">
            <ChipBlock label="declining ideas" items={state.decliningIdeas} tone="rose" />
          </div>
        </Card>
        <Card>
          <ChipBlock label="research directions" items={state.researchDirections} tone="emerald" />
        </Card>
      </div>

      {/* problems */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <SectionTitle>Problems</SectionTitle>
          <div className="mt-3 space-y-4">
            <ChipBlock label="new problems" items={state.newProblems} tone="violet" />
            <ChipBlock
              label="persistent problems"
              items={state.persistentProblems}
              tone="amber"
            />
          </div>
        </Card>
        <Card>
          <SectionTitle>Solved, reduced, unsolved</SectionTitle>
          <div className="mt-3 space-y-4">
            <ChipBlock
              label="solved or reduced"
              items={state.solvedOrReducedProblems}
              tone="emerald"
            />
            <ChipBlock label="unsolved problems" items={state.unsolvedProblems} tone="amber" />
          </div>
        </Card>
      </div>

      {/* developments */}
      <div>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Developments
        </h2>
        <div className="grid gap-4 lg:grid-cols-3">
          <Card>
            <ListBlock
              label="mathematical"
              items={state.mathematicalDevelopments}
              tone="violet"
            />
          </Card>
          <Card>
            <ListBlock
              label="architectural"
              items={state.architecturalDevelopments}
              tone="sky"
            />
          </Card>
          <Card>
            <ListBlock
              label="computational"
              items={state.computationalDevelopments}
              tone="emerald"
            />
          </Card>
        </div>
      </div>

      {/* applications + edge cases */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <ChipBlock label="applications" items={state.applications} tone="zinc" />
        </Card>
        <Card>
          <ChipBlock label="edge cases" items={state.edgeCases} tone="rose" />
        </Card>
      </div>

      {/* supporting papers */}
      <Card>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <SectionTitle>Supporting papers · {state.year}</SectionTitle>
          <span className="font-mono text-[10px] text-zinc-600">
            {state.supportingPapers.length} papers
          </span>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {state.supportingPapers.length === 0 ? (
            <span className="font-mono text-[11px] text-zinc-600">—</span>
          ) : (
            state.supportingPapers.map((id) => <PaperLink key={id} id={id} />)
          )}
        </div>
      </Card>

      <div className="flex flex-wrap items-center gap-2 font-mono text-[10px] text-zinc-600">
        <Badge tone="emerald">traced</Badge>
        <span>field-state claims above are grounded in this year&apos;s corpus papers</span>
      </div>
    </div>
  );
}
