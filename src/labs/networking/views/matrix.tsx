"use client";
import { Card, SectionTitle, Badge } from "@/labs/networking/ui";
import { ANALYSIS } from "@/labs/networking/data/analysis";

export function MatrixView() {
  const { periods, rows } = ANALYSIS.problemMatrix;
  const max = Math.max(1, ...rows.flatMap((r) => r.cells));

  return (
    <div className="space-y-6">
      <SectionTitle>Problem matrix</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          Domain × period incidence grid. Cell shade is paper count within the period;
          rows are ordered by total volume. This is the field&apos;s shape at a glance —
          where research concentrates, and which eras a domain was quiet.
        </p>
      </Card>

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[11px]">
            <thead>
              <tr>
                <th className="border-b border-zinc-800 px-2 py-1.5 text-left font-mono text-[10px] font-normal uppercase tracking-wide text-zinc-500">
                  domain
                </th>
                {periods.map((p) => (
                  <th
                    key={p}
                    className="border-b border-zinc-800 px-2 py-1.5 text-right font-mono text-[10px] font-normal uppercase tracking-wide text-zinc-500"
                  >
                    {p}
                  </th>
                ))}
                <th className="border-b border-zinc-800 px-2 py-1.5 text-right font-mono text-[10px] font-normal uppercase tracking-wide text-zinc-500">
                  total
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.domainId}>
                  <td className="border-b border-zinc-900 px-2 py-1.5 text-zinc-300">
                    {r.label}
                  </td>
                  {r.cells.map((n, i) => {
                    const alpha = n === 0 ? 0 : 0.15 + (n / max) * 0.85;
                    return (
                      <td
                        key={i}
                        className="border-b border-zinc-900 px-2 py-1.5 text-right font-mono"
                        style={{
                          background: n ? `rgba(14, 116, 144, ${alpha.toFixed(2)})` : "transparent",
                          color: n ? (alpha > 0.55 ? "#e4f4ff" : "#94a3b8") : "#3f3f46",
                        }}
                      >
                        {n || "·"}
                      </td>
                    );
                  })}
                  <td className="border-b border-zinc-900 px-2 py-1.5 text-right font-mono text-zinc-400">
                    {r.total}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <p className="font-mono text-[10px] uppercase tracking-wide text-zinc-500">
          reading the grid
        </p>
        <p className="mt-1 text-[12px] leading-6 text-zinc-400">
          A row that is dense only in 2011–2014 is a settled problem; a row that grows
          left-to-right is a rising one. Sparse rows are gaps: real domains with too few
          papers in this corpus to claim either answer. Click{" "}
          <span className="font-mono text-sky-300">Clusters</span> to see where the same
          problem was attacked repeatedly across periods.
        </p>
        <div className="mt-2 flex flex-wrap gap-1">
          {[0, 1, 2, 3].map((i) => (
            <Badge key={i}>{periods[i]}</Badge>
          ))}
        </div>
      </Card>
    </div>
  );
}
