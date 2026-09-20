"use client";

import { YEAR_LIST } from "@/labs/object-detection/data/years";
import { useLab } from "@/labs/object-detection/context";

export function YearTicker({ accent = false }: { accent?: boolean }) {
  const { year, setYear, setSection } = useLab();
  return (
    <div className="flex min-w-0 items-center gap-1 overflow-x-auto font-mono">
      {YEAR_LIST.map((y) => (
        <button
          key={y}
          onClick={() => {
            setYear(y);
            if (!accent) setSection("year");
          }}
          title={`${y}`}
          className={`shrink-0 rounded px-2 py-1 text-[11px] tabular-nums transition-colors ${
            y === year
              ? "bg-emerald-600/25 text-emerald-300 border border-emerald-700/60"
              : "text-zinc-500 border border-transparent hover:text-zinc-300"
          }`}
        >
          {y}
        </button>
      ))}
    </div>
  );
}