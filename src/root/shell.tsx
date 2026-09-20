"use client";

import { LABS, type Accent } from "./labs";
import { ActiveLab } from "./dynamic-loader";
import { RootProvider, useRoot } from "./root-context";

const ACCENT_ACTIVE: Record<Accent, string> = {
  emerald: "border-emerald-700/70 bg-emerald-600/15 text-emerald-200",
  sky: "border-sky-700/70 bg-sky-600/15 text-sky-200",
  amber: "border-amber-700/70 bg-amber-600/15 text-amber-200",
  violet: "border-violet-700/70 bg-violet-600/15 text-violet-200",
  rose: "border-rose-700/70 bg-rose-600/15 text-rose-200",
  cyan: "border-cyan-700/70 bg-cyan-600/15 text-cyan-200",
};

const ACCENT_INACTIVE =
  "border-transparent text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200";

function RootTabBar() {
  const { lab, setLab } = useRoot();
  return (
    <header className="sticky top-0 z-50 border-b border-zinc-800 bg-zinc-950/95 backdrop-blur">
      <div className="mx-auto flex w-full max-w-7xl items-center gap-1.5 overflow-x-auto px-4 py-2.5">
        <span className="mr-2 hidden shrink-0 font-mono text-[10px] uppercase tracking-widest text-zinc-600 sm:inline">
          Labs
        </span>
        <span
          className="-mx-2 mr-1 shrink-0 font-mono text-[13px] font-bold text-emerald-500 sm:hidden"
          aria-hidden
        >
          ⊙
        </span>
        {LABS.map((l) => {
          const active = l.id === lab;
          return (
            <button
              key={l.id}
              onClick={() => setLab(l.id)}
              className={`flex shrink-0 items-center gap-2 rounded-lg border px-3 py-2 font-mono text-[12px] font-semibold uppercase tracking-wide transition-colors sm:py-1.5 ${
                active ? ACCENT_ACTIVE[l.accent] : ACCENT_INACTIVE
              }`}
            >
              {l.label}
              <span
                className={`hidden font-normal normal-case tracking-normal sm:inline ${
                  active ? "text-[9px] opacity-70" : "text-[9px] opacity-50"
                }`}
              >
                {l.blurb}
              </span>
            </button>
          );
        })}
      </div>
    </header>
  );
}

export function RootShell() {
  return (
    <RootProvider>
      <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-300">
        <RootTabBar />
        <ActiveLab />
      </div>
    </RootProvider>
  );
}