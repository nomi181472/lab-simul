"use client";

export function ObjectTrackerLab() {
  return (
    <div className="flex flex-1 flex-col">
      <header className="border-b border-zinc-800 bg-zinc-950/95 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl items-center gap-3 px-4 py-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-sky-700/60 bg-sky-950/40 font-mono text-[13px] font-bold text-sky-300">
            ◉
          </div>
          <div className="leading-tight">
            <h1 className="text-sm font-semibold tracking-tight text-zinc-100">
              OBJECT TRACKER
            </h1>
            <p className="font-mono text-[10px] text-zinc-500">
              Research Lab · corpus pending · coming soon
            </p>
          </div>
        </div>
      </header>
      <main className="mx-auto flex w-full max-w-7xl flex-1 items-start px-4 py-12">
        <div className="w-full rounded-2xl border border-zinc-800 bg-zinc-900/40 p-8">
          <div className="font-mono text-[11px] uppercase tracking-widest text-sky-400">
            object tracker
          </div>
          <h2 className="mt-3 text-2xl font-semibold tracking-tight text-zinc-50">
            Coming Soon
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-400">
            This lab is not seeded yet. Its paper corpus, evidence pipeline, and
            interactive sections will be built out here — fully separated from
            the other labs.
          </p>
        </div>
      </main>
    </div>
  );
}