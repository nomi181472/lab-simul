"use client";

import { Component, type ReactNode } from "react";

type Props = { children: ReactNode; label?: string };
type State = { error: Error | null };

/** Catches a crashing simulator and keeps the rest of the lab alive.
 *  The previous IoU-loss key bug crashed the whole page on tab switch;
 *  with a boundary the sim degrades to a visible fallback instead. */
export class SimErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="rounded-xl border border-rose-800/60 bg-rose-950/20 p-4">
          <div className="font-mono text-[10px] uppercase tracking-wide text-rose-400">
            {this.props.label ?? "simulator"} — render error
          </div>
          <p className="mt-2 font-mono text-[12px] leading-5 text-rose-200">
            {this.state.error.message}
          </p>
          <p className="mt-1.5 text-[11px] leading-4 text-zinc-500">
            The lab stays intact. This sim could not render; refresh and switch to another
            tab. No claim about the corpus is affected.
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}