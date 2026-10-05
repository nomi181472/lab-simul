"use client";

import { NeuroLabProvider } from "./context";
import { NeuroLabShell, ActiveNeuroSection } from "./shell";

export function NeuroevolutionLab() {
  return (
    <NeuroLabProvider>
      <NeuroLabShell>
        <ActiveNeuroSection />
      </NeuroLabShell>
    </NeuroLabProvider>
  );
}