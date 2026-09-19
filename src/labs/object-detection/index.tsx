"use client";

import { LabProvider } from "./context";
import { LabShell, ActiveSection } from "./shell";

export function ObjectDetectionLab() {
  return (
    <LabProvider>
      <LabShell>
        <ActiveSection />
      </LabShell>
    </LabProvider>
  );
}