"use client";

import { TfLabProvider } from "./context";
import { TfLabShell, ActiveTfSection } from "./shell";

/**
 * Exported as `LlmsLab` because that is the id this lab has carried in
 * src/root/labs.ts since the placeholder; the corpus it now hosts is the
 * transformer-to-micro-LLM arc rather than LLMs alone.
 */
export function LlmsLab() {
  return (
    <TfLabProvider>
      <TfLabShell>
        <ActiveTfSection />
      </TfLabShell>
    </TfLabProvider>
  );
}