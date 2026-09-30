"use client";

import { TrackingLabProvider } from "./context";
import { TrackingLabShell, ActiveTrackingSection } from "./shell";

export function ObjectTrackingLab() {
  return (
    <TrackingLabProvider>
      <TrackingLabShell>
        <ActiveTrackingSection />
      </TrackingLabShell>
    </TrackingLabProvider>
  );
}
