"use client";

import { NetworkingProvider } from "@/labs/networking/context";
import { NetworkingShell, ActiveNetworkingSection } from "@/labs/networking/shell";

export function NetworkingLab() {
  return (
    <NetworkingProvider>
      <NetworkingShell>
        <ActiveNetworkingSection />
      </NetworkingShell>
    </NetworkingProvider>
  );
}
