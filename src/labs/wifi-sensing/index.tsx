"use client";

import { WifiLabProvider } from "./context";
import { WifiLabShell, ActiveWifiSection } from "./shell";

export function WifiSensingLab() {
  return (
    <WifiLabProvider>
      <WifiLabShell>
        <ActiveWifiSection />
      </WifiLabShell>
    </WifiLabProvider>
  );
}