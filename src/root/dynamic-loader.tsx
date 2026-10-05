"use client";

import dynamic from "next/dynamic";
import { useRoot } from "./root-context";
import type { RootLabId } from "./labs";

/** Each lab is its own webpack chunk, loaded only when its root tab is active. */
const ObjectDetectionLab = dynamic(
  () => import("@/labs/object-detection").then((m) => m.ObjectDetectionLab),
  { loading: RootTabLoading },
);
const ObjectTrackerLab = dynamic(
  () => import("@/labs/object-tracker").then((m) => m.ObjectTrackerLab),
  { loading: RootTabLoading },
);
const WifiSensingLab = dynamic(
  () => import("@/labs/wifi-sensing").then((m) => m.WifiSensingLab),
  { loading: RootTabLoading },
);
const NeuroevolutionLab = dynamic(
  () => import("@/labs/neuroevolution").then((m) => m.NeuroevolutionLab),
  { loading: RootTabLoading },
);
const LlmsLab = dynamic(
  () => import("@/labs/llms").then((m) => m.LlmsLab),
  { loading: RootTabLoading },
);

const LAB_COMPONENTS: Record<RootLabId, React.ComponentType> = {
  "object-detection": ObjectDetectionLab,
  "object-tracker": ObjectTrackerLab,
  "wifi-sensing": WifiSensingLab,
  neuroevolution: NeuroevolutionLab,
  llms: LlmsLab,
};

function RootTabLoading() {
  return (
    <div className="flex flex-1 items-center justify-center py-32 font-mono text-xs tracking-widest text-zinc-600">
      LOADING LAB…
    </div>
  );
}

export function ActiveLab() {
  const { lab } = useRoot();
  const Comp = LAB_COMPONENTS[lab];
  return <Comp />;
}