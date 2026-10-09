"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { SectionId } from "@/labs/object-detection/data/types";
import { useRoot } from "@/root/root-context";
import { SIM_IDS } from "@/labs/object-detection/sims/registry";

const VALID_SECTIONS: SectionId[] = [
  "overview",
  "year",
  "evolution",
  "problems",
  "clusters",
  "directions",
  "industry",
  "saturation",
  "benchmark",
  "explorer",
  "papers",
  "simulators",
  "analogy",
  "ask",
  "audit",
];

export type LabContextValue = {
  year: number;
  section: SectionId;
  sim: string;
  setYear: (y: number) => void;
  setSection: (s: SectionId) => void;
  setSim: (s: string) => void;
};

/** Parse the lab-local hash fragment (`sim-<id>` or `<section>`). */
function parseRest(rest: string): { section: SectionId; sim: string } {
  const raw = rest.replace(/^#/, "");
  if (raw.startsWith("sim-")) {
    const id = raw.slice(4);
    return { section: "simulators", sim: SIM_IDS.includes(id) ? id : "iou" };
  }
  if (VALID_SECTIONS.includes(raw as SectionId)) {
    return { section: raw as SectionId, sim: "iou" };
  }
  return { section: "overview", sim: "iou" };
}

const LabContext = createContext<LabContextValue | null>(null);

export function LabProvider({ children }: { children: ReactNode }) {
  const { rest, setRest } = useRoot();
  const [year, setYearState] = useState<number>(2020);

  // Section + active sim are derived from the root-owned hash fragment,
  // so there is no duplicated state to reconcile.
  const { section, sim } = parseRest(rest);

  const setSection = useCallback(
    (s: SectionId) => {
      setRest(s);
      if (typeof window !== "undefined") {
        window.scrollTo({ top: 0, behavior: "auto" });
      }
    },
    [setRest],
  );

  const setSim = useCallback((s: string) => setRest(`sim-${s}`), [setRest]);

  const setYear = useCallback((y: number) => setYearState(y), []);

  const value = useMemo(
    () => ({ year, section, sim, setYear, setSection, setSim }),
    [year, section, sim, setYear, setSection, setSim],
  );

  return <LabContext.Provider value={value}>{children}</LabContext.Provider>;
}

export function useLab() {
  const ctx = useContext(LabContext);
  if (!ctx) throw new Error("useLab must be used within LabProvider");
  return ctx;
}
