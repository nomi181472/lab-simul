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

const VALID_SECTIONS: SectionId[] = [
  "overview",
  "year",
  "evolution",
  "problems",
  "directions",
  "applications",
  "benchmark",
  "explorer",
  "ask",
  "missing",
  "papers",
  "math",
  "edge",
  "audit",
];

export type LabContextValue = {
  year: number;
  section: SectionId;
  mathSim: string;
  setYear: (y: number) => void;
  setSection: (s: SectionId) => void;
  setMathSim: (s: string) => void;
};

/** Parse the lab-local hash fragment (`math-<sim>` or `<section>`) into (section, mathSim). */
function parseRest(rest: string): { section: SectionId; mathSim: string } {
  const raw = rest.replace(/^#/, "");
  if (raw.startsWith("math-")) {
    return { section: "math", mathSim: raw.slice(5) };
  }
  if (VALID_SECTIONS.includes(raw as SectionId)) {
    return { section: raw as SectionId, mathSim: "iou" };
  }
  return { section: "overview", mathSim: "iou" };
}

const LabContext = createContext<LabContextValue | null>(null);

export function LabProvider({ children }: { children: ReactNode }) {
  const { rest, setRest } = useRoot();
  const [year, setYearState] = useState<number>(2020);

  // Section + math simulation are derived from the root-owned hash fragment,
  // so there is no duplicated state to reconcile.
  const { section, mathSim } = parseRest(rest);

  const setSection = useCallback(
    (s: SectionId) => {
      setRest(s);
      if (typeof window !== "undefined") {
        window.scrollTo({ top: 0, behavior: "auto" });
      }
    },
    [setRest],
  );

  const setMathSim = useCallback(
    (s: string) => {
      setRest(`math-${s}`);
    },
    [setRest],
  );

  const setYear = useCallback((y: number) => setYearState(y), []);

  const value = useMemo(
    () => ({ year, section, mathSim, setYear, setSection, setMathSim }),
    [year, section, mathSim, setYear, setSection, setMathSim],
  );

  return <LabContext.Provider value={value}>{children}</LabContext.Provider>;
}

export function useLab() {
  const ctx = useContext(LabContext);
  if (!ctx) throw new Error("useLab must be used within LabProvider");
  return ctx;
}