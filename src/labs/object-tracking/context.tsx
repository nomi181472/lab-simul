"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { SectionId } from "@/labs/object-tracking/data/types";
import { useRoot } from "@/root/root-context";

const VALID_SECTIONS: SectionId[] = [
  "overview",
  "path",
  "timeline",
  "graph",
  "gaps",
  "foundations",
  "pipeline",
  "math",
  "papers",
  "compare",
  "datasets",
  "metrics",
  "audit",
];

export type TrackingLabContextValue = {
  section: SectionId;
  mathSim: string;
  selectedPaper: string | null;
  setSection: (s: SectionId) => void;
  setMathSim: (s: string) => void;
  setSelectedPaper: (id: string | null) => void;
};

function parseRest(rest: string): { section: SectionId; mathSim: string } {
  const raw = rest.replace(/^#/, "");
  if (raw.startsWith("math-")) {
    return { section: "math", mathSim: raw.slice(5) };
  }
  if (raw.startsWith("paper-")) {
    return { section: "papers", mathSim: "iou-track" };
  }
  if ((VALID_SECTIONS as string[]).includes(raw)) {
    return { section: raw as SectionId, mathSim: "iou-track" };
  }
  return { section: "overview", mathSim: "iou-track" };
}

const TrackingLabContext = createContext<TrackingLabContextValue | null>(null);

export function TrackingLabProvider({ children }: { children: ReactNode }) {
  const { rest, setRest } = useRoot();
  const [selectedPaper, setSelectedPaperState] = useState<string | null>(null);

  const { section, mathSim } = parseRest(rest);

  const setSection = useCallback(
    (s: SectionId) => {
      setRest(s === "overview" ? "" : s);
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

  const setSelectedPaper = useCallback((id: string | null) => {
    setSelectedPaperState(id);
  }, []);

  const value = useMemo(
    () => ({
      section,
      mathSim,
      selectedPaper,
      setSection,
      setMathSim,
      setSelectedPaper,
    }),
    [section, mathSim, selectedPaper, setSection, setMathSim, setSelectedPaper],
  );

  return (
    <TrackingLabContext.Provider value={value}>
      {children}
    </TrackingLabContext.Provider>
  );
}

export function useTrackingLab() {
  const ctx = useContext(TrackingLabContext);
  if (!ctx)
    throw new Error("useTrackingLab must be used within TrackingLabProvider");
  return ctx;
}
