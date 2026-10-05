"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { SectionId } from "@/labs/wifi-sensing/data/types";
import { useRoot } from "@/root/root-context";

const VALID_SECTIONS: SectionId[] = [
  "overview",
  "path",
  "timeline",
  "graph",
  "gaps",
  "foundations",
  "signal",
  "math",
  "papers",
  "compare",
  "datasets",
  "metrics",
  "audit",
];

export type WifiLabContextValue = {
  section: SectionId;
  mathSim: string;
  selectedPaper: string | null;
  setSection: (s: SectionId) => void;
  setMathSim: (s: string) => void;
  setSelectedPaper: (id: string | null) => void;
};

function parseRest(rest: string): { section: SectionId; mathSim: string } {
  const raw = rest.replace(/^#/, "");
  if (raw.startsWith("math-")) return { section: "math", mathSim: raw.slice(5) };
  if (raw.startsWith("paper-")) return { section: "papers", mathSim: "csi-multipath" };
  if ((VALID_SECTIONS as string[]).includes(raw)) {
    return { section: raw as SectionId, mathSim: "csi-multipath" };
  }
  return { section: "overview", mathSim: "csi-multipath" };
}

const WifiLabContext = createContext<WifiLabContextValue | null>(null);

export function WifiLabProvider({ children }: { children: ReactNode }) {
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

  return <WifiLabContext.Provider value={value}>{children}</WifiLabContext.Provider>;
}

export function useWifiLab() {
  const ctx = useContext(WifiLabContext);
  if (!ctx) throw new Error("useWifiLab must be used within WifiLabProvider");
  return ctx;
}