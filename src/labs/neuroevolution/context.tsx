"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { SectionId } from "@/labs/neuroevolution/data/types";
import { useRoot } from "@/root/root-context";

const VALID_SECTIONS: SectionId[] = [
  "overview",
  "architecture",
  "modification",
  "path",
  "timeline",
  "graph",
  "foundations",
  "operators",
  "papers",
  "compare",
  "datasets",
  "metrics",
  "audit",
];

export type NeuroLabContextValue = {
  section: SectionId;
  sim: string;
  selectedPaper: string | null;
  /** Architecture view focus, kept in the URL so a diagram is linkable. */
  locusFilter: string | null;
  setSection: (s: SectionId) => void;
  setSim: (s: string) => void;
  setSelectedPaper: (id: string | null) => void;
  setLocusFilter: (locus: string | null) => void;
};

function parseRest(rest: string): {
  section: SectionId;
  sim: string;
  locusFilter: string | null;
} {
  const raw = rest.replace(/^#/, "");
  if (raw.startsWith("paper-")) {
    return { section: "papers", sim: "ga-demo", locusFilter: null };
  }
  if (raw.startsWith("locus-")) {
    return { section: "architecture", sim: "ga-demo", locusFilter: raw.slice(6) };
  }
  if (raw.startsWith("mod-")) {
    return { section: "modification", sim: "ga-demo", locusFilter: null };
  }
  if ((VALID_SECTIONS as string[]).includes(raw)) {
    return { section: raw as SectionId, sim: "ga-demo", locusFilter: null };
  }
  return { section: "overview", sim: "ga-demo", locusFilter: null };
}

const NeuroLabContext = createContext<NeuroLabContextValue | null>(null);

export function NeuroLabProvider({ children }: { children: ReactNode }) {
  const { rest, setRest } = useRoot();
  const [selectedPaper, setSelectedPaperState] = useState<string | null>(null);
  const { section, sim, locusFilter } = parseRest(rest);

  const setSection = useCallback(
    (s: SectionId) => {
      setRest(s === "overview" ? "" : s);
      if (typeof window !== "undefined") {
        window.scrollTo({ top: 0, behavior: "auto" });
      }
    },
    [setRest],
  );

  const setSim = useCallback(
    (s: string) => {
      setRest(`mod-${s}`);
    },
    [setRest],
  );

  const setSelectedPaper = useCallback((id: string | null) => {
    setSelectedPaperState(id);
  }, []);

  const setLocusFilter = useCallback(
    (locus: string | null) => {
      setRest(locus ? `locus-${locus}` : "architecture");
    },
    [setRest],
  );

  const value = useMemo(
    () => ({
      section,
      sim,
      selectedPaper,
      locusFilter,
      setSection,
      setSim,
      setSelectedPaper,
      setLocusFilter,
    }),
    [
      section,
      sim,
      selectedPaper,
      locusFilter,
      setSection,
      setSim,
      setSelectedPaper,
      setLocusFilter,
    ],
  );

  return (
    <NeuroLabContext.Provider value={value}>{children}</NeuroLabContext.Provider>
  );
}

export function useNeuroLab() {
  const ctx = useContext(NeuroLabContext);
  if (!ctx) throw new Error("useNeuroLab must be used within NeuroLabProvider");
  return ctx;
}