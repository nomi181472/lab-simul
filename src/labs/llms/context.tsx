"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { SectionId } from "@/labs/llms/data/sections";
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

const DEFAULT_SIM = "attention-demo";

export type TfLabContextValue = {
  section: SectionId;
  sim: string;
  selectedPaper: string | null;
  /** Architecture view focus, kept in the URL so a diagram is linkable. */
  kindFilter: string | null;
  setSection: (s: SectionId) => void;
  setSim: (s: string) => void;
  setSelectedPaper: (id: string | null) => void;
  setKindFilter: (kind: string | null) => void;
};

function parseRest(rest: string): {
  section: SectionId;
  sim: string;
  kindFilter: string | null;
} {
  const raw = rest.replace(/^#/, "");
  if (raw.startsWith("paper-")) {
    return { section: "papers", sim: DEFAULT_SIM, kindFilter: null };
  }
  if (raw.startsWith("kind-")) {
    return { section: "architecture", sim: DEFAULT_SIM, kindFilter: raw.slice(5) };
  }
  if (raw.startsWith("mod-")) {
    return { section: "modification", sim: raw.slice(4) || DEFAULT_SIM, kindFilter: null };
  }
  if ((VALID_SECTIONS as string[]).includes(raw)) {
    return { section: raw as SectionId, sim: DEFAULT_SIM, kindFilter: null };
  }
  return { section: "overview", sim: DEFAULT_SIM, kindFilter: null };
}

const TfLabContext = createContext<TfLabContextValue | null>(null);

export function TfLabProvider({ children }: { children: ReactNode }) {
  const { rest, setRest } = useRoot();
  const [selectedPaper, setSelectedPaperState] = useState<string | null>(null);
  const { section, sim, kindFilter } = parseRest(rest);

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

  const setKindFilter = useCallback(
    (kind: string | null) => {
      setRest(kind ? `kind-${kind}` : "architecture");
    },
    [setRest],
  );

  const value = useMemo(
    () => ({
      section,
      sim,
      selectedPaper,
      kindFilter,
      setSection,
      setSim,
      setSelectedPaper,
      setKindFilter,
    }),
    [
      section,
      sim,
      selectedPaper,
      kindFilter,
      setSection,
      setSim,
      setSelectedPaper,
      setKindFilter,
    ],
  );

  return <TfLabContext.Provider value={value}>{children}</TfLabContext.Provider>;
}

export function useTfLab() {
  const ctx = useContext(TfLabContext);
  if (!ctx) throw new Error("useTfLab must be used within TfLabProvider");
  return ctx;
}