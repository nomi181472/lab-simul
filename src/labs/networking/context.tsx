"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { SectionId } from "@/labs/networking/data/sections";
import { useRoot } from "@/root/root-context";

const VALID_SECTIONS: SectionId[] = [
  "overview", "corpus", "years", "evolution", "matrix", "clusters",
  "industry", "future", "benchmarks", "saturation", "gaps", "adoption",
  "simulators", "analogies", "audit",
];

export type NetworkingContextValue = {
  section: SectionId;
  selectedPaper: string | null;
  setSection: (s: SectionId) => void;
  setSelectedPaper: (id: string | null) => void;
};

function parseSection(rest: string): SectionId {
  const raw = rest.replace(/^#/, "").replace(/^sim-/, "");
  if ((VALID_SECTIONS as string[]).includes(raw)) return raw as SectionId;
  if (raw.startsWith("paper-")) return "corpus";
  return "overview";
}

const NetworkingContext = createContext<NetworkingContextValue | null>(null);

export function NetworkingProvider({ children }: { children: ReactNode }) {
  const { rest, setRest } = useRoot();
  const [selectedPaper, setSelectedPaperState] = useState<string | null>(null);
  const section = parseSection(rest);

  const setSection = useCallback(
    (s: SectionId) => {
      setRest(s === "overview" ? "" : s);
      if (typeof window !== "undefined") {
        window.scrollTo({ top: 0, behavior: "auto" });
      }
    },
    [setRest],
  );

  const setSelectedPaper = useCallback((id: string | null) => {
    setSelectedPaperState(id);
  }, []);

  const value = useMemo(
    () => ({ section, selectedPaper, setSection, setSelectedPaper }),
    [section, selectedPaper, setSection, setSelectedPaper],
  );

  return (
    <NetworkingContext.Provider value={value}>{children}</NetworkingContext.Provider>
  );
}

export function useNetworking() {
  const ctx = useContext(NetworkingContext);
  if (!ctx) throw new Error("useNetworking must be used within NetworkingProvider");
  return ctx;
}
