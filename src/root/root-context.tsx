"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  type ReactNode,
  useSyncExternalStore,
} from "react";
import { DEFAULT_LAB, LAB_IDS, type RootLabId } from "./labs";

export type RootContextValue = {
  lab: RootLabId;
  /** hash fragment after `#<lab>/`; owned per-lab (each lab parses its own rest) */
  rest: string;
  setLab: (id: RootLabId) => void;
  setRest: (rest: string) => void;
};

const RootContext = createContext<RootContextValue | null>(null);

function subscribeToHash(onChange: () => void) {
  window.addEventListener("hashchange", onChange);
  return () => window.removeEventListener("hashchange", onChange);
}

function getHash() {
  return typeof window === "undefined" ? "" : window.location.hash;
}

function getServerHash() {
  return "";
}

/** Parse `#<lab>/<rest>`; legacy `#<section>` / `#math-<sim>` resolve to the default lab. */
function parseHash(hash: string): { lab: RootLabId; rest: string } {
  const raw = hash.replace(/^#/, "");
  const [head, ...tail] = raw.split("/");
  const rest = tail.join("/");
  if ((LAB_IDS as readonly string[]).includes(head)) {
    return { lab: head as RootLabId, rest };
  }
  return { lab: DEFAULT_LAB, rest: raw };
}

function writeHash(lab: RootLabId, rest: string) {
  if (typeof window === "undefined") return;
  const target = rest ? `#${lab}/${rest}` : `#${lab}`;
  if (window.location.hash !== target) {
    window.history.replaceState(null, "", target);
    window.dispatchEvent(new Event("hashchange"));
  }
}

export function RootProvider({ children }: { children: ReactNode }) {
  const hash = useSyncExternalStore(subscribeToHash, getHash, getServerHash);
  const { lab, rest } = parseHash(hash);

  // keep the URL canonical for the current (lab, rest)
  useEffect(() => {
    writeHash(lab, rest);
  }, [lab, rest]);

  const setLab = useCallback((id: RootLabId) => {
    writeHash(id, "");
    if (typeof window !== "undefined") {
      window.scrollTo({ top: 0, behavior: "auto" });
    }
  }, []);

  const setRest = useCallback((next: string) => {
    if (typeof window === "undefined") return;
    const current = parseHash(window.location.hash);
    writeHash(current.lab, next);
  }, []);

  const value = useMemo(
    () => ({ lab, rest, setLab, setRest }),
    [lab, rest, setLab, setRest],
  );

  return <RootContext.Provider value={value}>{children}</RootContext.Provider>;
}

export function useRoot() {
  const ctx = useContext(RootContext);
  if (!ctx) throw new Error("useRoot must be used within RootProvider");
  return ctx;
}