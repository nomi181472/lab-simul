export type Accent =
  | "emerald"
  | "sky"
  | "amber"
  | "violet"
  | "rose"
  | "cyan";

/**
 * Root-tab registry. Metadata only — never imports lab code, so the root
 * shell stays decoupled from each lab's bundle.
 */
export const LABS = [
  {
    id: "object-detection",
    label: "Object Detection",
    blurb: "48 papers · 2015–2026",
    accent: "emerald",
  },
  {
    id: "object-tracker",
    label: "Object Tracker",
    blurb: "coming soon",
    accent: "sky",
  },
  {
    id: "evolutionary",
    label: "Evolutionary",
    blurb: "coming soon",
    accent: "amber",
  },
  {
    id: "nature-inspired",
    label: "Nature-Inspired Algorithms",
    blurb: "coming soon",
    accent: "violet",
  },
  {
    id: "neuroevolution",
    label: "Neuroevolution",
    blurb: "coming soon",
    accent: "rose",
  },
  {
    id: "llms",
    label: "LLMs",
    blurb: "coming soon",
    accent: "cyan",
  },
] as const;

export type RootLabId = (typeof LABS)[number]["id"];

export type RootLabMeta = {
  id: RootLabId;
  label: string;
  blurb: string;
  accent: Accent;
};

export const LAB_IDS: RootLabId[] = LABS.map((l) => l.id);

export const DEFAULT_LAB: RootLabId = "object-detection";