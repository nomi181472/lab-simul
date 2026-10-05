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
    label: "Object Tracking",
    blurb: "159 papers · 2014–2026",
    accent: "sky",
  },
  {
    id: "wifi-sensing",
    label: "WiFi Sensing",
    blurb: "300 papers · citation-ordered",
    accent: "cyan",
  },
  {
    id: "neuroevolution",
    label: "Neuroevolution",
    blurb: "300 papers · evolved architectures",
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