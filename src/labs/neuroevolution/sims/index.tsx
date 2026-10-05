"use client";

import type { ComponentType } from "react";
import {
  GaDemoSim,
  LocusDemoSim,
  QdDemoSim,
  NoveltyDemoSim,
  EncodingDemoSim,
  TopologyDemoSim,
  FitnessShapingSim,
  SelectionSim,
  EsGradSim,
  NasSearchSim,
} from "@/labs/neuroevolution/sims/modification";

/* Interactive simulator registry for the Neuroevolution lab.
 *
 * Every simulator is dependency-free (divs + SVG + range inputs) and fully
 * deterministic, so a shared link renders the same population, curve and genome
 * on every machine. The typed registry means a simulator cannot be added to the
 * component map without also appearing in the id list and the metadata table.
 *
 * These are the Modification section's simulators: each one shows a mechanism by
 * which a neural artefact gets changed, which is what the user needs in order to
 * read "how is the modification being done" in a paper.
 */

export const SIMULATOR_IDS = [
  "ga-demo",
  "locus-demo",
  "qd-demo",
  "novelty-demo",
  "encoding-demo",
  "topology-demo",
  "fitness-shaping",
  "selection-pressure",
  "es-gradient",
  "nas-budget",
];

export type SimulatorId = (typeof SIMULATOR_IDS)[number];

export const SIMULATORS: Record<SimulatorId, ComponentType> = {
  "ga-demo": GaDemoSim,
  "locus-demo": LocusDemoSim,
  "qd-demo": QdDemoSim,
  "novelty-demo": NoveltyDemoSim,
  "encoding-demo": EncodingDemoSim,
  "topology-demo": TopologyDemoSim,
  "fitness-shaping": FitnessShapingSim,
  "selection-pressure": SelectionSim,
  "es-gradient": EsGradSim,
  "nas-budget": NasSearchSim,
};

/** Which conceptual layer a simulator belongs to — used to group the picker. */
export const SIMULATOR_GROUP: Record<SimulatorId, string> = {
  "ga-demo": "Operators",
  "locus-demo": "Operators",
  "selection-pressure": "Operators",
  "fitness-shaping": "Selection",
  "encoding-demo": "Encoding",
  "topology-demo": "Encoding",
  "qd-demo": "Diversity",
  "novelty-demo": "Diversity",
  "es-gradient": "Basics",
  "nas-budget": "Applications",
};

export const SIMULATOR_META: { id: SimulatorId; label: string; blurb: string }[] = [
  { id: "ga-demo", label: "GA on widths", blurb: "Run a genetic algorithm over hidden widths under a compute budget." },
  { id: "locus-demo", label: "Single-locus mutation", blurb: "Apply one mutation sweep and see exactly which genes move." },
  { id: "qd-demo", label: "MAP-Elites archive", blurb: "Watch a quality-diversity archive fill one solution per behaviour cell." },
  { id: "novelty-demo", label: "Novelty pressure", blurb: "Score behavioural novelty against an archive instead of task reward." },
  { id: "encoding-demo", label: "Direct vs indirect", blurb: "Compare a per-weight genotype with a 3-gene generative rule." },
  { id: "topology-demo", label: "NEAT topology growth", blurb: "Grow a network by mutating node and link genes." },
  { id: "fitness-shaping", label: "Fitness shaping", blurb: "See a sparse reward stall the search where a smoothed one climbs." },
  { id: "selection-pressure", label: "Selection pressure", blurb: "Raise tournament size and watch population diversity collapse." },
  { id: "es-gradient", label: "ES gradient estimate", blurb: "Finite-difference perturbations recover a gradient estimate." },
  { id: "nas-budget", label: "NAS at fixed budget", blurb: "Compare evolutionary, random and RL architecture search at equal budget." },
];

export const DEFAULT_SIM: SimulatorId = "ga-demo";

export function SimShell({ id }: { id: string }) {
  const Sim = SIMULATORS[(SIMULATOR_IDS as string[]).includes(id) ? (id as SimulatorId) : DEFAULT_SIM];
  const meta = SIMULATOR_META.find((m) => m.id === id);
  return (
    <div className="flex flex-col gap-3">
      {meta ? (
        <div className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">
          {meta.label}
        </div>
      ) : null}
      <Sim />
    </div>
  );
}

/** Sim id -> paper ids it is an instance of. Validated by the audit view. */
export { SIM_BINDINGS } from "@/labs/neuroevolution/sims/papers";