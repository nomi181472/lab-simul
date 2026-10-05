"use client";

import type { ComponentType } from "react";
import {
  FresnelZoneSim,
  PathLossSim,
  MultipathCsiSim,
  DopplerSim,
  RssiVsCsiSim,
  PhaseWrapSim,
} from "@/labs/wifi-sensing/sims/propagation";
import {
  StftVelocitySim,
  PeriodogramSim,
  FeatureFamiliesSim,
  FftInvarianceSim,
  Conv1dWindowSim,
  QuantizationNoiseSim,
  CfoSfoSim,
} from "@/labs/wifi-sensing/sims/features";
import {
  ActivityClfSim,
  RssiFingerprintSim,
  KalmanTrajectorySim,
  VitalSignsSim,
  SnrMarginSim,
} from "@/labs/wifi-sensing/sims/tasks";
import { DomainShiftSim, ThroughWallSim } from "@/labs/wifi-sensing/sims/robustness";

/* Interactive simulator registry for the WiFi sensing lab.
 *
 * Every sim is dependency-free (divs + SVG + range inputs) so the lab chunk
 * stays small. The registry is typed so a new simulator cannot be added to the
 * map without also appearing in the id list and the metadata table.
 *
 * The first simulator is the default landing tab.
 */

export const SIMULATOR_IDS = [
  // propagation
  "csi-multipath",
  "doppler",
  "rssi-vs-csi",
  "phase-wrap",
  "fresnel-zone",
  "path-loss",
  // front-end
  "stft-velocity",
  "periodogram",
  "feature-families",
  "fft-invariance",
  "conv1d-window",
  "quantization-noise",
  "cfo-sfo",
  // tasks
  "activity-clf",
  "rssi-fingerprint",
  "kalman-trajectory",
  "vital-signs",
  "snr-margin",
  // robustness
  "domain-shift",
  "through-wall",
] as const;

export type SimulatorId = (typeof SIMULATOR_IDS)[number];

export const SIMULATORS: Record<SimulatorId, ComponentType> = {
  "csi-multipath": MultipathCsiSim,
  doppler: DopplerSim,
  "rssi-vs-csi": RssiVsCsiSim,
  "phase-wrap": PhaseWrapSim,
  "fresnel-zone": FresnelZoneSim,
  "path-loss": PathLossSim,
  "stft-velocity": StftVelocitySim,
  periodogram: PeriodogramSim,
  "feature-families": FeatureFamiliesSim,
  "fft-invariance": FftInvarianceSim,
  "conv1d-window": Conv1dWindowSim,
  "quantization-noise": QuantizationNoiseSim,
  "cfo-sfo": CfoSfoSim,
  "activity-clf": ActivityClfSim,
  "rssi-fingerprint": RssiFingerprintSim,
  "kalman-trajectory": KalmanTrajectorySim,
  "vital-signs": VitalSignsSim,
  "snr-margin": SnrMarginSim,
  "domain-shift": DomainShiftSim,
  "through-wall": ThroughWallSim,
};

/** which conceptual layer a simulator belongs to — used to group the picker */
export const SIMULATOR_GROUP: Record<SimulatorId, string> = {
  "csi-multipath": "Propagation",
  doppler: "Propagation",
  "rssi-vs-csi": "Propagation",
  "phase-wrap": "Propagation",
  "fresnel-zone": "Propagation",
  "path-loss": "Propagation",
  "stft-velocity": "Signal front-end",
  periodogram: "Signal front-end",
  "feature-families": "Signal front-end",
  "fft-invariance": "Signal front-end",
  "conv1d-window": "Signal front-end",
  "quantization-noise": "Signal front-end",
  "cfo-sfo": "Signal front-end",
  "activity-clf": "Tasks",
  "rssi-fingerprint": "Tasks",
  "kalman-trajectory": "Tasks",
  "vital-signs": "Tasks",
  "snr-margin": "Tasks",
  "domain-shift": "Robustness",
  "through-wall": "Robustness",
};

export const SIMULATOR_META: { id: SimulatorId; label: string; blurb: string }[] = [
  {
    id: "csi-multipath",
    label: "CSI multipath",
    blurb: "H(f) = Σ αᵢe^(−j2πfτᵢ) across subcarriers",
  },
  { id: "doppler", label: "Doppler shift", blurb: "f_d = v cosθ f_c/c and its aliasing limit" },
  { id: "rssi-vs-csi", label: "RSSI vs CSI", blurb: "What the missing phase costs you" },
  { id: "phase-wrap", label: "Phase unwrap", blurb: "±π ambiguity and SFRA spikes" },
  { id: "fresnel-zone", label: "Fresnel zone", blurb: "Clearance needed for a live link" },
  { id: "path-loss", label: "Path loss", blurb: "PL = 10n log₁₀(d/d₀) per environment" },
  { id: "stft-velocity", label: "STFT resolution", blurb: "Window length vs velocity accuracy" },
  { id: "periodogram", label: "Periodogram", blurb: "Dominant bin → a rate" },
  { id: "feature-families", label: "Feature families", blurb: "RSS / FCS / DFR / CCR / entropy" },
  { id: "fft-invariance", label: "FFT invariance", blurb: "Ignoring where the window starts" },
  { id: "conv1d-window", label: "Conv1D window", blurb: "Receptive field in seconds" },
  { id: "quantization-noise", label: "Quantization", blurb: "NIC bits and the phase blow-up" },
  { id: "cfo-sfo", label: "CFO / SFO", blurb: "Clock error that fakes a vital sign" },
  { id: "activity-clf", label: "Activity classes", blurb: "Separability sets the ceiling" },
  { id: "rssi-fingerprint", label: "RSSI fingerprint", blurb: "k-NN on an RSSI map" },
  { id: "kalman-trajectory", label: "Kalman track", blurb: "Q/R smoothness vs lag" },
  { id: "vital-signs", label: "Vital signs", blurb: "Breathing from phase, below noise" },
  { id: "snr-margin", label: "SNR margin", blurb: "Detection rate as an SNR statement" },
  { id: "domain-shift", label: "Cross-domain", blurb: "Trained in one room, tested in another" },
  { id: "through-wall", label: "Through-wall", blurb: "Range as a dB budget" },
];

export const DEFAULT_SIM: SimulatorId = "csi-multipath";

export function SimShell({ id }: { id: string }) {
  const Comp = (SIMULATORS as Record<string, ComponentType>)[id] ?? SIMULATORS[DEFAULT_SIM];
  return <Comp />;
}