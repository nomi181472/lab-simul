/* Sim → corpus binding.
 *
 * The repo convention is an inline `papers={[...]}` array on each <Sim>, which
 * `runAudit()` validates. This file is the single binding point for all 20
 * simulators: it keeps one edit in one place instead of 20 scattered arrays,
 * and the audit still enforces that every id here resolves to a corpus paper.
 *
 * `note` states what the sim is an instance of, so the binding can be checked
 * against the paper it names rather than trusted blindly.
 */

export type SimPaperBinding = {
  /** paper ids this simulator is grounded in */
  papers: string[];
  /** what those papers do that the sim reproduces */
  note: string;
};

export const SIM_PAPERS: Record<string, SimPaperBinding> = {
  // ---------- propagation ----------
  "fresnel-zone": {
    papers: [
      "W007",
      "W009",
      "W010",
      "W011",
    ],
    note: "Fresnel clearance condition for a radio link; obstruction inside the first Fresnel zone drives the reflection model.",
  },
  "path-loss": {
    papers: [
      "W001",
      "W023",
      "W029",
      "W031",
    ],
    note: "Log-distance path-loss exponent per environment; sets the usable sensing range.",
  },
  "csi-multipath": {
    papers: [
      "W001",
      "W002",
      "W003",
      "W004",
    ],
    note: "Channel frequency response as a coherent sum of delayed paths; the basis of CSI-based sensing.",
  },
  doppler: {
    papers: [
      "W001",
      "W004",
      "W007",
      "W014",
    ],
    note: "Doppler frequency shift of the moving path, used for velocity estimation.",
  },
  "rssi-vs-csi": {
    papers: [
      "W001",
      "W002",
      "W004",
      "W007",
    ],
    note: "RSSI discards phase, CSI keeps per-subcarrier complex response; motivates CSI over RSSI.",
  },
  "phase-wrap": {
    papers: [
      "W048",
      "W061",
      "W088",
      "W097",
    ],
    note: "CSI phase is reported modulo 2pi; unwrapping and FiLiMS sanitization are standard preprocessing steps.",
  },

  // ---------- front-end ----------
  "stft-velocity": {
    papers: [
      "W007",
      "W011",
      "W015",
      "W016",
    ],
    note: "Time-frequency resolution trade-off that sets Doppler/velocity estimation accuracy.",
  },
  "periodogram": {
    papers: [
      "W067",
      "W094",
      "W138",
      "W151",
    ],
    note: "Dominant-bin frequency estimation used to convert motion to a rate.",
  },
  "feature-families": {
    papers: [
      "W001",
      "W007",
      "W036",
      "W048",
    ],
    note: "Hand-crafted feature sets (RSS, DFR, FCS) for training-free activity recognition.",
  },
  "fft-invariance": {
    papers: [
      "W011",
      "W012",
      "W015",
      "W016",
    ],
    note: "FFT-domain features make recognition robust to the unknown start time of a window.",
  },
  "conv1d-window": {
    papers: [
      "W002",
      "W004",
      "W006",
      "W007",
    ],
    note: "Temporal receptive field of 1-D convolutions over CSI packet streams.",
  },
  "quantization-noise": {
    papers: [
      "W007",
      "W010",
      "W011",
      "W031",
    ],
    note: "NIC bit-width limits on reported CSI amplitude and phase.",
  },
  "cfo-sfo": {
    papers: [
      "W017",
      "W031",
      "W032",
      "W040",
    ],
    note: "Carrier and sampling frequency offset appear as phase drift that must be corrected before Doppler work.",
  },

  // ---------- tasks ----------
  "activity-clf": {
    papers: [
      "W004",
      "W005",
      "W007",
      "W008",
    ],
    note: "Class separability determines activity-recognition accuracy.",
  },
  "rssi-fingerprint": {
    papers: [
      "W001",
      "W002",
      "W004",
      "W005",
    ],
    note: "k-NN over RSSI fingerprints; the classical WiFi localization baseline.",
  },
  "kalman-trajectory": {
    papers: [
      "W001",
      "W002",
      "W013",
      "W021",
    ],
    note: "Smoothing noisy position estimates derived from CSI features.",
  },
  "vital-signs": {
    papers: [
      "W005",
      "W007",
      "W008",
      "W009",
    ],
    note: "Chest displacement modulates CSI phase; spectral estimation recovers the respiration rate.",
  },
  "snr-margin": {
    papers: [
      "W001",
      "W003",
      "W006",
      "W009",
    ],
    note: "Detection reliability as a function of feature SNR.",
  },

  // ---------- robustness ----------
  "domain-shift": {
    papers: [
      "W001",
      "W007",
      "W008",
      "W010",
    ],
    note: "CSI is device- and environment-specific, so cross-domain accuracy collapses without adaptation.",
  },
  "through-wall": {
    papers: [
      "W001",
      "W003",
      "W004",
      "W005",
    ],
    note: "Per-wall attenuation bounds through-wall sensing range.",
  },
};

/** Simulator id → display name. Must stay in sync with the SIMULATORS registry;
 *  `checkSim` uses this to validate the URL hash. */
export const SIMULATOR_INDEX: Record<string, string> = {
  "fresnel-zone": "Fresnel Zone",
  "path-loss": "Path Loss",
  "csi-multipath": "CSI Multipath",
  doppler: "Doppler Shift",
  "rssi-vs-csi": "RSSI vs CSI",
  "phase-wrap": "Phase Wrap & Unwrap",
  "stft-velocity": "STFT Velocity Resolution",
  periodogram: "Periodogram Peak",
  "feature-families": "Feature Families",
  "fft-invariance": "FFT Shift Invariance",
  "conv1d-window": "Conv1D Receptive Field",
  "quantization-noise": "Quantization Noise",
  "cfo-sfo": "CFO / SFO Drift",
  "activity-clf": "Activity Classification",
  "rssi-fingerprint": "RSSI Fingerprint Localization",
  "kalman-trajectory": "Kalman Trajectory",
  "vital-signs": "Vital Signs",
  "snr-margin": "SNR Margin",
  "domain-shift": "Cross-Domain Shift",
  "through-wall": "Through-Wall Attenuation",
};

export function checkSim(sim?: string): string | null {
  if (!sim) return null;
  return SIMULATOR_INDEX[sim] ? sim : null;
}
