"use client";

import { useMemo, useState } from "react";
import {
  PATH_LOSS_EXPONENTS,
  csiPacket,
  dopplerHz,
  excessTau,
  fresnelZone,
  maxUnambiguousVelocity,
  pathLossDb,
  sanitizeFiLiMS,
  unwrapSeries,
  velocityFromDoppler,
  wrapPhase,
  type Complex,
} from "@/labs/wifi-sensing/sim/math";
import {
  makeGeometry,
  mags,
  phases,
  scenePaths,
  walkTrajectory,
} from "@/labs/wifi-sensing/sim/mock";
import { Slider, Segmented, Sim } from "./shared";
import { CsiPlot, LinePlot, RoomPlan, Stat } from "./canvas";
import { SIM_PAPERS } from "./papers";

/* ============================ Fresnel zone ============================ */

export function FresnelZoneSim() {
  const [l0, setL0] = useState(6);
  const [clearance, setClearance] = useState(0.6);
  const zone = useMemo(() => fresnelZone(l0, clearance), [l0, clearance]);

  return (
    <Sim
      name="Fresnel zone (obstruction geometry)"
      identifies={`A ${l0} m link with a ${(clearance * 100).toFixed(0)}% clearance requirement: the first Fresnel zone bulges to ${zone.radius.toFixed(2)} m radius at ${(l0 / 2).toFixed(1)} m from the transmitter.`}
      math={[
        "r₁ = √( λ·d₁·d₂ / (d₁+d₂) )      (first Fresnel zone radius at the link midpoint)",
        "clearance requirement: the line TX–RX must stay ε·r₁ away from every obstruction",
        "ε = 0.6 is the classic engineering threshold; below it the direct path is treated as intact",
      ]}
      need="A WiFi link does not have to be blocked to fail — enough of the Fresnel zone being obstructed turns the direct ray into a reflection and changes the channel. This is the geometry behind every through-wall result."
      real="Through-wall and NLOS sensing papers all rest on this: a wall does not need to absorb the signal, it only needs to intrude on the clearance region."
      papers={SIM_PAPERS["fresnel-zone"].papers}
      params={
        <>
          <Slider label="link distance" value={l0} min={1} max={20} step={0.5} onChange={setL0} fmt={(v) => `${v.toFixed(1)} m`} />
          <Slider label="clearance ε" value={clearance} min={0} max={0.9} step={0.05} onChange={setClearance} fmt={(v) => v.toFixed(2)} />
        </>
      }
    >
      <RoomPlan
        tx={{ x: 0, y: 0 }}
        rx={{ x: l0, y: 0 }}
        zone={{ d1: zone.d1, d2: zone.d2, radius: zone.radius }}
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="zone radius" value={`${zone.radius.toFixed(2)} m`} tone="cyan" />
        <Stat label="ε·r₁ clearance" value={`${(clearance * zone.radius).toFixed(2)} m`} />
        <Stat label="d₁ (zone apex)" value={`${zone.d1.toFixed(2)} m`} />
        <Stat label="wavelength" value={`${(299792458 / 2.4e9 * 100).toFixed(2)} cm`} hint="2.4 GHz" />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Double the link distance and the clearance radius grows as √d — a 6 m link needs only a
        small gap, but a 20 m link needs twice the room. That is why long links are so much
        more fragile to an obstruction, and why a body standing between TX and RX degrades the
        link before it blocks it.
      </p>
    </Sim>
  );
}

/* ============================== Path loss ============================== */

export function PathLossSim() {
  const [d, setD] = useState(4);
  const [sel, setSel] = useState(1);
  const txDbm = -10;
  const noiseDbm = -92;
  const ex = PATH_LOSS_EXPONENTS[sel];

  const curves = useMemo(
    () =>
      PATH_LOSS_EXPONENTS.map((e) => ({
        label: e.label,
        color: e.label === ex.label ? "#22d3ee" : "#52525b",
        ys: Array.from({ length: 41 }, (_, i) => pathLossDb(i * 0.5, e.n)),
      })),
    [ex.label],
  );

  const pl = pathLossDb(d, ex.n);
  const rx = txDbm - pl;
  const snr = rx - noiseDbm;

  return (
    <Sim
      name="Log-distance path loss"
      identifies={`${ex.label} (n = ${ex.n}) at ${d.toFixed(1)} m costs ${pl.toFixed(1)} dB, leaving ${rx.toFixed(1)} dBm received and ${snr.toFixed(1)} dB SNR against a −92 dBm noise floor.`}
      math={[
        "PL(d) = PL(d₀) + 10·n·log₁₀(d / d₀)",
        "PL(d₀) = 40 dB reference at d₀ = 1 m;  n = path-loss exponent",
        "SNR = P_tx − PL(d) − N,   with P_tx = −10 dBm, N = −92 dBm",
      ]}
      need="The exponent n is the only thing separating environments. Going from free space (n=2) to two concrete walls (n≈5.2) costs about 25 dB at 6 m — the difference between a usable link and an unusable one."
      real="Sensing-range and deployment claims are all path-loss arithmetic. A paper that detects a person at 8 m through a wall is asserting a specific n for that wall."
      papers={SIM_PAPERS["path-loss"].papers}
      params={
        <>
          <Slider label="distance" value={d} min={0.5} max={20} step={0.5} onChange={setD} fmt={(v) => `${v.toFixed(1)} m`} />
          <Segmented
            label="environment"
            value={String(sel)}
            onChange={(v) => setSel(Number(v))}
            options={PATH_LOSS_EXPONENTS.map((e, i) => ({ value: String(i), label: e.label }))}
          />
        </>
      }
    >
      <LinePlot
        series={curves}
        yLabel="dB lost"
        xLabel="distance (m)"
        markers={[{ i: d / 0.5, color: "#fbbf24", label: `${d.toFixed(1)} m` }]}
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="path loss" value={`${pl.toFixed(1)} dB`} tone="amber" />
        <Stat label="received" value={`${rx.toFixed(1)} dBm`} />
        <Stat label="SNR" value={`${snr.toFixed(1)} dB`} tone={snr > 15 ? "emerald" : snr > 5 ? "amber" : "rose"} />
        <Stat label="range @ 6 dB SNR" value={`${(Math.pow(10, (txDbm - noiseDbm - 6) / 10 / ex.n)).toFixed(1)} m`} hint="where SNR falls to the 6 dB usable margin" />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Drag the distance slider: each curve is 20·log₁₀ faster or slower than free space. Switch
        environment to see how quickly the same room becomes unusable. The gap between the
        free-space and two-wall curves at 20 m is over 40 dB.
      </p>
    </Sim>
  );
}

/* ============================ CSI multipath ============================ */

export function MultipathCsiSim() {
  const [d, setD] = useState(3);
  const [dynamicAmp, setDynamicAmp] = useState(1);
  const [showStatic, setShowStatic] = useState(true);
  const g = useMemo(() => makeGeometry(5), []);

  const cs = useMemo(() => {
    const paths = scenePaths(g, d, dynamicAmp, showStatic ? 0.6 : 0);
    return csiPacket(paths);
  }, [g, d, dynamicAmp, showStatic]);

  const tau = excessTau(d, g.height, g.l0);

  return (
    <Sim
      name="CSI as a coherent sum of delayed paths"
      identifies={`A target at ${d.toFixed(1)} m gives excess path ${(tau * 1e9).toFixed(2)} ns, rotating each subcarrier's phase by ${((2 * Math.PI * 312500 * tau * 1e9) % 360).toFixed(0)}° relative to the static path.`}
      math={[
        "H(f) = Σᵢ αᵢ · e^(−j2πfτᵢ)      (channel frequency response)",
        "τᵢ = extra path length / c        (excess delay)",
        "CSI per subcarrier k:  Hₖ = H(k · 312.5 kHz)   for 20 MHz OFDM",
      ]}
      need="This single sum is the entire physics of WiFi sensing. A person is one extra αᵢ with a changing τᵢ. Move the target and every subcarrier's complex value moves — that motion is the entire input to every classifier in the corpus."
      real="Every CSI-based paper is, underneath, a method for estimating how one αᵢ and τᵢ changed. The differences between papers are mostly what front-end they use to read it out."
      papers={SIM_PAPERS["csi-multipath"].papers}
      params={
        <>
          <Slider label="target distance" value={d} min={0.5} max={9} step={0.1} onChange={setD} fmt={(v) => `${v.toFixed(1)} m`} />
          <Slider label="dynamic path gain" value={dynamicAmp} min={0} max={3} step={0.1} onChange={setDynamicAmp} />
          <Segmented
            label="static clutter"
            value={showStatic ? "on" : "off"}
            onChange={(v) => setShowStatic(v === "on")}
            options={[
              { value: "on", label: "with walls" },
              { value: "off", label: "free space only" },
            ]}
          />
        </>
      }
    >
      <CsiPlot cs={cs} xLabel="subcarrier index k" />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="excess delay τ" value={`${(tau * 1e9).toFixed(2)} ns`} tone="cyan" />
        <Stat label="|H| at k=16" value={mags(cs)[15].toFixed(3)} />
        <Stat label="phase at k=16" value={`${((phases(cs)[15] * 180) / Math.PI).toFixed(0)}°`} />
        <Stat label="subcarriers" value={String(cs.length)} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Turn static clutter off: the per-subcarrier ripple collapses to a single rotating phase
        ramp. Turn the dynamic gain to zero: the channel is frozen no matter where the target is.
        The wiggle across k in the magnitude plot is pure multipath — it is the reason CSI
        carries information RSSI threw away.
      </p>
    </Sim>
  );
}

/* =============================== Doppler =============================== */

export function DopplerSim() {
  const [velocity, setVelocity] = useState(1);
  const [cosTheta, setCosTheta] = useState(1);
  const [rate, setRate] = useState(100);
  const fs = rate;
  const vmax = maxUnambiguousVelocity(fs);

  const traj = useMemo(
    () => walkTrajectory(fs, 4, 2, velocity, 1, 8),
    [fs, velocity],
  );
  const fds = useMemo(
    () =>
      traj.map((_, i) => {
        const dt = traj[Math.min(i + 1, traj.length - 1)] - traj[Math.max(i - 1, 0)];
        const dtSec = 2 / fs;
        const v = dt / dtSec; // m/s, negative when receding
        const fd = dopplerHz(Math.abs(v), cosTheta);
        // alias back into [-fs/2, fs/2]
        const alias = ((fd + fs / 2) % fs + fs) % fs - fs / 2;
        return alias;
      }),
    [traj, fs, cosTheta],
  );

  const steadyFd = dopplerHz(velocity, cosTheta);
  const implied = velocityFromDoppler(steadyFd, cosTheta);
  const aliased = velocity > vmax;

  return (
    <Sim
      name="Doppler shift and its aliasing limit"
      identifies={`${velocity.toFixed(2)} m/s at cosθ=${cosTheta.toFixed(2)} gives ${steadyFd.toFixed(1)} Hz. Nyquist at ${fs} Hz packet rate caps observable speed at ${vmax.toFixed(3)} m/s — this target ${aliased ? "exceeds" : "fits"} that.`}
      math={[
        "f_d = v · cosθ · f_c / c        (f_c = 2.4 GHz, c = 3×10⁸ m/s)",
        "observable only while |f_d| < f_s/2   where f_s = packet rate",
        "⇒ v_max = (f_s/2) · c / f_c      e.g. 100 Hz packets ⇒ 6.25 m/s",
        "v̂ = f_d · c / (cosθ · f_c)        (inverse, used to report velocity)",
      ]}
      need="One packet is a snapshot; a target only becomes visible as a frequency. The same equation that estimates velocity also imposes a hard speed limit, and the limit is set by the sampling rate, not the radio."
      real="Sampling-rate choice in WiFi sensing papers is a Doppler-aliasing decision. At 100 Hz packets a normal walk already saturates; papers reporting walking velocity often raise the packet rate or restrict to slow motion."
      papers={SIM_PAPERS["doppler"].papers}
      params={
        <>
          <Slider label="velocity" value={velocity} min={0} max={2} step={0.05} onChange={setVelocity} fmt={(v) => `${v.toFixed(2)} m/s`} />
          <Slider label="cosθ (path projection)" value={cosTheta} min={0} max={1} step={0.05} onChange={setCosTheta} />
          <Slider label="packet rate" value={rate} min={20} max={400} step={10} onChange={setRate} fmt={(v) => `${v} Hz`} />
        </>
      }
    >
      <LinePlot
        series={[
          { ys: fds, color: "#22d3ee", label: "observed f_d (aliased)" },
          { ys: fds.map(() => steadyFd), color: "#34d399", dashed: true, label: "true f_d" },
        ]}
        yLabel="Hz"
        xLabel="packet index"
        zeroLine
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="true f_d" value={`${steadyFd.toFixed(1)} Hz`} tone="emerald" />
        <Stat label="implied v̂" value={`${implied.toFixed(2)} m/s`} />
        <Stat label="v_max" value={`${vmax.toFixed(3)} m/s`} tone={aliased ? "rose" : "zinc"} />
        <Stat label="regime" value={aliased ? "ALIASED" : "resolved"} tone={aliased ? "rose" : "emerald"} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Raise velocity past v_max and the observed trace stops following the true line — it folds
        back, because the aliased value lands on a plausible-looking frequency. Drop cosθ toward
        zero and the shift vanishes entirely: a target moving perpendicular to the link is
        invisible in Doppler no matter how fast it moves. This is the single most common failure
        mode in WiFi sensing evaluations.
      </p>
    </Sim>
  );
}

/* ============================ RSSI vs CSI ============================== */

export function RssiVsCsiSim() {
  const [noiseSd, setNoiseSd] = useState(0.02);
  const [velocity, setVelocity] = useState(0.6);
  const g = useMemo(() => makeGeometry(5), []);
  const fs = 50;

  const { rssiSeries, csiMag } = useMemo(() => {
    const traj = walkTrajectory(fs, 4, 2, velocity, 1, 8);
    let a = 0; // deterministic pseudo-noise
    const rnd = () => {
      a = (a * 1103515245 + 12345) % 2147483648;
      return a / 2147483648 - 0.5;
    };
    const r: number[] = [];
    const c: number[] = [];
    for (const d of traj) {
      const paths = scenePaths(g, d, 1, 0.6);
      const h: Complex = csiPacket(paths)[15];
      r.push(Math.hypot(h.re, h.im) + rnd() * 2 * noiseSd);
      c.push(Math.hypot(h.re, h.im));
    }
    return { rssiSeries: r, csiMag: c };
  }, [g, velocity, noiseSd, fs]);

  const rssiVar = rssiSeries.reduce(
    (s, _, i, a) => s + (a[i] - rssiSeries.reduce((x, y) => x + y, 0) / a.length) ** 2,
    0,
  ) / rssiSeries.length;

  return (
    <Sim
      name="RSSI discards phase; CSI does not"
      identifies={`Same walk, two receivers. RSSI magnitude of the summed channel varies by ${(Math.sqrt(rssiVar)).toFixed(3)} around its mean, while the CSI subcarrier keeps the coherent structure — but RSSI's phase is gone before the digitizer sees it.`}
      math={[
        "RSSI reports |H(f)| only — the phase term e^(−j2πfτ) is discarded",
        "|H| = | Σᵢ αᵢ e^(−j2πfτᵢ) |    (a scalar: constructive/destructive cancellation)",
        "CSI reports the complex H per subcarrier, so the interference pattern survives",
      ]}
      need="RSSI is one number per packet; CSI is 30 complex numbers. Because RSSI sums all paths before taking magnitude, two paths that cancel leave no trace of their individual delays. That is why vital-sign and fine-motion work needs CSI, and why the earliest occupancy work could get away with RSSI."
      real="The corpus splits cleanly along this axis: RSSI papers target coarse presence/occupancy, CSI papers target respiration, gesture and fine localization."
      papers={SIM_PAPERS["rssi-vs-csi"].papers}
      params={
        <>
          <Slider label="walk speed" value={velocity} min={0.1} max={2} step={0.05} onChange={setVelocity} fmt={(v) => `${v.toFixed(2)} m/s`} />
          <Slider label="receiver noise" value={noiseSd} min={0} max={0.2} step={0.005} onChange={setNoiseSd} />
        </>
      }
    >
      <LinePlot
        series={[
          { ys: rssiSeries, color: "#fbbf24", label: "RSSI (magnitude only)" },
          { ys: csiMag, color: "#22d3ee", label: "CSI |H| subcarrier 16" },
        ]}
        yLabel="amplitude"
        xLabel="packet index"
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="RSSI σ" value={Math.sqrt(rssiVar).toFixed(4)} tone="amber" hint="variation RSSI shows for this walk" />
        <Stat label="quantities / packet" value="1 vs 30" tone="cyan" />
        <Stat label="phase available" value="RSSI: no" tone="rose" />
        <Stat label="sampling" value={`${fs} Hz`} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Raise receiver noise and the RSSI trace fills with jitter proportional to the signal,
        because there is no averaging across subcarriers to fall back on. The CSI trace at one
        subcarrier is noisier still — the win comes from using all 30 together, which is why CSI
        work dominates the corpus.
      </p>
    </Sim>
  );
}

/* ========================== Phase wrap / unwrap ========================= */

export function PhaseWrapSim() {
  const [d, setD] = useState(3);
  const [noiseSd, setNoiseSd] = useState(0.05);
  const [sanitize, setSanitize] = useState(true);
  const g = useMemo(() => makeGeometry(5), []);
  const fs = 40;

  const { raw, unwrapped, clean } = useMemo(() => {
    const traj = walkTrajectory(fs, 3, 2, 0.4, 1, 8);
    let a = 7;
    const rnd = () => {
      a = (a * 1103515245 + 12345) % 2147483648;
      return a / 2147483648 - 0.5;
    };
    const r: number[] = [];
    for (const dd of traj) {
      const h = csiPacket(scenePaths(g, dd, 1, 0.6))[15];
      r.push(wrapPhase(Math.atan2(h.im, h.re) + rnd() * noiseSd));
    }
    const u = unwrapSeries(r);
    return { raw: r, unwrapped: u, clean: sanitizeFiLiMS(u) };
  }, [g, noiseSd]);

  const jumps = raw.filter((v, i) => i > 0 && Math.abs(v - raw[i - 1]) > Math.PI).length;
  const shown = sanitize ? clean : unwrapped;

  return (
    <Sim
      name="CSI phase is wrapped; unwrapping is not optional"
      identifies={`${jumps} of ${raw.length - 1} consecutive phase pairs jump by more than π — a receiver reports phase only modulo 2π, so raw phase is unusable without unwrapping.`}
      math={[
        "reported:  φ̂ = wrap(H) ∈ (−π, π]      (the 2π ambiguity is inherent)",
        "unwrap:    φᵢ = φᵢ₋₁ + wrap(φ̂ᵢ − φᵢ₋₁)   (accumulate the small steps)",
        "FiLiMS sanitize: if the phase *slope* flips sign between packets, invert the sample",
        "         Δφᵢ ← −Δφᵢ, which is the classic SFRA-induced spike artefact",
      ]}
      need="Every phase-based paper — vital signs, gesture, fine localization — begins here. Unwrap by the wrong rule and you fabricate a large sawtooth that no filter will remove; the sign-flip correction exists because commodity NICs sometimes report inverted samples."
      real="This is the most-repeated preprocessing step in the corpus and the most common source of irreproducible results: papers that skip unwrapping, or unwrap with a different rule, are not comparable."
      papers={SIM_PAPERS["phase-wrap"].papers}
      params={
        <>
          <Slider label="target distance" value={d} min={0.5} max={9} step={0.1} onChange={setD} fmt={(v) => `${v.toFixed(1)} m`} />
          <Slider label="phase noise" value={noiseSd} min={0} max={0.4} step={0.01} onChange={setNoiseSd} />
          <Segmented
            label="sanitize"
            value={sanitize ? "on" : "off"}
            onChange={(v) => setSanitize(v === "on")}
            options={[
              { value: "on", label: "FiLiMS sanitize" },
              { value: "off", label: "unwrap only" },
            ]}
          />
        </>
      }
    >
      <LinePlot
        series={[
          { ys: raw, color: "#71717a", width: 1, dashed: true, label: "raw (wrapped)" },
          { ys: shown, color: "#22d3ee", label: sanitize ? "unwrapped + sanitized" : "unwrapped" },
        ]}
        yLabel="radians"
        xLabel="packet index"
        domain={[-Math.PI, Math.PI]}
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="π-jumps" value={String(jumps)} tone={jumps > 0 ? "amber" : "emerald"} />
        <Stat label="raw range" value={`${(Math.max(...raw) - Math.min(...raw)).toFixed(2)} rad`} />
        <Stat label="unwrapped range" value={`${(Math.max(...unwrapped) - Math.min(...unwrapped)).toFixed(2)} rad`} tone="cyan" />
        <Stat label="sanitize" value={sanitize ? "on" : "off"} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        The grey dashed line is what the NIC reports — it is bounded to ±π and jumps
        discontinuously. The cyan line is the recovered motion. Toggle sanitize off and on at high
        phase noise: the corrected version trades a little smoothing for the removal of the
        inverted samples, which otherwise dominate any derivative-based feature.
      </p>
    </Sim>
  );
}