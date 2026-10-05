"use client";

import { useMemo, useState } from "react";
import {
  binHz,
  convReceptiveField,
  cfoPhaseRampHz,
  csiPacket,
  dfr,
  erf,
  fcs,
  freqResolution,
  gaussian,
  mulberry32,
  peakBin,
  periodogram,
  phaseNoiseFromAmpNoise,
  quantErrorBound,
  quantize,
  receptiveFieldSeconds,
  rms,
  spectralEntropy,
  variance,
  velocityResolution,
  wrapPhase,
} from "@/labs/wifi-sensing/sim/math";
import { makeGeometry, scenePaths, walkTrajectory } from "@/labs/wifi-sensing/sim/mock";
import { Slider, Segmented, Seeded, Sim } from "./shared";
import { BarPlot, LinePlot, Stat } from "./canvas";
import { SIM_PAPERS } from "./papers";

/* ====================== STFT / velocity resolution ====================== */

export function StftVelocitySim() {
  const [n, setN] = useState(100);
  const [fs, setFs] = useState(100);
  const [trueV, setTrueV] = useState(0.8);
  const cosTheta = 1;
  const df = freqResolution(n, fs);
  const dv = velocityResolution(df, cosTheta);
  const vBin = (trueV / 3e8) * 2.4e9;

  return (
    <Sim
      name="Window length trades resolution against latency"
      identifies={`An ${n}-packet window at ${fs} Hz spans ${(n / fs).toFixed(2)} s, resolves ${df.toFixed(2)} Hz, and can therefore report velocity no finer than ${dv.toFixed(3)} m/s — while the true velocity here is ${trueV.toFixed(2)} m/s.`}
      math={[
        "Δf = f_s / N                 (frequency resolution of an N-point window)",
        "Δv = Δf · c / (cosθ · f_c)   (velocity resolution implied by Δf)",
        "N / f_s = window latency     (how stale the estimate is)",
      ]}
      need="Resolution and latency are the same knob. Short windows track a moving target but cannot tell 0.8 m/s from 0.9; long windows can tell but respond after the target has already changed direction."
      real="Any paper reporting a velocity or respiration-rate number is implicitly choosing N and f_s. Changing either changes the claim, which is why these are not comparable across papers."
      papers={SIM_PAPERS["stft-velocity"].papers}
      params={
        <>
          <Slider label="window N (packets)" value={n} min={16} max={512} step={16} onChange={setN} fmt={(v) => String(v)} />
          <Slider label="sample rate" value={fs} min={20} max={400} step={10} onChange={setFs} fmt={(v) => `${v} Hz`} />
          <Slider label="true velocity" value={trueV} min={0.05} max={2} step={0.05} onChange={setTrueV} fmt={(v) => `${v.toFixed(2)} m/s`} />
        </>
      }
    >
      <LinePlot
        series={[
          {
            ys: Array.from({ length: 33 }, (_, i) =>
              freqResolution(i * 16 + 16, fs),
            ),
            color: "#22d3ee",
            label: "Δf vs N",
          },
        ]}
        yLabel="Δf (Hz)"
        xLabel="window length (packets)"
        markers={[{ i: Math.min(31, Math.floor(n / 16) - 1), color: "#fbbf24", label: `${n}` }]}
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="window span" value={`${(n / fs).toFixed(2)} s`} />
        <Stat label="Δf" value={`${df.toFixed(2)} Hz`} tone="cyan" />
        <Stat label="Δv" value={`${dv.toFixed(3)} m/s`} tone="cyan" />
        <Stat label="bins across v_max" value={String(Math.floor(fs / 2 / df))} hint="how many distinct velocities the window can express" />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Note the true velocity maps to ${" "}
        <span className="font-mono text-amber-200">{vBin.toFixed(1)} Hz</span>, which is{" "}
        {Math.round(vBin / df)} bins — so the estimate is only good to about ±Δv/2. At N=512 a
        gait cycle no longer fits in the window and the “motion” looks static.
      </p>
    </Sim>
  );
}

/* ============================ Periodogram ============================ */

export function PeriodogramSim() {
  const [seed, setSeed] = useState(4);
  const [fs, setFs] = useState(20);
  const [rateHz, setRateHz] = useState(1.2);
  const [noise, setNoise] = useState(0.15);
  const [n, setN] = useState(128);

  const x = useMemo(() => {
    const rand = mulberry32(seed);
    return Array.from({ length: n }, (_, i) => {
      const t = i / fs;
      return Math.sin(2 * Math.PI * rateHz * t) + gaussian(rand, 0, noise);
    });
  }, [seed, fs, rateHz, noise, n]);

  const spec = useMemo(() => periodogram(x), [x]);
  const pk = peakBin(spec);
  const est = binHz(n, fs, pk);
  const trueBin = binHz(n, fs, Math.round((rateHz * n) / fs));

  return (
    <Sim
      name="Dominant-bin rate estimation"
      identifies={`A ${rateHz.toFixed(2)} Hz oscillation buried in noise σ=${noise.toFixed(2)}: periodogram peak at bin ${pk} = ${est.toFixed(2)} Hz, error ${(100 * Math.abs(est - rateHz) / rateHz).toFixed(1)}%.`}
      math={[
        "X(k) = Σₙ x[n] · w[n] · e^(−j2πkn/N)      (Hann-windowed DFT)",
        "peak bin k* = argmax_k |X(k)|",
        "f̂ = k* · f_s / N",
      ]}
      need="Turns an oscillation into a number. Every respiration-rate, gait-cadence and keystroke paper ultimately estimates one dominant bin — and inherits its bias toward the nearest bin centre, which is why reported rates cluster on discrete values."
      real="Respiration and heartbeat papers report a bpm error; this is the estimator producing the number, and its bin quantisation is a floor under that error."
      papers={SIM_PAPERS["periodogram"].papers}
      params={
        <>
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
          <Slider label="true rate" value={rateHz} min={0.2} max={4} step={0.05} onChange={setRateHz} fmt={(v) => `${v.toFixed(2)} Hz`} />
          <Slider label="noise σ" value={noise} min={0} max={1.5} step={0.05} onChange={setNoise} />
          <Slider label="N" value={n} min={32} max={512} step={16} onChange={setN} fmt={(v) => String(v)} />
          <Slider label="sample rate" value={fs} min={5} max={100} step={5} onChange={setFs} fmt={(v) => `${v} Hz`} />
        </>
      }
    >
      <BarPlot ys={spec} yLabel="|X(k)|" xLabel="bin k" highlight={pk} />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="peak bin" value={String(pk)} tone="amber" />
        <Stat label="true bin" value={String(trueBin)} />
        <Stat label="f̂" value={`${est.toFixed(3)} Hz`} tone="cyan" />
        <Stat label="rel. error" value={`${(100 * Math.abs(est - rateHz) / rateHz).toFixed(1)}%`} tone={Math.abs(est - rateHz) / rateHz < 0.1 ? "emerald" : "rose"} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Sweep the true rate and watch the peak jump in steps — the estimator can only land on bin
        centres, so error is bounded below by Δf/2 and never reaches zero. Raise noise and the
        peak wanders to a neighbouring bin entirely, which is why respiration papers report a
        median error over long windows rather than an instantaneous rate.
      </p>
    </Sim>
  );
}

/* ========================== Feature families ========================== */

export function FeatureFamiliesSim() {
  const [velocity, setVelocity] = useState(0.8);
  const [presence, setPresence] = useState(true);
  const [seed, setSeed] = useState(9);
  const fs = 50;
  const n = 128;

  const feats = useMemo(() => {
    const traj = walkTrajectory(fs, n / fs, 2.5, velocity, 2, 5);
    // sample the channel at subcarrier 16 for every packet in the walk
    const samples = traj.map((d) => csiPacket(scenePaths(makeGeometry(5), d, presence ? 1 : 0, 0.6))[15]);
    const amp = samples.map((z) => Math.hypot(z.re, z.im));
    const spec = periodogram(amp);
    const movingHz = binHz(n, fs, peakBin(spec));
    return {
      rss: rms(amp),
      dfr: dfr(movingHz, 0.5),
      fcs: fcs(amp),
      ccr: variance(amp) / Math.max(rms(amp) ** 2, 1e-9),
      entropy: spectralEntropy(spec),
      series: amp,
    };
  }, [velocity, presence, n, fs]);

  return (
    <Sim
      name="What each classical feature responds to"
      identifies={`${presence ? "Target present" : "Empty room"} at ${velocity.toFixed(2)} m/s — RMS ${feats.rss.toFixed(3)}, FCS ${feats.fcs.toFixed(4)}, CCR ${feats.ccr.toFixed(3)}, entropy ${feats.entropy.toFixed(2)} bits.`}
      math={[
        "RMS   = √( mean |H|² )                      — overall energy, insensitive to motion",
        "FCS   = mean | |H|ᵢ − |H|ᵢ₋₁| |              — fluctuation, fires on any motion",
        "DFR   = f_d(moving) / f_d(static)           — ratio, cancels TX power and gain",
        "CCR   = var / mean²                        — coefficient of variation",
        "entropy = −Σ p log₂ p over the spectrum     — randomness of the CSI spectrum",
      ]}
      need="These four families are the training-free baseline every deep model in the corpus is measured against. They differ in what they can be confused by: RMS ignores motion entirely, FCS fires on anything, DFR survives gain changes, entropy tracks unpredictability."
      real="Papers introducing a new architecture almost always report accuracy against an FCS or DFR baseline. If the feature baseline already matches the network, the network is not the contribution."
      papers={SIM_PAPERS["feature-families"].papers}
      params={
        <>
          <Slider label="target speed" value={velocity} min={0} max={2} step={0.05} onChange={setVelocity} fmt={(v) => `${v.toFixed(2)} m/s`} />
          <Segmented
            label="state"
            value={presence ? "present" : "empty"}
            onChange={(v) => setPresence(v === "present")}
            options={[
              { value: "present", label: "target present" },
              { value: "empty", label: "empty room" },
            ]}
          />
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
        </>
      }
    >
      <LinePlot
        series={[{ ys: feats.series, color: "#22d3ee", label: "|H| over time" }]}
        yLabel="amplitude"
        xLabel="packet index"
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
        <Stat label="RMS" value={feats.rss.toFixed(3)} tone="cyan" hint="energy: barely moves with speed" />
        <Stat label="FCS" value={feats.fcs.toFixed(4)} tone="cyan" hint="fires on any change" />
        <Stat label="DFR" value={feats.dfr.toFixed(2)} tone="cyan" hint="gain-invariant ratio" />
        <Stat label="CCR" value={feats.ccr.toFixed(3)} />
        <Stat label="entropy" value={`${feats.entropy.toFixed(2)} b`} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Go from “target present” to “empty room”: FCS and CCR collapse, which is why they are the
        presence-detection features. Now hold the target present and change only its speed — RMS
        barely moves but DFR and the spectral entropy do, which is why speed-sensitive tasks
        (gait, fall, gesture) cannot use energy features alone.
      </p>
    </Sim>
  );
}

/* ========================= FFT shift invariance ========================= */

export function FftInvarianceSim() {
  const [offset, setOffset] = useState(40);
  const [seed, setSeed] = useState(6);
  const fs = 20;
  const n = 128;
  const rateHz = 1.0;

  const { raw, fft } = useMemo(() => {
    const rand = mulberry32(seed);
    const base = Array.from({ length: n + offset }, (_, i) => {
      const t = i / fs;
      return Math.sin(2 * Math.PI * rateHz * t) + gaussian(rand, 0, 0.2);
    });
    const window = base.slice(offset, offset + n);
    const rawSpec = periodogram(window);
    return { raw: window, fft: rawSpec };
  }, [seed, offset]);

  return (
    <Sim
      name="FFT features ignore where the window starts"
      identifies={`The same ${rateHz.toFixed(2)} Hz oscillation, windowed ${offset} packets later, has a completely different time-domain trace — but its FFT magnitude is identical, so the recognizer is unaffected by the arbitrary start.`}
      math={[
        "X[k] = | FFT( x[n] · w[n] ) |   (magnitude spectrum)",
        "a circular time shift by s multiplies X[k] by e^(−j2πks/N) → |X[k]| unchanged",
        "⇒ the feature is invariant to the window's position in the stream, not to its length",
      ]}
      need="Nobody can align a real-time CSI window to the start of a gesture. FFT-domain features sidestep that entirely, which is why a large fraction of WiFi activity-recognition papers operate on FFT magnitudes rather than raw time series."
      real="Papers that quote FFT-magnitude inputs are buying shift invariance for free — but paying resolution, since N bins over the whole window is a coarse summary of any sub-window structure."
      papers={SIM_PAPERS["fft-invariance"].papers}
      params={
        <>
          <Slider label="window start offset" value={offset} min={0} max={80} step={1} onChange={setOffset} fmt={(v) => `${v} pkts`} />
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
        </>
      }
    >
      <LinePlot
        series={[{ ys: raw.slice(0, n), color: "#22d3ee", label: `window starting at ${offset}` }]}
        yLabel="amplitude"
        xLabel="packet index"
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="dominant bin" value={String(peakBin(fft))} tone="amber" />
        <Stat label="dominant Hz" value={binHz(n, fs, peakBin(fft)).toFixed(2)} tone="cyan" />
        <Stat label="window offset" value={`${offset} pkts`} />
        <Stat label="time-domain L2" value={Math.sqrt(raw.slice(0, n).reduce((s, v) => s + v * v, 0)).toFixed(3)} hint="changes with offset" />
      </div>
      <p className="mt-3 text-[11] leading-5 text-zinc-500">
        Slide the offset across its whole range and watch the time-domain plot slide with it while
        the two frequency-domain statistics stay pinned. That invariance is the reason for the
        FFT feature pipeline — and the reason you cannot recover “when did this happen” from it.
      </p>
    </Sim>
  );
}

/* ======================== Conv1D receptive field ======================== */

export function Conv1dWindowSim() {
  const [k, setK] = useState(5);
  const [layers, setLayers] = useState(2);
  const [rate, setRate] = useState(100);
  const rf = convReceptiveField(k, layers);
  const secs = receptiveFieldSeconds(rf, rate);
  const gaitCycles = secs * 1.2;
  const breathCycles = secs * 0.25;

  return (
    <Sim
      name="Temporal receptive field of a 1-D conv over CSI"
      identifies={`${layers} conv layer(s) of width ${k} at ${rate} Hz packet rate see ${rf} packets = ${secs.toFixed(3)} s — covering ${gaitCycles.toFixed(2)} gait cycles but only ${breathCycles.toFixed(2)} respiration cycles.`}
      math={[
        "RF = 1 + (k−1)·Σⱼ₌₁ᴸ strideʲ⁻¹        (receptive field of stacked 1-D convs)",
        "t_window = RF / f_s                     (seconds of stream the filter can see)",
        "need t_window ≥ 1/f_signal for the feature to resolve the signal at all",
      ]}
      need="A conv kernel is a fixed window on the packet stream. Too short and the model cannot see a full physiological cycle; too long and it averages away the detail that distinguishes two activities."
      real="Every deep WiFi model is described by its kernel sizes, and those numbers decide which physical phenomena the architecture is even capable of representing."
      papers={SIM_PAPERS["conv1d-window"].papers}
      params={
        <>
          <Slider label="kernel width k" value={k} min={1} max={15} step={2} onChange={setK} fmt={(v) => String(v)} />
          <Slider label="layers" value={layers} min={1} max={6} step={1} onChange={setLayers} fmt={(v) => String(v)} />
          <Slider label="packet rate" value={rate} min={10} max={400} step={10} onChange={setRate} fmt={(v) => `${v} Hz`} />
        </>
      }
    >
      <LinePlot
        series={[
          {
            ys: Array.from({ length: 8 }, (_, i) => convReceptiveField(k, i)),
            color: "#22d3ee",
            label: "RF vs depth",
          },
        ]}
        yLabel="packets"
        xLabel="layers"
        markers={[{ i: Math.min(7, layers), color: "#fbbf24", label: `${rf}` }]}
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="RF" value={`${rf} pkts`} tone="cyan" />
        <Stat label="window" value={`${secs.toFixed(3)} s`} tone="cyan" />
        <Stat label="gait cycles (1.2 Hz)" value={gaitCycles.toFixed(2)} tone={gaitCycles >= 1 ? "emerald" : "rose"} />
        <Stat label="breath cycles (0.25 Hz)" value={breathCycles.toFixed(2)} tone={breathCycles >= 1 ? "emerald" : "rose"} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        The two cycle counts are the design constraint. Falling activity and walking need only
        about one gait cycle, which almost any window satisfies. Respiration at 0.25 Hz needs four
        seconds of stream — a 0.4 s window cannot represent it at all, no matter how many layers
        are stacked.
      </p>
    </Sim>
  );
}

/* ========================= Quantization noise ========================= */

export function QuantizationNoiseSim() {
  const [bits, setBits] = useState(8);
  const [amp, setAmp] = useState(0.5);
  const err = quantErrorBound(bits, 1);
  const qAmp = quantize(amp, bits, 1);
  const phaseErr = phaseNoiseFromAmpNoise(0.7, err, amp);

  const sweep = useMemo(
    () =>
      Array.from({ length: 64 }, (_, i) => {
        const trueVal = (i / 63) * 1.0;
        return quantize(trueVal, bits, 1) - trueVal;
      }),
    [bits],
  );

  return (
    <Sim
      name="NIC bit-width limits amplitude and phase"
      identifies={`At ${bits} bits, the quantisation step is ${(1 / 2 ** bits).toFixed(5)}, an amplitude reading of ${amp.toFixed(3)} lands on ${qAmp.toFixed(5)}, and near-zero amplitude the phase error blows up to ${((phaseErr * 180) / Math.PI).toFixed(1)}°.`}
      math={[
        "step = full_scale / 2^b          (uniform quantiser)",
        "|e| ≤ step/2 = full_scale / 2^(b+1)",
        "phase error ≈ atan2(ε, |A cosθ| + ε)  — diverges as |A| → 0",
        "⇒ deep fades destroy phase long before they destroy amplitude",
      ]}
      need="Commercial NICs report CSI with a small fixed number of bits. Amplitude degrades gracefully; phase does not, because phase is the argument of a complex number and any amplitude error rotates it most when the signal is weakest."
      real="Any paper claiming phase-based sensing through distance or walls is implicitly relying on the receiver keeping enough amplitude to preserve phase."
      papers={SIM_PAPERS["quantization-noise"].papers}
      params={
        <>
          <Slider label="bits b" value={bits} min={2} max={16} step={1} onChange={setBits} fmt={(v) => String(v)} />
          <Slider label="true amplitude" value={amp} min={0.01} max={1} step={0.01} onChange={setAmp} />
        </>
      }
    >
      <LinePlot
        series={[{ ys: sweep, color: "#fbbf24", label: "quantisation error" }]}
        yLabel="error"
        xLabel="true value"
        zeroLine
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="step" value={(1 / 2 ** bits).toFixed(5)} />
        <Stat label="error bound" value={err.toFixed(5)} />
        <Stat label="reported" value={qAmp.toFixed(5)} tone="cyan" />
        <Stat label="phase error" value={`${((phaseErr * 180) / Math.PI).toFixed(1)}°`} tone={phaseErr > 0.3 ? "rose" : "emerald"} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Sweep the amplitude toward zero and watch the phase-error tile climb without bound. This
        is the practical reason WiFi sensing has a hard range limit that is smaller than the
        link’s range: past that point the link is still up, but its phase is noise.
      </p>
    </Sim>
  );
}

/* ============================== CFO / SFO ============================== */

export function CfoSfoSim() {
  const [offsetHz, setOffsetHz] = useState(0.4);
  const [fs, setFs] = useState(100);
  const ramp = cfoPhaseRampHz(offsetHz, fs);
  const drift = useMemo(
    () => Array.from({ length: 80 }, (_, i) => wrapPhase((2 * Math.PI * ramp * i))),
    [ramp],
  );
  const totalDrift = ((offsetHz * 80) / fs) * 360;

  return (
    <Sim
      name="Oscillator offset masquerades as phase drift"
      identifies={`A ${offsetHz.toFixed(2)} Hz carrier offset at ${fs} Hz sampling adds ${(360 * ramp).toFixed(2)}° of phase every packet — ${totalDrift.toFixed(0)}° across an 80-packet window, a pure artifact of the radio's own clocks.`}
      math={[
        "phase ramp per packet = 2π · Δf / f_s",
        "carrier offset (CFO) and sampling offset (SFO) both appear in the same axis",
        "⇒ they cannot be separated by phase alone; amplitude-domain or pilot-based",
        "   correction is required before Doppler estimation",
      ]}
      need="Hardware is never on exactly the right frequency. The resulting phase looks identical to a slow-moving reflector, so an uncorrected pipeline reports phantom motion that grows linearly with window length."
      real="Papers reporting absolute displacement from CSI phase must handle CFO/SFO first; those that do not have an unmodelled drift term that grows with the integration window."
      papers={SIM_PAPERS["cfo-sfo"].papers}
      params={
        <>
          <Slider label="frequency offset" value={offsetHz} min={-2} max={2} step={0.05} onChange={setOffsetHz} fmt={(v) => `${v.toFixed(2)} Hz`} />
          <Slider label="sample rate" value={fs} min={10} max={400} step={10} onChange={setFs} fmt={(v) => `${v} Hz`} />
        </>
      }
    >
      <LinePlot
        series={[{ ys: drift, color: "#fbbf24", label: "residual phase after wrap" }]}
        yLabel="radians"
        xLabel="packet index"
        domain={[-Math.PI, Math.PI]}
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="ramp / packet" value={`${(360 * ramp).toFixed(2)}°`} tone="amber" />
        <Stat label="drift over 80 pkts" value={`${totalDrift.toFixed(0)}°`} tone={Math.abs(totalDrift) > 180 ? "rose" : "zinc"} />
        <Stat label="turns / window" value={(offsetHz * 80 / fs).toFixed(3)} />
        <Stat label="breathing-rate artefact" value={`${(Math.abs(offsetHz) * 60).toFixed(0)} bpm`} tone="rose" hint="a 0.4 Hz clock offset masquerades as a 24 bpm 'breathing' rate" />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        The last tile is the trap: a modest clock offset sits directly inside the respiration band
        (6–48 bpm), so an uncorrected phase stream reports a confident, entirely spurious
        breathing rate. Increasing the sample rate shrinks the drift per packet but does not
        remove it.
      </p>
    </Sim>
  );
}

/** erf re-export so views can reuse the same tail approximation. */
export { erf };