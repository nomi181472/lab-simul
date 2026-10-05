"use client";

import { useMemo, useState } from "react";
import {
  domainShiftAccuracy,
  erf,
  pathLossDb,
  snrDb,
  throughWallDb,
  usableAfterWalls,
} from "@/labs/wifi-sensing/sim/math";
import { WALL_MATERIALS, domainScenes } from "@/labs/wifi-sensing/sim/mock";
import { Slider, Segmented, Sim } from "./shared";
import { BarPlot, LinePlot, RoomPlan, Stat } from "./canvas";
import { SIM_PAPERS } from "./papers";

/* ========================== Cross-domain shift ========================== */

export function DomainShiftSim() {
  const [shift, setShift] = useState(1.4);
  const [spread, setSpread] = useState(1);
  const [adapt, setAdapt] = useState(false);

  const { a, b } = useMemo(() => domainScenes(shift, spread), [shift, spread]);
  const accNoAdapt = domainShiftAccuracy(shift, spread);
  const accAdapted = adapt ? domainShiftAccuracy(shift / 6, spread) : accNoAdapt;

  const sweep = useMemo(
    () => Array.from({ length: 60 }, (_, i) => domainShiftAccuracy(i * 0.1, spread)),
    [spread],
  );

  return (
    <Sim
      name="A model trained in one room fails in the next"
      identifies={`The same activity produces features ${shift.toFixed(2)}σ apart between domain A (trained) and domain B (tested). Accuracy without adaptation is ${(accNoAdapt * 100).toFixed(1)}%; with a mean-alignment correction ${(accAdapted * 100).toFixed(1)}%.`}
      math={[
        "train threshold learned on domain A:  t = 0",
        "after a mean shift of Δ:  accuracy = Φ( Δ / σ )   (target scores now straddle t)",
        "mean alignment (a one-line adaptation) removes first-order shift:  Δ → Δ/6",
      ]}
      need="CSI depends on the NIC's antenna pattern, the room's geometry, and everyone's furniture. None of that transfers, so a trained model's decision boundary is calibrated to the wrong distribution the moment it is deployed elsewhere."
      real="Cross-domain generalization is a standing problem across the corpus: same-activity-different-room, same-activity-different-device. Domain adaptation and deployment-aware training are the recurring answers."
      papers={SIM_PAPERS["domain-shift"].papers}
      params={
        <>
          <Slider label="domain shift Δ" value={shift} min={0} max={4} step={0.1} onChange={setShift} fmt={(v) => `${v.toFixed(2)} σ`} />
          <Slider label="within-domain spread σ" value={spread} min={0.2} max={3} step={0.1} onChange={setSpread} />
          <Segmented
            label="adaptation"
            value={adapt ? "on" : "off"}
            onChange={(v) => setAdapt(v === "on")}
            options={[
              { value: "on", label: "mean alignment" },
              { value: "off", label: "none" },
            ]}
          />
        </>
      }
    >
      <LinePlot
        series={[
          {
            ys: a,
            color: "#34d399",
            width: 1,
            dashed: true,
            label: "domain A (train)",
          },
          { ys: b, color: "#fbbf24", width: 1, label: "domain B (test)" },
        ]}
        yLabel="feature value"
        xLabel="sample index"
      />
      <BarPlot ys={sweep} yLabel="accuracy" xLabel="shift Δ (σ)" color="#22d3ee" highlight={Math.min(59, Math.round(shift * 10))} />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="acc, no adaptation" value={`${(accNoAdapt * 100).toFixed(1)}%`} tone={accNoAdapt > 0.85 ? "emerald" : "rose"} />
        <Stat label="acc, mean-aligned" value={`${(accAdapted * 100).toFixed(1)}%`} tone="cyan" />
        <Stat label="Δ / σ" value={(shift / spread).toFixed(2)} />
        <Stat label="shift regime" value={shift / spread < 0.5 ? "mild" : shift / spread < 1.5 ? "moderate" : "severe"} tone={shift / spread > 1.5 ? "rose" : "amber"} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        The two traces are the same physical activity recorded in two rooms, and they barely
        overlap. Increase Δ and the no-adaptation bar walks to chance while the model is
        unchanged. That gap — in-domain accuracy minus cross-domain accuracy — is the honest
        measure of how much of a WiFi paper’s result is about the method and how much is about
        the room it was tested in.
      </p>
    </Sim>
  );
}

/* ======================== Through-wall attenuation ======================== */

const TX_DBM = -10;
const NOISE_DBM = -92;

export function ThroughWallSim() {
  const [walls, setWalls] = useState(2);
  const [matIdx, setMatIdx] = useState(1);
  const [dist, setDist] = useState(4);
  const mat = WALL_MATERIALS[matIdx];
  const pl = pathLossDb(dist, 2.4);
  const wall = throughWallDb(walls, mat.perWallDb);
  const rx = TX_DBM - pl - wall;
  const snr = rx - NOISE_DBM;
  const ok = usableAfterWalls(TX_DBM, pl, walls, mat.perWallDb, NOISE_DBM);

  const rangeCurve = useMemo(
    () =>
      Array.from({ length: 40 }, (_, i) => {
        const d = i * 0.5 + 0.5;
        return TX_DBM - pathLossDb(d, 2.4) - throughWallDb(walls, mat.perWallDb) - NOISE_DBM;
      }),
    [walls, mat.perWallDb],
  );

  const maxRange = (() => {
    // solve snr = 0 for d
    const need = TX_DBM - NOISE_DBM - throughWallDb(walls, mat.perWallDb);
    return Math.pow(10, need / (10 * 2.4));
  })();

  return (
    <Sim
      name="Through-wall sensing range is an arithmetic budget"
      identifies={`${walls} × ${mat.label} wall(s) costs ${wall.toFixed(0)} dB. At ${dist.toFixed(1)} m indoors (${pl.toFixed(1)} dB) the received level is ${rx.toFixed(1)} dBm, SNR ${snr.toFixed(1)} dB — ${ok ? "usable" : "unusable"}.`}
      math={[
        "P_rx = P_tx − PL(d) − N_walls · A_wall",
        "SNR = P_rx − N",
        "usable ⇔ SNR ≥ margin (6 dB is a common working margin)",
        "⇒ d_max = 10^(( P_tx − N − N_walls·A_wall ) / (10 n))",
      ]}
      need="Every claimed through-wall range is this formula evaluated with a particular wall material and count. Concrete at 9 dB per wall and drywall at 3 dB differ by a factor of three in range — which is why papers must state the obstruction."
      real="Through-wall detection papers report the wall type for exactly this reason, and the impressive ranges in the literature are mostly single-drywall-plaster setups."
      papers={SIM_PAPERS["through-wall"].papers}
      params={
        <>
          <Slider label="wall count" value={walls} min={0} max={5} step={1} onChange={setWalls} fmt={(v) => String(v)} />
          <Slider label="distance" value={dist} min={0.5} max={15} step={0.5} onChange={setDist} fmt={(v) => `${v.toFixed(1)} m`} />
          <Segmented
            label="material"
            value={String(matIdx)}
            onChange={(v) => setMatIdx(Number(v))}
            options={WALL_MATERIALS.map((m, i) => ({ value: String(i), label: m.label }))}
          />
        </>
      }
    >
      <RoomPlan tx={{ x: 0, y: 0 }} rx={{ x: dist, y: 0 }} target={{ x: dist / 2, y: 1.5 }} />
      <LinePlot
        series={[{ ys: rangeCurve, color: "#22d3ee", label: "SNR vs distance" }]}
        yLabel="dB"
        xLabel="distance (m)"
        zeroLine
        markers={[{ i: Math.min(39, Math.round(dist * 2) - 1), color: "#fbbf24", label: `${dist.toFixed(1)} m` }]}
      />
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="wall loss" value={`${wall.toFixed(0)} dB`} tone="amber" />
        <Stat label="received" value={`${rx.toFixed(1)} dBm`} />
        <Stat label="SNR" value={`${snr.toFixed(1)} dB`} tone={snr > 15 ? "emerald" : snr > 6 ? "amber" : "rose"} />
        <Stat label="max range" value={maxRange > 50 ? ">50 m" : `${maxRange.toFixed(1)} m`} tone={ok ? "emerald" : "rose"} />
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Cycle the material from glass to a metal door and watch the max-range tile collapse. A
        metal door at 18 dB per wall eliminates the budget entirely at two walls. This is the
        honest shape of a through-wall claim: it is a joint statement about geometry, wall
        composition, and noise floor — not a single range number.
      </p>
    </Sim>
  );
}

/** erf re-exported so the Compare view can share the tail approximation. */
export { erf, snrDb };