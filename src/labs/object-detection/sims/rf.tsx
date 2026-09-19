"use client";

import { useMemo, useState } from "react";
import { receptiveField } from "@/labs/object-detection/sim/math";
import { Slider, Segmented, Sim } from "./shared";

type LayerKind = { kernel: number; stride: number; dilation: number };

export function RfSim() {
  const [nConv, setNConv] = useState(5);
  const [kernel, setKernel] = useState(3);
  const [dilate, setDilate] = useState(false);
  const [pool, setPool] = useState(0);

  const layers: LayerKind[] = useMemo(() => {
    const l: LayerKind[] = [];
    for (let i = 0; i < nConv; i++) {
      l.push({ kernel, stride: 1, dilation: dilate ? (i % 2 === 0 ? 1 : 2) : 1 });
    }
    for (let i = 0; i < pool; i++) l.push({ kernel: 2, stride: 2, dilation: 1 });
    return l;
  }, [nConv, kernel, dilate, pool]);

  const { rf, stride, perLayer } = useMemo(() => receptiveField(layers), [layers]);

  return (
    <Sim
      name="Receptive field growth"
      identifies={`${nConv} conv layers (k=${kernel}${dilate ? ", dilated" : ""})${pool ? ` + ${pool} 2× strided pools` : ""} → RF ${rf}px, total stride ${stride}.`}
      math={[
        "rf_new = rf_old + (k′ − 1) · stride_accumulated     (k′ = effective kernel)",
        "effective k′ = (k − 1) · dilation + 1",
        "rf grows graph-linearly in depth, but ×2 when strided (pool) layers pre-multiply",
      ]}
      need="A 3×3 unit can only see 3×3 of input unless the accumulated stride multiplies earlier contexts. Detectors need receptive fields sized to their objects — too small and context is missing, too large and small-object cues are diluted."
      real="SSD justified deep maps failing on small objects precisely via receptive region arguments; dilated convolution features in DSSD/RRC extended RF without stepping down resolution, and early YOLO models racified grid cells to cover the full image."
      papers={["P001", "P004", "P002", "P015"]}
      params={
        <>
          <Slider label="conv layers" value={nConv} min={1} max={10} step={1} onChange={setNConv} />
          <Slider label="kernel size" value={kernel} min={1} max={7} step={1} onChange={setKernel} />
          <Slider label="strided pools" value={pool} min={0} max={4} step={1} onChange={setPool} />
          <Segmented
            label="dilation"
            value={dilate ? "on" : "off"}
            onChange={(v) => setDilate(v === "on")}
            options={[
              { value: "off", label: "plain 3×3 stack" },
              { value: "on", label: "dilated (1,2 cycle)" },
            ]}
          />
        </>
      }
    >
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <div className="font-mono text-[9px] uppercase text-zinc-500">receptive field</div>
            <div className="mt-1 font-mono text-xl text-emerald-300">{rf}px</div>
          </div>
          <div>
            <div className="font-mono text-[9px] uppercase text-zinc-500">total stride</div>
            <div className="mt-1 font-mono text-xl text-sky-300">{stride}×</div>
          </div>
        </div>
        <div className="mt-3">
          <div className="mb-1 font-mono text-[9px] uppercase text-zinc-600">growth curve</div>
          <div className="flex h-24 items-end gap-0.5">
            {perLayer.map((rfv, i) => (
              <div
                key={i}
                title={`layer ${i}: ${rfv}px`}
                className="flex-1 rounded-t bg-emerald-700/60"
                style={{ height: `${Math.min(100, (rfv / Math.max(1, rf)) * 100)}%` }}
              />
            ))}
          </div>
          <div className="mt-1 flex justify-between font-mono text-[8px] text-zinc-700">
            <span>layer 0</span>
            <span>layer {perLayer.length - 1}</span>
          </div>
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Each 3×3 adds only 2px of context — but a single strided pool multiplies accumulated
        stride and can double the RF. Dilation stretches effective coverage without losing
        resolution, the trick DSSD/RRC use for small objects.
      </p>
    </Sim>
  );
}