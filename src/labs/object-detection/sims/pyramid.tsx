"use client";

import { useMemo, useState } from "react";
import { Slider, Sim } from "./shared";
import { generateFeatureMap } from "@/labs/object-detection/sim/mock";

export function PyramidSim() {
  const [levels, setLevels] = useState(4);
  const [maxDim, setMaxDim] = useState(32);

  const rows = useMemo(
    () => Array.from({ length: levels }, (_, i) => Math.max(4, Math.round(maxDim / 2 ** i))),
    [levels, maxDim],
  );

  const maps = useMemo(
    () => rows.map((r, i) => generateFeatureMap(100 + i, r, r, r)),
    [rows],
  );

  // a small object (say 6x6 px at input) after stride 32 collapses into <1 cell
  const objectAt = (stride: number) => {
    const size = 8 / stride;
    return size;
  };

  return (
    <Sim
      name="Feature pyramid for scale"
      identifies={`${levels} resolution levels (${rows.join(" → ")} cells). A small 8px object occupies ${objectAt(8).toFixed(2)} cells at 8× stride and ${objectAt(32).toFixed(2)} cells at 32× stride.`}
      math={[
        "feature maps shrink by k per level: h_i = h / stride_i",
        "an object of w px spans w/stride_i cells at level i",
        "detectors read from the level where the object still spans ≥1 cell",
      ]}
      need="Small objects vanish in deep, low-resolution maps: SSD itself noted deep maps 'may not have any information' for small objects. Multi-scale heads and top-down feature pyramids keep small and large objects alive in the same backbone."
      real="SSD’s maps, then feature pyramids/concat/concatenation webs (FSSD, RefineDet’s TCB, DSOD dense fusions), NAS-searched necks (DAMO-YOLO / Gold-YOLO), and RT-DETR’s hybrid encoder — all answer one question: where does a tiny object survive?"
      papers={["P001", "P004", "P005", "P008", "P027"]}
      params={
        <>
          <Slider label="pyramid levels" value={levels} min={2} max={6} step={1} onChange={setLevels} />
          <Slider label="top resolution" value={maxDim} min={16} max={64} step={2} onChange={setMaxDim} />
        </>
      }
    >
      <div className="flex flex-wrap items-end gap-3">
        {maps.map((m, i) => (
          <div key={i}>
            <div className="mb-1 text-center font-mono text-[9px] text-zinc-600">
              level {i} · {m.grid.length}×{m.grid[0].length}
            </div>
            <div
              className="grid gap-px border border-zinc-800 bg-zinc-800 p-px"
              style={{
                gridTemplateColumns: `repeat(${m.grid[0].length}, ${Math.max(2, 92 / m.grid[0].length)}px)`,
              }}
            >
              {m.grid.flat().map((v, j) => (
                <div
                  key={j}
                  className="h-2 w-2 rounded-[1px]"
                  style={{
                    backgroundColor:
                      v > 0.5
                        ? "#34d399"
                        : v < -0.5
                          ? "rgba(251,113,133,0.9)"
                          : `rgba(${70 + Math.floor(v * 60)}, ${90 + Math.floor(v * 40)}, 115, 1)`,
                  }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Watch the count of informative cells collapse as you add levels. At level 0 a small
        object still paints several cells; at the deepest level it is sub-pixel — exactly the
        phenomenon the pyramid and multi-scale supervision attack.
      </p>
    </Sim>
  );
}