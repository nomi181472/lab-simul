"use client";

import { useMemo, useState } from "react";
import { generateAnchors, locationStrands, type Box } from "@/labs/object-detection/sim/math";
import { BoxCanvas } from "./canvas";
import { Sim, Segmented } from "./shared";

const GT: Box = { x1: 84, y1: 46, x2: 150, y2: 96 };

export function AnchorsVsFreeSim() {
  const [mode, setMode] = useState<"anchor" | "free">("anchor");
  const [gridOn, setGridOn] = useState(true);

  const anchors = useMemo(
    () =>
      gridOn ? generateAnchors(10, 6, 24, [1], [1, 1.6, 0.6]).filter((a) => a.x1 >= 0 && a.y1 >= 0 && a.x2 <= 220 && a.y2 <= 130) : [],
    [gridOn],
  );

  const { strands, anchorPositives, freePositives } = useMemo(
    () => locationStrands(GT, anchors),
    [anchors],
  );

  const activePositive = mode === "anchor" ? anchorPositives : freePositives;
  const activeColor = mode === "anchor" ? "#34d399" : "#60a5fa";

  return (
    <Sim
      name="Anchor vs anchor-free strands"
      identifies={`Same GT box: anchor scoring finds ${anchorPositives} positive anchors (IoU>0.5); center-cell scoring finds ${freePositives} positive cells (center inside the GT).`}
      math={[
        "anchor strand:   cost = 1 − IoU(anchor, gt)      (a box must overlap the GT)",
        "anchor-free:     positive ≡ cell center inside gt  (the center IS the vote)",
        "scale handles the rest: which pyramid level matches the box size",
      ]}
      need="Two ways to answer “where is the object?” — so its size can be embedded (anchor boxes) or left out entirely (center + level). The corpus contains both strands; the anchor-free side is what lets transformer queries skip preset boxes."
      real="The lab’s corpus documents anchor-based dense heads and the anchor-free/NMS-free thread; choosing the strand is a training-design decision, not a fact a detector must carry."
      papers={["P001", "P003", "P017", "P024", "P027", "P038"]}
      params={
        <>
          <Segmented
            label="strand"
            value={mode}
            options={[
              { value: "anchor", label: "anchor (IoU)" },
              { value: "free", label: "anchor-free (center)" },
            ]}
            onChange={setMode}
          />
          <label className="flex items-center gap-2 font-mono text-[10px] text-zinc-500">
            show grid
            <input type="checkbox" checked={gridOn} onChange={(e) => setGridOn(e.target.checked)} className="accent-emerald-500" />
          </label>
        </>
      }
    >
      <BoxCanvas
        boxes={[
          { box: GT, color: "#fbbf24", label: "gt" },
          ...strands
            .filter((s) => (mode === "anchor" ? s.anchorCost < 0.5 : s.centerInGt))
            .map((s) => ({
              box: { x1: s.cx - s.w / 2, y1: s.cy - s.h / 2, x2: s.cx + s.w / 2, y2: s.cy + s.h / 2 },
              color: activeColor,
              dashed: true,
            })),
        ]}
      />
      <div className="mt-2 h-3 w-full rounded bg-zinc-800">
        <div
          className="h-3 rounded"
          style={{ width: `${(activePositive / Math.max(1, strands.length)) * 100}%`, background: activeColor }}
        />
      </div>
      <div className="mt-2 flex justify-between font-mono text-[10px] text-zinc-500">
        <span>
          {strands.length} anchor positions total · {activePositive} positive in this strand
        </span>
        <span>
          {`positive fraction ${((activePositive / Math.max(1, strands.length)) * 100).toFixed(0)}%`}
        </span>
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        In anchor mode the tiled boxes must overlap; in anchor-free mode every cell whose
        center lands inside the GT votes, and each pyramid level only “sees” boxes near its
        stride-scale. The center strand is viewpoint- and aspect-invariant — that is the
        anchor-free claim in one picture.
      </p>
    </Sim>
  );
}