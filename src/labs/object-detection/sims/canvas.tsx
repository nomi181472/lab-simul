"use client";

import type { Box } from "@/labs/object-detection/sim/math";

export const VIEW_W = 220;
export const VIEW_H = 130;

export function BoxCanvas({
  boxes,
  className = "",
}: {
  boxes: { box: Box; color?: string; label?: string; dashed?: boolean }[];
  className?: string;
}) {
  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      className={`w-full ${className}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <rect x={0} y={0} width={VIEW_W} height={VIEW_H} className="fill-zinc-900" rx={4} />
      {boxes.map((b, i) => (
        <g key={i}>
          <rect
            x={b.box.x1}
            y={b.box.y1}
            width={Math.max(1, b.box.x2 - b.box.x1)}
            height={Math.max(1, b.box.y2 - b.box.y1)}
            fill={b.color ?? "#34d399"}
            fillOpacity={0.18}
            stroke={b.color ?? "#34d399"}
            strokeWidth={1.5}
            strokeDasharray={b.dashed ? "3 3" : undefined}
          />
          {b.label && (
            <text x={b.box.x1 + 2} y={b.box.y1 + 9} fontSize={8} fill={b.color ?? "#34d399"} fontFamily="monospace">
              {b.label}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}

export function CrossMarker({ x, y, color = "#fbbf24" }: { x: number; y: number; color?: string }) {
  return (
    <g stroke={color} strokeWidth={1.2}>
      <line x1={x - 4} y1={y} x2={x + 4} y2={y} />
      <line x1={x} y1={y - 4} x2={x} y2={y + 4} />
    </g>
  );
}