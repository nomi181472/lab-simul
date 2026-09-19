"use client";

import { useMemo, useState } from "react";
import { attention } from "@/labs/object-detection/sim/math";
import { generateTokens } from "@/labs/object-detection/sim/mock";
import { Slider, Seeded, Sim } from "./shared";

export function AttentionSim() {
  const [seed, setSeed] = useState(3);
  const [q0, setQ0] = useState(0.9);
  const [q1, setQ1] = useState(-0.4);

  const { tokens, labels } = useMemo(() => generateTokens(seed, 6, 2), [seed]);
  const q = [q0, q1];
  const { logits, weights, output } = attention(q, tokens, tokens); // self-attention: keys = values

  const qNorm = Math.sqrt(q.reduce((s, v) => s + v * v, 0) || 1);

  return (
    <Sim
      name="Attention and querying"
      identifies={`A query focuses on a weighted mix of ${tokens.length} tokens. Token 3 (“${labels[3]}”) receives ${(weights[3] * 100).toFixed(0)}% weight here.`}
      math={[
        "logits_i = (q · k_i) / √d",
        "weights = softmax(logits)",
        "output = Σ_i weights_i · v_i        (soft lookup, differentiable)",
      ]}
      need="CNN layers only see a fixed local window; attention lets any position read from anywhere. Self-attention (q trained per position, keys=values=same map) is the core of the transformer detector, used by RT-DETR to fuse multi-scale features."
      real="RT-DETR’s hybrid encoder mixes convolutional and self-attention blocks; deformable/area attention limits cost (RT-DETR, D-FINE); YOLOv12 introduces an attention-only backbone keeping the CNN advantages."
      papers={["P029", "P037", "P041", "P038"]}
      params={
        <>
          <Seeded seed={seed} onReseed={() => setSeed(Math.floor(Math.random() * 1000))} />
          <Slider label="query[0]" value={q0} min={-1} max={1} step={0.05} onChange={setQ0} />
          <Slider label="query[1]" value={q1} min={-1} max={1} step={0.05} onChange={setQ1} />
        </>
      }
    >
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
        <div className="mb-2 font-mono text-[9px] uppercase text-zinc-500">
          query ·|q|={qNorm.toFixed(2)} — attention over tokens
        </div>
        <div className="space-y-1.5">
          {labels.map((l, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="w-24 shrink-0 truncate font-mono text-[10px] text-zinc-500">{l}</span>
              <div className="h-3 flex-1 overflow-hidden rounded-sm bg-zinc-800">
                <div
                  className="h-full bg-emerald-500"
                  style={{ width: `${weights[i] * 100}%` }}
                />
              </div>
              <span className="w-14 shrink-0 text-right font-mono text-[10px] text-zinc-400">
                {(weights[i] * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
        <div className="mt-3 grid grid-cols-2 gap-3">
          <div className="rounded-md border border-zinc-800 p-2">
            <div className="font-mono text-[9px] uppercase text-zinc-600">learned logits</div>
            <div className="mt-1 font-mono text-xs text-sky-300">
              [{logits.map((v) => v.toFixed(2)).join(", ")}]
            </div>
          </div>
          <div className="rounded-md border border-zinc-800 p-2">
            <div className="font-mono text-[9px] uppercase text-zinc-600">output embed</div>
            <div className="mt-1 font-mono text-xs text-emerald-300">
              [{output.map((v) => v.toFixed(2)).join(", ")}]
            </div>
          </div>
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Collision: make q parallel to one token — that token dominates. Rotate it toward the
        average — the mixture flattens. Transformers learn q per query box and produce a soft,
        differentiable mixture, which is what makes anchor-free end-to-end decoding possible.
      </p>
    </Sim>
  );
}

export function SelfAttnVsConvSim() {
  const [position, setPosition] = useState(5);
  const [sigma, setSigma] = useState(2);
  const signal = [0, 0, 1, 1, 1, 0, 0, 0];

  const attentionRows = (() => {
    const m: number[][] = [];
    for (let i = 0; i < signal.length; i++) {
      const raw = signal.map((_, j) => Math.exp(-((i - j) ** 2) / (2 * sigma * sigma)));
      const s = raw.reduce((a, b) => a + b, 0);
      m.push(raw.map((r) => r / s));
    }
    return m;
  })();

  const attended = attentionRows[position];
  const conv5 = [1, 1, 1, 1, 1].map((v) => v / 5);
  const convOut = (i: number) => {
    let s = 0;
    for (let j = 0; j < 5; j++) {
      const idx = i - 2 + j;
      if (idx >= 0 && idx < signal.length) s += signal[idx] * conv5[j];
    }
    return s;
  };

  return (
    <Sim
      name="Self-attention vs. fixed conv"
      identifies={`Reading position ${position}: conv always mixes its fixed 5-cell window; attention soft-mixes everywhere, σ=${sigma} concentrates or flattens the focus.`}
      math={[
        "conv: out[i] = Σ_k w_k · x[i + k]            (weights shared, window fixed)",
        "self-attn: out[i] = Σ_j softmax(q_i·k_j/√d)_j · v_j   (window = everywhere)",
        "attention weights depend on the input content; conv weights do not",
      ]}
      need="Why transformer detectors can join features from opposite corners while conv+pyramids must hop neighbour-by-neighbour. The tradeoff: attention mixes all positions but at quadratic cost — deformable/area attention recovers locality cheaply."
      real="YOLOv12’s area-attention keeps an attention backbone efficient; RT-DETR and D-FINE dispatch conv vs attention blocks; the debate is literally about this slider: how wide a context is safe and affordable."
      papers={["P041", "P037", "P038", "P029"]}
      params={
        <>
          <Slider label="reading position" value={position} min={0} max={7} step={1} onChange={setPosition} />
          <Slider label="σ attention spread" value={sigma} min={0.5} max={8} step={0.1} onChange={setSigma} />
        </>
      }
    >
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
        <div className="mb-2 font-mono text-[9px] uppercase text-zinc-500">signal · · ↔ signal positions</div>
        <div className="flex gap-1">
          {signal.map((v, i) => (
            <div key={i} className="flex flex-1 flex-col items-center gap-1">
              <div className={`h-2 w-2 rounded-full ${v === 1 ? "bg-sky-400" : "bg-zinc-700"}`} />
              <div className="font-mono text-[8px] text-zinc-600">{i}</div>
            </div>
          ))}
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3">
          <div>
            <div className="mb-1 font-mono text-[9px] uppercase text-zinc-600">conv 5-tap weights</div>
            <div className="flex gap-0.5">
              {conv5.map((w, i) => (
                <div key={i} className="flex-1 rounded-t bg-rose-600/70" style={{ height: `${w * 160}px` }} />
              ))}
            </div>
          </div>
          <div>
            <div className="mb-1 font-mono text-[9px] uppercase text-zinc-600">attention weights @ pos {position}</div>
            <div className="flex gap-0.5">
              {attended.map((w, i) => (
                <div key={i} className="flex-1 rounded-t bg-emerald-600/70" style={{ height: `${w * 160}px` }} />
              ))}
            </div>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-3">
          <div className="rounded-md border border-zinc-800 p-2 text-center">
            <div className="font-mono text-[9px] uppercase text-zinc-600">conv out</div>
            <div className="font-mono text-sm text-rose-300">{convOut(position).toFixed(2)}</div>
          </div>
          <div className="rounded-md border border-zinc-800 p-2 text-center">
            <div className="font-mono text-[9px] uppercase text-zinc-600">attn out</div>
            <div className="font-mono text-sm text-emerald-300">
              {signal.reduce((s, v, i) => s + attended[i] * v, 0).toFixed(2)}
            </div>
          </div>
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        At σ≈0.5 attention ≈ conv (local). At σ≫1 it approaches a global average that
        ignores content. The learnable sweet spot — content-adaptive dilation of the window —
        is exactly what area/deformable attention approximate in YOLOv12 and RT-DETR.
      </p>
    </Sim>
  );
}

export function TransformerQuerySim() {
  const [q, setQ] = useState(0.5);
  const [contentMix, setContentMix] = useState(0.6);

  const fmap = [0.1, 0.9, 0.7, 0.2];
  const posEnc = [0.2, 0.4, 0.3, 0.1];

  const keys = fmap.map((f, i) => [contentMix * f + (1 - contentMix) * posEnc[i]]);
  const { weights } = attention([q], keys, keys.map((_, i) => [i / 3]));

  const boxDecode = weights.reduce((s, w, i) => s + w * (i / 3), 0);

  return (
    <Sim
      name="Transformer query → box"
      identifies={`Query weighs feature cells and reads out a continuous box coordinate (decoded at ${boxDecode.toFixed(0)}/${boxDecode.toFixed(1)}/3 of the axis).`}
      math={[
        "each query is a learned embedding representing one object slot",
        "query softly selects features (content + position encoding)",
        "a regression head maps the aggregated features to box coordinates directly",
      ]}
      need="Conv heads need anchors + NMS to map heatmaps to boxes. A transformer query is the anchor’s learned successor: one query, one object, one direct decode — the foundation of NMS-free, end-to-end detection in RT-DETR and D-FINE."
      real="RT-DETR queries; DETR-style set loss with Hungarian shown in the next sim; D-FINE decodes continuous offsets from discrete distributions instead of a single scalar."
      papers={["P029", "P037", "P038"]}
      params={
        <>
          <Slider label="query slot embedding" value={q} min={0} max={1} step={0.05} onChange={setQ} />
          <Slider label="content vs positional" value={contentMix} min={0} max={1} step={0.05} onChange={setContentMix} />
        </>
      }
    >
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 p-3">
        <div className="mb-2 font-mono text-[9px] uppercase text-zinc-500">
          feature cells across the image axis (content in green, position in amber)
        </div>
        <div className="relative flex h-16 items-end gap-2">
          {fmap.map((f, i) => (
            <div key={i} className="relative flex-1">
              <div className="w-full rounded-t bg-emerald-600/70" style={{ height: `${f * 60}px` }} />
              <div
                className="absolute bottom-0 w-full rounded-t bg-amber-500/50"
                style={{ height: `${posEnc[i] * 60}px` }}
              />
              <div className="mt-0.5 text-center font-mono text-[8px] text-zinc-600">{i}</div>
            </div>
          ))}

          {/* query weight line */}
          <div className="pointer-events-none absolute bottom-0 left-0 right-0">
            <div className="relative h-0">
              <svg viewBox="0 0 100 10" className="w-full" preserveAspectRatio="none">
                <polyline
                  points={weights.map((_, i) => `${i * (100 / 4)},${10 - weights[i] * 10}`).join(" ")}
                  fill="none"
                  stroke="#34d399"
                  strokeWidth={1.5}
                />
              </svg>
            </div>
          </div>
        </div>
        <div className="mt-2 flex justify-between font-mono text-[9px] text-zinc-600">
          <span>left edge</span>
          <span>decoded object center → {boxDecode.toFixed(1)}/3</span>
          <span>right edge</span>
        </div>
        <div className="mt-1 h-1.5 w-full rounded bg-zinc-800">
          <div
            className="h-1.5 rounded bg-emerald-400"
            style={{ width: `${(boxDecode / 1) * 100}%` }}
          />
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-5 text-zinc-500">
        Move the query embedding or shift the content/position trade — the decoded center
        glides. This “soft anchor” is why transformer detectors skip anchor tiling and NMS.
      </p>
    </Sim>
  );
}