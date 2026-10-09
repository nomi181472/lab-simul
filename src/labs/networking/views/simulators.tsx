"use client";

import { useState } from "react";
import { Card, SectionTitle, Badge, Quote, PaperLink } from "@/labs/networking/ui";
import { ANALYSIS } from "@/labs/networking/data/analysis";
import { SIM_META, type SimKey } from "@/labs/networking/sims/registry";
import {
  CongestionControlSim,
  RoutingSim,
  CdnCachingSim,
  IotAccessSim,
  QueueingSim,
  AnomalyDetectionSim,
  DcLoadBalancingSim,
  InterferenceSim,
  TrafficAnalysisSim,
} from "@/labs/networking/sims/models";

const COMPONENTS: Record<SimKey, React.ComponentType> = {
  "congestion-control": CongestionControlSim,
  routing: RoutingSim,
  "cdn-caching": CdnCachingSim,
  "iot-networking": IotAccessSim,
  queueing: QueueingSim,
  "anomaly-detection": AnomalyDetectionSim,
  "traffic-analysis": TrafficAnalysisSim,
  "dc-load-balancing": DcLoadBalancingSim,
  interference: InterferenceSim,
};

const ORDER: SimKey[] = [
  "congestion-control",
  "routing",
  "cdn-caching",
  "iot-networking",
  "queueing",
  "anomaly-detection",
  "traffic-analysis",
  "dc-load-balancing",
  "interference",
];

export function SimulatorsView() {
  const [active, setActive] = useState<SimKey>("congestion-control");
  const meta = SIM_META[active];
  const Comp = COMPONENTS[active];

  const cluster = ANALYSIS.clusters.find((c) => c.simulatorKey === active);

  return (
    <div className="space-y-6">
      <SectionTitle>Interactive simulators</SectionTitle>

      <Card tone="accent">
        <p className="text-[13px] leading-6 text-zinc-400">
          Each simulator is a minimal model of one identical-problem cluster — the
          repeated problem statements the corpus reports. They are teaching models built
          from the quoted problem statements, not reproductions of any paper&apos;s results;
          no number on these screens is a claim about a specific paper.
        </p>
      </Card>

      <div className="flex flex-wrap gap-1">
        {ORDER.map((k) => (
          <button
            key={k}
            onClick={() => setActive(k)}
            className={`rounded-md border px-2.5 py-1.5 text-[11px] font-medium ${
              active === k
                ? "border-sky-700/50 bg-sky-600/20 text-sky-300"
                : "border-zinc-700 bg-zinc-900 text-zinc-400 hover:bg-zinc-800"
            }`}
          >
            {SIM_META[k].label}
          </button>
        ))}
      </div>

      <Card>
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <span className="text-[13px] font-semibold text-zinc-100">{meta.label}</span>
          {cluster && (
            <Badge tone="amber">
              {cluster.attemptCount} repeated attempts · {cluster.spanYears[0]}–
              {cluster.spanYears[1]}
            </Badge>
          )}
        </div>
        <p className="mt-1 text-[12px] leading-5 text-zinc-400">{meta.blurb}</p>

        <div className="mt-4">
          <Comp />
        </div>
      </Card>

      {cluster && (
        <Card>
          <p className="mb-2 font-mono text-[10px] uppercase tracking-wide text-zinc-500">
            the real problem this model is built from (quoted)
          </p>
          <div className="mb-1">
            <PaperLink id={cluster.corePaperId} />
          </div>
          <Quote>{cluster.coreStatement}</Quote>
        </Card>
      )}
    </div>
  );
}
