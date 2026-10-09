export type SimKey =
  | "congestion-control"
  | "routing"
  | "cdn-caching"
  | "iot-networking"
  | "queueing"
  | "anomaly-detection"
  | "traffic-analysis"
  | "dc-load-balancing"
  | "interference";

export interface SimMeta {
  key: SimKey;
  label: string;
  blurb: string;
  /** The identical-problem cluster this simulator models. */
  clusterDomain: string;
}

export const SIM_META: Record<SimKey, SimMeta> = {
  "congestion-control": {
    key: "congestion-control",
    label: "Congestion Control",
    blurb:
      "Bottleneck queue with competing senders: watch AIMD, delay-based and RL-style windows behave under the same load.",
    clusterDomain: "congestion-control",
  },
  routing: {
    key: "routing",
    label: "Routing & Forwarding",
    blurb:
      "Shortest-path vs traffic-engineered forwarding on a small topology; compare how each spreads load.",
    clusterDomain: "routing",
  },
  "cdn-caching": {
    key: "cdn-caching",
    label: "CDN & Caching",
    blurb:
      "Request stream against an edge cache: hit rate vs cache size under LRU and LFU.",
    clusterDomain: "cdn-caching",
  },
  "iot-networking": {
    key: "iot-networking",
    label: "IoT Access",
    blurb:
      "Massive random access: many devices, one channel — see collisions rise with population and traffic.",
    clusterDomain: "iot-networking",
  },
  queueing: {
    key: "queueing",
    label: "Queueing at a Link",
    blurb:
      "M/M/1 queue: utilisation vs delay trade-off, the structural reason latency-sensitive transports target low occupancy.",
    clusterDomain: "transport-protocols",
  },
  "anomaly-detection": {
    key: "anomaly-detection",
    label: "Intrusion Detection",
    blurb:
      "Threshold detector on flow features: precision/recall trade-off as the threshold moves.",
    clusterDomain: "network-security",
  },
  "traffic-analysis": {
    key: "traffic-analysis",
    label: "Traffic Classification",
    blurb:
      "Feature separation between traffic classes and what noise does to a classifier's margin.",
    clusterDomain: "traffic-analysis",
  },
  "dc-load-balancing": {
    key: "dc-load-balancing",
    label: "Datacenter Load Balancing",
    blurb:
      "Flow completion time under per-flow vs per-packet spreading on a Clos fabric.",
    clusterDomain: "datacenter-networking",
  },
  interference: {
    key: "interference",
    label: "Wireless Interference",
    blurb:
      "Shared spectrum: how many links fit before aggregate goodput collapses.",
    clusterDomain: "wireless",
  },
};

export const SIMS: Record<string, SimMeta> = SIM_META;
