/** Simulator contracts for the Transformers lab. */
export interface SimulatorBinding {
  id: string;
  label: string;
  papers: string[];
}

export interface SimPoint {
  label: string;
  value: number;
}

export interface SimSeries {
  key: string;
  label: string;
  points: SimPoint[];
}

/** One step of a simulator run, kept quotable so the UI can show its work. */
export interface SimStep {
  t: number;
  action: string;
  detail: string;
}

export interface SimRun {
  id: string;
  label: string;
  description: string;
  series: SimSeries[];
  steps: SimStep[];
  /** Papers this simulator is grounded in. */
  evidence: string[];
}