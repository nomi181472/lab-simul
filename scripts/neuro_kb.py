#!/usr/bin/env python3
"""Curated knowledge base for the Neuroevolution lab, emitted as typed records.

Concept, metric and dataset entries are hand-written background material, not
paper extracts: they exist so the Foundations and Metrics views can explain a
term without quoting anyone. What *is* corpus-derived is the linkage — each
entry gets `paperIds` from the generated records that actually reference it.

`characteristic` strings on datasets are the only paper-derived text here, and
they are filled by neuro_records.py from the PDF; they stay empty until then.
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------- concepts

# (id, name, category, intuition, formula, variables, prereqs, simulator)
CONCEPTS = [
    ("neuroevolution", "Neuroevolution", "basics",
     "Evolutionary search whose individual *is* a neural artefact: its weights, its topology, or both. Nothing is learned by backpropagation; the population is the optimiser.",
     None, None, [], "ga-demo"),
    ("genotype", "Genotype", "encoding",
     "The thing the operators actually touch. It need not look like the network: it can be a weight vector, an adjacency list, or a tree that grows into one.",
     None, None, ["neuroevolution"], "encoding-demo"),
    ("phenotype", "Phenotype", "encoding",
     "The evaluated artefact, produced by decoding the genotype. When decoding involves a developmental process, the mapping is part of the algorithm and a place where bias enters.",
     None, None, ["genotype"], "encoding-demo"),
    ("fitness", "Fitness", "basics",
     "The scalar the population is sorted by. A deliberately noisy, delayed or sparse approximation of the thing you actually want; every trick in this field is about making that approximation less bad.",
     "fitness(i) = reward(decide(phenotype(i)))", [("i", "genotype index"), ("reward", "environment or objective score")],
     ["neuroevolution"], "ga-demo"),
    ("population", "Population", "basics",
     "The set of candidates evaluated each generation. Its size is the search's main parallelism knob and the usual knob for total cost, since cost is roughly population x generations.",
     None, None, ["neuroevolution"], "ga-demo"),
    ("genetic-algorithm", "Genetic Algorithm", "basics",
     "Selection, crossover and mutation over a fixed-length or tree-shaped genotype. The reference implementation of the field and the baseline almost every paper compares against.",
     None, None, ["population", "fitness"], "ga-demo"),
    ("mutation", "Mutation", "operators",
     "Perturbation with injected randomness; the only operator that creates information the previous population did not have, so it is what prevents premature convergence.",
     "x' = x + sigma * N(0, I)", [("sigma", "step size"), ("N(0, I)", "isotropic Gaussian")],
     ["genetic-algorithm"], "ga-demo"),
    ("crossover", "Crossover", "operators",
     "Recombination of two parents' genes. It exploits structure already found, but it cannot invent anything new, which is why mutation is never optional.",
     None, None, ["genetic-algorithm"], "ga-demo"),
    ("selection", "Selection", "selection",
     "Choosing who survives and reproduces. Strong selection converges fast and then stops exploring; the classic failure is the population collapsing onto one solution.",
     None, None, ["fitness"], "ga-demo"),
    ("elitism", "Elitism", "selection",
     "Copying the best individual into the next generation untouched. Guarantees the search never regresses, at the cost of locking in whatever early bias the fitness function has.",
     None, None, ["selection"], "ga-demo"),
    ("evolution-strategy", "Evolution Strategy", "basics",
     "A (mu, lambda) or (mu + lambda) scheme that applies Gaussian perturbation to parameters. Clean, mutation-only, and it turns out to admit a gradient-estimation interpretation.",
     "m_theta = (1/lambda) * sum_i F(theta + sigma*eps_i) * eps_i", [("lambda", "number of perturbed samples"), ("eps_i", "Gaussian noise")],
     ["mutation"], "ga-demo"),
    ("cma-es", "CMA-ES", "basics",
     "Evolution strategy that learns the shape of its sampling distribution, adapting a covariance matrix. This is why it scales to high-dimensional weight vectors where plain ES stalls.",
     None, None, ["evolution-strategy"], "ga-demo"),
    ("genetic-programming", "Genetic Programming", "encoding",
     "Evolution over program trees under a grammar. Structure can grow and shrink, so the search changes the hypothesis class, not just the numbers in a fixed one.",
     None, None, ["genotype", "mutation"], None),
    ("neat", "NEAT / HyperNEAT", "encoding",
     "Topology itself is encoded as a list of (node, gene) pairs, so adding a node and wiring its links are ordinary mutation operators. The direct answer to 'is architecture search possible without gradients?'.",
     None, None, ["genotype", "mutation"], "topology-demo"),
    ("indirect-encoding", "Direct vs Indirect Encoding", "encoding",
     "Direct: one gene per weight. Indirect: a compact set of rules that generate the weights. Indirect encodings can produce far smaller search spaces and smoother structure, but they constrain what is reachable.",
     None, None, ["genotype", "phenotype"], "encoding-demo"),
    ("quality-diversity", "Quality-Diversity (MAP-Elites)", "diversity",
     "Search that returns a repertoire: one solution per cell of a behaviour-space grid, keeping the best in each. It answers 'find every good solution' rather than 'find the best solution'.",
     "fitness(x) = objective(x) * (1 + behaviour_diversity(x))^w", [("w", "diversity weight")],
     ["fitness", "novelty"], "qd-demo"),
    ("novelty-search", "Novelty Search", "diversity",
     "Fitness is behavioural novelty against recent population members, not task reward. Necessary when the reward is unknown, deceptive, or too sparse to guide search.",
     None, None, ["fitness"], "qd-demo"),
    ("open-ended-evolution", "Open-Ended Evolution", "diversity",
     "Search with no fixed goal, sustained by novelty and changing environments. Tests whether an evolutionary system keeps generating new structure indefinitely.",
     None, None, ["novelty-search", "aging"], None),
    ("aging", "Aging / Complexity Pressure", "diversity",
     "Selection pressure that rewards solutions for remaining viable as complexity accumulates. Without it, simple solutions always win and interesting structure never appears.",
     None, None, ["selection"], None),
    ("fitness-shaping", "Fitness Shaping", "basics",
     "Transforming a sparse, delayed or deceptive objective into the selection signal the search can climb. Often the difference between a working algorithm and a non-working one.",
     None, None, ["fitness"], "ga-demo"),
    ("locus", "Mutation Locus", "operators",
     "Which part of the artefact an operator is allowed to change: weights, topology, connections, learning rate, activation. A search is best described by where it may change things.",
     None, None, ["mutation", "genotype"], "modification-demo"),
    ("nas", "Neural Architecture Search", "applications",
     "Automating architecture design. Evolutionary NAS is the family here; its competitors are gradient-based and RL-based search, and it still has to beat them on cost.",
     None, None, ["neat", "mutation"], "modification-demo"),
    ("neurocontroller", "Neurocontroller", "applications",
     "A network whose weights are evolved to be a robot controller or agent policy. The classic embodied setting: no gradient flows through the environment, so search does the work.",
     None, None, ["phenotype"], None),
    ("modular-neuroevolution", "Modular Neuroevolution", "encoding",
     "Evolving a network of reusable modules whose boundaries can be inherited, so crossover swaps whole sub-networks instead of individual genes.",
     None, None, ["genotype", "crossover"], None),
]

# ---------------------------------------------------------------- metrics

# (id, name, family, formula, variables, meaning, intuition, example, limitations)
METRICS = [
    ("mean-reward", "Mean episodic return", "performance",
     "R = (1/N) * sum_{i=1..N} sum_t r_t^(i)",
     [("N", "episodes"), ("r_t", "reward at step t")],
     "Average task reward over N episodes.",
     "The headline number for control papers. With a gradient-free search it is also the landscape the search is walking, so reward shaping decisions show up here first.",
     "'mean reward 512 +/- 74 after 500 generations'",
     ["Averages hide the per-seed spread; report the spread too.",
      "Reward scale differs per environment, so cross-paper comparison needs the benchmark named."]),
    ("success-rate", "Task success rate", "performance",
     "P(success) = successes / trials",
     None,
     "Fraction of trials meeting the task's success condition.",
     "More informative than average reward when reward is unbounded but success is binary, which is common in manipulation tasks.",
     "'solved the task in 94% of 100 trials'",
     ["Depends entirely on the paper's success threshold.",
      "Can saturate at 100%, hiding differences in the harder tail."]),
    ("final-fitness", "Best / final fitness", "fitness",
     "F* = max_t fitness(t)",
     None,
     "The best individual the search ever evaluated.",
     "The search's own scorecard, independent of whether the resulting network is any good.",
     "'best fitness found: 0.98 after 200 generations'",
     ["Not comparable across papers with different fitness functions.",
      "Increasing monotonically by construction, so it says nothing about reliability."]),
    ("evaluations", "Fitness evaluations", "efficiency",
     "cost ~ population_size * generations",
     None,
     "How many candidates were scored in total.",
     "The fair cost unit, since evolutionary search is embarrassingly parallel and wall-clock is machine-dependent.",
     "'180,000 evaluations on 8 GPUs'",
     ["One evaluation can cost anywhere from microseconds to GPU-hours.",
      "Papers that report generations rather than evaluations hide population size."]),
    ("generations", "Generations", "efficiency",
     "G = iterations until stopping rule",
     None,
     "Number of population iterations.",
     "Useful only alongside population size; on its own it is not a cost measure.",
     "'500 generations, population 100'",
     ["Not comparable across papers with different population sizes."]),
    ("swept-areas", "Swept areas", "diversity",
     "coverage = |{cells with an elite}| / |cells|",
     None,
     "How much of the behaviour grid has been filled with a solution.",
     "Quality-diversity's answer to 'did it explore?', reported as a percentage of the archive.",
     "'swept 62% of the 2-D behaviour descriptor space'",
     ["Depends entirely on the descriptor chosen; not comparable across papers.",
      "A large archive of bad solutions scores well."]),
    ("qd-score", "QD-score / archive size", "diversity",
     "QD = sum over cells of (objective * (1 + d(w)))",
     None,
     "Coverage-weighted quality of a quality-diversity archive.",
     "Improves on raw coverage, which rewards filling cells with poor solutions.",
     "'QD-score 1.42 versus QD-score 1.05 for the baseline'",
     ["Requires a fixed behaviour descriptor and grid resolution to reproduce."]),
    ("l1-error", "L1 / RMSE error", "performance",
     "RMSE = sqrt( (1/n) * sum_i (y_i - y_hat_i)^2 )",
     None,
     "Numeric error against ground truth.",
     "The accuracy measure for symbolic regression and evolved-program papers.",
     "'RMSE 0.012 on the Nguyen benchmark'",
     ["Sensitive to outliers; papers must state whether they report median or mean."]),
    ("accuracy", "Accuracy / F1", "performance",
     "accuracy = correct / total",
     None,
     "Classification accuracy or F1.",
     "Used when the evolved model is evaluated on a classification task.",
     "'94.1% accuracy, 0.918 F1'",
     ["Accuracy alone is misleading on imbalanced datasets."]),
    ("inference-cost", "Inference cost", "efficiency",
     None,
     [("params", "parameter count"), ("FLOPs", "floating point ops per forward pass")],
     "Parameter count or FLOPs of the evolved network.",
     "What architecture search is really buying: a cheaper network at equal accuracy, not a better search story.",
     "'1.2M parameters, 41% fewer FLOPs than the baseline'",
     ["A smaller network is only valuable if accuracy holds; the two must be reported together."]),
    ("wall-clock", "Wall-clock time", "efficiency",
     None,
     None,
     "Elapsed time for the whole search.",
     "Evolutionary search parallelises, so wall-clock and evaluation count can disagree by orders of magnitude.",
     "'11 GPU-hours on 8 GPUs'",
     ["Hardware-dependent and rarely comparable across papers.",
      "Understates cost when the population is small and idle."]),
    ("coverage", "Behaviour coverage", "diversity",
     None,
     None,
     "How widely the search spread over behaviour space.",
     "Separates 'found one solution' from 'mapped the space', which novelty and QD papers both need.",
     "'covered 3 distinct locomotion gaits'",
     ["Only meaningful with the descriptor or gait taxonomy stated."]),
]

# ---------------------------------------------------------------- datasets

DATASETS = [
    ("1707.06347", "Evolution Gym", 2020, "benchmark-suite", "locomotion",
     "Benchmarks where the *task* is mutated, forcing the architecture itself to change between environments.",
     ["mutating-environments", "transfer"]),
    ("1806.09091", "MuJoCo continuous control", 2018, "benchmark-suite", "locomotion",
     "The default continuous-control suite for evolved controllers; versions differ in actuator and contact model.",
     ["continuous-control", "simulated"]),
    ("1802.01548", "Quality-diversity benchmark tasks", 2018, "benchmark-suite", "locomotion",
     "Standard hand-designed QD arenas, including the Arm and PointMaze tasks with known coverage ceilings.",
     ["quality-diversity", "behaviour-descriptor"]),
    ("1210.1053", "Gripper and Ball-in-cup", 2012, "benchmark-task", "manipulation",
     "Small analytic tasks where the theoretical QD coverage ceiling is known, which makes them diagnostic.",
     ["quality-diversity", "analytic"]),
    ("2007.14199", "Humanoid control tasks", 2020, "benchmark-suite", "locomotion",
     "Higher-dimensional humanoid tasks used to show whether an evolved method still scales.",
     ["continuous-control", "high-dimensional"]),
    ("1906.09656", "Ant / PointMass map-elites tasks", 2019, "benchmark-task", "locomotion",
     "Small arenas used to visualise behaviour-space coverage directly.",
     ["quality-diversity", "visualisable"]),
    ("1706.06670", "Sequence-to-sequence translation", 2017, "benchmark-suite", "sequence",
     "Translation benchmarks for architecture search and RNN/GRU evolution.",
     ["sequence", "architecture-search"]),
    ("1505.02397", "CIFAR-10 / CIFAR-100", 2015, "dataset", "vision",
     "Image classification used to compare search strategies at a fixed evaluation budget.",
     ["vision", "accuracy"]),
    ("2202.06502", "Symbolic regression benchmark suite", 2022, "benchmark-suite", "symbolic-regression",
     "The standard GP target functions, reported as error tables against an evolved-program baseline.",
     ["genetic-programming", "error-tables"]),
    ("2106.07597", "Ant morphology suite", 2021, "benchmark-suite", "locomotion",
     "Environment variation across morphologies, where co-evolution or novelty search adapts.",
     ["open-ended", "morphology"]),
    ("2006.14139", "PyBullet walker tasks", 2020, "benchmark-suite", "locomotion",
     "Continuous-control walkers evolved without gradients.",
     ["continuous-control", "simulated"]),
    ("1806.03762", "MNIST", 1998, "dataset", "vision",
     "Digit classification, used mostly as a sanity check that the search still trains at all.",
     ["vision", "sanity-check"]),
]


def _vars(v):
    if not v:
        return None
    return [{"symbol": a, "meaning": b} for a, b in v]


def emit(out_dir, papers) -> None:
    """Write concepts.ts, metrics.ts, datasets.ts, linking each to corpus papers."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_concept: dict[str, list[str]] = {}
    by_metric: dict[str, list[str]] = {}
    by_dataset: dict[str, list[str]] = {}
    ds_char: dict[str, list[str]] = {}
    for p in papers:
        for c in p.get("concepts", []):
            by_concept.setdefault(c, []).append(p["id"])
        for m in p.get("metrics", []):
            by_metric.setdefault(m, []).append(p["id"])
        for d in p.get("datasets", []):
            by_dataset.setdefault(d, []).append(p["id"])
            body = p.get("method", {}).get("evaluation", "")
            if body and body != "not stated in the retrieved text" and len(ds_char.get(d, [])) < 3:
                ds_char.setdefault(d, []).append(body)

    def j(x) -> str:
        return json.dumps(x, ensure_ascii=False)

    # ---- concepts
    L = [
        "/* GENERATED by scripts/neuro_kb.py — curated background, not paper extracts.",
        " * paperIds are corpus-derived: the papers that reference the concept.",
        " */",
        "",
        'import type { Concept } from "./types";',
        "",
        "export const CONCEPTS: Concept[] = [",
    ]
    for cid, name, cat, intu, formula, variables, prereqs, sim in CONCEPTS:
        parts = [f'    id: {j(cid)},', f'    name: {j(name)},', f'    category: {j(cat)},', f'    intuition: {j(intu)},']
        if formula:
            parts.append(f"    formula: {j(formula)},")
        if variables:
            parts.append(f"    variables: {j(_vars(variables))},")
        if sim:
            parts.append(f"    simulator: {j(sim)},")
        parts.append(f"    prereqs: {j(prereqs)},")
        pids = sorted(set(by_concept.get(cid, [])))
        if pids:
            parts.append(f"    paperIds: {j(pids[:40])},")
        L.append("  { " + " ".join(parts) + " },")
    L += ["];", "", "export const CONCEPT_BY_ID: Record<string, Concept> = Object.fromEntries(",
          "  CONCEPTS.map((c) => [c.id, c]),", ");", ""]
    (out_dir / "concepts.ts").write_text("\n".join(L))

    # ---- metrics
    M = [
        "/* GENERATED by scripts/neuro_kb.py — metric definitions with their traps. */",
        "",
        'import type { MetricRecord } from "./types";',
        "",
        "export const METRICS: MetricRecord[] = [",
    ]
    for mid, name, fam, formula, variables, meaning, intu, example, lims in METRICS:
        parts = [
            f'    id: {j(mid)},',
            f'    name: {j(name)},',
            f'    family: {j(fam)},',
        ]
        parts.append(f"    formula: {j(formula or name)},")
        if variables:
            parts.append(f"    variables: {j(_vars(variables))},")
        parts += [
            f"    meaning: {j(meaning)},",
            f"    intuition: {j(intu)},",
            f"    example: {j(example)},",
            f"    limitations: {j(lims)},",
        ]
        pids = sorted(set(by_metric.get(mid, [])))
        parts.append(f"    paperIds: {j(pids[:40])},")
        M.append("  { " + " ".join(parts) + " },")
    M += ["];", "", "export const METRIC_BY_ID: Record<string, MetricRecord> = Object.fromEntries(",
          "  METRICS.map((m) => [m.id, m]),", ");", ""]
    (out_dir / "metrics.ts").write_text("\n".join(M))

    # ---- datasets
    D = [
        "/* GENERATED by scripts/neuro_kb.py — benchmark descriptions.",
        " * `characteristics` quotes corpus PDFs; `paperIds` lists the papers that used it.",
        " */",
        "",
        'import type { DatasetRecord } from "./types";',
        "",
        "export const DATASETS: DatasetRecord[] = [",
    ]
    for arxiv, name, year, kind, task, purpose, tags in DATASETS:
        parts = [
            f'    id: {j(arxiv)},',
            f'    name: {j(name)},',
            f"    year: {year},",
            f'    purpose: {j(purpose)},',
            f'    domain: {j(kind)},',
            f'    task: {j(task)},',
            f'    characteristics: {j(ds_char.get(arxiv, [])[:3])},',
        ]
        pids = sorted(set(by_dataset.get(arxiv, [])))
        parts.append(f"    paperIds: {j(pids)},")
        # which corpus metrics papers evaluating this benchmark reported
        seen_m: list[str] = []
        for pid in pids:
            for mid in next((p.get("metrics", []) for p in papers if p["id"] == pid), []):
                if mid not in seen_m:
                    seen_m.append(mid)
        parts.append(f"    metrics: {j(seen_m[:6])},")
        D.append("  { " + " ".join(parts) + " },")
    D += ["];", "", "export const DATASET_BY_ID: Record<string, DatasetRecord> = Object.fromEntries(",
          "  DATASETS.map((d) => [d.id, d]),", ");", ""]
    (out_dir / "datasets.ts").write_text("\n".join(D))

    print(
        f"kb: {len(CONCEPTS)} concepts, {len(METRICS)} metrics, {len(DATASETS)} datasets; "
        f"linked {sum(len(v) for v in by_dataset.values())} dataset references"
    )

# ---------------------------------------------------------------- link terms

# Corpus -> concept/metric linking. Kept separate from the display text above so
# a reworded intuition never silently changes which papers a concept claims.
# Matching is against the paper's own head text (title + abstract + opening), so
# a link means the paper itself uses the term, not that a reviewer thought so.
CONCEPT_MATCH: dict[str, tuple[str, ...]] = {
    "neuroevolution": ("neuroevolution", "neuro-evolution"),
    "genotype": ("genotype", "genome", "encoding", "chromosome"),
    "phenotype": ("phenotype", "decoded network", "individual"),
    "fitness": ("fitness", "objective function", "reward"),
    "population": ("population", "generations"),
    "genetic-algorithm": ("genetic algorithm", "ga", "elitism"),
    "mutation": ("mutation", "mutate", "mutating"),
    "crossover": ("crossover", "recombination", "recombine"),
    "selection": ("selection", "tournament", "truncation", "survival"),
    "elitism": ("elitism", "elite"),
    "evolution-strategy": ("evolution strateg", "es algorithm", "natural evolution strategies"),
    "cma-es": ("cma-es", "covariance matrix adaptation", "cmaes"),
    "genetic-programming": ("genetic programming", "symbolic regression", "subtree", "adtf"),
    "neat": ("neat", "augmenting topologies", "hyper-neat", "hyperneat", "innovation number"),
    "indirect-encoding": ("indirect encoding", "direct encoding", "developmental"),
    "quality-diversity": ("map-elites", "map elites", "quality diversity", "quality-diversity", "illumination", "archive cell", "swept areas", "qd-score"),
    "novelty-search": ("novelty search", "novelty archive", "novelty", "behavioural novelty"),
    "open-ended-evolution": ("open-ended", "open ended", "unbounded", "open-endedness"),
    "aging": ("aging", "age of structure", "complexity pressure"),
    "fitness-shaping": ("fitness shaping", "surrogate fitness", "reward shaping", "sparse reward"),
    "locus": ("locus", "loci"),
    "nas": ("architecture search", "neural architecture search", "nas"),
    "neurocontroller": ("neurocontroller", "neuro-controller", "controller", "locomotion", "policy"),
    "modular-neuroevolution": ("modular", "module", "modularity"),
}

METRIC_MATCH: dict[str, tuple[str, ...]] = {
    "mean-reward": ("mean reward", "average reward", "episodic return", "average return", "cumulative reward"),
    "success-rate": ("success rate", "solved", "completion rate"),
    "final-fitness": ("best fitness", "highest fitness", "final fitness",
                      "maximum fitness", "best individual", "fitness of the best"),
    "evaluations": ("fitness evaluations", "number of evaluations", "objective evaluations", "evaluations"),
    "generations": ("generations", "number of generations"),
    "swept-areas": ("swept areas", "swept", "archive cells", "cells filled"),
    "qd-score": ("qd-score", "qd score", "archive size"),
    "l1-error": ("l1 error", "rmse", "root mean squared", "mean squared error", "mse", "absolute error"),
    "accuracy": ("accuracy", "f1", "precision", "recall"),
    "inference-cost": ("inference cost", "flops", "parameter count", "model size", "parameters"),
    "wall-clock": ("wall-clock", "wall clock", "runtime", "computational cost", "hours", "gpu"),
    "coverage": ("coverage", "explored"),
}
