#!/usr/bin/env python3
"""Generate deep Neuroevolution paper records.

Same provenance policy as the other labs: every field is grounded in the corpus
PDF named by `fileName`, results only carry numbers the paper printed, and an
unstated field stays empty so the UI can say so.

What this lab adds is `architecture`, which is required on every record. Two
things get extracted per paper, both with quoted evidence:

  1. the evolved network's layer list, when the paper enumerates it, so the
     Architecture view can draw something rather than show a placeholder;
  2. the loci the operators actually change (weights, topology, connections,
     learning rate, ...), which is the "how is the modification being done"
     axis the Modification view is built around.

Layer and operator detection reuses the rule tables in scripts/neuro_index.py
so the manifest classifier and the record generator cannot drift apart.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import neuro_index as X  # noqa: E402
import neuro_kb as KB  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "papers" / "neuroevolution"
CACHE = ROOT / ".cache" / "neuro_text"
OUTDIR = ROOT / "src" / "labs" / "neuroevolution" / "data" / "papers"

NOT_DESCRIBED = "not stated in the retrieved text"

# ------------------------------------------------------------------ cues

PROBLEM_CUES = (
    "however", "challenge", "difficult", "hard to", "suffer", "drawback",
    "limitation", "expensive", "intractable", "cannot", "struggle",
    "prohibitively", "bottleneck", "unsuitable", "difficult to",
)

GAP_CUES = (
    "however", "but", "nevertheless", "remains", "unclear", "open question",
    "little work", "few attempts", "has not been", "not been explored",
    "lack", "gap", "limited attention", "underexplored",
)

CONTRIB_CUES = (
    "we propose", "we present", "we introduce", "this paper", "in this paper",
    "we develop", "we describe", "our contribution", "contributions of this",
    "we show", "we demonstrate", "we introduce a", "we design",
    "the main contribution", "novelty of this",
)

RESULT_CUES = (
    "achieve", "achieves", "obtained", "outperform", "outperforms", "improve",
    "improves", "improve over", "reduces", "reduce", "error of", "accuracy of",
    "reaches", "yields", "demonstrates", "fitness of", "final reward",
    "average reward", "success rate", "mean error", "rmse", "f1", "auc",
    "surpasses", "better than", "state-of-the-art", "state of the art",
)

ABLATION_CUES = (
    "ablation", "we ablate", "sensitivity", "sensitive to", "we vary",
    "we compare population", "effect of", "impact of population",
    "number of generations", "we investigate the effect", "varying",
)

LIMIT_CUES = ("limitation", "drawback", "fails", "cannot", "struggles", "future work", "does not")

ASSUME_CUES = ("assume", "assumption", "we assume", "is assumed", "subject to")

COST_CUES = (
    "population size", "number of generations", "evaluations", "gpu", "cpu",
    "hours", "days", "runtime", "computational cost", "wall-clock",
    "parallel", "fitness evaluations", "converge",
)

FITNESS_CUES = (
    "fitness function", "fitness is", "objective function", "we maximize",
    "we minimize", "reward function", "objective is", "fitness measure",
    "fitness of", "fitness value", "fitness signal", "reward is",
)

# Results must be quoted from the experiments section. Scanning the whole body
# lets the introduction's "prior work improves accuracy" sentences become this
# paper's results, which is the single easiest way to fabricate a finding.
RESULTS_SECTION_RE = re.compile(
    r"\n\s*(?:\d+(?:\.\d+)*\s*[.)]?\s*)?"
    r"(?:RESULTS?|EXPERIMENTS?|EVALUATION|EXPERIMENTAL RESULTS|DISCUSSION"
    r"|RESULTS AND DISCUSSION|EMPIRICAL (?:RESULTS|EVALUATION))\b[^\n]*\n",
    re.I,
)
END_SECTION_RE = re.compile(
    r"\n\s*(?:\d+(?:\.\d+)*\s*[.)]?\s*)?"
    r"(?:CONCLUSION|CONCLUSIONS|DISCUSSION|REFERENCES|APPENDIX|LIMITATIONS"
    r"|FUTURE WORK|ACKNOWLEDG)",
    re.I,
)

# Title/author blocks must never become a field value: they are the paper's own
# title with affiliation superscripts and email addresses.
TITLEBLOCK_RE = re.compile(
    r"(\d\s*,\s*\d|@|\bAbstract\b|\bPreprint\b|\bunder review\b"
    r"|\bProceedings of\b|\bIEEE\b|\bACM\b|\bsubmitted to\b)"
    r"|^.{0,110}$\s+\w+\s+\w+(\s+\d+)?$",
)


def section_sents(text: str, start_re, stop_re) -> list[str]:
    """Sentences of the first section matching start_re, up to the next heading."""
    m = start_re.search(text)
    if not m:
        return []
    rest = text[m.end():]
    stop = stop_re.search(rest)
    if stop:
        rest = rest[: stop.start()]
    return sentences(rest)[:400]


def looks_like_title_block(s: str) -> bool:
    return bool(TITLEBLOCK_RE.search(s)) or len(s.split()) < 7


def _cue_hit(cue: str, low: str) -> bool:
    """Cues are regexes, so `perturb.*gaussian` is not a dead literal."""
    try:
        return bool(re.search(cue, low))
    except re.error:
        return cue in low


def bind_operator(s: str) -> str:
    """The operator this sentence names, else the honest operator *category*.

    A named variant is preferred; when the paper only says "we mutate the
    weights", saying "Mutation operator (variant not named in retrieved text)"
    is both truthful and useful, whereas "operator unnamed" reads as a failure.
    """
    low = s.lower()
    for _kind, label, cues in X.MUTATION_RULES:
        if any(_cue_hit(c, low) for c in cues):
            return label
    for _kind, label, cues in X.CROSSOVER_RULES:
        if any(_cue_hit(c, low) for c in cues):
            return label
    for _kind, label, cues in X.SELECTION_RULES:
        if any(_cue_hit(c, low) for c in cues):
            return label
    for pat, label in FALLBACK_OPERATORS:
        if re.search(pat, low):
            return label
    return "Not stated in retrieved text"


FALLBACK_OPERATORS = (
    (r"crossover|recombin|combine(?:d|s)? (?:the )?two parent", "Crossover (variant not named)"),
    (r"mutat|offspring", "Mutation operator (variant not named)"),
    (r"perturb|jitter|noise|gaussian|resampl|randomly (?:choose|sample|draw)", "Stochastic perturbation"),
    (r"hyperparameter|learning rate|batch size", "Hyperparameter tuning"),
    (r"add|insert|append|new (?:layer|node|neuron)", "Structural growth"),
    (r"remove|delete|prune|drop", "Structural pruning"),
    (r"select|surviv|keep the best|retain", "Selection"),
    (r"evolutionary strategy|natural gradient|rank-?\d|perturbation vector", "Natural-gradient update"),
    (r"archive|map-elites|map elites|repertoire|illumination", "Archive insertion"),
    (r"novelty|behavioural|behavioral distance", "Novelty selection"),
    (r"encode|decode|decoding|encoding|phenotyp|genotyp", "Genotype-phenotype mapping"),
)


PHENOTYPE_RE = re.compile(
    r"network|architecture|controller|policy|genotype|genome", re.I
)

REF_RE = re.compile(r"\b(?:REFERENCES|References|Bibliography)\b\s*\n(.*)$", re.S)
SURNAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")

BASELINE_CUES = (
    "baseline", "compared with", "compare with", "compared against",
    "outperform", "state-of-the-art", "state of the art", "versus", " vs ",
)


# Front matter, publisher metadata and licence blocks are not the paper's
# content. They survive pdftotext as ordinary sentences and otherwise end up
# quoted as if they were findings ("CCS CONCEPTS -- Computing methodologies").
JUNK_RE = re.compile(
    r"CCS CONCEPTS|\bKEYWORDS\b|Index Terms|ACM Reference|Reference Format"
    r"|\bDOI\b|arXiv:\d|This work is licensed|Permission to make"
    r"|Downloaded from|Preprint\. |under review|copyright|©|\bISBN\b"
    r"|Conference on|Proceedings of the \d|https?://|@\w+\.\w+"
    r"|Peer-review|subject descriptors",
    re.I,
)

# Sentences describing what *other* work does, or what people "typically" do.
# These are background, not this paper's operators, so they must not become
# evidence of what the paper modified.
FRAMING_RE = re.compile(
    r"^(?:often|typically|usually|commonly|generally|frequently|in general|many)\b"
    r"|\b(?:approaches|methods|works|algorithms|systems|researchers|authors)\s+"
    r"(?:often|typically|usually|commonly|generally|frequently)\b"
    r"|requires expert (?:knowledge|effort)"
    r"|is (?:an?|the) (?:open|challenging|difficult) problem"
    r"|has been (?:widely |extensively |long )?(?:studied|proposed|explored|investigated)"
    r"|existing (?:approaches|methods|works)"
    r"|prior (?:work|works|approaches)",
    re.I,
)

# Generalising language ("often", "commonly") is only acceptable when the paper
# is talking about itself. "we commonly use lr=0.01" is this paper's operator;
# "learning-rate schedules are often hand-crafted" is not.
GENERALIZING_RE = re.compile(
    r"\b(?:often|typically|usually|commonly|generally|frequently)\b", re.I
)
FIRST_PERSON_RE = re.compile(
    r"\b(?:we|our|ours|us|i)\b|\bthis (?:paper|work|article|study|section)\b",
    re.I,
)
# pdftotext often runs a section heading into the next sentence
RUNON_HEADING_RE = re.compile(
    r"^(?:Related Work|Related Works|Background|Introduction|Conclusion|"
    r"Discussion|Experiments?|Method|Methods|Results|Evaluation)\b",
    re.I,
)


def is_framing(sent: str) -> bool:
    """True when a sentence describes general practice, not this paper's search."""
    if RUNON_HEADING_RE.match(sent.strip()):
        return True
    if FRAMING_RE.search(sent):
        # an explicit framing phrase, unless the sentence is first-person
        return not FIRST_PERSON_RE.search(sent)
    if GENERALIZING_RE.search(sent):
        return not FIRST_PERSON_RE.search(sent)
    return False


# A locus claim only counts if the sentence also reads as search description.
SEARCH_CONTEXT_RE = re.compile(
    r"\b(?:mutat|crossover|recombin|evolv|optimi[sz]|search|operator|add|"
    r"delet|remov|insert|swap|sampl|tune|population|generation|variation|"
    r"select|offspring|perturb|jitter|perturb)\w*",
    re.I,
)


def is_junk(s: str) -> bool:
    return bool(JUNK_RE.search(s))


def sentences(text: str) -> list[str]:
    return [s for s in X.sentences(text) if not is_junk(s)]


def strip_cites(s: str) -> str:
    s = CITE_RE.sub("", s)
    s = re.sub(r"\s*,\s*,+", ", ", s)
    return " ".join(s.split())


CITE_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def pick(sents: list[str], cues, limit: int = 3, minlen: int = 60) -> list[str]:
    out: list[str] = []
    for s in sents:
        low = s.lower()
        if any(c in low for c in cues) and len(s) >= minlen:
            c = strip_cites(s)
            if c not in out:
                out.append(c)
        if len(out) >= limit:
            break
    return out


def first_pick(sents: list[str], cues, minlen: int = 50) -> str:
    p = pick(sents, cues, 1, minlen)
    return p[0] if p else ""


# Where a bibliography starts, and where it stops again. Both are searched on the
# raw extracted text rather than on the sentence list: pdftotext interleaves
# columns, so a paper's references often do not come last -- appendices, figure
# captions and author biographies can follow them. Anchoring on the heading is
# the only layout-independent way to find them.
REF_HEADING_RE = re.compile(
    r"\n[ \t]*(?:\d+(?:\.\d+)*\s*[.)]?\s*)?"
    r"(?:REFERENCES AND NOTES|REFERENCES|REFERENCE LIST|Bibliography|BIBLIOGRAPHY)"
    r"\b[^\n]{0,40}\n",
    re.I,
)
# IEEE style, where entries are numbered and there is no heading to anchor on.
BRACKET_ENTRY_RE = re.compile(r"(?m)^[ \t]*\[(\d{1,3})\][ \t]+[A-Z]")
REF_STOP_RE = re.compile(
    r"\n[ \t]*(?:\d+(?:\.\d+)*\s*[.)]?\s*)?"
    r"(?:APPENDIX|Acknowledg|ACKNOWLEDG|Supplementary|AUTHOR CONTRIBUTIONS|"
    r"CHECKLIST|Broader Impact)",
    re.I,
)


def split_refs(text: str) -> tuple[str, str]:
    """(body text, reference block) for one paper's extracted text."""
    m = REF_HEADING_RE.search(text)
    if m:
        tail = text[m.end():]
        stop = REF_STOP_RE.search(tail)
        if stop:
            tail = tail[: stop.start()]
        return text[: m.start()], tail

    # No heading: fall back to a run of numbered entries. Requires several, and
    # requires them to sit in the second half, so an in-text numbered list or a
    # figure list is not mistaken for a bibliography.
    hits = list(BRACKET_ENTRY_RE.finditer(text))
    if len(hits) >= 8 and hits[0].start() > len(text) * 0.4:
        first = hits[0].start()
        nums = [int(h.group(1)) for h in hits]
        # A real bibliography numbers consecutively; require the run to climb.
        rising = sum(1 for a, b in zip(nums, nums[1:]) if b > a)
        if rising >= len(nums) * 0.6:
            return text[:first], text[first:]

    return text, ""


# ------------------------------------------------------------------ record


def build_record(row: dict, corpus_by_id: dict[str, dict]) -> dict:
    pid = row["id"]
    pdf = CORPUS / row["fileName"]
    text = X.pdf_text(pdf) if pdf.exists() else ""
    head = X.head_text(text)
    body_text, refs_block = split_refs(text)
    body_sents = sentences(body_text)
    head_sents = sentences(head)

    abs_text = X.abstract_of(head)
    intro_sents = body_sents[:80]

    # ---- architecture (required, always present) ----
    layers, arch_quotes = X.detect_layers(intro_sents)
    arch_quotes += [
        s for s in intro_sents if any(c in s.lower() for c in X.FITNESS_CUES)
    ][:1]
    arch_quotes = [strip_cites(s) for s in dict.fromkeys(arch_quotes)][:4]

    phenotype_quote = ""
    for s in intro_sents:
        if PHENOTYPE_RE.search(s) and not looks_like_title_block(s):
            phenotype_quote = strip_cites(s)
            break

    phenotype = row.get("task", "other")

    details = pick(
        intro_sents,
        ("number of layers", "hidden units", "number of neurons", "hidden layer",
         "input layer", "output layer", "neurons in", "width", "depth of",
         "convolutional layer", "recurrent", "lstm", "attention"),
        5,
        50,
    )
    # Encoding is usually described in the method section rather than the
    # introduction, so the intro alone would leave most papers with none.
    genotype = first_pick(
        intro_sents,
        ("genotype", "genome", "genome is", "each gene", "gene encodes",
         "maps to", "encoding", "phenotype", "indirect", "developmental"),
        60,
    ) or first_pick(
        [x for x in body_sents if x not in intro_sents],
        ("genotype", "genome", "genome is", "each gene", "gene encodes",
         "maps to", "encoding", "phenotype", "indirect", "developmental"),
        60,
    )

    architecture = {
        "summary": strip_cites(phenotype_quote) or f"phenotype classified as {phenotype} from the paper's own text",
        "phenotype": phenotype,
        "layers": layers,
        "details": details,
        "genotypeToPhenotype": cap(genotype) if genotype else "",
        "quotes": arch_quotes,
    }

    # ---- evolutionary machinery ----
    search_sents = [s for s in body_sents if len(s) < 400]
    mut = X.detect_rules(search_sents, X.MUTATION_RULES, 5)
    cross = X.detect_rules(search_sents, X.CROSSOVER_RULES, 4)
    sel = X.detect_rules(search_sents, X.SELECTION_RULES, 4)

    # Loci: what the operators actually change. A sentence only supports a locus
    # if it names both the thing (a locus cue) and the search act (an operator or
    # a search step) -- otherwise a passing mention of "connection" in related
    # work would be recorded as this paper modifying its connectivity.
    ranked: list[tuple[int, str, str, str]] = []
    for locus, cues in X.LOCUS_RULES:
        best: tuple[int, str] | None = None
        for sent in search_sents:
            low = sent.lower()
            if not any(re.search(c, sent, re.I) for c in cues):
                continue
            if not SEARCH_CONTEXT_RE.search(low) or is_framing(sent):
                continue
            op = bind_operator(sent)
            named = op != "described by the paper, operator unnamed"
            # a sentence naming the operator outranks one that only implies it
            score = (2 if named else 1, len(sent))
            if best is None or score > best:
                best = score
                best_sent = (sent, op)
        if best is not None:
            sent, op = best_sent
            ranked.append((best[0], locus, op, sent))

    # Name an operator first, then fill remaining slots by locus order.
    ranked.sort(key=lambda t: -t[0])
    changes: list[dict] = []
    for _score, locus, op, sent in ranked:
        if len(changes) >= 6:
            break
        if any(c["locus"] == locus for c in changes):
            continue
        changes.append({"locus": locus, "operator": op, "quote": strip_cites(sent)[:220]})

    algorithm = first_pick(
        intro_sents,
        ("we use", "we employ", "we adopt", "the algorithm", "our algorithm",
         "search procedure", "we apply", "we run", "using"),
        50,
    )
    representation = first_pick(
        intro_sents,
        ("genome", "genotype", "individual is", "each individual",
         "candidate solution", "chromosome", "tree", "network is encoded"),
        50,
    )
    fitness = first_pick(body_sents, FITNESS_CUES, 50)

    evolution = {
        "family": row.get("family", "other"),
        "algorithm": cap(strip_cites(algorithm)),
        "representation": cap(strip_cites(representation)),
        "encoding": row.get("encoding", "other"),
        "mutation": [m["label"] for m in mut],
        "crossover": [c["label"] for c in cross],
        "selection": [s["label"] for s in sel],
        "changes": changes,
        "fitness": cap(strip_cites(fitness)),
        "phenotypeQuote": cap(phenotype_quote),
    }

    # ---- prose fields ----
    summary_src = strip_cites(abs_text) or first_pick(intro_sents, CONTRIB_CUES, 120) or head_sents[0] if head_sents else ""
    summary = summary_src[:700]
    if summary:
        first_stop = re.search(r"(?<=[.!?])\s", summary)
        if first_stop and first_stop.start() > 200:
            summary = summary[: first_stop.start()]

    problem = first_pick(intro_sents, PROBLEM_CUES, 60)
    gap = first_pick(intro_sents, GAP_CUES, 60)
    contributions = pick(intro_sents, CONTRIB_CUES, 6, 50)
    exp_sents = section_sents(text, RESULTS_SECTION_RE, END_SECTION_RE)
    if not exp_sents:
        exp_sents = body_sents  # papers without a detectable heading
    results = [s for s in pick(exp_sents, RESULT_CUES, 8, 40) if re.search(r"\d", s)]
    ablations = pick(exp_sents, ABLATION_CUES, 4, 50)
    author_limits = pick(body_sents, LIMIT_CUES, 4, 50)
    assumptions = pick(body_sents, ASSUME_CUES, 3, 40)
    computation = pick(body_sents, COST_CUES, 5, 40)

    evident: list[str] = []
    if not layers:
        evident.append("The retrieved text never enumerates the network's layers, so the architecture diagram stays empty rather than guessing one.")
    if not changes:
        evident.append("No locus-level modification is described, so the paper's search is not reproducible from its own text alone.")
    if not evolution["fitness"]:
        evident.append("No fitness/objective sentence was found in the retrieved text.")

    # ---- relations ----
    refs = refs_block
    relations: list[dict] = []
    for oid, other in corpus_by_id.items():
        if oid == pid or not refs:
            continue
        if cites(refs, other):
            rel = relation_type_for(pick(body_sents, list(BASELINE_CUES) + ["related", "prior", "existing", "however"], 2))
            relations.append({"to": oid, "type": rel[0], "note": rel[1]})
            if len(relations) >= 8:
                break

    hay = head.lower()
    concepts = [cid for cid, terms in KB.CONCEPT_MATCH.items() if any(m in hay for m in terms)]
    metrics = [mid for mid, terms in KB.METRIC_MATCH.items() if any(w in hay for w in terms)]
    datasets = [d[0] for d in KB.DATASETS if d[1].lower() in hay or d[0] in hay]

    difficulty = "intro"
    if (row.get("citations") or 0) >= 300:
        difficulty = "advanced"
    elif (row.get("citations") or 0) >= 60:
        difficulty = "intermediate"

    short = row["title"]
    short = re.sub(r"^(a|an|the)\s+", "", short, flags=re.I)
    words = short.split()
    short = " ".join(words[:7]) + ("…" if len(words) > 7 else "")

    return {
        "id": pid,
        "arxiv": row.get("arxiv", ""),
        "doi": row.get("doi", ""),
        "title": row["title"],
        "shortTitle": short,
        "year": row.get("year") or 0,
        "authors": row.get("authors") or [],
        "venue": row.get("venue", ""),
        "fileName": row["fileName"],
        "citations": row.get("citations") or 0,
        "pages": row.get("pages") or 0,
        "tags": row.get("tags") or [],
        "difficulty": difficulty,
        "summary": summary or NOT_DESCRIBED,
        "problem": problem or NOT_DESCRIBED,
        "researchGap": gap or NOT_DESCRIBED,
        "background": concepts[:3],
        "contribution": contributions or [NOT_DESCRIBED],
        "architecture": architecture,
        "method": {
            "evolution": evolution,
            "task": architecture["summary"],
            "evaluation": cap(fitness) if fitness else NOT_DESCRIBED,
        },
        "datasets": datasets,
        "metrics": metrics,
        "baselines": [r["to"] for r in relations if r["type"] == "uses-as-baseline"][:4],
        "results": results,
        "ablations": ablations,
        "limitations": {"authorStated": author_limits or [NOT_DESCRIBED], "evident": evident},
        "assumptions": assumptions,
        "computation": computation,
        "relations": relations,
        "concepts": concepts,
        "impact": pick(body_sents, ("generaliz", "transfer", "future work", "we plan"), 3, 50),
    }


def cites(refs: str, other: dict) -> bool:
    if not refs:
        return False
    low = refs.lower()
    title = other["title"].lower()
    words = [w for w in re.findall(r"[a-z0-9]+", title) if len(w) > 3]
    if words:
        bg = f"{words[0]} {words[1]}" if len(words) > 1 else words[0]
        if bg in low:
            return True
    authors = other.get("authors") or []
    year = other.get("year") or 0
    if authors and year:
        sur = SURNAME_RE.findall(authors[0])
        if sur and re.search(rf"{sur[0]}[^.]{{0,180}}\b{year}\b", refs, re.I):
            return True
    return False


def relation_type_for(citing_sents: list[str]) -> tuple[str, str]:
    joined = " ".join(citing_sents).lower()
    if any(c in joined for c in BASELINE_CUES):
        return "uses-as-baseline", "named as a comparison baseline in this paper's text"
    if any(c in joined for c in ("limitation", "drawback", "however", "fails", "cannot", "struggl")):
        return "addresses-limitation", "cited as prior work whose limitation this paper targets"
    if any(c in joined for c in ("build", "extend", "follow", "based on", "generaliz")):
        return "builds-on", "cited as prior work this paper builds on"
    return "related", "appears in this paper's reference block"


# ------------------------------------------------------------------ render


# Evidence fields are read in cards, not studied at length; a 700-character
# sentence crowds out everything else on screen.
EVIDENCE_CAP = 260


def cap(s: str, n: int = EVIDENCE_CAP) -> str:
    s = " ".join(s.split())
    if len(s) <= n:
        return s
    cut = s[:n]
    stop = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(", "))
    return (cut[:stop + 1] if stop > n * 0.5 else cut).rstrip()


def q(s: str) -> str:
    import json as _json

    return _json.dumps(s, ensure_ascii=False)


def qlist(xs) -> str:
    return "[" + ", ".join(q(x) for x in xs) + "]"


def render(rec: dict) -> str:
    a = rec["architecture"]
    e = rec["method"]["evolution"]
    b = [
        f"export const P{rec['id'][1:]}: PaperRecord = {{",
        f"  id: {q(rec['id'])},",
        f"  arxiv: {q(rec['arxiv'])},",
    ]
    if rec.get("doi"):
        b.append(f"  doi: {q(rec['doi'])},")
    b += [
        f"  title: {q(rec['title'])},",
        f"  shortTitle: {q(rec['shortTitle'])},",
        f"  year: {rec['year']},",
        f"  authors: {json.dumps(rec['authors'], ensure_ascii=False)},",
        f"  venue: {q(rec['venue'])},",
        f"  fileName: {q(rec['fileName'])},",
        f"  citations: {rec['citations']},",
        f"  pages: {rec['pages']},",
        f"  tags: {json.dumps(rec['tags'], ensure_ascii=False)},",
        f"  difficulty: {q(rec['difficulty'])},",
        f"  summary: {q(rec['summary'])},",
        f"  problem: {q(rec['problem'])},",
        f"  background: {qlist(rec['background'])},",
        f"  researchGap: {q(rec['researchGap'])},",
        f"  contribution: {qlist(rec['contribution'])},",
    ]
    # architecture block, hand-laid-out so the data model stays readable
    b.append("  architecture: {")
    b.append(f"    summary: {q(a['summary'])},")
    b.append(f"    phenotype: {q(a['phenotype'])},")
    if a["layers"]:
        b.append("    layers: [")
        for L in a["layers"]:
            units = f", units: {L['units']}" if L.get("units") else ""
            quote = f", quote: {q(L['quote'][:200])}" if L.get("quote") else ""
            b.append(f"      {{ kind: {q(L['kind'])}, label: {q(L['label'])}{units}{quote} }},")
        b.append("    ],")
    else:
        b.append("    layers: [],")
    b.append(f"    details: {qlist(a['details'])},")
    if a.get("genotypeToPhenotype"):
        b.append(f"    genotypeToPhenotype: {q(a['genotypeToPhenotype'])},")
    b.append(f"    quotes: {qlist(a['quotes'])},")
    b.append("  },")
    # evolution block
    b.append("  method: {")
    b.append("    evolution: {")
    b.append(f"      family: {q(e['family'])},")
    b.append(f"      algorithm: {q(e['algorithm'])},")
    b.append(f"      representation: {q(e['representation'])},")
    b.append(f"      encoding: {q(e['encoding'])},")
    b.append(f"      mutation: {qlist(e['mutation'])},")
    b.append(f"      crossover: {qlist(e['crossover'])},")
    b.append(f"      selection: {qlist(e['selection'])},")
    if e["changes"]:
        b.append("      changes: [")
        for c in e["changes"]:
            quote = f", quote: {q(c['quote'])}" if c.get("quote") else ""
            b.append(f"        {{ locus: {q(c['locus'])}, operator: {q(c['operator'])}{quote} }},")
        b.append("      ],")
    else:
        b.append("      changes: [],")
    if e.get("fitness"):
        b.append(f"      fitness: {q(e['fitness'])},")
    if e.get("phenotypeQuote"):
        b.append(f"      phenotypeQuote: {q(e['phenotypeQuote'])},")
    b.append("    },")
    b.append(f"    task: {q(rec['method']['task'])},")
    b.append(f"    evaluation: {q(rec['method']['evaluation'])},")
    b.append("  },")
    b += [
        f"  datasets: {qlist(rec['datasets'])},",
        f"  metrics: {qlist(rec['metrics'])},",
        f"  baselines: {qlist(rec['baselines'])},",
        f"  results: {qlist(rec['results'])},",
        f"  ablations: {qlist(rec['ablations'])},",
        "  limitations: {",
        f"    authorStated: {qlist(rec['limitations']['authorStated'])},",
        f"    evident: {qlist(rec['limitations']['evident'])},",
        "  },",
        f"  assumptions: {qlist(rec.get('assumptions') or [])},",
        f"  computation: {qlist(rec.get('computation') or [])},",
        "  relations: [",
    ]
    for r in rec["relations"]:
        note = f", note: {q(r['note'])}" if r.get("note") else ""
        b.append(f"    {{ to: {q(r['to'])}, type: {q(r['type'])}{note} }},")
    b += [
        "  ],",
        f"  concepts: {qlist(rec['concepts'])},",
    ]
    if rec.get("impact"):
        b.append(f"  impact: {qlist(rec['impact'])},")
    b.append("};")
    return "\n".join(b)


HEADER = """/* GENERATED by scripts/neuro_records.py — do not edit by hand.
 * Every field is grounded in the corpus PDF named by fileName.
 * `architecture` is required on every record: what the search produced, and
 * which loci its operators change.
 */

import type { PaperRecord } from "../types";"""


def main() -> int:
    if not (CORPUS / "manifest.json").exists():
        print("missing manifest; run the harvester first", file=sys.stderr)
        return 1
    rows = json.loads((CORPUS / "manifest.json").read_text())
    corpus_by_id = {r["id"]: r for r in rows}
    print(f"building records for {len(rows)} papers", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    built = []
    for i, row in enumerate(rows, 1):
        built.append(build_record(row, corpus_by_id))
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    batch_files: list[str] = []
    batch_ids: list[list[str]] = []
    for bi in range(0, len(built), 50):
        chunk = built[bi : bi + 50]
        name = f"batch{bi // 50 + 1:02d}.ts"
        body = [HEADER]
        for rec in chunk:
            body.append(render(rec))
            body.append("")
        (OUTDIR / name).write_text("\n".join(body))
        batch_files.append(name)
        batch_ids.append([r["id"] for r in chunk])
        print(f"wrote data/papers/{name}: {len(chunk)} records")

    names = [["P" + i[1:] for i in ids] for ids in batch_ids]
    idx = [
        HEADER,
        "",
        *[
            f'import {{ {", ".join(ns)} }} from "./{pathlib.Path(f).stem}";'
            for f, ns in zip(batch_files, names)
        ],
        "",
        "export const PAPERS: PaperRecord[] = [",
        *[f"  {n}," for ns in names for n in ns],
        "];",
        "",
        "export const PAPER_BY_ID: Record<string, PaperRecord> = Object.fromEntries(",
        "  PAPERS.map((p) => [p.id, p]),",
        ");",
        "",
        "export const PAPER_COUNT = PAPERS.length;",
        "",
    ]
    (OUTDIR / "index.ts").write_text("\n".join(idx))

    stats = {
        "records": len(built),
        "withSummary": sum(1 for r in built if r["summary"] != NOT_DESCRIBED),
        "withArchitectureDiagram": sum(1 for r in built if r["architecture"]["layers"]),
        "withLayers": sum(len(r["architecture"]["layers"]) for r in built),
        "withChanges": sum(1 for r in built if r["method"]["evolution"]["changes"]),
        "withMutation": sum(1 for r in built if r["method"]["evolution"]["mutation"]),
        "withFitness": sum(1 for r in built if r["method"]["evolution"]["fitness"]),
        "withResults": sum(1 for r in built if r["results"]),
        "withRelations": sum(1 for r in built if r["relations"]),
        "totalRelations": sum(len(r["relations"]) for r in built),
    }
    KB.emit(OUTDIR.parent, built)
    (CORPUS / "records_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())