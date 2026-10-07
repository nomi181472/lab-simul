#!/usr/bin/env python3
"""Build the Transformers / LLM / micro-LLM corpus artifacts.

Emits, from the verified manifest plus the real PDF text:

    papers/transformers/{manifest,index_summary,records_summary}.json
    src/labs/transformers/data/{manifest,concepts,metrics,datasets}.ts
    src/labs/transformers/data/papers/{index,batch01..batch06}.ts
    src/labs/transformers/sims/papers.ts

The evidence policy is the same one the neuroevolution lab uses: every
architecture layer, modification operator, objective and result carries a quote
lifted from the paper's own text, and anything not found stays an empty list or
an explicit `evident` note. Nothing is inferred.

Run:  python3 scripts/tf_build.py
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "papers" / "transformers"
TXT_CACHE = ROOT / ".cache" / "tf_text"
OUT_DATA = ROOT / "src" / "labs" / "llms" / "data"
OUT_PAPERS = OUT_DATA / "papers"
OUT_SIMS = ROOT / "src" / "labs" / "llms" / "sims"

HEAD_CHARS = 26_000

# ---------------------------------------------------------------- extraction


def pdf_text(path: pathlib.Path) -> str:
    """Extracted body text, cached per PDF.

    Raw `pdftotext` (no -layout): -layout interleaves the two-column body line by
    line and shreds every sentence this lab quotes.
    """
    if not path.exists():
        return ""
    TXT_CACHE.mkdir(parents=True, exist_ok=True)
    out = TXT_CACHE / (path.stem + ".txt")
    if out.exists() and out.stat().st_size > 200:
        return out.read_text(errors="replace")
    try:
        r = subprocess.run(
            ["pdftotext", "-q", str(path), "-"], capture_output=True, timeout=120
        )
        text = r.stdout.decode("utf-8", "replace")
    except Exception:
        text = ""
    out.write_text(text)
    return text


def pdf_pages(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    try:
        r = subprocess.run(["pdfinfo", str(path)], capture_output=True, timeout=60, text=True)
        m = re.search(r"^Pages:\s+(\d+)", r.stdout, re.M)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def sentences(text: str) -> list[str]:
    """Split into quote-able sentences, dropping broken PDF furniture."""
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    keep: list[str] = []
    for s in parts:
        s = s.strip()
        if len(s) < 60 or len(s) > 420:
            continue
        if s.count(" ") < 9:
            continue
        # references, page furniture and citation noise are not quotable evidence
        if re.search(r"\[\d+\]|\bet al\b|doi:|arxiv:|Proceedings of|Vol\.\s", s):
            continue
        keep.append(s)
    return keep


def head_text(text: str) -> str:
    return text[:HEAD_CHARS]


# ------------------------------------------------------------ vocabularies

# The structural stack. Each entry: (kind, human label, cue regexes).
# Ordered coarse -> fine so the emitted diagram reads top to bottom.
ARCH_LAYERS: list[tuple[str, str, tuple[str, ...]]] = [
    ("tokenizer", "Tokenizer", (r"tokeni[sz]er", r"\bBPE\b", r"wordpiece", r"sentencepiece")),
    ("embedding", "Embedding", (r"\bembedding layer\b", r"word embedding", r"token embedding", r"input embedding")),
    ("positional", "Positional encoding", (r"positional (?:encoding|embedding)", r"rotary (?:positional )?embedding", r"\bRoPE\b", r"relative position")),
    ("attention", "Attention", (r"self-attention", r"self attention", r"multi-head attention", r"multihead attention", r"attention (?:mechanism|module|block|layer)", r"cross-attention", r"linear attention", r"flash ?attention")),
    ("ffn", "Feed-forward / MLP", (r"feed-?forward (?:network|layer|block)", r"\bFFN\b", r"\bMLP\b", r"swi[gl]u", r"gated feed")),
    ("norm", "Normalization", (r"layer norm(?:ali[sz]ation)?", r"\bRMSNorm\b", r"normalization layer")),
    ("router", "MoE router", (r"\brouter\b", r"mixture[- ]of[- ]experts", r"\bMoE\b", r"gating network", r"load balancing")),
    ("adapter", "Adapter / low-rank", (r"\bLoRA\b", r"low-rank adapt", r"\badapter\b", r"parameter-efficient", r"prefix tuning")),
    ("cache", "KV cache", (r"kv[- ]cache", r"key[- ]value cache", r"cache mechanism")),
    ("head", "Output head", (r"output (?:projection|layer|head)", r"language modeling head", r"\bLM head\b", r"classifier head")),
]

# Modification operations. The lab's analogue of the neuroevolution operator:
# a concrete, nameable transformation applied to a model.
OPERATORS: list[tuple[str, str, tuple[str, ...]]] = [
    ("quantization", "Quantization", (r"quantiz", r"quantis", r"\d+-bit (?:weight|integer|precision)", r"integer arithmetic", r"\bint8\b", r"\bint4\b")),
    ("pruning", "Pruning", (r"\bprun", r"\bsparsif", r"structured pruning", r"unstructured sparsit")),
    ("distillation", "Distillation", (r"\bdistill", r"teacher[- ]student", r"dark knowledge")),
    ("low-rank-adaptation", "Low-rank adaptation", (r"\bLoRA\b", r"low-rank adapt")),
    ("mixed-precision", "Mixed precision", (r"mixed[- ]precision", r"\bfp16\b", r"\bbf16\b", r"half precision")),
    ("layer-reduction", "Layer reduction", (r"drop (?:a )?layer", r"layer (?:dropping|pruning|removal)", r"depth pruning", r"structural pruning", r"fewer layers")),
    ("token-reduction", "Token reduction", (r"token (?:pruning|dropping|reduction|merging)", r"merge tokens", r"token fusion")),
    ("attention-approximation", "Attention approximation", (r"linear(?:i[sz]ed)? attention", r"nystr", r"approximate attention", r"sparse attention", r"low-rank attention")),
    ("weight-sharing", "Weight sharing", (r"weight (?:tying|sharing)", r"shared (?:embedding|weights)", r"tie(?:d)? embeddings")),
    ("architecture-search", "Architecture search", (r"architecture search", r"\bNAS\b", r"evolutionary (?:search|prompt)", r"automl", r"neural architecture")),
    ("speculative-decoding", "Speculative decoding", (r"speculative decoding", r"draft model", r"speculative sampling")),
    ("caching", "Caching", (r"kv[- ]cache", r"prefix cach", r"reuse(?:d)? (?:computation|representations?)")),
    ("calibration", "Calibration", (r"calibrat", r"outlier handling", r"activation clipping", r"smoothing")),
]

# The objective a paper is optimising. Two axes matter in this literature:
# quality (accuracy/perplexity) and cost (latency/memory/energy).
OBJECTIVES: list[tuple[str, str, tuple[str, ...]]] = [
    ("accuracy", "Task accuracy", (r"accuracy", r"\bF1\b", r"exact match", r"pass@", r"BLEU", r"ROUGE", r"human evaluation", r"win rate")),
    ("perplexity", "Perplexity", (r"perplexit", r"\bPPL\b", r"bits per (?:byte|character|token)", r"\bBPC\b", r"cross[- ]entropy loss")),
    ("latency", "Latency", (r"latency", r"time per token", r"response time", r"decode speed", r"wall[- ]clock")),
    ("memory", "Memory / footprint", (r"memory footprint", r"gpu memory", r"\bKV[- ]cache size\b", r"peak memory", r"memory (?:usage|saving|reduction)")),
    ("throughput", "Throughput", (r"throughput", r"tokens per second", r"tokens/s", r"queries per second", r"batch throughput")),
    ("energy", "Energy", (r"\benergy\b", r"joules?", r"power consumption", r"carbon")),
    ("compression", "Compression ratio", (r"compression ratio", r"shrink(?:age)? factor", r"\d+\s*x\s+(?:smaller|compression|reduction)")),
    ("size", "Parameter count", (r"parameter count", r"\b\d+\.?\d*\s*[BM]\b parameters", r"model size", r"number of parameters")),
]

# Benchmarks that recur across the corpus, for the datasets view.
BENCHMARKS: list[tuple[str, str]] = [
    ("glue", "GLUE"), ("superglue", "SuperGLUE"), ("squad", "SQuAD"), ("xnli", "XNLI"),
    ("mmlu", "MMLU"), ("hellaswag", "HellaSwag"), ("arc-easy", "ARC-Easy"),
    ("arc-challenge", "ARC-Challenge"), ("winogrande", "WinoGrande"), ("piqa", "PIQA"),
    ("boolq", "BoolQ"), ("truthfulqa", "TruthfulQA"), ("gsm8k", "GSM8K"),
    ("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("wikitext", "WikiText-103"),
    ("lambada", "LAMBADA"), ("c4", "C4"), ("pile", "The Pile"), ("bookcorpus", "BookCorpus"),
    ("openwebtext", "OpenWebText"), ("anthropic-eval", "Anthropic eval"),
    ("mmlu-pro", "MMLU-Pro"), ("gpqa", "GPQA"), ("ifeval", "IFEval"),
    ("longbench", "LongBench"), ("naturalquestions", "NaturalQuestions"),
    ("hotpotqa", "HotpotQA"), ("squad2", "SQuAD 2.0"),
]

# Cost-side benchmarks.
EFFICIENCY_BENCHMARKS: list[tuple[str, str]] = [
    ("wiki2", "WikiText-2 perplexity"), ("wikitext2", "WikiText-2 perplexity"),
    ("c4-perplexity", "C4 perplexity"), ("lambada-accuracy", "LAMBADA accuracy"),
    ("wikitext103", "WikiText-103 perplexity"), ("opt-125m", "OPT-125M benchmark suite"),
    ("llama-7b", "LLaMA-7B benchmark suite"), ("mobilebert", "MobileBERT suite"),
]

CONCEPTS: list[tuple[str, str, tuple[str, ...]]] = [
    ("self-attention", "Self-Attention", (r"self-attention", r"self attention")),
    ("multi-head", "Multi-Head Attention", (r"multi-head attention", r"multihead attention", r"\bMHA\b")),
    ("positional-encoding", "Positional Encoding", (r"positional (?:encoding|embedding)", r"\bRoPE\b", r"\bALiBi\b")),
    ("encoder", "Encoder", (r"encoder[- ]only", r"bidirectional encoder", r"encoder stack")),
    ("decoder", "Decoder", (r"decoder[- ]only", r"autoregressive decoder", r"causal decoder")),
    ("encoder-decoder", "Encoder-Decoder", (r"encoder-decoder", r"encoder decoder architecture")),
    ("pretraining", "Pretraining", (r"pre-?train", r"self-supervised objective")),
    ("scaling", "Scaling", (r"scaling law", r"model scaling", r"compute-optimal")),
    ("emergence", "Emergent Abilities", (r"emergent abilit", r"emergent behaviour", r"sudden capability")),
    ("in-context-learning", "In-Context Learning", (r"in-context learning", r"in-context general")),
    ("instruction-tuning", "Instruction Tuning", (r"instruction tuning", r"instruction finetun", r"instruction-following")),
    ("rlhf", "RLHF", (r"reinforcement learning from human feedback", r"\bRLHF\b", r"human feedback")),
    ("alignment", "Alignment", (r"alignment", r"preference optimi[sz]ation", r"\bDPO\b")),
    ("prompt-tuning", "Prompt Tuning", (r"prompt tuning", r"soft prompt", r"prompt vector")),
    ("peft", "PEFT", (r"parameter-efficient", r"\bLoRA\b", r"adapter tuning")),
    ("moe", "Mixture of Experts", (r"mixture[- ]of[- ]experts", r"\bMoE\b", r"sparse expert")),
    ("long-context", "Long Context", (r"long[- ]context", r"context length", r"context window", r"extended context")),
    ("routing", "Retrieval / RAG", (r"retrieval[- ]augmented", r"\bRAG\b", r"retrieval augmented")),
    ("reasoning", "Reasoning", (r"chain[- ]of[- ]thought", r"step-by-step reasoning", r"reasoning trace")),
    ("multimodal", "Multimodal", (r"multimodal", r"vision-language", r"image-text", r"cross-modal")),
    ("quantization", "Quantization", (r"quantiz", r"quantis")),
    ("pruning", "Pruning", (r"\bprun", r"\bsparsif")),
    ("distillation", "Distillation", (r"\bdistill", r"teacher[- ]student")),
    ("kv-cache", "KV Cache", (r"kv[- ]cache", r"key[- ]value cache")),
    ("speculative-decoding", "Speculative Decoding", (r"speculative decoding", r"draft model")),
    ("edge", "Edge / On-Device", (r"on-device", r"edge device", r"mobile deployment", r"micro controller")),
    ("efficient-inference", "Efficient Inference", (r"efficient inference", r"inference efficien", r"serving efficien")),
]

METRICS: list[tuple[str, str, str]] = [
    ("perplexity", "Perplexity", "lower"), ("bits-per-byte", "Bits per Byte", "lower"),
    ("accuracy", "Accuracy", "higher"), ("f1", "F1", "higher"),
    ("exact-match", "Exact Match", "higher"), ("pass-at-1", "Pass@1", "higher"),
    ("mmlu", "MMLU", "higher"), ("human-eval", "Human Evaluation", "higher"),
    ("latency", "Latency", "lower"), ("memory", "Peak Memory", "lower"),
    ("throughput", "Tokens / Second", "higher"), ("energy", "Energy", "lower"),
    ("compression-ratio", "Compression Ratio", "higher"), ("params", "Parameters", "lower"),
]

DATASETS: list[tuple[str, str, tuple[str, ...]]] = [
    (k, label, (rf"\b{re.escape(k)}\b",))
    for k, label in BENCHMARKS + EFFICIENCY_BENCHMARKS
]

# ---------------------------------------------------------------- helpers


def find_quote(sents: list[str], patterns: tuple[str, ...], limit: int = 1) -> list[str]:
    """First `limit` sentences matching any cue, in document order."""
    out: list[str] = []
    for s in sents:
        for p in patterns:
            if re.search(p, s, re.I):
                out.append(s)
                break
        if len(out) >= limit:
            break
    return out


def ref_sentences(text: str) -> list[str]:
    """The reference block, anchored on the heading rather than assuming it is
    last: OpenAlex frequently serves a repository wrapper with extra front
    matter after the bibliography, and a bare "text[-N:]" window then returns
    nothing."""
    for pat in (
        r"^\s*(?:\d+\.?\s*)?(?:references|bibliography|reference list)\s*$",
        r"\n\s*references\s*\n",
    ):
        m = re.search(pat, text, re.I | re.M)
        if m:
            tail = text[m.end():]
            # the block is the first ~6k chars after the heading, which covers
            # a typical reference list without swallowing the appendix
            return sentences(tail[:8000])
    return []


def arxiv_id(title: str, text: str) -> str:
    m = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", text[:4000])
    return m.group(1) if m else ""


def authors_from_text(title: str, text: str) -> list[str]:
    """Fallback author list from the PDF, used only when OpenAlex has none."""
    for pat in (
        r"^\s*([A-Z][\w'’-]+(?:\s+[A-Z]\.)?\s+[A-Z][\w'’-]+(?:,?\s+[A-Z][\w'’-]+\s+[A-Z][\w'’-]+)*)\s*$",
    ):
        for line in text[:3000].splitlines():
            s = line.strip()
            if 8 < len(s) < 180 and re.match(pat, s) and "," in s:
                if not re.search(r"university|abstract|@|\bdoi\b", s, re.I):
                    return [a.strip() for a in re.split(r",\s*|\s+and\s+", s) if a.strip()][:8]
    return []


def ts_str(s: str) -> str:
    return json.dumps(s)


def ts_list(xs: list[str], indent: str = "") -> str:
    if not xs:
        return "[]"
    inner = ",\n".join(f"{indent}  {ts_str(x)}" for x in xs)
    return f"[\n{inner},\n{indent}]"


# ---------------------------------------------------------------- generation


def short_title(title: str, n: int = 52) -> str:
    """Trim on a word boundary so paper chips stay one line in the UI."""
    t = " ".join((title or "").split())
    if len(t) <= n:
        return t
    cut = t[:n].rsplit(" ", 1)[0]
    return (cut or t[:n]).rstrip(" ,;:") + "\u2026"


def build_records(rows: list[dict]) -> tuple[list[dict], dict]:
    papers: list[dict] = []
    for row in rows:
        pid = row["id"]
        path = PDF_DIR / row["fileName"]
        text = pdf_text(path)
        head = head_text(text)
        sents = sentences(head)
        body_sents = sentences(text)

        arch: list[dict] = []
        for kind, label, cues in ARCH_LAYERS:
            qs = find_quote(sents, cues, limit=1)
            if not qs:
                continue
            arch.append({"kind": kind, "label": label, "quote": qs[0]})

        changes: list[dict] = []
        for op, label, cues in OPERATORS:
            qs = find_quote(sents, cues, limit=1)
            if not qs:
                continue
            q = qs[0]
            changes.append({
                "operator": op,
                "label": label,
                "quote": q,
            })

        objectives: list[dict] = []
        for kind, label, cues in OBJECTIVES:
            qs = find_quote(sents, cues, limit=1)
            if not qs:
                continue
            objectives.append({"kind": kind, "label": label, "quote": qs[0]})

        results: list[dict] = []
        for key, label in BENCHMARKS + EFFICIENCY_BENCHMARKS:
            qs = find_quote(sents, (rf"\b{re.escape(key)}\b",), limit=1)
            if qs:
                results.append({"dataset": key, "label": label, "quote": qs[0]})

        refs = ref_sentences(text)
        arx = row.get("arxiv") or arxiv_id(row["title"], text)

        authors = row.get("authors") or authors_from_text(row["title"], text)
        pages = pdf_pages(path)

        papers.append({
            "id": pid,
            "openalex": row["openalex"],
            "arxiv": arx,
            "doi": row.get("doi", ""),
            "title": row["title"],
            "shortTitle": short_title(row["title"]),
            "year": row.get("year"),
            "venue": row.get("venue", ""),
            "authors": authors,
            "citations": row.get("citations", 0),
            "pages": pages,
            "fileName": row["fileName"],
            "pdfUrl": row.get("pdfUrl", ""),
            "relevance": "transformers",
            "abstractNote": row.get("abstract", "")[:600],
            "textChars": len(text.strip()),
            "architecture": {
                "summary": row["title"],
                "layers": arch,
            },
            "changes": changes,
            "objectives": objectives,
            "results": results,
            "referenceCount": len(refs),
        })

    stats = {
        "papers": len(papers),
        "withText": sum(1 for p in papers if p["textChars"] >= 12_000),
        "withArchitectureLayers": sum(1 for p in papers if p["architecture"]["layers"]),
        "architectureLayers": sum(len(p["architecture"]["layers"]) for p in papers),
        "withChanges": sum(1 for p in papers if p["changes"]),
        "withObjectives": sum(1 for p in papers if p["objectives"]),
        "withResults": sum(1 for p in papers if p["results"]),
        "withReferences": sum(1 for p in papers if p["referenceCount"]),
        "references": sum(p["referenceCount"] for p in papers),
        "byYear": {},
        "byOperator": {},
    }
    for p in papers:
        y = str(p.get("year") or "?")
        stats["byYear"][y] = stats["byYear"].get(y, 0) + 1
        for c in p["changes"]:
            stats["byOperator"][c["operator"]] = stats["byOperator"].get(c["operator"], 0) + 1
    return papers, stats


def emit_papers(papers: list[dict]) -> None:
    OUT_PAPERS.mkdir(parents=True, exist_ok=True)
    per = 80
    batches = [papers[i:i + per] for i in range(0, len(papers), per)]
    names: list[str] = []
    for bi, batch in enumerate(batches, 1):
        name = f"batch{bi:02d}"
        names.append(name)
        body = ",\n".join(
            "  " + json.dumps(p, indent=2).replace("\n", "\n  ") for p in batch
        )
        (OUT_PAPERS / f"{name}.ts").write_text(
            f"// GENERATED by scripts/tf_build.py -- do not edit.\n"
            f"import type {{ PaperRecord }} from \"../types\";\n\n"
            f"export const {name}: PaperRecord[] = [\n{body},\n];\n"
        )
    idx = ",\n".join(f"  {ts_str(n)}" for n in names)
    # Every batch must be imported and spread, not just the first: a hardcoded
    # `...batch01` silently truncates the corpus to 80 papers while the manifest
    # still reports 420, which is exactly the kind of drift the audit view
    # exists to catch.
    imports = "\n".join(f'import {{ {n} }} from "./{n}"' for n in names)
    spread = "\n".join(f"  ...{n}," for n in names)
    (OUT_PAPERS / "index.ts").write_text(
        f"// GENERATED by scripts/tf_build.py -- do not edit.\n"
        f"import type {{ PaperRecord }} from \"../types\"\n"
        f"{imports}\n\n"
        f"export const PAPER_BATCHES: string[] = [\n{idx},\n];\n\n"
        f"export const PAPERS: PaperRecord[] = [\n{spread}\n];\n\n"
        f"export const PAPER_COUNT = PAPERS.length;\n\n"
        f"export const PAPER_BY_ID: Record<string, PaperRecord> = Object.fromEntries(\n"
        f"  PAPERS.map((p) => [p.id, p]),\n"
        f");\n\n"
        f"export function paperById(id: string): PaperRecord | undefined {{\n"
        f"  return PAPER_BY_ID[id];\n}}\n"
    )


def emit_manifest(rows: list[dict], papers: list[dict], stats: dict) -> None:
    by_id = {p["id"]: p for p in papers}
    entries = []
    for r in rows:
        p = by_id[r["id"]]
        entries.append({
            "id": r["id"],
            "title": r["title"],
            "year": r.get("year"),
            "citations": r.get("citations", 0),
            "venue": r.get("venue", ""),
            "arxiv": p["arxiv"],
            "openalex": r["openalex"],
            "fileName": r["fileName"],
            "pages": p["pages"],
            "hasArchitecture": bool(p["architecture"]["layers"]),
            "changeCount": len(p["changes"]),
        })
    manifest = {
        "generatedBy": "scripts/tf_build.py",
        "window": "2019-01-01..2026-10-05",
        "ordering": "descending cited_by_count (OpenAlex)",
        "total": len(entries),
        "totalCitations": sum(e["citations"] for e in entries),
        "stats": stats,
        "papers": entries,
    }
    (OUT_DATA / "manifest.ts").write_text(
        f"// GENERATED by scripts/tf_build.py -- do not edit.\n"
        f"import type {{ CorpusManifest }} from \"./types\";\n\n"
        f"export const MANIFEST: CorpusManifest = {json.dumps(manifest, indent=2)};\n"
    )
    (PDF_DIR / "manifest.ts.json").unlink(missing_ok=True)


def emit_kb(papers: list[dict]) -> dict:
    def linked(pairs) -> list[dict]:
        out = []
        for key, label, cues in pairs:
            hits = []
            for p in papers:
                blob = f"{p['title']} {p.get('abstractNote','')}"
                if any(re.search(c, blob, re.I) for c in cues):
                    hits.append(p["id"])
            if hits:
                out.append({"key": key, "label": label, "papers": hits})
        return out

    concepts = linked(CONCEPTS)
    metrics = linked([(k, l, (rf"\b{re.escape(k)}\b", rf"\b{l}\b")) for k, l, d in METRICS])
    datasets = linked(DATASETS)

    for name, data, kind in (
        ("concepts", concepts, "ConceptLink"),
        ("metrics", metrics, "MetricLink"),
        ("datasets", datasets, "DatasetLink"),
    ):
        rows = ",\n".join(json.dumps(d).replace("`", "\\`") for d in data)
        (OUT_DATA / f"{name}.ts").write_text(
            f"// GENERATED by scripts/tf_build.py -- do not edit.\n"
            f"import type {{ {kind} }} from \"./types\";\n\n"
            f"export const {name.upper()}: {kind}[] = [\n{rows},\n];\n"
        )
    return {
        "concepts": len(concepts),
        "metrics": len(metrics),
        "datasets": len(datasets),
    }


def emit_sims(papers: list[dict]) -> dict:
    """Bind every paper that names a recognisable architecture family to one of
    the lab's simulators, so each simulator rests on real papers."""
    SIMS = [
        # ids must match SIMULATORS in src/labs/llms/sims/index.tsx exactly;
        # a mismatch here renders an empty "grounded in" panel with no error
        ("quantization", "Quantization", (r"quantiz", r"quantis", r"\d+-bit")),
        ("pruning", "Pruning", (r"\bprun", r"\bsparsif")),
        ("layer-reduction", "Layer reduction", (r"layer (?:dropping|pruning|removal|reduction)", r"drop (?:a )?layer", r"depth pruning", r"fewer layers", r"\bprun")),
        ("distillation", "Distillation", (r"\bdistill", r"teacher[- ]student")),
        ("peft", "PEFT", (r"\bLoRA\b", r"parameter-efficient", r"prompt tuning", r"adapter", r"prefix tuning")),
        ("moe", "Mixture of Experts", (r"mixture[- ]of[- ]experts", r"\bMoE\b", r"expert routing", r"sparse expert")),
        ("kv-cache", "KV Cache", (r"kv[- ]cache", r"key[- ]value cache")),
        ("speculative-decoding", "Speculative decoding", (r"speculative decoding", r"draft model", r"speculative sampling")),
        ("retrieval", "Retrieval", (r"retrieval[- ]augmented", r"\bRAG\b", r"retrieval augmented", r"retrieval")),
        ("reasoning", "Reasoning", (r"chain[- ]of[- ]thought", r"reasoning", r"self-consistency")),
    ]
    bindings: dict[str, list[str]] = {}
    detail: list[dict] = []
    for sim_id, label, cues in SIMS:
        hits = [
            p for p in papers
            if any(re.search(c, p["title"] + " " + p.get("abstractNote", ""), re.I) for c in cues)
        ]
        # most cited first, capped so no simulator dominates the lab
        hits.sort(key=lambda p: (-p["citations"], p["id"]))
        chosen = hits[:4]
        if not chosen:
            continue
        bindings[sim_id] = [p["id"] for p in chosen]
        detail.append({
            "id": sim_id,
            "label": label,
            "papers": [p["id"] for p in chosen],
        })
    rows = []
    for d in detail:
        rows.append(
            "  { id: %s, label: %s, papers: [%s] }"
            % (
                ts_str(d["id"]),
                ts_str(d["label"]),
                ", ".join(ts_str(x) for x in d["papers"]),
            )
        )
    OUT_SIMS.mkdir(parents=True, exist_ok=True)
    (OUT_SIMS / "papers.ts").write_text(
        f"// GENERATED by scripts/tf_build.py -- do not edit.\n"
        f"import type {{ SimulatorBinding }} from \"./types\";\n\n"
        f"export const SIMULATOR_PAPERS: SimulatorBinding[] = [\n"
        + ",\n".join(rows)
        + ",\n];\n"
    )
    return {"simulators": len(detail), "bindings": sum(len(d["papers"]) for d in detail)}


def main() -> int:
    mf = PDF_DIR / "manifest.json"
    if not mf.exists():
        print("manifest.json missing; run scripts/harvest_transformers.py first", file=sys.stderr)
        return 1
    rows = json.loads(mf.read_text())
    print(f"manifest: {len(rows)} papers")
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    papers, stats = build_records(rows)
    emit_papers(papers)
    emit_manifest(rows, papers, stats)
    kb = emit_kb(papers)
    sims = emit_sims(papers)

    stats["kb"] = kb
    stats["sims"] = sims
    (PDF_DIR / "index_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())