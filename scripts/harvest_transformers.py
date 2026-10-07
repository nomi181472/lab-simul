#!/usr/bin/env python3
"""Harvest the Transformers / LLM / micro-LLM corpus via OpenAlex.

Same pipeline as scripts/harvest_neuro_openalex.py, with the domain policy
swapped for the arc this lab actually covers:

    transformer architecture -> large language model -> micro / edge LLM

Scope. Papers whose subject is the transformer itself or a descendant of it:
attention mechanisms, pretraining and scaling, prompting and adaptation, the
efficient-LLM literature (quantisation, pruning, distillation, sparsity, mixture
of experts, efficient attention), LLM systems and serving, evaluation, and the
small / on-device / micro-LLM line. Application papers are only admitted when the
architecture or the model is the contribution.

Date window is 2017-01-01 through 2026-10-05. It originally started at 2019,
which excluded the papers the whole arc is built on -- Vaswani et al.,
GPT-1/2, BERT -- leaving the lab unable to show where transformer models came
from. The floor is now 2017 so the foundations are in the corpus alongside
their descendants.

Ordering is by descending real citation count, which is what makes the manifest
read as "what the field actually leaned on".

Why OpenAlex rather than the arXiv API: `export.arxiv.org/api/query` returns a
hard 429 in this environment, while OpenAlex is reachable, carries
`cited_by_count` directly (no fuzzy Crossref title matching needed), exposes
`abstract_inverted_index` so the gate can run before spending a download, and
resolves open-access PDFs through `locations[].pdf_url`.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import harvest_neuroevolution as H

ROOT = H.ROOT
OUT = ROOT / "papers" / "transformers"
# Never deleted: a rejected PDF is kept so a later policy change can re-admit
# it without re-downloading. `.cache` is gitignored.
REJECT = ROOT / ".cache" / "transformers-rejected"
TARGET = 420

# --- the window the user asked for
YEAR_MIN = 2017
YEAR_MAX = 2026
FROM_DATE = "2017-01-01"
TO_DATE = "2026-10-05"

OA = "https://api.openalex.org/works"
MAILTO = H.MAILTO
UA = H.UA

# This corpus is mostly short conference papers, so the neuroevolution
# MIN_TEXT (25k, chosen to survive long two-column journal papers) would reject
# perfectly good 5-page papers. 12k clears repository wrapper pages while keeping
# short workshop papers.
MIN_TEXT = 12_000
# Distinct signal terms required in the body when the title has none. Calibrated
# against the corpus: papers whose titles genuinely are about the architecture
# ("Attention Is All You Need", "An Image is Worth 16x16 Words: Transformers")
# carry a title signal and are exempt, while incidental single-term mentions are
# not enough to make a paper about transformers.
BODY_SIGNAL_MIN = 3

# ---------------------------------------------------------------- vocabulary

# The mechanism the field is built on.
SIGNAL_TERMS: tuple[str, ...] = (
    "transformer", "self-attention", "self attention", "attention mechanism",
    # bare "attention" is the mechanism word itself; without it the efficient-
    # attention line (FlashAttention and friends) is invisible, because those
    # abstracts say "attention algorithm" and never "attention mechanism".
    "attention", "memory-efficient", "io-aware", "hbm",
    "multi-head attention", "multihead attention", "positional encoding",
    "recurrent neural network", "encoder-decoder", "sequence-to-sequence",
    "context window", "tokenizer", "tokenization", "wordpiece", "sentencepiece",
    "language model", "large language model", "llm", "llms", "gpt", "bert",
    "llama", "mistral", "gemma", "t5", "vit", "vision transformer",
    "pretraining", "pre-training", "pretrained language model",
    # Model names whose family name is glued to a qualifier. `\bbert\b` cannot
    # match "BioBERT" (no boundary between "o" and "B"), so the biomedical,
    # legal, financial and multilingual BERT variants were invisible to the
    # gate. Listed explicitly rather than with a `\w*bert` wildcard: a wildcard
    # would match "Herbert" and "Egbert" in bibliographies and reference lists,
    # which pdftotext includes.
    "biobert", "clinicalbert", "financialbert", "scientificbert",
    "pubmedbert", "psychbert", "genericbert", "lilbert", "medbert", "roberta",
    "deberta", "bigbird", "hubert", "electra", "layoutlm", "videobert",
    "vilbert", "lxmert", "chatgpt", "chat-bots", "gpt-4", "gpt-3", "gpt-2",
    "instruction tuning", "prompt tuning", "in-context learning",
    "few-shot learning", "chain-of-thought", "chain of thought",
    "retrieval-augmented", "retrieval augmented", "knowledge distillation",
    "model compression", "quantization", "quantisation", "quantization-aware",
    "network pruning", "weight pruning", "sparsity", "mixture of experts",
    "flash attention", "linear attention", "efficient attention",
    "small language model", "edge llm", "on-device inference", "tinybert",
    "distilbert", "albert", "emergent abilit", "scaling law", "scaling laws",
    "rlhf", "human feedback", "foundation model", "multimodal large language",
    "speculative decoding", "kv cache", "token budget",
    "parameter-efficient", "lora", "adapter tuning", "instruction-following",
    "natural language processing", "text generation", "code generation",
    # LLM systems / training infrastructure, which the arc terminates in
    "large-scale deep learning", "trillion-parameter", "model parallelism",
    "distributed training", "mixed precision", "training efficiency",
    "inference optimization", "efficient inference", "serving throughput",
    "parallelism strategy",
)

# What the model is asked to do. Keeps "transformer" from admitting power-grid
# transformer studies, which use the word in a completely different sense.
TASK_TERMS: tuple[str, ...] = (
    "language", "text", "translation", "summariz", "summaris", "question answering",
    "generation", "generative", "dialogue", "conversation", "chatbot", "reasoning",
    "code", "programming", "classification", "retrieval", "understanding",
    "benchmark", "evaluation", "inference", "serving", "deployment", "fine-tun",
    "finetun", "alignment", "instruction", "prompt", "token", "sequence", "nlp",
    "semantic", "lexical", "syntax", "sentiment", "summarisation", "caption",
    "embedding", "pretraining", "knowledge", "reason", "math", "medical",
    "vision", "image", "speech", "multimodal", "agent", "tool use",
)

# Application areas where the transformer is incidental. Checked against the
# title only, so a paper whose title is about the architecture is not discarded
# for also touching one of these.
OFF_TERMS: tuple[str, ...] = (
    "power system", "power grid", "power distribution", "power flow", "volt/var",
    "wind turbine", "electricity", "electromagnetic", "antenna array", "volt-ampere",
    "radiology", "radiograph", "pathology", "histopatholog", "computed tomography",
    "eeg", "ecg", "electrocardiogram", "protein folding", "drug discovery",
    "retrosynthesis", "molecular dynamics", "gene expression", "single-cell",
    "protein", "molecular", "drug discovery", "drug design", "peptide",
    "molecular docking", "drug docking",
    "materials discovery", "traffic flow", "vehicle trajectory", "traffic congestion",
    "wireless sensor network", "recommender system", "stock market", "financial trading",
    "supply chain", "radio signal", "signal classification", "modulation",
    "speech enhancement", "speaker verification", "semantic segmentation",
    # Physics, mathematics, philosophy and social-science papers that cite the
    # transformer literature rather than being about it. "HISTORIAE, History of
    # Socio-Cultural Transformation as Linguistic Theory" (17.3k citations) has
    # eight distinct body signal terms, so no body threshold can catch it: only
    # a title veto will.
    "photonics", "topological", "quantum", "gravitational", "black hole",
    "neutron star", "stellar", "astronom", "cosmolog", "dark matter",
    "molecular", "protein", "crystal", "semiconductor", "catalys",
    "gravitational-wave", "ligo", "virgo", "ligo-virgo", "kagra", "pulsar",
    "epidemiolog", "morbidity", "mortality", "prevalence", "clinical trial",
    "cochrane", "guideline", "meta-analysis", "autism", "cancer",
    "hallmarks of", "aging", "immun", "microbiome", "gut inflam",
    "socio-cultural", "sociological", "discernment", "consciousness",
    "linguistic theory", "philosoph", "theology", "jurisprudence",
    # Legal *domain* papers are out of scope, but so is vetoing the term outright:
    # LegalBench measures reasoning in LLMs and LEGAL-BERT is a model paper, so
    # the veto is confined to titles that are plainly about law as a field.
    "jurisprudence", "case law", "legal doctrine", "statutory interpretation",
)

# Language-model signals: strong enough in the title ALONE to prove the paper is
# in scope, and strong enough to rescue it from OFF_TERMS.
LM_TITLE: tuple[str, ...] = (
    "language model", "llm", "llms", "gpt", "bert", "llama", "mistral", "gemma",
    "t5", "prompt", "token", "foundation model", "nlp", "instruction",
    "dialogue", "question answering", "in-context", "chain-of-thought",
)

# Architecture signals: prove scope, but do NOT rescue from OFF_TERMS. Bare
# "transformer" is the ambiguous word this whole OFF list exists for -- power-grid
# and RF papers say "a Transformer-based framework" in their titles -- so it must
# not be able to talk its way past its own off-topic list.
ARCH_TITLE: tuple[str, ...] = (
    "transformer", "attention", "vision transformer", "vit",
)

QUERIES: list[str] = [
    # the architecture and its attention variants
    "transformer attention mechanism",
    "efficient attention transformer long sequence",
    "linear attention transformer complexity",
    "flash attention",
    "positional encoding transformer",
    "sparse attention transformer",
    "vision transformer",
    "state space model sequence",
    # large language models: pretraining and scaling
    "large language model pretraining",
    "scaling laws language model",
    "emergent abilities large language model",
    "in-context learning language model",
    "instruction tuning language model",
    "chain of thought reasoning language model",
    "reinforcement learning human feedback language model",
    "foundation model",
    "multimodal large language model",
    "retrieval augmented generation",
    # adaptation
    "parameter efficient fine tuning large language model",
    "prompt tuning",
    "LoRA low rank adaptation",
    "few shot learning language model",
    # the efficient-LLM line
    "quantization large language model",
    "pruning large language model compression",
    "knowledge distillation language model",
    "mixture of experts language model",
    "model compression transformer inference",
    "sparse attention inference acceleration",
    "KV cache attention inference",
    "speculative decoding",
    # micro / small / edge LLMs -- the end of the arc
    "small language model edge",
    "on-device large language model",
    "micro language model",
    "edge intelligence language model",
    "model compression edge deployment transformer",
    "distilbert",
    "tiny language model",
    # systems and evaluation
    "LLM serving system throughput",
    "LLM inference memory efficient",
    "language model benchmark evaluation",
    "language model reasoning benchmark",
    "code generation language model",
]
QUERIES = list(dict.fromkeys(QUERIES))

# A second pass. The first pool ran out at a ~50% verification rate, which lands
# just under a 420 target, so these widen the tail rather than re-asking the
# same questions.
EXTRA_QUERIES: list[str] = [
    "efficient transformer memory long document",
    "perceiver io-efficient attention",
    "xformers memory efficient attention",
    "longformer sliding window attention",
    "bigbird sparse attention",
    "reformer reversible layers",
    "memory efficient transformer",
    "learned sparse attention",
    "alibi relative position bias",
    "rotary position embedding",
    "grouped query attention",
    "multi query attention",
    "sliding window attention",
    "context compression long context",
    "infini-attention compressive memory",
    "kv cache compression eviction",
    "h2o heavy hitter oracle",
    "scissorhands kv cache",
    "gptq post training quantization",
    "awq activation aware quantization",
    "smoothquant migration",
    "zero quantization",
    "bitsandbytes 8-bit optimizers",
    "spqr quantized attention",
    "atom mixed precision",
    "llama.cpp quantization",
    "gptq vs awq quantization accuracy",
    "weight-only quantization llm",
    "one-bit extreme compression",
    "product quantization embeddings",
    "deepseek mixture of experts",
    "switch transformer expert routing",
    "gshard mixture of experts",
    "expert choice routing",
    "upcycling sparse llm",
    "distilling step-by-step",
    "gpt-3 few shot learning",
    "bloom open multilingual",
    "opt open pretraining",
    "llama open efficient instruction",
    "alpaca instruction following",
    "vicuna chat assistant",
    "openchat open models",
    "tinychat compact chat models",
    "phi small language models",
    "orca progressive learning",
    "self-instruct aligning language models",
    "lIMA less is more alignment",
    "dpo direct preference optimization",
    "constitutional ai harmlessness",
    "WizardLM Evol-Instruct",
    "longchat long context chat",
    "chatglm efficient inference",
    "mobile llm on device",
    "llama.cpp edge deployment",
    "edge inference transformer accelerator",
    "tinyml edge bert",
    "micro controller transformer",
    "energy efficient llm",
    "carbon efficient inference",
    "lora qa lora multi adapter",
    "prefix tuning prompt tuning",
    "adapter parameter efficient transfer",
    "sparsity large language model training",
    "2:4 semi structured sparsity",
    "wanda pruning llm",
    "slice pruning llm",
    "shortgpt layer pruning",
    "llm compression survey",
    "efficient inference survey llm",
    "memory efficient inference llm",
]
EXTRA_QUERIES = list(dict.fromkeys(EXTRA_QUERIES))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _lit(terms: tuple[str, ...]) -> re.Pattern[str]:
    """Regex over the exact vocabulary, so gate() and relevance() agree.

    Word boundaries are mandatory, not cosmetic. Several entries are short and
    alphanumeric -- `vit`, `lora`, `t5`, `gpt`, `moe` -- and without \\b they
    match inside ordinary words: `vit` in "acti**vit**y" and "**vit**al",
    `lora` in "exp**lora**tion". That is how a bare-substring version of this
    policy admitted *Array programming with NumPy* (24k citations) and a
    socio-cultural history paper into a transformer corpus.

    The trailing `(?:s|es)?` keeps plurals matching, because a hard boundary
    after the term stops "language model" from matching "large language models",
    which would have thrown out the most on-topic papers in the corpus.
    """
    return re.compile(
        r"\b(?:"
        + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
        + r")(?:s|es)?\b",
        re.I,
    )


SIGNAL = _lit(SIGNAL_TERMS)
TASK = _lit(TASK_TERMS)
OFF_RE = _lit(OFF_TERMS)
LM_TITLE_RE = _lit(LM_TITLE)
ARCH_TITLE_RE = _lit(ARCH_TITLE)


def abstract_of(w: dict) -> str:
    """Rebuild the abstract OpenAlex stores as a positional inverted index."""
    inv = w.get("abstract_inverted_index")
    if not inv:
        return ""
    n = max((max(v) for v in inv.values() if v), default=-1) + 1
    buf = [""] * n
    for word, poss in inv.items():
        for p in poss:
            if 0 <= p < n:
                buf[p] = word
    return " ".join(buf)


def gate(text: str) -> tuple[bool, str]:
    """Admission is SIGNAL-only, deliberately.

    An earlier revision also required a TASK word, which silently threw away
    exactly the papers that define the arc -- "Attention Is All You Need" and
    "DistilBERT, a distilled version of BERT" contain no task noun at all.

    Off-topic filtering is not repeated here. OFF_TERMS is matched against the
    *title* alone in LLM_RELEVANCE; running it over title+abstract rejected good
    papers for mentioning "recommender system" or "medical imaging" once in
    passing, which is exactly the over-rejection the vocabulary was meant to
    avoid.
    """
    if not SIGNAL.search(text):
        return False, "no-signal-term"
    return True, "ok"


def LLM_RELEVANCE(title: str, hay: str) -> str:
    """transformers | adjacent | off-topic, from the paper's own text.

    Every test here goes through a word-bounded regex. Plain substring tests
    over this vocabulary are actively wrong: see `_lit`.
    """
    t = (title or "").lower()
    lm = bool(LM_TITLE_RE.search(t))
    arch = bool(ARCH_TITLE_RE.search(t))
    # OFF_TERMS is an unconditional veto, including for papers whose title
    # also contains an LM signal. Rescuing on `lm` let
    # "Recommender Systems in the Era of Large Language Models" and
    # "Translating radiology reports ... using ChatGPT and GPT-4" back in:
    # the lab is about architecture and efficiency, and a domain application
    # paper contributes no architecture or change evidence no matter how many
    # LLM words are in its title.
    if OFF_RE.search(t):
        return "off-topic"
    # A title that is plainly about the architecture or a language model is in
    # scope on its own; otherwise the signal has to appear in the body too.
    if lm or arch:
        return "transformers"
    if SIGNAL.search(hay):
        return "transformers" if TASK.search(hay) else "adjacent"
    return "adjacent"


def pdf_url_of(w: dict) -> str:
    """Prefer an arXiv copy, then the best OA location, then any other OA copy."""
    cands: list[str] = []
    best = w.get("best_oa_location") or {}
    if best.get("pdf_url"):
        cands.append(best["pdf_url"])
    for loc in w.get("locations") or []:
        if loc.get("pdf_url"):
            cands.append(loc["pdf_url"])
    if best.get("landing_page_url") and "arxiv.org" in best["landing_page_url"]:
        aid = best["landing_page_url"].rstrip("/").split("/abs/")[-1]
        if aid:
            cands.append(f"https://arxiv.org/pdf/{aid}")
    arxiv_first = [c for c in cands if "arxiv.org" in c]
    return (arxiv_first + cands)[0] if (arxiv_first or cands) else ""


class HarvestBlocked(RuntimeError):
    """OpenAlex stayed unreachable.

    Raised instead of returning an empty page, because an empty page is
    indistinguishable from a query that legitimately matched nothing -- and
    that ambiguity is what produced an empty corpus with a success message.
    """


def fetch_page(params: dict) -> dict:
    """One OpenAlex page, retrying patiently.

    OpenAlex throttles hard from a single host: the polite pool still returns
    429 for many minutes at a time. Giving up after four quick tries meant every
    query returned an empty page and the whole harvest silently completed with
    an empty pool -- 14 queries, 0 papers, and no error to notice. So retry long
    and hard (about 13 minutes of worst-case backoff per query), and treat an
    exhausted retry budget as a hard failure that stops the run rather than
    quietly returning {}.
    """
    u = OA + "?" + urllib.parse.urlencode(params)
    attempts = 7
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < attempts - 1:
                d = min(10.0 * (2**attempt), 150.0)
                # Jitter so concurrent-ish retries don't resonate.
                d *= 0.75 + 0.5 * ((attempt * 37) % 10) / 10.0
                log(f"    ~ HTTP {e.code}, backing off {d:.0f}s")
                time.sleep(d)
                continue
            raise HarvestBlocked(
                f"OpenAlex HTTP {e.code} on {u[:110]} after {attempts} attempts"
            )
        except Exception as e:
            if attempt < attempts - 1:
                d = min(5.0 * (2**attempt), 60.0)
                log(f"    ~ {type(e).__name__}, retrying in {d:.0f}s")
                time.sleep(d)
                continue
            raise HarvestBlocked(f"{type(e).__name__}: {str(e)[:80]}")
    raise HarvestBlocked("retry budget exhausted")


SELECT = (
    "id,doi,title,display_name,publication_year,cited_by_count,"
    "abstract_inverted_index,locations,best_oa_location,authorships,primary_location"
)


def discover(per_query: int, queries: list[str]) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    # Persisted after every query: this run spends minutes in OpenAlex 429
    # backoff, and losing the pool to a crash would cost all of it.
    cache = ROOT / ".cache" / "transformers_pool.json"
    if cache.exists():
        try:
            seen = json.loads(cache.read_text())
            log(f"resumed pool from cache: {len(seen)}")
        except Exception:
            seen = {}
    for q in queries:
        params = {
            "search": q,
            "filter": f"from_publication_date:{FROM_DATE},to_publication_date:{TO_DATE}",
            "sort": "cited_by_count:desc",
            "per-page": min(per_query, 200),
            "mailto": MAILTO,
            "select": SELECT,
        }
        d = fetch_page(params)
        rows = d.get("results") or []
        added = 0
        for w in rows:
            wid = w.get("id")
            if not wid or wid in seen:
                continue
            year = w.get("publication_year")
            if not year or not (YEAR_MIN <= year <= YEAR_MAX):
                continue
            title = w.get("title") or w.get("display_name") or ""
            if not title:
                continue
            abstract = abstract_of(w)
            ok, _reason = gate(f"{title} {abstract}")
            if not ok:
                continue
            # a thin or topic-less abstract must not let a paper through on its
            # title alone; the full text is checked again at download time
            if LLM_RELEVANCE(title, f"{title} {abstract}".lower()) == "off-topic":
                continue
            pdf = pdf_url_of(w)
            if not pdf:
                continue
            authors = [
                (a.get("author") or {}).get("display_name", "")
                for a in (w.get("authorships") or [])[:12]
            ]
            venue = ""
            pl = w.get("primary_location") or {}
            if pl.get("source"):
                venue = (pl["source"] or {}).get("display_name") or ""
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            seen[wid] = {
                "openalex": wid.rsplit("/", 1)[-1],
                "doi": doi,
                "title": title,
                "year": year,
                "authors": authors,
                "venue": venue,
                "citations": w.get("cited_by_count") or 0,
                "pdfUrl": pdf,
                "abstract": abstract[:4000],
            }
            added += 1
        cache.write_text(json.dumps(seen))
        log(f"  {added:>4}  {q}")
        time.sleep(1.5)
    # OpenAlex routinely returns the same work under several IDs (a preprint, a
    # proceedings version, a postprint). Keyed by OpenAlex ID they survive as
    # separate candidates, so collapse them by normalized title and keep the
    # most-cited record. Without this the lab shows the same paper twice.
    by_title: dict[str, dict] = {}
    for cand in seen.values():
        key = norm_title(cand["title"])
        cur = by_title.get(key)
        if cur is None or cand["citations"] > cur["citations"]:
            by_title[key] = cand
    if len(by_title) != len(seen):
        log(
            f"collapsed {len(seen) - len(by_title)} duplicate-title records "
            f"from distinct OpenAlex IDs"
        )
    log(f"OpenAlex gated+OA pool (sorted by citations): {len(by_title)}")
    return by_title


def arxiv_year(arxiv_id: str) -> int | None:
    """Year encoded in an arXiv id, e.g. 1706.03762 -> 2017.

    OpenAlex's `publication_year` is taken from whichever record it indexed
    last, so a reposted or re-derived version reports the wrong year: "Attention
    Is All You Need" (arXiv 1706.03762, the Vaswani et al. original) came back as
    2025. The arXiv id is authoritative about when the paper was first posted,
    so prefer it whenever the two disagree.
    """
    m = re.match(r"(\d{2})(\d{2})", (arxiv_id or "").strip())
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 91 else 2000 + yy


def norm_title(t: str) -> str:
    """Collapse title variants for duplicate detection: case, punctuation, spacing."""
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def _pad(real: str) -> str:
    """Grow a one-line sample to MIN_TEXT so gate tests exercise the real path."""
    return (real + "\n") * 400


def distinct_signals(text: str) -> set[str]:
    return {
        t for t in SIGNAL_TERMS if re.search(r"\b" + re.escape(t) + r"(?:s|es)?\b", text, re.I)
    }


def relevance_of(row: dict, text: str) -> str:
    """Second gate on the real PDF text, so a thin abstract cannot carry a paper."""
    title = row["title"]
    hay = (title + "\n" + text[:22000]).lower()
    ok, _ = gate(f"{title} {row['abstract']}")
    if not ok:
        return "off-topic"
    # OpenAlex often points at a repository wrapper (deposit notice, cover page,
    # scanned front matter): a title and a few thousand chars of boilerplate.
    if len(text.strip()) < MIN_TEXT:
        return "too-thin"
    rel = LLM_RELEVANCE(title, hay)
    if rel != "transformers":
        return rel
    # A paper with no architecture vocabulary in its *title* is only in scope if
    # its body carries several distinct signals. One incidental hit is not
    # enough: "Topological photonics" and "Billion-Scale Similarity Search with
    # GPUs" each contain exactly one body signal term and were being admitted on
    # that alone. Titles still pass on a single signal -- that is what lets
    # "Attention Is All You Need" in.
    if not distinct_signals(title) and len(distinct_signals(hay)) < BODY_SIGNAL_MIN:
        return "adjacent"
    return rel


def seed_from_manifest() -> list[dict]:
    """Rows already verified by an earlier run, in the internal record shape.

    Re-downloading and re-verifying them would cost another full pass over
    several hundred PDFs for no new information, so they are carried forward
    as-is and only their position in the citation ordering is recomputed.
    """
    mf = OUT / "manifest.json"
    if not mf.exists():
        return []
    try:
        rows = json.loads(mf.read_text())
    except Exception:
        return []
    log(f"seeded {len(rows)} already-verified papers from {mf}")
    return [
        {
            "openalex": r["openalex"],
            "doi": r.get("doi", ""),
            "title": r["title"],
            "year": r.get("year"),
            "authors": r.get("authors", []),
            "venue": r.get("venue", ""),
            "citations": r.get("citations", 0),
            "pdfUrl": r.get("pdfUrl", ""),
            "abstract": "",
            "fileName": r["fileName"],
            "arxiv": r.get("arxiv", ""),
            "relevance": "transformers",
        }
        for r in rows
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--per-query", type=int, default=200)
    ap.add_argument("--workers", type=int, default=H.DOWNLOAD_WORKERS)
    ap.add_argument(
        "--extra",
        action="store_true",
        help="run the supplementary query set as well as the base one",
    )
    ap.add_argument(
        "--discover-only",
        action="store_true",
        help="populate the pool cache and stop, without downloading",
    )
    ap.add_argument(
        "--cache-only",
        action="store_true",
        help=(
            "download and verify from the cached pool without querying OpenAlex. "
            "Discovery is the only rate-limited phase, and a 429 part-way through it "
            "otherwise throws away the whole run: the pool from the last complete "
            "query stays in the cache but nothing gets downloaded before the abort. "
            "4,656 candidates with 743 above 400 citations are already banked, which "
            "is far more than the corpus needs."
        ),
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    manifest = seed_from_manifest()
    have = {r["openalex"] for r in manifest}

    queries = list(QUERIES)
    if args.extra:
        queries += [q for q in EXTRA_QUERIES if q not in queries]
    if args.cache_only:
        cache = ROOT / ".cache" / "transformers_pool.json"
        if not cache.exists():
            log("cache-only requested but no pool cache exists")
            return 1
        pool = json.loads(cache.read_text())
        log(f"cache-only: using {len(pool)} cached candidates, no OpenAlex queries")
    else:
        pool = discover(args.per_query, queries)
    if args.discover_only:
        log("discover-only: stopping before download")
        return 0
    if not pool:
        log("empty pool; nothing to do")
        return 1

    ranked = sorted(
        pool.values(), key=lambda c: (-c["citations"], -(c.get("year") or 0), c["title"])
    )
    log(f"top 10 by citation:")
    for c in ranked[:10]:
        log(f"   {c['citations']:>6}  {c.get('year')}  {c['title'][:62]}")

    failed: list[list[str]] = []

    def job(c: dict) -> dict | None:
        if c["openalex"] in have:
            return None
        arxiv = ""
        m = re.search(r"arxiv\.org/pdf/([^?#]+)", c["pdfUrl"])
        if m:
            arxiv = m.group(1)
        name = f"{c['openalex']}.pdf"
        dest = OUT / name
        if not H.valid_pdf(dest):
            # Retry: the first pass lost 396 of 769 candidates to a single
            # transient fetch, which is most of the gap between 339 verified
            # papers and the 400 the corpus is supposed to hold. A second and
            # third attempt recover most of those, so the cost of a retry is far
            # below the cost of re-discovering the candidate later.
            why = "unknown"
            for attempt in range(3):
                time.sleep((H.ARXIV_PAUSE / args.workers) * (attempt + 1))
                ok, why = H.fetch(c["pdfUrl"], dest)
                if ok and H.valid_pdf(dest):
                    break
            if not H.valid_pdf(dest):
                dest.unlink(missing_ok=True)
                failed.append([c["openalex"], why])
                return None
        text = H.pdftotext(dest)
        rel = relevance_of(c, text)
        if rel == "too-thin":
            # Retry once: pdftotext returns almost nothing when the machine is
            # already saturated, which is indistinguishable from a genuinely
            # thin paper. Without the retry a paper can be lost outright on a
            # transient condition; without the reject afterwards the PDF stays
            # on disk forever, unmanifested, and re-fails on every later run.
            log(f"    ~ too-thin, refetching ({len(text)} ch) {c['title'][:48]}")
            for attempt in range(2):
                time.sleep((H.ARXIV_PAUSE / args.workers) * (attempt + 1))
                dest.unlink(missing_ok=True)
                ok, _ = H.fetch(c["pdfUrl"], dest)
                if not ok or not H.valid_pdf(dest):
                    break
                text = H.pdftotext(dest)
                rel = relevance_of(c, text)
                if rel != "too-thin":
                    break
            if rel == "too-thin":
                log(f"    x too-thin ({len(text)} ch) {c['title'][:56]}")
                rej = REJECT / name
                REJECT.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    shutil.move(str(dest), str(rej))
                failed.append([c["openalex"], "too-thin"])
                return None
        if rel != "transformers":
            dest.unlink(missing_ok=True)
            return None
        out = dict(c)
        out["fileName"] = name
        out["arxiv"] = arxiv
        out["relevance"] = rel
        return out

    log(f"downloading + verifying until {args.target} (seeded {len(manifest)})")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, r in enumerate(ex.map(job, ranked), 1):
            if r:
                manifest.append(r)
                have.add(r["openalex"])
            if n % 25 == 0:
                log(f"  {len(manifest)} verified / {n} attempted")
            if len(manifest) >= args.target:
                break

    manifest.sort(key=lambda c: (-c["citations"], -(c.get("year") or 0), c["title"]))
    rows = []
    for i, c in enumerate(manifest, 1):
        rows.append(
            {
                "id": f"T{i:03d}",
                "openalex": c["openalex"],
                "arxiv": c.get("arxiv", ""),
                "doi": c.get("doi", ""),
                "title": c["title"],
                "year": arxiv_year(c.get("arxiv", "")) or c.get("year"),
                "venue": c.get("venue", ""),
                "authors": c.get("authors", []),
                "citations": c["citations"],
                "fileName": c["fileName"],
                "pdfUrl": c["pdfUrl"],
                "relevance": "transformers",
            }
        )
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "download_summary.json").write_text(
        json.dumps(
            {
                "source": "openalex",
                "window": f"{FROM_DATE}..{TO_DATE}",
                "success": [r["openalex"] for r in rows],
                "failed": sorted(failed),
            },
            indent=2,
        )
        + "\n"
    )
    log(
        f"DONE manifest={len(rows)} failed={len(failed)} "
        f"pdfs={len(list(OUT.glob('*.pdf')))} "
        f"citations={sum(r['citations'] for r in rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())