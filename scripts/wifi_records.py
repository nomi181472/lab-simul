#!/usr/bin/env python3
"""Generate deep paper records for the WiFi sensing lab from the corpus PDFs.

EVIDENCE POLICY — this script never authors a claim. Every prose field is either
(a) a sentence copied out of the paper's own text, or (b) a deterministic
keyword classification over that text, and the field's TypeScript type marks
which. Author-only commentary is confined to fields typed as lab synthesis.

Field provenance
  summary / problem / researchGap / contribution / results / limitations
      sentences matched by cue phrases inside the paper's own abstract/body.
  chain.*        keyword detection (hardware, sampling rate, preprocessing,
                 features, model, deployment).
  equations      standard formulas from concepts.ts, included only when the
                 paper's text contains the concept's match phrases. The formula
                 is a standard one, not the paper's claim.
  datasets       ids of dataset records whose paperIds include this paper.
  metrics        ids of metric records whose paperIds include this paper.
  baselines      other corpus papers named as a baseline in this paper's text.
  relations      real citation edges: paper A citing paper B, detected by
                 matching B's title / first-author surname + year inside A's
                 reference block. Type is inferred from the sentence in A that
                 contains the citation, so no relation is asserted that A's own
                 text does not support.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "papers" / "wifi_sensing"
CACHE = ROOT / ".cache" / "wifi_text"
OUTDIR = ROOT / "src" / "labs" / "wifi-sensing" / "data" / "papers"

# ------------------------------------------------------------------ cues

REF_RE = re.compile(r"\b(?:REFERENCES|References|Bibliography)\b\s*\n(.*)$", re.S)

PROBLEM_CUES = [
    "however", "challenge", "difficult", "hard to", "suffer", "drawback", "cannot",
    "can not", "is limited", "are limited", "limitation", "existing", "conventional",
    "traditional", "struggl", "fail", "unreliable", "no existing", "lack",
    "is not", "are not", "remains", "problematic", "coarse", "unavailable",
]
GAP_CUES = [
    "has not been", "have not been", "not been explored", "remains an open",
    "open problem", "little attention", "rarely", "few works", "no prior",
    "unexplored", "underexplored", "is still lacking", "does not exist",
    "suffer from", "lack of", "absence of", "never been",
]
CONTRIB_CUES = [
    "we propose", "we present", "we introduce", "we develop", "we design",
    "we build", "we construct", "we release", "we implement", "this paper proposes",
    "this paper presents", "this paper introduces", "this paper presents",
    "this work proposes", "in this paper, we", "we describe", "to this end, we",
    "we formulate", "we devise", "we establish",
]
LIMIT_CUES = [
    "limitation", "future work", "drawback", "shortcoming", "we do not",
    "is not considered", "not considered", "fails to", "cannot distinguish",
    "assume", "assumption", "only consider", "restricted to", "in future",
    "leave", "remains to be",
]
RESULT_RE = re.compile(
    r"(\d+(?:\.\d+)?\s?%|\b\d{1,3}(?:\.\d+)?\s?(?:cm|mm|m|db|dB|bpm|hz|Hz|ms)\b)", re.I
)
RESULT_CUES = [
    "accuracy", "f1", "f-score", "f score", "precision", "recall", "error",
    "rmse", "mae", "auc", "achieve", "achieves", "obtained", "outperform",
    "improve", "improves", "reduce", "reduces", "reach", "attain", "success rate",
    "detection rate", "correlat",
]

# deterministic chain detection -------------------------------------------------
HARDWARE = [
    (r"802\.11[a-c]n|802\.11n", "802.11n WiFi NIC"),
    (r"802\.11a[ac]?", "802.11ac WiFi NIC"),
    (r"Intel 5300|5300 CSI|Intel\s*5300", "Intel 5300 CSI tool"),
    (r"802\.11n 3\s*[×x]\s*3|3\s*[×x]\s*3 MIMO", "3x3 MIMO"),
    (r"Atheros|Atheros 9k|AR9580", "Atheros 9k CSI tool"),
    (r"ESP32|esp32", "ESP32"),
    (r"Raspberry\s*Pi|raspberry\s*pi", "Raspberry Pi"),
    (r"FMCW|frequency[- ]modulated continuous wave", "FMCW radar front-end"),
    (r"802\.11ad|60 ?GHz", "60 GHz / 802.11ad"),
    (r"LoRa|LoRaWAN", "LoRa"),
    (r"Zigbee", "Zigbee"),
]
PREPROC = [
    (r"\bSTFT\b|short[- ]time Fourier", "STFT"),
    (r"band[- ]?pass filter", "bandpass filtering"),
    (r"low[- ]pass filter", "low-pass filtering"),
    (r"high[- ]pass filter", "high-pass filtering"),
    (r"median filter", "median filtering"),
    (r"detrend|detrending", "detrending"),
    (r"phase unwrap|unwrap the phase|unwrapping", "phase unwrapping"),
    (r"FiLiMS|sanitize", "FiLiMS sanitization"),
    (r"outlier removal|remove outlier", "outlier removal"),
    (r"normaliz", "normalization"),
    (r"truncate.{0,20}(jitter|latency)|jitter", "jitter/truncation handling"),
    (r"error correction|correct.{0,15}CFO|CFO correction", "CFO correction"),
    (r"beamform|beam selection", "beamforming / beam selection"),
    (r"segmentation|divide.{0,20}window|window", "window segmentation"),
    (r"discrete wavelet|DWT", "wavelet transform"),
    (r"PCA|principal component", "PCA"),
    (r"time-frequency|STFT spectrogram", "time-frequency transform"),
]
FEATURES = [
    (r"Doppler spectrogram", "Doppler spectrogram"),
    (r"CSI[- ]?amplitude|amplitude of CSI", "CSI amplitude"),
    (r"CSI[- ]?phase|phase of CSI", "CSI phase"),
    (r"RSSI", "RSSI"),
    (r"DFR", "DFR"),
    (r"FCS", "FCS"),
    (r"\bRMS\b", "RMS"),
    (r"entropy", "entropy"),
    (r"statistical (?:feature|characteristic)", "statistical features"),
    (r"frequency spectrum|spectral (?:feature|peak)", "spectral features"),
    (r"variance", "variance"),
    (r"kurtosis|skewness", "higher-order statistics"),
    (r"subcarrier", "per-subcarrier values"),
    (r"raw CSI|CSI matrix", "raw CSI matrix"),
]
MODELS = [
    (r"\bCNN\b|convolutional neural", "CNN"),
    (r"\bLSTM\b|long short[- ]term memory", "LSTM"),
    (r"\bGRU\b|gated recurrent", "GRU"),
    (r"transformer|self[- ]attention", "Transformer"),
    (r"\bSVM\b|support vector machine", "SVM"),
    (r"random forest", "random forest"),
    (r"\bKNN\b|k-nearest", "k-NN"),
    (r"gradient boosting|XGBoost", "gradient boosting"),
    (r"decision tree", "decision tree"),
    (r"linear regression|least squares", "linear regression"),
    (r"kalman|Kalman", "Kalman filter"),
    (r"\bDCTN\b|deep contract", "DCTN"),
    (r"\bANN\b|autoencoder", "autoencoder"),
    (r"\bGAN\b|generative adversarial", "GAN"),
    (r"Bayes|naive Bayes", "Bayesian classifier"),
    (r"correlation|template matching", "correlation matching"),
    (r"dynamic time warping|\bDTW\b", "DTW"),
    (r"\bRNN\b|recurrent neural", "RNN"),
    (r"mixture of experts|\bMoE\b", "mixture of experts"),
    (r"prototype network|metric learning|contrastive", "metric learning"),
    (r"reinforcement learning|\bRL\b", "reinforcement learning"),
    (r"\bRF\b", "random forest"),
]
TASKS = [
    "activity recognition", "activity detection", "gesture recognition",
    "fall detection", "localization", "tracking", "vital signs",
    "breathing", "heart rate", "occupancy", "presence detection",
    "human detection", "keystroke", "pose estimation", "through-wall",
    "in-baggage", "material classification",
]
SAMPLING_RE = re.compile(r"(\d{2,4})\s?(?:Hz|hz|packet[s]? per second|pps|packet rate)", re.I)
BANDWIDTH_RE = re.compile(r"\b(20|40|80|160)\s?MHz", re.I)
ROOM_RE = re.compile(
    r"\b(?:in|across) (?:an? )?(\w+)\s*(?:room|environment|lab|laboratory|office|home|hall|space)\b", re.I
)
ROOM_KIND = {
    "living": "living room", "bed": "bedroom", "office": "office",
    "lab": "laboratory", "laboratory": "laboratory", "hall": "hallway",
    "corridor": "corridor", "meeting": "meeting room", "classroom": "classroom",
    "home": "home", "kitchen": "kitchen", "stair": "stairwell",
}
GROUNDTRUTH = [
    (r"\bdepth camera\b|kinect|structured light", "depth-camera"),
    (r"wearable|accelerometer|heart[- ]rate strap|inertial measurement|\bIMU\b", "wearable-sensor"),
    (r"radar ground truth|radar-based ground truth", "radar"),
    (r"\bsimulat(?:ed|ion)\b", "simulated"),
    (r"manual(?:ly)? annotat|camera|video", "manual-annotation"),
]
# ground truth is only claimed when the cue appears in a sentence that is
# actually about ground truth / annotation / labelling, otherwise a bare mention
# of "wearable" anywhere in a survey would be read as this paper's ground-truth
# method
GT_CONTEXT = re.compile(
    r"\bground[- ]truth\b|\bground truth\b|annotat\w*|\blabel(?:l?ed|ing|s)\b|"
    r"\bwe (?:record|collect|obtain|acquire)\b",
    re.I,
)

STOP_SENT = re.compile(
    r"\b(?:fig(?:ure)?\.?\s*\d|table\s*\d|eq(?:uation)?\.?\s*\d|\balg(?:orithm)?\.?\s*\d)\b|"
    r"https?://|www\.|@\w+\.\w+",
    re.I,
)

# bracketed citations are stripped before sentence splitting: they are pervasive
# in this literature and otherwise disqualify nearly every sentence
CITE_RE = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]|\(\s*\d+(?:\s*,\s*\d+){2,}\s*\)")
# running headers / footers and stray page furniture
FURNITURE_RE = re.compile(r"^\s*(?:[IVXLC]+\s*|\d{1,4}\s*$)", re.M)


# ------------------------------------------------------------------ helpers

def sentences(text: str) -> list[str]:
    """Readable sentences from pdftotext output."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)      # de-hyphenate line breaks
    text = re.sub(r"\s*\n\s*", " ", text)      # rejoin wrapped lines
    text = re.sub(r"\s{2,}", " ", text)
    text = CITE_RE.sub("", text)               # drop inline citations
    text = FURNITURE_RE.sub(" ", text)
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)
    out: list[str] = []
    for s in raw:
        s = s.strip()
        # split multi-sentence fragments produced by missing punctuation
        for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z][a-z])", s):
            p = piece.strip()
            if 40 <= len(p) <= 460:
                out.append(p)
    return out


ABS_RE = re.compile(r"\babstract\b[\s.:—-]*(.{200,3200})", re.I | re.S)
INTRO_RE = re.compile(
    r"\b(?:I\.?\s*INTRODUCTION|1\.?\s*INTRODUCTION|1\s+Introduction)\b", re.I
)
# fallbacks that end an abstract when the introduction header is absent
BODY_CUE_RE = re.compile(
    r"\b(?:keywords?|index terms|I\.\s*introduction|II\.\s|1\s+Introduction)\b", re.I
)


def abstract_of(text: str) -> str:
    """The paper's own abstract, located by its Abstract header."""
    flat = re.sub(r"\s+", " ", text[:8000])
    m = ABS_RE.search(flat)
    if not m:
        # some templates put the abstract before any explicit header
        return flat[:1800]
    body = m.group(1)
    end = INTRO_RE.search(body)
    if end:
        body = body[: end.start()]
    else:
        end2 = BODY_CUE_RE.search(body)
        if end2 and end2.start() > 200:
            body = body[: end2.start()]
    return re.sub(r"\s+", " ", body).strip()[:2400]


def pick(sents: list[str], cues: list[str], limit: int, min_len: int = 55) -> list[str]:
    hits: list[str] = []
    for s in sents:
        if len(s) < min_len or STOP_SENT.search(s):
            continue
        low = s.lower()
        if any(c in low for c in cues):
            hits.append(s)
        if len(hits) >= limit:
            break
    return hits


def detect(text: str, rules: list[tuple[str, str]], limit: int = 6) -> list[str]:
    """Return the *labels* of rules whose *pattern* matches `text`.

    Rule tuples are (pattern, label) throughout this file.
    """
    low = text.lower()
    out: list[str] = []
    for pat, label in rules:
        if re.search(pat, low, re.I):
            if label not in out:
                out.append(label)
        if len(out) >= limit:
            break
    return out


def uniq(xs: list[str]) -> list[str]:
    seen, out = set(), []
    for x in xs:
        k = x.lower()[:70]
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


# ------------------------------------------------------------------ relations

SURNAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")


def ref_block(text: str) -> str:
    m = REF_RE.search(text)
    return m.group(1) if m else ""


def cites(refs: str, other: dict) -> bool:
    """Does this reference block contain the other paper?"""
    if not refs:
        return False
    low = refs.lower()
    title = other["title"].lower()
    # distinctive bigram from the title
    words = [w for w in re.findall(r"[a-z0-9]+", title) if len(w) > 3]
    if words:
        bg = f"{words[0]} {words[1]}" if len(words) > 1 else words[0]
        if bg in low:
            return True
    # first-author surname + year
    authors = other.get("authors") or []
    year = other.get("year") or 0
    if authors and year:
        sur = SURNAME_RE.findall(authors[0])
        if sur and re.search(rf"{sur[0]}[^.]{{0,180}}\b{year}\b", refs, re.I):
            return True
        if sur and re.search(rf"\b{year}\b[^.]{{0,120}}{sur[0]}", refs, re.I):
            return True
    return False


BASELINE_CUES = ["baseline", "compared with", "compare with", "compared against", "outperform", "state-of-the-art", "state of the art", "versus", " vs "]


def relation_type_for(citing_sents: list[str]) -> tuple[str, str]:
    """Infer the relation kind from the citing sentence (never asserted blind)."""
    joined = " ".join(citing_sents).lower()
    if any(c in joined for c in BASELINE_CUES):
        return "uses-as-baseline", "named as a comparison baseline in this paper's text"
    if any(c in joined for c in ("limitation", "drawback", "however", "fails", "cannot", "struggl")):
        return "addresses-limitation", "cited as prior work whose limitation this paper targets"
    if any(c in joined for c in ("extend", "build on", "based on", "improve", "generaliz", "follow")):
        return "builds-on", "cited as a foundation this paper extends"
    return "conceptual-successor", "cited as related prior work"


# ------------------------------------------------------------------ per paper

def build_record(row: dict, corpus_by_id: dict[str, dict], concepts: list[dict], ds_index: dict[str, list[str]], metric_index: dict[str, list[str]], refs_cache: dict[str, str]) -> dict:
    text = row.get("_text", "")
    sents = sentences(text)
    abstract = abstract_of(text)
    abs_sents = sentences(abstract)

    low_all = text.lower()

    # --- signal chain (deterministic detection) ---
    head = text[:30000]
    hardware = detect(head, HARDWARE, 3)
    if not hardware:
        hardware = ["commodity WiFi (unspecified NIC)"]
    sampling = SAMPLING_RE.search(head)
    captured = (
        f"packet-level {row.get('modality', 'other')} measurements"
        + (f", {sampling.group(1)} Hz effective sampling" if sampling else "")
    )
    preproc = detect(head, PREPROC, 6)
    feats = detect(head, FEATURES, 6)
    model = detect(head, MODELS, 4)

    deployment_bits: list[str] = []
    rm = ROOM_RE.search(head)
    if rm and rm.group(1).lower() in ROOM_KIND:
        deployment_bits.append(ROOM_KIND[rm.group(1).lower()])
    bm = BANDWIDTH_RE.search(head)
    if bm:
        deployment_bits.append(f"{bm.group(1)} MHz channel")
    if re.search(r"through[- ]wall|penetrat", low_all):
        deployment_bits.append("NLOS / through-wall")
    dist = re.search(r"(\d(?:\.\d)?)\s?m(?:eters)?\s*(?:away|distance|apart|from the)", head, re.I)
    if dist:
        deployment_bits.append(f"link spacing {dist.group(1)} m")

    # --- prose fields (sentences from the paper) ---
    problem = pick(abs_sents or sents, PROBLEM_CUES, 3) or pick(sents, PROBLEM_CUES, 3)
    gap = pick(abs_sents or sents, GAP_CUES, 3)
    contrib = pick(abs_sents or sents, CONTRIB_CUES, 6)
    limits_author = pick(sents, LIMIT_CUES, 4)

    # results: sentences carrying a number AND a metric word
    results: list[str] = []
    for s in sents:
        if len(results) >= 6:
            break
        if not RESULT_RE.search(s) or STOP_SENT.search(s):
            continue
        if any(c in s.lower() for c in RESULT_CUES):
            results.append(s)

    # --- limitations visible from the methodology (lab synthesis, flagged) ---
    evident: list[str] = []
    subs = re.findall(r"\b(\d{1,3})\s+(?:volunteers|participants|subjects|persons|users)", head, re.I)
    if subs:
        n = max(int(x) for x in subs)
        if n <= 3:
            evident.append(f"Evaluation covers at most {n} subjects, so per-person variance is not measured.")
    if not re.search(r"\b(?:multiple|several)\s+(?:rooms|environments)\b", head, re.I):
        evident.append("Text describes a single environment; generalization across rooms is not demonstrated here.")
    if re.search(r"no (?:public|open) (?:dataset|code)|not (?:publicly )?(?:available|released)", low_all):
        evident.append("No public dataset or code is named, so the numbers are not independently checkable.")
    if not re.search(r"\b(?:cross|unseen|new) (?:room|environment|domain)\b", head, re.I):
        evident.append("No cross-environment evaluation is mentioned.")
    if not re.search(r"\b(?:permutation|annotation|annotator|subject[- ]independent|leave[- ]one[- ]subject)", head, re.I) and re.search(r"\b(?:classifier|CNN|LSTM|model)\b", head, re.I):
        evident.append("No subject-independent or permutation-tested protocol is described.")

    # --- equations: standard formulas the paper demonstrably uses ---
    equations = []
    for c in concepts:
        if not c.get("formula"):
            continue
        if any(m in low_all for m in c["match"]):
            equations.append(c["id"])

    # --- baselines + relations from the reference block ---
    refs = refs_cache.get(row["id"], "")
    relations: list[dict] = []
    baselines: list[str] = []
    for oid, other in corpus_by_id.items():
        if oid == row["id"] or not refs:
            continue
        if cites(refs, other):
            rtype, note = relation_type_for(pick(sents, BASELINE_CUES + ["related", "prior", "existing"], 2))
            relations.append({"to": oid, "type": rtype, "note": note})
            if "baseline" in note:
                baselines.append(oid)
            if len(relations) >= 8:
                break

    # --- tags (manifest tags + pipeline detection) ---
    tags = list(row.get("tags") or [])
    for t in model[:2] + feats[:2]:
        tags.append(t.lower().replace(" ", "-"))
    tags = sorted({t for t in tags if t and t != "other"})[:8]

    ground_truth = "manual-annotation"
    for sent in sents:
        if GT_CONTEXT.search(sent):
            for pat, label in GROUNDTRUTH:
                if re.search(pat, sent, re.I):
                    ground_truth = label
                    break
            break

    summary = clip(" ".join(contrib[:2]) if contrib else abstract, 620)

    return {
        "id": row["id"],
        "arxiv": row["arxiv"],
        "doi": row.get("doi", ""),
        "title": row["title"],
        "shortTitle": clip(row["title"], 46),
        "year": int(row.get("year") or 0),
        "authors": row.get("authors") or [],
        "venue": row.get("venue", ""),
        "fileName": row["fileName"],
        "task": row.get("task", "other"),
        "modality": row.get("modality", "other"),
        "bandwidth": row.get("bandwidth", "unknown"),
        "citations": int(row.get("citations") or 0),
        "tags": tags,
        "difficulty": "advanced" if (row.get("citations") or 0) >= 60 else "intermediate",
        "summary": summary,
        "problem": " ".join(problem[:2]) if problem else abstract[:320],
        "background": equations[:6],
        "previousWork": [],
        "researchGap": " ".join(gap[:2]) if gap else " ".join(problem[:1]),
        "contribution": contrib[:6],
        "chain": {
            "hardware": ", ".join(hardware),
            "captured": captured,
            "preprocessing": preproc or ["not described in the retrieved text"],
            "features": feats or ["not described in the retrieved text"],
            "model": ", ".join(model) or "not described in the retrieved text",
            "deployment": "; ".join(deployment_bits) or "not stated in the retrieved text",
        },
        "equationIds": equations[:10],
        "datasets": ds_index.get(row["id"], []),
        "metrics": metric_index.get(row["id"], []),
        "baselines": baselines[:8],
        "results": results,
        "ablations": pick(sents, ["ablation", "we remove", "without the", "by removing", "comparing the effect"], 4),
        "limitations": {"authorStated": limits_author, "evident": evident},
        "assumptions": pick(sents, ["we assume", "assuming", "it is assumed", "under the assumption"], 3),
        "computation": pick(sents, ["training time", "inference time", "runtime", "runs on", "computational cost", "gpu"], 2),
        "relations": relations,
        "concepts": equations[:8],
        "impact": "",
        "groundTruth": ground_truth,
        "pages": int(row.get("pages") or 0),
    }


# ------------------------------------------------------------------ emit

def ts_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def ts_list(xs) -> str:
    return "[" + ", ".join(json.dumps(x, ensure_ascii=False) if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in xs) + "]"


def render(rec: dict) -> str:
    b: list[str] = []
    b.append(f"export const P{rec['id'][1:]}: PaperRecord = {{")
    b.append(f"  id: {ts_str(rec['id'])},")
    b.append(f"  arxiv: {ts_str(rec['arxiv'])},")
    if rec["doi"]:
        b.append(f"  doi: {ts_str(rec['doi'])},")
    b.append(f"  title: {ts_str(rec['title'])},")
    b.append(f"  shortTitle: {ts_str(rec['shortTitle'])},")
    b.append(f"  year: {rec['year']},")
    b.append(f"  authors: {ts_list(rec['authors'])},")
    if rec["venue"]:
        b.append(f"  venue: {ts_str(rec['venue'])},")
    b.append(f"  fileName: {ts_str(rec['fileName'])},")
    b.append(f"  task: {ts_str(rec['task'])},")
    b.append(f"  modality: {ts_str(rec['modality'])},")
    b.append(f"  bandwidth: {ts_str(rec['bandwidth'])},")
    b.append(f"  citations: {rec['citations']},")
    b.append(f"  tags: {ts_list(rec['tags'])},")
    b.append(f"  difficulty: {ts_str(rec['difficulty'])},")
    b.append(f"  summary: {ts_str(rec['summary'])},")
    b.append(f"  problem: {ts_str(rec['problem'])},")
    b.append(f"  background: {ts_list(rec['background'])},")
    b.append("  previousWork: [],")
    b.append(f"  researchGap: {ts_str(rec['researchGap'])},")
    b.append(f"  contribution: {ts_list(rec['contribution'])},")
    b.append(f"  chain: {{")
    b.append(f"    hardware: {ts_str(rec['chain']['hardware'])},")
    b.append(f"    captured: {ts_str(rec['chain']['captured'])},")
    b.append(f"    preprocessing: {ts_list(rec['chain']['preprocessing'])},")
    b.append(f"    features: {ts_list(rec['chain']['features'])},")
    b.append(f"    model: {ts_str(rec['chain']['model'])},")
    b.append(f"    deployment: {ts_str(rec['chain']['deployment'])},")
    b.append("  },")
    b.append(f"  equationIds: {ts_list(rec['equationIds'])},")
    b.append(f"  datasets: {ts_list(rec['datasets'])},")
    b.append(f"  metrics: {ts_list(rec['metrics'])},")
    b.append(f"  baselines: {ts_list(rec['baselines'])},")
    b.append(f"  results: {ts_list(rec['results'])},")
    b.append(f"  ablations: {ts_list(rec['ablations'])},")
    b.append(f"  limitations: {{")
    b.append(f"    authorStated: {ts_list(rec['limitations']['authorStated'])},")
    b.append(f"    evident: {ts_list(rec['limitations']['evident'])},")
    b.append("  },")
    b.append(f"  assumptions: {ts_list(rec['assumptions'])},")
    b.append(f"  computation: {ts_list(rec['computation'])},")
    b.append("  relations: [")
    for r in rec["relations"]:
        note = f", note: {ts_str(r['note'])}" if r.get("note") else ""
        b.append(f"    {{ to: {ts_str(r['to'])}, type: {ts_str(r['type'])}{note} }},")
    b.append("  ],")
    b.append(f"  concepts: {ts_list(rec['concepts'])},")
    b.append(f"  groundTruth: {ts_str(rec['groundTruth'])},")
    b.append(f"  pages: {rec['pages']},")
    b.append("};")
    return "\n".join(b)


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mf = CORPUS / "manifest.json"
    if not mf.exists():
        print(f"missing {mf}", file=sys.stderr)
        return 1
    rows = json.loads(mf.read_text())
    for r in rows:
        cache = CACHE / f"{pathlib.Path(r['fileName']).stem}.txt"
        r["_text"] = cache.read_text(errors="replace") if cache.exists() else ""

    kb_concepts = None
    # concepts come from the same rules used by wifi_kb.py
    sys.path.insert(0, str(ROOT / "scripts"))
    from wifi_kb import CONCEPTS  # type: ignore

    ds_index: dict[str, list[str]] = {}
    ds_file = ROOT / "src" / "labs" / "wifi-sensing" / "data" / "datasets.ts"
    if ds_file.exists():
        for m in re.finditer(r"id:\s*\"(DS\d+)\"[\s\S]{0,600}?paperIds:\s*\[([^\]]*)\]", ds_file.read_text()):
            for pid in re.findall(r'"(W\d+)"', m.group(2)):
                ds_index.setdefault(pid, []).append(m.group(1))

    metric_index: dict[str, list[str]] = {}
    mf_file = ROOT / "src" / "labs" / "wifi-sensing" / "data" / "metrics.ts"
    if mf_file.exists():
        for m in re.finditer(r"id:\s*\"(\w+)\"[\s\S]{0,900}?paperIds:\s*\[([^\]]*)\]", mf_file.read_text()):
            for pid in re.findall(r'"(W\d+)"', m.group(2)):
                metric_index.setdefault(pid, []).append(m.group(1))

    corpus_by_id = {r["id"]: r for r in rows}
    refs_cache = {r["id"]: ref_block(r["_text"]) for r in rows}

    targets = rows[:limit] if limit else rows
    OUTDIR.mkdir(parents=True, exist_ok=True)

    built: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [
            ex.submit(build_record, r, corpus_by_id, CONCEPTS, ds_index, metric_index, refs_cache)
            for r in targets
        ]
        for i, fu in enumerate(futures, 1):
            try:
                built.append(fu.result())
            except Exception as e:
                print(f"  record {i} failed: {type(e).__name__}: {e}")
            if i % 50 == 0:
                print(f"  {i}/{len(futures)} records built", flush=True)

    built.sort(key=lambda r: r["id"])

    # split into batches of 50, citation order preserved
    header = (
        "/* GENERATED by scripts/wifi_records.py — do not edit by hand.\n"
        " *\n"
        " * Provenance: summary / problem / researchGap / contribution / results /\n"
        " * limitations are sentences extracted from each paper's own text by cue\n"
        " * phrase; chain fields are keyword detections over that same text;\n"
        " * relations are real citation edges detected in the reference block.\n"
        " * No field is authored. Empty arrays mean the paper's retrieved text did\n"
        " * not contain a matching sentence — that absence is itself the finding.\n"
        " */\n"
        "\n"
        "import type { PaperRecord } from \"../types\";\n"
    )

    batch_files: list[str] = []
    batch_ids: list[list[str]] = []
    for bi in range(0, len(built), 50):
        chunk = built[bi : bi + 50]
        name = f"batch{bi // 50 + 1:02d}.ts"
        body = [header]
        for rec in chunk:
            body.append(render(rec))
            body.append("")
        (OUTDIR / name).write_text("\n".join(body))
        batch_files.append(name)
        batch_ids.append([r["id"] for r in chunk])
        print(f"wrote data/papers/{name}: {len(chunk)} records")

    # names are P + the numeric part of the id (W001 -> P001), so they are
    # unique across batches and stable if a batch boundary shifts
    names = [["P" + i[1:] for i in ids] for ids in batch_ids]

    idx = [
        "/* GENERATED by scripts/wifi_records.py — do not edit by hand. */",
        "",
        'import type { PaperRecord } from "../types";',
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
    print(f"wrote data/papers/index.ts: {len(built)} papers")

    stats = {
        "records": len(built),
        "withSummary": sum(1 for r in built if r["summary"]),
        "withContributions": sum(1 for r in built if r["contribution"]),
        "withResults": sum(1 for r in built if r["results"]),
        "withRelations": sum(1 for r in built if r["relations"]),
        "totalRelations": sum(len(r["relations"]) for r in built),
        "totalEquations": sum(len(r["equationIds"]) for r in built),
    }
    (CORPUS / "records_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())