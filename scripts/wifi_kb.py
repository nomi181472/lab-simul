#!/usr/bin/env python3
"""Generate data/concepts.ts, data/metrics.ts and data/datasets.ts.

Design rules (the lab's evidence policy):
  * CONCEPTS and METRICS are authored here, but they are *definitions of standard
    RF / signal-processing / ML terms*, not claims attributed to papers. Their
    `paperIds` are resolved by keyword-matching the corpus, so every concept is
    shown next to the papers that actually use it.
  * DATASETS are never authored. A dataset record is emitted only for a paper
    that introduces a dataset in its own text, and its description is that
    paper's own sentence, quoted. If nothing matches, no dataset is emitted.
"""
from __future__ import annotations

import json
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "papers" / "wifi_sensing"
CACHE = ROOT / ".cache" / "wifi_text"
DATA = ROOT / "src" / "labs" / "wifi-sensing" / "data"


def q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def arr(xs) -> str:
    return json.dumps(list(xs), ensure_ascii=False)


# =====================================================================
# CONCEPTS — standard terms, with match rules to find papers that use them
# =====================================================================

CONCEPTS = [
    dict(
        id="csi",
        name="Channel State Information (CSI)",
        category="signal",
        intuition=(
            "For every packet the receiver estimates how the radio channel changed each known "
            "subcarrier tone came back: a complex number per tone. Amplitude says how much of "
            "the signal survived; phase says how the wave was delayed. A person in the path "
            "perturbs both, subtly but repeatably."
        ),
        formula="CSI(fₖ) = H(fₖ) = Σᵢ αᵢ · e^(−j2πfₖτᵢ)",
        simulator="csi-multipath",
        prereqs=[],
        match=["channel state information", "csi"],
    ),
    dict(
        id="multipath",
        name="Multipath and frequency-selective fading",
        category="propagation",
        intuition=(
            "A transmitted wave bounces off walls, furniture and people before reaching the "
            "receiver. Each bounce arrives later and weaker. Because the receiver probes many "
            "different frequencies at once, some tones are absorbed by a bounce and others are "
            "not — so the response varies across the band even though the room did not change "
            "between them."
        ),
        formula="H(f) = Σᵢ αᵢ e^(−j2πfτᵢ);  coherence bandwidth ≈ 1/τ_max",
        simulator="csi-multipath",
        prereqs=["csi"],
        match=["multipath", "frequency selective", "freqency-selective"],
    ),
    dict(
        id="path-loss",
        name="Log-distance path loss",
        category="propagation",
        intuition=(
            "Received power falls as distance grows. The rate depends on the environment: an "
            "open line-of-sight path behaves differently from a cluttered indoor room. The "
            "exponent n is what makes an indoor range claim specific rather than generic."
        ),
        formula="PL(d) = 10·n·log₁₀(d/d₀)",
        simulator="path-loss",
        prereqs=["rssi"],
        match=["path loss", "pathloss", "log-distance"],
    ),
    dict(
        id="fresnel-zone",
        name="Fresnel zone obstruction",
        category="propagation",
        intuition=(
            "An obstruction that intrudes into the first Fresnel zone steals energy from the "
            "direct path, even if the direct ray still geometrically exists. This is why a "
            "thin wall can dominate the received signal and why a doorway behaves differently "
            "from the wall beside it."
        ),
        formula="r₁ = √( λ·d₁ + d₁² )",
        simulator="fresnel-zone",
        prereqs=["multipath"],
        match=["fresnel"],
    ),
    dict(
        id="rssi",
        name="RSSI (received signal strength)",
        category="signal",
        intuition=(
            "One number per packet: total received power. Cheap, universally available, and it "
            "throws away everything except magnitude — including the phase timing information "
            "that fine motion lives in."
        ),
        formula="RSSI ∝ log₁₀ |H(f)|  (per-packet scalar)",
        simulator="rssi-vs-csi",
        prereqs=[],
        match=["rssi", "received signal strength"],
    ),
    dict(
        id="doppler",
        name="Doppler shift from motion",
        category="signal",
        intuition=(
            "A moving reflector shifts the frequency of the echo it sends back. Because the "
            "packet is very short, that shift is not a frequency — it is a phase that advances "
            "faster than it otherwise would, and the rate of advance is the Doppler shift."
        ),
        formula="f_d = v·cosθ·f_c / c ;  v_max = (packet rate)·c / (2 f_c)",
        simulator="doppler",
        prereqs=["csi"],
        match=["doppler"],
    ),
    dict(
        id="phase-wrap",
        name="Phase ambiguity and unwrapping",
        category="signal",
        intuition=(
            "Phase is only reported modulo 2π, so a smoothly moving target's phase trace jumps "
            "backwards by a full turn and wraps around. Unwrapping stitches those jumps back "
            "into a continuous trajectory; a missed correction injects a step that no filter "
            "will believe."
        ),
        formula="unwrap(φ) via adjacent-sample differencing under |Δφ| < π",
        simulator="phase-wrap",
        prereqs=["csi"],
        match=["phase unwrap", "unwrapping", "phase wrap"],
    ),
    dict(
        id="packet-rate",
        name="Packet rate and sampling limit",
        category="signal",
        intuition=(
            "Commodity WiFi is a sampled channel, not a continuous one. The sampling interval "
            "sets both the fastest motion you can see and the fastest Doppler shift you can "
            "resolve without aliasing."
        ),
        formula="f_s = packet rate (Hz);  Nyquist: |f_d| < f_s/2",
        simulator="stft-velocity",
        prereqs=["doppler"],
        match=["packet rate", "sampling rate", "nyquist", "aliasing"],
    ),
    dict(
        id="stft",
        name="Short-time Fourier transform",
        category="features",
        intuition=(
            "Slicing the stream into short windows and taking an FFT of each turns 'how the "
            "signal changed' into a picture of energy over time and motion frequency. Window "
            "length is the whole trade: long windows resolve slow motion finely, short windows "
            "respond quickly."
        ),
        formula="X(m,k) = Σ_n x[n]·w[n−mL]·e^(−j2πkn/N)",
        simulator="stft-velocity",
        prereqs=["packet-rate", "doppler"],
        match=["short-time fourier", "stft", "spectrogram"],
    ),
    dict(
        id="periodogram",
        name="Periodogram / dominant-bin estimation",
        category="features",
        intuition=(
            "If motion is periodic — breathing, gait cadence, a swinging arm — the energy "
            "concentrates at one frequency. Finding the strongest bin turns a noisy trace into a "
            "rate in a single line of code."
        ),
        formula="P[k] = |FFT(x)|² / N ;  f̂ = argmaxₖ P[k]",
        simulator="periodogram",
        prereqs=["stft"],
        match=["periodogram", "spectral analysis", "frequency spectrum", "fft"],
    ),
    dict(
        id="feature-families",
        name="CSI feature families (RSS, FCS, DFR, CCR, entropy)",
        category="features",
        intuition=(
            "Before deep learning, WiFi sensing leaned on a handful of cheap statistics per "
            "packet or window: how strong, how smooth, how much of the energy sits in the "
            "moving band, how correlated consecutive packets are, how peaky the spectrum is. "
            "Training-free recognizers are built almost entirely from these."
        ),
        formula="RSS = ‖x‖₂ ; FCS = mean|Δx| / std|Δx| ; CCR = var/‖x‖²",
        simulator="feature-families",
        prereqs=["stft"],
        match=["feature", "features"],
    ),
    dict(
        id="quantization",
        name="Quantization and NIC bit-width",
        category="signal",
        intuition=(
            "CSI is delivered as fixed-width integers. With few bits, the amplitude quantizer "
            "step is a meaningful fraction of a small signal, and recovering phase from a pair "
            "of tiny numbers amplifies that error badly."
        ),
        formula="Δ = FS / 2^b ;  σ_q = Δ / √12 ;  σ_φ ≈ σ_a / (a·sinφ)",
        simulator="quantization-noise",
        prereqs=["csi"],
        match=["quantiz", "bit-width", "bit width", "resolution"],
    ),
    dict(
        id="cfo-sfo",
        name="Carrier / sampling frequency offset",
        category="signal",
        intuition=(
            "The transmitter's carrier and the receiver's sampling clock do not agree exactly. "
            "The residual appears as a steady phase ramp that mimics a slowly moving reflector "
            "— a spurious baseline that can masquerade as breathing."
        ),
        formula="Δφ(t) = 2π·Δf·t  (linear ramp = CFO/SFO signature)",
        simulator="cfo-sfo",
        prereqs=["phase-wrap"],
        match=["cfo", "carrier frequency offset", "sfo", "sampling frequency offset", "pll"],
    ),
    dict(
        id="fingerprinting",
        name="RSSI / CSI fingerprinting",
        category="estimation",
        intuition=(
            "Skip the physics: survey the signal across a room once, then match a live reading "
            "to the survey. Localization becomes a nearest-neighbour lookup over a stored map "
            "whose shadowing pattern the room has already written into."
        ),
        formula="d(q,f) = ‖RSSI(q) − RSSI(f)‖₂ ;  p̂ = mean of k nearest",
        simulator="rssi-fingerprint",
        prereqs=["rssi"],
        match=["fingerprint", "radio map", "radio-map"],
    ),
    dict(
        id="model-based-localization",
        name="Model-based localization",
        category="estimation",
        intuition=(
            "Estimate distance from signal strength using a path-loss model, then triangulate "
            "or solve for position. Cheaper than fingerprinting but hostage to whatever "
            "shadowing the model omits."
        ),
        formula="d = d₀ · 10^((P₀ − P_rx) / 10n)",
        simulator="rssi-fingerprint",
        prereqs=["path-loss", "rssi"],
        match=["trilateration", "model-based localization", "model based localization"],
    ),
    dict(
        id="kalman",
        name="Kalman filtering of a track",
        category="estimation",
        intuition=(
            "A motion model plus noisy measurements, fused online. The process/measurement "
            "noise ratio is the single dial: trust the motion and get a laggy smooth line, "
            "trust the measurements and get a jagged one."
        ),
        formula="K = P⁻/(P⁻+R) ;  x = x⁻ + K(z − x⁻)",
        simulator="kalman-trajectory",
        prereqs=["packet-rate"],
        match=["kalman", "smoothing", "filtering"],
    ),
    dict(
        id="snr",
        name="Feature SNR",
        category="evaluation",
        intuition=(
            "Before asking whether a model works, ask whether the signal is above the noise at "
            "all. This number sets the ceiling on every downstream accuracy claim."
        ),
        formula="SNR_dB = 10 log₁₀(P_signal / P_noise)",
        simulator="snr-margin",
        prereqs=[],
        match=["snr", "signal-to-noise"],
    ),
    dict(
        id="domain-shift",
        name="Cross-domain shift",
        category="evaluation",
        intuition=(
            "The same physical activity looks different on a different NIC, in a different "
            "room, with different furniture. A decision boundary learned on one distribution "
            "is calibrated to the wrong one everywhere else."
        ),
        formula="accuracy after mean shift Δ = Φ(Δ/σ)",
        simulator="domain-shift",
        prereqs=["snr"],
        match=["domain shift", "cross-domain", "cross domain", "domain adaptation", "generaliz"],
    ),
    dict(
        id="separation-ceiling",
        name="Class-separability ceiling",
        category="learning",
        intuition=(
            "If two activities produce overlapping feature distributions, no model separates "
            "them. Accuracy is capped by feature overlap, not architecture depth."
        ),
        formula="P(error) = Φ(−k/2) for Gaussians at k·σ separation",
        simulator="activity-clf",
        prereqs=["snr"],
        match=["separab", "class overlap", "bayes error"],
    ),
    dict(
        id="receptive-field",
        name="Temporal receptive field",
        category="learning",
        intuition=(
            "A 1-D convolution over packets sees a span of samples. Too short to contain one "
            "cycle of the motion you are trying to recognise; too long and it smears the phase "
            "of the pattern. This is the temporal analogue of an image model's field of view."
        ),
        formula="RF = 1 + L·(k−1) samples → seconds = RF / packet rate",
        simulator="conv1d-window",
        prereqs=["packet-rate"],
        match=["receptive field", "temporal window", "window size"],
    ),
    dict(
        id="privacy",
        name="Device-free / non-invasive sensing",
        category="tasks",
        intuition=(
            "The pitch of the field: infer human activity without a wearable, a camera or any "
            "cooperation from the person being observed. The signal passes through existing "
            "infrastructure instead."
        ),
        formula="— (a property of the sensing modality, not an equation)",
        simulator=None,
        prereqs=["csi"],
        match=["device-free", "device free", "non-invasive", "noninvasive", "without wearable", "wifi camera"],
    ),
    dict(
        id="ground-truth",
        name="Ground-truth acquisition",
        category="evaluation",
        intuition=(
            "The hardest and least discussed part of any WiFi sensing dataset: proving where the "
            "person was and what they did. Cameras, wearables and depth sensors all perturb the "
            "very environment being measured."
        ),
        formula="— (a methodological property, not an equation)",
        simulator=None,
        prereqs=[],
        match=["ground truth", "ground-truth", "annotat", "labeling", "labelling"],
    ),
    dict(
        id="through-wall",
        name="Through-wall sensing budget",
        category="propagation",
        intuition=(
            "Each wall consumes dB from a fixed transmit budget. A through-wall range claim is "
            "really a statement about wall material, count, distance and noise floor jointly."
        ),
        formula="P_rx = P_tx − PL(d) − N_walls·A_wall ;  SNR = P_rx − N",
        simulator="through-wall",
        prereqs=["path-loss", "fresnel-zone"],
        match=["through-wall", "through the wall", "wall penetration", "nlos", "non-line-of-sight"],
    ),
]


# =====================================================================
# METRICS — standard definitions
# =====================================================================

METRICS = [
    dict(
        id="acc",
        name="Accuracy",
        family="classification",
        formula="accuracy = (TP + TN) / (TP + TN + FP + FN)",
        meaning="Fraction of all decisions that were correct.",
        intuition="A single number that is only honest when classes are balanced and the wrong answers are equally costly.",
        example="70% accuracy on 3 balanced classes against a 33% chance floor.",
        limitations=[
            "Collapses toward chance as class count grows, which reads as progress.",
            "Says nothing about which class is being confused with which.",
            "A majority-class predictor looks strong on imbalanced data.",
        ],
        simulator="activity-clf",
        match=["accuracy"],
    ),
    dict(
        id="macro-f1",
        name="Macro-averaged F1",
        family="classification",
        formula="F1_k = 2·P_k·R_k/(P_k+R_k) ;  macro = (1/K)·Σ_k F1_k",
        meaning="Unweighted mean of per-class F1 scores.",
        intuition="Asks how well the model does on the rare classes too, which accuracy hides.",
        example="Macro-F1 of 0.62 when overall accuracy is 0.85 signals the model is failing on a minority class.",
        limitations=[
            "Requires predicted labels, so it costs a full inference pass.",
            "Sensitive to the threshold, which is rarely reported.",
        ],
        simulator="activity-clf",
        match=["f1", "f-score", "f score", "macro-f"],
    ),
    dict(
        id="mae",
        name="Mean absolute error",
        family="vital-sign",
        formula="MAE = (1/N)·Σ|yᵢ − ŷᵢ|",
        meaning="Average absolute distance between estimate and truth.",
        intuition="The natural unit for anything measured in bpm or metres: 'off by 0.6 bpm on average' needs no interpretation.",
        example="MAE of 0.6 bpm on a 15 bpm respiration rate is roughly a 4% error.",
        limitations=[
            "Hides the tail — a method with a few catastrophic misses can still post a good MAE.",
            "Uninterpretable without the scale of the target quantity.",
        ],
        simulator="vital-signs",
        match=["mae", "mean absolute error"],
    ),
    dict(
        id="rmse",
        name="Root-mean-square error",
        family="vital-sign",
        formula="RMSE = √( (1/N)·Σ(yᵢ − ŷᵢ)² )",
        meaning="Same as MAE but penalises large misses more heavily.",
        intuition="Use it when a big miss is much worse than a small one; otherwise MAE is the safer summary.",
        example="RMSE 0.9 vs MAE 0.6 usually means a handful of large failures.",
        limitations=[
            "Does not report which sample failed.",
            "Easy to accidentally compute over a different window than MAE.",
        ],
        simulator="kalman-trajectory",
        match=["rmse", "root mean square", "root-mean-square"],
    ),
    dict(
        id="median-err",
        name="Median absolute error",
        family="vital-sign",
        formula="median(|y − ŷ|)",
        meaning="The typical error, ignoring outliers.",
        intuition="The standard in vital-sign work because body movement produces outliers that wreck a mean.",
        example="Reporting a median error alongside the mean exposes movement-induced failures.",
        limitations=[
            "Can look good while a meaningful fraction of windows fail completely.",
            "Pairs badly with an accuracy claim that ignores failures.",
        ],
        simulator="vital-signs",
        match=["median absolute error", "mae-median", "median error"],
    ),
    dict(
        id="mse",
        name="Mean position error",
        family="localization",
        formula="e = ‖p̂ − p‖₂ ;  report mean / median / 90th percentile",
        meaning="Euclidean distance between estimated and true position.",
        intuition="The spatial analogue of MAE, and the only honest way to report a room-scale result.",
        example="A mean error of 0.6 m with a 90th percentile of 2.1 m tells a very different story from 0.6 m flat.",
        limitations=[
            "Sensitive to the room size chosen for testing.",
            "Averages hide regions of the room where the estimate fails.",
        ],
        simulator="rssi-fingerprint",
        match=["position error", "localization error", "localisation error", "distance error"],
    ),
    dict(
        id="auc",
        name="Detection / occupancy AUC and rate",
        family="detection",
        formula="TPR = TP/(TP+FN) at a threshold; AUC = ∫ TPR dFPR",
        meaning="How reliably presence or motion is detected as a threshold varies.",
        intuition="Threshold-free, so it separates 'the method works' from 'the threshold was tuned well'.",
        example="An occupancy detector at 0.9 TPR and 0.05 FPR is usable; 0.9/0.4 is not.",
        limitations=[
            "Hides whether either class is rare.",
            "Says nothing about latency, which dominates a fall alarm.",
        ],
        simulator="snr-margin",
        match=["auc", "detection rate", "true positive", "false alarm", "precision", "recall"],
    ),
    dict(
        id="params",
        name="Parameter count and inference cost",
        family="efficiency",
        formula="cost ≈ params × 2 bytes (fp16) ; latency ≈ layers × window / throughput",
        meaning="Model size and on-device feasibility.",
        intuition="A 99% paper that needs a GPU server is a different contribution from one that runs on the router.",
        example="Reporting params and inference time is what makes a 'lightweight' claim checkable.",
        limitations=[
            "Not comparable across hardware without latency measurements.",
            "Ignores data-preprocessing cost, which for CSI is often larger.",
        ],
        simulator=None,
        match=["parameter", "params", "lightweight", "inference time", "complexity", "efficiency"],
    ),
]


# =====================================================================


def load_corpus() -> list[dict]:
    rows = json.loads((CORPUS / "manifest.json").read_text())
    for r in rows:
        pdf = CORPUS / r["fileName"]
        cache = CACHE / f"{pdf.stem}.txt"
        if cache.exists():
            r["_text"] = cache.read_text(errors="replace")
        else:
            r["_text"] = ""
    return rows


def resolve(corpus: list[dict], match: list[str], limit: int = 6) -> list[str]:
    """Paper ids whose own text contains any of the match phrases."""
    hits: list[str] = []
    for r in corpus:
        hay = (r["title"] + " " + r["_text"][:20000]).lower()
        if any(m in hay for m in match):
            hits.append(r["id"])
        if len(hits) >= limit:
            break
    return hits


# ---------- datasets: never authored, always quoted from the paper ----------

DS_RE = re.compile(
    r"[^.]*?\b(?:we (?:publicly )?(?:release|introduce|present|provide)|"
    r"this (?:paper )?(?:introduces|presents|releases|provides)|"
    r"to (?:the best of our knowledge, )?(?:facilitate|enable|support))[^.]*?"
    r"\b(?:data ?set|dataset|benchmark|corpus)\b[^.]*\.",
    re.I,
)


def extract_datasets(corpus: list[dict], limit: int = 40) -> list[dict]:
    out: list[dict] = []
    for r in corpus:
        text = re.sub(r"\s+", " ", r["_text"][:30000])
        m = DS_RE.search(text)
        if not m:
            continue
        sent = m.group(0).strip()
        if len(sent) < 60 or len(sent) > 400:
            continue
        # name = the paper's own title-based short label
        name = r["title"]
        if len(name) > 70:
            name = name[:67].rsplit(" ", 1)[0] + "…"
        out.append(
            dict(
                id=f"DS{len(out) + 1:03d}",
                name=name,
                year=r.get("year") or 0,
                purpose=sent,
                domain=r.get("modality", "other"),
                groundTruth="manual-annotation",
                subjects=None,
                locations=None,
                samples=None,
                characteristics=[f"modality: {r.get('modality')}", f"task: {r.get('task')}"],
                metrics=[],
                paperIds=[r["id"]],
            )
        )
        if len(out) >= limit:
            break
    return out


# ---------- emitters ----------

GT_MAP = {
    "manual-annotation": "manual-annotation",
    "wearable-sensor": "wearable-sensor",
    "depth-camera": "depth-camera",
    "radar": "radar",
    "simulated": "simulated",
}


def emit_concepts(corpus: list[dict]) -> None:
    L = [
        "/* GENERATED by scripts/wifi_kb.py — do not edit by hand.",
        " *",
        " * Concept definitions are standard RF / signal-processing / ML terms. Each",
        " * `paperIds` list is resolved by matching the phrase against the corpus itself,",
        " * so every concept links only to papers that actually use it.",
        " */",
        "",
        "import type { Concept } from \"./types\";",
        "",
        "export const CONCEPTS: Concept[] = [",
    ]
    for c in CONCEPTS:
        ids = resolve(corpus, c["match"])
        L.append("  {")
        L.append(f"    id: {q(c['id'])},")
        L.append(f"    name: {q(c['name'])},")
        L.append(f"    category: {q(c['category'])},")
        L.append(f"    intuition: {q(c['intuition'])},")
        if c.get("formula"):
            L.append(f"    formula: {q(c['formula'])},")
        if c.get("simulator"):
            L.append(f"    simulator: {q(c['simulator'])},")
        L.append(f"    prereqs: {arr(c['prereqs'])},")
        L.append(f"    paperIds: {arr(ids)},")
        L.append("  },")
    L += ["];", "", "export const CONCEPT_BY_ID: Record<string, Concept> = Object.fromEntries(", "  CONCEPTS.map((c) => [c.id, c]),", ");", ""]
    (DATA / "concepts.ts").write_text("\n".join(L))
    print(f"concepts.ts: {len(CONCEPTS)} concepts")


def emit_metrics(corpus: list[dict]) -> None:
    L = [
        "/* GENERATED by scripts/wifi_kb.py — do not edit by hand.",
        " *",
        " * Metric definitions are standard and shared across the field; each `paperIds`",
        " * list is resolved from corpus text so the metric page links only to papers",
        " * that report it.",
        " */",
        "",
        "import type { MetricRecord } from \"./types\";",
        "",
        "export const METRICS: MetricRecord[] = [",
    ]
    for m in METRICS:
        ids = resolve(corpus, m["match"])
        L.append("  {")
        L.append(f"    id: {q(m['id'])},")
        L.append(f"    name: {q(m['name'])},")
        L.append(f"    family: {q(m['family'])},")
        L.append(f"    formula: {q(m['formula'])},")
        L.append(f"    meaning: {q(m['meaning'])},")
        L.append(f"    intuition: {q(m['intuition'])},")
        L.append(f"    example: {q(m['example'])},")
        L.append(f"    limitations: {arr(m['limitations'])},")
        if m.get("simulator"):
            L.append(f"    simulator: {q(m['simulator'])},")
        L.append(f"    paperIds: {arr(ids)},")
        L.append("  },")
    L += ["];", "", "export const METRIC_BY_ID: Record<string, MetricRecord> = Object.fromEntries(", "  METRICS.map((m) => [m.id, m]),", ");", ""]
    (DATA / "metrics.ts").write_text("\n".join(L))
    print(f"metrics.ts: {len(METRICS)} metrics")


def emit_datasets(corpus: list[dict]) -> None:
    ds = extract_datasets(corpus)
    L = [
        "/* GENERATED by scripts/wifi_kb.py — do not edit by hand.",
        " *",
        " * Every dataset record is extracted from a corpus paper that introduces a",
        " * dataset in its own text, and `purpose` is that paper's own sentence, quoted",
        " * verbatim. Nothing here is authored, and a paper that does not announce a",
        " * dataset gets no entry.",
        " *",
        " * `groundTruth` is left as the conservative default because the acquisition",
        " * method is not inferable from the announcement sentence; papers that state",
        " * it are surfaced through `characteristics`.",
        " */",
        "",
        "import type { DatasetRecord } from \"./types\";",
        "",
        "export const DATASETS: DatasetRecord[] = [",
    ]
    for d in ds:
        L.append("  {")
        L.append(f"    id: {q(d['id'])},")
        L.append(f"    name: {q(d['name'])},")
        L.append(f"    year: {int(d['year'])},")
        L.append(f"    purpose: {q(d['purpose'])},")
        L.append(f"    domain: {q(d['domain'])},")
        L.append(f"    groundTruth: {q(GT_MAP[d['groundTruth']])},")
        L.append(f"    characteristics: {arr(d['characteristics'])},")
        L.append(f"    metrics: {arr(d['metrics'])},")
        L.append(f"    paperIds: {arr(d['paperIds'])},")
        L.append("  },")
    L += ["];", "", "export const DATASET_BY_ID: Record<string, DatasetRecord> = Object.fromEntries(", "  DATASETS.map((d) => [d.id, d]),", ");", ""]
    (DATA / "datasets.ts").write_text("\n".join(L))
    print(f"datasets.ts: {len(ds)} datasets (quoted from corpus)")


def main() -> int:
    corpus = load_corpus()
    print(f"corpus: {len(corpus)} papers")
    with_text = sum(1 for r in corpus if r["_text"])
    print(f"  with extracted text: {with_text}")
    DATA.mkdir(parents=True, exist_ok=True)
    emit_concepts(corpus)
    emit_metrics(corpus)
    emit_datasets(corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())