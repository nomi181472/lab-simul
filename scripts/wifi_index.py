#!/usr/bin/env python3
"""Index the WiFi sensing corpus: enrich manifest.json and emit data/manifest.ts.

For every harvested PDF this script
  1. extracts text with pdftotext (cached under .cache/wifi_text/),
  2. records pdfBytes and the real page count from pdfinfo,
  3. classifies task / modality / bandwidth / tags **deterministically** from the
     paper's own title, abstract and opening text,
  4. writes the enriched fields back into manifest.json, and
  5. emits the typed TS projection the lab imports.

Nothing here is authored prose: classification is keyword-driven over text the
paper itself contains, so every label in the manifest is traceable to the PDF.
Deep per-paper records (data/papers/) are a separate, deliberately manual step.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from html import unescape
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "papers" / "wifi_sensing"
CACHE = ROOT / ".cache" / "wifi_text"
MANIFEST = CORPUS / "manifest.json"
TS_OUT = ROOT / "src" / "labs" / "wifi-sensing" / "data" / "manifest.ts"

# ---------------------------------------------------------------- taxonomy

TASK_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("vital-signs", ("heart rate", "heartbeat", "cardiac", "ecg", "pulse rate", "vital sign")),
    ("breathing", ("breathing", "respiration", "respiratory", "chest displacement", "inhale")),
    ("fall-detection", ("fall detection", "fall detect", "falling person", "fall recognit", "fall alarm")),
    ("gesture-recognition", ("gesture recognition", "gesture", "hand sign", "sign language", "air writing")),
    ("keystroke-identification", ("keystroke", "keystroke inference", "typing", "keyboard input")),
    ("occupancy-presence", ("occupancy", "presence detection", "human detection", "person detection", "presence estimation")),
    ("localization", ("localization", "localisation", "position estimation", "positioning", "indoor localization", "locating")),
    ("tracking", ("tracking", "trajectory", "motion tracking", "target tracking", "human tracking")),
    ("posture-pose", ("pose estimation", "posture recognition", "body pose", "skeleton", "joint angle")),
    ("in-baggage", ("baggage", "in-baggage", "luggage", "through the bag", "inside bag")),
    ("material-signature", ("material classification", "material signature", "material type", "liquids classification")),
    ("activity-recognition", ("activity recognition", "activity detection", "human activity", "gesture and activity", "gait recognition")),
    ("through-wall", ("through-wall", "through the wall", "wall penetration", "wall-penetrating")),
    ("benchmark", ("benchmark", "dataset for", "we release", "data set of")),
]

MODALITY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("wifi-radar", ("fmcw", "frequency-modulated continuous wave", "wifi radar", "chirp", "radar signal", "doppler radar")),
    ("wifi-sonar", ("sonar", "acoustic echolocation", "ultrasound")),
    ("csi-toolkit", ("csi tool", "toolkit for csi", "csi toolbox", "linux csi", "openwifi", "csikit", "intel 5300", "atheros 9k", "aixcl", "wifi-csi")),
    ("csi-80211n", ("channel state information", "csi", "mimo", "subcarrier", "802.11n", "802.11ac", "802.11ax")),
    ("rssi", ("rssi", "received signal strength")),
]

# How a paper obtains the measurement it reasons about, when it is stated.
CAPTURE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("complex per-subcarrier CSI", ("channel state information", "csi matrix", "per-subcarrier")),
    ("RSSI only", ("rssi", "received signal strength")),
]

BANDWIDTH_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("160 MHz", ("160 mhz", "160mhz")),
    ("80 MHz", ("80 mhz", "80mhz")),
    ("40 MHz", ("40 mhz", "40mhz")),
    ("20 MHz", ("20 mhz", "20mhz", "802.11n", "802.11a", "802.11g")),
]

TAG_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("deep-learning", ("deep learning", "cnn", "lstm", "transformer", "neural network", "gan", "autoencoder", "diffusion model")),
    ("transfer-learning", ("transfer learning", "domain adaptation", "few-shot", "meta-learning", "zero-shot")),
    ("signal-processing", ("doppler", "spectrogram", "stft", "fourier", "filter bank", "phase unwrap", "sampling")),
    ("fingerprinting", ("fingerprint", "radio map", "radio-map", "fingerprint database")),
    ("model-based", ("ray tracing", "ray-tracing", "monte carlo", "propagation model", "stochastic channel model")),
    ("privacy", ("privacy", "non-invasive", "without wearable", "device-free", "camera-free")),
    ("edge-deployment", ("embedded", "edge", "raspberry", "on-device", "fpga", "low-power")),
    ("multi-modal", ("multi-modal", "multimodal", "fusion of", "combine csi and", "vision-assisted")),
    ("benchmark-tooling", ("toolkit", "framework", "platform", "open source", "publicly available at")),
    ("vital-signs", ("breath", "heart rate", "vital")),
    ("localization", ("localiz", "localis", "position")),
    ("through-wall", ("through-wall", "wall penetration", "nlos")),
    ("human-detection", ("human detection", "person detection", "occupancy", "presence")),
]

# ---------------------------------------------------------------- extraction


def pdf_text(pdf: pathlib.Path) -> str:
    """Full text via pdftotext, cached as <arxiv>.txt.

    Deliberately NOT `-layout`: for the two-column IEEE/ACM papers that dominate
    this corpus, -layout emits both columns side by side and splits sentences
    mid-clause. Raw mode follows reading order, which is what sentence-level
    extraction needs.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{pdf.stem}.txt"
    if cache.exists() and cache.stat().st_size > 200:
        return cache.read_text(errors="replace")
    try:
        r = subprocess.run(
            ["pdftotext", str(pdf), "-"],
            capture_output=True,
            timeout=180,
        )
        text = r.stdout.decode("utf-8", "replace")
    except Exception:
        text = ""
    cache.write_text(text)
    return text


def pdf_pages(pdf: pathlib.Path) -> int:
    try:
        r = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, timeout=60)
        m = re.search(r"^Pages:\s+(\d+)", r.stdout.decode("utf-8", "replace"), re.M)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def head_text(text: str, chars: int = 22000) -> str:
    """Title + abstract + opening, which is where a paper states its own task."""
    return text[:chars].lower()


# ---------------------------------------------------------------- relevance

HUMAN_CUES = (
    "human", "person", "people", "pedestrian", "occupant", "occupancy", "activity",
    "gesture", "breathing", "heartbeat", "heart rate", "fall", "posture", "gait",
    "vital sign", "crowd", "patient", "elderly", "infant", "sleep", "keystroke",
    "presence detection", "human detection", "person detection", "user activity",
    "living", "walking", "standing", "sitting", "lying", "motion detection",
)
SENSING_CUES = (
    "wifi", "wi-fi", "wlan", "csi", "rssi", "fingerprint", "sensing", "radio signal",
    "through-wall", "through the wall", "wireless signal", "802.11", "beam", "channel state",
)
OFF_TOPIC_CUES = (
    "uav", "drone", "vehicle", "vehicular", "covert channel", "coexistence",
    "spectrum sensing", "throughput", "mac protocol", "handover", "handoff",
    "cryptograph", "key generation", "phishing", "malware", "worm attack",
    "agricultur", "smart grid", "routing protocol", "blockchain",
    # IT/network security that uses the radio or the network as an attack
    # surface rather than to sense a person. Bare "attack" is deliberately
    # absent: adversarial-attack robustness is legitimate sensing research.
    "intrusion detection", "intrusion-detection", "rogue access point",
    "rogue ap ", "netflow", "network traffic", "network security",
    "network monitoring", "traffic classification", "packet capture",
    "iot forensics", "forensic", "firewall", "penetration test",
    "denial of service", "denial-of-service", "botnet", "smart contract",
    "malicious", "tor network", "anomaly detection in network",
    # system/security studies and radio hardware papers: real WiFi work, but
    # the contribution is not a sensing result about people. "countermeasure"
    # alone is too broad -- "a countermeasure against adversarial wireless
    # sensing" is legitimate sensing research, so only the security-survey
    # phrasing is treated as off-topic.
    "attacks and countermeasures", "app-based attack", "app based attack",
    "detection of apps", "sensor network node",
)
HUMAN_PRIMARY_OFFTOPIC = ("uav", "drone", "vehicle", "agricultur", "blockchain")


def classify_relevance(title: str, hay: str) -> str:
    """human-sensing | adjacent | off-topic, from the paper's own text.

    Kept separate from `task` on purpose: a paper can be a WiFi paper and still
    not be about sensing people.
    """
    t = title.lower()
    human = any(c in hay for c in HUMAN_CUES) or any(c in t for c in HUMAN_CUES)
    sensing = any(c in hay for c in SENSING_CUES) or any(c in t for c in SENSING_CUES)
    off = [c for c in OFF_TOPIC_CUES if c in t]

    if not sensing:
        return "off-topic"
    if human and not off:
        return "human-sensing"
    if human and off:
        # title-level off-topic cue: on-topic only if the title also names people
        return "adjacent" if any(c in t for c in ("human", "person", "occupant", "activity")) else "off-topic"
    return "adjacent"


def first_match(hay: str, rules: list[tuple[str, tuple[str, ...]]]) -> tuple[str, list[str]]:
    """First rule that fires wins; also return every rule that fired (for tags)."""
    fired = [name for name, keys in rules if any(k in hay for k in keys)]
    return (fired[0] if fired else "other", fired)


# ---------------------------------------------------------------- enrich


def enrich_row(row: dict) -> dict:
    pdf = CORPUS / row["fileName"]
    text = pdf_text(pdf) if pdf.exists() else ""
    # title always counts, abstract/intro carries most of the signal
    hay = (row["title"] + "\n" + head_text(text)).lower()

    task, _ = first_match(hay, TASK_RULES)
    modality, _ = first_match(hay, MODALITY_RULES)
    bandwidth, _ = first_match(hay, BANDWIDTH_RULES)
    _, tags = first_match(hay, TAG_RULES)

    row["pdfBytes"] = pdf.stat().st_size if pdf.exists() else 0
    row["pages"] = pdf_pages(pdf) if pdf.exists() else 0
    row["task"] = task
    row["modality"] = modality
    # bandwidth is frequently simply unstated in WiFi sensing papers; that is a
    # real fact about the paper, not a classification failure.
    row["bandwidth"] = "unknown" if bandwidth == "other" else bandwidth
    row["tags"] = sorted(set(tags))[:6]
    row["textChars"] = len(text)
    row["relevance"] = classify_relevance(row["title"], hay)
    return row


# ---------------------------------------------------------------- emit ts


def q(s: str) -> str:
    # Crossref/arXiv metadata carries HTML entities, sometimes double-escaped
    # ("&amp;amp;"); unescape until stable so no entity survives into the data
    s = str(s)
    for _ in range(3):
        t = unescape(s)
        if t == s:
            break
        s = t
    return json.dumps(s, ensure_ascii=False)


def emit_ts(rows: list[dict]) -> None:
    L = [
        "/* GENERATED by scripts/wifi_index.py — do not edit by hand.",
        " * Source: papers/wifi_sensing/manifest.json, enriched from the PDFs themselves.",
        " *",
        " * task / modality / bandwidth / tags are keyword classifications over each",
        " * paper's own title + abstract + opening text (see scripts/wifi_index.py), not",
        " * editorial judgements.",
        " */",
        "",
        "export type ManifestEntry = {",
        "  id: string;",
        "  arxiv: string;",
        "  doi?: string;",
        "  title: string;",
        "  year: number;",
        "  authors: string[];",
        "  venue?: string;",
        "  task: string;",
        "  modality: string;",
        "  bandwidth: string;",
        "  citations: number;",
        "  tags: string[];",
        "  fileName: string;",
        "  pdfBytes: number;",
        "  pages: number;",
        "};",
        "",
        "/** Citation-ordered corpus. Index 0 is the most-cited paper. */",
        "export const MANIFEST: ManifestEntry[] = [",
    ]
    for p in rows:
        bits = [f"id: {q(p['id'])}", f"arxiv: {q(p['arxiv'])}"]
        if p.get("doi"):
            bits.append(f"doi: {q(p['doi'])}")
        bits += [
            f"title: {q(p['title'])}",
            f"year: {int(p.get('year') or 0)}",
            f"authors: {json.dumps(p.get('authors') or [], ensure_ascii=False)}",
        ]
        if p.get("venue"):
            bits.append(f"venue: {q(p['venue'])}")
        bits += [
            f"task: {q(p.get('task', 'other'))}",
            f"modality: {q(p.get('modality', 'other'))}",
            f"bandwidth: {q(p.get('bandwidth', 'unknown'))}",
            f"citations: {int(p.get('citations') or 0)}",
            f"tags: {json.dumps(p.get('tags') or [], ensure_ascii=False)}",
            f"fileName: {q(p['fileName'])}",
            f"pdfBytes: {int(p.get('pdfBytes') or 0)}",
            f"pages: {int(p.get('pages') or 0)}",
        ]
        L.append("  { " + ", ".join(bits) + " },")

    cites = [int(p.get("citations") or 0) for p in rows]
    yrs = [int(p.get("year") or 0) for p in rows if p.get("year")]
    L += [
        "];",
        "",
        "export const MANIFEST_BY_ID: Record<string, ManifestEntry> = Object.fromEntries(",
        "  MANIFEST.map((p) => [p.id, p]),",
        ");",
        "",
        f"export const MANIFEST_TOTAL_CITATIONS = {sum(cites)};",
        f"export const MANIFEST_YEAR_MIN = {min(yrs) if yrs else 0};",
        f"export const MANIFEST_YEAR_MAX = {max(yrs) if yrs else 0};",
        f"export const MANIFEST_PAGES = {sum(int(p.get('pages') or 0) for p in rows)};",
        "",
    ]
    TS_OUT.parent.mkdir(parents=True, exist_ok=True)
    TS_OUT.write_text("\n".join(L))


def main() -> int:
    if not MANIFEST.exists():
        print(f"missing {MANIFEST}", file=sys.stderr)
        return 1
    rows = json.loads(MANIFEST.read_text())
    print(f"indexing {len(rows)} papers…", flush=True)

    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(enrich_row, rows))

    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n")

    # corpus-level histograms, handy for the audit view
    def hist(field: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            k = str(r.get(field) or "other")
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    (CORPUS / "index_summary.json").write_text(
        json.dumps(
            {
                "count": len(rows),
                "totalCitations": sum(int(r.get("citations") or 0) for r in rows),
                "totalPages": sum(int(r.get("pages") or 0) for r in rows),
                "byTask": hist("task"),
                "byModality": hist("modality"),
                "byBandwidth": hist("bandwidth"),
                "byRelevance": hist("relevance"),
                "byYear": hist("year"),
            },
            indent=2,
        )
        + "\n"
    )

    emit_ts(rows)
    print(f"wrote {TS_OUT.relative_to(ROOT)} ({len(rows)} papers)")
    print(f"wrote {(CORPUS / 'index_summary.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())