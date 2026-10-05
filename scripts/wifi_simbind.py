#!/usr/bin/env python3
"""Bind every simulator to real papers from the corpus.

The simulator bindings in ``src/labs/wifi-sensing/sims/papers.ts`` must name
papers that actually exist, otherwise the lab's provenance audit reports the
simulator as ungrounded.  Hand-maintaining twenty of those arrays drifts out of
sync as soon as the corpus is re-harvested, so the ``papers:`` arrays are
generated here instead.

Selection is deterministic and evidence-based:

1. each simulator declares a regex over the paper's own text;
2. papers matching that regex are ranked by the corpus citation order;
3. the top ``LIMIT`` ids are written into the binding.

Only the ``papers:`` line of each binding is rewritten -- the editorial
``note`` next to it stays hand-written.  Running the script twice produces the
same file.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "wifi_text"
BINDINGS = ROOT / "src" / "labs" / "wifi-sensing" / "sims" / "papers.ts"
MANIFEST = ROOT / "papers" / "wifi_sensing" / "manifest.json"

LIMIT = 4

# binding keys are written both bare (``doppler:``) and quoted (``"doppler":``)
KEY_RE = re.compile(r'^  "([a-z0-9-]+)": \{$|^  ([a-z0-9-]+): \{$')

# simulator id -> regex matched against title + abstract + body text
QUERIES: dict[str, str] = {
    # ---------- propagation ----------
    "fresnel-zone": r"fresnel zone|fresnel clearance|fresnel",
    "path-loss": r"path[- ]loss exponent|log[- ]distance path|shadowing (?:factor|fading)|path loss",
    "csi-multipath": r"multipath|channel impulse response|delay spread|reflection coefficient",
    "doppler": r"doppler shift|doppler frequency|doppler",
    "rssi-vs-csi": r"received signal strength|\brssi\b",
    "phase-wrap": r"phase unwrapp|phase wrapping|unwrap\w* the phase|phase sanitiz|sanitizefilims",
    # ---------- features ----------
    "stft-velocity": r"short[- ]time fourier|\bstft\b|spectrogram",
    "periodogram": r"periodogram|power spectral density|\bpsd\b|welch",
    "feature-families": (
        r"doppler frequency ratio|\bdfr\b|\bfcs\b|\bfcc\b|csi ratio|"
        r"feature famil|entropy feature|signal[- ]domain feature"
    ),
    "fft-invariance": r"fft|fourier transform|translation invariant|shift[- ]invariant",
    "conv1d-window": r"window size|sliding window|window function|segmentation window|windowed",
    "quantization-noise": r"quantiz\w+|bit depth|\bint8\b|precision of the csi|amplitude quantization",
    "cfo-sfo": r"carrier frequency offset|sampling frequency offset|\bcfo\b|\bsfo\b|packet detection",
    # ---------- tasks ----------
    "activity-clf": r"activity recognition|gesture recognition|human activity",
    "rssi-fingerprint": r"fingerprint\w*|radiomap|radio map|\bsvm\b",
    "kalman-trajectory": r"kalman|particle filter|trajectory (?:smoothing|track)|tracking filter",
    "vital-signs": r"vital sign|breathing|breath rate|heart rate|respiration",
    "snr-margin": r"signal[- ]to[- ]noise|\bsnr\b|noise floor|link margin|receiver sensitivity",
    # ---------- robustness ----------
    "domain-shift": (
        r"cross[- ](?:domain|room|environment)|domain adaptation|domain generali|"
        r"transfer learning|environment[- ]independent"
    ),
    "through-wall": r"through[- ]wall|\bnlos\b|non[- ]line[- ]of[- ]sight|occluded",
}


def load_text(file_name: str) -> str:
    path = CACHE / (pathlib.Path(file_name).stem + ".txt")
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def main() -> int:
    if not CACHE.exists():
        print(f"missing {CACHE}; run scripts/wifi_index.py first", file=sys.stderr)
        return 1

    rows = json.loads(MANIFEST.read_text())
    haystacks: dict[str, str] = {}
    for row in rows:
        haystacks[row["id"]] = " ".join(
            [row.get("title") or "", load_text(row.get("fileName", ""))]
        )

    # citation order == manifest order, so a stable sort by id rank is enough
    rank = {row["id"]: i for i, row in enumerate(rows)}

    picked: dict[str, list[str]] = {}
    for sim_id, pattern in QUERIES.items():
        rx = re.compile(pattern, re.I)
        hits = [r["id"] for r in rows if rx.search(haystacks.get(r["id"], ""))]
        hits.sort(key=lambda pid: rank[pid])
        picked[sim_id] = hits[:LIMIT]

    src = BINDINGS.read_text()
    missing = [s for s in QUERIES if not any(m and (m.group(1) or m.group(2)) == s for m in (KEY_RE.match(l) for l in src.splitlines()))]
    if missing:
        print(f"bindings.ts is missing simulator ids: {missing}", file=sys.stderr)
        return 1

    # Walk the file once, tracking the current binding key. When a `papers: [`
    # line is found, emit a fresh block and *consume the old block up to and
    # including its closing `],` so re-running the script is idempotent.
    lines = src.splitlines()
    out_lines: list[str] = []
    current: str | None = None
    replaced = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        m = KEY_RE.match(line)
        if m:
            current = m.group(1) or m.group(2)
            out_lines.append(line)
            i += 1
            continue
        if re.match(r"^    papers: \[", line):
            ids = picked.get(current or "", [])
            out_lines.append("    papers: [")
            for pid in ids:
                out_lines.append(f"      {json.dumps(pid)},")
            out_lines.append("    ],")
            replaced += 1
            # skip the stale block: everything up to the next `    ],`
            j = i + 1
            while j < len(lines) and not re.match(r"^    \],", lines[j]):
                j += 1
            i = j + 1
            continue
        out_lines.append(line)
        i += 1

    BINDINGS.write_text("\n".join(out_lines) + "\n")

    empty = [s for s, ids in picked.items() if not ids]
    print(f"rewrote {replaced} bindings from {len(rows)} papers")
    for sim_id in sorted(picked):
        print(f"  {sim_id:20s} {len(picked[sim_id])} papers  {picked[sim_id]}")
    if empty:
        print(f"WARNING: no corpus paper matched: {empty}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())