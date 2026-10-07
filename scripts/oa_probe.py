#!/usr/bin/env python3
"""Probe whether OpenAlex has lifted its 429, with real backoff between tries.

Written because both OpenAlex and Semantic Scholar started returning 429 for
every request, including a minimal `?per-page=3` call, which points at a
per-IP quota rather than a malformed query. arXiv and Crossref answered
immediately, so this only answers the narrower question of when OpenAlex comes
back -- it is still the only source that supplies the `cited_by_count` the
corpus is ordered by, and mixing in Crossref counts would mean sorting the lab
by two different metrics at once.

Run:  python3 scripts/oa_probe.py [--wait-minutes 60]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import harvest_transformers as T

UA = {"User-Agent": T.UA["User-Agent"]}
PROBE = {
    "search": "efficient attention",
    "filter": f"from_publication_date:{T.FROM_DATE},to_publication_date:{T.TO_DATE}",
    "sort": "cited_by_count:desc",
    "per-page": 25,
    "mailto": T.MAILTO,
    "select": T.SELECT,
}


def probe() -> tuple[int, int]:
    """Return (http_status, result_count); 0 means a non-HTTP failure."""
    u = T.OA + "?" + urllib.parse.urlencode(PROBE)
    try:
        with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=90) as r:
            return 200, len(json.load(r).get("results") or [])
    except urllib.error.HTTPError as e:
        return e.code, 0
    except Exception as e:
        print(f"    ! {type(e).__name__}: {str(e)[:60]}")
        return 0, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-minutes", type=float, default=60.0)
    ap.add_argument("--interval", type=float, default=180.0)
    args = ap.parse_args()

    deadline = time.time() + args.wait_minutes * 60
    n = 0
    while time.time() < deadline:
        n += 1
        status, count = probe()
        stamp = time.strftime("%H:%M:%S")
        if status == 200:
            print(f"[{stamp}] probe {n}: HTTP 200, {count} results -- OpenAlex is back")
            return 0
        print(f"[{stamp}] probe {n}: HTTP {status}, waiting {args.interval:.0f}s")
        time.sleep(args.interval)
    print(f"still HTTP-limited after {args.wait_minutes:.0f} min / {n} probes")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())