# Pose Estimation corpus

300 papers, ranked by citation count. Raw PDFs live in this directory; metadata is in `manifest.csv`.

## Provenance

| | |
| --- | --- |
| Source | arXiv API only (`export.arxiv.org/api/query`) |
| Harvest window | `submittedDate:[200708010000 TO 202610010000]` |
| Citation counts | Semantic Scholar Graph API `citationCount` |
| Selection | strict top-300 by `citationCount`, no era quotas |
| PDFs | `https://arxiv.org/pdf/<id>v<n>`, `%PDF` magic-byte validated |
| Total size | ~1.85 GB |

## The 2000–2007 gap

This corpus **cannot** cover 2000–2007, and the omission is a hard source
limitation rather than a curation choice:

- **arXiv launched in August 2007.** No paper from 2000–2007 exists on arXiv, so
  there is nothing to download for that window.
- **CVF Open Access starts at CVPR/ICCV 2013.** There is no free conference-PDF
  archive covering the 2000s.
- The remaining 2000–2007 pose literature (pictorial structures, PASCAL VOC-era
  work) sits behind IEEE/Elsevier/Springer paywalls.

Within arXiv's own window the earliest pose papers land in **2009**; the two
2009–2012 candidates never ranked high enough to enter the top 300, so the
selected corpus effectively starts at **2013**.

## Query union

Harvested from eight arXiv queries covering every pose branch, then deduped:

```
all:"pose estimation"        all:"6D pose"           all:"mesh recovery"
all:"human pose"             all:"keypoint detection" all:"category-level pose"
all:"landmark detection"     all:"pose tracking"
```

| Stage | Count |
| --- | --- |
| Raw hits (pre-dedup) | 9,716 |
| Unique arXiv IDs | 7,571 |
| Resolved by Semantic Scholar | 7,305 (96.5%) |
| Title-relevant (see below) | 4,039 |
| Selected (top 300) | 300 |

## Relevance gate

A bare `all:"pose estimation"` search matches papers that merely *mention* the
phrase. Ranking those by citation put **Mask R-CNN** (33k citations, "instance
pose"), **ControlNet** ("Adding Conditional Control to Text-to-Image Diffusion
Models", 7.8k) and **YOLOv11 overviews** at the top. Selection therefore
requires the **title** to carry a pose term (pose estimation/tracking/recovery,
keypoint, landmark, skeleton, SMPL, mesh recovery, 6D/category-level/novel-object
pose, hand/face/animal/head pose). That removed 3,532 candidates and left every
branch you asked for represented.

## Year distribution

Citations accumulate with time, so a pure top-300 ranking is structurally
biased toward the deep-learning era. This is a property of the ranking rule,
not of the harvest.

```
2013:  2      2018: 47
2014:  7      2019: 54
2015: 17      2020: 29
2016: 29      2021: 34
2017: 49      2022: 16
                 2023: 14
                 2024:  1
                 2025:  1
```

2026 is absent by construction: the window closes 2026-10-01 and nothing from
that year has accumulated meaningful citations yet.

## Manifest

`manifest.csv` extends the schema used by `papers/object_trackers/manifest.csv`:

| Column | Notes |
| --- | --- |
| `rank` | 1–300, contiguous, strictly descending in `citations` |
| `arxiv_id` | versioned, e.g. `1611.08050v2` |
| `file` | matching PDF in this directory |
| `title` | whitespace-normalized arXiv title |
| `published` | ISO 8601 arXiv submission timestamp |
| `year` | derived from `published` |
| `citations` | Semantic Scholar `citationCount` at harvest time |

Citation counts are a snapshot and will drift. Unresolved papers (266 of 7,571)
were scored 0 and therefore excluded from the top 300 rather than silently
ranked.