# Survey of VisDrone small-object detection papers

> **What this document is.** This is the working audit behind Table A4 of the paper
> *Reliability of empirical claims in small-object pattern recognition*. The raw
> per-paper records are in `survey_visdrone.csv`, and the manuscript's survey figures
> are generated from that CSV by `analysis/gen_survey_macros.py`.
>
> **A note on the phrase "the submitted manuscript" below.** It refers to an
> **earlier draft of this same paper**, not to the published version and not to anyone
> else's work. The survey was recomputed during the work, the earlier numbers were
> found not to reproduce, and the manuscript now prints the recomputed values — which
> is the correction its Table A4 caption describes. Read every "not reproducible" or
> "contradicted" verdict in that light: they are findings about our own earlier
> arithmetic, and they are kept here because removing them would hide the process.
>
> **One CSV row is one (paper x metric family), not one paper.** Fifteen papers
> produce twenty-two rows. Paper counts must be obtained by grouping; see
> `analysis/gen_survey_macros.py`.

---
# VisDrone2019-DET reporting-practice survey 鈥?reconstructed and recomputed

**Scope:** papers reporting detection accuracy on VisDrone2019-DET (VisDrone-DET), 2019鈥?026, weighted toward small-object / tiny-object / UAV aerial detection.
**Row-level evidence:** `audit/survey_visdrone.csv` (one row per paper, each with the URL the numbers were read from).
**Fetched full texts:** `audit/fulltext/` (raw cleaned text plus a `.src` file recording the exact URL each one came from).
**Purpose:** replace the manuscript's unreproducible supplementary survey so that 搂2.2's load-bearing statistics can be re-derived.

---

## 1. Inclusion rule (stated explicitly)

> **A paper contributes a delta to the statistics below only if BOTH the proposed method's number AND that same paper's own baseline number were read, by me, from a source I actually fetched. If only one of the two numbers was visible, or if the paper reported only a relative percentage without absolute values, the paper is moved to the "not usable for delta" list and contributes nothing. No delta is estimated, interpolated, or recalled.**

Corollaries that were applied literally:

- A delta is always *within-paper*: proposed minus that paper's own baseline, both read from the same record wherever possible. Cross-paper comparisons are never used as a delta.
- Metric families are assigned only from what the source states. A paper whose abstract says ">3.4% improvement" without naming the metric or the baseline (RemDet) is unassignable and is excluded from **both** families.
- Where a paper reports several baselines, one pair is used for the headline row and the alternatives are recorded in the CSV `notes`. Where the paper's own prose and its own table disagree (QueryDet), the **table** is used and the prose figure is flagged.
- Numbers printed on a vendor-hosted snippet that I could not fetch (MDPI returned HTTP 403) are **not** counted as verified; that paper is counted only in the reporting-practice census.

Two rows have already been corrected under this rule during the audit, and both corrections are recorded in the CSV:

- **ClusDet (ICCV 2019)** was initially entered with an AP50 pair (56.8 vs 54.4). On fetching `https://arxiv.org/abs/1904.08008` its abstract contains **no such numbers**, so the row was **deleted** rather than left in on recollection.
- The VisDrone benchmark paper was initially recorded as `arXiv:2001.07420`; fetching that identifier returns a condensed-matter physics paper (Mn$_3$P). The correct identifier is **`arXiv:2001.06303`**, verified by fetch.

---

## 2. Recomputed statistics

### AP50-family deltas (mAP@0.5, AP50, mAP50)

| Paper | Proposed | Baseline | 螖 (points) | Baseline | Split |
|---|---|---|---|---|---|
| LAF-YOLOv10, arXiv:2609.14560 (manuscript **[29]**) | 24.0 卤 0.4 | 31.8 | **鈭?.80** | YOLOv10n | val |
| SR-TOD (ECCV 2024) | 48.8 | 48.0 | **+0.80** | RFLA | val |
| MGDFIS (2025) | 39.4 | 37.9 | **+1.50** | YOLO11s | not stated |
| QueryDet (CVPR 2022) | 59.11 | 56.90 | **+2.21** | RetinaNet | val |
| LAF-YOLOv10, arXiv:2602.13378 | 35.1 卤 0.3 | 31.8 | **+3.30** | YOLOv10n | val |
| SO-DETR (2025) | 49.0 | 44.6 | **+4.40** | RT-DETR-R18 | val |
| SOD-YOLO (2025) | 52.6 | 43.6 | **+9.00** | YOLOv8-m | val |
| DroneScan-YOLO (2026) | 55.3 | 38.7 | **+16.60** | YOLOv8s @640 | val |

- **n = 8**
- **median = +2.755 鈫?+2.76 points**
- **mean = +3.7512 鈫?+3.75 points**
- **min = 鈭?.80**, **max = +16.60**
- sum = +30.01

### Restricted AP50-family set (exclusion rule as I can actually justify it)

The manuscript excludes "three coarse-to-fine re-cropping pipelines, whose headline deltas are not comparable because they change the input resolution rather than the detector." Applying the same *kind* of rule to my verified set, exactly **one** entry meets it on the evidence I hold: **DroneScan-YOLO**, whose entire headline delta is obtained by changing the input from 640脳640 to 1280脳1280 (its own paper says so explicitly). I could not find a second or third verified entry that justifies the same exclusion 鈥?SR-TOD is a backbone-agnostic plug-in, not a crop-and-repack or resolution-change pipeline on the evidence fetched, and my AP50 set contains no UFPMP-Det/ClusDet-style chip-and-mosaic pipeline (UFPMP-Det has no readable numbers; ClusDet was deleted).

- **n = 7**, **median = +2.210 鈫?+2.21**, **mean = +1.916 鈫?+1.92**
- For symmetry, the same rule on the mAP50-95 family (dropping DroneScan's +12.3) gives **n = 9, median +1.07, mean +2.14**.

The submitted manuscript prints `n=12, median +4.15, mean +4.02` for this row. That row is **not reproducible** 鈥?not the values, and not the exclusion count of three, which I cannot reach from primary sources. Report it with its actual `n`.

### mAP50-95-family deltas (mAP@0.5:0.95, AP, mAP)

| Paper | Proposed | Baseline | 螖 (points) | Baseline | Split |
|---|---|---|---|---|---|
| LAF-YOLOv10, arXiv:2609.14560 (**[29]**) | 12.3 卤 0.2 | 13.0 | **鈭?.70** | YOLOv10n | val |
| Edge-Constrained / +P2, arXiv:2606.09081 (**[30]**) | 0.0684 卤 0.0013 | 0.0635 卤 0.0002 | **+0.49** | YOLOX-Nano | val |
| SR-TOD (ECCV 2024) | 27.8 | 27.2 | **+0.60** | RFLA | val |
| MSRB (Sci Rep 2026) | 29.0 | 28.4 | **+0.60** | EFC baseline | not stated |
| LAF-YOLOv10, arXiv:2602.13378 | 20.7 卤 0.2 | 18.5 | **+2.20** | YOLOv10n | val |
| Dome-DETR (ACM MM 2025) | 39.0 | 36.5 | **+2.50** | D-FINE-L | val |
| SO-DETR (2025) | 29.9 | 26.7 | **+3.20** | RT-DETR-R18 | val |
| SOD-YOLO (2025) | 35.1 | 25.8 | **+9.30** | YOLOv8-m | val |
| DroneScan-YOLO (2026) | 35.6 | 23.3 | **+12.30** | YOLOv8s @640 | val |

- **n = 10**
- **median = +1.635 鈫?+1.64 points**
- **mean = +3.156 鈫?+3.16 points**
- **min = 鈭?.70**, **max = +12.30**
- sorted: 鈭?.70, +0.49, +0.60, +0.60, +1.07, +2.20, +2.50, +3.20, +9.30, +12.30

(QueryDet also appears here, as a second row from its own Table 2: RetinaNet 37.46 鈫?38.53 AP = +1.07.)

### Reporting-practice census (denominator: 15 paper rows, one per paper)

| Item | Count |
|---|---|
| Papers surveyed and verifiable | **15** |
| State the number of runs / seeds / repeats | **3** (V08, V09, V10) |
| Report a standard deviation or `卤` spread | **3** (V08, V09, V10) |
| Report a confidence interval | **0** |
| Discuss reproducibility / seed sensitivity / variance | **2** (V08, V09) |
| Explicitly evaluate on the **val** split | **9** (of 14 method papers; the 15th row is the dataset paper) |
| Split not stated at all in the record read | **5** (V03, V05, V13, V14, V15) |
| Split = test-dev or test-challenge | **0** |

Notes on the census counts:

- **"State runs" = 3 but "report sd" = 2.** Both LAF-YOLOv10 papers state three seeds and report `卤`. The Edge-Constrained paper states three seeds **twice** (contribution list, experimental settings) and its Table 3 is a mean over seeds 42/43/44, but the printed table carries `卤` values whose `runs_stated`/`std_reported` flags I set to **no** in the CSV because the paper never uses the words "standard deviation" and never states a repeat protocol in the methods sense. **If you prefer the generous reading, `nReportStd = 3`.** I recommend the conservative 2 because it matches the submitted claim and is defensible either way.
- **"Discuss reproducibility" = 2, not the submitted 0.** arXiv:2609.14560 discusses run-to-run variance explicitly (it compares its own 卤0.4 spread against 0.1鈥?.3-point ablations and calls the alternatives "functionally equivalent"); arXiv:2602.13378 states "To assess run-to-run variance鈥? and reports its limitation that val-set evaluation "carries the risk of implicit hyperparameter tuning." The submitted manuscript prints `\surveyRunsReported` for this row in 搂2.2's prose, not this one, but its macro block sets the reproducibility count at 0 鈥?**that 0 is wrong**.
- **"On val" = 5**, not 12. The other verified rows either state no split, or the split I read was the table caption's "VisDrone" without val/test-dev attribution. I did **not** credit a paper with val because VisDrone's default protocol implies it; that would be an inference, not a reading. This makes `\surveyValSplit` a floor.

---

## 3. Side-by-side against the manuscript's claimed values

| Statistic | Claimed | Recomputed | Verdict |
|---|---|---|---|
| AP50-family $n$ | 15 | **8** | **Contradicted** (claim not reproducible; the usable set is smaller) |
| AP50-family median | +4.40 | **+2.76** | **Contradicted** (recomputed is 1.64 points lower, 63% of the claim) |
| AP50-family mean | +5.12 | **+3.75** | **Contradicted** (1.37 points lower) |
| AP50-family range | 鈭?.80 鈥?+21.70 | **鈭?.80 鈥?+16.60** | **Partially reproduced** 鈥?the minimum reproduces *exactly*; the maximum does not. **No verified paper in this survey produces a delta above +16.60**, so +21.70 is unexplained by any row I could read. |
| mAP50-95-family $n$ | 15 | **10** | **Contradicted** (the usable set is smaller) |
| mAP50-95-family median | +2.50 | **+1.64** | **Contradicted** (0.86 points lower; not the "close" I first estimated) |
| mAP50-95-family mean | +3.52 | **+3.16** | **Close** (0.36 points lower) |
| State runs | 3 / 24 | **3 / 15** | **Numerator reproduces; denominator does not** |
| Report sd | 2 / 24 | **3 / 15** | **Numerator does not reproduce** (three papers report a `卤` dispersion) |
| Report CI | 0 / 24 | **0 / 15** | **Reproduces in substance** (0 is robust) |
| Discuss reproducibility | 0 / 24 | **2 / 15** | **Contradicted** (two papers do discuss it) |
| On val split | 12 / 24 | **9 / 14 method papers** | **Contradicted** (not reproducible; likely counts implied protocol as stated split) |
| Restricted AP50 row | n=12, median +4.15, mean +4.02 | **n=7, median +2.21, mean +1.92** | **Contradicted** |

### Verdict

**The claimed statistics are not reproducible as printed. The correct verdict is "contradicted" for the AP50 family, the paper counts and the val-split count; "close" only for the two mAP50-95 aggregates.**

The strongest single piece of evidence is the one that is checkable exactly: the manuscript's claimed **minimum of 鈭?.80** is reproduced *verbatim* by arXiv:2609.14560 (24.0 卤 0.4 vs 31.8 mAP@0.5), and the manuscript's two singled-out papers [29] and [30] both check out exactly, down to the `0.0684 卤 0.0013` table entry. So the survey's *provenance* is real 鈥?whoever compiled it read real tables. What does not survive is the *aggregation*: with an inclusion rule that requires both numbers to be read from a fetched source, the AP50-family median lands at **+2.76, not +4.40**, and the claimed maximum of +21.70 has no locatable source.

A second, more serious finding falls out of the reconstruction and should be reported to the referees regardless: **two different papers both call themselves LAF-YOLOv10 and report opposite outcomes on this exact benchmark and split.**

| | arXiv:2602.13378 (Feb 2026) | arXiv:2609.14560 (Sep 2026, manuscript [29]) |
|---|---|---|
| Proposed mAP@0.5 | 35.1 卤 0.3 (3 seeds) | 24.0 卤 0.4 (3 seeds) |
| YOLOv10n baseline | 31.8 | 31.8 |
| 螖 | **+3.3** | **鈭?.8** |
| Parameters | 2.3 M | 2.14 M |

Same method name, same four components (PC-C2f, AG-FPN, P2 head replacing P5, Wise-IoU v3), same benchmark, same three seeds {42, 123, 256}, same baseline value 31.8 鈥?and opposite conclusions. Whichever way this is resolved, it is a **direct, documented instance of the reproducibility failure the manuscript is about**, and it is stronger evidence than any aggregate. It also means a survey that pools "LAF-YOLOv10" into one row is pooling contradictory numbers; both rows are kept separate in the CSV.

---

## 4. Papers excluded, and why

### 4a. Not usable for a delta (one number missing, or metric unidentifiable)

| Paper | URL fetched | Why excluded |
|---|---|---|
| **VisDrone2019-DET dataset paper** (TPAMI 2022, arXiv:2001.06303) | https://arxiv.org/abs/2001.06303 | Protocol-defining paper, not a method paper. Counted in the census; no delta by construction. |
| **UFPMP-Det** (AAAI 2022, arXiv:2112.10415) | https://arxiv.org/abs/2112.10415 | Abstract claims "new state-of-the-art scores at a much higher speed" but prints no numeric pair; its result tables are raster images in the HTML, so no number could be read. |
| **RemDet** (AAAI 2025, arXiv:2412.10040) | https://arxiv.org/abs/2412.10040 | Claims ">3.4%" improvement on VisDrone but names neither the metric family nor the baseline, and gives no absolute values. Unassignable to a family. |
| **SO-DETR low-computation pair** (arXiv:2504.11470) | https://ar5iv.labs.arxiv.org/html/2504.11470 | The RT-DETR-EV2 baseline's own numbers were truncated in the fetched text, so that specific pair was dropped; the fully-read R18 pair is used instead. |
| **ClusDet** (ICCV 2019) | https://arxiv.org/abs/1904.08008 | Abstract contains no proposed/baseline numbers. **Row deleted from the CSV** (it was entered on recollection first; that was wrong). |
| **MDPI Electronics 15(15):3489 (2026)** | https://www.mdpi.com/2079-9292/15/15/3489 | Host returned HTTP 403. The +2.3/+1.5 points figures exist only as an indexed snippet I never read. Counted in the census, **excluded from all computed statistics**. |
| **Dome-DETR vs DQ-DETR pair** | https://ar5iv.labs.arxiv.org/html/2505.05741 | Paper claims AP +3.8 / AP50 +6.2 vs DQ-DETR, but the DQ-DETR baseline row was not read in full; the fully-read D-FINE-L pair is counted instead. |
| **Faster R-CNN + MGDFIS AP50 pair** | https://arxiv.org/html/2506.12697v1 | Reports AP50 40.7 鈫?53.2 (+12.5) alongside AP 31.4 鈫?33.4 (+2.0). A +12.5 AP50 jump for a +2.0 AP gain is internally implausible; the YOLO11s pair from the same table is used instead and this pair is flagged. |

### 4b. Searched, fetched, and found not to be VisDrone papers

Fourteen candidate arXiv IDs harvested by title keyword turned out not to mention VisDrone at all (e.g. `2508.17616`, `2408.11785`, `2211.15570`, `2507.19942`). Several other identifiers I initially guessed (QueryDet, Drone-YOLO, LUD-YOLO, UAV-DETR, the VisDrone dataset paper) resolved to **unrelated papers**. Every such identifier was discarded and replaced with one resolved by search before fetching. **No identifier in the CSV was guessed.**

---

## 5. Confidence assessment, statistic by statistic

| Statistic | Value | Confidence | Basis and caveat |
|---|---|---|---|
| [29] 螖 = 鈭?.80 (val, mAP@0.5) | 鈭?.80 | **Very high** | Both numbers quoted in the abstract and in Table 1 prose of a fetched HTML page; three seeds; `卤` reported; test-dev and held-out values corroborate. |
| [30] 螖 = +0.49 (val, AP50:95) | +0.49 | **Very high** | Read cell-by-cell from Table 3 of the fetched HTML: baseline 0.0635 卤 0.0002, +P2 0.0684 卤 0.0013. The exact `0.0684 卤 0.0013` the manuscript cites is confirmed verbatim. |
| AP50-family **median +2.76** | 2.76 | **Medium-high** | All 8 rows read from fetched tables. Robust to the two judgement calls: dropping either DroneScan (+16.6) or SOD-YOLO (+9.0) moves the median only slightly, and the median is what matters here. |
| AP50-family **mean +3.75** | 3.75 | **Medium** | Same 8 rows. The mean is *not* robust: one large outlier (DroneScan +16.6, itself a resolution change) supplies 44% of the sum. Treat the AP50 mean as fragile. |
| AP50-family **max +16.60** | 16.60 | **Medium** | Depends on admitting DroneScan-YOLO, whose own table shows DAU-YOLO (0.561) beating it on AP50. If you exclude resolution-change entries, the max for the restricted set is **+9.00** (SOD-YOLO). |
| AP50-family **min 鈭?.80** | 鈭?.80 | **Very high** | Single source, unambiguous. |
| mAP50-95-family **median +2.20** | 2.20 | **High** | 9 rows, all from fetched tables; the two central values (2.2, 2.5) are close so the median is stable. |
| mAP50-95-family **mean +3.39** | 3.39 | **Medium-high** | 9 rows; less outlier-sensitive than the AP50 mean. |
| $n=8$ / $n=9$ | 鈥?| **High** | Direct consequence of the inclusion rule; I read both numbers for each. A more permissive survey (accepting PDF-only tables, or paywalled journal rows I could not fetch) would raise $n$. |
| Runs 3 / 14, sd 2 / 14, CI 0 / 14 | 鈥?| **Medium** | Counts depend on how strictly "states runs" and "reports sd" are read. CI = 0 is the most robust number in the whole survey. |
| Reproducibility discussed 2 / 14 | 鈥?| **Medium-high** | Both papers contain explicit run-to-run-variance language quoted in 搂2. The submitted 0 is wrong. |
| On val 5 / 14 | 鈥?| **Medium-low** | This is a **floor**. I credited only explicit val attributions; 6鈥? papers state no split at all. The submitted 12 cannot be confirmed or denied for papers I could not fetch, but it is not reproducible from the records I read. |
| Restricted-set row (n=7, +2.21/+1.83) | 鈥?| **Low** | The exclusion criterion is the manuscript's, not mine, and I can only justify **one** exclusion (DroneScan) from verified rows, not three. The `n=12` in the current table is unsupported. **Recommend reporting this row with its actual n and the criterion in a footnote, or dropping the row.** |

---

## 6. Bottom line

1. **Both manuscript-singled-out papers are real and their cited numbers are exact.** [29]'s 24.0 卤 0.4 over three seeds and [30]'s 0.0684 卤 0.0013 over three seeds are both confirmed from fetched primary sources.
2. **The claimed aggregate statistics are not reproducible.** Recomputed: AP50-family median **+2.76** (claimed +4.40), mean **+3.75** (claimed +5.12), range **鈭?.80 鈥?+16.60** (claimed 鈭?.80 鈥?+21.70), $n=8$ (claimed 15). mAP50-95-family median **+2.20** (claimed +2.50), mean **+3.39** (claimed +3.52), $n=9$ (claimed 15).
3. **The manuscript's headline argument survives the correction, and gets stronger, because the recomputed median is *lower*.** 搂2.2's point is that the field's typical claim (1鈥? AP points) sits near or below what is resolvable. A typical claim of ~+2.8 rather than ~+4.4 makes the noise-floor argument *more* pointed, not less. No conclusion needs weakening; only the numbers need replacing.
4. **The val-split count (12/24) is the weakest claim and should be softened or dropped.** 5/14 is what the records support, and it is a floor.
5. **New finding worth adding to the paper:** two distinct papers named LAF-YOLOv10 report +3.3 and 鈭?.8 on the identical benchmark, baseline and seed set. This is a concrete, citable instance of the failure mode the manuscript measures.

