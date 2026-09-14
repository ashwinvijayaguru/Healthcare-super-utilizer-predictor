# Model card — superutilizer-risk v1.2.0

Generated from `artifacts/metrics.json`. Re-running `python -m ml.train` regenerates every number here.

## What it predicts

Probability that a member meets the super-utilizer definition within 6–12 months:
**≥4 ED visits, or ≥2 inpatient admissions, or total cost in the top 5% of the panel.**

## What it is for, and not for

**For:** ranking a panel so a care-management team can decide who to contact first, and explaining that ranking to the member.

**Not for:** denying or limiting care, setting premiums or benefit eligibility, individual clinical diagnosis, or any decision where being in a tier is itself the consequence. It is a prioritisation aid with a human in the loop.

## Training data

Synthetic cohort of 60,000 adults calibrated to published population statistics (MEPS, HCUP, NHIS, USDA, CDC). **No PHI.** See `docs/DATASETS.md`.

The metrics below describe the generator, not a population. Published models on real claims report AUROC 0.78–0.86, so treat this as the optimistic end.

Split: 36,000 fit / 12,000 calibrate / 12,000 holdout, stratified.

## Model

`logistic_regression`, isotonic-calibrated on a held-out split.

Both candidates were trained and compared on calibration-split AUPRC:

| Candidate | AUPRC | AUROC |
|---|---|---|
| lightgbm | 0.5381 | 0.9073 |
| logistic_regression | 0.5416 | 0.9104 |

Paired bootstrap on the difference: **-0.0038, 95% CI [-0.0124, +0.0048]** — not significant. Since the CI spans zero the models are indistinguishable on this data, so the selection rule (a challenger must beat the incumbent by 2% relative) keeps the simpler, more inspectable model.

This is partly an artifact of a log-linear generator, where a GLM is near optimal. On real claims data the booster usually wins, and the harness will select it automatically without a code change.

## Holdout performance (12,000 unseen members, 6.3% prevalence)

| Metric | Value |
|---|---|
| AUROC | 0.8825 |
| AUPRC | 0.5120 |
| Brier score | 0.0407 |
| Log loss | 0.1714 |
| Recall @ top 1% | 14.3% |
| Recall @ top 5% | 46.7% |
| Recall @ top 10% | 62.3% |
| Lift @ top 5% | 9.34× |

Read the operational line, not the AUROC: contacting the top 5% of the panel reaches **47% of everyone who will meet the outcome**.

## Calibration (holdout deciles)

| Decile | n | Predicted | Observed |
|---|---|---|---|
| 2 | 1,435 | 0.0% | 0.5% |
| 3 | 1,239 | 0.3% | 0.6% |
| 4 | 1,722 | 0.5% | 0.8% |
| 5 | 915 | 1.2% | 1.2% |
| 6 | 1,624 | 1.3% | 1.8% |
| 7 | 1,035 | 2.5% | 2.7% |
| 8 | 1,597 | 3.9% | 5.5% |
| 9 | 1,165 | 11.2% | 8.0% |
| 10 | 1,268 | 40.4% | 37.7% |

## Risk tiers

Cut points are set by care-management capacity, not by the model.

| Tier | Share of panel | Mean predicted | Observed outcome rate |
|---|---|---|---|
| low | 84.5% | 1.7% | 2.2% |
| rising | 9.0% | 15.6% | 13.0% |
| high | 4.2% | 39.8% | 35.9% |
| very high | 2.3% | 82.7% | 77.9% |

## Probability bounds

Isotonic calibration emits exactly 0.0 and 1.0 at the tails. Serving that to a patient as "100 out of 100 people like you will be hospitalised" is false — the top calibration bin has an observed rate of 81.7% — and harmful.

Output is bounded to **[0.001, 0.881]**, derived from the observed event rate in the top 240 calibration records plus Wilson-style padding. The unbounded value is still returned as `raw_score` for audit.

At the ceiling the score stops discriminating. That is correct behaviour — everyone there is already top-tier and the action is identical — but the extreme end cannot be used for fine-grained ranking.

## Subgroup performance

| Subgroup | n | Prevalence | AUROC | Calibration ratio |
|---|---|---|---|---|
| age 18 44 | 3,522 | 2.9% | 0.847 | 0.90 |
| age 45 64 | 6,028 | 5.1% | 0.853 | 1.00 |
| age 65 plus | 2,450 | 14.0% | 0.891 | 1.09 |
| medicaid or uninsured | 3,044 | 10.2% | 0.879 | 1.08 |
| commercial or medicare | 8,956 | 5.0% | 0.875 | 0.99 |
| high deprivation area | 1,298 | 19.4% | 0.834 | 1.06 |
| low deprivation area | 1,244 | 1.7% | 0.873 | 0.56 |
| any social risk | 3,749 | 11.5% | 0.869 | 1.04 |
| no social risk | 8,251 | 3.9% | 0.863 | 1.01 |

Calibration ratio is mean predicted ÷ observed; 1.00 is ideal.

**Known disparity:** the model under-predicts in low-deprivation areas (ratio 0.56). Absolute numbers there are small (prevalence 1.7%), and under-prediction in a low-risk group is the less harmful direction, but it is a real pattern to re-check on real data rather than assume away.

## Most influential features

| Feature | Importance |
|---|---|
| care continuity score | 0.702 |
| has primary care | 0.682 |
| medication count | 0.493 |
| medication complexity | 0.335 |
| area deprivation index | 0.306 |
| missed appointments 12mo | 0.246 |
| prior 30d readmission | 0.243 |
| medication adherence pdc | 0.177 |
| adherence gap | 0.177 |
| acute utilization index | 0.158 |
| ed visits 12mo | 0.140 |
| inpatient admits 12mo | 0.139 |

## Limitations

- **Trained on synthetic data.** Do not drive real outreach with this model.
- **Association, not causation.** Housing instability predicts utilisation; it does not follow that this model can tell you housing support will reduce it.
- **Historical utilisation dominates.** Members who have not engaged with care look low-risk, so under-served populations can be systematically under-scored. This is the failure mode most likely to cause real harm.
- **Area-level SDOH is ecological inference.** A county rate is not a person's circumstance. Prefer the individual answers the intake form collects.
- **No feedback loop.** If outreach works, treated members stop having events, and naive retraining learns they were never at risk.

## Human oversight

- Red flags are computed independently of the score; a low score never suppresses one.
- Every assessment stores its full input, so any score is reproducible and appealable.
- Patient summaries are validated and intended for care-team review before being shared.

