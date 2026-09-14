# Datasets

## Position on PHI

Nothing in this repository is protected health information. The demo cohort is
**generated**, not sampled from any patient record, and every real dataset named
below is a public-use file with no data use agreement and no identifiers.

Be precise with the client about what that buys and what it costs. It removes
the BAA, the IRB conversation and the breach surface entirely — you can put this
on a laptop and demo it anywhere. It also means the reported metrics describe the
generator, not a population, and must not be presented as validated clinical
performance. See "How to read the numbers" below.

## What actually trained the shipped model

`ml/generate_cohort.py` — 60,000 synthetic adults with a 6–12 month outcome window.

The generator is not arbitrary. It samples two correlated latent variables
(clinical burden, social adversity), generates observed features conditioned on
them, then generates future utilisation from those features **plus a fresh shock
term**. The shock is what produces regression to the mean and keeps achievable
AUROC in the 0.80–0.88 band reported in the super-utilizer literature rather than
an unrealistic 0.99.

Marginal distributions and effect sizes are calibrated to published population
statistics, and `tests/test_system.py::test_cohort_matches_published_benchmarks`
fails the build if the generator drifts away from them:

| Quantity | Generated | Published benchmark | Source |
|---|---|---|---|
| Top 5% share of total spend | 52.9% | ~50% | MEPS spending concentration |
| Adults with ≥1 ED visit/year | 20.2% | ~20% | NHIS / CDC |
| Adults with ≥4 ED visits/year | 1.05% | 1–2% | HCUP NEDS |
| Polypharmacy (≥10 medications) | 10.5% | ~10% overall | CDC NHANES |
| Multimorbidity (≥2 chronic) | 42.2% | 27–42% | CDC / RAND |
| Food insecurity | 17.1% | 13.5% households | USDA ERS |
| Mean age | 54.1 | skews above the 48-year adult mean | care-management panel |
| Super-utilizer prevalence | 6.3% | 5–10% of a programme panel | CMMI / Camden |

Two of these deserve a caveat rather than a tick. Food insecurity at 17% runs
above the USDA household rate; it is defensible for a Medicaid-weighted panel but
it is a choice, not a match. Mean age is deliberately older than the general
adult population because a care-management panel is.

## Public datasets the adapters target

`ml/ingest/` contains working loaders that map each source onto the same feature
contract. Once an adapter returns `FEATURE_NAMES + is_super_utilizer`,
`python -m ml.train --data <file>` needs no other change.

### CMS DE-SynPUF — `ml/ingest/cms_desynpuf.py`
Synthetic 5% sample of Medicare beneficiaries, 2008–2010, released by CMS with
no use agreement. The best public substitute for real claims.
<https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files>

- **Gives you:** real claims structure, chronic condition flags, inpatient and
  outpatient utilisation, payment amounts, a natural observation/outcome split
  (2008 features → 2009–2010 outcome).
- **Does not give you:** any social determinant at all, medication adherence,
  missed appointments, or an ED flag (outpatient ED revenue-centre claims are the
  standard proxy and the adapter uses it).
- **Gotcha:** condition flags are coded `1 = has`, `2 = does not`. Treating them
  as 0/1 inverts every condition in your model and is a genuinely easy mistake.

### MEPS Household Component — `ml/ingest/meps.py`
The only major public US dataset carrying individual-level utilisation, spending
**and** social circumstance on the same person. This is the right source for the
SDOH half of the model.
<https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp>

- **Gives you:** ED and inpatient counts, total expenditure, prescription counts,
  poverty category, insurance, usual source of care, K6 distress, food security,
  cost-related delayed care, household size.
- **Panel structure gives the split for free:** each panel follows a household for
  two calendar years — year 1 features, year 2 outcome. Use the longitudinal file.
- **Gotcha:** column names carry a two-digit year suffix (`ERTOT21`, `TOTEXP22`)
  that changes every release. The adapter builds names from a `year` argument;
  check the codebook for your download before trusting a run.

### AHRQ SDOH Database — `ml/ingest/sdoh_ahrq.py`
County and census-tract social measures, free and unrestricted.
<https://www.ahrq.gov/sdoh/data-analytics/sdoh-data.html>

Used to attach social context to claims files that carry none.

**State this caveat to the client directly:** these are *area* rates. Assigning a
county's food-insecurity rate to a person is ecological inference. It is a
reasonable prior when nothing better exists; it is not the same as asking the
patient. This is exactly why the web form asks the patient directly, and why
individual answers should always override the area join.

### Others worth knowing
- **MIMIC-IV** (PhysioNet) — real ICU/ED data, de-identified, but requires
  credentialing and a DUA. Better for acute physiology than for a 6–12 month
  social-driver model.
- **HCUP NRD / NEDS** (AHRQ) — the gold standard for readmission and ED
  denominators. Purchase required; excellent for validating your base rates.
- **CDC BRFSS** — large annual survey with an SDOH module. Self-reported and
  cross-sectional, so useful for calibrating prevalence, not for training.
- **Neighborhood Atlas ADI** (Wisconsin) — the canonical Area Deprivation Index,
  free with registration. Drop-in replacement for the synthetic `area_deprivation_index`.

## Suggested path to a real model

1. Train on **MEPS** first. It is the only public source with individual social
   data, so the SDOH coefficients mean something. Expect AUROC ~0.75–0.80 — lower
   than the synthetic figure, and the honest number.
2. Validate the utilisation half on **DE-SynPUF**, which has real claims structure.
3. Replace the synthetic ADI with **Neighborhood Atlas** values joined on ZIP.
4. Refit on the client's own claims once a BAA exists. At that point re-run the
   candidate selection: the boosted model very likely wins on real data (see below).

## How to read the numbers

Holdout performance on 12,000 unseen synthetic members:

| Metric | Value |
|---|---|
| AUROC | 0.883 |
| AUPRC | 0.512 (prevalence 6.3%) |
| Brier | 0.041 |
| Recall @ top 5% of panel | 46.7% (9.3× lift) |
| Top tier | 2.3% of panel, 78% observed event rate |

**These describe the generator, not a population.** Real super-utilizer models
published on claims data report AUROC 0.78–0.86, so this sits at the optimistic
end — expect a few points lower on real data, and say so before the client
anchors on 0.88.

One finding to carry forward: **LightGBM did not beat regularised logistic
regression here.** A paired bootstrap on the AUPRC difference gave −0.0038, CI
[−0.0124, +0.0048] — indistinguishable — so the selection rule in `ml/train.py`
picks the simpler model. That is partly an artifact of a log-linear generator,
where a GLM is close to optimal. On real claims with threshold effects and
interactions the booster usually wins, and the harness will pick it up
automatically without a code change.
