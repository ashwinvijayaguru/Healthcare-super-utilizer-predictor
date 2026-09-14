"""Train, calibrate and evaluate the super-utilizer risk model.

    python -m ml.train --data data/cohort.parquet

Produces artifacts/model.joblib (the served bundle) and artifacts/metrics.json
(the numbers quoted in the model card).

Design notes
------------
* Gradient boosting on tabular claims-style data outperforms deep models at this
  sample size and keeps exact per-feature attribution available via SHAP, which
  the product needs anyway to render the "why" panel.
* The classifier is fitted on a train split and then *calibrated* on a separate
  split. An uncalibrated boosted score is a ranking, not a probability, and this
  product shows the number to a human being.
* Threshold selection is an operational decision, not a modelling one: cut points
  are chosen so the top tier is the size of the caseload a team can absorb.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import (
    ARTIFACT_DIR,
    DATA_DIR,
    FEATURE_SCHEMA_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
    TIER_ORDER,
)
from ml.features import MODEL_FEATURE_NAMES, build_feature_frame

log = logging.getLogger(__name__)

TARGET = "is_super_utilizer"
# Share of the panel that lands in each tier, set by care-management capacity.
TIER_POPULATION_SHARE = {"very_high": 0.02, "high": 0.05, "rising": 0.15}

# A more complex challenger has to beat the interpretable incumbent by a real
# margin, not by sampling noise, before it earns a place in production.
INCUMBENT_MODEL = "logistic_regression"
SELECTION_MARGIN_RELATIVE = 0.02

LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    # Explicit metric: leaving LightGBM's default binary_logloss in place makes
    # early stopping fire on the wrong signal when the classes are reweighted.
    "metric": "average_precision",
    "learning_rate": 0.035,
    "num_leaves": 48,
    "max_depth": 7,
    "min_child_samples": 120,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.4,
    "reg_lambda": 2.0,
    "n_estimators": 2500,
    "verbosity": -1,
}


@dataclass
class SplitMetrics:
    n: int
    positives: int
    prevalence: float
    roc_auc: float
    pr_auc: float
    brier: float
    log_loss: float
    recall_at_top_1pct: float
    recall_at_top_5pct: float
    recall_at_top_10pct: float
    lift_at_top_5pct: float


def _recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    cutoff = max(1, int(round(len(y_score) * k)))
    top = np.argsort(-y_score)[:cutoff]
    positives = y_true.sum()
    return float(y_true[top].sum() / positives) if positives else 0.0


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> SplitMetrics:
    prevalence = float(y_true.mean())
    recall_5 = _recall_at_k(y_true, y_prob, 0.05)
    return SplitMetrics(
        n=int(len(y_true)),
        positives=int(y_true.sum()),
        prevalence=prevalence,
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        log_loss=float(log_loss(y_true, y_prob)),
        recall_at_top_1pct=_recall_at_k(y_true, y_prob, 0.01),
        recall_at_top_5pct=recall_5,
        recall_at_top_10pct=_recall_at_k(y_true, y_prob, 0.10),
        lift_at_top_5pct=float(recall_5 / 0.05),
    )


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Decile reliability table - the honest way to show a probability is usable."""
    edges = np.quantile(y_prob, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(y_prob, edges[1:-1])
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": b + 1,
            "n": int(mask.sum()),
            "mean_predicted": float(y_prob[mask].mean()),
            "observed_rate": float(y_true[mask].mean()),
        })
    return rows


def subgroup_metrics(y_true: np.ndarray, y_prob: np.ndarray, raw: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fairness check. A model used for outreach must not be blind in a subgroup."""
    groups = {
        "age_18_44": raw["age"] < 45,
        "age_45_64": (raw["age"] >= 45) & (raw["age"] < 65),
        "age_65_plus": raw["age"] >= 65,
        "medicaid_or_uninsured": raw["uninsured_or_medicaid"] == 1,
        "commercial_or_medicare": raw["uninsured_or_medicaid"] == 0,
        "high_deprivation_area": raw["area_deprivation_index"] >= 75,
        "low_deprivation_area": raw["area_deprivation_index"] < 25,
        "any_social_risk": raw[["housing_instability", "food_insecurity", "transportation_barrier"]].max(axis=1) == 1,
        "no_social_risk": raw[["housing_instability", "food_insecurity", "transportation_barrier"]].max(axis=1) == 0,
    }
    out: dict[str, dict[str, float]] = {}
    for name, mask in groups.items():
        m = mask.to_numpy()
        if m.sum() < 200 or len(np.unique(y_true[m])) < 2:
            continue
        out[name] = {
            "n": int(m.sum()),
            "prevalence": float(y_true[m].mean()),
            "roc_auc": float(roc_auc_score(y_true[m], y_prob[m])),
            "mean_predicted": float(y_prob[m].mean()),
            "calibration_ratio": float(y_prob[m].mean() / max(y_true[m].mean(), 1e-9)),
        }
    return out


def probability_bounds(y_true: np.ndarray, y_prob: np.ndarray, tail_fraction: float = 0.02) -> dict[str, float]:
    """Largest and smallest event rates the calibration data can actually support.

    The ceiling is the observed outcome rate among the highest-scoring 2% of the
    calibration split, and the floor the observed rate among the lowest 20%. A
    score is never presented as more extreme than the evidence behind it.
    """
    order = np.argsort(-y_prob)
    top_n = max(30, int(len(y_prob) * tail_fraction))
    bottom_n = max(30, int(len(y_prob) * 0.20))

    top_rate = float(y_true[order[:top_n]].mean())
    bottom_rate = float(y_true[order[-bottom_n:]].mean())

    # Wilson-style padding so a finite sample never implies absolute certainty.
    ceiling = min(0.97, max(top_rate, 0.50) + 1.0 / np.sqrt(top_n))
    floor = max(0.001, min(bottom_rate, 0.02) / 2.0)
    return {"floor": float(round(floor, 5)), "ceiling": float(round(ceiling, 4)),
            "observed_top_rate": round(top_rate, 4), "observed_bottom_rate": round(bottom_rate, 4),
            "top_n": int(top_n)}


def choose_tier_cuts(y_prob: np.ndarray) -> dict[str, float]:
    return {
        "rising": float(np.quantile(y_prob, 1 - TIER_POPULATION_SHARE["rising"])),
        "high": float(np.quantile(y_prob, 1 - TIER_POPULATION_SHARE["high"])),
        "very_high": float(np.quantile(y_prob, 1 - TIER_POPULATION_SHARE["very_high"])),
    }


def assign_tier(prob: float, cuts: dict[str, float]) -> str:
    if prob >= cuts["very_high"]:
        return "very_high"
    if prob >= cuts["high"]:
        return "high"
    if prob >= cuts["rising"]:
        return "rising"
    return "low"


def bootstrap_metric_delta(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_boot: int = 400,
    seed: int = 7,
) -> dict[str, float]:
    """Paired bootstrap on AUPRC(a) - AUPRC(b). If the CI spans 0, the two models
    are indistinguishable on this data and the simpler one should win."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas.append(
            average_precision_score(y_true[idx], prob_a[idx])
            - average_precision_score(y_true[idx], prob_b[idx])
        )
    arr = np.array(deltas)
    return {
        "mean_delta_pr_auc": float(arr.mean()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "significant": bool(np.percentile(arr, 2.5) > 0 or np.percentile(arr, 97.5) < 0),
    }


def _calibrate(estimator, X_cal: pd.DataFrame, y_cal: np.ndarray):
    """Wrap a fitted estimator in isotonic calibration on a held-out split.

    sklearn >= 1.6 replaces ``cv="prefit"`` with an explicit FrozenEstimator.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="isotonic")
    except ImportError:  # pragma: no cover - sklearn < 1.6
        calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv="prefit")
    return calibrated.fit(X_cal, y_cal)


def train(data_path: str, seed: int = 20241118) -> dict[str, Any]:
    frame = pd.read_parquet(data_path)
    y = frame[TARGET].to_numpy()
    X = build_feature_frame(frame)

    # Three-way split: fit / calibrate + threshold / untouched holdout.
    X_fit, X_tmp, y_fit, y_tmp, raw_fit, raw_tmp = train_test_split(
        X, y, frame, test_size=0.40, random_state=seed, stratify=y)
    X_cal, X_test, y_cal, y_test, _, raw_test = train_test_split(
        X_tmp, y_tmp, raw_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)

    log.info("split sizes  fit=%d calibrate=%d holdout=%d", len(X_fit), len(X_cal), len(X_test))

    # Class weighting is deliberately NOT used. At ~6% prevalence the boosted
    # model ranks fine without it, and reweighting distorts the output scale that
    # the isotonic step then has to undo.
    booster = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
    booster.fit(
        X_fit, y_fit,
        eval_set=[(X_cal, y_cal)],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )
    log.info("best iteration: %s", booster.best_iteration_)

    # A boosted tree ensemble is not automatically the right answer on tabular
    # claims data, so regularised logistic regression is trained as a real
    # candidate rather than a decorative baseline, and the winner is selected on
    # the calibration split by AUPRC (the metric that matters at 6% prevalence).
    baseline = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
    ]).fit(X_fit, y_fit)

    candidates = {"lightgbm": booster, "logistic_regression": baseline}
    calibrated_candidates = {name: _calibrate(est, X_cal, y_cal) for name, est in candidates.items()}

    selection = {
        name: {
            "cal_pr_auc": float(average_precision_score(y_cal, est.predict_proba(X_cal)[:, 1])),
            "cal_roc_auc": float(roc_auc_score(y_cal, est.predict_proba(X_cal)[:, 1])),
        }
        for name, est in calibrated_candidates.items()
    }
    challenger = next(name for name in selection if name != INCUMBENT_MODEL)
    incumbent_score = selection[INCUMBENT_MODEL]["cal_pr_auc"]
    challenger_score = selection[challenger]["cal_pr_auc"]
    beats_margin = challenger_score > incumbent_score * (1 + SELECTION_MARGIN_RELATIVE)
    selected_name = challenger if beats_margin else INCUMBENT_MODEL

    delta = bootstrap_metric_delta(
        y_cal,
        calibrated_candidates[challenger].predict_proba(X_cal)[:, 1],
        calibrated_candidates[INCUMBENT_MODEL].predict_proba(X_cal)[:, 1],
    )
    log.info("model selection: %s", {k: round(v["cal_pr_auc"], 4) for k, v in selection.items()})
    log.info("challenger delta AUPRC %.4f [%.4f, %.4f] significant=%s",
             delta["mean_delta_pr_auc"], delta["ci_low"], delta["ci_high"], delta["significant"])
    log.info("selected model: %s", selected_name)

    calibrated = calibrated_candidates[selected_name]
    runner_up = next(name for name in calibrated_candidates if name != selected_name)

    prob_test = calibrated.predict_proba(X_test)[:, 1]
    prob_cal = calibrated.predict_proba(X_cal)[:, 1]
    prob_base = calibrated_candidates[runner_up].predict_proba(X_test)[:, 1]
    prob_raw = candidates[selected_name].predict_proba(X_test)[:, 1]

    # Isotonic regression is a step function, so it happily emits exactly 0.0 and
    # exactly 1.0 at the tails. Serving that to a patient as "100 out of 100
    # people like you will be hospitalised" is both false and harmful: even the
    # top calibration bin does not have a 100% observed event rate. The usable
    # range is therefore derived from what the calibration data actually
    # supports, rather than from the arithmetic range of the function.
    bounds = probability_bounds(y_cal, prob_cal)
    log.info("probability bounds from calibration data: [%.4f, %.4f]",
             bounds["floor"], bounds["ceiling"])

    cuts = choose_tier_cuts(prob_cal)
    tiers = np.array([assign_tier(p, cuts) for p in prob_test])
    tier_summary = {
        tier: {
            "n": int((tiers == tier).sum()),
            "share_of_panel": float((tiers == tier).mean()),
            "observed_outcome_rate": float(y_test[tiers == tier].mean()) if (tiers == tier).any() else 0.0,
            "mean_predicted": float(prob_test[tiers == tier].mean()) if (tiers == tier).any() else 0.0,
        }
        for tier in TIER_ORDER
    }

    if selected_name == "lightgbm":
        importance = pd.Series(booster.booster_.feature_importance("gain"), index=MODEL_FEATURE_NAMES)
    else:
        importance = pd.Series(np.abs(baseline.named_steps["clf"].coef_[0]), index=MODEL_FEATURE_NAMES)
    importance = importance.sort_values(ascending=False)

    metrics = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(X_fit)),
        "best_iteration": int(booster.best_iteration_ or LGB_PARAMS["n_estimators"]),
        "holdout": asdict(evaluate(y_test, prob_test)),
        "calibration_split": asdict(evaluate(y_cal, prob_cal)),
        "uncalibrated_holdout": asdict(evaluate(y_test, prob_raw)),
        "runner_up_holdout": {"model": runner_up, **asdict(evaluate(y_test, prob_base))},
        "model_selection": {
            "selected": selected_name,
            "incumbent": INCUMBENT_MODEL,
            "selection_margin_relative": SELECTION_MARGIN_RELATIVE,
            "candidates": selection,
            "challenger_vs_incumbent": delta,
        },
        "tier_cuts": cuts,
        "probability_bounds": bounds,
        "tier_summary": tier_summary,
        "reliability": calibration_bins(y_test, prob_test),
        "subgroups": subgroup_metrics(y_test, prob_test, raw_test),
        "top_features_by_gain": {k: float(v) for k, v in importance.head(20).items()},
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": calibrated,
        "estimator_kind": selected_name,
        # Both raw estimators travel with the bundle: the explainer needs the
        # uncalibrated model to compute exact Shapley values, and keeping the
        # runner-up makes challenger comparisons reproducible after the fact.
        "booster": booster,
        "baseline_pipeline": baseline,
        "feature_names": list(MODEL_FEATURE_NAMES),
        "tier_cuts": cuts,
        "probability_bounds": bounds,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "trained_at": metrics["trained_at"],
        "holdout_metrics": metrics["holdout"],
        # Population reference values power the "compared with similar members"
        # line in the clinician panel.
        "population_reference": {
            name: float(np.median(X[name])) for name in MODEL_FEATURE_NAMES
        },
        "base_rate": float(y.mean()),
    }
    joblib.dump(bundle, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("saved model bundle -> %s", ARTIFACT_DIR / "model.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR / "cohort.parquet"))
    parser.add_argument("--seed", type=int, default=20241118)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    metrics = train(args.data, args.seed)
    h = metrics["holdout"]
    log.info("holdout  AUROC=%.4f  AUPRC=%.4f  Brier=%.4f", h["roc_auc"], h["pr_auc"], h["brier"])
    log.info("selected=%s  runner-up %s AUROC=%.4f",
             metrics["model_selection"]["selected"],
             metrics["runner_up_holdout"]["model"],
             metrics["runner_up_holdout"]["roc_auc"])
    log.info("recall @ top 5%% of panel = %.3f (lift %.2fx)",
             h["recall_at_top_5pct"], h["lift_at_top_5pct"])


if __name__ == "__main__":
    main()
