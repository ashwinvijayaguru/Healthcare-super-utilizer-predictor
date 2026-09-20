import type { Assessment, ModelInfo } from "../lib/types";

const TIER_COLOR: Record<string, string> = {
  low: "var(--risk-low)",
  rising: "var(--risk-rising)",
  high: "var(--risk-high)",
  very_high: "var(--risk-very-high)",
};

/**
 * The hero. A continuous band with the operational tier cut points drawn as
 * real tick marks, so a care manager can see not just the score but how close
 * it sits to the boundary that changes what they do about it.
 *
 * The scale is capped at the model's supported ceiling rather than 100%,
 * because the model is not permitted to claim certainty it cannot support.
 */
export default function RiskBand({
  assessment,
  model,
}: {
  assessment: Assessment;
  model: ModelInfo | null;
}) {
  const ceiling = Math.max(assessment.risk_score, 0.9);
  const pct = (value: number) => Math.min(100, (value / ceiling) * 100);

  const cuts = model?.tier_cuts ?? {};
  const ticks = [
    { key: "rising", value: cuts.rising, label: "Rising" },
    { key: "high", value: cuts.high, label: "High" },
    { key: "very_high", value: cuts.very_high, label: "Very high" },
  ].filter((t) => typeof t.value === "number");

  return (
    <div>
      <div className="readout">
        <div
          className="readout__number"
          style={{ color: TIER_COLOR[assessment.risk_tier] }}
        >
          {assessment.risk_percent.toFixed(1)}
          <sup>%</sup>
        </div>
        <div
          className="readout__tier"
          style={{ color: TIER_COLOR[assessment.risk_tier] }}
        >
          {assessment.risk_tier_label}
        </div>
        <div className="readout__context">
          {assessment.lift_vs_baseline.toFixed(1)}× the panel baseline of{" "}
          {(assessment.baseline_rate * 100).toFixed(1)}%
          <br />
          around the {Math.round(assessment.percentile ?? 0)}th percentile
        </div>
      </div>

      <div className="band">
        <div className="band__track">
          {ticks.map((tick) => (
            <span
              key={tick.key}
              className="band__tick"
              style={{ left: `${pct(tick.value as number)}%` }}
            />
          ))}
          <span
            className="band__marker"
            style={{ left: `${pct(assessment.risk_score)}%` }}
            role="img"
            aria-label={`Score ${assessment.risk_percent.toFixed(1)} percent, ${assessment.risk_tier_label} tier`}
          />
        </div>
        <div className="band__scale">
          <span style={{ left: 0 }}>0%</span>
          {ticks.map((tick) => (
            <span key={tick.key} style={{ left: `${pct(tick.value as number)}%` }}>
              {tick.label}
            </span>
          ))}
          <span style={{ left: "100%" }}>{(ceiling * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
}
