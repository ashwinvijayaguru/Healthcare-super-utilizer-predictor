import type { Assessment } from "../lib/types";

const GROUP_COLOR: Record<string, string> = {
  demographics: "#7d8b97",
  utilization: "#b4562f",
  clinical: "#8a2b2b",
  medications: "#c08a2e",
  sdoh: "#2b5ca8",
};

const GROUP_LABEL: Record<string, string> = {
  demographics: "Demographics",
  utilization: "Prior use",
  clinical: "Conditions",
  medications: "Medications",
  sdoh: "Social drivers",
};

/**
 * Bar length is |Shapley value|, scaled against the largest contribution on
 * screen. These are the model's actual attributions, not a narrative written
 * after the fact to sound plausible.
 */
export default function Attribution({ assessment }: { assessment: Assessment }) {
  const all = [...assessment.drivers, ...assessment.protective_factors];
  const scale = Math.max(...all.map((d) => Math.abs(d.contribution)), 0.001);

  const split = Object.entries(assessment.group_attribution)
    .filter(([, share]) => share > 0.001)
    .sort((a, b) => b[1] - a[1]);

  return (
    <>
      <div className="panel">
        <div className="panel__head">
          <h2>Where the risk comes from</h2>
          <span>share of the increase</span>
        </div>
        <div className="panel__body">
          <div className="split">
            {split.map(([group, share]) => (
              <i
                key={group}
                style={{ width: `${share * 100}%`, background: GROUP_COLOR[group] }}
                title={`${GROUP_LABEL[group]}: ${(share * 100).toFixed(0)}%`}
              />
            ))}
          </div>
          <div className="split-legend">
            {split.map(([group, share]) => (
              <span key={group}>
                <b style={{ background: GROUP_COLOR[group] }} />
                {GROUP_LABEL[group]} {(share * 100).toFixed(0)}%
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel__head">
          <h2>What is pushing the score up</h2>
        </div>
        <div className="panel__body">
          {assessment.drivers.length === 0 ? (
            <p className="summary-text">Nothing in this record raises the score materially.</p>
          ) : (
            assessment.drivers.map((d) => (
              <div className="driver" key={d.feature}>
                <div>
                  <div className="driver__label">{d.state}</div>
                  <div className="driver__group">{d.group_label}</div>
                </div>
                <div className="driver__bar">
                  <i style={{ width: `${(Math.abs(d.contribution) / scale) * 100}%` }} />
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {assessment.protective_factors.length > 0 && (
        <div className="panel">
          <div className="panel__head">
            <h2>Working in their favour</h2>
          </div>
          <div className="panel__body">
            {assessment.protective_factors.map((d) => (
              <div className="driver driver--protective" key={d.feature}>
                <div>
                  <div className="driver__label">{d.state}</div>
                  <div className="driver__group">{d.group_label}</div>
                </div>
                <div className="driver__bar">
                  <i style={{ width: `${(Math.abs(d.contribution) / scale) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
