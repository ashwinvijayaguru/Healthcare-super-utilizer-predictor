import type { RedFlag } from "../lib/types";

const SEVERITY_LABEL: Record<string, string> = {
  critical: "Act now",
  warning: "Review",
  advisory: "Note",
};

/**
 * Rule-based, evaluated independently of the model score. A low score never
 * suppresses a flag: the model can drift or simply be wrong, and a patient on
 * an anticoagulant with a 0.4 adherence ratio still needs a phone call.
 */
export default function RedFlags({ flags }: { flags: RedFlag[] }) {
  const critical = flags.filter((f) => f.severity === "critical").length;

  return (
    <div className="panel">
      <div className="panel__head">
        <h2>Clinical red flags</h2>
        <span>
          {flags.length === 0
            ? "none raised"
            : `${flags.length} raised${critical ? `, ${critical} needing action now` : ""}`}
        </span>
      </div>
      <div className="panel__body">
        {flags.length === 0 ? (
          <p className="summary-text">
            No safety rules were triggered by this record. Flags are checked separately from the
            score, so this is not simply a consequence of a low result.
          </p>
        ) : (
          flags.map((flag) => (
            <div className="flag" key={flag.code} data-severity={flag.severity}>
              <div className="flag__rail" />
              <div>
                <div className="severity">{SEVERITY_LABEL[flag.severity]}</div>
                <h3>{flag.title}</h3>
                <p>{flag.detail}</p>
                <p className="flag__action">{flag.suggested_action}</p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
