import type { PatientSummary } from "../lib/types";

/**
 * The patient-facing voice. Set in a serif on warmer stock so it reads as a
 * letter rather than a chart entry, and so nobody mistakes which half of the
 * screen is safe to hand to the person it is about.
 */
export default function PatientLetter({ summary }: { summary: PatientSummary }) {
  return (
    <div className="letter">
      <h2>{summary.headline}</h2>

      {summary.body.split("\n\n").map((para, i) => (
        <p key={i}>{para}</p>
      ))}

      {summary.what_this_means.length > 0 && (
        <div className="letter__section">
          <h3>What this means</h3>
          <ul>
            {summary.what_this_means.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      {summary.next_steps.length > 0 && (
        <div className="letter__section">
          <h3>What happens next</h3>
          <ul>
            {summary.next_steps.map((step, i) => (
              <li key={i}>
                {step.text}
                <span className="owner">
                  {step.owner === "patient" ? "you" : "your care team"}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="letter__close">{summary.reassurance}</p>

      <div className="provenance">
        {summary.source === "llm"
          ? "Written by a language model from the factors listed above, then checked automatically. Your care team reviews it before sharing."
          : "Assembled from the factors listed above. Your care team reviews it before sharing."}
      </div>
    </div>
  );
}
