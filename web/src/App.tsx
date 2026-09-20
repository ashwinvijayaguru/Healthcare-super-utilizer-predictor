import { useCallback, useEffect, useMemo, useState } from "react";
import IntakeForm from "./components/IntakeForm";
import RiskBand from "./components/RiskBand";
import DotMatrix from "./components/DotMatrix";
import Attribution from "./components/Attribution";
import RedFlags from "./components/RedFlags";
import PatientLetter from "./components/PatientLetter";
import {
  DEFAULT_INTAKE,
  PRESETS,
  getFormSchema,
  getModelInfo,
  scorePatient,
} from "./lib/api";
import type { Assessment, FormSchema, Intake, ModelInfo } from "./lib/types";

export default function App() {
  const [schema, setSchema] = useState<FormSchema | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [intake, setIntake] = useState<Intake>({ ...DEFAULT_INTAKE });
  const [preset, setPreset] = useState<string | null>(null);
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [scoring, setScoring] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getFormSchema(), getModelInfo()])
      .then(([s, m]) => {
        setSchema(s);
        setModel(m);
      })
      .catch((e: Error) => setBootError(e.message));
  }, []);

  const update = useCallback((name: string, value: number) => {
    setIntake((current) => ({ ...current, [name]: value }));
    setPreset(null);
  }, []);

  const applyPreset = (id: string) => {
    const found = PRESETS.find((p) => p.id === id);
    if (!found) return;
    setIntake({ ...found.intake });
    setPreset(id);
    setAssessment(null);
    setError(null);
  };

  const reset = () => {
    setIntake({ ...DEFAULT_INTAKE });
    setPreset(null);
    setAssessment(null);
    setError(null);
  };

  const run = async () => {
    setScoring(true);
    setError(null);
    try {
      const ref = `DEMO-${Date.now().toString(36).toUpperCase()}`;
      setAssessment(await scorePatient(ref, intake));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setScoring(false);
    }
  };

  const trained = useMemo(
    () => (model ? new Date(model.trained_at).toLocaleDateString() : null),
    [model],
  );

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1>Care Risk Console</h1>
          <p>
            Find members likely to need urgent hospital care in the next 6–12 months, and
            explain why in language they can read.
          </p>
        </div>
        <div className="masthead-meta">
          <span>
            <span className="status-dot" data-state={bootError ? "down" : "up"} />
            {bootError ? "API unreachable" : "Connected"}
          </span>
          {model && (
            <span>
              {model.model_name} v{model.model_version} · trained {trained}
            </span>
          )}
        </div>
      </header>

      <div className="workspace">
        <div className="column">
          <div className="panel">
            <div className="panel__head">
              <h2>Start from an example</h2>
            </div>
            <div className="panel__body">
              <div className="presets">
                {PRESETS.map((p) => (
                  <button
                    key={p.id}
                    className="preset"
                    aria-pressed={preset === p.id}
                    onClick={() => applyPreset(p.id)}
                  >
                    <strong>{p.name}</strong>
                    <span>{p.note}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {schema && (
            <IntakeForm groups={schema.groups} values={intake} onChange={update} />
          )}

          <div className="actions">
            <button className="button" onClick={run} disabled={scoring || !schema}>
              {scoring ? "Scoring…" : "Score this member"}
            </button>
            <button className="button button--quiet" onClick={reset}>
              Clear the form
            </button>
          </div>
        </div>

        <div className="column column--results">
          {bootError && (
            <div className="notice">
              <strong>The API is not responding.</strong>
              {bootError}. Start it with <code>uvicorn api.main:app --reload</code> and reload
              this page.
            </div>
          )}

          {error && !bootError && (
            <div className="notice">
              <strong>That record could not be scored.</strong>
              {error}
            </div>
          )}

          {!assessment && !bootError && (
            <div className="empty">
              <h2>No member scored yet</h2>
              <p>
                Pick an example or fill in the record on the left, then score it. You will get a
                risk band, the factors behind it, any clinical red flags, and a summary written
                for the patient.
              </p>
              {schema && <p>{schema.outcome_definition}</p>}
            </div>
          )}

          {assessment && (
            <>
              <div className="panel">
                <div className="panel__head">
                  <h2>Risk of becoming a high utilizer</h2>
                  <span>scored in {assessment.latency_ms} ms</span>
                </div>
                <div className="panel__body">
                  <RiskBand assessment={assessment} model={model} />
                  <DotMatrix filled={assessment.expected_in_100} />
                </div>
              </div>

              <div className="panel">
                <div className="panel__head">
                  <h2>Summary for the care team</h2>
                </div>
                <div className="panel__body">
                  <p className="summary-text">{assessment.clinician_summary}</p>
                </div>
              </div>

              <Attribution assessment={assessment} />
              <RedFlags flags={assessment.red_flags} />

              <div className="panel">
                <div className="panel__head">
                  <h2>Summary for the patient</h2>
                  <span>
                    {assessment.patient_summary.source === "llm"
                      ? "language model"
                      : "template"}
                  </span>
                </div>
                <div className="panel__body">
                  <PatientLetter summary={assessment.patient_summary} />
                </div>
              </div>

              <p className="disclaimer">{assessment.disclaimer}</p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
