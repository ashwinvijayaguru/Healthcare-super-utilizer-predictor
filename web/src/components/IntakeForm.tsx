import { useState } from "react";
import type { FormGroup, Intake } from "../lib/types";

/**
 * Fields are rendered from the schema the API serves at /api/v1/form-schema,
 * so adding a feature to the model does not require a frontend release. Only
 * the presentation hints below live here.
 */
const STEP: Record<string, number> = { medication_adherence_pdc: 0.05 };

const HELP: Record<string, string> = {
  days_since_last_discharge: "999 if never admitted",
  medication_adherence_pdc: "Proportion of days covered, 0–1",
  area_deprivation_index: "National percentile, 100 = most deprived",
  charlson_index: "Weighted comorbidity score",
  high_risk_med_count: "Anticoagulants, insulin, antiarrhythmics",
};

export default function IntakeForm({
  groups,
  values,
  onChange,
}: {
  groups: FormGroup[];
  values: Intake;
  onChange: (name: string, value: number) => void;
}) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const activeCount = (group: FormGroup) =>
    group.fields.filter((f) => {
      const v = values[f.name];
      if (f.kind === "binary") return f.higher_is_protective ? v === 0 : v === 1;
      return false;
    }).length;

  return (
    <div className="panel">
      <div className="panel__head">
        <h2>Member record</h2>
        <span>de-identified input only</span>
      </div>

      {groups.map((group) => {
        const isCollapsed = collapsed[group.key];
        const flagged = activeCount(group);
        return (
          <div className="fieldgroup" key={group.key}>
            <button
              className="fieldgroup__title"
              aria-expanded={!isCollapsed}
              onClick={() => setCollapsed((c) => ({ ...c, [group.key]: !c[group.key] }))}
            >
              {group.label}
              <span>
                {flagged > 0 ? `${flagged} noted` : "\u2014"} {isCollapsed ? "+" : "\u2212"}
              </span>
            </button>

            {!isCollapsed && (
              <div className="fieldgroup__fields">
                {group.fields.map((field) => {
                  const value = values[field.name] ?? 0;
                  const id = `f-${field.name}`;

                  if (field.kind === "binary") {
                    return (
                      <div className="field" key={field.name}>
                        <label htmlFor={id}>
                          {field.label}
                          {HELP[field.name] && <div className="unit">{HELP[field.name]}</div>}
                        </label>
                        <button
                          id={id}
                          className="toggle"
                          role="switch"
                          aria-checked={value === 1}
                          aria-label={field.label}
                          onClick={() => onChange(field.name, value === 1 ? 0 : 1)}
                        />
                      </div>
                    );
                  }

                  return (
                    <div className="field" key={field.name}>
                      <label htmlFor={id}>
                        {field.label}
                        {HELP[field.name] && <div className="unit">{HELP[field.name]}</div>}
                      </label>
                      <input
                        id={id}
                        type="number"
                        value={value}
                        min={field.min ?? undefined}
                        max={field.max ?? undefined}
                        step={STEP[field.name] ?? 1}
                        onChange={(e) => {
                          const next = e.target.value === "" ? 0 : Number(e.target.value);
                          onChange(field.name, Number.isNaN(next) ? 0 : next);
                        }}
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
