"""
Deterioration label engineering for Stream 1.

Event definition (per stay): the EARLIER of
  (a) in-hospital death, at hour = survival_days * 24 (falls back to
      length_of_stay_days * 24 if survival_days is unusable), or
  (b) a vitals-instability trend: sustained tachycardia (HR > 120 for >= 2
      consecutive hours) OR sustained hypotension (systolic BP < 90 for >= 2
      consecutive hours, from either invasive SysABP or non-invasive NISysABP),
      onset = the first hour of that 2-hour run.

For every (stay_id, hour=t) with no event yet, label_h(t) = 1 if the event
occurs in (t, t+h], else 0, for h in {24, 48, 72}. Rows at/after the event are
dropped (you can't predict something that has already happened, and leaving
them in as implicit negatives would understate the true positive rate).
Stays with no event at all keep all rows, all labels 0.
"""
import numpy as np
import pandas as pd

HORIZONS = {"24h": 24, "48h": 48, "72h": 72}
TACHYCARDIA_HR = 120.0
HYPOTENSION_SBP = 90.0
SUSTAINED_HOURS = 2


def _instability_onset_hour(stay_wide: pd.DataFrame) -> float:
    """stay_wide: rows for one stay, sorted by hour, columns incl. HR,
    HR_missing, SysABP, SysABP_missing, NISysABP, NISysABP_missing."""
    def observed(col):
        return (stay_wide[f"{col}_missing"] == 0) if f"{col}_missing" in stay_wide else pd.Series(False, index=stay_wide.index)

    hr_high = (stay_wide.get("HR", pd.Series(np.nan, index=stay_wide.index)) > TACHYCARDIA_HR) & observed("HR")
    sbp_low = pd.Series(False, index=stay_wide.index)
    for col in ("SysABP", "NISysABP"):
        if col in stay_wide:
            sbp_low = sbp_low | ((stay_wide[col] < HYPOTENSION_SBP) & observed(col))

    unstable = (hr_high | sbp_low).to_numpy()
    for i in range(len(unstable) - SUSTAINED_HOURS + 1):
        if unstable[i:i + SUSTAINED_HOURS].all():
            return float(stay_wide["hour"].iloc[i])
    return np.inf


def compute_event_hours(wide_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame [stay_id, event_hour] (event_hour = inf if none)."""
    instability = (
        wide_df.sort_values(["stay_id", "hour"])
        .groupby("stay_id")
        .apply(_instability_onset_hour, include_groups=False)
        .rename("instability_hour")
    )

    outcomes = outcomes_df.set_index("stay_id")
    death_hour = pd.Series(np.inf, index=instability.index)
    died = outcomes.reindex(instability.index)["in_hospital_death"] == 1
    survival_hours = outcomes.reindex(instability.index)["survival_days"] * 24
    los_hours = outcomes.reindex(instability.index)["length_of_stay_days"] * 24
    usable_survival = survival_hours.where(survival_hours > 0, los_hours)
    death_hour[died] = usable_survival[died]

    event_hour = pd.concat([instability, death_hour.rename("death_hour")], axis=1).min(axis=1)
    return event_hour.rename("event_hour").reset_index()


def add_deterioration_labels(wide_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> pd.DataFrame:
    """Joins event hours onto the per-(stay,hour) wide table, drops post-event
    rows, and adds one binary column per horizon (deteriorate_24h/48h/72h)."""
    event_hours = compute_event_hours(wide_df, outcomes_df)
    df = wide_df.merge(event_hours, on="stay_id", how="left")
    df = df[df["hour"] < df["event_hour"]].copy()

    for name, h in HORIZONS.items():
        will_happen = (df["event_hour"] - df["hour"]) <= h
        df[f"deteriorate_{name}"] = (will_happen & np.isfinite(df["event_hour"])).astype(np.float32)

    return df.drop(columns=["event_hour"])
