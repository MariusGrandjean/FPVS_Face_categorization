# -*- coding: utf-8 -*-
""" Behavioral data analysis – FPVS face categorization experiment

Author: Grandjean Marius
Revised by: Milan Jalocha

This script analyzes the behavioral data collected during an EEG face
categorization experiment using the FPVS method.

The behavioral task is mainly included to keep participants engaged during the
experiment. These data are analyzed to assess the presence or
absence of an attentional bias that could influence the interpretation of the
EEG results.

The task is similar to a go/no-go paradigm: participants are expected to respond
to specific events and withhold their response to others. The data are extracted
from experimental triggers, which indicate either a condition or a participant's
response, as well as from reaction times.

Behavioral responses are classified into four categories:
- Hit: correct response when a response was expected.
- Miss: no response when a response was expected.
- Correct rejection: correct absence of response when no response was expected.
- False alarm: response given when no response was expected.

The analysis mainly relies on:
- d', used as a sensitivity index to quantify behavioral performance;
- a 2 x 2 ANOVA, used to test whether the observed differences are statistically
  significant across experimental conditions.

Expected filename format:  (1|2)_Peripheral_<participant_name>.xls
Example:                    2_Peripheral_Charles0102.xls

"""
#%%
import re
import xlrd
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.stats import f as f_dist

#%%

DATA_FOLDER = Path("C:/Users/milan/OneDrive - UCL/Behavioral_data")
# Filename pattern: starts with 1 or 2, then _Peripheral_, then the participant name
FILE_PATTERN = "[12]_Peripheral_*.xls"

#%%

TOTAL_TRIALS = 28       # 9 for foveal, 28 for peripheral
RED_DURATION = 2        # seconds a red period lasts after Trigger 60 or 61

CONDITION_MAP = {
    "Trigger_226": "Natural-HM",
    "Trigger_228": "Natural-VM",
    "Trigger_230": "Negative-HM",
    "Trigger_232": "Negative-VM",
}

#%%
def process_one_file(excel_path):
    """
    Read one participant's .xls file and return a DataFrame with
    one row per fixation event across all trials.
    """
    stem_parts = excel_path.stem.split("_")
    session_num, participant = stem_parts[0], "_".join(stem_parts[2:])

    print("\n=== Processing " + excel_path.name + "  (participant: " + participant + ", session: " + session_num + ") ===")

    wb = xlrd.open_workbook(str(excel_path), on_demand=True)
    all_trials = []

    for trial_num in range(1, TOTAL_TRIALS + 1):
        sh = wb.sheet_by_name("Trial" + str(trial_num) + "_results")
        results_rows = [sh.row_values(r) for r in range(sh.nrows)]
        wb.unload_sheet("Trial" + str(trial_num) + "_results")

        sh = wb.sheet_by_name("Trial" + str(trial_num) + "_events_all")
        events_rows = [sh.row_values(r) for r in range(sh.nrows)]
        wb.unload_sheet("Trial" + str(trial_num) + "_events_all")

        df_res = pd.DataFrame(results_rows)
        df_ev  = pd.DataFrame(events_rows)

        # Results sheet structure: rows 2+ contain data; columns 0–2 are fixation time, key time, and RT.
        fix_time = pd.to_numeric(df_res.iloc[2:, 0], errors="coerce")
        key_time = pd.to_numeric(df_res.iloc[2:, 1], errors="coerce")
        rt       = pd.to_numeric(df_res.iloc[2:, 2], errors="coerce")

        # Remove trailing empty rows after the last valid fixation time.
        if fix_time.isna().any():
            last_good = fix_time.isna().idxmax() - 1
            fix_time, key_time, rt = fix_time.loc[:last_good], key_time.loc[:last_good], rt.loc[:last_good]

        trial_df = pd.DataFrame({"trial": trial_num, "fix_time": fix_time.values,
                                  "key_time": key_time.values, "rt": rt.values})
        trial_df["pressed"] = trial_df["key_time"] > 0

        ev_data = df_ev.iloc[1:]  # Drop the header row from the events sheet.
        # Red-period onsets are identified from event columns 26 and 27.
        mask_col26 = ev_data.iloc[:, 26].astype(str).str.strip() != ""
        mask_col27 = (ev_data.iloc[:, 27].astype(str).str.strip() != ""
                      if ev_data.shape[1] > 27 else pd.Series(False, index=ev_data.index))

        red_onsets = np.sort(np.concatenate([
            pd.to_numeric(ev_data.loc[mask_col26, ev_data.columns[0]], errors="coerce").dropna().values,
            pd.to_numeric(ev_data.loc[mask_col27, ev_data.columns[0]], errors="coerce").dropna().values,  ]))

        ft = trial_df["fix_time"].to_numpy()
        # Red-period onsets are identified from the event columns containing stop-period triggers.
        if len(red_onsets) == 0:
            trial_df["during_red"] = False
        else:
            in_any_window = (ft[:, np.newaxis] >= red_onsets) & (ft[:, np.newaxis] <= red_onsets + RED_DURATION)
            trial_df["during_red"] = in_any_window.any(axis=1)

        # Extract the condition trigger from the stimulation-start event.
        condition = "Unknown"
        for col in range(1, 5):
            col_text  = ev_data.iloc[:, col].astype(str)
            match_rows = ev_data[col_text.str.contains("Stimulation start", case=False, na=False)]
            if match_rows.empty:
                continue
            m = re.search(r"Trigger (\d+)", str(match_rows.iloc[0, col]), re.IGNORECASE)
            if m:
                condition = CONDITION_MAP.get("Trigger_" + m.group(1), "Trigger_" + m.group(1))
            break

        trial_df["condition"] = condition
        # Green/go event:
        #   Pressed     = Hit
        #   Not pressed = Miss
        # Red/stop event:
        #   Pressed     = False Alarm
        #   Not pressed = Correct Rejection
        green, pressed = ~trial_df["during_red"], trial_df["pressed"]
        trial_df["sdt"] = np.select(
            condlist=[green & pressed, green & ~pressed, ~green & pressed, ~green & ~pressed],
            choicelist=["H", "M", "FA", "CR"], default="CR",
        ).astype(str)

        all_trials.append(trial_df)
        print("   Trial {:02d}  |  {:<12}  |  H={:2d}  M={:2d}  FA={:2d}  CR={:2d}".format(
            trial_num, condition,
            (trial_df["sdt"] == "H").sum(),
            (trial_df["sdt"] == "M").sum(),
            (trial_df["sdt"] == "FA").sum(),
            (trial_df["sdt"] == "CR").sum()))

    wb.release_resources()
    file_df = pd.concat(all_trials, ignore_index=True)
    file_df["participant"], file_df["session"] = participant, session_num
    return file_df

#%%

all_files = sorted(DATA_FOLDER.glob(FILE_PATTERN))

if not all_files:
    raise FileNotFoundError("No files matched " + FILE_PATTERN + " in " + str(DATA_FOLDER))

print("Found " + str(len(all_files)) + " file(s) to process.")
per_participant_dfs = []

for excel_path in all_files:
    try:
        per_participant_dfs.append(process_one_file(excel_path))
    except Exception as exc:
        print("⚠️  Skipped " + excel_path.name + " — " + str(exc))

# Combine all participants into one master table
if not per_participant_dfs:
    raise RuntimeError("No participant data could be processed. Check skipped files above.")

master = pd.concat(per_participant_dfs, ignore_index=True)[[
    "participant", "session", "trial", "condition",
    "fix_time", "key_time", "rt", "pressed", "during_red", "sdt",  ]]

master_display = master.rename(columns={
    "participant": "Participant", "session": "Session", "trial": "Trial",
    "condition": "Condition", "fix_time": "Fixation onset time (s)",
    "key_time": "Key press time (s)", "rt": "Reaction time (s)",
    "pressed": "Key pressed", "during_red": "During red stop period",
    "sdt": "SDT response type",  })

print("\n\n" + "═" * 120)
print("  TABLE 1 · Event-level master table — first 10 rows\n  One row = one fixation event")
print("═" * 120)
print(master_display.head(10).to_string(index=False))
print("═" * 120)

#%%
# Summary table per participant × condition
# Create binary indicators so each SDT response type can be counted by summing.
for sdt_code in ["H", "M", "FA", "CR"]:
    master["is_" + sdt_code] = (master["sdt"] == sdt_code).astype(int)
# Reaction times are averaged only for hits.
master["rt_on_hit"] = master["rt"].where(master["sdt"] == "H")

participant_summary = master.groupby(["participant", "condition"]).agg(
    n_fixation_events=("sdt", "size"),
    Hits=("is_H", "sum"), Misses=("is_M", "sum"),
    False_Alarms=("is_FA", "sum"), Correct_Rej=("is_CR", "sum"),
    mean_RT_hits=("rt_on_hit", "mean"),  )

n_green = participant_summary["Hits"] + participant_summary["Misses"]
n_red   = participant_summary["False_Alarms"] + participant_summary["Correct_Rej"]

participant_summary["hit_rate_%"]  = (100 * participant_summary["Hits"] / n_green.clip(lower=1)).round(1)
participant_summary["FA_rate_%"]   = (100 * participant_summary["False_Alarms"] / n_red.clip(lower=1)).round(1)
participant_summary["mean_RT_hits"] = participant_summary["mean_RT_hits"].round(3)

# D-PRIME (d')
# d' = z(Hit Rate) − z(False Alarm Rate)
# Edge-case correction:
# corrected_HR = (Hits + 0.5) / (n_green + 1)
# corrected_FAR = (FA + 0.5) / (n_red + 1)
hr_corrected  = (participant_summary["Hits"] + 0.5) / (n_green + 1)
far_corrected = (participant_summary["False_Alarms"] + 0.5) / (n_red + 1)
participant_summary["d_prime"] = (norm.ppf(hr_corrected) - norm.ppf(far_corrected)).round(3)

ps_display = participant_summary.reset_index()[[
    "participant", "condition", "n_fixation_events", "Hits", "Misses",
    "False_Alarms", "Correct_Rej", "hit_rate_%", "FA_rate_%", "mean_RT_hits", "d_prime",
]].rename(columns={
    "participant": "Participant", "condition": "Condition",
    "n_fixation_events": "N fixation events", "False_Alarms": "False alarms",
    "Correct_Rej": "Correct rejections", "hit_rate_%": "Hit rate (%)",
    "FA_rate_%": "False alarm rate (%)", "mean_RT_hits": "Mean RT for hits (s)",
    "d_prime": "d-prime",
}).sort_values(["Participant", "Condition"])

print("\n\n" + "═" * 130)
print("  TABLE 2 · Per-participant SDT results by condition\n  Main behavioural outcomes first, followed by d'")
print("═" * 130)
print("  Participant      Condition      N events   Hits   Misses   False alarms   Correct rej.   Hit rate (%)   FA rate (%)   Mean RT hits (s)        d′")
print("─" * 130)

prev_pp = None
for _, row in ps_display.iterrows():
    if prev_pp and row["Participant"] != prev_pp:
        print("─" * 130)
    print("  {:<16} {:<14} {:>8} {:>6} {:>8} {:>14} {:>14} {:>13.1f} {:>12.1f} {:>17.3f} {:>7.3f}".format(
        row["Participant"], row["Condition"], int(row["N fixation events"]),
        int(row["Hits"]), int(row["Misses"]), int(row["False alarms"]),
        int(row["Correct rejections"]), row["Hit rate (%)"],
        row["False alarm rate (%)"], row["Mean RT for hits (s)"], row["d-prime"]))
    prev_pp = row["Participant"]

print("═" * 130)

dprime_display = ps_display[[
    "Participant", "Condition", "Hits", "Misses",
    "False alarms", "Correct rejections", "Hit rate (%)", "False alarm rate (%)", "d-prime",
]]

print("\n\n" + "═" * 115)
print("  TABLE 3 · Data used to compute d'\n  d' is computed from hit rate and false alarm rate")
print("═" * 115)
print(dprime_display.to_string(index=False))
print("═" * 115)

# Group-level summary
# Average participant-level rates instead of pooling all events,
# so each participant has equal weight.
gs = participant_summary.groupby("condition")[
    ["hit_rate_%", "FA_rate_%", "mean_RT_hits", "d_prime"]
].agg(["mean", "std", "count"]).round(2)
gs.columns = [m + "_" + s for m, s in gs.columns]
gs = gs.reset_index()

print("\n\n" + "═" * 105)
print("  TABLE 4 · Group-level summary\n  Mean ± SD across participants, with N participants per condition")
print("═" * 105)
print("  Condition          Hit rate (%)         FA rate (%)       Mean RT hits (s)                d′       N")
print("─" * 105)

for _, row in gs.iterrows():
    n = int(row["hit_rate_%_count"])
    print("  {:<14} {:>7.1f} ± {:<6.1f} {:>7.1f} ± {:<6.1f} {:>9.3f} ± {:<7.3f} {:>6.3f} ± {:<6.3f} {:>5}".format(
        row["condition"],
        row["hit_rate_%_mean"], row["hit_rate_%_std"],
        row["FA_rate_%_mean"], row["FA_rate_%_std"],
        row["mean_RT_hits_mean"], row["mean_RT_hits_std"],
        row["d_prime_mean"], row["d_prime_std"], n))

print("═" * 105)

# 2 × 2 Repeated-Measures ANOVA on d'
# Design:
#   Factor 1 – Contrast polarity : Natural vs Negative
#   Factor 2 – Meridian          : HM vs VM
#   Interaction                  : Contrast polarity × Meridian
anova_df = participant_summary[["d_prime"]].reset_index()
anova_df["Contrast polarity"] = anova_df["condition"].str.split("-").str[0]  # Natural / Negative
anova_df["Meridian"]          = anova_df["condition"].str.split("-").str[1]  # HM / VM

wide = anova_df.pivot(index="participant", columns="condition", values="d_prime")

try:
    Y = wide[["Natural-HM", "Natural-VM", "Negative-HM", "Negative-VM"]].values.astype(float)
except KeyError as e:
    raise KeyError("Condition missing from data: " + str(e) + ". Check CONDITION_MAP matches your trigger numbers.") from e

n = Y.shape[0]
if n < 2:
    raise RuntimeError("The repeated-measures ANOVA requires at least 2 participants.")

# Factor A = Contrast polarity
#   Levels: Natural = cols 0–1, Negative = cols 2–3
# Factor B = Meridian
#   Levels: HM = cols 0 and 2, VM = cols 1 and 3
grand_mean = Y.mean()

A_means = np.stack([Y[:, [0, 1]].mean(axis=1),  # Natural mean per participant
                    Y[:, [2, 3]].mean(axis=1)],  # Negative mean per participant
                   axis=1)

B_means = np.stack([Y[:, [0, 2]].mean(axis=1),  # HM mean per participant
                    Y[:, [1, 3]].mean(axis=1)],  # VM mean per participant
                   axis=1)

# SS for Factor A: Contrast polarity
A_grand    = A_means.mean(axis=0)
SS_A       = n * 2 * np.sum((A_grand - grand_mean) ** 2)
df_A       = 1
MS_A       = SS_A / df_A
A_subj_dev = A_means - A_means.mean(axis=1, keepdims=True)
SS_errA    = 2 * np.sum((A_subj_dev - (A_grand - grand_mean)) ** 2)
df_errA    = (n - 1) * df_A
MS_errA    = SS_errA / df_errA

# SS for Factor B: Meridian
B_grand    = B_means.mean(axis=0)
SS_B       = n * 2 * np.sum((B_grand - grand_mean) ** 2)
df_B       = 1
MS_B       = SS_B / df_B
B_subj_dev = B_means - B_means.mean(axis=1, keepdims=True)
SS_errB    = 2 * np.sum((B_subj_dev - (B_grand - grand_mean)) ** 2)
df_errB    = (n - 1) * df_B
MS_errB    = SS_errB / df_errB

# SS for Interaction A × B
# Interaction contrast:
#   (Natural-HM - Natural-VM) - (Negative-HM - Negative-VM)
AB_contrast = (Y[:, 0] - Y[:, 1]) - (Y[:, 2] - Y[:, 3])
SS_AB       = n * (AB_contrast.mean() ** 2) / 2
df_AB       = 1
MS_AB       = SS_AB / df_AB
SS_errAB    = np.sum((AB_contrast - AB_contrast.mean()) ** 2) / 4
df_errAB    = (n - 1) * df_AB
MS_errAB    = SS_errAB / df_errAB

F_A, F_B, F_AB       = MS_A / MS_errA, MS_B / MS_errB, MS_AB / MS_errAB
p_A, p_B, p_AB       = (1 - f_dist.cdf(F, df, dfe) for F, df, dfe in
                         [(F_A, df_A, df_errA), (F_B, df_B, df_errB), (F_AB, df_AB, df_errAB)])
eta2_A, eta2_B, eta2_AB = (SS / (SS + SSe) for SS, SSe in
                            [(SS_A, SS_errA), (SS_B, SS_errB), (SS_AB, SS_errAB)])

def sig_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

print("\n\n" + "═" * 85)
print("  TABLE 5 · 2 × 2 Repeated-Measures ANOVA on d'\n  Factors: Contrast polarity (Natural/Negative) × Meridian (HM/VM), N = " + str(n))
print("═" * 85)
print("  Effect                                F(1,{})            p     partial η²    sig".format(n - 1))
print("─" * 85)

for label, F, p, eta2 in [
    ("Contrast polarity", F_A, p_A, eta2_A),
    ("Meridian", F_B, p_B, eta2_B),
    ("Contrast polarity × Meridian", F_AB, p_AB, eta2_AB),  ]:
    p_str = "< .001" if p < 0.001 else "= {:.3f}".format(p)
    print("  {:<34} {:>12.3f} p {:<9} {:>14.3f} {:>6}".format(label, F, p_str, eta2, sig_stars(p)))

print("═" * 85)
print("  sig: *** p < .001   ** p < .01   * p < .05   ns p ≥ .05\n  partial η²: effect size; small ≥ .01, medium ≥ .06, large ≥ .14")

anova_results = pd.DataFrame({
    "Effect": ["Contrast polarity", "Meridian", "Contrast polarity × Meridian"],
    "df_effect": [df_A, df_B, df_AB],
    "df_error":  [df_errA, df_errB, df_errAB],
    "F":                  np.round([F_A, F_B, F_AB], 3),
    "p":                  np.round([p_A, p_B, p_AB], 4),
    "partial_eta_squared": np.round([eta2_A, eta2_B, eta2_AB], 3),
    "significance": [sig_stars(p_A), sig_stars(p_B), sig_stars(p_AB)], })
# Uncomment to export all results to a single Excel file with one sheet per table.
# with pd.ExcelWriter(DATA_FOLDER / "fpvs_behavioral_results.xlsx", engine="openpyxl") as writer:
# ps_display.to_excel(writer, sheet_name="SDT by participant-condition", index=False)
# dprime_display.to_excel(writer, sheet_name="d-prime inputs", index=False)
# anova_results.to_excel(writer, sheet_name="ANOVA on d-prime", index=False)