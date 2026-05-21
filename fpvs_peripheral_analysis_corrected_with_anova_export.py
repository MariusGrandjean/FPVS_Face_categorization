# -*- coding: utf-8 -*-
"""
FPVS Peripheral Task – Multi-participant Analysis
==================================================
For each participant file in a folder, this script builds a table with one
row per fixation event, then combines everything into a master table and
prints participant-level, group-level, and ANOVA summaries.

Expected filename format:  (1|2)_Peripheral_<participant_name>.xls
Example:                    2_Peripheral_Charles.xls

Author: grandjeamari
Modified for clearer table headers and ANOVA factor naming.
"""

import re                    # for extracting trigger numbers and parsing filenames
import xlrd                  # for reading the legacy .xls format, not .xlsx
import numpy as np           # for fast numeric array operations
import pandas as pd          # for tabular data, DataFrames
from pathlib import Path     # cleaner than os.path for handling file paths
from scipy.stats import norm # for norm.ppf, used to compute d'
from scipy.stats import f as f_dist   # for p-values from F distribution


# ─────────────────────────────────────────────────────────────
# SETTINGS  ← only edit this section
# ─────────────────────────────────────────────────────────────

# Folder that contains ALL participant .xls files
DATA_FOLDER = Path("C:/Users/milan/OneDrive - UCL/Behavioral_data")

# Filename pattern: starts with 1 or 2, then _Peripheral_, then the participant name
FILE_PATTERN = "[12]_Peripheral_*.xls"

TOTAL_TRIALS = 28       # 9 for foveal, 28 for peripheral
RED_DURATION = 2        # seconds a red period lasts after Trigger 60 or 61

# Mapping from raw trigger strings to readable condition names.
CONDITION_MAP = {
    "Trigger_226": "Natural-HM",
    "Trigger_228": "Natural-VM",
    "Trigger_230": "Negative-HM",
    "Trigger_232": "Negative-VM",
}


# ─────────────────────────────────────────────────────────────
# PER-FILE PROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────

def process_one_file(excel_path: Path) -> pd.DataFrame:
    """
    Read one participant's .xls file and return a DataFrame with
    one row per fixation event across all trials.
    """

    # Extract participant and session from filename.
    stem_parts = excel_path.stem.split("_")
    session_num = stem_parts[0]
    participant = "_".join(stem_parts[2:])

    print(
        f"\n=== Processing {excel_path.name}  "
        f"(participant: {participant}, session: {session_num}) ==="
    )

    wb = xlrd.open_workbook(str(excel_path), on_demand=True)
    all_trials = []

    for trial_num in range(1, TOTAL_TRIALS + 1):

        # ── Read results sheet ───────────────────────────────
        sh = wb.sheet_by_name(f"Trial{trial_num}_results")
        results_rows = [sh.row_values(r) for r in range(sh.nrows)]
        wb.unload_sheet(f"Trial{trial_num}_results")

        # ── Read events sheet ────────────────────────────────
        sh = wb.sheet_by_name(f"Trial{trial_num}_events_all")
        events_rows = [sh.row_values(r) for r in range(sh.nrows)]
        wb.unload_sheet(f"Trial{trial_num}_events_all")

        df_res = pd.DataFrame(results_rows)
        df_ev = pd.DataFrame(events_rows)

        # ── Parse results sheet ──────────────────────────────
        # Row 0 = sheet title, Row 1 = column headers, Rows 2+ = data
        # Col 0 = fixation change time, Col 1 = key press time, Col 2 = RT
        fix_time = pd.to_numeric(df_res.iloc[2:, 0], errors="coerce")
        key_time = pd.to_numeric(df_res.iloc[2:, 1], errors="coerce")
        rt = pd.to_numeric(df_res.iloc[2:, 2], errors="coerce")

        # Stop at the first blank row in fixation times.
        if fix_time.isna().any():
            last_good = fix_time.isna().idxmax() - 1
            fix_time = fix_time.loc[:last_good]
            key_time = key_time.loc[:last_good]
            rt = rt.loc[:last_good]

        trial_df = pd.DataFrame({
            "trial": trial_num,
            "fix_time": fix_time.values,
            "key_time": key_time.values,
            "rt": rt.values,
        })

        # Pressed = True whenever the participant pressed a key.
        trial_df["pressed"] = trial_df["key_time"] > 0

        # ── Find red-period onsets from the events sheet ─────
        ev_data = df_ev.iloc[1:]  # drop header row

        mask_col26 = ev_data.iloc[:, 26].astype(str).str.strip() != ""

        if ev_data.shape[1] > 27:
            mask_col27 = ev_data.iloc[:, 27].astype(str).str.strip() != ""
        else:
            mask_col27 = pd.Series(False, index=ev_data.index)

        red_times_26 = pd.to_numeric(
            ev_data.loc[mask_col26, ev_data.columns[0]],
            errors="coerce"
        ).dropna().values

        red_times_27 = pd.to_numeric(
            ev_data.loc[mask_col27, ev_data.columns[0]],
            errors="coerce"
        ).dropna().values

        red_onsets = np.sort(np.concatenate([red_times_26, red_times_27]))

        # ── Label each fixation: was the centre cross red? ───
        ft = trial_df["fix_time"].to_numpy()

        if len(red_onsets) == 0:
            trial_df["during_red"] = False
        else:
            in_any_window = (
                (ft[:, np.newaxis] >= red_onsets) &
                (ft[:, np.newaxis] <= red_onsets + RED_DURATION)
            )
            trial_df["during_red"] = in_any_window.any(axis=1)

        # ── Extract condition label ──────────────────────────
        condition = "Unknown"

        for col in range(1, 5):
            col_text = ev_data.iloc[:, col].astype(str)
            match_rows = ev_data[col_text.str.contains(
                "Stimulation start",
                case=False,
                na=False
            )]

            if match_rows.empty:
                continue

            m = re.search(
                r"Trigger (\d+)",
                str(match_rows.iloc[0, col]),
                re.IGNORECASE
            )

            if m:
                raw = f"Trigger_{m.group(1)}"
                condition = CONDITION_MAP.get(raw, raw)

            break

        trial_df["condition"] = condition

        # ── SDT classification ───────────────────────────────
        # Green/go event:
        #   Pressed     = Hit
        #   Not pressed = Miss
        # Red/stop event:
        #   Pressed     = False Alarm
        #   Not pressed = Correct Rejection
        green = ~trial_df["during_red"]
        pressed = trial_df["pressed"]

        trial_df["sdt"] = np.select(
            condlist=[
                green & pressed,
                green & ~pressed,
                ~green & pressed,
                ~green & ~pressed,
            ],
            choicelist=["H", "M", "FA", "CR"],
            default="CR",
        ).astype(str)

        all_trials.append(trial_df)

        print(
            f"   Trial {trial_num:02d}  |  {condition:<12}  |  "
            f"H={(trial_df['sdt'] == 'H').sum():2d}  "
            f"M={(trial_df['sdt'] == 'M').sum():2d}  "
            f"FA={(trial_df['sdt'] == 'FA').sum():2d}  "
            f"CR={(trial_df['sdt'] == 'CR').sum():2d}"
        )

    wb.release_resources()

    file_df = pd.concat(all_trials, ignore_index=True)
    file_df["participant"] = participant
    file_df["session"] = session_num

    return file_df


# ─────────────────────────────────────────────────────────────
# MAIN LOOP – iterate over every matching file in DATA_FOLDER
# ─────────────────────────────────────────────────────────────

all_files = sorted(DATA_FOLDER.glob(FILE_PATTERN))

if not all_files:
    raise FileNotFoundError(f"No files matched {FILE_PATTERN} in {DATA_FOLDER}")

print(f"Found {len(all_files)} file(s) to process.")

per_participant_dfs = []

for excel_path in all_files:
    try:
        per_participant_dfs.append(process_one_file(excel_path))
    except Exception as exc:
        print(f"⚠️  Skipped {excel_path.name} — {exc}")


# ─────────────────────────────────────────────────────────────
# STEP 3 – Stack all participants into one master table
# ─────────────────────────────────────────────────────────────

if not per_participant_dfs:
    raise RuntimeError("No participant data could be processed. Check skipped files above.")

master = pd.concat(per_participant_dfs, ignore_index=True)

# Reorder columns so the most useful information appears first.
# This is the raw/event-level table: one row = one fixation event.
master = master[
    [
        "participant",
        "session",
        "trial",
        "condition",
        "fix_time",
        "key_time",
        "rt",
        "pressed",
        "during_red",
        "sdt",
    ]
]

# Readable version for display/export.
master_display = master.rename(columns={
    "participant": "Participant",
    "session": "Session",
    "trial": "Trial",
    "condition": "Condition",
    "fix_time": "Fixation onset time (s)",
    "key_time": "Key press time (s)",
    "rt": "Reaction time (s)",
    "pressed": "Key pressed",
    "during_red": "During red stop period",
    "sdt": "SDT response type",
})

print("\n\n" + "═" * 120)
print("  TABLE 1 · Event-level master table — first 10 rows")
print("  One row = one fixation event")
print("═" * 120)
print(master_display.head(10).to_string(index=False))
print("═" * 120)


# ─────────────────────────────────────────────────────────────
# STEP 4 – Summary table per participant × condition
# ─────────────────────────────────────────────────────────────

# Add helper 0/1 columns so we can sum them to count each SDT category.
master["is_H"] = (master["sdt"] == "H").astype(int)
master["is_M"] = (master["sdt"] == "M").astype(int)
master["is_FA"] = (master["sdt"] == "FA").astype(int)
master["is_CR"] = (master["sdt"] == "CR").astype(int)

# For RT, we only want values on Hit rows.
master["rt_on_hit"] = master["rt"].where(master["sdt"] == "H")

participant_summary = master.groupby(["participant", "condition"]).agg(
    n_fixation_events=("sdt", "size"),
    Hits=("is_H", "sum"),
    Misses=("is_M", "sum"),
    False_Alarms=("is_FA", "sum"),
    Correct_Rej=("is_CR", "sum"),
    mean_RT_hits=("rt_on_hit", "mean"),
)

# Compute hit rate and false alarm rate.
n_green = participant_summary["Hits"] + participant_summary["Misses"]
n_red = participant_summary["False_Alarms"] + participant_summary["Correct_Rej"]

participant_summary["hit_rate_%"] = (
    100 * participant_summary["Hits"] / n_green.clip(lower=1)
).round(1)

participant_summary["FA_rate_%"] = (
    100 * participant_summary["False_Alarms"] / n_red.clip(lower=1)
).round(1)

participant_summary["mean_RT_hits"] = participant_summary["mean_RT_hits"].round(3)


# ── D-PRIME (d') ──────────────────────────────────────────────────────────────
# d' = z(Hit Rate) − z(False Alarm Rate)
#
# Edge-case correction:
# corrected_HR  = (Hits + 0.5) / (n_green + 1)
# corrected_FAR = (FA   + 0.5) / (n_red   + 1)

hr_corrected = (
    participant_summary["Hits"] + 0.5
) / (n_green + 1)

far_corrected = (
    participant_summary["False_Alarms"] + 0.5
) / (n_red + 1)

participant_summary["d_prime"] = (
    norm.ppf(hr_corrected) - norm.ppf(far_corrected)
).round(3)


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT 2 — Per-participant SDT results by condition
# ═════════════════════════════════════════════════════════════════════════════

ps_display = participant_summary.reset_index()[[
    "participant",
    "condition",
    "n_fixation_events",
    "Hits",
    "Misses",
    "False_Alarms",
    "Correct_Rej",
    "hit_rate_%",
    "FA_rate_%",
    "mean_RT_hits",
    "d_prime",
]].rename(columns={
    "participant": "Participant",
    "condition": "Condition",
    "n_fixation_events": "N fixation events",
    "False_Alarms": "False alarms",
    "Correct_Rej": "Correct rejections",
    "hit_rate_%": "Hit rate (%)",
    "FA_rate_%": "False alarm rate (%)",
    "mean_RT_hits": "Mean RT for hits (s)",
    "d_prime": "d'",
}).sort_values(["Participant", "Condition"])


print("\n\n" + "═" * 130)
print("  TABLE 2 · Per-participant SDT results by condition")
print("  Main behavioural outcomes first, followed by d'")
print("═" * 130)

print(
    f"  {'Participant':<16} "
    f"{'Condition':<14} "
    f"{'N events':>8} "
    f"{'Hits':>6} "
    f"{'Misses':>8} "
    f"{'False alarms':>14} "
    f"{'Correct rej.':>14} "
    f"{'Hit rate (%)':>13} "
    f"{'FA rate (%)':>12} "
    f"{'Mean RT hits (s)':>17} "
    f"{'d′':>7}"
)

print("─" * 130)

prev_pp = None

for _, row in ps_display.iterrows():
    if prev_pp and row["Participant"] != prev_pp:
        print("─" * 130)

    print(
        f"  {row['Participant']:<16} "
        f"{row['Condition']:<14} "
        f"{int(row['N fixation events']):>8} "
        f"{int(row['Hits']):>6} "
        f"{int(row['Misses']):>8} "
        f"{int(row['False alarms']):>14} "
        f"{int(row['Correct rejections']):>14} "
        f"{row['Hit rate (%)']:>13.1f} "
        f"{row['False alarm rate (%)']:>12.1f} "
        f"{row['Mean RT for hits (s)']:>17.3f} "
        f"{row["d'"]:>7.3f}"
    )

    prev_pp = row["Participant"]

print("═" * 130)


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT 3 — d' input table
# ═════════════════════════════════════════════════════════════════════════════
# This table keeps only the information directly useful for interpreting d'.

dprime_display = ps_display[[
    "Participant",
    "Condition",
    "Hits",
    "Misses",
    "False alarms",
    "Correct rejections",
    "Hit rate (%)",
    "False alarm rate (%)",
    "d'",
]]

print("\n\n" + "═" * 115)
print("  TABLE 3 · Data used to compute d'")
print("  d' is computed from hit rate and false alarm rate")
print("═" * 115)
print(dprime_display.to_string(index=False))
print("═" * 115)


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT 4 — Group-level summary
# ═════════════════════════════════════════════════════════════════════════════
# Important: we average the already-computed per-participant rates rather than
# pooling all events together. This gives the same weight to each participant.

group_summary = participant_summary.groupby("condition")[
    ["hit_rate_%", "FA_rate_%", "mean_RT_hits", "d_prime"]
].agg(["mean", "std", "count"]).round(2)

gs = group_summary.copy()
gs.columns = [f"{metric}_{stat}" for metric, stat in gs.columns]
gs = gs.reset_index()

print("\n\n" + "═" * 105)
print("  TABLE 4 · Group-level summary")
print("  Mean ± SD across participants, with N participants per condition")
print("═" * 105)

print(
    f"  {'Condition':<14} "
    f"{'Hit rate (%)':>18} "
    f"{'FA rate (%)':>18} "
    f"{'Mean RT hits (s)':>22} "
    f"{'d′':>16} "
    f"{'N':>5}"
)

print("─" * 105)

for _, row in gs.iterrows():
    n = int(row["hit_rate_%_count"])

    print(
        f"  {row['condition']:<14} "
        f"{row['hit_rate_%_mean']:>7.1f} ± {row['hit_rate_%_std']:<6.1f} "
        f"{row['FA_rate_%_mean']:>7.1f} ± {row['FA_rate_%_std']:<6.1f} "
        f"{row['mean_RT_hits_mean']:>9.3f} ± {row['mean_RT_hits_std']:<7.3f} "
        f"{row['d_prime_mean']:>6.3f} ± {row['d_prime_std']:<6.3f} "
        f"{n:>5}"
    )

print("═" * 105)


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT 5 — 2 × 2 Repeated-Measures ANOVA on d'
# ═════════════════════════════════════════════════════════════════════════════
# Design:
#   Factor 1 – Contrast polarity : Natural vs Negative
#   Factor 2 – Meridian          : HM vs VM
#   Interaction                  : Contrast polarity × Meridian
#
# Implementation: hand-rolled repeated-measures ANOVA using only numpy/scipy.

# ── Build the wide table: one row per participant, one col per condition ──────

anova_df = participant_summary[["d_prime"]].reset_index()

anova_df["Contrast polarity"] = (
    anova_df["condition"].str.split("-").str[0]
)  # Natural / Negative

anova_df["Meridian"] = (
    anova_df["condition"].str.split("-").str[1]
)  # HM / VM

wide = anova_df.pivot(
    index="participant",
    columns="condition",
    values="d_prime"
)

# ── Extract the four cell vectors ─────────────────────────────────────────────
# Condition order:
#   Natural-HM, Natural-VM, Negative-HM, Negative-VM

try:
    Y = wide[[
        "Natural-HM",
        "Natural-VM",
        "Negative-HM",
        "Negative-VM",
    ]].values.astype(float)

except KeyError as e:
    raise KeyError(
        f"Condition missing from data: {e}. "
        "Check CONDITION_MAP matches your trigger numbers."
    ) from e

n = Y.shape[0]

if n < 2:
    raise RuntimeError(
        "The repeated-measures ANOVA requires at least 2 participants."
    )

# ── Standard SS decomposition for 2-factor within-subject ANOVA ──────────────
# Factor A = Contrast polarity
#   Levels: Natural = cols 0–1, Negative = cols 2–3
#
# Factor B = Meridian
#   Levels: HM = cols 0 and 2, VM = cols 1 and 3

grand_mean = Y.mean()

A_means = np.stack([
    Y[:, [0, 1]].mean(axis=1),   # Natural mean per participant
    Y[:, [2, 3]].mean(axis=1),   # Negative mean per participant
], axis=1)

B_means = np.stack([
    Y[:, [0, 2]].mean(axis=1),   # HM mean per participant
    Y[:, [1, 3]].mean(axis=1),   # VM mean per participant
], axis=1)

# SS for Factor A: Contrast polarity
A_grand = A_means.mean(axis=0)

SS_A = n * 2 * np.sum((A_grand - grand_mean) ** 2)
df_A = 1
MS_A = SS_A / df_A

A_subj_dev = A_means - A_means.mean(axis=1, keepdims=True)

SS_errA = 2 * np.sum(
    (A_subj_dev - (A_grand - grand_mean)) ** 2
)

df_errA = (n - 1) * df_A
MS_errA = SS_errA / df_errA

# SS for Factor B: Meridian
B_grand = B_means.mean(axis=0)

SS_B = n * 2 * np.sum((B_grand - grand_mean) ** 2)
df_B = 1
MS_B = SS_B / df_B

B_subj_dev = B_means - B_means.mean(axis=1, keepdims=True)

SS_errB = 2 * np.sum(
    (B_subj_dev - (B_grand - grand_mean)) ** 2
)

df_errB = (n - 1) * df_B
MS_errB = SS_errB / df_errB

# SS for Interaction A × B
# Interaction contrast:
#   (Natural-HM - Natural-VM) - (Negative-HM - Negative-VM)

AB_contrast = (Y[:, 0] - Y[:, 1]) - (Y[:, 2] - Y[:, 3])

SS_AB = n * (AB_contrast.mean() ** 2) / 2
df_AB = 1
MS_AB = SS_AB / df_AB

SS_errAB = np.sum((AB_contrast - AB_contrast.mean()) ** 2) / 4
df_errAB = (n - 1) * df_AB
MS_errAB = SS_errAB / df_errAB

# F-ratios and p-values
F_A = MS_A / MS_errA
F_B = MS_B / MS_errB
F_AB = MS_AB / MS_errAB

p_A = 1 - f_dist.cdf(F_A, df_A, df_errA)
p_B = 1 - f_dist.cdf(F_B, df_B, df_errB)
p_AB = 1 - f_dist.cdf(F_AB, df_AB, df_errAB)

# Partial eta-squared
eta2_A = SS_A / (SS_A + SS_errA)
eta2_B = SS_B / (SS_B + SS_errB)
eta2_AB = SS_AB / (SS_AB + SS_errAB)


def sig_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ── Print ANOVA table ─────────────────────────────────────────────────────────

print("\n\n" + "═" * 85)
print("  TABLE 5 · 2 × 2 Repeated-Measures ANOVA on d'")
print(f"  Factors: Contrast polarity (Natural/Negative) × Meridian (HM/VM), N = {n}")
print("═" * 85)

print(
    f"  {'Effect':<34} "
    f"{'F(1,' + str(n - 1) + ')':>12} "
    f"{'p':>12} "
    f"{'partial η²':>14} "
    f"{'sig':>6}"
)

print("─" * 85)

for label, F, p, eta2 in [
    ("Contrast polarity", F_A, p_A, eta2_A),
    ("Meridian", F_B, p_B, eta2_B),
    ("Contrast polarity × Meridian", F_AB, p_AB, eta2_AB),
]:
    p_str = "< .001" if p < 0.001 else f"= {p:.3f}"

    print(
        f"  {label:<34} "
        f"{F:>12.3f} "
        f"p {p_str:<9} "
        f"{eta2:>14.3f} "
        f"{sig_stars(p):>6}"
    )

print("═" * 85)
print("  sig: *** p < .001   ** p < .01   * p < .05   ns p ≥ .05")
print("  partial η²: effect size; small ≥ .01, medium ≥ .06, large ≥ .14")


# Create ANOVA results table for export
anova_results = pd.DataFrame({
    "Effect": [
        "Contrast polarity",
        "Meridian",
        "Contrast polarity × Meridian",
    ],
    "df_effect": [
        df_A,
        df_B,
        df_AB,
    ],
    "df_error": [
        df_errA,
        df_errB,
        df_errAB,
    ],
    "F": [
        F_A,
        F_B,
        F_AB,
    ],
    "p": [
        p_A,
        p_B,
        p_AB,
    ],
    "partial_eta_squared": [
        eta2_A,
        eta2_B,
        eta2_AB,
    ],
    "significance": [
        sig_stars(p_A),
        sig_stars(p_B),
        sig_stars(p_AB),
    ],
})

anova_results["F"] = anova_results["F"].round(3)
anova_results["p"] = anova_results["p"].round(4)
anova_results["partial_eta_squared"] = anova_results["partial_eta_squared"].round(3)



# ─────────────────────────────────────────────────────────────
# STEP 6 – Export
# ─────────────────────────────────────────────────────────────
# Uncomment if you want to save the tables.

# master_display.to_csv(
#     DATA_FOLDER / "table1_event_level_master.csv",
#     index=False,
#     encoding="utf-8-sig"
# )

# ps_display.to_csv(
#     DATA_FOLDER / "table2_participant_sdt_results.csv",
#     index=False,
#     encoding="utf-8-sig"
# )

# dprime_display.to_csv(
#     DATA_FOLDER / "table3_dprime_input_table.csv",
#     index=False,
#     encoding="utf-8-sig"
# )

gs.to_csv(
    DATA_FOLDER / "table4_group_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


anova_results.to_csv(
   DATA_FOLDER / "table5_anova_dprime_results.csv",
   index=False,
   encoding="utf-8-sig"
)
