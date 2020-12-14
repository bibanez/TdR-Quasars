from squeze.squeze_common_functions import deserialize, load_json
import pandas as pd
import matplotlib.pyplot as plt


def v_error(z_try, z_true):
    return (z_try - z_true) / (1 + z_try) * c


print("[?] Loading...")
df = deserialize(load_json("../candidates_width70_sig6_test_64plates_sample0.json"))
line_shifts = deserialize(load_json("../processed/combined_line_shifts.json"))
print("[?] File loaded!")

c = 3e5  # km/s
probability_cut = 0.32

quasars = df[(df["prob"] > probability_cut) & df["is_correct"]].copy()

q_v_error = []

for index, row in quasars.iterrows():
    q_v_error.append(v_error(row["z_try"], row["z_true"]))

quasars["v_error"] = q_v_error

lines = dict(tuple(quasars.groupby("assumed_line")))

lines_df = pd.concat([lines[line]["v_error"] for line in lines],
                     ignore_index=True, axis=1)
lines_df.columns = ["CIII", "CIV", "Hα", "Hβ", "Lyα", "MgII"]

print(lines_df)

fig, axes = plt.subplots(nrows=3)

axes[0].set_xlabel("v [km/s]")
axes[0].set_ylabel("Frequency")
for line in ["Lyα", "CIV", "CIII", "MgII"]:
    lines_df[line].plot.hist(ax=axes[0], title="Line by line $Δv$", alpha=0.5)
axes[0].legend()


def correction_shift(row, line_shifts):
    return row["z_try"] - line_shifts[line_shifts["line"] == row["assumed_line"]]["z_shift"].values[0]


quasars["z_try"] = quasars.apply(correction_shift, args=(line_shifts,), axis=1)

q_v_error = []

for index, row in quasars.iterrows():
    q_v_error.append(v_error(row["z_try"], row["z_true"]))

quasars["v_error"] = q_v_error

lines = dict(tuple(quasars.groupby("assumed_line")))

lines_df = pd.concat([lines[line]["v_error"] for line in lines],
                     ignore_index=True, axis=1)
lines_df.columns = ["CIII", "CIV", "Hα", "Hβ", "Lyα", "MgII"]

axes[1].set_xlabel("v [km/s]")
axes[1].set_ylabel("Frequency")
for line in ["Lyα", "CIV", "CIII", "MgII"]:
    lines_df[line].plot.hist(ax=axes[1], title="Line by line $Δv$", alpha=0.5)
axes[1].legend()

axes[2].set_xlabel("v [km/s]")
axes[2].set_ylabel("Frequency")
for line in ["Lyα", "CIV", "CIII", "MgII"]:
    lines_df[line].plot.hist(ax=axes[2], title="Sample-wide $Δv$", color="sandybrown")

plt.tight_layout()
plt.show()
