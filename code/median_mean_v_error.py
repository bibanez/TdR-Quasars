from squeze.squeze_common_functions import deserialize, load_json, save_json
import pandas as pd
import numpy as np

print("[?] Loading file...")
df = deserialize(load_json("../candidates_width70_sig6_validation_64plates_sample0.json"))

line_shifts = deserialize(load_json("../processed/line_shifts_sample0.json"))

print("[?] Finished loading!...")

c = 3e5
prob_ranges = np.arange(0.05, 1.0, 0.05)

entries = pd.DataFrame(columns=["prob", "entries", "mean_v_std", "mean_v_mean",
                                "median_v_std", "median_v_mean"])


def find_weights(row, line_shifts):
    return line_shifts[line_shifts["line"] == row["assumed_line"]]["weight"].values[0]


def find_z(data):
    """ Takes a DataFrame split by "specid" and
    computes the mean z_try for the object
    """
    new_z_try = (data["z_try"] * data["weight"]).sum() / data["weight"].sum()
    return pd.Series({"z_true": data["z_true"].mean(),
                      "z_try": new_z_try,
                      "class_person": data["class_person"].values[0]})


def is_correct(row):
    # Returns True if a candidate is a true quasar and False otherwise.
    return bool((row["Delta_z"] <= 0.15)
                and (row["Delta_z"] >= -0.15)
                and (np.isin(row["class_person"], [3, 30])))


def find_z_median(data):
    """ Takes a DataFrame split by "specid" and
    computes the median z_try for the object
    """
    new_z_try = data["z_try"].median()
    return pd.Series({"z_true": data["z_true"].mean(),
                      "z_try": new_z_try,
                      "class_person": data["class_person"].values[0]})


for probability_cut in prob_ranges:
    entry = pd.Series(index=entries.columns)

    quasars = df[(df["prob"] > probability_cut)].copy()

    entry["prob"] = probability_cut
    entry["entries"] = quasars.shape[0]

    quasars["weight"] = quasars.apply(find_weights, axis=1, args=(line_shifts,))

    quasars_corrected = quasars.groupby("specid").apply(find_z).reset_index()

    quasars_corrected["Delta_z"] = quasars_corrected["z_try"] - quasars_corrected["z_true"]

    quasars_corrected["is_correct"] = quasars_corrected.apply(is_correct, axis=1)

    quasars_corrected = quasars_corrected[(quasars_corrected["is_correct"])]
    v_error = (quasars_corrected["Delta_z"] / (quasars_corrected["z_try"] + 1) * c)

    entry["mean_v_std"] = v_error.std()
    entry["mean_v_mean"] = v_error.mean()

    quasars_corrected = quasars.groupby("specid").apply(find_z_median).reset_index()

    quasars_corrected["Delta_z"] = quasars_corrected["z_try"] - quasars_corrected["z_true"]

    quasars_corrected["is_correct"] = quasars_corrected.apply(is_correct, axis=1)

    quasars_corrected = quasars_corrected[(quasars_corrected["is_correct"])]
    v_error = (quasars_corrected["Delta_z"]/(quasars_corrected["z_try"]+1)*3e5)

    entry["median_v_std"] = v_error.std()
    entry["median_v_mean"] = v_error.mean()

    entries = entries.append(entry, ignore_index=True)
    print("··· {} quasars entries with p_min = {}".format(quasars.shape[0], probability_cut))

save_json("probability_cuts.json", entries)
