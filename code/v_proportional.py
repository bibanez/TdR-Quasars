from squeze.squeze_common_functions import deserialize, load_json
import pandas as pd
import numpy as np

print("[?] Loading file...")
df = deserialize(load_json("../candidates_width70_sig6_validation_64plates_sample0.json"))

line_shifts = deserialize(load_json("../processed/line_shifts_sample0.json"))

print("[?] Finished loading!...")

c = 3e5
probability_cut = 0.32

quasars = df[(df["prob"] > probability_cut)].copy()


def find_weights(row, line_shifts):
    return line_shifts[line_shifts["line"] == row["assumed_line"]]["weight"].values[0]


quasars["weight"] = quasars.apply(find_weights, axis=1, args=(line_shifts,))


def find_z(data):
    """ Takes a DataFrame split by "specid" and
    computes the mean z_try for the object
    """
    data = data.sort_values(by=["prob"]).head(4)
    new_z_try = (data["z_try"] * data["weight"]).sum() / data["weight"].sum()
    return pd.Series({"z_true": data["z_true"].mean(),
                      "z_try": new_z_try,
                      "class_person": data["class_person"].values[0]})


quasars_corrected = quasars.groupby("specid").apply(find_z).reset_index()

quasars_corrected["Delta_z"] = quasars_corrected["z_try"] - quasars_corrected["z_true"]


def is_correct(row):
    # Returns True if a candidate is a true quasar and False otherwise.
    return bool((row["Delta_z"] <= 0.15)
                and (row["Delta_z"] >= -0.15)
                and (np.isin(row["class_person"], [3, 30])))


quasars_corrected["is_correct"] = quasars_corrected.apply(is_correct, axis=1)

quasars_corrected = quasars_corrected[(quasars_corrected["is_correct"])]
v_error = (quasars_corrected["Delta_z"] / (quasars_corrected["z_try"] + 1) * c)

print("··· mean v_error std: {} km/s".format(v_error.std()))
print("··· mean v_error mean: {} km/s".format(v_error.mean()))


def find_z_median(data):
    """ Takes a DataFrame split by "specid" and
    computes the median z_try for the object
    """
    new_z_try = data["z_try"].median()
    return pd.Series({"z_true": data["z_true"].mean(),
                      "z_try": new_z_try,
                      "class_person": data["class_person"].values[0]})


quasars_corrected = quasars.groupby("specid").apply(find_z_median).reset_index()

quasars_corrected["Delta_z"] = quasars_corrected["z_try"] - quasars_corrected["z_true"]

quasars_corrected["is_correct"] = quasars_corrected.apply(is_correct, axis=1)

quasars_corrected = quasars_corrected[(quasars_corrected["is_correct"])]
v_error = (quasars_corrected["Delta_z"]/(quasars_corrected["z_try"]+1)*3e5)

print("··· median v_error std:", v_error.std())
print("··· median v_error mean:", v_error.mean())

print("··· {} quasars entries with p_min = {}".format(quasars.shape[0], probability_cut))
