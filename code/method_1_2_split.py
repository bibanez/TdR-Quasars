from squeze.squeze_common_functions import deserialize, load_json, save_json
import pandas as pd
import numpy as np

print("[?] Loading file...")
df = deserialize(load_json("../candidates_width70_sig6_validation_64plates_sample0.json"))

line_shifts = deserialize(load_json("../processed/combined_line_shifts.json"))
low_cut = 0.32
high_cut = 0.77
c = 3e5

# # # # # # # # # # # #
# Method 1: Line mean #
# # # # # # # # # # # #

# Only the probability cut is applied in order to perform method 2 later on.
quasars = df[(df["prob"] > low_cut)].copy()

correct_quasars = quasars[~(quasars["duplicated"]) & (quasars["is_correct"])]
v_error = (correct_quasars["z_try"] - correct_quasars["z_true"]) / (1 + correct_quasars["z_try"]) * c
print("··· initial std = {} km/s".format(v_error.std()))
print("··· initial mean = {} km/s".format(v_error.mean()))


def correction_shift(row, line_shifts):
    return row["z_try"] - line_shifts[line_shifts["line"] == row["assumed_line"]]["z_shift"].values[0]


quasars["z_try"] = quasars.apply(correction_shift, args=(line_shifts,), axis=1)

# To print the true error, we filter out the duplicated quasars and leave only
# the ones deemed correct (Delta_z <= 0.15).
correct_quasars = quasars[~(quasars["duplicated"]) & (quasars["is_correct"])]
corrected_v_error = (correct_quasars["z_try"] - correct_quasars["z_true"]) / (1 + correct_quasars["z_try"]) * c

print("... first corrected std = {} km/s".format(corrected_v_error.std()))
print("... first corrected mean = {} km/s".format(corrected_v_error.mean()))

# # # # # # # # # # # # # # # #
# Method 2: Line combination  #
# # # # # # # # # # # # # # # #

# Separating lines between high_cut < p and low_cut < p < high_cut
quasars = {"high": quasars[(quasars["prob"] >= high_cut)].copy(),
           "low": quasars[(quasars["prob"] < high_cut)].copy()}

# Spectra found in quasars["high"] are removed from quasars["low"]
quasars["low"] = quasars["low"][(~quasars["low"]["specid"].isin(quasars["high"]["specid"].unique()))]


# Method 2 by mean

# A column containing the line error weight is added.
def find_weights(row, line_shifts):
    return line_shifts[line_shifts["line"] == row["assumed_line"]]["weight"].values[0]


for quasar in quasars:
    quasars[quasar]["weight"] = quasars[quasar].apply(find_weights, axis=1, args=(line_shifts,))


def find_z(data):
    """ Takes a DataFrame split by "specid" and
    computes the mean z_try for the object
    """
    new_z_try = (data["z_try"] * data["weight"]).sum() / data["weight"].sum()
    new_prob = (data["prob"] * data["weight"]).sum() / data["weight"].sum()
    return pd.Series({"z_true": data["z_true"].mean(),
                      "z_try": new_z_try,
                      "prob": new_prob,
                      "class_person": data["class_person"].values[0]})


quasars_split = {"high": 0, "low": 0}

for quasar in quasars_split:
    quasars_split[quasar] = quasars[quasar].groupby("specid").apply(find_z).reset_index()

quasars_corrected = pd.concat([quasars_split[quasar] for quasar in quasars_split])

# Delta_z is recalculated with the new z_try
quasars_corrected["Delta_z"] = quasars_corrected["z_try"] - quasars_corrected["z_true"]


# The class of the quasar is reassigned with the new Delta_z values.
def is_correct(row):
    # Returns True if a candidate is a true quasar and False otherwise.
    return bool((row["Delta_z"] <= 0.15)
                and (row["Delta_z"] >= -0.15)
                and (np.isin(row["class_person"], [3, 30])))


quasars_corrected["is_correct"] = quasars_corrected.apply(is_correct, axis=1)

# To print the true error, we filter out the duplicated quasars and leave only
# the ones deemed correct (Delta_z <= 0.15).
quasars_corrected = quasars_corrected[(quasars_corrected["is_correct"])]
v_error = (quasars_corrected["Delta_z"] / (quasars_corrected["z_try"] + 1) * c)

print("··· mean v_error std: {} km/s".format(v_error.std()))
print("··· mean v_error mean: {} km/s".format(v_error.mean()))


# Method 2 by median
def find_z_median(data):
    """ Takes a DataFrame split by "specid" and
    computes the median z_try for the object
    """
    new_z_try = data["z_try"].median()
    new_prob = data["prob"].median()
    return pd.Series({"z_true": data["z_true"].mean(),
                      "z_try": new_z_try,
                      "prob": new_prob,
                      "class_person": data["class_person"].values[0]})


quasars_split = {"high": 0, "low": 0}

for quasar in quasars_split:
    quasars_split[quasar] = quasars[quasar].groupby("specid").apply(find_z_median).reset_index()

quasars_corrected = pd.concat([quasars_split[quasar] for quasar in quasars_split])

quasars_corrected["Delta_z"] = quasars_corrected["z_try"] - quasars_corrected["z_true"]

quasars_corrected["is_correct"] = quasars_corrected.apply(is_correct, axis=1)

purity = (quasars_corrected[(quasars_corrected["is_correct"])]["specid"].unique().shape[0]
          / df[(df["is_correct"])]["specid"].unique().shape[0])
completeness = quasars_corrected["specid"].unique().shape[0]/df["specid"].unique().shape[0]

quasars_corrected = quasars_corrected[(quasars_corrected["is_correct"])]
v_error = (quasars_corrected["Delta_z"]/(quasars_corrected["z_try"]+1)*3e5)

print("··· median v_error std: {} km/s".format(v_error.std()))
print("··· median v_error mean: {} km/s".format(v_error.mean()))


# Method 2 by combined median and mean
def find_z_comb(data):
    """ Takes a DataFrame split by "specid" and
    computes the mean z_try for the object
    """
    # Samples that differ more than 0.15 from the median z_try are discarded
    median = data["z_try"].median()
    if data.shape[0] % 2 == 1:
        data = data[(abs(data["z_try"] - median) <= 0.15)]
        new_z_try = (data["z_try"] * data["weight"]).sum() / data["weight"].sum()
        new_prob = (data["prob"] * data["weight"]).sum() / data["weight"].sum()
        return pd.Series({"z_true": data["z_true"].mean(),
                          "z_try": new_z_try,
                          "prob": new_prob,
                          "class_person": data["class_person"].values[0]})
    else:
        new_z_try = data["z_try"].median()
        new_prob = data["prob"].median()
        return pd.Series({"z_true": data["z_true"].mean(),
                          "z_try": new_z_try,
                          "prob": new_prob,
                          "class_person": data["class_person"].values[0]})


quasars_split = {"high": 0, "low": 0}

for quasar in quasars_split:
    quasars_split[quasar] = quasars[quasar].groupby("specid").apply(find_z_comb).reset_index()

quasars_corrected = pd.concat([quasars_split[quasar] for quasar in quasars_split])

quasars_corrected["Delta_z"] = quasars_corrected["z_try"] - quasars_corrected["z_true"]

quasars_corrected["is_correct"] = quasars_corrected.apply(is_correct, axis=1)
print(quasars_corrected.head(20))
print(quasars_corrected.reset_index().head(20))

save_json("../sample_quasars/processed_validation_64plates_sample0.json", quasars_corrected.reset_index())

quasars_final = quasars_corrected[(quasars_corrected["is_correct"])]
v_error = (quasars_final["Delta_z"]/(quasars_final["z_try"]+1)*3e5)

print("··· combined v_error std: {} km/s".format(v_error.std()))
print("··· combined v_error mean: {} km/s".format(v_error.mean()))


print("··· {} quasars entries with p = {}, {}".format(quasars_final["specid"].unique().shape[0],
                                                      low_cut, high_cut))
