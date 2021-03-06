from squeze.squeze_common_functions import deserialize, load_json
import pandas as pd
import numpy as np

print("[?] Loading file...")
df = deserialize(load_json("../candidates_width70_sig6_validation_64plates_sample1.json"))

line_shifts = deserialize(load_json("../processed/line_shifts_sample0.json"))

print("[?] Finished loading!...")

c = 3e5
probability_cut = [0.32, 0.82]

quasars = {"high": df[(df["prob"] > probability_cut[1])].copy(),
           "low": df[(df["prob"] < probability_cut[1]) & (df["prob"] > probability_cut[0])].copy()}

count = {"high": quasars["high"].groupby("specid")["specid"].count(),
         "low": quasars["low"].groupby("specid")["specid"].count()}

spec_mid = []
for index, row in quasars["high"].iterrows():
    specid = row["specid"]
    if specid in quasars["low"]["specid"].unique() and ~(specid in spec_mid):
        if count["high"][specid] < count["low"][specid]:
            spec_mid.append(specid)

quasars["mid"] = df[(df["prob"] > probability_cut[0]) & (df["specid"].isin(spec_mid))].copy()

quasars["high"] = quasars["high"][~quasars["high"]["specid"].isin(quasars["mid"]["specid"].unique())]

quasars["low"] = quasars["low"][~quasars["low"]["specid"].isin(quasars["mid"]["specid"].unique())]
quasars["low"] = quasars["low"][~(quasars["low"]["specid"].isin(quasars["high"]["specid"].unique()))]


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
    return pd.Series({"z_true": data["z_true"].mean(),
                      "z_try": new_z_try,
                      "class_person": data["class_person"].values[0]})


quasars_split = {"high": 0, "mid": 0, "low": 0}

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


def find_z_median(data):
    """ Takes a DataFrame split by "specid" and
    computes the median z_try for the object
    """
    new_z_try = data["z_try"].median()
    return pd.Series({"z_true": data["z_true"].mean(),
                      "z_try": new_z_try,
                      "class_person": data["class_person"].values[0]})


quasars_split = {"high": 0, "mid": 0, "low": 0}

for quasar in quasars_split:
    quasars_split[quasar] = quasars[quasar].groupby("specid").apply(find_z_median).reset_index()

quasars_corrected = pd.concat([quasars_split[quasar] for quasar in quasars_split])

quasars_corrected["Delta_z"] = quasars_corrected["z_try"] - quasars_corrected["z_true"]

quasars_corrected["is_correct"] = quasars_corrected.apply(is_correct, axis=1)

quasars_corrected = quasars_corrected[(quasars_corrected["is_correct"])]
v_error = (quasars_corrected["Delta_z"]/(quasars_corrected["z_try"]+1)*3e5)

print("··· median v_error std:", v_error.std())
print("··· median v_error mean:", v_error.mean())

print("··· {} quasars entries with p = {}".format(quasars_corrected.shape[0],
                                                  probability_cut))

print(df[(df["prob"] > probability_cut[0])]["specid"].unique().shape[0])
