from squeze.squeze_common_functions import deserialize, load_json
import pandas as pd
import numpy as np

print("[?] Loading file...")
df = deserialize(load_json("../candidates_width70_sig6_validation_64plates_sample0.json"))

line_shifts = deserialize(load_json("../processed/line_shifts_sample0.json"))
probability_cut = 0.62
c = 3e5

# # # # # # # # # # # # #
# Mètode 1 (modificat): #
# # # # # # # # # # # # #

# Filtrem només per probability_cut per a poder aplicar després
# el mètode 2
quasars = df[(df["prob"] > probability_cut)].copy()

# calculem l'error inicial abans de canviar el valor de z_try
# per trobar els resultats correctes del primer mètode cal
# filtrar pels quasars que són correctes i que no estan
# repetits (no ho hem fet abans per a poder aplicar després
# el segon mètode)
correct_quasars = quasars[~(quasars["duplicated"]) & (quasars["is_correct"])]
v_error = (correct_quasars["z_try"] - correct_quasars["z_true"]) / (1 + correct_quasars["z_try"]) * c
print("initial std = {} km/s".format(v_error.std()))
print("initial mean = {} km/s".format(v_error.mean()))

# calculem la correccó del mètode 1 i (IMPORTANT!) la guardem
# a la mateixa variable
def correction_shift(row, line_shifts):
    return row["z_try"] - line_shifts[line_shifts["line"] == row["assumed_line"]]["z_shift"].values[0]
quasars["z_try"] = quasars.apply(correction_shift, args=(line_shifts,), axis=1)

# per trobar els resultats correctes del primer mètode cal
# filtrar pels quasars que són correctes i que no estan
# repetits (no ho hem fet abans per a poder aplicar després
# el segon mètode)
correct_quasars = quasars[~(quasars["duplicated"]) & (quasars["is_correct"])]
corrected_v_error = (correct_quasars["z_try"] - correct_quasars["z_true"]) / (1 + correct_quasars["z_try"]) * c
print("... first corrected std = {} km/s".format(corrected_v_error.std()))
print("... first corrected mean = {} km/s".format(corrected_v_error.mean()))


# # # # # # #
# Mètode 2: #
# # # # # # #

# Afegim una columna que contingui el pes que li toca a cada entrada
def find_weights(row, line_shifts):
    return line_shifts[line_shifts["line"] == row["assumed_line"]]["weight"].values[0]
quasars["weight"] = quasars.apply(find_weights, axis=1, args=(line_shifts,))

# Recalculem el redshift per a cada objecte
def find_z(data):
    """ Takes a DataFrame split by "specid" and
    computes the mean z_try for the object
    """
    new_z_try = (data["z_try"]*data["weight"]).sum()/data["weight"].sum()
    return pd.Series({"z_true": data["z_true"].mean(),
                      "z_try": new_z_try,
                      "class_person": data["class_person"].values[0],
                     })
quasars_corrected = quasars.groupby("specid").apply(find_z).reset_index()

# calculem el Delta_z
quasars_corrected["Delta_z"] = quasars_corrected["z_try"] - quasars_corrected["z_true"]
# recalculo quins són correctes
def is_correct(row):
        """ Returns True if a candidate is a true quasar and False otherwise."""
        return bool((row["Delta_z"] <= 0.15)
                    and (row["Delta_z"] >= -0.15)
                    and (np.isin(row["class_person"], [3, 30])))
quasars_corrected["is_correct"] = quasars_corrected.apply(is_correct, axis=1)

# per trobar els resultats correctes del primer mètode cal
# filtrar pels quasars que són correctes i que no estan
# repetits (no ho hem fet abans per a poder aplicar després
# el segon mètode)
quasars_corrected = quasars_corrected[(quasars_corrected["is_correct"])]
v_error = (quasars_corrected["Delta_z"]/(quasars_corrected["z_try"]+1)*3e5)

print("... v_error std:", v_error.std())
print("... v_error mean:", v_error.mean())



# Recalculem el redshift per a cada objecte fent servir el mètode de la median
# la primera correcció està aplicada però la primera vegada que hem aplicat el
# mètode 2 no
def find_z_median(data):
    """ Takes a DataFrame split by "specid" and
    computes the mean z_try for the object
    """
    new_z_try = data["z_try"].median()
    return pd.Series({"z_true": data["z_true"].mean(),
                      "z_try": new_z_try,
                      "class_person": data["class_person"].values[0],
                     })
quasars_corrected = quasars.groupby("specid").apply(find_z_median).reset_index()

# calculem el Delta_z
quasars_corrected["Delta_z"] = quasars_corrected["z_try"] - quasars_corrected["z_true"]
# recalculo quins són correctes
def is_correct(row):
        """ Returns True if a candidate is a true quasar and False otherwise."""
        return bool((row["Delta_z"] <= 0.15)
                    and (row["Delta_z"] >= -0.15)
                    and (np.isin(row["class_person"], [3, 30])))
quasars_corrected["is_correct"] = quasars_corrected.apply(is_correct, axis=1)

# per trobar els resultats correctes del primer mètode cal
# filtrar pels quasars que són correctes i que no estan
# repetits (no ho hem fet abans per a poder aplicar després
# el segon mètode)
quasars_corrected = quasars_corrected[(quasars_corrected["is_correct"])]
v_error = (quasars_corrected["Delta_z"]/(quasars_corrected["z_try"]+1)*3e5)

print("... v_error std:", v_error.std())
print("... v_error mean:", v_error.mean())
