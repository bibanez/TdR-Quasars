from squeze.squeze_common_functions import deserialize, load_json
import pandas as pd
import numpy as np

print("[?] Loading file...")
df = deserialize(load_json("candidates_width70_sig6_validation_64plates_sample0.json"))

line_shifts = deserialize(load_json("../processed/line_shifts_sample0.json"))

probability_cut = 0.32

# Seleccionem els quàsars. Fixa't-hi que crido la funció
# copy(). Això és per a crear un nou dataframe (i no només
# una vista del que teníem. Té vàries aplicacions però
# l'important aquí és que em permet calcular totes les
# velocitats de cop i de forma ràpida.
# Aquí no podem triar encara els que són correctes perquè a priori això
# no ho sabem, així que cal fer servir la info dels objectes que sí que sabem
quasars = df[(df["prob"] > probability_cut)].copy()


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

print("··· v_error std:", v_error.std())
print("··· v_error mean:", v_error.mean())



# Recalculem el redshift per a cada objecte fent servir el mètode de la median
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

print("··· v_error std:", v_error.std())
print("··· v_error mean:", v_error.mean())
