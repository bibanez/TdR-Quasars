from squeze.squeze_common_functions import deserialize, load_json, save_json
import pandas as pd

file_names = ["candidates_width70_sig6_train_64plates_sample{}.json"]
lines_file_name = "processed/line_shifts_sample{}.json"

c = 3e5  # km/s

for file in range(len(file_names)):
    for sample in range(8):
        print("[?] File {} Sample {} loading...".format(file, sample))
        df = deserialize(load_json(file_names[file].format(sample)))

        print("[?] Sample {} loaded.".format(sample))

        # Seleccionem els quàsars. Fixa't-hi que crido la funció
        # copy(). Això és per a crear un nou dataframe (i no només
        # una vista del que teníem. Té vàries aplicacions però
        # l'important aquí és que em permet calcular totes les
        # velocitats de cop i de forma ràpida
        quasars = df[(df["is_correct"])].copy()

        # Ara calculem les velocitats. Fixa'thi que la resta del
        # numerador ja la tens feta a Delta_z així que no cal
        # tornar-la a fer
        quasars["Delta_v"] = quasars["Delta_z"] / (1 + quasars["z_true"]) * c

        # aquí calculo, per a una línia, el promig de Delta_z
        # (pel primer mètode) i la desviació estàndard de Delta_v
        # (pel segon mètode)
        def line_stats(data):
            """ Takes a DataFrame split by "assumed_line" and
            computes the mean Delta-z and the standard deviation
            """
            return pd.Series({'z_shift': data["Delta_z"].mean(),
                              'v_disp': data["Delta_v"].std()})

        # aplico la funció directament al groupby per a que
        # m'ho faci de forma eficient a cada línia
        # poso també la funció reset_index() per a que el nom
        # de la línia sigui una nova columna
        lines = quasars.groupby("assumed_line").apply(line_stats).reset_index()
        # canvio el nom de la primera columna
        lines = lines.rename(columns={"assumed_line": "line"})
        # afegeixo una columna amb els pesos
        lines["weight"] = 1.0/lines["v_disp"].abs()

        # guardo el fitxer
        save_json(lines_file_name.format(sample), lines)

        print("[?] Sample {} completed!".format(sample), "\n")
