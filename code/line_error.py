from squeze.squeze_common_functions import deserialize, load_json, save_json
import pandas as pd

file_names = ["../candidates_width70_sig6_train_64plates_sample{}.json"]
lines_file_name = "../processed/line_shifts_sample{}.json"

c = 3e5  # km/s

for file in range(len(file_names)):
    for sample in range(8):
        print("[?] File {} Sample {} loading...".format(file, sample))
        df = deserialize(load_json(file_names[file].format(sample)))

        print("[?] Sample {} loaded.".format(sample))

        quasars = df[(df["is_correct"])].copy()

        quasars["Delta_v"] = quasars["Delta_z"] / (1 + quasars["z_try"]) * c

        def line_stats(data):
            """ Takes a DataFrame split by "assumed_line" and
            computes the mean Delta-z and the standard deviation
            """
            return pd.Series({'z_shift': data["Delta_z"].mean(),
                              'v_shift': data["Delta_v"].mean(),
                              'v_disp': data["Delta_v"].std()})

        lines = quasars.groupby("assumed_line").apply(line_stats).reset_index()
        lines = lines.rename(columns={"assumed_line": "line"})

        lines["weight"] = 1.0/lines["v_disp"].abs()

        save_json(lines_file_name.format(sample), lines)

        print("[?] Sample {} completed!".format(sample), "\n")
