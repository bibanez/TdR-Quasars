from squeze.squeze_common_functions import deserialize, load_json
import matplotlib.pyplot as plt

print("[?] Loading file...")
df = deserialize(load_json("../candidates_width70_sig6_validation_64plates_sample0.json"))

lines = {"ciii": "CIII", "civ": "CIV", "ha": "Hα", "hb": "Hβ", "lya": "Lyα",
         "mgii": "MgII"}

c = 3e5

quasars = df[(df["is_correct"])].copy()
quasars["Delta_v"] = quasars["Delta_z"] / (1 + quasars["z_try"]) * c

quasar_lines = quasars[~(quasars["assumed_line"].isin(["ha", "hb"]))].copy()

for index, row in quasar_lines.iterrows():
    row["assumed_line"] = lines[row["assumed_line"]]

quasar_lines = quasar_lines.groupby("assumed_line")

fig, axes = plt.subplots(nrows=2)

quasar_lines["Delta_v"].plot.hist(ax=axes[0], bins=20,
                                  title="$Δv$ for each line",
                                  alpha=0.5, legend=True)

axes[0].set_xlabel("$Δv$ [km/s]")
axes[0].set_ylabel("Number of objects")
axes[0].legend()

quasars["Delta_v"].plot.hist(ax=axes[1],
                             title="$Δv$ sample wide",
                             color="seagreen", bins=20)

axes[1].set_xlabel("$Δv$ [km/s]")
axes[1].set_ylabel("Number of objects")

plt.tight_layout()
plt.show()
