from squeze.squeze_common_functions import deserialize, load_json
import pandas as pd
import matplotlib.pyplot as plt

df = deserialize(load_json("candidates_width70_sig6_test_64plates_sample0.json"))


def v_error(delta_z, z_try):
    return delta_z / (1 + z_try) * 3e5


probability_cut = 0.28

quasars = df[(df["prob"] > probability_cut) & (df["is_correct"])]

print(v_error(quasars["Delta_z"], quasars["z_try"]).std())

lines = dict(tuple(quasars.groupby("assumed_line")))

lines_df = pd.concat([lines[line]["Delta_z"] for line in lines], ignore_index=True, axis=1)
lines_df.columns = [line for line in lines]

# lines_df.plot.hist(alpha=0.5)
plt.subplot(3, 1, 1)
for line in ["lya", "civ", "ciii", "mgii"]:
    lines_df[line].plot.hist(alpha=0.5)

plt.subplot(3, 1, 2)
quasars["Delta_z"].plot.hist()

for line in lines:
    lines[line]["Delta_z"] -= lines[line]["Delta_z"].mean()

lines_df = pd.concat([lines[line]["Delta_z"] for line in lines], ignore_index=True, axis=1)

plt.subplot(3, 1, 3)
lines_df.plot.hist(alpha=0.5)

quasars = pd.concat([lines[line] for line in lines])

print(v_error(quasars["Delta_z"], quasars["z_try"]).std())

plt.show()

"""
Aquest fitxer probablement l'hauré de refer perquè no va molt bé, però era el
que anava a utilitzar per fer els gràfics per l'article. Ara he vist que ja vas
posar-hi el teu gràfic. Pels següents déixe'm encarregar-me'n a mí, però moltes
gràcies igualment!
"""
