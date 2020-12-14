import pandas as pd
import matplotlib.pyplot as plt

complete_errors = pd.read_csv("../plots/high_cut.csv")

fig = plt.figure()
ax = plt.axes()

ax.plot(complete_errors["high_cut"], complete_errors["mean_std"])
ax.plot(complete_errors["high_cut"], complete_errors["median_std"])
ax.plot(complete_errors["high_cut"], complete_errors["comb_std"])
ax.set_xlabel("High cut")
ax.set_ylabel("$Δv$ [km/s]")
ax.legend()
plt.show()
