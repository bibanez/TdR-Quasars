from squeze.squeze_common_functions import load_json, deserialize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

fig, ax = plt.subplots()

for n in [1, 2, 4, 8]:
    samples = ["../sample_quasars/processed_validation_64plates_sample{}.json".format(i) for i in range(n)]
    n_bins = 20
    c = 3e5

    sample_dfs = [deserialize(load_json(sample)) for sample in samples]

    df = pd.concat(sample_dfs, sort=False).copy()

    df = df[df["is_correct"] == True]

    df = df.sort_values(by="prob")
    df["Delta_v"] = df["Delta_z"] / (1 + df["z_try"]) * c

    bins = np.linspace(0.32, 1, n_bins)
    group = df.groupby(pd.cut(df.prob, bins))

    plot_centers = (bins[:-1] + bins[1:])/2
    plot_values = group.Delta_v.std()

    ax.plot(plot_centers, plot_values, label=n, alpha=n*0.125)

plt.xlabel("Confidence ($p$)")
plt.ylabel("$\Delta v$ std")
plt.title("$\Delta v$ std by {} bins of $p$".format(n_bins))
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
ax.xaxis.set_major_locator(plt.FixedLocator(bins))
ax.set_ylim(ymin=0)
ax.set_xlim(xmin=bins[0], xmax=bins[-1])
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
