import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
#%matplotlib notebook

def plotModelStats(orig_data, alt_data, label):
    # plot settings
    figsize = (10, 5)
    fontsize = 12
    labelsize= 8
    ticksize = 8
    pad = 6
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.2, hspace=0.0, bottom=0.15, left=0.1, right=0.95, top=0.9)
    ax = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0,1])
    ax4 = fig.add_subplot(gs[1, 1])

    # compute original data
    orig = orig_data[(orig_data["prob"] > 0)].groupby("prob").mean().reset_index()

    # compute mean and errors to plot in alternative data
    model = alt_data[(alt_data["prob"] > 0) & (alt_data["model"] == "orig")].groupby("prob").mean().reset_index()
    error = alt_data[(alt_data["prob"] > 0) & (alt_data["model"] == "orig")].groupby("prob").std().reset_index()
    model2 = alt_data[(alt_data["prob"] > 0) & (alt_data["model"] == label)].groupby("prob").mean().reset_index()
    error2 = alt_data[(alt_data["prob"] > 0) & (alt_data["model"] == label)].groupby("prob").std().reset_index()

    # subtract original data to models
    model["purity z >= 2.1"] -= orig["purity z >= 2.1"]
    model["purity"] -= orig["purity"]
    model["completeness z >= 2.1"] -= orig["completeness z >= 2.1"]
    model["completeness"] -= orig["completeness"]
    model2["purity z >= 2.1"] -= orig["purity z >= 2.1"]
    model2["purity"] -= orig["purity"]
    model2["completeness z >= 2.1"] -= orig["completeness z >= 2.1"]
    model2["completeness"] -= orig["completeness"]

    # plot model
    ax.plot(model["prob"], model["purity z >= 2.1"],
            color="r", linestyle="solid", label="original training")
    ax2.plot(model["prob"], model["completeness z >= 2.1"],
             color="r", linestyle="solid")
    ax3.plot(model["prob"], model["purity"], color="r",
                     linestyle="solid", linewidth=1, label="original training")
    ax4.plot(model["prob"], model["completeness"], color="r",
             linestyle="solid", linewidth=1)

    # plot model 2
    ax.plot(model2["prob"], model2["purity z >= 2.1"],
            color="b", linestyle="solid", label="{} training".format(label))
    ax2.plot(model2["prob"], model2["completeness z >= 2.1"],
             color="b", linestyle="solid")
    ax3.plot(model2["prob"], model2["purity"], color="b",
             linestyle="solid", linewidth=1, label="{} training".format(label))
    ax4.plot(model2["prob"], model2["completeness"], color="b",
             linestyle="solid", linewidth=1)

    # plot legend
    #labs = [l.get_label() for l in lns]
    ax.legend(loc=0, fontsize=fontsize, frameon=False)

    # plot errors
    ax.fill_between(model["prob"],
                    model["purity z >= 2.1"] + error["purity z >= 2.1"],
                    model["purity z >= 2.1"] - error["purity z >= 2.1"],
                    color="r", alpha=0.5)
    ax2.fill_between(model["prob"],
                     model["completeness z >= 2.1"] + error["completeness z >= 2.1"],
                     model["completeness z >= 2.1"] - error["completeness z >= 2.1"],
                     color="r", alpha=0.5)
    ax3.fill_between(model["prob"],
                     model["purity"] + error["purity"],
                     model["purity"] - error["purity"],
                     color="r", alpha=0.5)
    ax4.fill_between(model["prob"],
                     model["completeness"] + error["completeness"],
                     model["completeness"] - error["completeness"],
                     color="r", alpha=0.5)

    # plot errors 2
    ax.fill_between(model2["prob"],
                    model2["purity z >= 2.1"] + error2["purity z >= 2.1"],
                    model2["purity z >= 2.1"] - error2["purity z >= 2.1"],
                    color="b", alpha=0.5)
    ax2.fill_between(model2["prob"],
                     model2["completeness z >= 2.1"] + error2["completeness z >= 2.1"],
                     model2["completeness z >= 2.1"] - error2["completeness z >= 2.1"],
                     color="b", alpha=0.5)
    ax3.fill_between(model2["prob"],
                     model2["purity"] + error["purity"],
                     model2["purity"] - error["purity"],
                     color="b", alpha=0.5)
    ax4.fill_between(model2["prob"],
                     model2["completeness"] + error2["completeness"],
                     model2["completeness"] - error2["completeness"],
                     color="b", alpha=0.5)

    # format axes
    ax.set_ylabel("purity - orig [%]", fontsize=fontsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=fontsize, pad=pad, top=True,
                   right=True, length=ticksize, direction="inout",
                   labelbottom=False)
    ax.set_title(r"$z \geq 2.1$", fontsize=fontsize, y=1.04)
    ax2.set_xlabel(r"$p_{\rm min}$", fontsize=fontsize)
    ax2.set_ylabel("comp - orig [%]", fontsize=fontsize)
    ax2.tick_params(axis='both', labelsize=fontsize, pad=pad, top=True,
                   right=True, length=ticksize, direction="inout")
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.tick_params(axis='both', labelsize=fontsize, pad=pad, top=True,
                   right=True, length=ticksize, direction="inout",
                   labelbottom=False)
    ax3.set_title(r"$z \geq 0.0$", fontsize=fontsize, y=1.04)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax4.set_xlabel(r"$p_{\rm min}$", fontsize=fontsize)
    ax4.tick_params(axis='both', labelsize=fontsize, pad=pad, top=True,
                   right=True, length=ticksize, direction="inout")
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.savefig("performance_vs_p_comparison_diff_{}.png".format(label))
