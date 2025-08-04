import matplotlib.pyplot as plt


def plot_hist(hist, title="Histogram", xlabel=None, ylabel="Entries"):
    fig, ax = plt.subplots()
    hist.plot(ax=ax)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax
