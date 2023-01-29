import matplotlib.pyplot as plt


def plot_ts(data, events=None, features=None, show=True):
    fig, ax = plt.subplots(1)
    ax.plot(data)
    if events is not None:
        for name, event_idx in events:
            ax.axvline(event_idx, label=name)
    if features is not None:
        ydata = [data[f] for f in features]
        ax.plot(features, ydata, '.')
    if show:
        plt.show()
    return ax
