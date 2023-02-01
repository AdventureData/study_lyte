import matplotlib.pyplot as plt


def plot_ts(data, time_data=None, events=None, features=None, show=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    n_samples = len(data)
    if n_samples < 1000:
        mark = 'o--'
    else:
        mark = '-'

    if time_data is not None:
        ax.plot(time_data, data, mark)
    else:
        ax.plot(data, mark)

    if events is not None:
        for name, event_idx in events:
            ax.axvline(event_idx, label=name)
    if features is not None:
        ydata = [data[f] for f in features]
        if time_data is not None:
            ax.plot([time_data[f] for f in features], ydata, '.')
        else:
            ax.plot(features, ydata, '.')

    if show:
        plt.show()
    return ax
