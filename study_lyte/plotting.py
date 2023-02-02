import matplotlib.pyplot as plt
from enum import Enum


class EventStyle(Enum):
    START = 'g', '--'
    STOP = 'r', '--'
    UNKNOWN = 'k', '--'

    @classmethod
    def from_name(cls, name):
        result = cls.UNKNOWN
        for e in cls:
            if e.name == name.upper():
                result = e
                break
        return result

    @property
    def color(self):
        return self.value[0]

    @property
    def linestyle(self):
        return self.value[-1]


def plot_ts(data, time_data=None, events=None, features=None, show=True, ax=None, alpha=1.0):
    if ax is None:
        fig, ax = plt.subplots(1)
        ax.grid(True)
    n_samples = len(data)
    if n_samples < 100:
        mark = 'o--'
    else:
        mark = '-'

    if time_data is not None:
        ax.plot(time_data, data, mark, alpha=alpha)
    else:
        ax.plot(data, mark, alpha=alpha)

    if events is not None:
        for name, event_idx in events:
            s = EventStyle.from_name(name)
            if time_data is not None:
                v = time_data[event_idx]
            else:
                v = event_idx
            ax.axvline(v, color=s.color, linestyle=s.linestyle, label=name)

    if features is not None:
        ydata = [data[f] for f in features]
        if time_data is not None:
            ax.plot([time_data[f] for f in features], ydata, '.')
        else:
            ax.plot(features, ydata, '.')

    if show:
        plt.show()

    return ax
