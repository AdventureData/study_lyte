import matplotlib.pyplot as plt
from enum import Enum


class EventStyle(Enum):
    START = 'g', '--'
    STOP = 'r', '--'
    SURFACE = 'lightsteelblue', '--'

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


def plot_ts(data, data_label=None, time_data=None, events=None, features=None, show=True, ax=None, alpha=1.0, color=None):
    if ax is None:
        fig, ax = plt.subplots(1)
        ax.grid(True)
    n_samples = len(data)
    if n_samples < 100:
        mark = 'o--'
    else:
        mark = '-'

    if time_data is not None:
        ax.plot(time_data, data, mark, alpha=alpha, label=data_label, color=color)
    else:
        ax.plot(data, mark, alpha=alpha, label=data_label, color=color)

    if data_label is not None:
        ax.legend()

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


def plot_constrained_baro(orig, partial, full, acc_pos, top, bottom, start, stop,
                          baro='filtereddepth', acc_axis='Y-Axis'):

    # zero it out
    partial[baro] = partial[baro] - partial[baro].iloc[0]
    # partial = partial.reset_index('time')
    # orig = orig.set_index('time')

    mid = int((start+stop)/2)

    orig[baro] = orig[baro] - orig[baro].iloc[0]
    ax = plot_ts(orig[baro], time_data=orig['time'], color='steelblue', alpha=0.2,
                 data_label='Orig.', show=False, features=[top, bottom])
    ax = plot_ts(acc_pos[acc_axis], time_data=acc_pos['time'], color='black', alpha=0.5,
                 ax=ax, data_label='Acc.', show=False,
                 events=[('start', start), ('stop', stop), ('mid', mid)])
    ax = plot_ts(partial[baro], time_data=partial['time'], color='blue',
                 ax=ax, show=False, data_label='Part. Const.', alpha=0.3)
    ax = plot_ts(full, time_data=partial['time'], color='magenta', alpha=1,
                 ax=ax, show=True, data_label='Constr.')