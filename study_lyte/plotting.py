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

    @property
    def label(self):
        return self.name.title()

class SensorStyle(Enum):
    """
    Enum to handle plotting titles and preferred colors
    """
    # Df column name, plot title, color
    RAW_FORCE = 'Sensor1', 'Raw Force', 'black'
    RAW_AMBIENT_NIR = 'Sensor2', 'Ambient', 'darkorange'
    RAW_ACTIVE_NIR = 'Sensor3', 'Raw Active', 'crimson'
    ACTIVE_NIR = 'nir', 'NIR', 'crimson'
    ACC_X_AXIS = 'X-Axis', 'X-Axis', 'darkslategrey'
    ACC_Y_AXIS = 'Y-Axis', 'Y-Axis', 'darkgreen'
    ACC_Z_AXIS = 'Z-Axis', 'Z-Axis', 'darkorange'
    ACCELERATION = 'acceleration', 'Acc. Magn.', 'darkgreen'
    BAROMETER = 'barometer', 'Baro.', ''
    UNKNOWN = 'UNKNOWN', 'UNKNOWN', None

    @property
    def column(self):
        return self.value[0]

    @property
    def label(self):
        return self.value[1].title()

    @property
    def color(self):
        return self.value[2]

    @classmethod
    def from_column(cls, column):
        result = cls.UNKNOWN
        for e in cls:
            if e.column.upper() == column.upper():
                result = e
                break
        return result


def plot_events(ax, profile_events, plot_type='normal', event_alpha=0.6):
    """
    Plots the hline or vline for each event on a plot
    Args:
        ax: matplotlib.Axes to add horizontal or vertical lines to
        start: Array index to represent start of motion
        surface: Array index to represent snow surface
        stop: Array index to represent stop of motion
        nir_stop: Array index to represent stop estimated by nir
        plot_type: string indicating whether the index is on the y (vertical) or the x (normal)
    """
    # PLotting sensor data on the x axis and time/or depth on y axis
    if plot_type == 'vertical':
        line_fn = ax.axhline

    # Normal time series data with y = values, x = time
    elif plot_type == 'normal':
        line_fn = ax.axvline

    else:
        raise ValueError(f'Unrecognized plot type {plot_type}, options are vertical or normal!')

    for event in profile_events:
        style = EventStyle.from_name(event.name)
        line_fn(event.time, linestyle=style.linestyle, color=style.color, label=style.label, alpha=event_alpha)


def plot_ts(data, data_label=None, time_data=None, events=None, thresholds=None, features=None, show=True, ax=None, alpha=1.0, color=None):
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
    if thresholds is not None:
        for name, tr in thresholds:
            ax.axhline(tr, label=name, alpha=0.2, linestyle='--')

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