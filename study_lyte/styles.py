from enum import Enum


class EventStyle(Enum):
    """
    Styles for plotting events in a timeseries, enums defined by
    color, line style and line width
    """
    START = 'g', '--', 1
    STOP = 'r', '--',  1
    SURFACE = 'lightsteelblue', '--', 1
    ERROR = 'orangered', 'dotted',   1
    UNKNOWN = 'k', '--', 1

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
        return self.value[1]

    @property
    def linewidth(self):
        return self.value[2]

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
    FUSED = 'fused', 'Fused', 'magenta'
    CONSTRAINED_BAROMETER = 'barometer', 'Constr. Baro.', 'navy'
    RAW_BARO = 'filtereddepth', 'Raw Baro.', 'Brown'
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