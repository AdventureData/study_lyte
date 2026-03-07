from pathlib import Path
from typing import Tuple, Union
import pandas as pd
import numpy as np

def find_metadata(f:str) -> [int, dict]:
    """Read just the metadata from the probe files"""

    # Collect the header
    metadata = {}

    # Use the header position
    header_position = 0

    # Read info as header until there is '=' is not found in the line
    with open(f) as fp:
        for i, line in enumerate(fp):
            if '=' in line:
                k, v = line.split('=')
                k = k.strip().strip('"')
                v = v.strip().strip('"')
                metadata[k] = v
            else:
                header_position = i
                break
    return header_position, metadata


def read_data(f: Union[str, Path], metadata: dict, header_position: int) -> Tuple[pd.DataFrame, dict]:
    """
    Reads just the data from the Lyte probe CSV file
    Args:
        f: Path to csv, or file buffer
        metadata: Dictionary of metadata from the header
        header_position: Line number where the header ends
    Returns:
        tuple:
            **df**: pandas Dataframe
            **metadata**: dictionary containing header info
    """
    # Use engine='c' explicitly and specify dtypes if known
    df = pd.read_csv(f, header=header_position, engine='c')

    # Faster column dropping - avoid regex
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)

    if 'time' not in df.columns and 'SAMPLE RATE' in metadata:
        sr = int(metadata['SAMPLE RATE'])
        n = len(df)
        df['time'] = np.linspace(0, n / sr, n)
    return df, metadata


def read_csv(f: Union[str, Path]) -> Tuple[pd.DataFrame, dict]:
    """
    Reads any Lyte probe CSV and returns a dataframe
    and metadata dictionary from the header

    Args:
        f: Path to csv, or file buffer
    Returns:
        tuple:
            **df**: pandas Dataframe
            **header**: dictionary containing header info
    """
    header_position, metadata = find_metadata(f)
    df, metadata = read_data(f, metadata, header_position)
    return df, metadata


def write_csv(df: pd.DataFrame, meta: dict, f: str) -> None:
    """
    Write out the results with a header using the dictionary

    Args:
        df: Pandas Dataframe
        meta: Dictionary of information to write above the data as a header
        f: String path to write the data to
    """

    with open(f, 'w+') as fp:
        for k, v in meta.items():
            fp.write(f'{k} = {v}\n')
    # write out time if it is the index
    if df.index.name == 'time':
        write_index = True
    else:
        write_index = False

    # Format columns for cleaner output - round before writing
    df_out = df.copy()

    # 4 decimals for g measurements (accelerometer axes)
    accel_cols = [c for c in ['X-Axis', 'Y-Axis', 'Z-Axis'] if c in df_out.columns]
    for col in accel_cols:
        df_out[col] = df_out[col].round(4)

    # 6 decimals for 16kHz timing (62.5µs resolution)
    if 'time' in df_out.columns:
        df_out['time'] = df_out['time'].round(6)
    elif 'time' in df_out.index.names:
        df_out.index = df_out.index.round(6)

    # 1 decimal for depth in cm (0.1cm resolution)
    if 'depth' in df_out.columns:
        df_out['depth'] = df_out['depth'].round(1)

    # Sensor columns are integers
    sensor_cols = [c for c in ['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4'] if c in df_out.columns]
    for col in sensor_cols:
        df_out[col] = df_out[col].astype(int)

    df_out.to_csv(f, mode='a', index=write_index)
