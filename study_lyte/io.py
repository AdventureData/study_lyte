<<<<<<< HEAD
import numpy as np
from typing.io import TextIO
=======
>>>>>>> 71d37af8cc290d8c27a4668c005a9882d422659e
from typing import Tuple
import pandas as pd
import numpy as np


def read_csv(f: str) -> Tuple[pd.DataFrame, dict]:
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

        df = pd.read_csv(f, header=header_position)
        # Drop any columns written with the plain index
        df.drop(df.filter(regex="Unname"), axis=1, inplace=True)

        if 'time' not in df and 'SAMPLE RATE' in metadata:
            sr = int(metadata['SAMPLE RATE'])
            n = len(df)
            df['time'] = np.linspace(0, n/sr, n)

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

    df.to_csv(f, mode='a', index=write_index)
