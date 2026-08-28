from pathlib import Path
from typing import Optional, Tuple, Union
import uuid
import pandas as pd
import numpy as np

# Header key a measurement's own identifier is written under.
#
# Until this existed the only way to name a measurement was the pair
# (Serial Num., RECORDED), which collides when two are taken inside the same
# second and carries no timezone. The id is generated once, at capture, and
# travels inside the file, so the same measurement stays recognisable wherever
# it ends up: re-exported, backed up to the cloud, or pulled onto a different
# machine.
#
# Shared deliberately. radicl writes it, the apps write it, and the sync API
# keys on it, so it should be spelled in exactly one place.
MEASUREMENT_ID_KEY = 'MEASUREMENT ID'


def new_measurement_id() -> str:
    """
    A fresh identifier for a measurement.

    Returns:
        str: A uuid4 in the usual hyphenated form
    """
    return str(uuid.uuid4())


def find_measurement_id(metadata: dict) -> Optional[str]:
    """
    The measurement id from a parsed header, if it carries one.

    Matched loosely on the key, since three clients write these files
    independently and an underscore or a different case should not lose the
    id. Files written before the key existed simply have none, which is why
    this returns None rather than inventing one — a caller that needs an id
    for such a file should mint it once and write it back, not derive a fresh
    one on every read.

    Args:
        metadata: Header dictionary, as returned by find_metadata

    Returns:
        str: The id, or None when the header does not carry one
    """
    wanted = MEASUREMENT_ID_KEY.replace(' ', '')

    for key, value in metadata.items():
        if str(key).strip().upper().replace('_', '').replace(' ', '') == wanted:
            value = str(value).strip()

            return value or None

    return None

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

    df.to_csv(f, mode='a', index=write_index)
