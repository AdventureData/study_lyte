from typing.io import TextIO
from typing import Union
import pandas as pd


def read_csv(f: TextIO) -> Union[pd.DataFrame, dict]:
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

        return df, metadata


def write_csv(df: pd.DataFrame, meta: dict, f:str) -> None:
    """
    Write out the results with a header using the dictionary

    Args:
        df: Pandas Dataframe
        meta: Dictionary of information to write above the data as a header
        f: String path to write the data to
    """

    with open(f, 'w+') as fp:
        for k,v in meta.items():
            fp.write(f'{k} = {v}\n')

    df.to_csv(f, mode='a', index=False)
