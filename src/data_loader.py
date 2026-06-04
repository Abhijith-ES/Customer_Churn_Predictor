import pandas as pd
from pathlib import Path


def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)

def explore_data(df : pd.DataFrame):
    print(f'''
    FIRST 3 ROWS:
    {df.head(3)}

    LAST 3 ROWS:
    {df.tail(3)}

    DATAFRAME SHAPE:
    {df.shape[0]} ROWS x {df.shape[1]} COLS

    COLUMN NAMES:
    {df.columns}

    MISSING VALUES:
    {df.isnull().sum()}

    STATISTICAL INFO:
    {df.describe()}
    '''
    )

    print("DATATYPE INFORMATION: ")
    df.info()


    