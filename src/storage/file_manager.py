from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw")


def save_uploaded_dataset(
    df: pd.DataFrame,
    filename: str
):
    """
    Saves uploaded dataset locally.

    Args:
        df: uploaded dataframe
        filename: original filename
    """


    RAW_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )


    save_path = RAW_DATA_PATH / filename


    df.to_csv(
        save_path,
        index=False
    )


    return save_path