import pandas as pd


def validate_dataset(df: pd.DataFrame):
    """
    Validates uploaded dataset.

    Returns:
        (bool, message)
    """


    if df.empty:
        return False, "Dataset is empty."


    if df.shape[0] < 10:
        return False, "Dataset should contain at least 10 rows."


    if df.shape[1] < 2:
        return False, "Dataset should contain at least two columns."


    if df.columns.duplicated().any():
        return False, "Dataset contains duplicate column names."


    return True, "Dataset validation successful."