import pandas as pd


def get_dataset_overview(df: pd.DataFrame):
    """
    Generates basic dataset metadata.

    Args:
        df: Uploaded dataframe

    Returns:
        Dictionary containing dataset summary
    """

    overview = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }


    return overview



def get_column_details(df: pd.DataFrame):
    """
    Returns column level information.
    """


    details = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        }
    )


    return details