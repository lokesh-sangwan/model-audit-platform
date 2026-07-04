from sklearn.model_selection import train_test_split
import pandas as pd


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Splits dataset into training and testing sets.

    Args:
        df: Complete dataset
        target_column: Column to predict
        test_size: Percentage reserved for testing
        random_state: Reproducibility seed

    Returns:
        X_train, X_test, y_train, y_test
    """


    X = df.drop(
        columns=[target_column]
    )


    y = df[target_column]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


    return (
        X_train,
        X_test,
        y_train,
        y_test
    )