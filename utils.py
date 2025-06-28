import os 
import yaml 
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd



def load_config(file_path: str, encoding = "UTF-8") -> dict:
    """Load a YAML file and return its content."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding=encoding) as file:
        return yaml.safe_load(file)

def load_data(file_path: str, extension: str) -> pd.DataFrame:
    """Load data from a file using pandas based on the file extension.
    Args:
        file_path (str): Path to the data file.
        extension (str): File extension (e.g., 'csv', 'json').
    Returns:
        pd.DataFrame: Data loaded into a pandas DataFrame."""
    try:
        return getattr(pd, f"read_{extension}")(file_path)
    except Exception as e:
        raise Exception(
            f"Pandas v.{pd.__version__} doesn't support extension: {extension}"
        )


def pad_series(input_arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad or truncate an input array to match the target length.

    Parameters:
    - input_arr (numpy.ndarray): The input array to be padded or truncated.
    - target_len (int): The desired target length of the output array.

    Returns:
    - numpy.ndarray: The padded or truncated array.
    """
    curr_len = len(input_arr)

    if curr_len < target_len:
        # Pad the 2d series with edge values to match the target length
        pad_width = target_len - curr_len
        output_arr = np.pad(input_arr, ((0, pad_width), (0, 0)), mode="edge")
    else:
        # Truncate or return the input 2d series as is
        output_arr = input_arr[:target_len, :]

    return output_arr


def stratified_split(df_list: list, label_list: list, test_size: float, random_state: int) -> tuple[list, list, list, list]:
    """Perform a stratified split of the dataset into training and testing sets.
    Args:
        df_list (list): List of dataframes to be split.
        label_list (list): List of labels corresponding to the dataframes.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple: Four lists containing the training dataframes, testing dataframes, training labels, and testing labels.
    """
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idxes, test_idxes = next(sss.split(X=df_list, y=label_list))
    train_df_list = [df_list[i] for i in train_idxes]
    test_df_list = [df_list[i] for i in test_idxes]
    train_label_list = [label_list[i] for i in train_idxes]
    test_label_list = [label_list[i] for i in test_idxes]

    return train_df_list, test_df_list, train_label_list, test_label_list