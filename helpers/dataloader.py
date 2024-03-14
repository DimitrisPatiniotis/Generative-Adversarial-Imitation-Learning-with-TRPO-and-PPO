import os
from dataclasses import dataclass, field
import pandas as pd
from settings import *
import utils
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

@dataclass
class TensorDataset(Dataset):
    x_values: torch.Tensor
    y_values: torch.Tensor

    def __post_init__(self) -> None:
        """
        Transform the x_values and y_values
        from DataFrame into Tensors.
        """
        self.x_values = utils.dataframe_to_tensor(self.x_values)
        self.y_values = utils.dataframe_to_tensor(self.y_values)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.y_values[0])

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a specific item from the dataset given an index.
        Retrieves normalized input and output tensors at the specified index.
        """
        x = torch.stack([self.x_values[i][idx] for i in range(len(self.x_values))])
        y = torch.stack([self.y_values[i][idx] for i in range(len(self.y_values))])
        return x, y

@dataclass
class TrajDataloader:

    normalize: bool = True
    ci_value: float = 14.0
    mpl_value: float = 0.6
    cluster: int = 5
    include_terminal: bool = True
    cluster_csv_path: str = BASE_DIR + DATA_PATH + 'test_clusters.csv'
    traj_folder_path: str = BASE_DIR + DATA_PATH + TRAJECTORIES_PATH
    drop_terminal_states: bool = True

    raw_data: pd.DataFrame = pd.DataFrame()
    normalized_data: pd.DataFrame = pd.DataFrame()
    scalers: dict = field(default_factory=dict)
    tensor_dataset = TensorDataset
    dataloader = DataLoader

    def __post_init__(self) -> None:
        """
        Performs post-initialization operations
        including creating trajectories dataframe,
        applying interpolation, adding step columns,
        remove all terminal states (if enabled),
        normalizing data (if enabled), converting data to a tensor
        dataset, and creating a dataloader.
        """
        self.raw_data = self.create_trajectories_dataframe()
        self.raw_data = self.add_step_columns()
        if self.drop_terminal_states: self.raw_data = utils.drop_terminal_states(self.raw_data)
        if self.normalize:self.normalized_data, self.scalers = utils.normalize_dataframe(self.raw_data)
        self.tensor_dataset = self.load_data_to_torch_dataset()        
        self.dataloader = self.dataset_to_dataloader()

    def create_trajectories_dataframe(self) -> pd.DataFrame:
        """
        Creates a dataframe containing trajectories based
        on the specified cluster, ci and mpl parameters.
        """
        initial_dataframe = utils.get_cluster_df(cluster = self.cluster, ci_value = self.ci_value, mpl_value = self.mpl_value, cluster_csv_path=self.cluster_csv_path,trajectory_folder_path= self.traj_folder_path)
        if self.include_terminal:DEFAULT_X_COLUMNS.append('terminal_state')
        return utils.keep_df_columns(df = initial_dataframe, col_names = DEFAULT_X_COLUMNS)

    def create_demo_trajectory_plot(self) -> plt:
        """
        Creates a demo trajectory plot based on the first
        trajectory encountered in the raw data dataframe.
        """
        return utils.get_single_trajectory_plot(df = utils.get_rows_until_value(self.raw_data, 'terminal_state', 1), lon_col_name= 'Lon[º]', lat_col_name='Lat[º]')

    def get_one_trajectory(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a single trajectory in 2 forms.
        The first is the raw data dataframe, and
        the second is the normalized data dataframe.
        """
        return (utils.get_rows_until_value(self.raw_data, 'terminal_state', 1), utils.get_rows_until_value(self.normalized_data, 'terminal_state', 1))

    def get_scalers(self) -> dict:
        """
        Returns the scalers dictionary.
        """
        return self.scalers

    def add_step_columns(self) -> pd.DataFrame:
        """
        Adds a step column to the raw data dataframe
        for each of the y columns for easier computation.
        """
        return utils.apply_add_step_column(self.raw_data, [i for i in DEFAULT_Y_COLUMNS.keys()], [DEFAULT_Y_COLUMNS.get(i) for i in DEFAULT_Y_COLUMNS.keys()])


    def load_data_to_torch_dataset(self) -> TensorDataset:
        """
        Loads the normalized data into our custom
        TensorDataset class.
        """
        return TensorDataset(self.normalized_data[DEFAULT_X_COLUMNS], self.normalized_data[DEFAULT_Y_COLUMNS.values()])

    def dataset_to_dataloader(self) -> DataLoader:
        """
        Converts the tensor dataset
        into a PyTorch DataLoader.
        """
        return DataLoader(self.tensor_dataset, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == '__main__':
    td = TrajDataloader()
    print(td.raw_data)