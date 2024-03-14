import os
import json
import torch
import functools
import tilemapbase
import numpy as np
import pandas as pd
import networks
import time
import datetime
import settings
from torch import nn
from settings import *
import matplotlib.pyplot as plt
from colorama import Fore, Style
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Tuple, Dict, Union, Callable, Optional, Any

device = "cuda" if torch.cuda.is_available() else "cpu"
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def keep_df_columns(df: pd.DataFrame, col_names:  List[str]) -> pd.DataFrame:
    """
    Filter the input DataFrame and retain only the columns specified.
    """
    if len(df.columns.intersection(col_names)) == 0:
        print(Fore.YELLOW + f'Warning: No columns from {col_names} were found in dataframe. Returning empty DataFrame.')
        Style.RESET_ALL
    return df[df.columns.intersection(col_names)]

def get_rows_until_value(df, column, value):
    """
    Extracts all rows of a dataframe until a
    specific value is encountered in a column.
    """
    mask = df[column] == value
    index = df.index[mask].tolist()[0] if any(mask) else len(df)
    return df.iloc[:index]

def import_config_settings(settings_path, settings_name):
    """
    Retrieves specific settings from a
    config json file.
    """
    with open(settings_path) as file:
        config = json.load(file)
    config1_settings = config[settings_name]
    return config1_settings

def flat_list(l: list) -> list:
    """
    Flattens a list.
    """
    return [item for sublist in l for item in sublist]

def net_weight_init(model: torch.nn.Module) -> None:
    """
    Takes a PyTorch model as input and initializes the weights of its linear layers.
    """
    classname = model.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)

def normalize_column(col: List[float]) -> Tuple[List[float], MinMaxScaler]:
    """
    Takes a list of float values col as input and performs min-max
    normalization on the column.
    """
    scaler = MinMaxScaler()
    col = flat_list(scaler.fit_transform(np.array(col).reshape(-1,1)))
    return col, scaler

def normalize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
    """
    Applies the normalize_column function to every column of a DataFrame and returns the
    normalized DataFrame and a dictionary containing column_name - scaler pairs.
    """
    normalized_df = df.copy()
    scaler_dict = {}

    for column in df.columns:
        normalized_col, scaler = normalize_column(df[column].tolist())
        normalized_df[column] = normalized_col
        scaler_dict[column] = scaler

    return normalized_df, scaler_dict

def get_flat_parameters(network: nn.Module) -> torch.Tensor:
    """
    Takes a PyTorch nn.Module object, network, as input and returns a
    flattened tensor containing all the trainable parameters of the network.
    """
    return torch.cat([param.view(-1) for param in network.parameters()])

def set_params(net: nn.Module, new_flat_params: torch.Tensor) -> None:
    """
    Updates the parameters of a neural network with new flat parameters.
    """
    start_idx = 0
    for param in net.parameters():
        end_idx = start_idx + np.prod(list(param.shape))
        param.data = torch.reshape(
            new_flat_params[start_idx:end_idx], param.shape
        )

        start_idx = end_idx

def get_flat_grads(f, net):
    """
    Returns the flattened gradients of a given
    function f with respect to the parameters
    of a neural network net. The gradients are
    computed using autograd and returned as a
    single flattened tensor.
    """
    flat_grads = torch.cat([
        grad.view(-1)
        for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
    ])

    return flat_grads

def split_dataframe(df: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
    """
    Splits a DataFrame into smaller
    chunks of a specified size.
    """
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def exclude_columns(series: pd.Series, columns_to_exclude: List[str]) -> pd.DataFrame:
    """
    Exclude specified columns from a Pandas DataFrame.
    """
    return series.drop(columns_to_exclude)

def policy_act(network : networks.PolicyNetwork, state : pd.Series) -> np.ndarray:
    """
    Takes a PolicyNetwork object and a state as input and returns an action.
    """
    network.to(device)
    network.eval()
    state = FloatTensor(state).to(device)
    distb = network(state)
    action = distb.sample().detach().cpu().numpy()
    return action

def transform_df(df: pd.DataFrame, scaler_dict: Dict[str, Any], inverse: bool = False) -> pd.DataFrame:
    """
    Transform the columns of a DataFrame using scalers from a scaler dictionary.
    """
    lists = []
    for c in df.columns:
        scaler = scaler_dict.get(c)
        if inverse:
            col = scaler.inverse_transform(df[c].values.reshape(-1,1)).flatten()
        else:
            col = scaler.transform(df[c].values.reshape(-1,1)).flatten()
        lists.append(col)
    return pd.DataFrame(np.array(lists).T, columns=df.columns)

def conjugate_gradient(Av_func: Callable, b: torch.Tensor, max_iter: int = 10, residual_tol: float = 1e-10) -> torch.Tensor:
    """
    Performs the conjugate gradient method to solve 
    a linear system of equations of the form Ax = b,
    where A is a linear operator, x is the unknown 
    vector, and b is the right-hand side vector.
    """
    x = torch.zeros_like(b)
    r = b - Av_func(x)
    p = r
    rsold = r.norm() ** 2

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x

def get_flat_gradients(f: Callable, network: torch.nn.Module) -> torch.Tensor:
    """
    Calculates and flattens the gradients of a given function f
    with respect to the parameters of a neural network network.
    """
    return torch.cat([grad.view(-1) for grad in torch.autograd.grad(f, network.parameters(), create_graph=True)])

def Hessian(gradient_difference: torch.Tensor, value_network: torch.nn.Module, v: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Hessian matrix approximation
    using the gradient difference method.
    """
    return get_flat_gradients(torch.dot(gradient_difference, v), value_network).detach()

def rescale_and_linesearch(
    g: torch.Tensor,
    s: torch.Tensor,
    Hs: torch.Tensor,
    max_kl: float,
    L: Callable[[], torch.Tensor],
    kld: Callable[[], torch.Tensor],
    old_params: torch.Tensor,
    pi: Any,
    max_iter: int = 10,
    success_ratio: float = 0.1
    ) -> torch.Tensor:
    """
    Performs a rescaling and line search optimization step.

    Args:
        g: The gradient vector.
        s: The search direction vector.
        Hs: The product of the Hessian matrix and the search direction.
        max_kl: The maximum KL divergence.
        L: A function that computes the objective function.
        kld: A function that computes the KL divergence.
        old_params: The old parameter values.
        pi: An object representing a policy.
        max_iter: The maximum number of iterations for the line search.
        success_ratio: The success ratio threshold for accepting the new parameters.

    Returns:
        The updated parameter values.
    """
    
    set_params(pi, old_params)
    L_old = L().detach()

    beta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))

    for _ in range(max_iter):
        new_params = old_params + beta * s

        set_params(pi, new_params)
        kld_new = kld().detach()

        L_new = L().detach()

        actual_improv = L_new - L_old
        approx_improv = torch.dot(g, beta * s)
        ratio = actual_improv / approx_improv

        if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
            return new_params

        beta *= 0.5

    print("The line search was failed!")
    return old_params

def get_list_steps(ls: List[float], padding: bool = True) -> List[float]:
    """
    Calculates the differences between consecutive
    elements in a given list.
    """
    if padding:
        return [ls[i+1] - ls[i] for i in range(len(ls[:-1]))] + [0]
    else:
        return [ls[i+1] - ls[i] for i in range(len(ls[:-1]))]
    
def add_step_column(df: pd.DataFrame, origin_column: str, target_column: str) -> pd.DataFrame:
    """
    Ddds a new column to a DataFrame that
    represents the differences between 
    consecutive elements of an existing column.
    """
    df[target_column] = get_list_steps(df[origin_column].values.tolist(), padding=True)
    return df

def apply_add_step_column(df: pd.DataFrame, origin_columns: List[str], target_columns: List[str]) -> pd.DataFrame:
    """
    Applies the add_step_column function
    to multiple columns in a DataFrame.
    """
    for origin_column, target_column in zip(origin_columns, target_columns):
        df = add_step_column(df, origin_column, target_column)

    return df

def dataframe_to_tensor(df: pd.DataFrame) -> torch.Tensor:
    """
    Converts a DataFrame into a PyTorch tensor of
    tensors with the values of each column.
    """
    tensor_list = []

    for column in df.columns:
        column_values = df[column].values
        tensor_list.append(torch.tensor(column_values))
    
    return torch.stack(tensor_list)

def create_cluster_dict(csv_file: str = BASE_DIR + DATA_PATH + 'test_clusters.csv') -> dict:
    """
    Reads a CSV file with cluster and folder columns,
    iterates over the rows, and creates a dictionary
    where the keys are cluster numbers and the values
    are lists of dictionaries with keys of the 
    corresponding folder names and the valid csv file
    names in them.
    """
    df = pd.read_csv(csv_file)
    cluster_dict = {}
    # Create cluster - folder_filename dict
    for _, row in df.iterrows():
        cluster = row['Cluster']
        folder = row['trajectory_ID']

        if cluster in cluster_dict:
            cluster_dict[cluster].append(folder)
        else:
            cluster_dict[cluster] = [folder]

    # Create folder_filenames dict
    def group_by_first_element(input_list: List[List]) -> Dict:
        """
        Groups the elements of a list
        of lists by their first element.
        """
        grouped_dict = {}
        for sublist in input_list:
            first_element = sublist[0]
            if first_element not in grouped_dict:
                grouped_dict[first_element] = []
            grouped_dict[first_element].append(sublist[1])
        return grouped_dict
    # Create folder_filenames dict

    for cl in list(cluster_dict.keys()):
        cluster_dict[cl] = group_by_first_element([item.split('_') for item in cluster_dict.get(cl)])

    return cluster_dict

def extract_ci_from_filename(filename: str) -> float:
    """
    Extracts the CI value from a given filename
    """
    ci_start = filename.find('CI_') + 3
    ci_end = filename.find('_', ci_start)
    ci_value = float(filename[ci_start:ci_end])
    return ci_value

def extract_mpl_from_filename(filename: str) -> float:
    """
    Extracts the MPL value from a given filename
    """
    mpl_start = filename.find('MPL_') + 4
    mpl_end = filename.find('_', mpl_start)
    mpl_value = float(filename[mpl_start:mpl_end])
    return mpl_value

def check_string_presence(strings: List[str], target_string: str) -> bool:
    """
    Checks if at least one string from a list of
    strings is present in a different target string.
    """
    for string in strings:
        if string in target_string:
            return True
    return False

def interpolate_dataframe(df: pd.DataFrame, time_column: str = settings.TIME_COLUMN, time_step: int = settings.INTERPOLATION_TIME_STEP) -> pd.DataFrame:
    """
    Interpolates a DataFrame with constant time steps
    """
    
    # Convert the time column to datetime type
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Set the time column as the index
    df.set_index(time_column, inplace=True)
    
    # Resample the dataframe with the desired time step
    resampled_df = df.resample(f'{time_step}S').first()
    
    # Drop duplicate labels in the time column
    resampled_df = resampled_df.loc[~resampled_df.index.duplicated(keep='first')]
    
    # Interpolate the resampled dataframe
    interpolated_df = resampled_df.interpolate(method='linear')
    
    # Reset the index
    interpolated_df.reset_index(inplace=True)

    # Set the terminal state column
    interpolated_df[settings.TERMINAL_STATE_COLUMN] = 0
    interpolated_df.loc[interpolated_df.index[-1], settings.TERMINAL_STATE_COLUMN] = 1
  
    return interpolated_df

def drop_terminal_states(df: pd.DataFrame, terminal_state_column: str = settings.TERMINAL_STATE_COLUMN) -> pd.DataFrame:
    """
    Drops the terminal states from a dataframe.
    """
    return df[df[terminal_state_column] == 0]

def combine_csv_by_id_ci_mpl(folder_path: str, ci_value: float, mpl_value: float, ids: List[str], interpolation: bool = True) -> pd.DataFrame:
    """
    Takes a folder path, a CI value, and an MPL value and a list of 
    valid ids in string format as inputs. It reads multiple CSV files from the 
    specified folder that match the CI and MPL values in their
    filenames and combines them into a single dataframe.
    It also inclueds a terminal_state column that
    informs us if this is the last state of a trajectory.
    """
    combined_df = pd.DataFrame()

    for filename in os.listdir(folder_path):
        
        if filename.endswith('.csv'):
            ci = extract_ci_from_filename(filename)
            mpl = extract_mpl_from_filename(filename)
            
            if ci == ci_value and mpl == mpl_value and check_string_presence(ids, filename):
                
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                df[settings.TERMINAL_STATE_COLUMN] = 0.0
                if interpolation: df = interpolate_dataframe(df)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        
    return combined_df


def create_dataframe_from_folders(folders: List[str], ci_value: float, mpl_value: float, traj_path: str) -> Optional[pd.DataFrame]:
    """
    Creates a dataframe by combining CSV files from
    multiple folders with a specific id, CI and MPL value.
    """
    dfs = []

    for folder_name in list(folders.keys()):
        try:
            combined_df = combine_csv_by_id_ci_mpl(traj_path + folder_name + '/', ci_value, mpl_value, ids = folders[folder_name])
            if combined_df is not None:
                dfs.append(combined_df)
        except:
            print(f'Folder {folder_name} was not found in {BASE_DIR + DATA_PATH + TRAJECTORIES_PATH} direcotry.')
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    else:
        print(f'No files found with CI={ci_value} and MPL={mpl_value} in the specified folders.')
        return None
    
def get_cluster_df(cluster: int, ci_value: float, mpl_value: float, cluster_csv_path: str = BASE_DIR + DATA_PATH + 'test_clusters.csv', trajectory_folder_path: str = BASE_DIR + DATA_PATH + 'trajectories/') -> pd.DataFrame:
    """
    Obtain a dataframe for a specific cluster, CI value,
    and MPL value from all the available trajectories.
    """
    cluster_dict = create_cluster_dict(cluster_csv_path)
    interpolated_dataframe = create_dataframe_from_folders(cluster_dict.get(cluster), ci_value, mpl_value, trajectory_folder_path)
    return interpolated_dataframe

def get_instances_by_ci_mpl(ci_value: float, mpl_value: float, cluster_dict: dict, rel_folders = List[str], trajectory_folder_path: str = BASE_DIR + DATA_PATH + 'trajectories/') -> Tuple[float, float, int]:
    """
    Get the number of csv instances for a specific CI and MPL value.
    """
    instances = 0
    for folder in rel_folders:
        try:
            ids = cluster_dict[folder]
            for filename in os.listdir(trajectory_folder_path + folder + '/'):
                if filename.endswith('.csv'):
                    ci = extract_ci_from_filename(filename)
                    mpl = extract_mpl_from_filename(filename)
                    if ci == ci_value and mpl == mpl_value and check_string_presence(ids, filename):
                        instances += 1
        except:
            pass
    return (ci_value, mpl_value, instances)

    
def get_largest_cluster_chunk(cluster_csv_path: str = BASE_DIR + DATA_PATH + 'test_clusters.csv', trajectory_folder_path: str = BASE_DIR + DATA_PATH + 'trajectories/') -> Tuple[float, float]:
    """
    Obtains the largest cluster and its corresponding CI and MPL values.
    """
    cluster_dict = create_cluster_dict(cluster_csv_path)
    largest_cluster = max(cluster_dict, key=lambda k: len(cluster_dict[k]))
    folders = [item for item in os.listdir(trajectory_folder_path) if os.path.isdir(os.path.join(trajectory_folder_path, item))]
    pair_instances_list = []
    c_value = 0
    while c_value < 100:
        m_value = 0.6
        while m_value < 1:
            pair_instances_list.append(get_instances_by_ci_mpl(ci_value=round(c_value, 1), mpl_value=round(m_value, 1), cluster_dict=cluster_dict[largest_cluster], rel_folders=folders, trajectory_folder_path=trajectory_folder_path))
            m_value += 0.1
        c_value += 2
    print(pair_instances_list)
    return largest_cluster, cluster_dict[largest_cluster]

def get_single_trajectory_plot(df:pd.DataFrame, lon_col_name:str, lat_col_name:str, save_dir: str = BASE_DIR + 'plots/default_plot.png',save_fig:bool = True):
    """
    Generates a trajectory plot based on longitude
    and latitude columns from a dataframe.
    """
    tilemapbase.init(create=True)
    expand=5
    extent = tilemapbase.Extent.from_lonlat(
        df[lon_col_name].min() - expand,
        df[lon_col_name].max() + expand,
        df[lat_col_name].min() - expand,
        df[lat_col_name].max() + expand,
    )
    trip_projected = df.apply(
        lambda x: tilemapbase.project(x[lon_col_name], x[lat_col_name]), axis=1
    ).apply(pd.Series)
    trip_projected.columns = ["x", "y"]
    tiles = tilemapbase.tiles.build_OSM()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plotter = tilemapbase.Plotter(extent, tiles, height=600)
    plotter.plot(ax, tiles, alpha=0.8)
    ax.plot(trip_projected.x, trip_projected.y, color='blue', linewidth=1)
    plt.axis('off')
    if save_fig:fig.savefig(save_dir, bbox_inches='tight',pad_inches=0, dpi=300)

    return fig

def plot_disc(exp_losses, agent_losses):
    """
    Plot the discriminator losses over epochs.
    """
    epochs = range(1, len(exp_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, exp_losses, label='Expert Discriminator Loss')
    plt.plot(epochs, agent_losses, label='Agent Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Discriminator Loss')
    plt.title('Discriminator Loss Progression Over Epochs')
    plt.legend()
    plt.show() 

def timeThis(func):
    """
    Times a function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_start = time. time()
        result = func(*args, **kwargs)
        print(f'Function {func.__name__} run in {round(time.time() - func_start, 5)} seconds.')
        return result
    return wrapper

def cacheThis(func):
    """
    Uses cache for reruns of the same function
    """
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(kwargs.items()))
        if key in cache:
            return cache[key]
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper

@cacheThis
def get_disc_values(constant, episode):
    return constant ** episode