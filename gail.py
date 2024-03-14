import sys
sys.path.insert(1, 'helpers/')
sys.path.insert(1, 'optimizers/onpolicy/')
from ppo import PPO
from trpo import TRPO
import os
import dataloader
import utils
import settings
import networks
import torch
import argparse
import numpy as np
import pandas as pd
from time import time
from torch import optim
from sklearn.preprocessing import MinMaxScaler
from statistics import fmean
from datetime import datetime
from typing import Optional, Union, Tuple, Dict
from dataclasses import dataclass

device = "cuda" if torch.cuda.is_available() else "cpu"
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

@dataclass
class GAIL():
    state_dim: int
    action_dim: int
    epochs: int
    optimizer_name: str
    dataloader: dataloader.TrajDataloader
    discrete: bool = False
    normalize_advantage: bool = False
    save_model: bool = False
    should_load_model: bool = False
    load_model_path: str = None
    model_save_note: str = None
    batch_size: int = settings.BATCH_SIZE
    optimizer: Optional[Union[TRPO, PPO]] = None
    discriminator: networks.Discriminator = None
    discriminator_optimizer: optim.Adam = None

    def __post_init__(self) -> None:
        self.optimizer = self.load_optimizer()
        self.discriminator = networks.Discriminator(state_dim = self.state_dim, action_dim = self.action_dim, discrete = self.discrete).to(device)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)
        if self.should_load_model:
            self.load_model()

    def load_optimizer(self) -> Optional[Union[TRPO, PPO]]:
        """
        Loads and returns an optimizer object based
        on the specified optimizer_name.
        """
        try:
            if self.optimizer_name == 'TRPO':
                return TRPO(state_dim = self.state_dim, action_dim = self.action_dim, config_file='optimizers/settings.json')
            elif self.optimizer_name == 'PPO':
                return PPO(state_dim = self.state_dim, action_dim = self.action_dim, config_file='optimizers/settings.json')
        except:
            print(f'{self.optimizer_name} is not a valid optimizer name.')
            return None
    
    def save_model(self) -> None:
        """
        Saves the optimizer models and
        gail discriminator.
        """
        self.optimizer.save_model(f'./models/{self.optimizer_name}/')
        torch.save(self.discriminator.state_dict(), f'./models/{self.optimizer_name}/discriminator.pt')
        with open(f'./models/{self.optimizer_name}/model_save_notes.txt', 'w') as f:
            f.write(self.model_save_note)

    def load_model(self) -> None:
        """
        Loads the optimizer models and
        gail discriminator.
        """
        self.optimizer.load_model(f'./models/{self.optimizer_name}/')
        self.discriminator.load_state_dict(torch.load(f'./models/{self.optimizer_name}/discriminator.pt'))

    def train_loop(self):
        exp_obs_batched = [FloatTensor(np.array(i.values.tolist())) for i in utils.split_dataframe(self.dataloader.normalized_data[settings.DEFAULT_X_COLUMNS], self.batch_size)]
        exp_acts_batched = [FloatTensor(np.array(i.values.tolist())) for i in utils.split_dataframe(self.dataloader.normalized_data[settings.DEFAULT_Y_COLUMNS.keys()], self.batch_size)]

        save_model_folder = f'./models/{self.optimizer_name}/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}/'
        current_saved_model_paths = []

        act_loss_criterion = torch.nn.MSELoss()
        loss_iter_means = []
        exp_d_losses = []
        agent_d_losses = []

        for e in range(self.epochs):
            loss_iter_epoch = []
            epoch_start = time()
            ep_exp_d_losses = []
            ep_agent_d_losses=[]
            for i in range(len(exp_obs_batched)):
                loss_iter = []
                obs = exp_obs_batched[i]
                rets = []
                advs = []
                acts = []
                gms = []
                lmbs = []
                t = 0

                acts = [np.array(row, dtype=np.float32) for row in self.optimizer.act(exp_obs_batched[i])]
                for l in range(len(exp_obs_batched[i])):
                    gms.append(utils.get_disc_values(self.optimizer.train_config.get('gae_gamma'), l))
                    lmbs.append(utils.get_disc_values(self.optimizer.train_config.get('gae_lambda'), l))
                    # for progress graph
                    loss_iter.append(act_loss_criterion(FloatTensor(np.array(acts[l])), exp_acts_batched[i][l]).item())
                    t += 1

                loss_iter_epoch.append(np.mean(loss_iter))
                acts = FloatTensor(np.array(acts))
                gms = FloatTensor(gms)
                lmbs = FloatTensor(lmbs)

                ep_costs = (-1) * torch.log(self.discriminator(exp_obs_batched[i], acts)).squeeze().detach()
                ep_disc_costs = gms * ep_costs
                ep_disc_rets = torch.flip(torch.cumsum(torch.flip(ep_disc_costs, dims=[0]), dim=0), dims=[0])
                rets = ep_disc_rets / gms

                self.optimizer.v.eval()
                curr_vals = self.optimizer.v(obs).detach()
                next_vals = torch.cat((self.optimizer.v(obs)[1:], FloatTensor([[0.]]))).detach()
                ep_deltas = ep_costs.unsqueeze(-1) + self.optimizer.train_config.get('gae_gamma') * next_vals - curr_vals
                advs = FloatTensor([((gms * lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:]).sum() for j in range(t)])

                if self.normalize_advantage:
                    advs = (advs - advs.mean()) / advs.std()

                self.discriminator.train()
                exp_scores = self.discriminator.get_logits(FloatTensor(exp_obs_batched[i]), FloatTensor(exp_acts_batched[i]))
                nov_scores = self.discriminator.get_logits(FloatTensor(exp_obs_batched[i]), acts)
                
                exp_d_loss = torch.nn.functional.binary_cross_entropy_with_logits(exp_scores, torch.zeros_like(exp_scores))
                nov_d_loss = torch.nn.functional.binary_cross_entropy_with_logits(nov_scores, torch.ones_like(nov_scores))

                ep_exp_d_losses.append(exp_d_loss.item())
                ep_agent_d_losses.append(nov_d_loss.item())

                self.discriminator.zero_grad()
                loss_d = exp_d_loss + nov_d_loss
                loss_d.backward()
                self.discriminator_optimizer.step()

                self.optimizer.step(exp_obs = exp_obs_batched[i], rets = rets, advs = advs, acts = acts, gms=gms)

            loss_iter_means.append(fmean(loss_iter_epoch))
            exp_d_losses.append(fmean(ep_exp_d_losses))
            agent_d_losses.append(fmean(ep_agent_d_losses))
            epoch_end = time()
            if e % 10 == 0 and e != 0:
                utils.plot_disc(exp_losses=exp_d_losses, agent_losses=agent_d_losses)
            print(f"Epoch {e +1} of {self.epochs} with {self.optimizer.name} optimizer - Ran in {round(epoch_end - epoch_start, 2)} seconds - Agent Loss: {round(loss_iter_means[-1], 6)}")
            if self.save_model and loss_iter_means[-1] == min(loss_iter_means):
                try:
                    for i in current_saved_model_paths:
                        os.remove(i)
                    current_saved_model_paths = self.optimizer.save_model(f'{save_model_folder}epoch_{e}_agent_loss_{str(loss_iter_means[-1]).replace(".", "_")}')
                except:
                    # For initial save
                    os.mkdir(save_model_folder)
                    current_saved_model_paths = self.optimizer.save_model(f'{save_model_folder}epoch_{e}_agent_loss_{str(loss_iter_means[-1]).replace(".", "_")}')
                    
                print(f"Saved model at epoch {e} with agent loss {loss_iter_means[-1]}")

        if self.save_model:
            self.save_model()
        return self.epochs, loss_iter_means, exp_d_losses, agent_d_losses



def create_model_trajectory(model_path: str, ci_value: float, mpl_value: float):
    raw_states, normalized_states, scalers, policy_net = load_network(model_path, ci_value, mpl_value)

    # Create Expert Trajectory
    expert_trajectory = utils.get_single_trajectory_plot(df = raw_states, lon_col_name = 'Lon[º]', lat_col_name = 'Lat[º]', save_dir='plots/demo.png')

    # Create Agent Trajectory
    agent_traj = normalized_states.head(1)
    row_num = 0
    dt_scalers = list(scalers.values())

    inv_abs_scalers = [scalers.get('Lon[º]'), scalers.get('Lat[º]'), scalers.get('FF[kg/h]')]
    while len(agent_traj) != len(normalized_states):
        actions = utils.policy_act(policy_net, utils.exclude_columns(agent_traj.iloc[row_num], ['dLon[º]', 'dLat[º]', 'dFF[kg/h]']))

        # Inverse transform the scaled actions to get unscaled actions
        unscaled_actions = [scalers.get(['dLon[º]', 'dLat[º]', 'dFF[kg/h]'][i]).inverse_transform(actions[i].reshape(-1,1)).item() for i in range(len(actions))]

        # Calculate unscaled absolute action values by adding unscaled actions to corresponding raw state values
        unscaled_abs_action_values = [raw_states[['Lon[º]', 'Lat[º]', 'FF[kg/h]']].iloc[row_num][i] + unscaled_actions[i] for i in range(len(unscaled_actions))]
        
        # Rescale the unscaled absolute action values using inverse absolute scalers
        rescaled_abs_action_values = [inv_abs_scalers[i].transform(unscaled_abs_action_values[i].reshape(-1,1)).item() for i in range(len(unscaled_abs_action_values))]
        
        # Update the 'Lon[º]' and 'Lat[º]' values of the next row with the rescaled absolute action values
        next_row = normalized_states.iloc[row_num+1]

        next_row_copy = next_row.copy() # avoid pandas warnings
        next_row_copy.loc['Lon[º]'], next_row_copy.loc['Lat[º]'] = rescaled_abs_action_values[0], rescaled_abs_action_values[1]
        agent_traj = pd.concat([agent_traj, next_row_copy.to_frame().T], ignore_index=True)
        row_num += 1

    agent_traj = utils.transform_df(agent_traj, scalers, inverse=True)
    utils.get_single_trajectory_plot(df = agent_traj, lon_col_name = 'Lon[º]', lat_col_name = 'Lat[º]', save_dir='plots/trpo.png')

def load_network(model_path: str, ci_value:float, mpl_value:float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, MinMaxScaler], networks.PolicyNetwork]:
    # Create Scalers
    dl = dataloader.TrajDataloader(ci_value = ci_value, mpl_value = mpl_value, cluster_csv_path = settings.DATA_PATH + 'test_clusters.csv', traj_folder_path= settings.DATA_PATH + settings.TRAJECTORIES_PATH, drop_terminal_states=False)
    raw_trajectory, normalized_trajectory = dl.get_one_trajectory()
    scalers = dl.get_scalers()

    # Load Model
    policy = networks.PolicyNetwork(state_dim=len(settings.DEFAULT_X_COLUMNS), action_dim=len(settings.DEFAULT_Y_COLUMNS), discrete=False)
    try:
        policy.load_state_dict(torch.load(model_path))
    except:
        print(f'Path of {model_path} is not valid. Please check the path and try again.')
        return None
    return raw_trajectory, normalized_trajectory, scalers, policy

def train_gail(ci_value: float, mpl_value: float) -> None:
    torch.autograd.set_detect_anomaly(True)
    dl = dataloader.TrajDataloader(ci_value = ci_value, mpl_value=mpl_value, cluster_csv_path = settings.DATA_PATH + 'test_clusters.csv', traj_folder_path= settings.DATA_PATH + settings.TRAJECTORIES_PATH)
    gail = GAIL(state_dim=len(settings.DEFAULT_X_COLUMNS), action_dim=len(settings.DEFAULT_Y_COLUMNS), dataloader = dl, epochs=3000, optimizer_name='TRPO', save_model=True)
    num_epochs, agent_loss_progretion, exp_d_losses, agent_d_losses = gail.train_loop()
    return num_epochs, agent_loss_progretion, exp_d_losses, agent_d_losses

def flow_handler() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ci', dest='ci_value', type=float, help='Select CI Value')
    parser.add_argument('--mpl', dest='mpl_value', type=float, help='Select MPL Value')
    parser.add_argument('--model_path', dest='model_path', type=str, help='Select Agent Model Path for Trajectory Creation')
    parser.add_argument('--mode', dest='mode', type=str, help='Select GAIL Module Mode')
    args = parser.parse_args()

    if args.mode == 'train':
        train_gail(args.ci_value, args.mpl_value)
    elif args.mode == 'create_trajectory':
        create_model_trajectory(args.model_path, float(args.ci_value), float(args.mpl_value))

if __name__ == '__main__':
    flow_handler()