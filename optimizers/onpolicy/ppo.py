import torch
import sys
sys.path.insert(1, '../../helpers/')

import utils
import networks
import settings
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

@dataclass
class PPO:
    state_dim: int
    action_dim: int
    train_config: dict = field(default_factory=dict)
    config_file: str = '../settings.json'
    settings_name: str = 'ppo'
    pi: nn.Module = None
    old_pi: nn.Module = None
    v: nn.Module = None
    ppo_opt_pi: optim.Adam = None
    ppo_opt_v: optim.Adam = None
    name: str = 'PPO'

    def __post_init__(self) -> None:
        """
        Load train configurations and
        initialize networks.
        """
        self.train_config = utils.import_config_settings(self.config_file, self.settings_name)
        self.pi = networks.PolicyNetwork(self.state_dim, self.action_dim, False).to(device)
        self.v = networks.ValueNetwork(self.state_dim).to(device)
        self.old_pi = networks.PolicyNetwork(self.state_dim, self.action_dim, False).to(device)
        self.old_pi.load_state_dict(self.pi.state_dict())
        self.ppo_opt_pi = torch.optim.Adam(self.pi.parameters(), lr=0.001)
        self.ppo_opt_v = torch.optim.Adam(self.v.parameters(), lr=0.001)

    def get_networks(self) -> List[nn.Module]:
        """
        Return the policy and
        value networks.
        """
        return [self.pi, self.v]
    
    def act(self, state) -> np.ndarray:
        """
        Act by taking the distribution from
        policy network and sampling from it.
        """
        self.pi.eval()
        state = FloatTensor(state)
        distb = self.pi(state)
        action = distb.sample().detach().cpu().numpy()
        return action
    
    def save_model(self, path: str) -> List[str]:
        """
        Save the policy and value networks.
        """
        torch.save(self.pi.state_dict(), path + 'pi.pth')
        torch.save(self.v.state_dict(), path + 'v.pth')
        return [path + 'pi.pth', path + 'v.pth']
    
    def load_model(self, path: str) -> None:
        """
        Load the policy and value networks.
        """
        self.pi.load_state_dict(torch.load(path + 'pi.pth'))
        self.v.load_state_dict(torch.load(path + 'v.pth'))

    def step(self, exp_obs, rets, advs, acts, *args, **kwargs):
        """
        Implement PPO step.
        """
        # Compute the value loss
        values = self.v(exp_obs)
        # Could also use mse for value_loss
        value_loss = F.mse_loss(values.squeeze(), rets.detach())
        self.old_pi.to(device)
        old_dist = self.old_pi(exp_obs)
        log_probs = old_dist.log_prob(acts)
        new_dist = self.pi(exp_obs)
        new_log_probs = new_dist.log_prob(acts)

        # Update old pi network params
        self.old_pi = type(self.pi)(self.state_dim, self.action_dim, False) # or PolicyNetwork(self.state_dim, self.action_dim, False) 
        self.old_pi.load_state_dict(self.pi.state_dict())

        # Get Policy Network Loss
        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - self.train_config.get('ppo_clip'), 1.0 + self.train_config.get('ppo_clip')) * advs
        policy_loss = -torch.min(surr1, surr2)

        self.ppo_opt_pi.zero_grad()
        policy_loss.mean().backward()
        self.ppo_opt_pi.step()

        self.ppo_opt_v.zero_grad()
        value_loss.backward()
        self.ppo_opt_v.step()

        
if __name__ == '__main__':
    # Test init
    tr = PPO(state_dim=5, action_dim=5)