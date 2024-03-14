import torch
import sys
sys.path.insert(1, '../../helpers/')

import utils
import networks
import settings
import numpy as np
from torch import nn
from torch import optim
from dataclasses import dataclass, field
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

@dataclass
class TRPO:
    state_dim: int
    action_dim: int
    train_config: dict = field(default_factory=dict)
    config_file: str = '../settings.json'
    settings_name: str = 'trpo'
    pi: nn.Module = None
    v: nn.Module = None
    name: str = 'TRPO'

    def __post_init__(self) -> None:
        """
        Load train configurations and
        initialize networks.
        """
        self.train_config = utils.import_config_settings(self.config_file, self.settings_name)
        self.pi = networks.PolicyNetwork(self.state_dim, self.action_dim, False).to(device)
        self.v = networks.ValueNetwork(self.state_dim).to(device)

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
        Returns their respective paths.
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

    def step(self, exp_obs, rets, advs, acts, gms, *args, **kwargs):
        """
        Implement TRPO step
        """
        # Update value network
        self.v.train()
        old_params = utils.get_flat_parameters(self.v).detach()
        old_v = self.v(exp_obs).detach()
        def constraint():
            return ((old_v - self.v(exp_obs)) ** 2).mean()
        grad_diff = utils.get_flat_grads(constraint(), self.v)
        # Calculate heessian vector
        def Hv(v):
            return utils.get_flat_grads(torch.dot(grad_diff, v), self.v).detach()
        # Policy Gradient
        g = utils.get_flat_grads(((-1) * (self.v(exp_obs).squeeze() - rets) ** 2).mean(), self.v).detach()
        s = utils.conjugate_gradient(Hv, g).detach()
        Hs = Hv(s).detach()
        alpha = torch.sqrt(2 * self.train_config.get("eps") / torch.dot(s, Hs))
        new_params = old_params + alpha * s
        utils.set_params(self.v, new_params)

        # Update policy network
        self.pi.train()
        old_params = utils.get_flat_parameters(self.pi).detach()
        old_distb = self.pi(exp_obs)

        def L():
            distb = self.pi(exp_obs)
            return (advs * torch.exp(distb.log_prob(acts)- old_distb.log_prob(acts).detach())).mean()
        
        def kld():
            distb = self.pi(exp_obs)
            old_mean = old_distb.mean.detach()
            old_cov = old_distb.covariance_matrix.sum(-1).detach()
            mean = distb.mean
            cov = distb.covariance_matrix.sum(-1)
            return (0.5) * ((old_cov / cov).sum(-1) + (((old_mean - mean) ** 2) / cov).sum(-1) - self.action_dim + torch.log(cov).sum(-1) - torch.log(old_cov).sum(-1)).mean()

        grad_kld_old_param = utils.get_flat_grads(kld(), self.pi)

        def Hv(v):
            hessian = utils.get_flat_grads(torch.dot(grad_kld_old_param, v), self.pi).detach()
            return hessian + self.train_config.get('cg_damping') * v

        g = utils.get_flat_grads(L(), self.pi).detach()

        s = utils.conjugate_gradient(Hv, g).detach()
        Hs = Hv(s).detach()
        # Line search on surrogate loss and kl constraint
        new_params = utils.rescale_and_linesearch(g, s, Hs, self.train_config.get('max_kl'), L, kld, old_params, self.pi)

        disc_causal_entropy = ((-1) * gms * self.pi(exp_obs).log_prob(acts)).mean()
        grad_disc_causal_entropy = utils.get_flat_grads(
            disc_causal_entropy, self.pi
        )
        new_params += self.train_config.get('lambda') * grad_disc_causal_entropy

        utils.set_params(self.pi, new_params)

        
if __name__ == '__main__':
    # Test init
    tr = TRPO(state_dim=5, action_dim=5)