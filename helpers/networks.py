import torch
from torch import nn
from torch.distributions import Categorical, MultivariateNormal

device = "cuda" if torch.cuda.is_available() else "cpu"

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, discrete) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, action_dim),
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if not self.discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, states):
        if self.discrete:
            probs = torch.softmax(self.net(states), dim=-1)
            distb = Categorical(probs)
        else:
            mean = self.net(states)
            std = torch.exp(self.log_std)
            cov_mtx = torch.eye(self.action_dim).to(device) * (std ** 2)
            distb = MultivariateNormal(mean, cov_mtx)

        return distb
    
class ValueNetwork(nn.Module):
    def __init__(self, state_dim) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

    def forward(self, states):
        return self.net(states)
    
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, discrete) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if self.discrete:
            self.act_emb = nn.Embedding(
                action_dim, state_dim
            )
            self.net_in_dim = 2 * state_dim
        else:
            self.net_in_dim = state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(self.net_in_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        if self.discrete:
            actions = self.act_emb(actions.long())
        sa = torch.cat([states, actions], dim=-1)
        return self.net(sa)