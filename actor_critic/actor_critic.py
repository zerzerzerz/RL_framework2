import torch
from torch import nn
from model.model import MLP
from gym import Space
# from distributions.distributions import Categorical, MultivariateNormal
from torch.distributions import Categorical, MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(self, actor_critic_shared_model:nn.Module, obs_space:Space, act_space:Space, hidden_dim:int, init_std:float=1.0) -> None:
        super().__init__()
        assert issubclass(obs_space.__class__, Space)
        assert issubclass(act_space.__class__, Space)
        self.shared_model = actor_critic_shared_model
        self.obs_space = obs_space
        self.act_space = act_space
        self.hidden_dim = hidden_dim


        if self.act_space.__class__.__name__ == "Discrete":
            self.actor_header = nn.Sequential(
                MLP(hidden_dim, self.act_space.n, hidden_dim, 1),
                nn.Softmax(dim=-1)
            )
        elif self.act_space.__class__.__name__ == "Box":
            self.actor_header = MLP(hidden_dim, self.act_space.shape[0], hidden_dim, 1)
            self.register_buffer('cov_matrix', torch.diag(torch.full(self.act_space.shape[0], init_std)).unsqueeze(0))
        else:
            raise NotImplementedError
        
        self.critic_header = MLP(hidden_dim, 1, hidden_dim, 1)
    
    def forward(self):
        raise NotImplementedError
    

    def act(self, obs):
        '''return value, action, logprob'''
        shared_feature = self.shared_model(obs)

        value = self.critic_header(shared_feature)

        act_prob = self.actor_header(shared_feature)
        if self.act_space.__class__.__name__ == "Discrete":
            dist = Categorical(act_prob)
        elif self.act_space.__class__.__name__ == "Box":
            dist = MultivariateNormal(act_prob, self.cov_matrix)
        else:
            raise NotImplementedError
        action = dist.sample()
        logprob = dist.log_prob(action)

        return value, action, logprob
    
    
    def evaluate(self, obs, action):
        shared_feature = self.shared_model(obs)

        value = self.critic_header(shared_feature)

        act_prob = self.actor_header(shared_feature)
        if self.act_space.__class__.__name__ == "Discrete":
            dist = Categorical(act_prob)
        elif self.act_space.__class__.__name__ == "Box":
            dist = MultivariateNormal(act_prob, self.cov_matrix)
        else:
            raise NotImplementedError
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return value, logprob, entropy