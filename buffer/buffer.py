import torch
from torch import nn
from torch.utils.data import SubsetRandomSampler, BatchSampler
from gym import Space

class Buffer(nn.Module):
    def __init__(self, state_dim, action_space:Space, capacity, batch_size, gamma, lambda_, device, compute_advantage_method) -> None:
        super().__init__()
        self.register_buffer('states', torch.zeros(capacity, state_dim))
        self.register_buffer('values', torch.zeros(capacity, 1))
        self.register_buffer('rewards', torch.zeros(capacity, 1))
        self.register_buffer('action_logprobs', torch.zeros(capacity, 1))
        self.register_buffer('dones', torch.zeros(capacity, 1))
        self.register_buffer('returns', torch.zeros(capacity, 1))
        if action_space.__class__.__name__ == "Box":
            self.register_buffer('actions', torch.zeros(capacity, action_space.shape[0]))
        elif action_space.__class__.__name__ == "Discrete":
            self.register_buffer('actions', torch.zeros(capacity, 1))
        else:
            raise NotImplementedError

        self.size = 0
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambda_ = lambda_
        self.device = device
        self.compute_advantage_method = compute_advantage_method
        self.to(self.device)


    def insert(self, state, value, reward, action, action_logprob, done):
        self.states[self.size].copy_(state[0])
        self.values[self.size].copy_(value[0])
        self.rewards[self.size].copy_(reward)
        self.actions[self.size].copy_(action[0])
        self.action_logprobs[self.size].copy_(action_logprob)
        self.dones[self.size].copy_(done)
        self.size += 1
    

    def clear(self):
        torch.zero_(self.states)
        torch.zero_(self.values)
        torch.zero_(self.rewards)
        torch.zero_(self.actions)
        torch.zero_(self.action_logprobs)
        torch.zero_(self.dones)
        torch.zero_(self.returns)
        self.size = 0

    def is_full(self):
        return self.size == self.capacity

    def compute_return(self):
        if self.compute_advantage_method == 'gae':
            NotImplementedError
        elif self.compute_advantage_method == 'normal':
            discounted_reward = 0
            for i in reversed(range(self.capacity)):
                if self.dones[i]:
                    discounted_reward = 0
                discounted_reward = self.gamma * discounted_reward + self.rewards[i]
                self.returns[i].copy_(discounted_reward)
            self.returns = (self.returns - self.returns.mean()) / (self.returns.std() + 1e-6)
        else:
            NotImplementedError
    

    def get_generator(self):
        if self.compute_advantage_method == 'normal':
            self.compute_return()
            for i in BatchSampler(SubsetRandomSampler(range(self.capacity)), batch_size=self.batch_size, drop_last=True):
                yield self.states[i], self.values[i], self.rewards[i], self.actions[i], \
                    self.action_logprobs[i], self.dones[i], self.returns[i]
        else:
            raise NotImplementedError(f'compute_advantage_method = {self.compute_advantage_method} is not implemented')
        
