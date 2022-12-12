import gym
from gym import Space
import torch

class MyBaseEnv():
    def __init__(self, obs_space:Space, act_space:Space, device:str) -> None:
        assert issubclass(obs_space.__class__, Space)
        assert issubclass(act_space.__class__, Space)
        self.obs_space = obs_space
        self.act_space = act_space
        self.device = device
    
    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError


class MyEnv(MyBaseEnv):
    def __init__(self, obs_space, act_space, device) -> None:
        self.env = gym.make('LunarLander-v2')
        super().__init__(self.env.observation_space, self.env.action_space, device)
    
    def reset(self):
        '''return obs, shape is [1,...]'''
        obs, info = self.env.reset()
        obs = torch.from_numpy(obs).unsqueeze(0).to(dtype=torch.float32, device=self.device)
        return obs
    
    def step(self, action):
        '''
        Input:
            @action: shape == [1,...]
        Output:
            @obs: shape == [1,...], Tensor
            @reward: shape == [1,], Tensor 
            @done: bool
            @info: dict
        '''
        if self.act_space.__class__.__name__ == "Discrete":
            action = action.item()
        elif self.act_space.__class__.__name__ == "Box":
            action = action.squeeze(1).detach().cpu().numpy()
        else:
            raise NotImplementedError

        obs, reward, done, trunc, info = self.env.step(action)
        obs = torch.from_numpy(obs).unsqueeze(0).to(dtype=torch.float32, device=self.device)
        reward = torch.Tensor([reward]).to(self.device)
        done = torch.Tensor([done]).to(self.device)
        return obs, reward, done, info

