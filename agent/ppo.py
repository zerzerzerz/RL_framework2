import torch
from buffer.buffer import Buffer
from actor_critic.actor_critic import ActorCritic
from env.env import MyBaseEnv, MyEnv
from torch.nn import functional as F
import copy

class PPO(torch.nn.Module):
    def __init__(
            self,
            actor_critic: ActorCritic,
            optimizer: torch.optim.Optimizer,
            lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
            buffer: Buffer,
            device: str,
            # episode_len: int,
            env: MyEnv,
            clip_param: float,
            coef_loss_value: float,
            coef_loss_action: float,
            coef_loss_entropy: float,
            update_epoch: int,
            max_episode_len: int,
        ) -> None:
        super().__init__()
        self.actor_critic = actor_critic
        self.actor_critic_old = copy.deepcopy(actor_critic)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.buffer = buffer
        self.device = device
        # self.episode_len = episode_len
        self.env = env
        self.clip_param = clip_param
        self.coef_loss_value = coef_loss_value
        self.coef_loss_action = coef_loss_action
        self.coef_loss_entropy = coef_loss_entropy
        self.update_epoch = update_epoch
        self.max_episode_len = max_episode_len
        self.to(self.device)
    

    def collect_episode(self):
        with torch.no_grad():
            self.eval()
            self.buffer.clear()
            obs = self.env.reset()
            for _ in range(self.buffer.capacity):
                value, action, action_logprob = self.actor_critic_old.act(obs)
                next_obs, reward, done, info = self.env.step(action)
                self.buffer.insert(obs, value, reward, action, action_logprob, done)
                obs = self.env.reset() if done[0] else next_obs


    def update(self):
        epoch_loss_value = 0
        epoch_loss_action = 0
        epoch_loss_entropy = 0
        epoch_advantage = 0
        epoch_ratio = 0
        self.train()
        for epoch in range(self.update_epoch):
            dataloader = self.buffer.get_generator()
            for sample in dataloader:
                if self.buffer.compute_advantage_method == 'normal':
                    state, value, reward, action, \
                        old_action_logprob, done, return_ = sample
                else:
                    raise NotImplementedError
                
                new_value, new_action_logprob, entropy = self.actor_critic.evaluate(state, action)
                advantage = return_ - new_value

                loss_value = F.mse_loss(new_value, return_, reduction='mean')
                loss_entropy = -entropy.mean()
                ratio = torch.exp(new_action_logprob - old_action_logprob.detach())
                surr1 = ratio * advantage
                surr2 = torch.clip(ratio, 1-self.clip_param, 1+self.clip_param) * advantage
                objective = torch.min(surr1, surr2)
                loss_action = -objective.mean()

                loss = self.coef_loss_value * loss_value + \
                    self.coef_loss_action * loss_action + \
                        self.coef_loss_entropy * loss_entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss_value += loss_value.item()
                epoch_loss_action += loss_action.item()
                epoch_loss_entropy += loss_entropy.item()
                epoch_advantage += advantage.mean().item()
                epoch_ratio += ratio.mean().item()
                
        self.lr_scheduler.step()
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())
        res = {
            "loss_value": epoch_loss_value,
            "loss_action": epoch_loss_action,
            "loss_entropy": epoch_loss_entropy,
            "advantage": epoch_advantage, 
            "ratio": epoch_ratio,
        }
        for k in res.keys():
            res[k] /= self.update_epoch
        
        return res


    def evaluate(self):
        with torch.no_grad():
            self.eval()
            obs = self.env.reset()
            done = False
            i = 0
            rewards = 0
            while not done and i < self.max_episode_len:
                value, action, action_prob = self.actor_critic.act(obs)
                obs, reward, done, info = self.env.step(action)
                rewards += reward.item()
                i += 1
            return rewards, i
                
                
                