from arguments.arguments import get_args
import utils.utils as utils
import utils.log as log
import utils.plot as plot
from os.path import join
import pandas as pd
from buffer.buffer import Buffer
from actor_critic.actor_critic import ActorCritic
from model.model import MLP
from env.env import MyEnv
from agent.ppo import PPO
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

def main(args):
    utils.mkdir(args.output_dir)
    utils.mkdir(args.ck_dir)
    utils.mkdir(args.log_dir)
    utils.mkdir(args.fig_dir)
    utils.save_json(vars(args), join(args.output_dir, 'args.json'))
    utils.setup_seed(args.seed)

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    
    env = MyEnv(None, None, args.device)
    buffer = Buffer(args.state_dim, env.act_space, args.buffer_capacity, args.batch_size, args.gamma, args.lambda_, args.device, args.compute_advantage_method)
    actor_critic_shared_model = MLP(args.state_dim, args.hidden_dim, args.hidden_dim, args.num_layers)
    actor_critic = ActorCritic(actor_critic_shared_model, env.obs_space, env.act_space, args.hidden_dim)
    optimizer = Adam([
        {
            "params": actor_critic.actor_header.parameters(),
            "lr": args.lr_actor,
        },
        {
            "params": actor_critic.critic_header.parameters(),
            "lr": args.lr_critic,
        },
        {
            "params": actor_critic.shared_model.parameters(),
            "lr": args.lr_shared,
        }
    ])
    lr_scheduler = ExponentialLR(optimizer, args.lr_decay_rate, verbose=False)
    agent = PPO(actor_critic, optimizer, lr_scheduler, buffer, args.device, env, args.clip_param, args.coef_loss_value, args.coef_loss_action, args.coef_loss_entropy, args.update_epoch, args.max_episode_len)

    for epoch in tqdm(range(args.total_steps // args.buffer_capacity)):
        agent.collect_episode()
        log_dict = agent.update()
        episode_reward, num_steps = agent.evaluate()

        df_train = log.save_log(df_train, join(args.log_dir, 'train.csv'), epoch, args.log_interval, **log_dict)
        df_test = log.save_log(df_test, join(args.log_dir, 'test.csv'), epoch, args.log_interval, episode_reward=episode_reward, num_steps=num_steps)
        plot.draw_and_save(df_train, epoch, args.plot_interval, args.fig_dir, *log_dict.keys())
        plot.draw_and_save(df_test, epoch, args.plot_interval, args.fig_dir, 'episode_reward', 'num_steps')



if __name__ == "__main__":
    args = get_args()
    main(args)