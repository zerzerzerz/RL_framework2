import argparse
from os.path import join
from utils.utils import get_datetime
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--output_dir', type=str, default='result-'+get_datetime())
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--max_episode_len', type=int, default=256)
    parser.add_argument('--state_dim', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--action_dim', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--buffer_capacity', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--update_epoch', type=int, default=10)
    parser.add_argument('--total_steps', type=int, default=int(1e7))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda_', type=float, default=0.99)
    parser.add_argument('--lr_actor', type=float, default=3e-4)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--lr_shared', type=float, default=5e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.998)
    parser.add_argument('--coef_loss_value', type=float, default=0.5)
    parser.add_argument('--coef_loss_action', type=float, default=1.0)
    parser.add_argument('--coef_loss_entropy', type=float, default=0.01)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--log_interval', type=int, default=300)
    parser.add_argument('--plot_interval', type=int, default=300)
    parser.add_argument('--compute_advantage_method', type=str, default='normal')
    
    args = parser.parse_args()

    args.ck_dir = join(args.output_dir, 'checkpoint')
    args.log_dir = join(args.output_dir, 'log')
    args.fig_dir = join(args.output_dir, 'fig')

    return args