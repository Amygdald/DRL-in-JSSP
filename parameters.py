import argparse
parser = argparse.ArgumentParser(description='DRL-LSJSP')

# env parameters
parser.add_argument('--j', type=int, default=15)
parser.add_argument('--m', type=int, default=15)
parser.add_argument('--l', type=int, default=1)
parser.add_argument('--h', type=int, default=99)
parser.add_argument('--init_type', type=str, default='fdd-divide-mwkr')
parser.add_argument('--reward_type', type=str, default='yaoxin')
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--lmbda', type=float, default=0.95)
# model parameters
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--embedding_layer', type=int, default=4)
parser.add_argument('--policy_layer', type=int, default=4)
parser.add_argument('--embedding_type', type=str, default='gin+dghan')  # 'gin', 'dghan', 'gin+dghan'
parser.add_argument('--heads', type=int, default=1)  # dghan parameters
parser.add_argument('--drop_out', type=float, default=0.)  # dghan parameters
# training parameters
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--steps_learn', type=int, default=10)
parser.add_argument('--update_batch', type=int, default=10)
parser.add_argument('--update_times', type=int, default=3)
parser.add_argument('--transit', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--episodes', type=int, default=96000)
parser.add_argument('--sample_len', type=int, default=10)
parser.add_argument('--step_validation', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float,default=0.01)



args = parser.parse_args()