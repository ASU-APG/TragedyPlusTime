import argparse

parser = argparse.ArgumentParser(description='WTALC')
parser.add_argument('--lr', type=float, default=0.0001,help='learning rate (default: 0.0001)')
parser.add_argument('--k', type=int, default=2,help='MILL parameter')
parser.add_argument('--p', type=float, default=100,help='Overlap Loss parameter')
parser.add_argument('--q', type=float, default=80,help='Order Loss parameter')
parser.add_argument('--batch-size', type=int, default=16, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--model-name', default='transmodel', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature-size', default=1024, type=int, help='size of feature (default: 1024)')
parser.add_argument('--dataset-name', default='Oops', help='dataset to train on (default: )')
parser.add_argument('--max-seqlen', type=int, default=100, help='maximum sequence length during training (default: 30)')
parser.add_argument('--Lambda', type=float, default=0.8, help='Weight Tradeoff parameter')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-iter', type=int, default=15000, help='maximum iteration to train (default: 10000)')
parser.add_argument('--model-type', type=str, default='gru', help='Video Embedding Module')
parser.add_argument('--feature-type', type=str, default='i3d', help='type of feature to be used I3D or R3D (default: I3D)')
parser.add_argument('--run-name', type=str, default='transmodel', help='Specify run name')
parser.add_argument('--no-overlap', action='store_true', help='Remove overlap loss (Default: False)')
parser.add_argument('--no-order',action='store_true',help='Remove order loss (Default: False)')
parser.add_argument('--num-similar', default=3, type=int,help='number of similar pairs in a batch of data  (default: 3)')


