from __future__ import print_function
import argparse
import os
import torch
from model import Model
from transnet import TransModel
from transnet_transformer import TransformModel
from transnet_joint_transformer import JointTransformModel
from stpn import TemporalProposal
from video_dataset import Dataset
from test import test
from train import train
import options
import torch.optim as optim
import json

torch.set_default_tensor_type('torch.cuda.FloatTensor')
if __name__ == '__main__':

    args = options.parser.parse_args()
    HYPERPARAMS_FIELDS = ['k','p','q','Lambda','lr','seed','max_seqlen','no_overlap','no_order','model_type']
    hyperparam_cfg = {}
    for param in HYPERPARAMS_FIELDS:
      hyperparam_cfg[param] = vars(args)[param]

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    
    dataset = Dataset(args)
    if not os.path.exists('./ckpt/'):
       os.makedirs('./ckpt/')
    if not os.path.exists('./previous-runs/'):
       os.makedirs('./previous-runs/')
    if not os.path.exists('./run-logs/'):
       os.makedirs('./run-logs/')
    
   #  model = Model(dataset.feature_size, dataset.num_goal_class, dataset.num_wentwrong_class).to(device)
    if args.model_type == 'gru':
      model = TransModel(dataset.feature_size, 3, dataset.feature_size, dataset.num_goal_class, dataset.num_wentwrong_class).to(device)
    elif args.model_type == 'transformer':
      model = TransformModel(dataset.feature_size, 3, dataset.feature_size, 8, dataset.num_goal_class, dataset.num_wentwrong_class).to(device)
   #  model = JointTransformModel(dataset.feature_size, 1, dataset.feature_size, 8, dataset.num_goal_class, dataset.num_wentwrong_class).to(device)
   #  model = TemporalProposal(dataset.feature_size, dataset.num_goal_class, dataset.num_wentwrong_class).to(device)

    if args.pretrained_ckpt is not None:
       model.load_state_dict(torch.load(args.pretrained_ckpt))
   #  model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_name = "-".join([args.model_name,
                    args.feature_type,
                    'no_overlap' if args.no_overlap else 'overlap',
                    'no_order' if args.no_order else 'order',
                     str(args.k), str(args.p), str(args.q), str(args.Lambda),
                     str(args.max_seqlen)
                   ])

    cur_run_config = {
       'Goal IoU': None,
       'UnInt IoU': None,
       'Avg IoU': None,
       'itr': None
    }

    for itr in range(args.max_iter):
       loss = train(itr, dataset, args, model, optimizer, device)
       if ((itr>3000 and itr % 500 == 0)) and not itr == 0:
          model.eval()
          goal_cmap, wentwrong_cmap, avg_goal_dmap, avg_wentwrong_dmap = test(itr, dataset, args, model, device, run_name)
          if cur_run_config['Avg IoU'] is None or (avg_goal_dmap+avg_wentwrong_dmap)/2.0 > cur_run_config['Avg IoU']:
             cur_run_config['Goal IoU'] = avg_goal_dmap
             cur_run_config['UnInt IoU'] = avg_wentwrong_dmap
             cur_run_config['Avg IoU'] = (avg_goal_dmap+avg_wentwrong_dmap)/2.0
             cur_run_config['itr'] = itr
             with open('./previous-runs/'+run_name+'.json','w') as f:
               f.truncate(0)
               json.dump(cur_run_config,f)
             torch.save(model.state_dict(), './ckpt/' + run_name + '.pkl')
          model.train()
    f.close()
    
