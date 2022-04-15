import torch
import torch.nn.functional as F
import utils
import numpy as np
from torch.autograd import Variable
from classificationMAP import getClassificationMAP as cmAP
from detectionMAP import getDetectionMAP as dmAP

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def test(itr, dataset, args, model, device, run_name):
    done = False
    goal_instance_logits_stack = []
    goal_tcams_stack = []
    goal_labels_stack = []
    wentwrong_instance_logits_stack = []
    wentwrong_tcams_stack = []
    wentwrong_labels_stack = []
    goal_attn_stack = []
    wentwrong_attn_stack = []
    goal_outputs_stack = []
    wentwrong_outputs_stack = []

    final_goal_result = dict()
    final_wentwrong_result = dict()
    # final_result['version'] = 'VERSION 1.3'
    final_goal_result['results'] = {}
    final_wentwrong_result['results'] = {}
    # final_result['external_data'] = {'used': True, 'details': 'Features from I3D Net'}
    

    while not done:
        if dataset.currenttestidx % 100 ==0:
            print('Testing test data point %d of %d' %(dataset.currenttestidx, len(dataset.testidx)))

        features, pose_features, goal_labels, wentwrong_labels, done, vid_name = dataset.load_data(is_training=False,n_similar=args.num_similar)
        seq_len = torch.tensor([features.shape[0]])
        seq_len = torch.as_tensor(seq_len, dtype=torch.int64, device='cpu')

        features = torch.from_numpy(features).float().to(device)
        features = features.unsqueeze(0)
        pose_features = torch.from_numpy(pose_features).float().to(device)
        pose_features = pose_features.unsqueeze(0)
        with torch.no_grad():
            # goal_output,wentwrong_output,goal_tcams, wentwrong_tcams, goal_rgb_inp, goal_pose_inp, wentwrong_rgb_inp, wentwrong_pose_inp, goal_cams, wentwrong_cams = model(Variable(features),Variable(pose_features), seq_len, is_training=False)
            goal_output,wentwrong_output,goal_element_logits, wentwrong_element_logits, goal_inp, wentwrong_inp, goal_cams, wentwrong_cams = model(Variable(features),Variable(pose_features), seq_len, is_training=False)

            goal_tcams = goal_cams.squeeze()
            wentwrong_tcams = wentwrong_cams.squeeze()

        goal_tmp = F.softmax(torch.mean(torch.topk(goal_tcams, k=max(1,int(np.ceil(len(features)/args.k))), dim=0)[0], dim=0), dim=0).cpu().data.numpy()
        wentwrong_tmp = F.softmax(torch.mean(torch.topk(wentwrong_tcams, k=max(1,int(np.ceil(len(features)/args.k))), dim=0)[0], dim=0), dim=0).cpu().data.numpy()
        
        goal_tcams = goal_tcams.squeeze(0).cpu().data.numpy()
        wentwrong_tcams = wentwrong_tcams.squeeze(0).cpu().data.numpy()
        goal_cams = goal_cams.squeeze(0).cpu().data.numpy()
        wentwrong_cams = wentwrong_cams.squeeze(0).cpu().data.numpy()
        
    
        goal_instance_logits_stack.append(goal_tmp)
        goal_tcams_stack.append(goal_tcams)
        goal_labels_stack.append(goal_labels)
        goal_attn_stack.append(goal_inp.squeeze())
        goal_outputs_stack.append(goal_output.squeeze())

        wentwrong_instance_logits_stack.append(wentwrong_tmp)
        wentwrong_tcams_stack.append(wentwrong_tcams)
        wentwrong_labels_stack.append(wentwrong_labels)
        wentwrong_attn_stack.append(wentwrong_inp.squeeze())
        wentwrong_outputs_stack.append(wentwrong_output.squeeze())

    goal_instance_logits_stack = np.array(goal_instance_logits_stack)
    wentwrong_instance_logits_stack = np.array(wentwrong_instance_logits_stack)
    goal_labels_stack = np.array(goal_labels_stack)
    wentwrong_labels_stack = np.array(wentwrong_labels_stack)
    goal_attn_stack = np.array(goal_attn_stack)
    wentwrong_attn_stack = np.array(wentwrong_attn_stack)

    goal_dmap, goal_iou = dmAP(goal_tcams_stack, dataset.path_to_annotations, args, mode='goal', attn_stack=goal_attn_stack, remove_ixs=dataset.remove_ixs)
    wentwrong_dmap, wentwrong_iou = dmAP(wentwrong_tcams_stack, dataset.path_to_annotations, args, mode='wentwrong', attn_stack=wentwrong_attn_stack,remove_ixs=dataset.remove_ixs)
    
    avg_goal_dmap = np.mean(np.array(goal_dmap))
    avg_wentwrong_dmap = np.mean(np.array(wentwrong_dmap))
    goal_cmap = cmAP(goal_instance_logits_stack, goal_labels_stack)

    wentwrong_cmap = cmAP(wentwrong_instance_logits_stack, wentwrong_labels_stack)
    

    utils.write_to_file(run_name, goal_dmap, goal_cmap, itr, mode='goal')
    utils.write_to_file(run_name, wentwrong_dmap, wentwrong_cmap, itr, mode='wentwrong')
    return goal_cmap, wentwrong_cmap, avg_goal_dmap, avg_wentwrong_dmap

