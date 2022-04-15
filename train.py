import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def MILL(element_logits, seq_len, batch_size, labels, device, k):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over, 
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''

    t = np.ceil(seq_len/k).astype('int32')
    t[t==0]=1
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(batch_size):
        tmp, _ = torch.topk(element_logits[i][:seq_len[i]], k=int(t[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(Variable(labels) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def OverlapLoss(goal_attn, wentwrong_attn, seq_len, batch_size, device, args):

    overlap_loss = torch.zeros(0).to(device)
    for i in range(batch_size):
        goal_inp = goal_attn[i][:seq_len[i]]
        wentwrong_inp = wentwrong_attn[i][:seq_len[i]]

        if not args.no_overlap:
            #Calculating Goal Activated Indices
            goal_idx = torch.where(goal_inp > (torch.max(goal_inp)- 0.5*(torch.max(goal_inp)-torch.min(goal_inp))))
            goal_idx = torch.cat([goal_idx[0].unsqueeze(-1),goal_idx[1].unsqueeze(-1)],dim=1)
            if goal_idx.shape[0]!=0:
                overlap_loss = torch.cat([overlap_loss, (torch.max(torch.tensor(0).float(),torch.sum(wentwrong_inp[goal_idx[:,0],goal_idx[:,1]])/float(goal_idx.shape[0]) - seq_len[i]/args.p)).view(1)],dim=0)

            #Calculating UnInt Activated Indices
            wentwrong_idx = torch.where(wentwrong_inp > (torch.max(wentwrong_inp)- 0.5*(torch.max(wentwrong_inp)-torch.min(wentwrong_inp))))
            wentwrong_idx = torch.cat([wentwrong_idx[0].unsqueeze(-1),wentwrong_idx[1].unsqueeze(-1)],dim=1)
            # print(goal_idx.shape[0], wentwrong_idx.shape[0])
            if wentwrong_idx.shape[0]!=0:
                overlap_loss = torch.cat([overlap_loss, (torch.max(torch.tensor(0).float(),torch.sum(goal_inp[wentwrong_idx[:,0],wentwrong_idx[:,1]])/float(wentwrong_idx.shape[0]) - seq_len[i]/args.p)).view(1) ],dim=0)
        
        if not args.no_order:
            # Calculating order loss
            goal_inp_pmf = torch.softmax(goal_inp, dim=0).squeeze()
            goal_mean = torch.sum(goal_inp_pmf * torch.arange(seq_len[i]))

            wentwrong_inp_pmf = torch.softmax(wentwrong_inp, dim=0).squeeze()
            wentwrong_mean = torch.sum(wentwrong_inp_pmf * torch.arange(seq_len[i]))
            

            overlap_loss = torch.cat([overlap_loss, (torch.max(torch.tensor(0).float(),-(wentwrong_mean-goal_mean)/seq_len[i]  + seq_len[i]/args.q)).view(1)],dim=0)
            # print(overlap_loss[-3],overlap_loss[-2],overlap_loss[-1])

    return torch.sum(overlap_loss)/float(batch_size)

    


def train(itr, dataset, args, model, optimizer, device):

    features, pose_features, goal_labels, wentwrong_labels = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:,:np.max(seq_len),:]
    pose_features = pose_features[:,:np.max(seq_len),:]

    features = torch.from_numpy(features).float().to(device)
    pose_features = torch.from_numpy(pose_features).float().to(device)
    goal_labels = torch.from_numpy(goal_labels).float().to(device)
    wentwrong_labels = torch.from_numpy(wentwrong_labels).float().to(device)

    goal_output,wentwrong_output,goal_element_logits, wentwrong_element_logits, goal_rgb_inp, wentwrong_rgb_inp = model(Variable(features), Variable(pose_features), seq_len)
        
    goal_milloss = MILL(goal_element_logits, seq_len, args.batch_size, goal_labels, device, args.k)
    wentwrong_milloss = MILL(wentwrong_element_logits, seq_len, args.batch_size, wentwrong_labels, device, args.k)
    overlap_loss = OverlapLoss(goal_rgb_inp, wentwrong_rgb_inp, seq_len,args.batch_size, device, args)

    total_loss = args.Lambda*(goal_milloss + wentwrong_milloss) + (1-args.Lambda)* (overlap_loss)

    print('Iteration: %d, Loss: %.3f' %(itr, total_loss.data.cpu().numpy()))
    print('MillLoss: ',(goal_milloss+wentwrong_milloss).item())
    print('OverlapLoss: ',(overlap_loss).item())

    optimizer.zero_grad()
    total_loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    return total_loss