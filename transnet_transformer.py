import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from transformer_encoder import TransformerEncoderModel, TransformerDecoderModel, SAN

def hotfix_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices)

class TransformModel(nn.Module):
    def __init__(self, emb_dim, num_layers, num_hidden_units, n_head, num_goals, num_wentwrongs):
        """
            emb_dim -> Embedding dimension of Input
            num_layers -> Number of LSTM/Transformer layers
            num_hidden_units -> Number of hidden units (LSTM, GRU)
            num_goals -> Number of goal labels
            num_wentwrongs -> Number of went wrong labels
        """
        super(TransformModel, self).__init__()
        self.num_layers = num_layers
        self.goal_encoder = TransformerEncoderModel(emb_dim, n_head, 4*emb_dim, num_layers, 0.1)

        self.wentwrong_encoder = TransformerDecoderModel(emb_dim, n_head, 4*emb_dim, num_layers, 0.1 )

        self.dropout = nn.Dropout(p=0.5)
        self.relu = m = nn.ReLU()
        # self.goal_fc = nn.Linear(num_hidden_units,256)
        self.goal_fc = nn.Conv1d(num_hidden_units, 256, kernel_size=1, stride=1, padding=0)
        self.context_encoder = nn.Linear(emb_dim*2, emb_dim)
        # self.wentwrong_fc = nn.Linear(num_hidden_units, 256)
        self.wentwrong_fc = nn.Conv1d(num_hidden_units, 256, kernel_size=1, stride=1, padding=0)
        # self.goal_atn = nn.Linear(256,1)
        self.goal_atn = nn.Conv1d(256,1, kernel_size=1, stride=1, padding=0)
        # self.wentwrong_atn = nn.Linear(256,1)
        self.wentwrong_atn = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)
        self.goal_classifier = nn.Linear(num_hidden_units, num_goals)
        self.wentwrong_classifier = nn.Linear(num_hidden_units, num_wentwrongs)
        self.sigmoid = nn.Sigmoid()
        self.attn = nn.MultiheadAttention(emb_dim,n_head, dropout=0.2)
        self.num_hidden_units = num_hidden_units
    
    def forward(self, vid_feats, vid_len, is_training=True):
        vid_len = torch.as_tensor(vid_len, dtype=torch.int64, device='cpu')
        maxlen = vid_feats.size(1)
        
        # with torch.no_grad():
        idx = torch.arange(maxlen).unsqueeze(0).expand((vid_feats.size(0), vid_feats.size(1)))
        # print(idx)
        len_expanded = vid_len.unsqueeze(1).expand((vid_feats.size(0), vid_feats.size(1))).cuda()
        # print(len_expanded)
        src_key_padding_mask = (idx >= len_expanded).bool()

        # print(mask)
        # packed_input = hotfix_pack_padded_sequence(vid_feats,vid_len, batch_first=True, enforce_sorted=False)
        # packed_goal_output, final_goal_output = self.goal_encoder(packed_input)
        # final_goal_output = torch.cat([final_goal_output[:self.num_layers,:,:], final_goal_output[self.num_layers:,:,:]], dim=2)
        # final_goal_output = self.relu(self.context_encoder(self.relu(final_goal_output)))
        # final_goal_output = final_goal_output.view(self.num_layers*2, -1, self.num_hidden_units )
        # final_goal_output = final_goal_output.view(self.num_layers*2, )
        # print(_.shape)
        # goal_output, _ = pad_packed_sequence(packed_goal_output, batch_first=True)
        # goal_output = self.context_encoder(goal_output)
        # contextualized_input = hotfix_pack_padded_sequence(torch.cat([vid_feats, goal_output], dim=2), vid_len, batch_first=True, enforce_sorted=False)
        # print(contextualized_input.shape)
        
        fused_output = self.goal_encoder(vid_feats, src_key_padding_mask).permute(1,0,2)*~src_key_padding_mask.unsqueeze(-1)
        # new_padding_mask[torch.where(new_padding_mask==True)] = float('-inf')
        # new_padding_mask[new_padding_mask==False] = float(1)
        # print(vid_feats[0])
        # print(fused_output[0])
        # print(vid_len[0])
        # print(fused_output[0].shape, vid_feats[0].shape)
    
        # goal_output = self.goal_encoder(vid_feats, src_key_padding_mask))

        # wentwrong_output = 
        # final_goal_states = goal_output[range(len(goal_output)), vid_len - 1]
        # print(final_goal_states.shape)
        # packed_wentwrong_output,_ = self.wentwrong_encoder(packed_input, final_goal_states.unsqueeze(0))
        # packed_wentwrong_output,_ = self.wentwrong_encoder(contextualized_input)
        # wentwrong_output, _  = pad_packed_sequence(packed_wentwrong_output, batch_first=True)
        # contextual_input = torch.cat([vid_feats*(1-goal_inp), goal_output], dim=-1)
        # print()
        # print(contextual_input.shape)
        # wentwrong_output = self.wentwrong_encoder(contextual_input, src_key_padding_mask).permute(1,0,2)
        # print(fused_output.shape)
        goal_inp = self.goal_fc(self.relu(fused_output).permute(0,2,1))
        # print(goal_inp.shape)
        goal_inp = self.goal_atn(self.relu(goal_inp)).permute(0,2,1)
        goal_inp = self.sigmoid(goal_inp)
        # goal_output = fused_output*goal_inp
        # og_wentwrong_output = self.wentwrong_encoder(vid_feats, goal_output, src_key_padding_mask, src_key_padding_mask).permute(1,0,2)
        # og_wentwrong_output = self.wentwrong_encoder(vid_feats, src_key_padding_mask).permute(1,0,2)*~src_key_padding_mask.unsqueeze(-1)
        # fused_output = torch.cat([goal_output, wentwrong_output], dim=-1)
        # fused_output = self.context_encoder(self.relu(fused_output))
        # print(wentwrong_output.shape)


        wentwrong_inp = self.wentwrong_fc(self.relu(fused_output).permute(0,2,1))
        wentwrong_inp = self.wentwrong_atn(self.relu(wentwrong_inp)).permute(0,2,1)
        wentwrong_inp = self.sigmoid(wentwrong_inp)


        # out_reverse = output[:, 0, self.num_hidden_units:]
        # out_reduced = torch.cat((out_forward, out_reverse), 1)
        # print(goal_output.shape, goal_inp.shape)
        # print(wentwrong_output.shape, wentwrong_inp.shape)
        # print(goal_inp[0], wentwrong_inp[0], vid_len[0])
     
        # if is_training:
        #     goal_output = self.dropout(goal_output)
        #     wentwrong_output = self.dropout(wentwrong_output)
        # print(goal_inp[0].squeeze())
        # print(fused_output[0])
        # wentwrong_output = fused_output*wentwrong_inp
        # print(goal_output[0])
        # if is_training:
        #     fused_output =
            # goal_output = self.dropout(goal_output)
            # wentwrong_output = self.dropout(wentwrong_output)
        goal_logits = self.goal_classifier(fused_output * goal_inp)
        wentwrong_logits = self.wentwrong_classifier(fused_output * wentwrong_inp)
        # goal_logits = goal_logits
        # wentwrong_logits = wentwrong_logits

        # print(goal_logits.shape)
        # print(wentwrong_logits.shape)
        if is_training:
            return fused_output, fused_output, goal_logits, wentwrong_logits, goal_inp, wentwrong_inp
        else:
            return fused_output, fused_output, goal_logits, wentwrong_logits, goal_inp, wentwrong_inp,self.goal_classifier(fused_output), self.wentwrong_classifier(fused_output)

class GoalEncoder(nn.Module):
    def __init__(self, emb_dim, num_layers, num_hidden_units, n_head, n_class):
        super(GoalEncoder,self).__init__()
        self.num_layers = num_layers
        self.encoder = TransformerEncoderModel(emb_dim, n_head, 4*emb_dim, num_layers, 0.1)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = m = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(num_hidden_units,256)
        self.fc1 = nn.Linear(256,1)
        self.classifier = nn.Linear(num_hidden_units, n_class)
        self.attn = nn.MultiheadAttention(emb_dim,n_head, dropout=0.2)
        self.num_hidden_units = num_hidden_units
    
    def forward(self, vid_feats, vid_len, is_training=True):
        vid_len = torch.as_tensor(vid_len, dtype=torch.int64, device='cpu')
        maxlen = vid_feats.size(1)
        idx = torch.arange(maxlen).unsqueeze(0).expand((vid_feats.size(0), vid_feats.size(1)))
        len_expanded = vid_len.unsqueeze(1).expand((vid_feats.size(0), vid_feats.size(1))).cuda()
        src_key_padding_mask = (idx >= len_expanded).bool()

        og_feats = self.encoder(vid_feats, src_key_padding_mask).permute(1,0,2)*~src_key_padding_mask.unsqueeze(-1)

        inp = self.fc(self.relu(og_feats))
        inp = self.fc1(self.relu(inp))
        inp = self.sigmoid(inp)
        feats = og_feats*inp

        if is_training:
            feats = self.dropout(feats)

        logits = self.classifier(feats)

        if is_training:
            return feats, logits, inp, src_key_padding_mask
        else:
            return feats, logits, inp, self.classifier(og_feats)

class UnintEncoder(nn.Module):
    def __init__(self, emb_dim, num_layers, num_hidden_units, n_head, n_class):
        super(UnintEncoder,self).__init__()
        self.num_layers = num_layers
        self.encoder = TransformerDecoderModel(emb_dim, n_head, 4*emb_dim, num_layers, 0.1)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = m = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(num_hidden_units,256)
        self.fc1 = nn.Linear(256,1)
        self.classifier = nn.Linear(num_hidden_units, n_class)
        self.attn = nn.MultiheadAttention(emb_dim,n_head, dropout=0.2)
        self.num_hidden_units = num_hidden_units
    
    def forward(self, memory, vid_feats, vid_len, src_key_padding_mask,is_training=True):
        # vid_len = torch.as_tensor(vid_len, dtype=torch.int64, device='cpu')
        # maxlen = vid_feats.size(1)
        # idx = torch.arange(maxlen).unsqueeze(0).expand((vid_feats.size(0), vid_feats.size(1)))
        # len_expanded = vid_len.unsqueeze(1).expand((vid_feats.size(0), vid_feats.size(1))).cuda()
        # src_key_padding_mask = (idx >= len_expanded).bool()

        og_feats = self.encoder(vid_feats, memory, src_key_padding_mask, src_key_padding_mask).permute(1,0,2)

        inp = self.fc(self.relu(og_feats))
        inp = self.fc1(self.relu(inp))
        inp = self.sigmoid(inp)
        feats = og_feats*inp

        if is_training:
            feats = self.dropout(feats)

        logits = self.classifier(feats)

        if is_training:
            return feats, logits, inp
        else:
            return feats, logits, inp, self.classifier(og_feats)



