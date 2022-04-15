import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

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

class TransModel(nn.Module):
    def __init__(self, emb_dim, num_layers, num_hidden_units, num_goals, num_wentwrongs):
        """
            emb_dim -> Embedding dimension of Input
            num_layers -> Number of LSTM layers
            num_hidden_units -> Number of hidden units (LSTM, GRU)
            num_goals -> Number of goal labels
            num_wentwrongs -> Number of went wrong labels
        """
        super(TransModel, self).__init__()
        num_hidden_units = num_hidden_units
        emb_dim = emb_dim
        self.num_layers = num_layers
        self.rgb_encoder = nn.GRU(
            input_size=emb_dim,
            hidden_size=num_hidden_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(p=0.5)
        self.inp_dropout = nn.Dropout(p=0.1)
        self.relu = m = nn.ReLU()
  
        self.goal_fc = nn.Conv1d(num_hidden_units*2, 256, kernel_size=1, stride=1, padding=0)
        self.wentwrong_fc = nn.Conv1d(num_hidden_units*2, 256, kernel_size=1, stride=1, padding=0)

        self.goal_atn = nn.Conv1d(256,1, kernel_size=1, stride=1, padding=0)
        self.wentwrong_atn = nn.Conv1d(256, 1, kernel_size=1, stride=1, padding=0)

        self.goal_classifier = nn.Linear(num_hidden_units*2, num_goals)
        self.wentwrong_classifier = nn.Linear(num_hidden_units*2, num_wentwrongs)

        self.sigmoid = nn.Sigmoid()

        self.num_hidden_units = num_hidden_units

    def forward(self, rgb_feats, pose_feats, vid_len, is_training=True):
        # rgb_feats = torch.cat([rgb_feats,pose_feats],dim=-1)
        vid_len = torch.as_tensor(vid_len, dtype=torch.int64, device='cpu')
        packed_rgb_input = hotfix_pack_padded_sequence(rgb_feats,vid_len, batch_first=True, enforce_sorted=False)
        packed_rgb_output, final_rgb_output= self.rgb_encoder(packed_rgb_input)
        rgb_output, _ = pad_packed_sequence(packed_rgb_output, batch_first=True)

        goal_rgb_inp = self.goal_fc(self.relu(rgb_output).permute(0,2,1))
        goal_rgb_inp = self.goal_atn(self.relu(goal_rgb_inp)).permute(0,2,1)
        goal_rgb_inp = self.sigmoid(goal_rgb_inp)

        wentwrong_rgb_inp = self.wentwrong_fc(self.relu(rgb_output).permute(0,2,1))
        wentwrong_rgb_inp = self.wentwrong_atn(self.relu(wentwrong_rgb_inp)).permute(0,2,1)
        wentwrong_rgb_inp = self.sigmoid(wentwrong_rgb_inp)

        goal_rgb_output = rgb_output*goal_rgb_inp
        wentwrong_rgb_output = rgb_output*wentwrong_rgb_inp

        if is_training:
            goal_rgb_output = self.dropout(goal_rgb_output)
            wentwrong_rgb_output = self.dropout(wentwrong_rgb_output)

        goal_rgb_logits = self.goal_classifier(goal_rgb_output)
        wentwrong_rgb_logits = self.wentwrong_classifier(wentwrong_rgb_output)

        if is_training:
            return goal_rgb_output, wentwrong_rgb_output, goal_rgb_logits, wentwrong_rgb_logits, goal_rgb_inp, wentwrong_rgb_inp
        else:
            return goal_rgb_output, wentwrong_rgb_output, goal_rgb_logits, wentwrong_rgb_logits, goal_rgb_inp, wentwrong_rgb_inp, self.goal_classifier(goal_rgb_output), self.wentwrong_classifier(wentwrong_rgb_output)
