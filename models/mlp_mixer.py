import torch
from torch import nn


class MlpBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = torch.dropout(x, self.dropout, train=self.training)
        x = self.gelu(self.fc2(x))
        return x


class MixerBlock(nn.Module):
    def __init__(self, patch_dim, channel_dim, token_hidden_dim, channel_hidden_dim, dropout=0.5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channel_dim)
        self.tokens_mlp_block = MlpBlock(in_dim=patch_dim, hidden_dim=token_hidden_dim, dropout=dropout)
        self.channels_mlp_block = MlpBlock(in_dim=channel_dim, hidden_dim=channel_hidden_dim, dropout=dropout)

    def forward(self, x):
        """
        x: (bs,tokens,channels)
        """
        ### tokens mixing
        y = self.layer_norm(x)
        y = y.transpose(0, 1)
        y = self.tokens_mlp_block(y)
        ### channels mixing
        y = y.transpose(0, 1)
        out = x + y
        y = self.layer_norm(out)
        y = self.channels_mlp_block(y)
        y = out + y
        return y


class MlpMixer(nn.Module):

    def __init__(self, patch_dim, channel_dim, num_blocks, token_hidden_dim, channel_hidden_dim, dropout=0.5):
        super().__init__()
        self.num_blocks = num_blocks  # num of mlp layers
        self.layer_norm = nn.LayerNorm(channel_dim)
        self.mlp_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_blocks.append(MixerBlock(
                patch_dim=patch_dim, channel_dim=channel_dim,
                token_hidden_dim=token_hidden_dim, channel_hidden_dim=channel_hidden_dim
            ))

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.mlp_blocks[i](x)
        x = self.layer_norm(x)
        x = torch.mean(x, dim=1, keepdim=False)
        x = (x - x.min()) / (x.max() - x.min())
        return x



if __name__ == '__main__':

    import os
    import sys
    sys.path.insert(0, os.path.abspath('../'))
    from common.utils import load_data

    pyg_data = load_data(name='cora')
    test_mask = pyg_data.test_mask
    graph_statistics = torch.load('../attack/cora-greedy-rbcd-gcn.pth')

    x = []
    for key in graph_statistics.keys():
        if not key.endswith('mask'):
            x.append(graph_statistics[key])
    x = torch.stack(x)
    x = torch.sigmoid(x).T
    x = x[test_mask]

    mlp_mixer = MlpMixer(patch_dim=1000, channel_dim=7, token_hidden_dim=2000, channel_hidden_dim=16, num_blocks=5)
    output = mlp_mixer(x)
    print(output.shape)