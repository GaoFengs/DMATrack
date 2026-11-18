import math

import torch
import torch.nn as nn

class CrossLayerAttention(nn.Module):
    def __init__(self, channels=256, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Linear(channels, channels)
        self.kv_proj = nn.Linear(channels, channels * 2)

        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(768)

        self.input_norm = nn.LayerNorm(768)

        self._init_weights()

    def _init_weights(self):

        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.kv_proj.weight, gain=1 / math.sqrt(2))
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.kv_proj.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, deep_feat, shallow_feats):

        deep_feat = self.input_norm(deep_feat)
        shallow_feats = [self.input_norm(feat) for feat in shallow_feats]

        bn, c, l = deep_feat.shape

        query = self.q_proj(deep_feat.transpose(1, 2))

        shallow_all = torch.cat(shallow_feats, dim=2)
        k, v = self.kv_proj(shallow_all.transpose(1, 2)).chunk(2, dim=-1)

        query = query.view(bn, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bn, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = v.view(bn, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(query, k) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(bn, l, c)
        attn_output = self.out_proj(attn_output).transpose(1, 2)

        return self.norm(attn_output + deep_feat)

    def extra_repr(self):
        return f"channels={self.q_proj.in_features}, num_heads={self.num_heads}"
