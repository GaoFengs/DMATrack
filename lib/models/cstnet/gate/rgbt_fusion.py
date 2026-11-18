import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_ir = nn.Linear(dim, dim)
        self.k_rgb = nn.Linear(dim, dim)
        self.v_rgb = nn.Linear(dim, dim)

        self.q_rgb = nn.Linear(dim, dim)
        self.k_ir = nn.Linear(dim, dim)
        self.v_ir = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.out_ir = nn.Linear(dim, dim)
        self.out_rgb = nn.Linear(dim, dim)

        self._init_weights()

    def _init_weights(self):

        nn.init.xavier_uniform_(self.q_ir.weight)
        nn.init.xavier_uniform_(self.k_rgb.weight)
        nn.init.xavier_uniform_(self.v_rgb.weight)

        nn.init.xavier_uniform_(self.q_rgb.weight)
        nn.init.xavier_uniform_(self.k_ir.weight)
        nn.init.xavier_uniform_(self.v_ir.weight)

        nn.init.zeros_(self.q_ir.bias)
        nn.init.zeros_(self.k_rgb.bias)
        nn.init.zeros_(self.v_rgb.bias)

        nn.init.zeros_(self.q_rgb.bias)
        nn.init.zeros_(self.k_ir.bias)
        nn.init.zeros_(self.v_ir.bias)

        nn.init.xavier_uniform_(self.out_ir.weight)
        nn.init.xavier_uniform_(self.out_rgb.weight)
        nn.init.zeros_(self.out_ir.bias)
        nn.init.zeros_(self.out_rgb.bias)

    def forward(self, ir_feat, rgb_feat):
        bn, seq, _ = ir_feat.shape

        q_ir = self.q_ir(ir_feat).view(bn, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k_rgb = self.k_rgb(rgb_feat).view(bn, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v_rgb = self.v_rgb(rgb_feat).view(bn, seq, self.num_heads, self.head_dim).transpose(1, 2)

        q_rgb = self.q_rgb(rgb_feat).view(bn, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k_ir = self.k_ir(ir_feat).view(bn, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v_ir = self.v_ir(ir_feat).view(bn, seq, self.num_heads, self.head_dim).transpose(1, 2)

        attn_ir = (q_ir @ k_rgb.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_ir = F.softmax(attn_ir, dim=-1)
        attn_ir = self.dropout(attn_ir)
        out_ir = (attn_ir @ v_rgb).transpose(1, 2).reshape(bn, seq, self.dim)
        out_ir = self.out_ir(out_ir)

        attn_rgb = (q_rgb @ k_ir.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_rgb = F.softmax(attn_rgb, dim=-1)
        attn_rgb = self.dropout(attn_rgb)
        out_rgb = (attn_rgb @ v_ir).transpose(1, 2).reshape(bn, seq, self.dim)
        out_rgb = self.out_rgb(out_rgb)

        output = out_ir + out_rgb

        return output

class Bi_direct_adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)
        self.adapter_up = nn.Linear(dim, 768)
        self.adapter_mid = nn.Linear(dim, dim)

        self.cross_attn = CrossModalAttention(dim=dim, num_heads=8)

        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x_v, x_i):
        b, n, c = x_v.shape
        xv_down = self.adapter_down(x_v)
        xi_down = self.adapter_down(x_i)

        xv_down = self.adapter_mid(xv_down)
        xi_down = self.adapter_mid(xi_down)
        x_down = self.cross_attn(xv_down, xi_down)

        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up

class rgbt_layer(nn.Module):
    def __init__(self, dim=768, drop_path=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.adap_t = Bi_direct_adapter()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_v, x_i):

        x = self.drop_path(self.adap_t(self.norm(x_v), self.norm(x_i)))

        return x
