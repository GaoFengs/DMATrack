"""
SPM: Score Prediction Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .functional import prroi_pool2d
from timm.models.layers import trunc_normal_

def bbox_cxcywh_to_xyxy(bbox):
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx), (cy), (cx + w), (cy + h)]
    return torch.cat(bbox_new, dim=-1)

def bbox_cxcywh_to_xyxy_center(bbox):
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)

__all__ = ['PrRoIPool2D']

class PrRoIPool2D(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super().__init__()

        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return prroi_pool2d(features, rois, self.pooled_height, self.pooled_width, self.spatial_scale)

    def extra_repr(self):
        return 'kernel_size=({pooled_height}, {pooled_width}), spatial_scale={spatial_scale}'.format(**self.__dict__)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ScoreDecoder(nn.Module):
    def __init__(self, num_heads=12, hidden_dim=768, nlayer_head=3, pool_size=4):
        super().__init__()
        self.num_heads = num_heads
        self.pool_size = pool_size
        self.score_head = MLP(hidden_dim, hidden_dim, 1, nlayer_head)
        self.scale = hidden_dim ** -0.5
        self.search_prroipool = PrRoIPool2D(pool_size, pool_size, spatial_scale=1.0)
        self.template_prroipool = PrRoIPool2D(pool_size, pool_size, spatial_scale=1.0)
        self.proj_q = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))
        self.proj_k = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))
        self.proj_v = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))

        self.proj = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(2))

        self.normt = nn.LayerNorm(hidden_dim)
        self.norms = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(2))

        self.score_token_t = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.score_token_s = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        trunc_normal_(self.score_token_t, std=.02)
        trunc_normal_(self.score_token_s, std=.02)

    def forward(self, feat_last, feat_last_t, search_box, gt_bboxs_search=None, gt_bboxs_template=None):

        feat_last = F.layer_norm(feat_last, feat_last.shape[-1:])
        feat_last_t = F.layer_norm(feat_last_t, feat_last_t.shape[-1:])

        if gt_bboxs_search == None:

            search_box = bbox_cxcywh_to_xyxy_center(search_box['pred_boxes'].clone().view(-1, 4))

        else:
            search_box = bbox_cxcywh_to_xyxy(gt_bboxs_search.clone())

        template_box = bbox_cxcywh_to_xyxy(gt_bboxs_template.clone().view(-1, 4))

        num_template_token = 0
        num_search_token = 256

        enc_opt_search = feat_last
        opt = (enc_opt_search.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        enc_opt_search = opt.view(-1, C, 16, 16)

        enc_opt_template = feat_last_t
        opt = (enc_opt_template.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()

        enc_opt_template = opt.view(-1, C, 8, 8)

        b, c, h, w = enc_opt_search.shape
        search_box = search_box.clone() * w

        bb_pool = search_box.view(-1, 4)

        batch_size = bb_pool.shape[0]

        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb_pool.device)

        target_roi = torch.cat((batch_index, bb_pool), dim=1)

        b, c, h, w = enc_opt_template.shape
        template_box = template_box.clone() * w

        bb_pool_template = template_box.view(-1, 4)

        batch_size = bb_pool_template.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb_pool_template.device)
        target_roi_template = torch.cat((batch_index, bb_pool_template), dim=1)

        x = [self.normt(self.score_token_t.expand(b, -1, -1)), self.norms(self.score_token_s.expand(b, -1, -1))]

        search_box_feat = rearrange(self.search_prroipool(enc_opt_search, target_roi), 'b c h w -> b (h w) c')

        template_feat = rearrange(self.template_prroipool(enc_opt_template, target_roi_template.to(enc_opt_template.device)), 'b c h w -> b (h w) c')

        kv_memory = [template_feat, search_box_feat]
        result = []
        for i in range(2):
            q = rearrange(self.proj_q[i](x[i]), 'b t (n d) -> b n t d', n=self.num_heads)
            k = rearrange(self.proj_k[i](kv_memory[i]), 'b t (n d) -> b n t d', n=self.num_heads)
            v = rearrange(self.proj_v[i](kv_memory[i]), 'b t (n d) -> b n t d', n=self.num_heads)

            attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
            attn = F.softmax(attn_score, dim=-1)
            x_cls = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
            x_cls = rearrange(x_cls, 'b h t d -> b t (h d)')
            x_cls = self.proj[i](x_cls)
            x_cls = self.norm2[i](x_cls)
            result.append(x_cls)

        return self.score_head(result[0] + result[1])
