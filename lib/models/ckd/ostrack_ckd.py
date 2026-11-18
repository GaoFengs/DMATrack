"""

"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from .vit import vit_base_patch16_224, resize_pos_embed
from .vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from greenlet import greenlet
from lib.models.cstnet.gate import CrossLayerAttention
from lib.models.cstnet.gate.score_decoder import ScoreDecoder

class OSTrack_CKD(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, rgb_branch, tir_branch, teacher_rgb, teacher_tir, box_head, box_head_v, box_head_i,\
                 aux_loss=False, head_type="CORNER",mask_probability=0.0, mask_ratio=0.0, training=True,
                 rgbt_score_fusion="sum"):
        super().__init__()
        self.rgb_branch = rgb_branch
        self.tir_branch = tir_branch
        self.rgb_branch.mask_probability = mask_probability
        self.tir_branch.mask_probability = mask_probability
        self.rgb_branch.mask_ratio = mask_ratio
        self.tir_branch.mask_ratio = mask_ratio
        self.box_head = box_head

        self.teacher_rgb = teacher_rgb
        self.teacher_tir = teacher_tir
        self.box_head_v = box_head_v
        self.box_head_i = box_head_i

        self.mask_probability = mask_probability
        self.mask_ratio = mask_ratio

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.cross_attn_rgb = CrossLayerAttention(channels=256, num_heads=8)
        self.cross_attn_t = CrossLayerAttention(channels=256, num_heads=8)
        if rgbt_score_fusion not in ("sum", "concat"):
            raise ValueError("Unknown RGBT score fusion: " + str(rgbt_score_fusion))
        self.rgbt_score_fusion = rgbt_score_fusion
        score_dim = 1536 if rgbt_score_fusion == "concat" else 768
        self.score_branch_rgbt = ScoreDecoder(pool_size=4, hidden_dim=score_dim, num_heads=score_dim//64)
        self.score_branch_rgb = ScoreDecoder(pool_size=4, hidden_dim=768, num_heads=768//64)
        self.score_branch_t = ScoreDecoder(pool_size=4, hidden_dim=768, num_heads=768//64)

    def rgbt_score_inputs(self, rgb, tir, template_rgb, template_tir):
        """Only the score input changes; tracking-head fusion remains untouched."""
        if self.rgbt_score_fusion == "concat":
            return torch.cat((rgb, tir), dim=-1), torch.cat((template_rgb, template_tir), dim=-1)
        return rgb + tir, template_rgb + template_tir

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,gt_bboxs_search=None, gt_bboxs_template=None,
                inference_strategy="hard_switch",
                soft_temperature=1.0,
                tau_rgb=0.8,
                tau_tir=0.8,
                tau_rgbt=0.8,
                return_analysis=False,
                force_tir_stage=False,
                force_rgbt_stage=False,
                ):
        import time
        start_time = time.time()
        self.training = False
        x_rgb, aux_dict_rgb = self.rgb_branch(z_li=template[:1], x_li=search[:1],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, feature_lists=[] )

        x_rgb = self.cross_attn_rgb(aux_dict_rgb['x_list'][11][:, -self.feat_len_s:], [aux_dict_rgb['x_list'][2][:, -self.feat_len_s:], aux_dict_rgb['x_list'][5][:, -self.feat_len_s:], aux_dict_rgb['x_list'][8][:, -self.feat_len_s:]])
        out_rgb = self.forward_head(x_rgb, None, head=self.box_head_v)
        rgb_t = aux_dict_rgb['x_list'][11][:, :-self.feat_len_s]
        rgb_score = self.score_branch_rgb(x_rgb, rgb_t, search_box=out_rgb, gt_bboxs_search=None, gt_bboxs_template=gt_bboxs_template)
        rgb_score_sigmoid = torch.sigmoid(rgb_score)
        nan_score = torch.full_like(rgb_score_sigmoid, float("nan"))
        rgb_exit = bool((rgb_score_sigmoid > tau_rgb).reshape(-1)[0].item())
        if inference_strategy == "always_rgb":
            out = out_rgb
            out['expert'] = "rgb"
            out['source_gate'] = [nan_score, rgb_score_sigmoid, nan_score]
            out['best_score'] = rgb_score_sigmoid
            out['time'] = time.time() - start_time
            return out
        if inference_strategy == "hard_switch" and rgb_score_sigmoid > tau_rgb and not (return_analysis or force_tir_stage or force_rgbt_stage):
            out = out_rgb
            out['expert'] = "rgb"
            out['source_gate'] = [nan_score, rgb_score_sigmoid, nan_score]
            out['best_score'] = rgb_score_sigmoid
            out['time'] =  time.time() - start_time
            return out

        x_tir, aux_dict_tir = self.tir_branch(z_li=template[1:], x_li=search[1:],
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, feature_lists=aux_dict_rgb['x_list'])

        x_tir = self.cross_attn_t(aux_dict_tir['x_list'][11][:, -self.feat_len_s:], [aux_dict_tir['x_list'][2][:, -self.feat_len_s:], aux_dict_tir['x_list'][5][:, -self.feat_len_s:], aux_dict_tir['x_list'][8][:, -self.feat_len_s:]])
        out_t = self.forward_head(x_tir, None, head=self.box_head_i)
        t_t = aux_dict_tir['x_list'][11][:, :-self.feat_len_s]
        tir_score =self.score_branch_t(x_tir, t_t, search_box=out_t, gt_bboxs_search=None, gt_bboxs_template=gt_bboxs_template)
        tir_score_sigmoid = torch.sigmoid(tir_score)
        tir_exit = bool((tir_score_sigmoid > tau_tir).reshape(-1)[0].item())
        if inference_strategy == "always_tir":
            out = out_t
            out['expert'] = "t"
            out['source_gate'] = [nan_score, rgb_score_sigmoid, tir_score_sigmoid]
            out['best_score'] = tir_score_sigmoid
            out['time'] = time.time() - start_time
            return out
        if inference_strategy == "hard_switch" and tir_score_sigmoid > tau_tir and not (return_analysis or force_rgbt_stage):
            out = out_t
            out['expert'] = "t"
            out['source_gate'] = [nan_score, rgb_score_sigmoid, tir_score_sigmoid]
            out['best_score'] = tir_score_sigmoid
            out['time'] =  time.time() - start_time
            return out

        x = torch.cat([x_rgb,x_tir], 2)

        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out_rgbt = self.forward_head(feat_last, None)

        rgbt_x, rgbt_t = self.rgbt_score_inputs(
            x_rgb, x_tir, aux_dict_rgb['x_list'][11][:, :-self.feat_len_s],
            aux_dict_tir['x_list'][11][:, :-self.feat_len_s])
        rgbt_score = self.score_branch_rgbt(rgbt_x, rgbt_t, search_box=out_rgbt, gt_bboxs_search=None, gt_bboxs_template=gt_bboxs_template)
        rgbt_score_sigmoid = torch.sigmoid(rgbt_score)
        if inference_strategy == "always_rgbt":
            out = out_rgbt
            out['expert'] = "rgbt"
            out['source_gate'] = [rgbt_score_sigmoid, rgb_score_sigmoid, tir_score_sigmoid]
            out['best_score'] = rgbt_score_sigmoid
            out['time'] = time.time() - start_time
            return out

        if inference_strategy == "soft_switch":
            weight_logits = torch.stack(
                [
                    (rgb_score_sigmoid - tau_rgb).reshape(-1)[0],
                    (tir_score_sigmoid - tau_tir).reshape(-1)[0],
                    (rgbt_score_sigmoid - tau_rgbt).reshape(-1)[0],
                ]
            )
            weights = torch.softmax(weight_logits, dim=0)
            w_rgb, w_tir, w_rgbt = [w.view(1, 1, 1, 1) for w in weights]
            fused_out = {}
            for key in ["pred_boxes", "score_map", "size_map", "offset_map"]:
                fused_out[key] = w_rgb * out_rgb[key] + w_tir * out_t[key] + w_rgbt * out_rgbt[key]
            fused_out['expert'] = "soft"
            fused_out['source_gate'] = [rgbt_score_sigmoid, rgb_score_sigmoid, tir_score_sigmoid]
            fused_out['best_score'] = torch.max(
                torch.stack(
                    [
                        rgbt_score_sigmoid.reshape(-1)[0],
                        rgb_score_sigmoid.reshape(-1)[0],
                        tir_score_sigmoid.reshape(-1)[0],
                    ]
                )
            ).view(1)
            fused_out['time'] = time.time() - start_time
            return fused_out

        if inference_strategy == "softmax_argmax":
            weight_logits = torch.stack(
                [
                    (rgb_score_sigmoid - tau_rgb).reshape(-1)[0],
                    (tir_score_sigmoid - tau_tir).reshape(-1)[0],
                    (rgbt_score_sigmoid - tau_rgbt).reshape(-1)[0],
                ]
            )
            weights = torch.softmax(weight_logits, dim=0)
            winner = int(torch.argmax(weights).item())
            if winner == 0:
                out = out_rgb
                out['expert'] = "rgb"
                out['best_score'] = rgb_score_sigmoid
            elif winner == 1:
                out = out_t
                out['expert'] = "t"
                out['best_score'] = tir_score_sigmoid
            else:
                out = out_rgbt
                out['expert'] = "rgbt"
                out['best_score'] = rgbt_score_sigmoid
            out['source_gate'] = [rgbt_score_sigmoid, rgb_score_sigmoid, tir_score_sigmoid]
            out['softmax_weights'] = [weights[2].view(1), weights[0].view(1), weights[1].view(1)]
            out['time'] = time.time() - start_time
            return out

        if inference_strategy == "hard_selection":
            if rgb_score_sigmoid > tau_rgb:
                out = out_rgb
                out['expert'] = "rgb"
                out['best_score'] = rgb_score_sigmoid
            elif tir_score_sigmoid > tau_tir:
                out = out_t
                out['expert'] = "t"
                out['best_score'] = tir_score_sigmoid
            else:
                out = out_rgbt
                out['expert'] = "rgbt"
                out['best_score'] = rgbt_score_sigmoid
            out['source_gate'] = [rgbt_score_sigmoid, rgb_score_sigmoid, tir_score_sigmoid]
            out['time'] = time.time() - start_time
            return out

        normal_route = "rgb" if rgb_exit else ("t" if tir_exit else "rgbt")
        if normal_route == "rgb":
            out = out_rgb
            out['expert'] = "rgb"
            out['source_gate'] = [rgbt_score_sigmoid, rgb_score_sigmoid, tir_score_sigmoid]
            out['best_score'] = rgb_score_sigmoid
        elif normal_route == "t":
            out = out_t
            out['expert'] = "t"
            out['source_gate'] = [rgbt_score_sigmoid, rgb_score_sigmoid, tir_score_sigmoid]
            out['best_score'] = tir_score_sigmoid
        else:
            out = out_rgbt
            out['expert'] = "rgbt"
            out['source_gate'] = [rgbt_score_sigmoid, rgb_score_sigmoid, tir_score_sigmoid]
            out['best_score'] = rgbt_score_sigmoid
        out['time'] =  time.time() - start_time
        if return_analysis:
            out['analysis'] = {
                'rgb_out': out_rgb,
                'tir_out': out_t,
                'rgbt_out': out_rgbt,
                'rgb_score': rgb_score_sigmoid,
                'tir_score': tir_score_sigmoid,
                'rgbt_score': rgbt_score_sigmoid,
                'rgb_exit': rgb_exit,
                'tir_exit': tir_exit,
                'normal_route': normal_route,
                'tir_stage_reached': not rgb_exit,
                'rgbt_stage_reached': not rgb_exit and not tir_exit,
            }
        return out

    def forward_head(self, cat_feature, gt_score_map=None, head=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if head==None:
            box_head = self.box_head
        else:
            box_head = head
        enc_opt = cat_feature[:, -self.feat_len_s:]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":

            pred_box, score_map = box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":

            score_map_ctr, bbox, size_map, offset_map = box_head(opt_feat, gt_score_map)

            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_ostrack_ckd(cfg, training=True):
    patch_start_index = 1
    rgbt_loc = getattr(getattr(cfg.MODEL.BACKBONE, "PARAMS", {}), "rgbt_loc", list(range(1, 12)))

    rgb_branch = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO, branch_mode="rgb")
    rgb_branch.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    if cfg.MODEL.SHARE_STUDENT:
        tir_branch = rgb_branch
    else:
        tir_branch = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO, branch_mode="t",
                                            rgbt_loc=rgbt_loc)
        tir_branch.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    if cfg.MODEL.RGB_TEACHER:
        teacher_rgb = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO, )
        teacher_rgb.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
        box_head_v = build_box_head(cfg, teacher_rgb.embed_dim)
    else:
        teacher_rgb = None
        box_head_v = None

    if cfg.MODEL.TIR_TEACHER:
        teacher_tir = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO, )
        teacher_tir.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
        box_head_i = build_box_head(cfg, teacher_tir.embed_dim)
    else:
        teacher_tir = None
        box_head_i = None

    box_head = build_box_head(cfg, teacher_tir.embed_dim * 2)

    def pos_embed_filter(param):
        param['backbone.pos_embed_z'] = resize_pos_embed(param['backbone.pos_embed_z'], posemb_new=torch.zeros(1,64,768),num_tokens=0)
        param['backbone.pos_embed_x'] = resize_pos_embed(param['backbone.pos_embed_x'], posemb_new=torch.zeros(1,256,768),num_tokens=0)
        param['backbone.pos_embed_z'] += param['backbone.temporal_pos_embed_z']
        param['backbone.pos_embed_x'] += param['backbone.temporal_pos_embed_x']
        return param
        if param['backbone.pos_embed_z'].shape[1]==144:
            param['backbone.pos_embed_z'] = param['backbone.pos_embed_z'].reshape(1, 12, 12, 768)
            param['backbone.pos_embed_z'] = param['backbone.pos_embed_z'][:, 2:-2, 2:-2, :].reshape(1, 64, 768)

        if param['backbone.pos_embed_x'].shape[1]==576:
            param['backbone.pos_embed_x'] = param['backbone.pos_embed_x'].reshape(1, 24, 24, 768)
            param['backbone.pos_embed_x'] = param['backbone.pos_embed_x'][:, 4:-4, 4:-4, :].reshape(1, 256, 768)

        return param

    model = OSTrack_CKD(
        rgb_branch,
        tir_branch,
        teacher_rgb,
        teacher_tir,
        box_head = box_head,
        box_head_v = box_head_v,
        box_head_i = box_head_i,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        mask_ratio=cfg.TRAIN.INPUT_MASK_RATIO,
        mask_probability=cfg.TRAIN.MASK_PROBABILITY,
        training = training,
        rgbt_score_fusion=getattr(cfg.MODEL, 'RGBT_SCORE_FUSION', 'sum'),
    )
    if cfg.MODEL.PRETRAIN_FILE !="" and training:
        print(cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location='cpu')
        new_checkpoint = {"net": {}}

        for key in list(checkpoint['net'].keys()):
            break

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['net'], strict=False)
        print("missing keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
        print("load finsh")
    return model
