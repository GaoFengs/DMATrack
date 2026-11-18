"""Freeze tracking predictions and learn three independent IoU classifiers."""
import hashlib
from copy import deepcopy
import torch
from torch import nn
from lib.models.ckd import build_ostrack_ckd
from lib.models.cstnet.gate.score_decoder import ScoreDecoder
from lib.test.tracker.ckd_repro_7578 import freeze_box_bn, original_precise_pool, PrecisePool

SCORES = ('score_branch_rgb', 'score_branch_t', 'score_branch_rgbt')

def is_score(name):
    return name.split('.')[0] in SCORES

def frozen_digest(network):
    digest = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        if not is_score(name):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def score_digests(network):
    result = {}
    for branch in SCORES:
        digest = hashlib.sha256()
        for name, tensor in getattr(network, branch).state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        result[branch] = digest.hexdigest()
    return result

def aligned_iou(pred_cxcywh, gt_xywh):
    """One prediction per sample; never use NxN pairwise IoU as labels."""
    pred = pred_cxcywh.detach().float().reshape(-1, 4)
    gt = gt_xywh.detach().float().reshape(-1, 4)
    p0, p1 = pred[:, :2]-pred[:, 2:]/2, pred[:, :2]+pred[:, 2:]/2
    g0, g1 = gt[:, :2], gt[:, :2]+gt[:, 2:]
    intersection = (torch.minimum(p1, g1)-torch.maximum(p0, g0)).clamp(min=0).prod(-1)
    union = pred[:, 2:].clamp(min=0).prod(-1)+gt[:, 2:].clamp(min=0).prod(-1)-intersection
    return intersection / union.clamp(min=1e-8)

class FloatPrecisePool(PrecisePool):
    def forward(self, features, rois):

        with torch.cuda.amp.autocast(enabled=False):
            return super().forward(features.float(), rois.float())

class ScoreOnlyModel(nn.Module):
    def __init__(self, cfg, checkpoint, reset_scores=True, threshold=.8, fusion='sum',
                 branch_reduction='mean', positive_boundary='gt'):
        super().__init__()
        if not 0 < threshold < 1:
            raise ValueError('IoU label threshold must be between zero and one')
        if branch_reduction not in ('mean', 'sum') or positive_boundary not in ('gt', 'ge'):
            raise ValueError('Invalid classification protocol')
        self.branch_reduction = branch_reduction
        self.positive_boundary = positive_boundary
        if fusion not in ('sum', 'concat'):
            raise ValueError('Unknown fusion: ' + str(fusion))
        cfg = deepcopy(cfg)
        cfg.MODEL.BACKBONE.PARAMS.rgbt_loc = list(range(1, 12))

        cfg.MODEL.RGBT_SCORE_FUSION = 'sum'
        self.network = build_ostrack_ckd(cfg, training=False)
        freeze_box_bn(self.network)
        self.network.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
        original_frozen = frozen_digest(self.network)
        if reset_scores:
            for name in SCORES:
                width = 1536 if name == 'score_branch_rgbt' and fusion == 'concat' else 768
                setattr(self.network, name, ScoreDecoder(pool_size=4, hidden_dim=width, num_heads=width//64))
        elif fusion != 'sum':
            raise ValueError('Concatenation requires freshly initialized score weights')
        self.network.rgbt_score_fusion = fusion
        self.fusion = fusion
        self.initial_score_digests = score_digests(self.network)
        operation = original_precise_pool()
        for name, parameter in self.network.named_parameters():
            parameter.requires_grad_(is_score(name))
        for name in SCORES:
            decoder = getattr(self.network, name)
            for pool_name in ('search_prroipool', 'template_prroipool'):
                setattr(decoder, pool_name, FloatPrecisePool(getattr(decoder, pool_name), operation))
        self.threshold = threshold
        self.train()
        self.initial_frozen_digest = original_frozen
        assert frozen_digest(self.network) == original_frozen

    def train(self, mode=True):

        super().train(False)
        for name in SCORES:
            getattr(self.network, name).train(mode)
        self.training = mode
        return self

    @torch.no_grad()
    def tracking_outputs(self, template, search):
        n = self.network
        with torch.cuda.amp.autocast(enabled=False):
            xr, ar = n.rgb_branch(z_li=template[:1], x_li=search[:1], feature_lists=[])
            xt, at = n.tir_branch(z_li=template[1:], x_li=search[1:], feature_lists=ar['x_list'])
            length = n.feat_len_s
            def cross(module, aux):
                return module(aux['x_list'][11][:, -length:],
                              [aux['x_list'][i][:, -length:] for i in (2, 5, 8)])
            xr, xt = cross(n.cross_attn_rgb, ar), cross(n.cross_attn_t, at)
            tr, tt = ar['x_list'][11][:, :-length], at['x_list'][11][:, :-length]
            boxes = (n.forward_head(xr, head=n.box_head_v), n.forward_head(xt, head=n.box_head_i),
                     n.forward_head(torch.cat((xr, xt), 2)))
        fused_x, fused_z = n.rgbt_score_inputs(xr, xt, tr, tt)
        return (xr, xt, fused_x), (tr, tt, fused_z), boxes

    def forward(self, template, search, template_box, search_box):
        features, templates, boxes = self.tracking_outputs(template, search)
        logits = torch.stack([getattr(self.network, name)(x, z, box, gt_bboxs_template=template_box).reshape(-1)
                              for name, x, z, box in zip(SCORES, features, templates, boxes)], dim=1)
        iou = torch.stack([aligned_iou(box['pred_boxes'], search_box) for box in boxes], dim=1)
        labels = self.make_labels(iou).to(logits.dtype)
        loss = self.classification_loss(logits, labels)
        return loss, logits.detach().float(), labels.detach().float(), iou

    def make_labels(self, iou):
        return iou >= self.threshold if self.positive_boundary == 'ge' else iou > self.threshold

    def classification_loss(self, logits, labels):
        per_branch = nn.functional.binary_cross_entropy_with_logits(
            logits.float(), labels.float(), reduction='none').mean(dim=0)
        return per_branch.sum() if self.branch_reduction == 'sum' else per_branch.mean()
