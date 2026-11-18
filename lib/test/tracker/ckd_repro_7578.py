"""Restore the operations used by the archived ep10 LasHeR run."""
import importlib.util
import os
from pathlib import Path

import torch
from lib.models.layers.frozen_bn import FrozenBatchNorm2d
from .ckd import OSTrack_twobranch

def original_precise_pool():
    from external.prroi_pool import functional as module

    cached = Path(__file__).resolve().parents[3] / '.build/prroi/_prroi_pooling.so'
    cached = Path(os.environ.get('DMATRACK_PRROI_LIBRARY', str(cached)))
    if cached.is_file():
        native_spec = importlib.util.spec_from_file_location('_prroi_pooling', str(cached))
        native = importlib.util.module_from_spec(native_spec)
        native_spec.loader.exec_module(native)
        module._prroi_pooling = native
    return module.prroi_pool2d

class PrecisePool(torch.nn.Module):
    def __init__(self, original, operation):
        super().__init__()
        self.pooled_height = original.pooled_height
        self.pooled_width = original.pooled_width
        self.spatial_scale = original.spatial_scale
        self.operation = operation

    def forward(self, features, rois):
        return self.operation(features, rois, self.pooled_height, self.pooled_width, self.spatial_scale)

def freeze_box_bn(module):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features).to(child.weight.device)
            frozen.load_state_dict({k: v for k, v in child.state_dict().items() if k != 'num_batches_tracked'}, strict=True)
            setattr(module, name, frozen)
        else:
            freeze_box_bn(child)

class HistoricalCKD(OSTrack_twobranch):
    def __init__(self, params, dataset_name):
        params.cfg.MODEL.BACKBONE.PARAMS.rgbt_loc = list(range(1, 12))
        params.inference_strategy = getattr(params, 'inference_strategy', 'always_rgbt')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        super().__init__(params, dataset_name)
        freeze_box_bn(self.network)
        checkpoint = torch.load(params.checkpoint, map_location='cpu')
        self.network.load_state_dict(checkpoint['net'], strict=True)
        operation = original_precise_pool()
        for branch in ('score_branch_rgbt', 'score_branch_rgb', 'score_branch_t'):
            decoder = getattr(self.network, branch)
            decoder.search_prroipool = PrecisePool(decoder.search_prroipool, operation)
            decoder.template_prroipool = PrecisePool(decoder.template_prroipool, operation)
        self.network.eval()
        print('Historical profile: strict checkpoint load; fusion=1..11; FrozenBatchNorm2d; native PreciseRoIPooling; TF32 enabled')

    def initialize(self, image_v, image_i, info):
        result = super().initialize(image_v, image_i, info) or {}
        self.before_update_info[2]['init_bbox'] = list(info['init_bbox'])
        result.update(expert='start', source_gate='start')
        return result

    def _make_template_record(self, image_v, image_i, bbox, score=None):
        record = super()._make_template_record(image_v, image_i, bbox, score)

        record['bbox'] = list(bbox)
        return record

    def _maybe_update_template(self):
        super()._maybe_update_template()
        self.before_update_info[2]['init_bbox'] = list(self.current_template_info['bbox'])

    def track(self, image_v, image_i, info=None):
        result = super().track(image_v, image_i, info)

        if self.inference_strategy == 'always_rgbt':
            score = str(result['best_score'].item())[:6]
            result['source_gate'] = 'rgbt:{} rgb:{} t:{}'.format(score, score, score)
        else:
            result['source_gate'] = 'rgbt:{:.9g} rgb:{:.9g} t:{:.9g}'.format(*self._last_gate_values)
        return result

    def track_with_details(self, image_v, image_i, info=None):
        details = super().track_with_details(image_v, image_i, info)
        output = details['out_dict']
        self._last_gate_values = [float(value.item()) for value in output['source_gate']]
        if self.inference_strategy == 'hard_switch':
            rgb_exit = bool((output['source_gate'][1] > self.tau_rgb).reshape(-1)[0].item())
            tir_exit = bool((output['source_gate'][2] > self.tau_tir).reshape(-1)[0].item())
            expected = 'rgb' if rgb_exit else ('t' if tir_exit else 'rgbt')
            if output['expert'] != expected:
                raise RuntimeError('Gate routing violates the configured thresholds')
        return details

def get_tracker_class():
    return HistoricalCKD
