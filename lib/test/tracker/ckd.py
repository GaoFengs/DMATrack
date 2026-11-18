import math
import copy

from lib.models.ckd import build_ostrack_ckd
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
import torch.nn.functional as F
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out

class OSTrack_twobranch(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack_twobranch, self).__init__(params)
        network = build_ostrack_ckd(params.cfg, training=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.params.checkpoint)
        checkpoint = torch.load(self.params.checkpoint, map_location='cpu')
        state_dict = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
        missing_keys, unexpected_keys = network.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print("missing_keys", missing_keys)
        if unexpected_keys:
            print("unexpected_keys", unexpected_keys)
        self.cfg = params.cfg
        self.network = network.to(self.device)
        self.network.eval()
        self.preprocessor = Preprocessor(device=self.device)
        self.state = None
        self.stste_n = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)

        self.debug = getattr(params, "debug", 0)
        self.use_visdom = self.debug
        self.frame_id = 0
        if self.debug:
            raise ValueError('Only non-interactive inference is supported')

        self.save_all_boxes = params.save_all_boxes
        self.inference_strategy = getattr(params, "inference_strategy", "hard_switch")
        self.soft_temperature = float(getattr(params, "soft_temperature", 1.0))
        self.tau_rgb = float(getattr(params, "tau_rgb", 0.8))
        self.tau_tir = float(getattr(params, "tau_tir", 0.8))
        self.tau_rgbt = float(getattr(params, "tau_rgbt", 0.8))
        self.return_analysis = bool(getattr(params, "return_analysis", False))
        self.force_tir_stage = bool(getattr(params, "force_tir_stage", False))
        self.force_rgbt_stage = bool(getattr(params, "force_rgbt_stage", False))
        self.enable_template_update = bool(getattr(params, "enable_template_update", True))
        self.template_update_interval = int(getattr(params, "template_update_interval", 50))
        self.template_score_threshold = float(getattr(params, "template_score_threshold", 0.3))
        self.template_iou_threshold = float(getattr(params, "template_iou_threshold", 0.4))
        print(
            f"inference_strategy={self.inference_strategy}, "
            f"soft_temperature={self.soft_temperature:.3f}, "
            f"tau_rgb={self.tau_rgb:.3f}, tau_tir={self.tau_tir:.3f}, tau_rgbt={self.tau_rgbt:.3f}"
        )
        print(
            f"template_update={'enabled' if self.enable_template_update and self.template_update_interval > 0 else 'disabled'}, "
            f"interval={self.template_update_interval}, "
            f"score_thr={self.template_score_threshold:.3f}, iou_thr={self.template_iou_threshold:.3f}"
        )

    def initialize(self, image_v, image_i, info: dict):
        self.temps = []
        self.temps_score = []
        self.state = info['init_bbox']
        self._apply_template(image_v, image_i, info['init_bbox'])
        self.current_template_info = self._make_template_record(image_v, image_i, info['init_bbox'])
        self.confidences = []
        self.up_images = []
        self.up_bboxs = []
        self.before_update_info = [image_v.copy(), image_i.copy(), {"init_bbox": [float(v) for v in info["init_bbox"]]}]
        self.frame_id = 0

        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def update_template(self, image_v, image_i, bbox):
        self._apply_template(image_v, image_i, bbox)
        self.current_template_info = self._make_template_record(image_v, image_i, bbox)

    def _apply_template(self, image_v, image_i, bbox):
        z_patch_arr_rgb, resize_factor_rgb, z_amask_arr_rgb = sample_target(
            image_v, bbox, self.params.template_factor, output_sz=self.params.template_size
        )
        z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir = sample_target(
            image_i, bbox, self.params.template_factor, output_sz=self.params.template_size
        )

        self.z_patch_arr_rgb = z_patch_arr_rgb
        self.z_patch_arr_tir = z_patch_arr_tir

        template_rgb = self.preprocessor.process(z_patch_arr_rgb, z_amask_arr_rgb)
        template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)
        with torch.no_grad():
            self.z_dict1_rgb = template_rgb
            self.z_dict1_tir = template_tir
            self.z_dict = [self.z_dict1_rgb, self.z_dict1_tir]

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox_rgb = self.transform_bbox_to_crop(
                bbox, resize_factor_rgb, template_rgb.tensors.device
            ).squeeze(1)
            self.box_mask_z_rgb = generate_mask_cond(self.cfg, 1, template_rgb.tensors.device, template_bbox_rgb)

            template_bbox_tir = self.transform_bbox_to_crop(
                bbox, resize_factor_tir, template_tir.tensors.device
            ).squeeze(1)
            self.box_mask_z_tir = generate_mask_cond(self.cfg, 1, template_tir.tensors.device, template_bbox_tir)

            self.box_mask_z = [self.box_mask_z_rgb, self.box_mask_z_tir]
            self.box_mask_z = self.box_mask_z[0]

        with torch.no_grad():
            crop_sz = torch.Tensor([self.params.template_size, self.params.template_size])
            self.stste_n = [
                transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip([torch.tensor(bbox)], [torch.tensor(bbox)], [resize_factor_rgb])
            ][0].float()
            self.stste_n = self.stste_n.view(1, 1, -1).to(self.device)

    def _make_template_record(self, image_v, image_i, bbox, score=None):
        return {
            "image_v": image_v.copy(),
            "image_i": image_i.copy(),
            "bbox": [float(v) for v in bbox],
            "score": None if score is None else float(score),
        }

    def _score_to_float(self, score):
        if score is None:
            return None
        if isinstance(score, torch.Tensor):
            return float(score.detach().view(-1)[0].item())
        return float(score)

    def _bbox_iou(self, box_a, box_b):
        ax1, ay1, aw, ah = [float(v) for v in box_a]
        bx1, by1, bw, bh = [float(v) for v in box_b]
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, aw) * max(0.0, ah)
        area_b = max(0.0, bw) * max(0.0, bh)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _prepare_search(self, image_v, image_i):
        x_patch_arr_rgb, resize_factor_rgb, x_amask_arr_rgb = sample_target(
            image_v, self.state, self.params.search_factor, output_sz=self.params.search_size
        )
        x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(
            image_i, self.state, self.params.search_factor, output_sz=self.params.search_size
        )
        search_rgb = self.preprocessor.process(x_patch_arr_rgb, x_amask_arr_rgb)
        search_tir = self.preprocessor.process(x_patch_arr_tir, x_amask_arr_tir)
        return {
            "search_rgb": search_rgb,
            "search_tir": search_tir,
            "resize_factor_rgb": resize_factor_rgb,
            "resize_factor_tir": resize_factor_tir,
            "x_patch_arr_rgb": x_patch_arr_rgb,
            "x_patch_arr_tir": x_patch_arr_tir,
        }

    def _decode_branch_output(self, out_dict, head, resize_factor_rgb, image_shape, state_for_mapping=None):
        H, W = image_shape[:2]
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, head_score = head.cal_bbox(
            response, out_dict['size_map'], out_dict['offset_map'], return_score=True
        )
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor_rgb).tolist()
        state_ref = self.state if state_for_mapping is None else state_for_mapping
        cx_prev, cy_prev = state_ref[0] + 0.5 * state_ref[2], state_ref[1] + 0.5 * state_ref[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor_rgb
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        mapped = [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]
        mapped = clip_box(mapped, H, W, margin=10)
        return mapped, head_score

    def _maybe_store_window_candidate(self, image_v, image_i, bbox, score):
        if not self.enable_template_update or self.template_update_interval <= 0:
            return
        if score is None:
            return
        candidate = self._make_template_record(image_v, image_i, bbox, score=score)
        if len(self.confidences) < self.template_update_interval:
            self.confidences.append(score)
            self.up_images.append([image_v.copy(), image_i.copy()])
            self.up_bboxs.append(candidate)
        else:
            self.confidences.pop(0)
            self.confidences.append(score)
            self.up_images.pop(0)
            self.up_images.append([image_v.copy(), image_i.copy()])
            self.up_bboxs.pop(0)
            self.up_bboxs.append(candidate)

    def _maybe_update_template(self):
        if not self.enable_template_update or self.template_update_interval <= 0:
            return
        if self.frame_id % self.template_update_interval != 0:
            return
        if not self.confidences:
            print(f"[template_update] frame={self.frame_id} skipped: empty confidence window")
            return
        max_value = max(self.confidences)
        print(f"[template_update] frame={self.frame_id} window_max_score={max_value:.4f}")
        if max_value <= self.template_score_threshold:
            print(
                f"[template_update] frame={self.frame_id} skipped: "
                f"max_score={max_value:.4f} <= threshold={self.template_score_threshold:.4f}"
            )
            return
        max_index = self.confidences.index(max_value)

        previous_template = copy.deepcopy(self.before_update_info)
        candidate_template = copy.deepcopy(self.up_bboxs[max_index])
        print(
            f"[template_update] frame={self.frame_id} try_update: "
            f"old_bbox={previous_template[2]['init_bbox']} "
            f"candidate_bbox={candidate_template['bbox']}"
        )

        self._apply_template(
            self.up_images[max_index][0], self.up_images[max_index][1], candidate_template["bbox"]
        )
        verify_out = self.track_back(
            previous_template[0],
            previous_template[1],
            info=previous_template[2],
        )
        verify_bbox = verify_out["target_bbox"]
        verify_iou = self._bbox_iou(verify_bbox, previous_template[2]["init_bbox"])
        print(
            f"[template_update] frame={self.frame_id} verify: "
            f"tracked_bbox={verify_bbox} ref_bbox={previous_template[2]['init_bbox']} "
            f"iou={verify_iou:.4f}"
        )

        if verify_iou > self.template_iou_threshold:
            self.current_template_info = self._make_template_record(
                self.up_images[max_index][0], self.up_images[max_index][1], candidate_template["bbox"], score=max_value
            )
            self.before_update_info = [
                self.up_images[max_index][0].copy(),
                self.up_images[max_index][1].copy(),
                {"init_bbox": [float(v) for v in candidate_template["bbox"]]},
            ]
            print(
                f"[template_update] frame={self.frame_id} success: "
                f"iou={verify_iou:.4f} > threshold={self.template_iou_threshold:.4f}"
            )
        else:
            self._apply_template(
                previous_template[0], previous_template[1], previous_template[2]["init_bbox"]
            )
            self.current_template_info = self._make_template_record(
                previous_template[0], previous_template[1], previous_template[2]["init_bbox"]
            )
            print(
                f"[template_update] frame={self.frame_id} rollback: "
                f"iou={verify_iou:.4f} <= threshold={self.template_iou_threshold:.4f}"
            )

    def track_with_details(self, image_v, image_i, info: dict = None):
        H, W, _ = image_v.shape
        self.frame_id += 1
        search_bundle = self._prepare_search(image_v, image_i)

        with torch.no_grad():
            out_dict = self.network.forward(
                template=[self.z_dict[0].tensors, self.z_dict[1].tensors],
                search=[search_bundle["search_rgb"].tensors, search_bundle["search_tir"].tensors],
                ce_template_mask=self.box_mask_z,
                gt_bboxs_search=None,
                gt_bboxs_template=self.stste_n,
                inference_strategy=self.inference_strategy,
                soft_temperature=self.soft_temperature,
                tau_rgb=self.tau_rgb,
                tau_tir=self.tau_tir,
                tau_rgbt=self.tau_rgbt,
                return_analysis=self.return_analysis,
                force_tir_stage=self.force_tir_stage,
                force_rgbt_stage=self.force_rgbt_stage,
            )

        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(
            response, out_dict['size_map'], out_dict['offset_map'], return_score=True
        )
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / search_bundle["resize_factor_rgb"]).tolist()
        final_state = clip_box(self.map_box_back(pred_box, search_bundle["resize_factor_rgb"]), H, W, margin=10)

        details = {
            "final_state": final_state,
            "best_score": out_dict.get("best_score", best_score),
            "out_dict": out_dict,
            "search_bundle": search_bundle,
        }

        analysis = out_dict.get("analysis")
        if analysis is not None:
            rgb_box, rgb_head_score = self._decode_branch_output(
                analysis["rgb_out"], self.network.box_head_v, search_bundle["resize_factor_rgb"], image_v.shape
            )
            tir_box, tir_head_score = self._decode_branch_output(
                analysis["tir_out"], self.network.box_head_i, search_bundle["resize_factor_rgb"], image_v.shape
            )
            rgbt_box, rgbt_head_score = self._decode_branch_output(
                analysis["rgbt_out"], self.network.box_head, search_bundle["resize_factor_rgb"], image_v.shape
            )
            details["analysis"] = {
                "rgb_box": rgb_box,
                "tir_box": tir_box,
                "rgbt_box": rgbt_box,
                "rgb_stage_score": self._score_to_float(analysis["rgb_score"]),
                "tir_stage_score": self._score_to_float(analysis["tir_score"]),
                "rgbt_stage_score": self._score_to_float(analysis["rgbt_score"]),
                "rgb_exit": bool(analysis["rgb_exit"]),
                "tir_exit": bool(analysis["tir_exit"]),
                "normal_route": analysis["normal_route"],
                "tir_stage_reached": bool(analysis["tir_stage_reached"]),
                "rgbt_stage_reached": bool(analysis["rgbt_stage_reached"]),
                "rgb_head_score": self._score_to_float(rgb_head_score),
                "tir_head_score": self._score_to_float(tir_head_score),
                "rgbt_head_score": self._score_to_float(rgbt_head_score),
            }
        return details

    def track(self, image_v,image_i, info: dict = None):
        H, W, _ = image_v.shape
        details = self.track_with_details(image_v, image_i, info=info)
        out_dict = details["out_dict"]
        search_bundle = details["search_bundle"]
        self.state = details["final_state"]
        pred_score_map = out_dict["score_map"]
        score_value = self._score_to_float(details["best_score"])
        self._maybe_store_window_candidate(image_v, image_i, self.state, score_value)
        self._maybe_update_template()

        if self.debug:
            raise ValueError('Only non-interactive inference is supported')

        if 'expert' in out_dict.keys():
                source_gate = 'rgbt:' + str(out_dict['source_gate'][0].item())[0:6] + ' ' + 'rgb:' + str(out_dict['source_gate'][1].item())[0:6] + ' ' + 't:' + str(out_dict['source_gate'][2].item())[0:6]

                return {"target_bbox": self.state, "expert": out_dict['expert'], "source_gate": source_gate, "best_score": out_dict["best_score"], "time": out_dict['time']}
        else:
            return {"target_bbox": self.state, "best_score": details["best_score"]}

    def track_back(self, image_v,image_i, info: dict = None):
        H, W, _ = image_v.shape

        x_patch_arr_rgb, resize_factor_rgb, x_amask_arr_rgb = sample_target(
            image_v, info['init_bbox'], self.params.search_factor, output_sz=self.params.search_size
        )
        x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(
            image_i, info['init_bbox'], self.params.search_factor, output_sz=self.params.search_size
        )

        search_rgb = self.preprocessor.process(x_patch_arr_rgb, x_amask_arr_rgb)
        search_tir = self.preprocessor.process(x_patch_arr_tir, x_amask_arr_tir)

        with torch.no_grad():
            x_dict_rgb = search_rgb
            x_dict_tir = search_tir
            out_dict = self.network.forward(
                template=[self.z_dict[0].tensors, self.z_dict[1].tensors],
                search=[x_dict_rgb.tensors, x_dict_tir.tensors],
                ce_template_mask=self.box_mask_z,
                gt_bboxs_search=None,
                gt_bboxs_template=self.stste_n,
                inference_strategy=self.inference_strategy,
                soft_temperature=self.soft_temperature,
                tau_rgb=self.tau_rgb,
                tau_tir=self.tau_tir,
                tau_rgbt=self.tau_rgbt,
            )

        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        pred_boxes, best_score = self.network.box_head.cal_bbox(
            response, out_dict['size_map'], out_dict['offset_map'], return_score=True
        )
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor_rgb).tolist()
        state = clip_box(self.map_box_back_back(pred_box, resize_factor_rgb, info['init_bbox']), H, W, margin=10)

        if 'expert' in out_dict.keys():
            source_gate = 'rgbt:' + str(out_dict['source_gate'][0].item())[0:6] + ' ' + 'rgb:' + str(out_dict['source_gate'][1].item())[0:6] + ' ' + 't:' + str(out_dict['source_gate'][2].item())[0:6]
            return {"target_bbox": state, "expert": out_dict['expert'], "source_gate": source_gate, "best_score": best_score, "time": out_dict['time']}
        else:
            return {"target_bbox": state, "best_score": best_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_back(self, pred_box: list, resize_factor: float, state):
        cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return OSTrack_twobranch
