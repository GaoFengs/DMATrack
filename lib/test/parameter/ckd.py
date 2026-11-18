from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.ckd.config import cfg, update_config_from_file

DEFAULT_CKD_CHECKPOINTS = {
    "CKD_ba_rgbt_base_gate": os.environ.get('DMATRACK_CHECKPOINT', os.path.join(os.path.dirname(__file__), '../../../checkpoints/DMATrack_ep0010.pth.tar')),
}

def parameters(yaml_name: str, checkpoint_path=None):
    params = TrackerParams()
    settings = env_settings()
    prj_dir = settings.prj_dir
    save_dir = settings.save_dir

    yaml_file = os.path.join(prj_dir, 'experiments/ckd/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    if checkpoint_path is not None:
        params.checkpoint = checkpoint_path
    elif yaml_name in DEFAULT_CKD_CHECKPOINTS:
        params.checkpoint = DEFAULT_CKD_CHECKPOINTS[yaml_name]
    else:
        params.checkpoint = os.path.join(
            save_dir,
            "checkpoints/train/ckd/{}/OSTrack_CKD_ep{:04d}.pth.tar".format(yaml_name, cfg.TEST.EPOCH),
        )

    params.save_all_boxes = False
    params.tau_rgb = 0.8
    params.tau_tir = 0.8
    params.tau_rgbt = 0.8
    params.return_analysis = False
    params.force_tir_stage = False
    params.force_rgbt_stage = False
    params.template_update_interval = 50
    params.template_score_threshold = 0.3
    params.template_iou_threshold = 0.4
    params.enable_template_update = True

    return params
