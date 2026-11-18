"""Historical LasHeR result profile; independent of current ablations."""
from .ckd import parameters as base_parameters

def parameters(yaml_name, checkpoint_path=None):
    params = base_parameters(yaml_name, checkpoint_path)
    params.cfg.MODEL.BACKBONE.PARAMS.rgbt_loc = list(range(1, 12))
    params.inference_strategy = 'always_rgbt'
    params.enable_template_update = True
    return params
