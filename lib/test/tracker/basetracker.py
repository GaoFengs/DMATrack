import torch
from lib.train.data.processing_utils import transform_image_to_crop


class BaseTracker:
    def __init__(self, params):
        self.params = params
        self.visdom = None

    def transform_bbox_to_crop(self, box_in, resize_factor, device,
                               box_extract=None, crop_type='template'):
        crop_size = self.params.template_size if crop_type == 'template' else self.params.search_size
        box_in = torch.tensor(box_in)
        box_extract = box_in if box_extract is None else torch.tensor(box_extract)
        result = transform_image_to_crop(box_in, box_extract, resize_factor,
                                        torch.Tensor([crop_size, crop_size]), normalize=True)
        return result.view(1, 1, 4).to(device)
