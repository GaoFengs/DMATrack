import torch
import numpy as np
from lib.utils.misc import NestedTensor

class Preprocessor(object):
    def __init__(self, device=None):
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.449, 0.449, 0.449], device=self.device).view((1, 6, 1, 1))
        self.std = torch.tensor([0.229, 0.224, 0.225, 0.226, 0.226, 0.226], device=self.device).view((1, 6, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):

        img_tensor = torch.tensor(img_arr, device=self.device).float().permute((2,0,1)).unsqueeze(dim=0)
        if img_tensor.shape[1] == 3:
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view((1, 3, 1, 1))
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view((1, 3, 1, 1))
        else:
            self.mean = torch.tensor([0.485, 0.456, 0.406, 0.449, 0.449, 0.449], device=self.device).view((1, 6, 1, 1))
            self.std = torch.tensor([0.229, 0.224, 0.225, 0.226, 0.226, 0.226], device=self.device).view((1, 6, 1, 1))
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std

        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).to(self.device).unsqueeze(dim=0)
        return NestedTensor(img_tensor_norm, amask_tensor)

