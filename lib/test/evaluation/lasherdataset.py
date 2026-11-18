import os
import re

import numpy as np

from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList
from lib.test.utils.load_text import load_text


class LasHeRDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.lasher_path
        self.sequence_path = os.path.join(self.base_path, "testingset")
        self.sequence_list = self._get_sequence_list()

    def _get_sequence_list(self):
        return sorted(
            [
                name
                for name in os.listdir(self.sequence_path)
                if os.path.isdir(os.path.join(self.sequence_path, name))
            ]
        )

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_dir = os.path.join(self.sequence_path, sequence_name)
        anno_path = os.path.join(sequence_dir, "init.txt")
        ground_truth_rect = load_text(str(anno_path), delimiter=",", dtype=np.float64)

        rgb_dir = os.path.join(sequence_dir, "visible")
        tir_dir = os.path.join(sequence_dir, "infrared")
        rgb_frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
        tir_frames = sorted([f for f in os.listdir(tir_dir) if f.endswith(".jpg")])
        rgb_ids = [re.search(r'(\d+)$', os.path.splitext(name)[0]) for name in rgb_frames]
        tir_ids = [re.search(r'(\d+)$', os.path.splitext(name)[0]) for name in tir_frames]
        aligned = (all(rgb_ids) and all(tir_ids) and
                   [int(m.group(1)) for m in rgb_ids] == [int(m.group(1)) for m in tir_ids])
        if not aligned or len(rgb_frames) != len(ground_truth_rect):
            raise ValueError('Test frames must be complete and aligned with annotations: '+sequence_name)
        frames = list(
            zip(
                [os.path.join(rgb_dir, frame) for frame in rgb_frames],
                [os.path.join(tir_dir, frame) for frame in tir_frames],
            )
        )

        return Sequence(sequence_name, frames, "lasher", ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)
