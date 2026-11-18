"""LasHeR official splits with the project's shared sampling/crop transforms."""
from collections import OrderedDict
from pathlib import Path
import re
import random
import cv2
import numpy as np
import torch
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data.processing import BATProcessing
from lib.train.data.sampler import TrackingSampler
import lib.train.data.transforms as tfm
from lib.utils import TensorDict

class CheckedTrackingSampler(TrackingSampler):
    """Same one-template/one-search causal rule, but expose real I/O errors."""
    def getitem(self):
        assert self.num_template_frames == self.num_search_frames == 1
        assert self.frame_sample_mode == 'causal'
        for _ in range(100):
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            seq_id, visible, info = self.sample_seq_from_dataset(dataset, True)
            search_ids = None
            gap_increase = 0
            for _ in range(1000):
                template_ids = self._sample_visible_ids(visible, 1, min_id=0, max_id=len(visible)-1)
                search_ids = self._sample_visible_ids(visible, 1, min_id=template_ids[0]+1,
                                                     max_id=template_ids[0]+self.max_gap+gap_increase)
                gap_increase += 5
                if search_ids is not None:
                    break
            if search_ids is None:
                raise RuntimeError('Unable to sample a causal pair')
            z, za, meta = dataset.get_frames(seq_id, template_ids, info)
            x, xa, _ = dataset.get_frames(seq_id, search_ids, info)
            data = self.processing(TensorDict(template_images=z, template_anno=za['bbox'],
                search_images=x, search_anno=xa['bbox'], dataset=dataset.get_name(),
                test_class=meta.get('object_class_name')))
            if data['valid']:
                return data
        raise RuntimeError('100 consecutive invalid crops')

class OfficialLasHeR(BaseVideoDataset):
    def __init__(self, root, split, sequence_names=None):
        root = Path(root)
        super().__init__('LasHeR_'+split, str(root))
        list_name, folder = ('trainingsetList.txt', 'trainingset') if split == 'train' else ('testingsetList.txt', 'testingset')
        self.sequence_list = (root/list_name).read_text().splitlines()
        if sequence_names is not None:
            if not set(sequence_names).issubset(self.sequence_list):
                raise ValueError('Requested sequences are outside the official split')
            self.sequence_list = list(sequence_names)
        self.folder = root/folder
        self.records = []
        self.missing_frame_audit = []
        for name in self.sequence_list:
            directory = self.folder/name
            bbox = torch.tensor(np.loadtxt(directory/'init.txt', delimiter=','), dtype=torch.float32)
            frames = [sorted((directory/modality).glob('*.[jp][pn]g')) for modality in ('visible', 'infrared')]
            if any(len(f) != len(bbox) for f in frames):

                aligned = []
                origins = []
                for paths in frames:
                    ids = [int(re.search(r'(\d+)$', p.stem).group(1)) for p in paths]
                    origin = 0 if 0 in ids else 1
                    if len(set(ids)) != len(ids) or not ids or max(ids)-origin >= len(bbox):
                        raise ValueError('Cannot align original frame IDs: '+str(directory))
                    table = {number-origin: path for number, path in zip(ids, paths)}
                    aligned.append([table.get(i) for i in range(len(bbox))])
                    origins.append(origin)
                if origins[0] != origins[1]:
                    raise ValueError('Modalities disagree on frame origin: '+str(directory))
                frames = aligned
                missing = [i for i in range(len(bbox)) if any(f[i] is None for f in frames)]
                self.missing_frame_audit.append(dict(sequence=name, annotation_rows=len(bbox),
                    original_frame_origin=origins[0], unavailable_gt_rows_zero_based=missing,
                    unavailable_original_frame_ids=[i+origins[0] for i in missing]))
            valid = torch.isfinite(bbox).all(1) & (bbox[:, 2:] > 0).all(1)
            for i in range(len(bbox)):
                if any(f[i] is None for f in frames):
                    valid[i] = False
            self.records.append((frames, dict(bbox=bbox, valid=valid, visible=valid.byte())))

    def get_name(self):
        return self.name

    def get_sequence_info(self, seq_id):
        return self.records[seq_id][1]

    def get_frames(self, seq_id, frame_ids, anno=None):
        paths, stored = self.records[seq_id]
        images = []
        for frame in frame_ids:
            pair = [cv2.imread(str(p[frame])) for p in paths]
            if any(image is None for image in pair):
                raise IOError('Unreadable image in '+self.sequence_list[seq_id])
            images.append(np.concatenate([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in pair], axis=2))
        anno = anno if anno is not None else stored
        return images, {k: [v[f].clone() for f in frame_ids] for k, v in anno.items()}, OrderedDict(object_class_name=None)

def make_sampler(cfg, root, split, samples=None, sequence_names=None):
    dataset = OfficialLasHeR(root, split, sequence_names)

    assert cfg.DATA.MEAN[:3] == cfg.DATA.MEAN[3:]
    assert cfg.DATA.STD[:3] == cfg.DATA.STD[3:]
    transform = tfm.Transform(tfm.ToTensorAndJitter(.2) if split == 'train' else tfm.ToTensor(),
                              *([tfm.RandomHorizontalFlip_Norm(.5)] if split == 'train' else []),
                              tfm.Normalize(mean=cfg.DATA.MEAN[:3], std=cfg.DATA.STD[:3]))
    processing = BATProcessing(
        search_area_factor={'template': cfg.DATA.TEMPLATE.FACTOR, 'search': cfg.DATA.SEARCH.FACTOR},
        output_sz={'template': cfg.DATA.TEMPLATE.SIZE, 'search': cfg.DATA.SEARCH.SIZE},
        center_jitter_factor={'template': cfg.DATA.TEMPLATE.CENTER_JITTER, 'search': cfg.DATA.SEARCH.CENTER_JITTER},
        scale_jitter_factor={'template': cfg.DATA.TEMPLATE.SCALE_JITTER, 'search': cfg.DATA.SEARCH.SCALE_JITTER},
        mode='sequence', transform=transform,
        joint_transform=tfm.Transform(tfm.ToGrayscale(0), tfm.RandomHorizontalFlip(.5)))
    count = cfg.DATA.TRAIN.SAMPLE_PER_EPOCH if split == 'train' else cfg.DATA.VAL.SAMPLE_PER_EPOCH
    return CheckedTrackingSampler([dataset], [1], samples or count, cfg.DATA.MAX_SAMPLE_INTERVAL,
                           num_search_frames=1, num_template_frames=1, processing=processing,
                           frame_sample_mode='causal', train_cls=False)

def collate_pairs(rows):
    return {key: torch.stack([r[key][0] for r in rows])
            for key in ('template_images', 'search_images', 'template_anno', 'search_anno')}
