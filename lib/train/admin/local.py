"""Legacy checkpoint metadata compatibility with relocatable paths."""
import os
from pathlib import Path


class EnvironmentSettings:
    def __init__(self):
        root = Path(__file__).resolve().parents[3]
        self.workspace_dir = str(root/'output')
        self.tensorboard_dir = str(root/'output/tensorboard')
        self.pretrained_networks = str(root/'checkpoints')
        self.lasher_dir = os.environ.get('LASHER_ROOT', str(root/'data/LasHeR'))
