"""Legacy checkpoint metadata compatibility; not a training recipe."""
from .local import EnvironmentSettings


class Settings:
    def __init__(self):
        self.env = EnvironmentSettings()
        self.use_gpu = True
