import os
from pathlib import Path

class EnvSettings:
    def __init__(self):
        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

        self.prj_dir = project_root
        self.save_dir = os.path.join(project_root, "output")
        self.results_path = os.path.join(self.save_dir, "test", "tracking_results")
        self.segmentation_path = os.path.join(self.save_dir, "test", "segmentation_results")
        self.network_path = os.path.join(self.save_dir, "test", "networks")
        self.result_plot_path = os.path.join(self.save_dir, "test", "result_plots")
        self.lasher_path = os.environ.get('LASHER_ROOT', str(Path(project_root) / 'data/LasHeR'))

def env_settings(env_num=None):
    del env_num
    return EnvSettings()
