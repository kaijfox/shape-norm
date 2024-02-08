from pathlib import Path
from ..config import (
    load_model_config,
    save_model_config,
    deepen,
    recursive_update,
)


class Project:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir)

    def root(self):
        return self.project_dir

    def scan(self, scan_name):
        return self.project_dir / "scans" / scan_name

    def model(self, model_name):
        return self.project_dir / "models" / model_name

    def main_config(self):
        return self.project_dir / "project.yml"

    def base_model_config(self):
        return self.project_dir / "base_model.yml"

    def model_config(self, model_name):
        return self.project_dir / "models" / model_name / "model.yml"

    def calibration_data(self):
        return self.project_dir / "calibration.p"


def ensure_dirs(project):
    project.root().mkdir(parents=True, exist_ok=True)
    (project.root() / "scans").mkdir(exist_ok=True)
    (project.root() / "models").mkdir(exist_ok=True)


def create_model(project: Project, name: str, config=None, config_overrides={}):
    """Create a new model directory and config file."""

    if config is None:
        config = load_model_config(project.base_model_config())

    config = recursive_update(config, deepen(config_overrides))
    model_dir = project.model(name)
    model_dir.mkdir(exist_ok=True)
    save_model_config(project.model_config(name), config)
    return model_dir, config
