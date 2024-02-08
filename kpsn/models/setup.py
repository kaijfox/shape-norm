from .pose import pose_models
from .morph import morph_models
from ..project.paths import ensure_dirs
from ..config import save_config, loads
from ..fitting.methods import fit_types

import os.path


def setup_base_model_config(
    project, pose_type="gmm", morph_type="lowrank_affine", fit_type="standard"
):
    if pose_type is None:
        pose_type = default_pose_type
    if morph_type is None:
        morph_type = default_morph_type
    if fit_type is None:
        fit_type = default_fit_type

    model_cfg = model_cfg_structure.copy()

    model_cfg["project"] = os.path.realpath(str(project.main_config()))
    model_cfg["pose"]["type"] = pose_type
    model_cfg["pose"].update(pose_models[pose_type].defaults)
    model_cfg["morph"]["type"] = morph_type
    model_cfg["morph"].update(morph_models[morph_type].defaults)
    model_cfg["fit"]["type"] = fit_type
    model_cfg["fit"].update(fit_types[fit_type].defaults)

    ensure_dirs(project)
    save_config(project.base_model_config(), model_cfg)


model_cfg_structure = loads(
    """\
    project: null
    pose:
        type: null
    morph:
        type: null
    fit:
        type: null"""
)
default_pose_type = "gmm"
default_morph_type = "lowrank_affine"
default_fit_type = "standard"
