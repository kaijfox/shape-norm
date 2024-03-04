from .pose import pose_models
from .morph import morph_models
from ..project.paths import ensure_dirs
from ..config import save_config, loads
from ..fitting.methods import fit_types

import os, os.path


def setup_base_model_config(
    project,
    file_path=None,
    pose_type="gmm",
    morph_type="lowrank_affine",
    fit_type="standard",
    project_config=None,
):
    """
    Parameters
    ----------
    project : kpsn.project.Paths or str or Path
        Project paths object, or string/Path giving the location of the project
        config. If a string or Path is given, then `file_path` is required.
    file_path : str or Path, optional
        Path to write config. If `project` is a string or Path, then this is
        required.
    """

    if file_path is None:
        if isinstance(project, (str, os.PathLike)):
            raise ValueError(
                "File path is required when project is a string or PathLike."
            )
        project_path = project.main_config()
        ensure_dirs(project_path)
        file_path = project.setup_base_model_config()
    else:
        project_path = project

    if pose_type is None:
        pose_type = default_pose_type
    if morph_type is None:
        morph_type = default_morph_type
    if fit_type is None:
        fit_type = default_fit_type

    model_cfg = model_cfg_structure.copy()

    model_cfg["project"] = os.path.realpath(str(project_path))
    model_cfg["pose"]["type"] = pose_type
    model_cfg["pose"].update(pose_models[pose_type].defaults)
    model_cfg["morph"]["type"] = morph_type
    model_cfg["morph"].update(morph_models[morph_type].defaults)
    model_cfg["fit"]["type"] = fit_type
    model_cfg["fit"].update(fit_types[fit_type].defaults)

    save_config(file_path, model_cfg)


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
