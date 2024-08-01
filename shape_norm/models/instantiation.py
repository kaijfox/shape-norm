from .pose import pose_models
from .morph import morph_models
from .joint import JointModel


def get_model(config):
    """
    Parameters
    ----------
    config : dict
        Full model config file (uses `pose` and `morph` sections).
    """
    pose = pose_models[config["pose"]["type"]]
    morph = morph_models[config["morph"]["type"]]
    return JointModel(pose, morph)
