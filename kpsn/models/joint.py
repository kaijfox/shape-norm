from ..io.dataset import FeatureDataset, PytreeDataset
from ..project.paths import Project

from typing import NamedTuple, Callable, Tuple, Any
from jaxtyping import Array, Float, Scalar, Integer
import logging


def _getter(key):
    return property(lambda self: self._tree[key])


class ModelParams(object):
    """Abstract base class wrapping a dictionary of model parameters."""

    def __init__(self, param_dict: dict):
        self._tree = param_dict

    def by_type(self):
        static = tuple(self._tree[k] for k in self._static)
        hyper = tuple(self._tree[k] for k in self._hyper)
        trained = tuple(self._tree[k] for k in self._trained)
        return (static, hyper, trained)

    @classmethod
    def from_types(cls, static, hyper, trained):
        return cls(
            {
                **{k: v for k, v in zip(cls._static, static)},
                **{k: v for k, v in zip(cls._hyper, hyper)},
                **{k: v for k, v in zip(cls._trained, trained)},
            }
        )


class PoseModelParams(ModelParams):
    """Abstract base class wrapping a dictionary of pose model parameters."""

    n_sessions: int = _getter("n_sessions")
    n_feats: int = _getter("n_feats")
    ref_session: int = _getter("ref_session")

    _static = ["n_sessions", "n_feats"]
    _hyper = ["ref_session"]
    _trained = []


class MorphModelParams(ModelParams):
    """Abstract base class wrapping a dictionary of morph model parameters."""

    n_bodies: int = _getter("n_bodies")
    n_feats: int = _getter("n_feats")
    ref_body: int = _getter("ref_body")

    _static = [
        "n_bodies",
        "n_feats",
    ]
    _hyper = ["ref_body"]
    _trained = []


class PoseModel(NamedTuple):
    type_name: str
    defaults: dict
    ParamClass: type
    discrete_mle: Callable[
        [PoseModelParams, PytreeDataset], Integer[Array, "*#K"]
    ]
    pose_logprob: Callable[[PoseModelParams, PytreeDataset], Float[Array, "*#K n_discrete"]]
    aux_distribution: Callable[
        [PoseModelParams, PytreeDataset], Float[Array, "*#K n_discrete"]
    ]
    log_prior: Callable[[PoseModelParams], Scalar]
    init_hyperparams: Callable[[PytreeDataset, dict], PoseModelParams]
    init: Callable[[PoseModelParams, PytreeDataset, dict], PoseModelParams]
    reports: Callable[[PoseModelParams], dict]
    plot_calibration: Callable[[Project, dict], Any]


class MorphModel(NamedTuple):
    type_name: str
    defaults: dict
    ParamClass: type
    transform: Callable[[MorphModelParams, PytreeDataset], PytreeDataset]
    inverse_transform: Callable[
        [MorphModelParams, PytreeDataset, bool],
        Tuple[PytreeDataset, Float[Array, "*#K"]],
    ]
    log_prior: Callable[[MorphModelParams], dict]
    init_hyperparams: Callable[[PytreeDataset, dict], MorphModelParams]
    init: Callable[[MorphModelParams, PytreeDataset, dict], MorphModelParams]
    reports: Callable[[MorphModelParams], dict]
    apply_bodies: Callable[
        [MorphModelParams, PytreeDataset, Integer[Array, "n_sessions"]],
        PytreeDataset,
    ]
    plot_calibration: Callable[[Project, dict], Any]


class JointModel(NamedTuple):
    pose: PoseModel
    morph: MorphModel


class JointModelParams(object):
    """Wrapper for a pair of morph and pose model parameters."""

    def __init__(self, pose: PoseModelParams, morph: MorphModelParams):
        self.pose = pose
        self.morph = morph

    def by_type(self):
        pose_dicts = self.pose.by_type()
        morph_dicts = self.morph.by_type()
        return tuple((pose_dicts[i], morph_dicts[i]) for i in range(3))

    @staticmethod
    def from_types(model: JointModel, static: dict, hyper: dict, trained: dict):
        morph_params = model.morph.ParamClass.from_types(
            static[1], hyper[1], trained[1]
        )
        return JointModelParams(
            model.pose.ParamClass.from_types(static[0], hyper[0], trained[0]),
            model.morph.ParamClass.from_types(static[1], hyper[1], trained[1]),
        )


def initialize_joint_model(
    model: JointModel, observations: FeatureDataset, config: dict
) -> JointModelParams:
    """Initialize a model based on a config dictionary.

    Parameters
    ----------
    model : JointModel
    observations : FeatureDataset
    config : dict
        Full model config file (uses `pose` and `morph` sections).


    Returns
    -------
    params : JointParams
    """
    pt_obs = PytreeDataset.from_pythonic(observations)

    logging.info("Initializing morph model")
    morph_params = model.morph.init_hyperparams(pt_obs, config["morph"])
    morph_params = model.morph.init(morph_params, pt_obs, config["morph"])

    poses = model.morph.inverse_transform(
        morph_params, pt_obs, return_determinants=False
    )
    logging.info("Initializing pose model")
    pose_params = model.pose.init_hyperparams(pt_obs, config["pose"])
    pose_params = model.pose.init(pose_params, poses, config["pose"])

    return JointModelParams(pose=pose_params, morph=morph_params)
