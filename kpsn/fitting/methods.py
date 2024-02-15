from ..models.joint import JointModel, JointModelParams, initialize_joint_model
from ..models.instantiation import get_model
from ..io.utils import split_sessions
from . import em
from ..io.dataset import Dataset
from ..io.loaders import load_dataset
from ..io.features import reduce_to_features
from ..io.alignment import align
from ..config import load_model_config, recursive_eq, load_calibration_data
from ..project.paths import Project

from typing import NamedTuple, Callable, Optional, Tuple
import joblib as jl
from pathlib import Path
import logging


def fit_standard(
    model_dir: Path,
    model: JointModel,
    dataset: Dataset,
    config: dict,
    checkpoint_extra: dict = {},
    checkpoint_every: int = 10,
    log_every: int = -1,
    progress: bool = False,
):
    """Fit a model using EM given a dataset and a loaded config.

    Parameters
    ----------
    config : dict
        Full config dictionary."""

    # if model was partway through fitting, load params and recall metadata
    checkpoint = load_fit(model_dir, silent=True)
    if fit_status(model_dir) == "in_progress" or (
        checkpoint and checkpoint["step"] < config["fit"]["n_steps"] - 1
    ):
        logging.info(
            f"Continuing from checkpoint at step {checkpoint['step']}."
        )
        init_params = checkpoint["params"]
        first_step = checkpoint["step"] + 1
        fit_metadata = checkpoint["meta"]
    else:
        init_params = initialize_joint_model(model, dataset, config)
        first_step = 0
        fit_metadata = None

    checkpoint = em.iterate_em(
        model,
        init_params,
        dataset,
        config["fit"],
        first_step=first_step,
        meta=fit_metadata,
        checkpoint_dir=model_dir,
        checkpoint_every=checkpoint_every,
        checkpoint_extra=checkpoint_extra,
        log_every=log_every,
        progress=progress,
        return_param_hist="trace",
        return_mstep_losses=True,
    )
    return checkpoint


def modify_dataset_split(dataset, cfg):
    """
    Split a dataset into multiple datasets based on the configuration.

    Parameters
    ----------
    dataset : Dataset
    cfg : dict
        `fit` section of config dictionary.
    """
    split_dataset = split_sessions(
        dataset,
        split_all=cfg["split_all"],
        mode=cfg["split_type"],
        count=cfg["split_count"],
        chunk_size=cfg["split_size"],
    )
    return split_dataset.with_sess_bodies(
        {s: f"body-{s}" for s in split_dataset.sessions}, new_names=True
    )


def fit_split(
    model_dir: Path,
    model: JointModel,
    dataset: Dataset,
    config: dict,
    checkpoint_extra: dict = {},
    checkpoint_every: int = 10,
    log_every: int = -1,
    progress: bool = False,
):
    """Fit a model on a dataset with body appearing in more than one session.

    Parameters
    ----------
    config : dict
        `fit` section of config dictionary."""

    standard_config = {**config, "fit": config["fit"]["em"]}
    return fit_standard(
        model_dir,
        model,
        dataset,
        standard_config,
        checkpoint_extra,
        checkpoint_every,
        log_every,
        progress,
    )


def status_standard(model_dir: Path):
    """Get the status of a model fit.

    Parameters
    ----------
    model_dir : pathlib.Path
        Path to model directory.
    """

    config = load_model_config(model_dir / "model.yml")
    checkpoint = load_fit(model_dir, silent=True)
    if not checkpoint:
        return "nonexistent"
    if not recursive_eq(config, checkpoint["config"]):
        return "new_config"
    else:
        return checkpoint["status"]


def load_standard(model_dir: Path):
    """Load a model fit.

    Parameters
    ----------
    model_dir : pathlib.Path
        Path to model directory.
    """
    return jl.load(model_dir / "checkpoint.p")


class FitMethod(NamedTuple):
    type_name: str
    defaults: dict
    modify_dataset: Callable[[Dataset, dict], Dataset]
    run: Callable[
        [
            Path,
            JointModel,
            Dataset,
            dict,
            Optional[int],
            Optional[int],
            Optional[bool],
        ],
        Tuple[JointModelParams, dict],
    ]
    status: Callable[[Path], str]
    load: Callable[[Path], Tuple[JointModelParams, dict]]


standard_method = FitMethod(
    type_name="standard",
    defaults=dict(
        n_steps=100,
        tol=None,
        stop_window=5,
        batch_size=None,
        batch_seed=23,
        update_blacklist=None,
        use_priors=True,
        learning_rate=8,
        scale_lr=True,
        full_dataset_objectives=True,
        mstep=dict(
            reinit_opt=False,
            n_steps=400,
            tol=1e-7,
            stop_window=25,
            batch_size=None,
            batch_seed=29,
            update_max=1e2,
        ),
    ),
    run=fit_standard,
    load=load_standard,
    modify_dataset=lambda dataset, config: dataset,
    status=status_standard,
)

split_method = FitMethod(
    type_name="split",
    defaults=dict(
        split_all=False,
        split_type="consecutive",
        split_count=2,
        split_size=120,
        em=standard_method.defaults,
    ),
    run=fit_split,
    load=load_standard,
    modify_dataset=modify_dataset_split,
    status=status_standard,
)


fit_types = {
    standard_method.type_name: standard_method,
    split_method.type_name: split_method,
}


def fit(
    model_dir: Path,
    checkpoint_every: int = 10,
    log_every: int = -1,
    progress: bool = False,
    force_restart: bool = False,
):
    """Run a model fitting method.

    Config structure:
    method: standard | split
    params:
        <method-specific params> # usually including standard params

    Parameters
    ----------
    model_dir : Path
        Path to model directory.
    checkpoint_every : int
        Number of iterations between checkpoints.
    log_every : int
        Number of iterations between log messages.
    progress : bool
        Whether to display a progress bar.
    force_restart : bool
        Whether to restart fitting from scratch, even if a checkpoint exists.
    """

    # don't go any further if model is up to date with config
    if fit_status(model_dir).startswith("finished") and not force_restart:
        logging.info(f"Model at {model_dir} is already up to date.")
        return load_fit(model_dir)

    # load config/dataset and run preprocessing
    config = load_model_config(model_dir / "model.yml")
    dataset, align_inv = load_and_prepare_dataset(config)
    model = get_model(config)

    fit_type = config["fit"]["type"]
    if fit_type not in fit_types:
        raise NotImplementedError(f"Unknown dataset type: {fit_type}")

    checkpoint = fit_types[fit_type].run(
        model_dir,
        model,
        dataset,
        config,
        checkpoint_extra=dict(
            align_inverse=align_inv,
            config=config,
        ),
        checkpoint_every=checkpoint_every,
        log_every=log_every,
        progress=progress,
    )

    return checkpoint


def load_and_prepare_dataset(
    config: dict, modify=True, all_versions=False, allow_subsample=True
):
    """
    Parameters
    ----------
    config : dict
        Full config dictionary.
    """
    dataset = {}
    dataset["raw"] = load_dataset(
        config["dataset"], allow_subsample=allow_subsample
    )
    dataset["aligned"], align_inverse = align(
        dataset["raw"], config["alignment"]
    )
    dataset["reduced"] = reduce_to_features(
        dataset["aligned"], config["features"]
    )

    if modify:
        fit_type = config["fit"]["type"]
        if fit_type not in fit_types:
            raise NotImplementedError(f"Unknown dataset type: {fit_type}")
        dataset["train"] = fit_types[fit_type].modify_dataset(
            dataset["reduced"], config["fit"]
        )

    if all_versions:
        return dataset, align_inverse
    elif modify:
        return dataset["train"], align_inverse
    else:
        return dataset["reduced"], align_inverse


def modify_dataset(model_dir: Path, dataset):
    """Modify a base dataset according to the fit method."""
    config = load_model_config(model_dir / "model.yml")
    fit_type = config["fit"]["type"]
    if fit_type not in fit_types:
        raise ValueError(f"Unknown fit type {fit_type}")
    return fit_types[fit_type].modify_dataset(dataset, config["fit"])


def fit_status(model_dir):
    """Get the status of a model fit.

    Parameters
    ----------
    model_dir : pathlib.Path
        Path to model directory.
    """

    config = load_model_config(model_dir / "model.yml")
    fit_type = config["fit"]["type"]
    if fit_type not in fit_types:
        raise ValueError(f"Unknown fit type {fit_type}")
    return fit_types[fit_type].status(model_dir)


def load_fit(model_dir: Path, silent=False):
    """Load a model fit.

    Parameters
    ----------
    model_dir : pathlib.Path
        Path to model directory.
    """

    if not (model_dir / "checkpoint.p").exists():
        if not silent:
            raise IOError(f"No checkpoint found in {model_dir}")
        return False
    config = load_model_config(model_dir / "model.yml")
    fit_type = config["fit"]["type"]
    if fit_type not in fit_types:
        raise ValueError(f"Unknown fit type {fit_type}")
    try:
        return fit_types[fit_type].load(model_dir)
    except Exception as e:
        if silent:
            return False
        raise (e)
