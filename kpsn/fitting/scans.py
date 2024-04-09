from ..config import (
    loads,
    load_project_config,
    load_config,
    flatten,
    deepen,
    save_config,
    load_model_config,
    save_model_config,
)
from ..project.paths import ensure_dirs, Project, recursive_update, create_model
from .methods import fit_types, fit, modify_dataset, load_and_prepare_dataset
from ..io.loaders import load_dataset
from ..io.alignment import align
from ..io.features import reduce_to_features, inflate
from ..io.dataset import FeatureDataset, KeypointDataset
from ..io.utils import split_body_inv
from ..logging import ArrayTrace
from ..models.joint import JointModelParams, JointModel
from ..models.instantiation import get_model
from ..models.util import (
    apply_bodies,
    reconst_errs,
    induced_reference_keypoints,
    _optional_pbar,
)
from ..clouds import PointCloudDensity, ball_cloud_js
from .methods import load_fit

import jax.tree_util as pt
from typing import Tuple
from pathlib import Path
import jax.numpy as jnp
import logging
import shutil
import tqdm


def setup_scan_config(
    project: Project,
    name: str,
    scan_params: dict,
    scan_config_overrides: dict = {},
    model_name_fmt: str = "{scan_name}_{i}",
    model_overrides: dict = {},
):
    """Setup a config for a parameter scan.

    Parameters
    ----------
    config_path : pathlib.Path
        Path to config file.
    scan_params : dict
        Dictionary mapping parameter to a list of values. May be nested or
        flattened, i.e. `fit.n_steps`.
    """

    # preprocess args
    if isinstance(model_name_fmt, str):
        name_fmt_string = model_name_fmt
        model_name_fmt = lambda **kwargs: name_fmt_string.format(**kwargs)
    scan_params = flatten(scan_params)
    n_models = len(scan_params[list(scan_params.keys())[0]])

    # Fill out main scan config
    config = scan_cfg_structure.copy()
    config["models"] = {
        model_name_fmt(scan_name=name, i=i): {
            p_name: p_vals[i] for p_name, p_vals in scan_params.items()
        }
        for i in range(n_models)
    }

    # set up model config modified to run with `split` fit method
    model_config = load_model_config(project.base_model_config())
    model_config = recursive_update(model_config, deepen(model_overrides))
    if model_config["fit"]["type"] != "standard":
        raise ValueError("Scan only supports standard fit type for base model.")
    scan_config = recursive_update(
        fit_types["split"].defaults, deepen(scan_config_overrides)
    )
    model_config["fit"] = {
        "type": "split",
        **scan_config,
        **{"em": model_config["fit"]},
    }

    # create scan directory and save configs
    ensure_dirs(project)
    scan_dir = project.scan(name)
    scan_dir.mkdir(exist_ok=True)
    save_config(scan_dir / "scan.yml", config)
    save_model_config(scan_dir / "base_model.yml", model_config)
    return config, model_config


def run_scan(
    project: Project,
    scan_name: str,
    checkpoint_every: int = 10,
    log_every: int = -1,
    progress: bool = False,
    force_restart: bool = False,
):
    """Load configs and run the desired scan."""
    scan_config = load_config(project.scan(scan_name) / "scan.yml")
    model_config = load_model_config(project.scan(scan_name) / "base_model.yml")
    for model_name, scan_params in scan_config["models"].items():
        if force_restart:
            model_dir = project.model(model_name)
            if model_dir.exists():
                logging.info(f"Removing existing model {model_dir}")
                shutil.rmtree(model_dir)
        model_dir, model_cfg = create_model(
            project,
            model_name,
            config=model_config,
            config_overrides=scan_params,
        )
        fit(model_dir, checkpoint_every, log_every, progress)


scan_cfg_structure = loads(
    """\
    models: null"""
)


def load_scan_dataset(
    project: Project, scan_name
) -> Tuple[FeatureDataset, KeypointDataset]:
    """Load modified dataset for/from a scan.

    Returns
    -------
    dataset : FeatureDataset
        Dataset with modified sessions.
    keypoint_datasset: KeypointDataset
        Inflated dataset with keypoints."""
    scan_cfg = load_config(project.scan(scan_name) / "scan.yml")
    model_name = list(scan_cfg["models"].keys())[0]

    # load split and non-split versions of the dataset
    cfg = load_model_config(project.model_config(model_name))
    dataset = load_dataset(cfg["dataset"])
    dataset_aligned, align_inverse = align(dataset, cfg["alignment"])
    dataset_reduced, reduction_inverse = reduce_to_features(
        dataset_aligned, cfg["features"]
    )
    dataset_train = modify_dataset(project.model(model_name), dataset_reduced)
    return (
        inflate(dataset_train, cfg["features"], reduction_inverse),
        dataset_train,
    )


def _dataset_and_bodies_inv(
    project, model_name=None, return_session_inv=False, allow_subsample=True
):

    if model_name is None:
        model_path = Path(project)
        config_path = model_path / "model.yml"
    else:
        model_path = project.model(model_name)
        config_path = project.model_config(model_name)

    cfg = load_model_config(config_path)
    dataset_orig, _ = load_and_prepare_dataset(
        cfg, modify=False, allow_subsample=allow_subsample
    )
    dataset = modify_dataset(model_path, dataset_orig)
    _inflate = lambda d: inflate(d, cfg["features"])

    if cfg["fit"]["type"] != "split":
        raise ValueError("This analysis is only for fits of type 'split'.")

    # map from original dataset bodies to sessions in the split dataset
    _body_inv, _session_inv = split_body_inv(
        dataset_orig,
        cfg["fit"]["split_all"],
        cfg["fit"]["split_type"],
        cfg["fit"]["split_count"],
    )

    if return_session_inv:
        return dataset, (_body_inv, _session_inv), _inflate
    return dataset, _body_inv, _inflate


def model_withinbody_reconst_errs(
    project,
    model_name,
    dataset=None,
    _body_inv=None,
    _inflate=None,
    progress=False,
):
    """Keypoint errors induced by morphing across examples of the same body
    in a split-dataset scan."""

    # for each body in the dataset, calculate the reconstruction error
    # after morphing to a within-body reference session of the model that
    # did not know these bodies should be identical

    checkpoint = load_fit(project.model(model_name))
    cfg = load_model_config(project.model_config(model_name))
    if dataset is None:
        dataset, _body_inv, _inflate = _dataset_and_bodies_inv(
            project, model_name
        )
    model = get_model(cfg)

    # select session/body for canonical pose space
    global_ref_body = dataset.sess_bodies[dataset.ref_session]

    errs = {}
    pbar = _optional_pbar(_body_inv[b], progress)
    for b in pbar:
        nonref_sessions = _body_inv[b][1:]
        ref_body = dataset.sess_bodies[_body_inv[b][0]]
        # nonref sessions mapped to canonical pose space
        subset = dataset.session_subset(nonref_sessions, bad_ref_ok=True)
        mapped_split_body = apply_bodies(
            model.morph,
            checkpoint["params"].morph,
            subset,
            {s: global_ref_body for s in nonref_sessions},
        )
        mapped_split_body = _inflate(mapped_split_body)
        # pretend all nonref sessions have body `ref_body`, mapped to canonical pose space
        with_ref_body = subset.with_sess_bodies(
            {s: ref_body for s in nonref_sessions}
        )
        mapped_ref_body = apply_bodies(
            model.morph,
            checkpoint["params"].morph,
            with_ref_body,
            {s: global_ref_body for s in nonref_sessions},
        )
        mapped_ref_body = _inflate(mapped_ref_body)

        errs[b] = {
            s: reconst_errs(
                mapped_ref_body.get_session(s), mapped_split_body.get_session(s)
            )
            for s in nonref_sessions
        }

    return errs


def model_withinbody_induced_errs(
    project,
    model_name,
    dataset=None,
    _body_inv=None,
):
    """Keypoint errors induced by morphing across examples of the same body
    in a split-dataset scan."""

    # for each body in the dataset, calculate the reconstruction error
    # after morphing to a within-body reference session of the model that
    # did not know these bodies should be identical

    checkpoint = load_fit(project.model(model_name))
    cfg = checkpoint["config"]
    if dataset is None:
        dataset, _body_inv, _inflate = _dataset_and_bodies_inv(
            project, model_name
        )
    model = get_model(cfg)

    induced_kpts = induced_reference_keypoints(
        dataset,
        cfg,
        model.morph,
        checkpoint["params"].morph,
        to_body=None,  # map to all bodies
        include_reference=True,
    )

    errs = {}
    for b in _body_inv:
        # _body_inv: map (pre-split) body to sessions with that body
        # Now map sessions in _body_inv[b] to their (post-split) body name
        # Also separate out into a reference session within _body_inv[b] (the
        # first) entry and the other sessions
        body_ref = dataset.sess_bodies[_body_inv[b][0]]
        nonref_sessions = _body_inv[b][1:]
        nonref_bodies = [dataset.sess_bodies[s] for s in nonref_sessions]
        # induced_kpts is indexed by (post-split) body names
        # measure errors between the reference session for this (pre-split)
        # body, that is `body_ref` and each of the non-reference sessions
        errs[b] = {
            s: reconst_errs(induced_kpts[b], induced_kpts[body_ref])
            for b, s in zip(nonref_bodies, nonref_sessions)
        }
    return errs


def withinbody_reconst_errs(project, scan_name, progress=False):
    """Keypoint errors induced by morphing across examples of the same body for
    each model in a scan."""
    if isinstance(scan_name, str):
        scan_cfg = load_config(project.scan(scan_name) / "scan.yml")
        models = list(scan_cfg["models"].keys())
    else:
        models = scan_name
    dataset, _body_inv, _inflate = _dataset_and_bodies_inv(project, models[0])
    return {
        model: model_withinbody_reconst_errs(
            project,
            model,
            dataset=dataset,
            _body_inv=_body_inv,
            _inflate=_inflate,
            progress=model if progress else False,
        )
        for model in models
    }


def withinbody_induced_errs(project, scan_name, progress=False):
    """Keypoint errors induced by morphing across examples of the same body for
    each model in a scan."""
    if isinstance(scan_name, str):
        scan_cfg = load_config(project.scan(scan_name) / "scan.yml")
        models = list(scan_cfg["models"].keys())
    else:
        models = scan_name
    dataset, _body_inv, _ = _dataset_and_bodies_inv(project, models[0])
    return {
        model: model_withinbody_induced_errs(
            project,
            model,
            dataset=dataset,
            _body_inv=_body_inv,
        )
        for model in _optional_pbar(models, progress)
    }


def withinsession_induced_errs(project, scan_name, progress=False):
    """Keypoint errors induced by morphing across examples of the same session
    for each model in a scan with split_all = True."""
    if isinstance(scan_name, str):
        scan_cfg = load_config(project.scan(scan_name) / "scan.yml")
        models = list(scan_cfg["models"].keys())
    else:
        models = scan_name
    dataset, (_body_inv, _session_inv), _ = _dataset_and_bodies_inv(
        project, models[0], return_session_inv=True
    )
    return _body_inv, {
        model: model_withinbody_induced_errs(
            project,
            model,
            dataset=dataset,
            _body_inv=_session_inv,
        )
        for model in _optional_pbar(models, progress)
    }


def base_jsds_to_reference(
    project,
    model_name=None,
    dataset=None,
    _body_inv=None,
    ref_cloud=None,
    progress=False,
):
    """
    Compute JSDs to reference session for each body in the dataset.
    """

    assert (
        model_name is not None or dataset is not None
    ), "Need either `model_name` or `dataset`."
    if dataset is None:
        dataset, _body_inv, _ = _dataset_and_bodies_inv(project, model_name)
        ref_cloud = PointCloudDensity(k=15).fit(
            dataset.get_session(dataset.ref_session)
        )

    # transform all sessions to the global reference session's body
    pbar = _optional_pbar(_body_inv, progress)
    jsds = {
        b: {
            s: ball_cloud_js(
                ref_cloud,
                PointCloudDensity(k=15).fit(dataset.get_session(s)),
            )
            for s in _body_inv[b]
        }
        for b in pbar
    }

    return jsds


def model_jsds_to_reference(
    project,
    model_name,
    dataset=None,
    _body_inv=None,
    ref_cloud=None,
    progress=False,
    average=True,
):
    cfg = load_model_config(project.model_config(model_name))
    if dataset is None:
        dataset, _body_inv, _ = _dataset_and_bodies_inv(project, model_name)
        ref_cloud = PointCloudDensity(k=15).fit(
            dataset.get_session(dataset.ref_session)
        )
    model = get_model(cfg)
    checkpoint = load_fit(project.model(model_name))

    # transform all sessions to the global reference session's body
    jsds = {}
    pbar = _optional_pbar(_body_inv, progress)
    for b in pbar:
        ref_body = dataset.sess_bodies[dataset.ref_session]
        subset = dataset.session_subset(_body_inv[b], bad_ref_ok=True)
        mapped = apply_bodies(
            model.morph,
            checkpoint["params"].morph,
            subset,
            {s: ref_body for s in _body_inv[b]},
        )

        # compute JS distances
        jsds[b] = {
            s: ball_cloud_js(
                ref_cloud,
                PointCloudDensity(k=15).fit(mapped.get_session(s)),
            )
            for s in _body_inv[b]
        }

    return jsds


def jsds_to_reference(project, scan_name, progress=False):
    """JS distances to reference session for each model in a scan."""
    if isinstance(scan_name, str):
        scan_cfg = load_config(project.scan(scan_name) / "scan.yml")
        models = list(scan_cfg["models"].keys())
    else:
        models = scan_name
    dataset, _body_inv, _ = _dataset_and_bodies_inv(project, models[0])
    ref_cloud = PointCloudDensity(k=15).fit(
        dataset.get_session(dataset.ref_session)
    )
    model_jsds = {
        model: model_jsds_to_reference(
            project,
            model,
            dataset=dataset,
            _body_inv=_body_inv,
            ref_cloud=ref_cloud,
            progress=str(model) if progress else False,
        )
        for model in models
    }
    base_jsds = base_jsds_to_reference(
        project,
        dataset=dataset,
        _body_inv=_body_inv,
        ref_cloud=ref_cloud,
        progress="Unmorphed",
    )
    return model_jsds, base_jsds, dataset


def merge_param_hist_with_hyperparams(
    model: JointModel, params: JointModelParams, param_hist: ArrayTrace
):
    # extend hyperparameters to match the length of the param_hist
    stat, hype, _ = params.by_type()
    lengthen = lambda arr: jnp.broadcast_to(
        jnp.array(arr)[None], (len(param_hist), *jnp.array(arr).shape)
    )
    long_stat = pt.tree_map(lengthen, stat)
    long_hype = pt.tree_map(lengthen, hype)

    # form model params with batch/step dimension
    full_params = JointModelParams.from_types(
        model, long_stat, long_hype, param_hist._tree
    )
    return full_params


def select_param_step(
    model: JointModel,
    params: JointModelParams,
    param_hist: ArrayTrace,
    step: int,
):
    """Select a single step from a parameter history."""
    stat, hype, _ = params.by_type()
    return JointModelParams.from_types(
        model,
        stat,
        hype,
        pt.tree_map(lambda arr: arr[step], param_hist._tree),
    )
