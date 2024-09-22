from ..joint import PoseModelParams, PoseModel, _getter
from ...io.dataset_refactor import Dataset
from ...config import get_entries, load_calibration_data, save_calibration_data
from ...util import (
    extract_tril_cholesky,
    expand_tril_cholesky,
    sq_mahalanobis,
    InverseWishart,
)
from ...project.paths import Project

from typing import NamedTuple, Tuple, Optional
from jaxtyping import Array, Float, Float32, Integer, Scalar
import jax._src.random as prng
from tensorflow_probability.substrates import jax as tfp
from sklearn import mixture
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import numpy as np
import jax.random as jr
import jax.nn as jnn
import logging


PRNGKey = prng.KeyArray

type_name = "gmm"
defaults = dict(
    n_components=None,
    ref_session=None,
    diag_eps=None,
    pop_weight_uniformity=10,
    subj_weight_uniformity=100,
    wish_var=None,
    wish_dof=None,
    logit_max=5,
    init=dict(count_eps=1e-3, cov_eigen_eps=1e-3, subsample=False, seed=0),
    calibration=dict(
        max_components=10, n_iter=5, sessions="ref", n_repeat=1, mode="bic"
    ),
)


def calibrate_base_model(
    dataset: Dataset, config: dict, n_components=None, wishart=True
):
    """
    Calibrate a pose model on a dataset.

    Parameters
    ----------
    config : dict
        Full model and project config.
    n_components : int
        Do not run BIC analysis to select number of components.
    wishart : bool
        Set up a wishart prior on the covariance matrix.
    """

    full_config = config
    config = config["pose"]
    init_pts = dataset.get_session(dataset.ref_session)

    if n_components is None:
        # set up a range of component numbers to try
        ns = jnp.unique(
            jnp.floor(
                jnp.logspace(
                    0,
                    jnp.log10(config["calibration"]["max_components"]),
                    config["calibration"]["n_iter"],
                )
            )
        ).astype(int)

        if config["calibration"].get("sessions", "ref") == "ref":
            sessions = [dataset.ref_session]
        elif config["calibration"]["sessions"] == "all":
            sessions = list(dataset.sessions)
        sessions = sessions * config["calibration"].get("n_repeat", 1)
        scores = [[] for _ in sessions]

        for j, session in enumerate(sessions):
            for i, n in enumerate(ns):
                if config["calibration"].get("mode", "bic") == "bic":
                    mix = _fit_gmm(
                        dataset,
                        int(n),
                        config["init"]["subsample"],
                        config["init"]["seed"],
                        large_dataset_ok=i != 0,
                        session=session,
                    )
                    pts = dataset.get_session(session)
                    scores[j].append(mix.bic(pts))

                elif config["calibration"]["mode"] == "likelihood":
                    pts = dataset.get_session(session)
                    split = int(len(pts) * 0.8)
                    shuf = jr.permutation(
                        jr.PRNGKey(config["init"]["seed"]) + i + int(j * n), pts
                    )
                    train_pts, val_pts = shuf[:split], shuf[split:]

                    mix = _fit_gmm(
                        None,
                        int(n),
                        config["init"]["subsample"],
                        config["init"]["seed"],
                        large_dataset_ok=i != 0,
                        pts=train_pts,
                    )

                    scores[j].append(
                        [
                            mix.score(val_pts).mean(),
                            mix.score(train_pts).mean(),
                        ]
                    )

        print(np.array(scores).shape)

        if config["calibration"].get("mode", "bic") == "bic":
            scores = jnp.array(scores).mean(axis=0)
            selected_ix = jnp.argmin(scores)
            selected_n = ns[selected_ix]
        elif config["calibration"]["mode"] == "likelihood":
            scores = jnp.array(scores).mean(axis=0)
            selected_ix = jnp.argmax(scores[:, 0])
            selected_n = ns[selected_ix]
        else:
            raise ValueError(
                f"Unknown calibration mode: {config['calibration']['mode']}"
            )

    else:
        ns = [n_components]
        scores = [0]
        selected_ix = 0
        selected_n = n_components

    # save calibration data for plotting
    calib = config["calibration_data"] = dict(
        ns_tested=ns,
        bics=scores,
        best_n=selected_n,
        best_test=selected_ix,
        dim_variances=jnp.var(init_pts, axis=0),
    )

    # update configs
    config["n_components"] = int(ns[selected_ix])
    if not isinstance(config.get("wish_var", None), float) and wishart:
        config["wish_var"] = float(
            jnp.var(init_pts, axis=0).mean() / selected_n
        )
    if not isinstance(config.get("wish_dof", None), (float, int)) and wishart:
        config["wish_dof"] = float(20.0 + dataset.data.shape[-1])

    return full_config


def _fit_gmm(
    dataset: Dataset,
    n_components: int,
    subsample,
    seed,
    large_dataset_ok=False,
    pts=None,
    session=None,
):
    if pts is None:
        session = dataset.ref_session if session is None else session
        init_pts = dataset.get_session(session)
    else:
        init_pts = pts

    if subsample is not False:
        if subsample < 1:
            n_frames = int(len(init_pts) * subsample)
        else:
            n_frames = subsample
        init_pts = jr.choice(
            jr.PRNGKey(seed), init_pts, (n_frames,), replace=False
        )

    if init_pts.shape[0] > 10000 and not large_dataset_ok:
        logging.warn(
            "Reference session contains over 10000 frames. GMM calibration "
            "can be slow and memory intensive, but often does not require the "
            "full dataset. Consider subsampling."
        )

    # logging.info(f"Fitting GMM to {init_pts.shape[0]} frames")
    init_mix = mixture.GaussianMixture(
        n_components=n_components,
        random_state=seed,
    )
    return init_mix.fit(init_pts)


def plot_calibration(config, colors):
    """
    Plot calibration data for a pose model.

    Parameters
    ----------
    config : dict
        Full model and project config.
    """
    from ...viz.util import legend

    config = config["pose"]
    calibration_data = config["calibration_data"]
    mode = config["calibration"].get("mode", "bic")

    fig, ax = plt.subplots(figsize=(3, 2))
    if mode == "bic":
        ax.plot(
            calibration_data["ns_tested"],
            calibration_data["bics"],
            "o-",
            ms=2,
            color=colors.neutral,
            lw=1,
        )
    else:
        for i, (clr, lbl) in enumerate([("C0", "Test"), ("k", "Train")]):
            ax.plot(
                calibration_data["ns_tested"],
                calibration_data["bics"][:, i],
                "o-",
                ms=2,
                color=clr,
                lw=1,
                label=lbl,
            )

    ax.axvline(
        calibration_data["best_n"],
        color=colors.subtle,
        lw=1,
        zorder=-3,
        label="Selected",
    )
    ax.set_xlabel("Number of components")
    if mode == "bic":
        ax.set_yticks(ax.get_ylim())
        ax.set_ylabel("BIC")
        ax.set_yticklabels(["", ""])
    else:
        ax.set_ylabel("Log likelihood")

    ax.set_title("Pose model component selection")
    sns.despine(ax=ax)
    legend(ax)
    return fig


class GMMParams(PoseModelParams):
    # --- static/shape params
    n_components: int = _getter("n_components")
    # --- hyperparams
    diag_eps: float = _getter("diag_eps")
    pop_weight_uniformity: float = _getter("pop_weight_uniformity")
    subj_weight_uniformity: float = _getter("subj_weight_uniformity")
    wish_var: float = _getter("wish_var")
    wish_dof: float = _getter("wish_dof")
    logit_max: float = _getter("logit_max")
    # --- trainable params
    subj_weight_logits: Float[Array, "n_sessions n_components"] = _getter(
        "subj_weight_logits"
    )
    pop_weight_logits: Float[Array, "n_components"] = _getter(
        "pop_weight_logits"
    )
    means: Float[Array, "n_components n_feats"] = _getter("means")
    cholesky: Float[Array, "n_components n_feats**2/2+n_feats/2"] = _getter(
        "cholesky"
    )

    # --- specify how to split params during jitting & differentiation
    _static = PoseModelParams._static + [
        "n_components",
    ]
    _hyper = PoseModelParams._hyper + [
        "diag_eps",
        "pop_weight_uniformity",
        "subj_weight_uniformity",
        "wish_var",
        "wish_dof",
        "logit_max",
    ]
    _trained = PoseModelParams._trained + [
        "subj_weight_logits",
        "pop_weight_logits",
        "means",
        "cholesky",
    ]

    # --- derived parameters

    @property
    def covariances(self) -> Float32[Array, "*#K L M M"]:
        covs = expand_tril_cholesky(self.cholesky, n=self.n_feats)
        # addition of diagonal before extracting cholesky
        if self.diag_eps is not None:
            diag_ixs = jnp.diag_indices(self.n_feats)
            covs = covs.at[..., diag_ixs[0], diag_ixs[1]].add(self.diag_eps)
        return covs

    @property
    def subj_weights(self) -> Float[Array, "*#K N L"]:
        logits = self.subj_weight_logits
        centered = logits - logits.mean(axis=-1)[..., None]
        logit_max = jnp.array(self.logit_max)[
            ..., None, None
        ]  # deal with batched parameter sets
        saturated = logit_max * jnp.tanh(centered / logit_max)
        return jnn.softmax(saturated, axis=-1)

    @property
    def pop_weights(self) -> Float[Array, "*#K L"]:
        logits = self.pop_weight_logits
        centered = logits - logits.mean(axis=-1)[..., None]
        logit_max = jnp.array(self.logit_max)[
            ..., None, None
        ]  # deal with batched parameter sets
        saturated = logit_max * jnp.tanh(centered / logit_max)
        return jnn.softmax(saturated, axis=-1)

    # --- helper functions

    @staticmethod
    def cholesky_from_covariances(
        covariances: Float32[Array, "L M M"], diag_eps: float
    ):
        # undo addition of diagonal before extracting cholesky
        if diag_eps is not None:
            diag_ixs = jnp.diag_indices(covariances.shape[-1])
            covariances = covariances.at[..., diag_ixs[0], diag_ixs[1]].add(
                -diag_eps
            )
        return extract_tril_cholesky(covariances)


ParamClass = GMMParams


def aux_distribution(
    params: GMMParams,
    poses: Dataset,
) -> Tuple[Float[Array, "*#K L"]]:
    """
    Calculate the auxiliary distribution for EM.

    Args:
        observations: Observations from combined model.
        morph_matrix: Subject-wise linear transform from poses to
            keypoints.
        morph_ofs: Subject-wise affine component of transform from
            poses to keypoints.
        params, hyperparams: Parameters of the mixture model.

    Returns:
        probs: Array component probabilities at each data point.
    """

    pose_logits = pose_logprob(params, poses)
    return jnn.softmax(pose_logits, axis=-1)


def pose_logprob(params: GMMParams, poses: Dataset) -> Float[Array, "*#K L"]:
    norm_probs = tfp.distributions.MultivariateNormalFullCovariance(
        loc=params.means,
        covariance_matrix=params.covariances,
    ).log_prob(poses.data[..., None, :])

    component_logprobs = jnp.log(params.subj_weights)[poses.stack_session_ids]

    return norm_probs + component_logprobs


def init_hyperparams(poses: Dataset, config: dict) -> GMMParams:
    wish_var, wish_dof = config["wish_var"], config["wish_dof"]
    return GMMParams(
        dict(
            n_sessions=poses.n_sessions,
            n_feats=poses.data.shape[-1],
            ref_session=poses.ref_session,
            wish_var=float(wish_var) if wish_var is not None else None,
            wish_dof=float(wish_dof) if wish_var is not None else None,
            **get_entries(
                config,
                [
                    "n_components",
                    "diag_eps",
                    "pop_weight_uniformity",
                    "subj_weight_uniformity",
                    "logit_max",
                ],
            ),
        )
    )


def init(
    hyperparams: GMMParams,
    dataset: Dataset,
    config: dict,
) -> GMMParams:
    """
    Initialize a GMMPoseSpaceModel based on observed keypoint data.

    This function uses a single subject's keypoint data to initialize a
    `GMMPoseSpaceModel` based on a standard Gaussian Mixture.

    Args:
        hyperparams: GMMParams
            Hyperparameters of the pose space model.
        poses:
            Pose state latents as given by initialization of a morph model.
        observations: pose.Observations
            Keypoint-space observations to estimate the pose space model from.
        reference_sibject: int
            Subject ID in `observations` to initialize
        count_eps: float
            Count representing no observations in a component. Component weights
            are represented as logits (logarithms) which cannot capture zero
            counts, so instances of zero counts are replaced with `count_eps` to
            indicate a very small cluster weight.
        uniform: bool
            Do not initialize cluster weights
    Returns:
        init_params: GMMParams
            Initial parameters for a `GMMPoseSpaceModel`
    """

    # fit GMM to reference subject
    init_cfg = config["init"]
    init_mix = _fit_gmm(
        dataset,
        hyperparams.n_components,
        init_cfg["subsample"],
        init_cfg["seed"],
    )

    # get component labels & counts across all subjects

    init_components = init_mix.predict(dataset.data)  # List<subj>[(T,)]
    init_counts = np.zeros([hyperparams.n_sessions, hyperparams.n_components])
    for i_subj in range(hyperparams.n_sessions):
        if init_cfg.get("uniform", False):
            lookup_subj = dataset.ref_session
        else:
            lookup_subj = i_subj
        slc = dataset.get_slice(lookup_subj)
        uniq, count = jnp.unique(
            init_components[slc[0] : slc[1]], return_counts=True
        )
        init_counts[i_subj][uniq] = count
    init_counts[init_counts == 0] = init_cfg["count_eps"]
    init_counts = jnp.array(init_counts)

    ref_counts = init_counts[dataset.ref_session]

    # Correct any negative eigenvalues
    # In the case of a non positive semidefinite covariance output by
    # the GMM fit (happens in moderate dimensionality), snap to the
    # nearest (in Frobenius norm) symmetric matrix with eigenvalues
    # not less than `cov_eigenvalue_eps`
    if (
        hyperparams.diag_eps is not None
        and init_cfg["cov_eigen_eps"] < hyperparams.diag_eps
    ):
        init_cfg["cov_eigen_eps"] = hyperparams.diag_eps
    cov_vals, cov_vecs = jnp.linalg.eigh(init_mix.covariances_)
    if jnp.any(cov_vals < 0):
        clipped_vals = jnp.clip(cov_vals, init_cfg["cov_eigen_eps"])
        init_mix.covariances_ = (
            cov_vecs * clipped_vals[..., None, :]
        ) @ jnp.swapaxes(cov_vecs, -2, -1)

    hyperparams._tree.update(
        dict(
            subj_weight_logits=jnp.log(init_counts),
            pop_weight_logits=jnp.log(ref_counts),
            means=jnp.array(init_mix.means_),
            cholesky=GMMParams.cholesky_from_covariances(
                jnp.array(init_mix.covariances_), hyperparams.diag_eps
            ),
        )
    )
    return hyperparams


def discrete_mle(
    estimated_params: GMMParams,
    poses: Dataset,
) -> Integer[Array, "*#K"]:
    """
    Estimate pose model discrete latents given poses.
    """
    dists = sq_mahalanobis(
        poses.data[:, None],
        estimated_params.means[None, :],
        estimated_params.covariances[None, :],
    )
    return (dists.argmin(axis=1),)


def log_prior(
    params: GMMParams,
    observations: Dataset,
    poses: Dataset,
):
    # Heirarchical dirichlet prior on component weights
    if params.n_components > 1:
        pop_weights = params.pop_weights
        pop_logpdf = tfp.distributions.Dirichlet(
            jnp.ones([params.n_components])
            * params.pop_weight_uniformity
            / params.n_components,
        ).log_prob(pop_weights)
        subj_logpdf = tfp.distributions.Dirichlet(
            params.subj_weight_uniformity * pop_weights
        ).log_prob(params.subj_weights)
    else:
        pop_logpdf = jnp.array(0)
        subj_logpdf = jnp.array(0)

    # wishart prior on covariances
    if params.wish_var is not None:
        cov_logpdf = InverseWishart(
            params.wish_dof, params.wish_var * jnp.eye(params.n_feats)
        ).log_prob(params.covariances)
    else:
        cov_logpdf = jnp.array(0)

    return dict(pop_weight=pop_logpdf, subj_weight=subj_logpdf, cov=cov_logpdf)


def reports(params: GMMParams):
    return dict()


model = PoseModel(
    type_name=type_name,
    defaults=defaults,
    ParamClass=ParamClass,
    discrete_mle=discrete_mle,
    pose_logprob=pose_logprob,
    aux_distribution=aux_distribution,
    log_prior=log_prior,
    init_hyperparams=init_hyperparams,
    init=init,
    reports=reports,
    plot_calibration=plot_calibration,
)
