"""
Getting crashes using sklearn PCA on M1 chip - custom implementation.

For fitting with only a partial set of PCs, see scipy.sparse.linalg.svds
"""

from typing import NamedTuple, Tuple
from jaxtyping import Float, Array
import jax.numpy as jnp
import numpy as np
import logging
import optax
import tqdm
import jax


class PCAData(NamedTuple):
    n_fit: int
    s: Float[Array, "*#K n_feats"]
    v: Float[Array, "*#K n_feats n_feats"]

    def __getitem__(self, slc):
        return PCAData(self.n_fit, self.s[..., slc], self.v[..., slc])

    def pcs(self) -> Float[Array, "*#K n_feats n_feats"]:
        """
        Returns
        -------
        pcs: (..., n_components = n_feats, n_feats)
            Array of normalized principal components.
            The `i`th component vector is at position `i` along the
            second to last dimension.
        """
        return jnp.swapaxes(self.v, -2, -1)

    def variances(self):
        """
        Returns:
        vars: (..., n_feats)
            Variance explained by each principal component.
        """
        return self.s**2 / (self.n_fit)

    def coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        """
        Returns:
        coords: (..., n_pts, n_feats)
            Coordinates in principal component space.
        Each row (second to last dimension) of the array is a vector
        that when right-multiplied by $V$, reconstructs the
        corresponding row of `arr`.
        """

        """
        Derivation:
        X = USV'
        columns of V are singular vectors
        coords in are column vectors to left-multiply by V
            or row vectors to right-multiply by V'
        CV' = X => C = XV = USV'V = US
        For a different matrix of observations, we do the same:
        CV' = A => C = AV
        """
        return arr @ self.v

    def whitened_coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        """
        Whitened coords have second-moment 1 alond each axis.
        Returns:
        coords: (..., n_pts, n_feats)
            Coordinates in whitened principal component space.
        Each row (second to last dimension) of the array is a vector
        that when right-multiplied by $V$, reconstructs the
        corresponding row of `arr`.
        """

        """
        Derivation:
        lambda_i = s_i ** 2 / (n - 1) are variances of PCs
        sqrt(lambda_i) = s_i / sqrt(n - 1) are stddevs of PCs
        AV are coords for data matrix A of row-vector points
        diag(s_i / sqrt(n - 1))^{-1} @ AV gives whitened coords
        """
        norm = jnp.sqrt(self.n_fit)
        return self.coords(arr) / (self.s[..., None, :] / norm)

    def from_coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        """
        Returns:
        pts: (..., n_pts, n_feats)
            Points transformed back from PC space.

        Coordinate vector achieved via C = AV
        To retrieve data points, calculate A = CV'
        """
        return arr @ jnp.swapaxes(self.v, -2, -1)

    def from_whitened_coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        """
        Returns:
        pts: (..., n_pts, n_feats)
            Points transformed back from whitened PC space.

        Coordinate vector achieved via C = diag(s_i / sqrt(n - 1))^{-1} @ AV
        To retrieve data points, calculate A = diag(s_i / sqrt(n - 1)) @CV'
        """
        norm = jnp.sqrt(self.n_fit)
        vt = jnp.swapaxes(self.v, -2, -1)
        return (arr * (self.s[..., None, :] / norm)) @ vt


class CenteredPCA:
    def __init__(self, center, pcadata):
        self._pcadata = pcadata
        self._center = center

    def from_coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        return self._pcadata.from_coords(arr) + self._center[..., None, :]

    def whitened_coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        return self._pcadata.whitened_coords(arr - self._center[..., None, :])

    def coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        return self._pcadata.coords(arr - self._center[..., None, :])


def fit(
    data: Float[Array, "*#K n_samples n_feats"],
    sign_correction: str = None,
) -> PCAData:
    """
    Parameters:
        centered: boolean
            Whether sample mean of the data is zero in all features.
            If it is not, then components will be given a canonical
            orientation (+/-) such that the mean of the data has positive
            coordinates."""

    cov = data.T @ data
    _, s2, vt = np.linalg.svd(cov)

    if sign_correction is not None:
        if sign_correction == "mean":
            # coords for mean of data
            standard_vec = data.mean(axis=-2)
        if sign_correction == "ones":
            standard_vec = jnp.ones(data.shape[:-2] + (data.shape[-1],))
        # standard_vec: (..., n_components)
        # coord_directions: (..., n_components)
        coord_directions = jnp.sign(
            standard_vec[..., None, :] @ jnp.swapaxes(vt, -2, -1)
        )[..., 0, :]
        # coords for vector of ones
        # flip PCs acoording to sign of mean coords
        # (..., n_components, n_feats)
        vt = coord_directions[..., :, None] * vt

    return PCAData(
        data.shape[-2], np.sqrt(s2), jnp.array(jnp.swapaxes(vt, -2, -1))
    )


def fit_with_center(
    data: Float[Array, "*#K n_samples n_feats"],
    sign_correction: str = None,
) -> CenteredPCA:
    center = data.mean(axis=-2)
    pcs = fit(data - center[..., None, :], sign_correction=sign_correction)
    return CenteredPCA(center, pcs)


def second_moment(arr: Float[Array, "*#K n_samples n_feats"]):
    return jnp.swapaxes(arr, -2, -1) @ arr / (arr.shape[-2])


def _covariance_alignment_objective(
    ref_data, alt_data, n_upd, alpha=0, centered=False, ref_pc=None
):
    """
    Parameters:
    ref_data, alt_data : jnp.ndarray, shape (..., n_samples, n_dims)
        The reference and alternative datasets, with samples arranged in rows.
    n_upd : int
        The number of principal components allowed to update
    """
    T = lambda x: jnp.swapaxes(x, -2, -1)
    if not centered:
        ref_data = ref_data - ref_data.mean(axis=-2, keepdims=True)
        alt_data = alt_data - alt_data.mean(axis=-2, keepdims=True)
    ref_cov = T(ref_data) @ ref_data / ref_data.shape[-2]
    if ref_pc is None:
        ref_pc, _, _ = jnp.linalg.svd(ref_cov)
        ref_pc = ref_pc[..., :n_upd]
    else:
        n_upd = ref_pc.shape[-1]
    alt_cov = T(alt_data) @ alt_data / alt_data.shape[-2]
    N = ref_data.shape[-1]
    I = jnp.eye(N).reshape((1,) * (ref_data.ndim - 2) + (N, N))

    @jax.jit
    def _objective(upds):
        """
        Parameters:
        upds : jnp.ndarray, shape (..., n_dims, n_upd <= n_dims)
            The updates to the principal components.
        """
        # P[:n_upd] + P_hat[:n_upd]
        updated = I + upds @ T(ref_pc)
        # P + P_hat
        # (I + P_hat P') X X' (I + P P_hat')
        morphed = updated @ ref_cov @ T(updated)
        reg = alpha * (upds**2).mean()
        return ((morphed - alt_cov) ** 2).mean() + reg

    return ref_pc, _objective


def _check_should_stop_early(loss_hist, step_i, tol, stop_window):
    """
    Check for median decrease in loss that is worse than `tol`.

    Args:
        loss_hist: np.ndarray, shape: (N,)
            Last `N` observations of loss.
        tol: Scalar
            Required average improvement to continue.
    """
    if tol is not None and step_i > stop_window:
        loss_hist = loss_hist[step_i - stop_window : step_i + 1]
        diff = np.diff(loss_hist)
        median = np.median(diff)
        return median > -tol
    return False


def fit_covariance_alignment(
    ref_data,
    alt_data,
    n_upd,
    alpha=1e-3,
    lr=1e-2,
    max_iter=5000,
    stop_window=100,
    tol=1e-5,
    progress=False,
    ref_pc=None,
):
    """
    Parameters
    ----------
    ref_data : jnp.ndarray, shape (samples, features)
        The reference data to map onto alternative covariance structures.
    alt_data : jnp.ndarray, shape (n_alts, samples, features)
        The alternative data to map onto.
    n_upd : int
        The number of principal components to update.
    alpha : float
        The regularization strength.
    lr : float
        The learning rate.
    max_iter : int
        The maximum number of iterations.
    tol, stop_window : int
        The criteria for early stopping: if loss has not decreased by more than
        `tol` for `stop_window` steps, stop early.
    progress : bool
        Whether to display a progress bar.
    ref_pc : jnp.ndarray, shape (features, n_pc <= n_features)
        Precomputed othonormal dimensions of the reference data to be adjusted.
        Not checked for orthonormality. If provided, `n_upd` is ignored and set
        to the last dimension of `ref_pc`.


    Returns
    -------
    ref_pc : jnp.ndarray, shape (features, n_pc)
        The principal components of the reference data.
    upds_best : jnp.ndarray, shape (n_alts, features, n_upd)
        The best updates to the principal components.
    ref_center, alt_center : jnp.ndarray, shapes (features,), (n_alts, features)
        The means of the reference and alternative datasets.
    """

    ref_center = ref_data.mean(axis=0)
    alt_center = alt_data.mean(axis=1)
    ref_data = (
        ref_data[None] - ref_center[None, None]
    )  # shape: (1, samples, features)
    alt_data = (
        alt_data - alt_center[:, None]
    )  # shape: (n_alts, samples, features)

    opt = optax.adam(learning_rate=lr)
    ref_pc, obj = _covariance_alignment_objective(
        ref_data, alt_data, n_upd, alpha, centered=True, ref_pc=ref_pc
    )
    upds_init = jnp.zeros(
        (
            len(alt_data),
            alt_data.shape[-1],
            n_upd,
        )
    )

    losses = np.zeros(max_iter)
    best_loss = np.inf
    upds_curr = upds_init
    opt_state = opt.init(upds_init)
    upds_best = upds_curr

    reason = f"max iter ({max_iter}) reached"
    pbar = tqdm.trange(max_iter) if progress else range(max_iter)
    for i in pbar:
        loss, grad = jax.value_and_grad(obj)(upds_curr)
        losses[i] = loss
        param_step, opt_state = opt.update(grad, opt_state)
        upds_curr = optax.apply_updates(upds_curr, param_step)

        if loss < best_loss:
            best_loss = loss
            upds_best = upds_curr

        if _check_should_stop_early(losses, i, tol, stop_window):
            reason = "converged"
            break
        if not jnp.isfinite(loss):
            reason = f"diverged at step {i}"
            break

    if reason != "converged":
        logging.warning(f"Maximal PC align did not converge, {reason}.")
        upds_best = jnp.zeros_like(upds_best)

    return ref_pc[0], upds_best, ref_center, alt_center


def covariance_alignment_transform(ref_pc, upds):
    """
    Construct mapping from PCs and updates as calculated by
    `fit_covariance_alignment`.

    Note the resulting transform assumes the centroids `ref_center` and
    `alt_center` have been subtracted from input data.

    Parameters
    ----------
    ref_pc : jnp.ndarray, shape (features, n_pc)
        The principal components of the reference data.
    upds : jnp.ndarray, shape (n_alts, features, n_upd)
        The updates to the principal components.

    Returns
    -------
    morphs : jnp.ndarray, shape (n_alts, features, features)
        The morphing matrices for each alternative dataset.
    """
    I = jnp.eye(ref_pc.shape[-2])[None]
    T = lambda x: jnp.swapaxes(x, -2, -1)
    return I + upds @ T(ref_pc)
