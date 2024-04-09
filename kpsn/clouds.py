from scipy import spatial
from scipy import special
import numpy as np
import logging
import jax.numpy as jnp


def ball_volume(r, d):
    """Volume of a d-dimensional ball of radius r"""
    normalizer = np.pi ** (d / 2) / special.gamma(d / 2 + 1)
    return normalizer * (r**d)


class PointCloudDensity:
    """Density estimation in a point cloud using a kdtree"""

    _is_cloud = True

    def __init__(self, k, eps=1e-10):
        """Initialize a point cloud density estimator

        Parameters
        ----------
        k : int
            k-th nearest neighbor to use for density estimation
        eps : float
            Small number to avoid division by zero.
        distance_eps : float
            Distance for max independent set reduction when querying a function
            to be averaged. If None, no reduction is performed.
        """
        self._k = k
        self.is_fitted = False
        self._eps = eps
        self._pdf = None
        self._internal_dist = None

    def fit(self, data):
        """Fit a point cloud density estimator"""
        self._tree = spatial.KDTree(data)
        self._n = len(data)
        self._d = data.shape[-1]
        self.is_fitted = True
        return self

    def predict(self, x):
        """Density estimation in a point cloud with pre-prepocessed kdtree

        Parameters
        ----------
        x : array_like (n, d)
            Points to estimate density for

        Returns
        -------
        densities : array_like (n,)
            Estimated densities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        distances, _ = self._tree.query(x, self._k)
        distances = distances[:, -1]
        volumes = ball_volume(distances, self._d)
        if (np.mean(volumes / self._eps) < 1e-3) > 0.5:
            logging.warn(
                "More than half of measured densities are smaller "
                + "than epsilon. Consider increasing."
            )
        return (self._k - 1) / (self._n * (volumes + self._eps))

    def normalized_distance(self, x):
        """Normalized distance measure to nearest points in the cloud"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        distances = self._tree.query(x, self._k)[0][:, -1]
        return distances / self._n

    def measure(self, func):
        """Evaluate func at points in the cloud, for example to compute expectation

        todo: maximal independent set reduction and weighted average

        Parameters
        ----------
        func : callable[array]
            Function to measure. Should take a single argument x of shape (n, d)
            and return a scalar.
        return_evals : bool
            If True, return the values of func at each queried point and weights
            of the queried points in the expectation.

        Returns
        -------
        expectation : float
        evaluations : array_like (n_queried,)
            Values of func at the query points
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        # evaluate function at each point
        query_points = self._tree.data
        evaluations = func(query_points)
        return evaluations

    def __len__(self):
        return self._n

    def pdf(self):
        """Density at each point in the cloud

        Returns
        -------
        pdf : array_like (n_pts,)
            Densities
        """
        if self._pdf is None:
            self._pdf = self.measure(self.predict)
        return self._pdf

    def internal_distances(self):
        if self._internal_dist is None:
            self._internal_dist = self.measure(self.normalized_distance)
        return self._internal_dist


def ball_cloud_js(cloud_a, cloud_b, average=True):
    a = cloud_a
    b = cloud_b

    # kNN density estimate
    # keep track of local distances at each point
    a_dists = a.internal_distances() * a._n
    pdf_a = a._k / a._n / ball_volume(a_dists, a._d)
    b_dists = b.internal_distances() * b._n
    pdf_b = b._k / b._n / ball_volume(b_dists, b._d)

    # kNN density estimates can be very non-normalized when q has significant
    # mass where p is supported but at low probability.
    # this can occur when q is a mixture of p and another distribution, so we
    # so for mixture PDF we rely on a histogram method based on the number
    # of points within a fixed radius
    count_b_at_a = b._tree.query_ball_point(
        a._tree.data, a_dists, return_length=True
    )
    pdf_b_at_a = count_b_at_a / a._n / ball_volume(a_dists, a._d)
    count_a_at_b = a._tree.query_ball_point(
        b._tree.data, b_dists, return_length=True
    )
    pdf_a_at_b = count_a_at_b / b._n / ball_volume(b_dists, b._d)

    # KL divergences to the mixture
    # KL(p||m) = E_p[log(p/m)] = E_p[log(p)] - E_p[log(m)]
    summ_fn = (lambda x: x.mean()) if average else lambda x: x
    kl_a_to_mix = (
        # summ_fn(np.log(pdf_a)) - summ_fn(np.log(0.5 * (pdf_b + pdf_a_at_b)))
        summ_fn(np.log(pdf_a)) - summ_fn(np.log(0.5 * (pdf_a + pdf_b_at_a)))
    )
    kl_b_to_mix = (
        # summ_fn(np.log(pdf_b)) - summ_fn(np.log(0.5 * (pdf_a + pdf_b_at_a)))
        summ_fn(np.log(pdf_b)) - summ_fn(np.log(0.5 * (pdf_b + pdf_a_at_b)))
    )

    if average:
        return 0.5 * (kl_a_to_mix + kl_b_to_mix) / jnp.log(2)
    else:
        return kl_a_to_mix, kl_b_to_mix
