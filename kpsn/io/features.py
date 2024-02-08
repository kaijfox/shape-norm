import jax.numpy as jnp
from ..io.dataset import KeypointDataset, FeatureDataset, Dataset
from ..project.paths import Project
from ..config import load_calibration_data, save_calibration_data
from ..pca import fit_with_center, CenteredPCA


class reducer(object):
    type_name = None
    defaults = dict()

    @classmethod
    def reduce(cls: type, dataset: KeypointDataset, config) -> FeatureDataset:
        """Reduce a dataset to a set of features.

        Only valid for alignment method 'locked_pts'.

        Parameters
        ----------
        dataset : FeatureDataset
        config : dict
            `features` section of config file.
        """
        return dataset.as_features().with_data(
            cls.reduce_array(dataset.data, config)
        )

    @classmethod
    def inflate(cls, dataset: FeatureDataset, config: dict):
        """Reconstruct a dataset from a set of features."""
        # reinflate removed axes as zeros
        inflated = dataset.with_data(cls.inflate_array(dataset.data, config))
        return inflated.as_keypoints(config["calibration_data"]["kpt_names"])

    @staticmethod
    def reduce_array(arr, config):
        """Reduce an array of keypoints to an array of features.

        Only valid for alignment method 'locked_pts'.

        Parameters
        ----------
        arr : jnp.ndarray (n_samples, ...)
        config : dict
            `features` section of config file.
        """
        raise NotImplementedError

    @staticmethod
    def inflate_array(arr, config):
        """Reconstruct an array of keypoints from an array of features.

        arr : array, shape (n_samples, n_features) or (n_features,)
            Array of reduced features.
        """
        raise NotImplementedError

    @classmethod
    def calibrate(cls, project: Project, dataset: KeypointDataset, config):
        """Calibrate feature extraction for a dataset.

        Parameters
        ----------
        dataset : KeypointDataset
        config : dict
            Full config
        """
        calib = config["features"]["calibration_data"] = dict(
            kpt_names=dataset.keypoint_names
        )
        calib.update(cls._calibrate(dataset, config))
        save_calibration_data(project.calibration_data(), calib)
        return config

    @staticmethod
    def _calibrate(dataset: KeypointDataset, config):
        """Calibrate feature extraction for a dataset."""
        raise NotImplementedError

    @staticmethod
    def plot_calibration(project: Project, config: dict, colors=None):
        """Plot calibration data."""
        raise NotImplementedError

    @staticmethod
    def _validate_config(config):
        raise NotImplementedError


class locked_pts(reducer):
    type_name = "locked_pts"
    defaults = dict(
        reduce_ixs=None,
        n_kpts=None,
    )

    @staticmethod
    def reduce_array(arr, config):
        """Reduce an array of keypoints to an array of features by removing
        dimensions marked in config"""
        flat_data = arr.reshape((arr.shape[0], -1))
        ix_mask = jnp.isin(
            jnp.arange(flat_data.shape[-1]),
            jnp.array(config["calibration_data"]["reduce_ixs"]),
        )
        return flat_data[..., ~ix_mask]

    @staticmethod
    def inflate_array(arr, config):
        """Reconstruct an array of keypoints from an array of features by
        inserting dimensions."""
        calib = config["calibration_data"]
        inflated_shape = (arr.shape[0], calib["n_kpts"], -1)
        # if passed a single frame
        if arr.ndim < 2:
            arr = arr[None]
            inflated_shape = (calib["n_kpts"], -1)
        # add indices that were dropped
        for reduced_ix in sorted(calib["reduce_ixs"]):
            arr = jnp.insert(arr, reduced_ix, 0, axis=-1)
        # return, reshaped to number of keypoints
        return arr.reshape(inflated_shape)

    @staticmethod
    def _calibrate(dataset: KeypointDataset, config: dict):
        """Calibrate feature extraction for a dataset."""
        # find keypoints with zero std
        flat_data = dataset.as_features()
        reduce_ixs = jnp.where(flat_data.data.std(axis=0) < 1e-5)[0].tolist()

        return dict(
            reduce_ixs=reduce_ixs,
            n_kpts=dataset.n_points,
        )

    @staticmethod
    def plot_calibration(project: Project, config: dict, colors=None):
        """Plot calibration data."""
        return None

    @staticmethod
    def _validate_config(config):
        assert (
            config.get("reduce_ixs") is not None
        ), "Feature reduction not calibrated."


class pcs(reducer):
    """Feature reduction based on principal components of a full dataset."""

    type_name = "pcs"
    defaults = dict(
        calibration=dict(tgt_variance=0.98),
        n_dims=None,
        n_kpts=None,
    )

    @staticmethod
    def reduce_array(arr, config):
        """Reduce an array of keypoints to an array of features."""
        flat_data = arr.reshape((arr.shape[0], -1))
        calib = config["calibration_data"]
        pcs = CenteredPCA(calib["center"], calib["pcs"])
        return pcs.coords(flat_data)[..., : config["n_dims"]]

    @staticmethod
    def inflate_array(arr, config):
        """Reconstruct an array of keypoints from an array of features."""
        calib = config["calibration_data"]
        pcs = CenteredPCA(calib["center"], calib["pcs"])
        # add zeros for any dimensions that were removed
        n_pcs = pcs._pcadata.s.shape[0]
        zeroes = jnp.zeros(arr.shape[:-1] + (n_pcs - config["n_dims"],))
        flat_data = jnp.concatenate([arr, zeroes], axis=-1)
        # transform out of PC space
        flat_arr = pcs.from_coords(flat_data)
        kpt_shape = (config["n_kpts"], flat_arr.shape[-1] // config["n_kpts"])
        return flat_arr.reshape(arr.shape[:-1] + kpt_shape)

    @staticmethod
    def _calibrate(dataset: KeypointDataset, config: dict):
        """Calibrate feature extraction for a dataset."""
        config = config["features"]
        # fit PCA
        flat_data = dataset.as_features()
        pcs = fit_with_center(flat_data.data)

        # choose number of dimensions in reduced data
        scree = (
            jnp.cumsum(pcs._pcadata.variances())
            / pcs._pcadata.variances().sum()
        )
        selected_ix = jnp.argmax(scree > config["calibration"]["tgt_variance"])

        # outputs to main config and calibration_data
        config["n_dims"] = int(selected_ix + 1)
        config["n_kpts"] = dataset.n_points
        return dict(pcs=pcs._pcadata, center=pcs._center)

    @staticmethod
    def plot_calibration(project: Project, config: dict, colors=None):
        """Plot calibration data."""
        from ..viz.general import scree

        config = config["features"]
        calib = config["calibration_data"]
        pcs = CenteredPCA(calib["center"], calib["pcs"])
        cumsum = (
            jnp.cumsum(pcs._pcadata.variances())
            / pcs._pcadata.variances().sum()
        )
        fig, ax = scree(
            cumsum, config["n_dims"], config["calibration"]["tgt_variance"]
        )
        ax.set_title("Feature reduction PCA scree")
        return fig

    @staticmethod
    def _validate_config(config):
        if config.get("n_components") is None:
            raise ValueError("n_components must be set for PCA features.")
        if config.get("n_kpts") is None:
            raise ValueError("n_kpts must be set for PCA features.")


feature_types = {
    locked_pts.type_name: locked_pts,
    pcs.type_name: pcs,
}
default_feature_type = "locked_pts"


def reduce_to_features(dataset: KeypointDataset, config: dict):
    """Reduce keypoint dataset to non-singular features.

    dataset : Dataset
        Dataset whose `ndim` is compatible with the alignment type specified in
        `config`.
    config : dict
        `features` section of config file.
    """
    feat_type = config["type"]
    if feat_type not in feature_types:
        raise NotImplementedError(f"Unknown dataset type: {feat_type}")
    if hasattr(dataset, "_is_dataset"):
        return feature_types[feat_type].reduce(dataset, config)
    else:
        return feature_types[feat_type].reduce_array(dataset, config)


def inflate(dataset, config):
    """Reconstruct a dataset from a set of features."""
    feat_type = config["type"]
    if feat_type not in feature_types:
        raise NotImplementedError(f"Unknown dataset type: {feat_type}")
    if hasattr(dataset, "_is_dataset"):
        return feature_types[feat_type].inflate(dataset, config)
    else:
        return feature_types[feat_type].inflate_array(dataset, config)
