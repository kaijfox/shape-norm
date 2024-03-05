from ..project.paths import Project
from ..io.dataset import KeypointDataset

import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
import jax.numpy.linalg as jla


def _align_scales(dataset: KeypointDataset, config):
    """
    Calculate behavior-naiive scales and dilate keypoints to match.

    Parameters
    ----------
    dataset : List[numpy array (frames, dataset, spatial)]

    Returns
    -------
    scaled_dataset :List[numpy array (frames, keypts, spatial)]
        Dilated keypoint data.
    scales : array (sessions)
        Relative scale factors of the original sessions.
    """

    anterior_ixs = [dataset.keypoint_ids[config["anterior"]]]
    posterior_ixs = [dataset.keypoint_ids[config["origin"]]]
    absolute_scales = []
    for sess in dataset.sessions:
        anterior_com = dataset.get_session(sess)[:, anterior_ixs].mean(axis=1)
        posterior_com = dataset.get_session(sess)[:, posterior_ixs].mean(
            axis=1
        )
        absolute_scales.append(
            jnp.median(jla.norm(anterior_com - posterior_com, axis=-1), axis=0)
        )
    absolute_scales = jnp.array(absolute_scales)
    scales = absolute_scales / absolute_scales.mean()
    scaled = dataset.with_data(
        dataset.data / scales[dataset.session_ids, None, None]
    )

    return scaled, scales


def _inverse_align_scales(dataset: KeypointDataset, scales: jnp.ndarray):
    """Invert `_align_scales`.

    Parameters
    ----------
    dataset : KeypointDataset
    scales : array (n_sessions,)
        Relative scale factors of the original sessions.
    """
    return dataset.with_data(
        dataset.data * scales[dataset.session_ids, None, None]
    )


class AlignmentMethod(object):
    pass


class sagittal(AlignmentMethod):
    type_name = "sagittal"

    @staticmethod
    def _align(dataset: KeypointDataset, config):
        """
        Config structure:
        alignment:
            method: locked_pts
            origin_keypt: <str> # keypoint name to use as origin
            anterior_keypt: <str> # keypoint used to determine direction
            scale: bool # scale animals to matching behavior-naive size
        keypoints: <list of str> # ordered keypoint names

        Parameters
        ----------
        dataset : KeypointDataset
        config : dict
            `alignment` section of config file.

        Returns
        -------
        dataset : KeypointDataset
        align_meta : dict
            Metadata required to invert alignment. Contains "centroid" and
            "angle", arrays of shape (n_samples, 3) and (n_sampled,),
            respectively, and "scale", an array of shape (n_sessions,).
        """

        # center hips/back or origin-keypt on (0,0,0)
        com = dataset.data[:, [dataset.keypoint_ids[config["origin"]]]]
        centered = dataset.data - com

        # rotate shoulders/head to align with (1,1,0)
        centered_ant_com = centered[:, dataset.keypoint_ids[config["anterior"]]]
        theta = jnp.arctan2(centered_ant_com[:, 1], centered_ant_com[:, 0])
        # shape: (t, 3, 3)
        R = Rotation.from_rotvec(
            (-theta[:, None]) * jnp.array([0, 0, 1])[None, :]
        ).as_matrix()
        rotated = (R[:, None] @ centered[..., None])[..., 0]
        aligned = dataset.with_data(rotated)

        if config["rescale"]:
            scaled, scales = _align_scales(aligned, config)
        else:
            scaled, scales = aligned, None

        return scaled, {"centroid": com, "angle": theta, "scale": scales}

    @staticmethod
    def _inverse(dataset, align_meta, config):
        """

        Parameters
        ----------
        dataset : KeypointDataset
        align_meta : dict
            Metadata required to invert alignment, as returned by `sagittal_align`
        config : dict
            `alignment` section of config file.

        Returns
        -------
        dataset : KeypointDataset
        """
        if config["rescale"]:
            aligned = _inverse_align_scales(dataset, align_meta["scale"])

        R = Rotation.from_rotvec(
            (align_meta["angle"][:, None]) * jnp.array([0, 0, 1])[None, :]
        ).as_matrix()
        rotated = (R[:, None] @ aligned.data[..., None])[..., 0]
        uncentered = rotated + align_meta["centroid"]
        return dataset.with_data(uncentered)

    defaults = dict(
        origin=None,
        anterior=None,
        rescale=True,
    )

    @staticmethod
    def calibrate(
        dataset, full_config, origin=None, anterior=None, rescale=None
    ):
        """Setup config for a dataset.

        Parameters
        ----------
        full_config : dict
            Full project config.
        """
        sagittal._setup_config(
            full_config["alignment"], origin, anterior, rescale
        )
        aligned_dataset, align_inverse = sagittal._align(
            dataset, full_config["alignment"]
        )
        return aligned_dataset, full_config

    @staticmethod
    def plot_calibration(project: Project, config: dict, colors=None):
        """Plot calibration data.

        Parameters
        ----------
        project : Project
            Full project config.
        config : dict
        """
        return None

    @staticmethod
    def _setup_config(config, origin, anterior, rescale):
        """Validate and complete config from uncalibrated state."""
        if config["origin"] is None:
            config["origin"] = origin
        if config["anterior"] is None:
            config["anterior"] = anterior
        if config["rescale"] is None:
            config["rescale"] = rescale

    @staticmethod
    def _validate_config(config):
        assert (
            config.get("origin") is not None
        ), "Must specify origin keypoint name."
        assert (
            config.get("anterior") is not None
        ), "Must specify anterior keypoint name."
        assert (
            config.get("rescale") is not None
        ), "Must set `rescale` to specify whether or not to rescale sessions."


alignment_types = {
    sagittal.type_name: sagittal,
}
default_alignment_type = "sagittal"


def align(dataset: KeypointDataset, config: dict):
    """Reduce keypoint dataset to non-singular features.

    dataset : Dataset
    config : dict
        `features` section of config file.

    Returns
    -------
    dataset : KeypointDataset
    align_inverse : dict
        Metadata required to invert alignment.
    """
    aln_type = config["type"]
    if aln_type not in alignment_types:
        raise NotImplementedError(f"Unknown dataset type: {aln_type}")
    return alignment_types[aln_type]._align(dataset, config)
