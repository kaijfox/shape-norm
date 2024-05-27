from ..project.paths import Project
from ..io.dataset_refactor import Dataset

import numpy as np
from scipy.spatial.transform import Rotation
import numpy.linalg as la


def _align_scales(dataset: Dataset, config):
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

    anterior_ixs = [dataset.aux["keypoint_ids"][config["anterior"]]]
    posterior_ixs = [dataset.aux["keypoint_ids"][config["origin"]]]
    absolute_scales = []
    for sess in dataset.sessions:
        anterior_com = dataset.get_session(sess)[:, anterior_ixs].mean(axis=1)
        posterior_com = dataset.get_session(sess)[:, posterior_ixs].mean(axis=1)
        absolute_scales.append(
            np.median(la.norm(anterior_com - posterior_com, axis=-1), axis=0)
        )
    absolute_scales = np.array(absolute_scales)
    scales = absolute_scales / absolute_scales.mean()
    scaled = dataset.update(
        data=dataset.data / scales[dataset.stack_session_ids, None, None]
    )

    return scaled, scales


def _inverse_align_scales(dataset: Dataset, scales: np.ndarray):
    """Invert `_align_scales`.

    Parameters
    ----------
    dataset : Dataset
    scales : array (n_sessions,)
        Relative scale factors of the original sessions.
    """
    return dataset.update(
        data=dataset.data * scales[dataset.stack_session_ids, None, None]
    )


class AlignmentMethod(object):
    pass


class sagittal(AlignmentMethod):
    type_name = "sagittal"

    @staticmethod
    def _align(dataset: Dataset, config):
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
        dataset : Dataset
        config : dict
            `alignment` section of config file.

        Returns
        -------
        dataset : Dataset
        align_meta : dict
            Metadata required to invert alignment. Contains "centroid" and
            "angle", arrays of shape (n_samples, 3) and (n_sampled,),
            respectively, and "scale", an array of shape (n_sessions,).
        """

        # center hips/back or origin-keypt on (0,0,0)
        com = dataset.data[:, [dataset.aux["keypoint_ids"][config["origin"]]]]
        centered = dataset.data - com


        # rotate shoulders/head to align with (1,1,0)
        centered_ant_com = centered[
            :, dataset.aux["keypoint_ids"][config["anterior"]]
        ]
        theta = np.arctan2(centered_ant_com[:, 1], centered_ant_com[:, 0])

        # shape: (t, 3, 3)
        R = Rotation.from_rotvec(
            (-theta[:, None]) * np.array([0, 0, 1])[None, :]
        ).as_matrix()
        rotated = (R[:, None] @ centered[..., None])[..., 0]
        aligned = dataset.update(data=rotated)

        if config["rescale"]:
            scaled, scales = _align_scales(aligned, config)
        else:
            scaled, scales = aligned, None

        return scaled, {"centroid": com, "angle": theta, "scale": scales}

    @staticmethod
    def _inverse(dataset: Dataset, align_meta: dict, config: dict, scale=True):
        """

        Parameters
        ----------
        dataset : Dataset
        align_meta : dict
            Metadata required to invert alignment, as returned by `sagittal_align`
        config : dict
            `alignment` section of config file.
        scale : bool
            If `False`, do not invert rigid scale transformation.

        Returns
        -------
        dataset : Dataset
        """
        if config["rescale"] and scale:
            aligned = _inverse_align_scales(dataset, align_meta["scale"])
        else:
            aligned = dataset

        R = Rotation.from_rotvec(
            (align_meta["angle"][:, None]) * np.array([0, 0, 1])[None, :]
        ).as_matrix()
        rotated = (R[:, None] @ aligned.data[..., None])[..., 0]
        uncentered = rotated + align_meta["centroid"]
        return dataset.update(uncentered)

    defaults = dict(
        origin=None,
        anterior=None,
        rescale=True,
    )

    @staticmethod
    def calibrate(
        dataset: Dataset,
        full_config: dict,
        origin: str = None,
        anterior: str = None,
        rescale: bool = None,
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
        print("[alignment]", "I wonder..")
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
    def _setup_config(config: dict, origin: str, anterior: str, rescale: bool):
        """Validate and complete config from uncalibrated state."""
        if config["origin"] is None:
            config["origin"] = origin
        if config["anterior"] is None:
            config["anterior"] = anterior
        if config["rescale"] is None:
            config["rescale"] = rescale

    @staticmethod
    def _validate_config(config: dict):
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


def align(dataset: Dataset, config: dict):
    """Align keypoint data to egocentric coordinates

    dataset : Dataset
    config : dict
        `features` section of config file.

    Returns
    -------
    dataset : Dataset
    align_inverse : dict
        Metadata required to invert alignment.
    """
    aln_type = config["type"]
    if aln_type not in alignment_types:
        raise NotImplementedError(f"Unknown dataset type: {aln_type}")
    return alignment_types[aln_type]._align(dataset, config)


def invert_align(dataset: Dataset, align_meta: dict, config: dict, **kwargs):
    """Invert alignment of keypoint data to egocentric coordinates."""
    aln_type = config["type"]
    if aln_type not in alignment_types:
        raise NotImplementedError(f"Unknown dataset type: {aln_type}")
    return alignment_types[aln_type]._inverse(
        dataset, align_meta, config, **kwargs
    )
