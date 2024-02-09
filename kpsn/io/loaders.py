from .dataset import KeypointDataset
from ..config import save_config, loads
from .features import locked_pts, feature_types
from .alignment import sagittal, alignment_types
from ..project.paths import ensure_dirs
from .utils import UUIDGenerator, select_keypt_ixs

import jax.numpy as jnp
from textwrap import dedent
import jax.tree_util as pt
from pathlib import Path
import logging
import os.path

# Common defaults required by all datasets
datasetdefaults = """
root_path: null
sessions:
    session_name_1:
        data: null"""


# Common defaults to all keypoint datasets
keypt_datasetdefaults = (
    datasetdefaults
    + """
keypoint_names:
- null
keypoint_parents: null
"""
)


def _get_root_path(paths: dict):
    """Get the root path of a set of paths."""
    paths = {k: Path(fp).parts for k, fp in paths.items()}
    first_path = paths[list(paths.keys())[0]]
    root = ()
    while all(fp[: len(root)] == root for fp in paths.values()):
        root = first_path[: len(root) + 1]
    return (
        str(Path(*root[:-1])),
        {k: str(Path(*fp[len(root) - 1 :])) for k, fp in paths.items()},
    )


class DatasetLoader(object):
    @classmethod
    def _write_project_config(
        cls, project, dataset_detail, alignment_type, feature_type
    ):
        """Write a project config for a dataset.

        Parameters
        ----------
        project : kpsn.Project
        dataset_detail : dict
            Dataset-specific config.
        alignment_type : str, default None
            Alignment method to use. If None, will use default for the dataset
            type.
        feature_type : str, default None
            Feature reduction method to use. If None, will use default for the
            dataset type.
        """

        ensure_dirs(project)
        cls._validate_config(dataset_detail)

        if alignment_type is None:
            alignment_type = cls.default_alignment.type_name
        if feature_type is None:
            feature_type = cls.default_features.type_name

        full_cfg = construct_nondataset_project_config(
            os.path.realpath(project.root() / "calibration.p"),
            dataset_detail,
            alignment_type,
            feature_type,
        )
        return save_config(project.main_config(), full_cfg)

    @classmethod
    def load(cls, config, meta_only=False):
        """
        Parameters
        ----------
        config : dict
            Full config.
        """

        cls._validate_config(config)

        # setup
        data = {}
        bodies = {}
        kpt_ixs = select_keypt_ixs(
            config["keypoint_names"], config["use_keypoints"]
        )

        # load data and assign a body name to each session
        for sess, sess_meta in config["sessions"].items():
            data_path = os.path.join(config["root_path"], sess_meta["path"])

            if not meta_only:
                data_arr = cls._load_keypoint_array(data_path)
                if data_arr.shape[1] != len(config["keypoint_names"]):
                    logging.error(
                        f"SEVERE: loaded {data_arr.shape[1]} keypoints, but "
                        f"expected {len(config['keypoint_names'])} from {data_path}."
                    )
                data_arr = data_arr[:, kpt_ixs]
            else:
                data_arr = jnp.empty((0, len(kpt_ixs), 0))

            if config["subsample"] is not None:
                data_arr = data_arr[:: config["subsample"]]
            data[sess] = data_arr

            if sess_meta["body"] is None:
                bodies[sess] = f"body-{sess}"
            else:
                bodies[sess] = sess_meta["body"]

        # form dataset from dictionary of keypoint arrays
        return KeypointDataset.from_arrays(
            data, bodies, config["ref_session"], config["use_keypoints"]
        )

    @staticmethod
    def _load_keypoint_array(path):
        """Load a keypoint array from a file (abstract)"""
        raise NotImplementedError

    @staticmethod
    def _validate_config(config):
        """Validate a dataset config."""
        assert "sessions" in config, "No sessions specified."
        for sess, sess_meta in config["sessions"].items():
            assert (
                "path" in sess_meta
            ), f"No data path specified for session {sess}."
            assert (
                "body" in sess_meta
            ), f"No body specified for session {sess}. Use `null` to assign unique body."
        assert "ref_session" in config, "No reference session specified."
        assert config["ref_session"] in config["sessions"], (
            f"Reference session {config['ref_session']} not in session list"
            f"{list(config['sessions'].keys())}."
        )
        assert "keypoint_names" in config, "No keypoint names specified."
        assert "use_keypoints" in config, "No keypoint filter specified."
        assert (
            "subsample" in config
        ), "No subsample rate specified. Use `null` to disable frame subsampling."


class raw_npy(DatasetLoader):
    """Collection of numpy array files containing keypoint data."""

    @staticmethod
    def _load_keypoint_array(path):
        """Load a keypoint array from a .npy file."""
        return jnp.load(path)

    @classmethod
    def setup_project_config(
        cls,
        project,
        filepaths,
        keypoint_names,
        ref_session,
        use_keypoints=None,
        exclude_keypoints=None,
        keypoint_parents=None,
        bodies=None,
        subsample=None,
        alignment_type=None,
        feature_type=None,
    ):
        """Set up a project config for a raw_npy dataset.

        Parameters
        ----------
        project : kpsn.Project
        filepaths : str, list of str
            List of paths to .npy files containing keypoint data or a directory
            to search for .npy files.
        keypoint_names : list of str
            Ordered list of keypoint names as appearing in the .npy files.
        ref_session : str
            Name of session to use as reference, whose poses should be treated
            as canoncial.
        use_keypoints : list of str, default None
            List of keypoint names to include, or None to use all keypoints.
        exclude_keypoints : list of str, default None
            List of keypoint names to exclude, or None to use all keypoints. If
            `use_keypoints` is passed, this is ignored.
        keypoint_parents : dict[str, Optional[str, None]]
            Mapping of keypoint names to their parent keypoint names, or None
            to indicate no parent, i.e. the root keypoint.
        bodies : list of Union[str, NoneType], default None
            List of body ids to assign to each session, or None to assign
            unique body to each session. Entries may be None to assign unique
            bodies to specific sessions.
        subsample : int, default None
            Reduce frame rate by taking every `subsample`th frame.
        alignment_type : str, default None
            Alignment method to use. If None, will use 'sagittal'.
        feature_type : str, default None
            Feature reduction method to use. If None, will use 'locked_pts'.
        """

        # Process filepaths:
        # - if a directory, search for .npy files
        # - if not a dict, assign names as basename of filepaths
        # - expand relative path and extract root directory
        if isinstance(filepaths, str):
            filepaths = [
                filepaths + "/" + fn
                for fn in os.listdir(filepaths)
                if fn.endswith(".npy")
            ]
        if not isinstance(filepaths, dict):
            filepaths_dict = {}
            for fp in filepaths:
                root_name = os.path.basename(fp)[:-4]
                name = root_name
                i = 0
                while name in filepaths_dict:
                    i += 1
                    name = f"{root_name}_{i}"
                filepaths_dict[name] = fp
            filepaths = filepaths_dict
        filepaths = pt.tree_map(os.path.realpath, filepaths)
        root, filepaths = _get_root_path(filepaths)

        # Process other args: bodies, use/exclude_keypoints, skeleton
        bodies = {n: None for n in filepaths} if bodies is None else bodies
        if use_keypoints is None:
            if exclude_keypoints is None:
                use_keypoints = keypoint_names
            else:
                use_keypoints = [
                    kp for kp in keypoint_names if kp not in exclude_keypoints
                ]
        if not isinstance(keypoint_parents, dict):
            keypoint_parents = dict(zip(use_keypoints, keypoint_parents))

        # Form dataset-specific config
        dataset_detail = dict(
            type="raw_npy",
            root_path=root,
            ref_session=ref_session,
            sessions={
                sess_name: dict(path=fp, body=bodies[sess_name])
                for sess_name, fp in filepaths.items()
            },
            keypoint_names=keypoint_names,
            use_keypoints=use_keypoints,
            subsample=subsample,
            viz=dict(armature=keypoint_parents),
        )

        # Write project config, including alignment and feature defaults
        return cls._write_project_config(
            project, dataset_detail, alignment_type, feature_type
        )

    @staticmethod
    def _validate_config(config):
        DatasetLoader._validate_config(config)

    default_alignment = sagittal
    default_features = locked_pts


dataset_types = dict(
    raw_npy=raw_npy,
)


def load_dataset(config):
    """Load a dataset based on a config dictionary.

    Parameters
    ----------
    config : dict
        `dataset` section of config.
    """
    dset_type = config["type"]
    if dset_type not in dataset_types:
        raise NotImplementedError(f"Unknown dataset type: {dset_type}")
    return dataset_types[dset_type].load(config)


def construct_nondataset_project_config(
    calibration_file,
    dataset_config,
    alignment_type="sagittal",
    feature_type="locked_pts",
):
    # set up ordered yaml
    struct = loads(
        dedent(
            f"""\
        calibration_file: "{str(calibration_file)}"
        dataset:
            type: null
        alignment:
            type: null
        features:
            type: null"""
        )
    )

    struct["dataset"] = dataset_config
    struct["alignment"]["type"] = alignment_type
    struct["alignment"].update(alignment_types[alignment_type].defaults)
    struct["features"]["type"] = feature_type
    struct["features"].update(feature_types[feature_type].defaults)

    return struct
