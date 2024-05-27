from .dataset_refactor import Dataset
from ..config import save_config, loads
from .features import locked_pts, feature_types
from .alignment import sagittal, alignment_types
from ..project.paths import ensure_dirs
from .utils import UUIDGenerator, select_keypt_ixs

import numpy as np
from textwrap import dedent
import jax.tree_util as pt
from pathlib import Path
import logging
import os.path


def _find_files(filepaths, ext):
    """Locate files with a given extension.

    Parameters
    ----------
    filepaths : str, list of str, dict[str, str]
        File paths or directory to search for files. If a string, will be
        interpreted as a directory to search for files. If a dictionary, is
        interpeted as a mapping from session names to file paths. If a list, is
        treated like a dictionary with session names taken to be file basename.
    Returns
    -------
    root_path : str
        Common root path of found files.
    filepaths : dict[str, str]
        Mapping of session names to file paths."""
    if isinstance(filepaths, str):
        filepaths = [
            filepaths + "/" + fn
            for fn in os.listdir(filepaths)
            if fn.endswith(ext)
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
    return _get_root_path(filepaths)


def _get_root_path(paths: dict):
    """Get the root path of a set of paths."""
    paths = {k: Path(fp).parts for k, fp in paths.items()}
    first_path = paths[list(paths.keys())[0]]
    root = ()
    i = 0
    while all(fp[: len(root)] == root for fp in paths.values()):
        root = first_path[: len(root) + 1]
        i += 1
        if i > 100:
            root = ()
            break
    if len(root):
        return (
            str(Path(*root[:-1])),
            {k: str(Path(*fp[len(root) - 1 :])) for k, fp in paths.items()},
        )
    else:
        return "", {k: str(Path(*fp)) for k, fp in paths.items()}


def _session_file_config(
    filepaths,
    ref_session,
    bodies=None,
    keypoint_names=None,
):
    """Generate common config for datasets based on one session / file.

    Parameters
    ----------
    keypoint_names : list of str
        Ordered list of keypoint names as appearing in the files, or callable
        which will be passed an example file and should return the keypoint
        names.
    """

    root, filepaths = _find_files(filepaths, ".npy")
    bodies = {n: None for n in filepaths} if bodies is None else bodies

    if callable(keypoint_names):
        keypoint_names = keypoint_names(
            os.path.join(root, list(filepaths.values())[0])
        )

    return (
        dict(
            root_path=root,
            ref_session=ref_session,
            sessions={
                sess_name: dict(path=fp, body=bodies[sess_name])
                for sess_name, fp in filepaths.items()
            },
        ),
        keypoint_names,
    )


def _keypoint_names_config(
    keypoint_names,
    use_keypoints=None,
    exclude_keypoints=None,
    keypoint_parents=None,
    anterior=None,
    posterior=None,
    invert_axes=None,
):
    """Generate common dataset config components for keypoint data."""

    # use/exclude_keypoints
    if use_keypoints is None:
        if exclude_keypoints is None:
            use_keypoints = keypoint_names
        else:
            use_keypoints = [
                kp for kp in keypoint_names if kp not in exclude_keypoints
            ]
    # armature as dictionary
    if not isinstance(keypoint_parents, dict):
        keypoint_parents = dict(zip(use_keypoints, keypoint_parents))
    else:
        keypoint_parents = {
            c: p for c, p in keypoint_parents.items() if c in use_keypoints
        }

    return dict(
        keypoint_names=keypoint_names,
        use_keypoints=use_keypoints,
        anterior=anterior,
        posterior=posterior,
        invert_axes=(
            (invert_axes,)
            if isinstance(invert_axes, int) and invert_axes is not None
            else invert_axes
        ),
        viz=dict(armature=keypoint_parents),
    )


def _extract_keypoints_from_array(
    keypoint_array,
    keypoint_names,
    use_keypoints,
    invert_axes,
    subsample,
    subsample_to,
    allow_subsample=True,
    session_name_for_error=None,
):
    data_arr = keypoint_array
    kpt_ixs = select_keypt_ixs(keypoint_names, use_keypoints)

    if data_arr.shape[1] != len(keypoint_names):
        logging.error(
            f"SEVERE: loaded {data_arr.shape[1]} keypoints, but "
            f"expected {len(keypoint_names)}"
            + (
                f" from {session_name_for_error}."
                if session_name_for_error
                else "."
            )
        )
    data_arr = data_arr[:, kpt_ixs]
    if invert_axes is not None:
        inv_ix = invert_axes
        data_arr[:, :, inv_ix] = -data_arr[:, :, inv_ix]

    if allow_subsample:
        if subsample_to is not None:
            N = data_arr.shape[0]
            tgt = subsample_to
            if N > tgt:
                data_arr = data_arr[: tgt * (N // tgt) : N // tgt]
            else:
                logging.warn(
                    f"Session " + f"{session_name_for_error} "
                    if session_name_for_error
                    else "" + f"has fewer frames than `subsample_to`"
                    f" (={tgt}). Subsampling disabled for this session."
                )
        elif subsample is not None:
            data_arr = data_arr[::subsample]

    return data_arr


class DatasetLoader(object):
    @classmethod
    def _write_project_config(
        cls, project, dataset_detail, alignment_type, feature_type
    ):
        """Write a project config for a dataset.

        Parameters
        ----------
        project : kpsn.Project or str or pathlib.Path
        dataset_detail : dict
            Dataset-specific config.
        alignment_type : str, default None
            Alignment method to use. If None, will use default for the dataset
            type.
        feature_type : str, default None
            Feature reduction method to use. If None, will use default for the
            dataset type.
        """

        cls._validate_config(dataset_detail)

        if alignment_type is None:
            alignment_type = cls.default_alignment.type_name
        if feature_type is None:
            feature_type = cls.default_features.type_name

        if isinstance(project, (str, Path)):
            abspath = Path(os.path.realpath(str(project)))
            calib_path = abspath.parent / f"{abspath.stem}.calib.p"
        else:
            calib_path = os.path.realpath(project.root() / "calibration.p")

        full_cfg = construct_nondataset_project_config(
            calib_path,
            dataset_detail,
            alignment_type,
            feature_type,
        )

        if isinstance(project, (str, Path)):
            return save_config(project, full_cfg)
        else:
            ensure_dirs(project)
            return save_config(project.main_config(), full_cfg)

    @classmethod
    def from_arrays(cls, data, config, meta_only=False, allow_subsample=True):
        cls._validate_config(config)

        # setup
        data_arranged = {}
        bodies = {}

        # load data and assign a body name to each session
        for sess, sess_meta in config["sessions"].items():

            if not meta_only:
                data_arranged[sess] = _extract_keypoints_from_array(
                    data[sess],
                    config["keypoint_names"],
                    config["use_keypoints"],
                    config["invert_axes"],
                    config["subsample"],
                    config.get("subsample_to", None),
                    allow_subsample,
                    sess,
                )
            else:
                data_arranged[sess] = np.empty(
                    (0, len(config["use_keypoints"]), 0)
                )

            if sess_meta["body"] is None:
                bodies[sess] = f"body-{sess}"
            else:
                bodies[sess] = sess_meta["body"]

        # form dataset from dictionary of keypoint arrays
        return Dataset.from_arrays(
            data_arranged,
            bodies,
            config["ref_session"],
            {
                "keypoint_names": config["use_keypoints"],
                "keypoint_ids": {
                    kp: i for i, kp in enumerate(config["use_keypoints"])
                },
            },
        )

    @classmethod
    def load(cls, config, meta_only=False, allow_subsample=True):
        """
        Parameters
        ----------
        config : dict
            `dataset` sections of config.
        """

        cls._validate_config(config)

        # setup
        data = {}
        bodies = {}

        # load data and assign a body name to each session
        for sess, sess_meta in config["sessions"].items():
            data_path = os.path.join(config["root_path"], sess_meta["path"])

            if not meta_only:
                data[sess] = _extract_keypoints_from_array(
                    cls._load_keypoint_array(data_path, config),
                    config["keypoint_names"],
                    config["use_keypoints"],
                    config["invert_axes"],
                    config["subsample"],
                    config.get("subsample_to", None),
                    allow_subsample,
                    sess,
                )
            else:
                data[sess] = np.empty((0, len(config["use_keypoints"]), 0))

            if sess_meta["body"] is None:
                bodies[sess] = f"body-{sess}"
            else:
                bodies[sess] = sess_meta["body"]

        # form dataset from dictionary of keypoint arrays
        return Dataset.from_arrays(
            data,
            bodies,
            config["ref_session"],
            {
                "keypoint_names": config["use_keypoints"],
                "keypoint_ids": {
                    kp: i for i, kp in enumerate(config["use_keypoints"])
                },
            },
        )

    @staticmethod
    def _load_keypoint_array(path, config):
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
        assert (config["subsample"] is None) or (
            config.get("subsample_to", None) is None
        ), "One of subsample or subsample_to must be null."


class raw_npy(DatasetLoader):
    """Collection of numpy array files containing keypoint data."""

    type_name = "raw_npy"

    @staticmethod
    def _load_keypoint_array(path, config):
        """Load a keypoint array from a .npy file."""
        return np.load(path)

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
        anterior=None,
        posterior=None,
        invert_axes=None,
        subsample=None,
        subsample_to=None,
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
        subsample_to : int, default None
            Reduce frame rate by taking every `n`th frame where `n` is chosen
            for each session such that the total number of frames in each loaded
            session is `subsample_to`.
        alignment_type : str, default None
            Alignment method to use. If None, will use 'sagittal'.
        feature_type : str, default None
            Feature reduction method to use. If None, will use 'locked_pts'.
        """

        # form session file data
        session_detail, found_kp_names = _session_file_config(
            filepaths, ref_session, bodies, keypoint_names
        )
        keypoint_names = found_kp_names or keypoint_names
        keypoint_detail = _keypoint_names_config(
            keypoint_names,
            use_keypoints,
            exclude_keypoints,
            keypoint_parents,
            anterior,
            posterior,
            invert_axes,
        )

        # Form dataset-specific config
        dataset_detail = dict(
            type="raw_npy",
            subsample=subsample,
            subsample_to=subsample_to,
            **session_detail,
            **keypoint_detail,
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


class arrays(DatasetLoader):
    """Collection of numpy array files containing keypoint data."""

    type_name = "arrays"

    @staticmethod
    def _load_keypoint_array(path, config):
        """Load a keypoint array from a .npy file."""
        raise NotImplementedError(
            "Dataset loading not supported for dataset"
            "type 'arrays'. Please instantiate a Dataset instance directly."
        )

    @classmethod
    def load(cls, config, meta_only=False, allow_subsample=True):
        raise NotImplementedError(
            "Dataset loading not supported for dataset"
            "type 'arrays'. Please instantiate a Dataset instance directly."
        )

    @classmethod
    def setup_project_config(
        cls,
        project,
        session_names,
        keypoint_names,
        ref_session,
        use_keypoints=None,
        exclude_keypoints=None,
        keypoint_parents=None,
        bodies=None,
        anterior=None,
        posterior=None,
        invert_axes=None,
        subsample=None,
        alignment_type=None,
        feature_type=None,
    ):
        """Set up a project config for a arrays dataset.

        Parameters
        ----------
        project : kpsn.Project
        session_names : list of str
            Names of the sessions that will be provided in the dataset.
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

        # form session file data
        bodies = {n: None for n in session_names} if bodies is None else bodies
        session_detail = dict(
            ref_session=ref_session,
            sessions={s: dict(body=bodies[s]) for s in session_names},
        )
        keypoint_detail = _keypoint_names_config(
            keypoint_names,
            use_keypoints,
            exclude_keypoints,
            keypoint_parents,
            anterior,
            posterior,
            invert_axes,
        )

        # Form dataset-specific config
        dataset_detail = dict(
            type="arrays",
            subsample=subsample,
            **session_detail,
            **keypoint_detail,
        )

        # Write project config, including alignment and feature defaults
        return cls._write_project_config(
            project, dataset_detail, alignment_type, feature_type
        )

    @staticmethod
    def _validate_config(config):
        return True

    default_alignment = sagittal
    default_features = locked_pts


class h5(DatasetLoader):
    """Collection of HDF5 archives with data stored under a given key."""

    type_name = "h5"

    @staticmethod
    def _load_keypoint_array(path, config):
        """Load keypoint array from an HDF5 archive.

        Parameters
        ----------
        path : str
            Path to the HDF5 archive.
        config : dict
            `datasset` section of config.
        """

        import h5py

        with h5py.File(path, "r") as f:
            return f[config["h5_key"]]

    @classmethod
    def setup_project_config(
        cls,
        project,
        filepaths,
        h5_key,
        keypoint_names,
        ref_session,
        use_keypoints=None,
        exclude_keypoints=None,
        keypoint_parents=None,
        bodies=None,
        subsample=None,
        subsample_to=None,
        alignment_type=None,
        anterior=None,
        posterior=None,
        invert_axes=None,
        feature_type=None,
    ):
        # Form dataset-specific config
        dataset_detail = dict(
            type="h5",
            subsample=subsample,
            subsample_to=subsample_to,
            h5_key=h5_key,
            **_session_file_config(
                filepaths,
                keypoint_names,
                ref_session,
                use_keypoints,
                exclude_keypoints,
                keypoint_parents,
                anterior,
                posterior,
                invert_axes,
                bodies,
            ),
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


dataset_types = {
    raw_npy.type_name: raw_npy,
    h5.type_name: h5,
    arrays.type_name: arrays,
}


def load_dataset(config, allow_subsample=True):
    """Load a dataset based on a config dictionary.

    Parameters
    ----------
    config : dict
        `dataset` section of config.
    """
    dset_type = config["type"]
    if dset_type not in dataset_types:
        raise NotImplementedError(f"Unknown dataset type: {dset_type}")
    return dataset_types[dset_type].load(
        config, allow_subsample=allow_subsample
    )


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

    if alignment_type not in alignment_types:
        raise ValueError(f"Unknown alignment type: {alignment_type}")
    if feature_type not in feature_types:
        raise ValueError(f"Unknown feature type: {feature_type}")

    struct["alignment"]["type"] = alignment_type
    struct["alignment"].update(alignment_types[alignment_type].defaults)
    struct["features"]["type"] = feature_type
    struct["features"].update(feature_types[feature_type].defaults)

    return struct
