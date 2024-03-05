from .utils import stack_dict

from jaxtyping import Integer, Array
from typing import Tuple
from bidict import bidict
import jax.numpy as jnp
import jax.random as jr
import jax._src.random as prng
import jax.tree_util as pt
import logging
import numpy as np

PRNGKey = prng.KeyArray


class Dataset(object):
    """Base class for a collection of data and metadata.

    Config parameters:
    sessions:
        session_name:
            body: <str,int> # id of body to use for this session

    Attributes
    ----------
    data : array, shape (n_samples, ...)
        Concatenated data for all sessions.
    ref_session : str
        Name of session to treat as a reference, whose poses should be treated
        as canoncial.
    slices : dict[str, slice]
        Efficient index into `data` for each session by name.
    bodies : dict[str, str]
        Mapping from session names to body names.
    body_ids, session_ids : array of int (n_samples,)
        Body or session id associated with each sample.
    _body_names, _session_names : bidict[int, str]
        Mapping between session / body names and internal integer ids.
    """

    _is_dataset = True

    def __init__(
        self,
        data,
        slices,
        bodies,
        ref_session,
        compute_ids=True,
        _body_names=None,
        _session_names=None,
    ):
        self.data = data
        if any([s.step != 1 and s.step is not None for s in slices.values()]):
            raise ValueError("Dataset does not support slices with step != 1.")
        self.slices = slices
        self.sess_bodies = bodies
        self.ref_session = ref_session
        self._body_names = _body_names
        self._session_names = _session_names
        if compute_ids:
            self._init_names_and_ids()

    def with_data(self, new_data):
        """Return a new dataset with identical metadata."""
        raise NotImplementedError

    def with_sessions(
        self, new_data, new_slices, new_bodies, ref_session, compute_ids=True
    ):
        """Return new dataset with new sessions and core metadata, preserving
        type-specific metadata."""
        raise NotImplementedError

    def __len__(self):
        """Return the total number of samples."""
        return self.data.shape[0]

    def session_length(self, session):
        """Return the number of samples in a session."""
        return self.get_slice(session).stop - self.get_slice(session).start

    def _init_names_and_ids(self):
        if self._body_names is None:
            self._body_names = bidict(
                zip(
                    range(len(self.sess_bodies)),
                    sorted(list(set(self.sess_bodies.values()))),
                )
            )
        else:
            self._body_names = bidict(self._body_names)
        if self._session_names is None:
            self._session_names = bidict(
                zip(range(len(self.slices)), sorted(self.slices.keys()))
            )
        else:
            self._session_names = bidict(self._session_names)

        _bodies_inv = {b: set() for b in set(self._body_names.values())}
        for sess, body in self.sess_bodies.items():
            _bodies_inv[body].add(sess)
        self._bodies_inv = {b: tuple(s) for b, s in _bodies_inv.items()}

        # Assign ids at each slice according to body_names and session_names
        body_ids = np.zeros([self.data.shape[0]], dtype=int)
        session_ids = np.zeros([self.data.shape[0]], dtype=int)
        for sess, slc in self.slices.items():
            body_ids[slc] = self._body_names.inverse[self.sess_bodies[sess]]
            session_ids[slc] = self._session_names.inverse[sess]
        self.body_ids = jnp.array(body_ids)
        self.session_ids = jnp.array(session_ids)

    @staticmethod
    def _copy_names_and_ids(source, dest):
        dest.sess_bodies = source.sess_bodies
        dest._bodies_inv = source._bodies_inv
        dest.body_ids = source.body_ids
        dest.session_ids = source.session_ids
        dest._body_names = source._body_names
        dest._session_names = source._session_names

    def ordered_session_names(self):
        return tuple(self._session_names[i] for i in range(self.n_sessions))

    def get_session(self, session):
        """Return a data for a single session."""
        return self.data[self.slices[self._session_name(session)]]

    def get_all_with_body(self, body):
        """Return data with only the given body.

        Note: order not guaranteed.
        """
        if not (isinstance(body, str)):
            body = self._body_names[body]
        return jnp.concatenate(
            [self.get_session(s) for s in self._bodies_inv[body]], axis=0
        )

    def get_slice(self, session):
        """Return a slice for a single session."""
        return self.slices[self._session_name(session)]

    def _session_name(self, session):
        """Return name corresponding to either session name or id."""
        return (
            session
            if isinstance(session, str)
            else self._session_names[session]
        )

    def session_subset(
        self, session_names, ref_session=None, bad_ref_ok=False
    ) -> "Dataset":
        """Return a new dataset with only the given sessions."""
        new_data, new_slices = stack_dict(
            {s: self.get_session(s) for s in session_names}
        )

        if ref_session is None:
            ref_session = self.ref_session
        if ref_session not in session_names:
            if not bad_ref_ok:
                logging.warn(
                    f"Reference session {ref_session} not in subset. "
                    f"Assigning new reference {session_names[0]}. Use `bad_ref_ok` "
                    "to suppress this warning."
                )
            ref_session = session_names[0]

        new_set = self.with_sessions(
            new_data,
            new_slices,
            self.sess_bodies,
            ref_session,
            compute_ids=True,
            _body_names=self._body_names,
            _session_names=self._session_names,
        )
        new_set._bodies_inv = {
            b: tuple(s for s in ss if s in session_names)
            for b, ss in self._bodies_inv.items()
        }
        return new_set

    def batch_generator(self, size, replace=False):
        """Return a subsample of frames from each session.

        Parameters
        ----------
        rkey : jax.random.PRNGKey
            Random key to use for sampling.
        size : int
            Number of frames to sample from each session.
        replace : bool, default = False
            Whether to sample with replacement. If any session is too small to
            be sampled without replacement, warning will be raised and sampling
            will be performed with replacement.

        Returns
        -------
        generate_batch : callable
            Callable that returns a new random key and a batch of data.

        Returns of `generate_batch`
        ---------------------------
        rkey : jax.random.PRNGKey
            New random key not used for sampling.
        batch : array, shape (n_sessions * size, ...)
            Concatenated batch of data from each session.
        batch_ixs : array, shape (n_sessions * size,)
            Indices of frames in the original dataset.
        """

        # determine which sessions may be sampled with replacement
        sess_replace = []
        for i, (sess, slc) in enumerate(self.slices.items()):
            if not replace:
                if self.session_length(sess) < size:
                    logging.warn(
                        f"Batch size too large for session {sess}. "
                        "Sampling with replacement."
                    )
                    sess_replace.append(True)
                else:
                    sess_replace.append(False)
            else:
                sess_replace.append(True)

        def generate_batch(
            rkey,
        ) -> Tuple[PRNGKey, Dataset, Integer[Array, "n_sessions * size"]]:
            rkeys = jr.split(rkey, self.n_sessions + 1)
            # select indices of frames from each session
            all_ixs = np.arange(len(self))
            batch = jnp.concatenate(
                [
                    jr.choice(rkey, all_ixs[self.get_slice(i)], size, replace)
                    for i, rkey, replace in zip(
                        range(self.n_sessions), rkeys, sess_replace
                    )
                ]
            )
            # create new dataset with selected frames and identical metadata
            new_set: Dataset = self.with_data(self.data[batch])
            # update metadata to reflect subset
            new_set.slices = {
                self._session_names[i]: slice(i * size, (i + 1) * size)
                for i in range(self.n_sessions)
            }
            new_set.body_ids = self.body_ids[batch]
            new_set.session_ids = self.session_ids[batch]

            return rkeys[-1], new_set, batch

        return generate_batch

    def with_sess_bodies(self, new_bodies, new_names=False):
        """Return a new dataset with the same data and slices, but new session
        bodies."""
        sess_bodies = {
            s: new_bodies.get(s, self.sess_bodies[s])
            for s in self._session_names.values()
        }
        return self.with_sessions(
            self.data,
            self.slices,
            sess_bodies,
            self.ref_session,
            compute_ids=True,
            _body_names=None if new_names else self._body_names,
            _session_names=self._session_names,
        )

    # important semantic point: n_sessions / sessions, (resp. bodies) describes
    # the sessions actually present in the data array, not those known by the
    # dataset object / assigned name-id pairs.
    n_sessions = property(lambda self: len(self.slices))
    n_bodies = property(lambda self: len(self._bodies_inv))
    sessions = property(lambda self: self.slices.keys())
    bodies = property(
        lambda self: set(self.sess_bodies[s] for s in self.sessions)
    )



class KeypointDataset(Dataset):
    """Collection of keypoint data."""

    ndim = 2
    # _is_dataset = True

    def __init__(
        self,
        data,
        slices,
        bodies,
        ref_session,
        keypoint_names,
        compute_ids=True,
        _body_names=None,
        _session_names=None,
    ):
        super().__init__(
            data,
            slices,
            bodies,
            ref_session,
            compute_ids,
            _body_names,
            _session_names,
        )
        self.keypoint_names = keypoint_names
        self.keypoint_ids = bidict({n: i for i, n in enumerate(keypoint_names)})

    @staticmethod
    def from_arrays(data, bodies, ref_session, keypoint_names):
        """Create a KeypointDataset from a dict of arrays."""
        data, slices = stack_dict(data)
        return KeypointDataset(
            data, slices, bodies, ref_session, keypoint_names
        )

    def as_features(self):
        """Return a FeatureDataset with the same data."""
        feats = FeatureDataset(
            self.data.reshape(len(self), -1),
            self.slices,
            self.sess_bodies,
            self.ref_session,
            compute_ids=False,
        )
        Dataset._copy_names_and_ids(self, feats)
        return feats

    def with_data(self, new_data):
        """Return a new dataset with identical metadata."""
        new_set = KeypointDataset(
            new_data,
            self.slices,
            self.sess_bodies,
            self.ref_session,
            self.keypoint_names,
            compute_ids=False,
        )
        Dataset._copy_names_and_ids(self, new_set)
        return new_set

    def with_sessions(
        self,
        new_data,
        new_slices,
        new_bodies,
        ref_session,
        compute_ids=True,
        _body_names=None,
        _session_names=None,
    ):
        """Return new dataset with new sessions and core metadata, preserving
        type-specific metadata."""
        return KeypointDataset(
            new_data,
            new_slices,
            new_bodies,
            ref_session,
            self.keypoint_names,
            compute_ids,
            _body_names,
            _session_names,
        )

    n_points = property(lambda self: self.data.shape[-2])
    n_dims = property(lambda self: self.data.shape[-1])


class FeatureDataset(Dataset):
    """Collection of flattened feature data."""

    ndim = 1

    n_feats: int = property(lambda self: self.data.shape[-1])

    def with_data(self, new_data):
        """Return a new dataset with identical metadata."""
        new_set = FeatureDataset(
            new_data,
            self.slices,
            self.sess_bodies,
            self.ref_session,
            compute_ids=False,
        )
        Dataset._copy_names_and_ids(self, new_set)
        return new_set

    def with_sessions(
        self,
        new_data,
        new_slices,
        new_bodies,
        ref_session,
        compute_ids=True,
        _body_names=None,
        _session_names=None,
    ):
        """Return new dataset with new sessions and core metadata, preserving
        type-specific metadata."""
        return FeatureDataset(
            new_data,
            new_slices,
            new_bodies,
            ref_session,
            compute_ids,
            _body_names,
            _session_names,
        )

    def as_keypoints(self, keypoint_names):
        """Return a KeypointDataset with the reshaped data.

        Parameters
        ----------
        keypoint_names
            Keypoint names of the new dataset.
        """
        kpts = KeypointDataset(
            self.data.reshape(len(self), len(keypoint_names), -1),
            self.slices,
            self.sess_bodies,
            self.ref_session,
            keypoint_names,
            compute_ids=False,
        )
        Dataset._copy_names_and_ids(self, kpts)
        return kpts

    @staticmethod
    def from_pytree(
        dataset: "PytreeDataset", body_names: dict, session_names: dict
    ):
        """Convert a PytreeDataset to a FeatureDataset.

        Parameters
        ----------
        dataset : PytreeDataset
        body_names, session_names : dict
            Mapping from integer body/session ids to string body/session names.
        """

        sess_bodies = {
            session_names[i]: body_names[int(dataset.sess_bodies[i])]
            for i in range(dataset.n_sessions)
            if dataset.slices[i] is not None
        }
        new_dataset = FeatureDataset(
            dataset.data,
            {
                session_names[i]: dataset.slices[i]
                for i in range(dataset.n_sessions)
                if dataset.slices[i] is not None
            },
            sess_bodies,
            session_names[dataset.ref_session],
            compute_ids=False,
            _body_names=body_names,
            _session_names=session_names,
        )
        new_dataset._bodies_inv = {
            body_names[b]: tuple(
                session_names[int(s)] for s in dataset._bodies_inv[b]
            )
            for b in range(dataset.n_bodies)
        }
        new_dataset.session_ids = dataset.session_ids
        new_dataset.body_ids = dataset.body_ids
        return new_dataset


@pt.register_pytree_node_class
class PytreeDataset:
    """Collection of data and metadata passable to jitted functions.

    Config parameters:
    sessions:
        session_name:
            body: <str,int> # id of body to use for this session

    Attributes
    ----------
    data : array, shape (n_samples, ...)
        Concatenated data for all sessions.
    ref_session : int
        Name of session to treat as a reference, whose poses should be treated
        as canoncial.
    slices : dict[int, slice]
        Efficient index into `data` for each session by name.
    bodies : dict[int, int]
        Mapping from session names to body names.
    body_ids, session_ids : array of int (n_samples,)
        Body or session id associated with each sample.
    """

    ndim = 1

    n_feats: int = property(lambda self: self.data.shape[-1])

    def __init__(
        self,
        data,
        slices,
        bodies,
        ref_session,
        body_ids,
        session_ids,
        bodies_inv,
    ):
        """"""
        self.data = data
        self.slices = slices
        self.sess_bodies = bodies
        self.ref_session = ref_session
        self.body_ids = body_ids
        self.session_ids = session_ids
        self._bodies_inv = bodies_inv
        if isinstance(self.slices[0], tuple):
            self.slices = tuple(slice(*s) for s in self.slices)

    def __len__(self):
        """Return the total number of samples."""
        return self.data.shape[0]

    def session_length(self, session):
        """Return the number of samples in a session."""
        slc = self.get_slice(session)
        return slc.stop - slc.start

    def get_session(self, session):
        """Return a data for a single session."""
        return self.data[self.get_slice(session)]

    def get_all_with_body(self, body):
        """Return data with only the given body.

        Note: order not guaranteed.
        """
        return jnp.concatenate(
            [self.data[self.slices[s]] for s in self._bodies_inv[body]],
            axis=0,
        )

    def get_slice(self, session):
        """Return a slice for a single session."""
        if self.slices[session] is None:
            raise ValueError(f"Session {session} not in PytreeDataset.")
        return self.slices[session]

    def batch_generator(self, size, replace=False, session_names=None):
        """Return a subsample of frames from each session.

        Parameters
        ----------
        rkey : jax.random.PRNGKey
            Random key to use for sampling.
        size : int
            Number of frames to sample from each session.
        replace : bool, default = False
            Whether to sample with replacement. If any session is too small to
            be sampled without replacement, warning will be raised and sampling
            will be performed with replacement.

        Returns
        -------
        generate_batch : callable
            Callable that returns a new random key and a batch of data.

        Returns of `generate_batch`
        ---------------------------
        rkey : jax.random.PRNGKey
            New random key not used for sampling.
        batch : array, shape (n_sessions * size, ...)
            Concatenated batch of data from each session.
        batch_ixs : array, shape (n_sessions * size,)
            Indices of frames in the original dataset.
        """

        # determine which sessions may be sampled with replacement
        sess_replace = []
        for i, slc in enumerate(self.slices):
            if session_names is not None:
                sess = session_names[i]
            else:
                sess = i
            if not replace:
                if self.session_length(sess) < size:
                    logging.warn(
                        f"Batch size too large for session {sess}. "
                        "Sampling with replacement."
                    )
                    sess_replace.append(True)
                else:
                    sess_replace.append(False)
            else:
                sess_replace.append(True)

        def generate_batch(
            rkey,
        ) -> Tuple[PRNGKey, PytreeDataset, Integer[Array, "n_sessions * size"]]:
            rkeys = jr.split(rkey, self.n_sessions + 1)
            # select indices of frames from each session
            all_ixs = np.arange(len(self))
            batch = jnp.concatenate(
                [
                    jr.choice(rkey, all_ixs[self.get_slice(i)], size, replace)
                    for i, rkey, replace in zip(
                        range(self.n_sessions), rkeys, sess_replace
                    )
                ]
            )
            # create new dataset with selected frames and identical metadata
            new_set = self.with_data(self.data[batch])
            # update metadata to reflect subset
            new_set.slices = {
                self._session_names[i]: slice(i * size, (i + 1) * size)
                for i in range(self.n_sessions)
            }
            new_set.body_ids = self.body_ids[batch]
            new_set.session_ids = self.session_ids[batch]

            return rkeys[-1], new_set, batch

        return generate_batch

    def with_data(self, new_data):
        """Return a new dataset with identical metadata."""
        return PytreeDataset(
            new_data,
            self.slices,
            self.sess_bodies,
            self.ref_session,
            self.body_ids,
            self.session_ids,
            self._bodies_inv,
        )

    @staticmethod
    def from_pythonic(dataset: Dataset):
        """Convert a Dataset to a PytreeDataset."""
        dset = dataset
        sessions = [
            dset._session_names[i] for i in range(len(dataset._session_names))
        ]
        bodies = [dset._body_names[i] for i in range(len(dset._body_names))]
        body_names_inv = dataset._body_names.inverse
        bodies_inv = tuple(
            tuple(
                dset._session_names.inverse[s] for s in dataset._bodies_inv[b]
            )
            for b in bodies
        )
        return PytreeDataset(
            dataset.data,
            tuple(dataset.slices.get(s) for s in sessions),
            tuple(
                (
                    body_names_inv[dataset.sess_bodies[s]]
                    if s in dataset.slices
                    else None
                )
                for s in sessions
            ),
            dset._session_names.inverse[dset.ref_session],
            dataset.body_ids,
            dataset.session_ids,
            bodies_inv,
        )

    def tree_flatten(self):
        return pt.tree_flatten(
            (
                self.data,
                tuple(
                    None if slc is None else (slc.start, slc.stop)
                    for slc in self.slices
                ),
                self.sess_bodies,
                self.ref_session,
                self.body_ids,
                self.session_ids,
                self._bodies_inv,
            )
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*pt.tree_unflatten(aux_data, children))

    # important semantic difference for pytreee datasets: n_sessions is the
    # number of sessions known to the dataset, not the number of sessions
    # present in the data array (could be different after subsetting)
    n_sessions = property(lambda self: len(self.slices))
    n_bodies = property(lambda self: len(self._bodies_inv))
