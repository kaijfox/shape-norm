from bidict import bidict
import numpy as np
import jax.numpy as jnp
from typing import Optional, Union, Tuple
from jaxtyping import Integer, Array
import jax.random as jr
from .utils import stack_dict
from frozendict import frozendict
import logging

import jax._src.random as prng

PRNGKey = prng.KeyArray


class SessionMetadata:
    """Mapping between session/body names/ids.

    Immutable. Contains bijections:
    - session id <-> session name
    - body id <-> body name
    And maps:
    - session_ids <-> body_ids
    """

    def __init__(
        self,
        serialized: Optional[Union["SessionMetadata", dict]] = None,
        session_ids: Optional[Union[dict, bidict]] = None,
        body_ids: Optional[Union[dict, bidict]] = None,
        session_bodies: Optional[dict[Union[str, int], Union[str, int]]] = None,
    ):
        if serialized is None:
            serialized = {
                "session_ids": session_ids,
                "body_ids": body_ids,
                "session_bodies": session_bodies,
            }
        if isinstance(serialized, (dict, frozendict)):
            self._session_ids = bidict(serialized["session_ids"])
            self._body_ids = bidict(serialized["body_ids"])
            self._session_bodies = serialized["session_bodies"]
            # session bodies may be passed using string names or integer ids
            # but will be stored using integer ids
            self._session_bodies = {
                self._session_ids[s] if isinstance(s, str) else s: (
                    self._body_ids[b] if isinstance(b, str) else b
                )
                for s, b in self._session_bodies.items()
            }
            self._bodies_inv = None
        # initialization from another SessionMetadata object: keep cached values
        else:
            self._session_ids = serialized._session_ids
            self._body_ids = serialized._body_ids
            self._session_bodies = serialized._session_bodies
            self._bodies_inv = serialized._bodies_inv

    def update(self, session_ids=None, body_ids=None, session_bodies=None):
        return SessionMetadata(
            session_ids=(
                session_ids if session_ids is not None else self._session_ids
            ),
            body_ids=(body_ids if body_ids is not None else self._body_ids),
            session_bodies=(
                session_bodies
                if session_bodies is not None
                else self._session_bodies
            ),
        )

    def session_name(self, id: int) -> str:
        return self._session_ids.inverse[id]

    def session_id(self, name: str) -> int:
        return self._session_ids[name]

    def body_name(self, id: int) -> str:
        return self._body_ids.inverse[id]

    def body_id(self, name: str) -> int:
        return self._body_ids[name]

    def session_body_name(self, session: Union[str, int]) -> dict:
        if isinstance(session, str):
            session = self.session_id(session)
        return self.body_name(self._session_bodies[session])

    def session_body_id(self, session: Union[str, int]) -> dict:
        if isinstance(session, str):
            session = self.session_id(session)
        return self._session_bodies[session]

    def bodies_inv(self) -> dict[str, tuple[str]]:
        if self._bodies_inv is None:
            _bodies_inv = {b: set() for b in set(self._body_ids.keys())}
            for sess_id, body_id in self._session_bodies.items():
                _bodies_inv[self.body_name(body_id)].add(
                    self.session_name(sess_id)
                )
            self._bodies_inv = {b: tuple(s) for b, s in _bodies_inv.items()}
        return self._bodies_inv

    def serialize(self) -> dict:
        """Return a dictionary that may be passed to JAX jit functions."""
        return frozendict(
            {
                "session_ids": frozendict(self._session_ids),
                "body_ids": frozendict(self._body_ids),
                "session_bodies": frozendict(self._session_bodies),
            }
        )


class StackedArrayMetadata:
    """Mapping from session ids to slices into a stacked array.

    Immutable. Offers a (cached) array mapping indices in the stacked axis to
    session ids."""

    def __init__(
        self,
        serialized: Optional[Union["StackedArrayMetadata", dict]] = None,
        slices: Optional[dict[int, tuple[int]]] = None,
        length: Optional[int] = None,
    ):
        if serialized is None:
            serialized = {"slices": slices, "length": length}
        if isinstance(serialized, (dict, frozendict)):
            self._slices = dict(serialized["slices"])
            self._length = serialized["length"]
            self._stack_session_ids = None
        else:
            self._slices = serialized._slices
            self._length = serialized._length
            self._stack_session_ids = None

    @property
    def slices(self) -> dict:
        return self._slices

    @property
    def length(self) -> int:
        return self._length

    def update(self, slices=None, length=None):
        return StackedArrayMetadata(
            slices=(slices if slices is not None else self._slices),
            length=(length if length is not None else self._length),
        )

    @property
    def stack_session_ids(self) -> np.ndarray:
        if self._stack_session_ids is None:
            self._stack_session_ids = np.zeros([self.length], dtype=int)
            for session_id, slice_ in self.slices.items():
                self._stack_session_ids[slice_[0] : slice_[1]] = session_id
        return self._stack_session_ids

    def serialize(self) -> dict:
        """Return a dictionary that may be passed to JAX jit functions."""
        return frozendict(
            {"slices": frozendict(self._slices), "length": self._length}
        )


def _ensure_length_match(
    stack_meta: Union[StackedArrayMetadata, dict],
    data: Union[np.ndarray, jnp.ndarray],
) -> StackedArrayMetadata:
    if isinstance(stack_meta, (dict, frozendict)):
        if "length" not in stack_meta or stack_meta["length"] != data.shape[0]:
            return StackedArrayMetadata({"length": data.shape[0], **stack_meta})
        return StackedArrayMetadata(stack_meta)
    if stack_meta.length != data.shape[0]:
        return stack_meta.update(length=data.shape[0])
    return StackedArrayMetadata(stack_meta)


class Dataset:
    """A stacked array with session and body id metadata."""

    def __init__(
        self,
        serialized: Optional[Union["Dataset", dict]] = None,
        data: Optional[Union[np.array, jnp.array]] = None,
        stack_meta: Optional[Union[StackedArrayMetadata, dict]] = None,
        session_meta: Optional[Union[SessionMetadata, dict]] = None,
        ref_session: Optional[Union[str, int]] = None,
        aux: Optional[dict] = None,
    ):
        # initialization from `data`, `stack_meta`, and `session_meta`
        if serialized is None:
            serialized = {
                "data": data,
                "stack_meta": stack_meta,
                "session_meta": session_meta,
                "ref_session": ref_session,
                "aux": aux,
            }

        # initialization from `serialized`
        if isinstance(serialized, (dict, frozendict)):
            if "aux" not in serialized:
                serialized["aux"] = aux if aux is not None else {}
            self._data = serialized["data"]
            self._stack_meta = _ensure_length_match(
                serialized["stack_meta"], self._data
            )
            self._session_meta = SessionMetadata(serialized["session_meta"])
            self._ref_session = serialized["ref_session"]
            if isinstance(self._ref_session, str):
                self._ref_session = self._session_meta.session_id(
                    self._ref_session
                )
            self._aux = serialized["aux"]
            self._stack_body_ids = None
            self._stack_session_ids = None

        # serialized is a Dataset object: copy attributes and cached values
        else:
            self._data = serialized.data
            self._stack_meta = serialized.stack_meta
            self._session_meta = serialized.session_meta
            self._ref_session = serialized._ref_session
            self._aux = serialized._aux
            self._stack_body_ids = serialized._stack_body_ids
            self._stack_session_ids = serialized._stack_session_ids

    # ------------------------------------------- constructors / modifiers ----

    @staticmethod
    def from_arrays(
        data: dict[Union[np.ndarray, jnp.ndarray]],
        bodies: dict[str, str],
        ref_session: str,
        aux: Optional[dict] = None,
    ) -> "Dataset":
        """Create a Dataset object from a dictionary of arrays and a dictionary
        of body names."""
        stacked, slices = stack_dict(data)
        session_ids = bidict({s: i for i, s in enumerate(sorted(data.keys()))})
        slices = {
            session_ids[s]: (slice_.start, slice_.stop)
            for s, slice_ in slices.items()
        }
        session_bodies = {s: bodies[s] for s in data.keys()}
        body_ids = {
            b: i for i, b in enumerate(sorted(list(set(bodies.values()))))
        }
        session_meta = SessionMetadata(
            session_ids=session_ids,
            body_ids=body_ids,
            session_bodies=session_bodies,
        )
        stack_meta = StackedArrayMetadata(
            slices=slices, length=stacked.shape[0]
        )
        return Dataset(
            data=stacked,
            stack_meta=stack_meta,
            session_meta=session_meta,
            ref_session=ref_session,
            aux=aux,
        )

    def serialize(self) -> dict:
        """Return a dictionary that may be passed to JAX jit functions
        containing all the information needed to reconstruct the object except
        auxiliary metadata."""
        return jnp.array(self.data), frozendict(
            {
                "session_meta": self._session_meta.serialize(),
                "stack_meta": self._stack_meta.serialize(),
                "ref_session": self._ref_session,
            }
        )

    def update(
        self,
        data=None,
        stack_meta=None,
        session_meta=None,
        ref_session=None,
        aux=None,
    ) -> "Dataset":
        ret = Dataset(
            data=data if data is not None else self._data,
            session_meta=(
                session_meta if session_meta is not None else self._session_meta
            ),
            stack_meta=(
                stack_meta if stack_meta is not None else self._stack_meta
            ),
            ref_session=(
                ref_session if ref_session is not None else self._ref_session
            ),
            aux=aux if aux is not None else self._aux,
        )
        # preserve cache if key metadata is unchanged
        if stack_meta is None:
            ret._stack_session_ids = self._stack_session_ids
            if session_meta is None:
                ret._stack_body_ids = self._stack_body_ids
        return ret

    # ------------------------------------------------------ simple access ----

    @property
    def data(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._data

    @property
    def session_meta(self) -> SessionMetadata:
        return self._session_meta

    @property
    def stack_meta(self) -> StackedArrayMetadata:
        return self._stack_meta

    @property
    def ref_session(self) -> int:
        return self._ref_session

    @property
    def aux(self) -> dict:
        return self._aux

    def set_cache(self, stack_session_ids=None, stack_body_ids=None):
        if stack_session_ids is not None:
            self._stack_session_ids = stack_session_ids
        if stack_body_ids is not None:
            self._stack_body_ids = stack_body_ids

    # ---- passthroughs to session_meta ----
    @property
    def stack_session_ids(self):
        return self.stack_meta.stack_session_ids

    def session_name(self, id: int) -> str:
        return self._session_meta.session_name(id)

    def session_id(self, name: str) -> int:
        return self._session_meta.session_id(name)

    def body_name(self, id: int) -> str:
        return self._session_meta.body_name(id)

    def body_id(self, name: str) -> int:
        return self._session_meta.body_id(name)

    def session_body_name(self, session: Union[str, int]) -> dict:
        return self._session_meta.session_body_name(session)

    def session_body_id(self, session: Union[str, int]) -> dict:
        return self._session_meta.session_body_id(session)

    # ---- passthroughs to stack_meta ----
    @property
    def slices(self) -> dict:
        return self.stack_meta.slices

    @property
    def length(self) -> int:
        return self.stack_meta.length

    def __len__(self) -> int:
        return self.stack_meta.length

    # ------------------------------------------------------ derived access ----

    @property
    def sessions(self):
        return tuple(
            self._session_meta.session_name(i)
            for i in self.stack_meta.slices.keys()
        )

    def ordered_session_names(self):
        return tuple(
            self._session_meta._session_ids.inverse[i]
            for i in range(len(self._session_meta._session_ids))
        )

    @property
    def bodies(self):
        return set(
            self._session_meta.session_body_name(i)
            for i in self.stack_meta.slices.keys()
        )

    @property
    def n_sessions(self):
        return len(self.stack_meta.slices)

    @property
    def n_bodies(self):
        return len(self.bodies)

    @property
    def ndim(self):
        return self.data.ndim - 1

    def get_slice(self, session: Union[str, int]) -> slice:
        if isinstance(session, str):
            session = self.session_meta.session_id(session)
        return self.stack_meta.slices[session]

    def session_length(self, session: Union[str, int]) -> int:
        slc = self.get_slice(session)
        return slc[1] - slc[0]

    @property
    def bodies_inv(self) -> dict[list, set]:
        """Partial inverse of session_bodies mapping."""
        full_inv = self._session_meta.bodies_inv()
        return {b: full_inv[b] for b in self.bodies}

    @property
    def stack_body_ids(self) -> np.ndarray:
        """Array mapping indices in a stacked axis to body ids."""
        if self._stack_body_ids is None:
            self._stack_body_ids = np.zeros([self.stack_meta.length], dtype=int)
            for session_id, slice_ in self.stack_meta.slices.items():
                self._stack_body_ids[slice_[0] : slice_[1]] = (
                    self.session_meta.session_body_id(session_id)
                )
            if isinstance(self._stack_body_ids, jnp.ndarray):
                self._stack_body_ids = jnp.array(self._stack_body_ids)
        return self._stack_body_ids

    @property
    def stack_session_ids(self):
        if self._stack_session_ids is None:
            if isinstance(self.stack_meta.stack_session_ids, jnp.ndarray):
                self._stack_session_ids = jnp.array(
                    self.stack_meta.stack_session_ids
                )
            else:
                self._stack_session_ids = self.stack_meta.stack_session_ids
        return self._stack_session_ids

    def get_session(
        self, session: Union[str, int]
    ) -> Union[np.ndarray, jnp.ndarray]:
        if isinstance(session, str):
            session = self.session_meta.session_id(session)
        slice_ = self.stack_meta.slices[session]
        return self.data[slice_[0] : slice_[1]]

    def get_all_with_body(
        self, body: Union[str, int]
    ) -> Union[np.ndarray, jnp.ndarray]:
        if isinstance(body, int):
            body = self.body_name(body)
        slices = [
            self.stack_meta.slices[self.session_meta.session_id(s)]
            for s in self.bodies_inv[body]
        ]
        return np.concatenate(
            [self.data[slc[0] : slc[1]] for slc in slices], axis=0
        )

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
        sess_replace = {}
        for sess_name, sess_id in self._session_meta._session_ids.items():
            sess_replace[sess_id] = True
            if not replace:
                if self.session_length() < size:
                    logging.warn(
                        f"Batch size too large for session {sess_name}. "
                        "Sampling with replacement."
                    )
                else:
                    sess_replace[sess_name] = False

        n_sessions = self.n_sessions
        all_ixs = np.arange(len(self))
        ord_sess_names = self.ordered_session_names()
        new_slices = {
            self.session_id(sess): (i * size, (i + 1) * size)
            for i, sess in enumerate(ord_sess_names)
        }

        def generate_batch(
            rkey,
        ) -> Tuple[PRNGKey, Dataset, Integer[Array, "n_sessions * size"]]:
            rkeys = jr.split(rkey, n_sessions + 1)
            # select indices of frames from each session
            batch = jnp.concatenate(
                [
                    jr.choice(
                        rkey,
                        all_ixs[self.get_slice(sess)],
                        size,
                        sess_replace[sess],
                    )
                    for rkey, sess in zip(ord_sess_names, rkeys)
                ]
            )
            # create new dataset with selected frames and identical metadata
            new_set = self.update(
                data=self.data[batch],
                stack_meta=StackedArrayMetadata(
                    slices=new_slices,
                    length=n_sessions * size,
                ),
            )

            return rkeys[-1], new_set, batch

        return generate_batch
