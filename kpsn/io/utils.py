import jax.numpy as jnp
from collections import defaultdict


def array_split_slice(l, n, i):
    """Start and end indices of i'th chunk returned by
    array_split(range(l), n)."""
    k = l // n
    m = l % n
    if i < m:
        return slice(i * (k + 1), (i + 1) * (k + 1))
    else:
        return slice(m * (k + 1) + (i - m) * k, m * (k + 1) + (i - m + 1) * k)


def compute_n_splits(
    dataset,
    split_all: bool,
    mode: str,
    count: int,
):
    """Calculate number of splits to be performed for each session."""
    if split_all:
        n_split = {s: count for s in dataset.sessions}
    else:
        n_split = {
            s: int(
                jnp.ceil(count / len(sessions))
                if i < len(sessions) % count
                else jnp.floor(count / len(sessions))
            )
            for b, sessions in dataset._bodies_inv.items()
            for i, s in enumerate(dataset.sessions)
        }
    return n_split


def split_body_inv(
    dataset,
    split_all: bool,
    mode: str,
    count: int,
):
    """Compute _body_inv for a split dataset."""
    n_split = compute_n_splits(dataset, split_all, mode, count)
    # split sessions according to n_split
    body_inv = defaultdict(list)
    for b, ss in dataset._bodies_inv.items():
        for s in ss:
            if n_split[s] == 1:
                body_inv[b].append(s)
                continue
            for i in range(n_split[s]):
                body_inv[b].append(f"{s}.{i}")
    return dict(body_inv)


def split_sessions(
    dataset, split_all: bool, mode: str, count: int, chunk_size: int
):
    """
    Parameters
    ----------
    split_all : bool
        Split only sessions which are the unique appearance of the associated
        body.
    mode : consecutive | interleaved
        Split sessions into maximal consecutive chunks or interleaved chunks of
        length `chunk_size`.
    count : int
        Number of desired sessions from each body if `split_all` is False, or
        number of split sections for each session if `split_all` is True.
    chunk_size : int
        Size of chunks to split sessions into if mode is 'interleaved', in frames.
    """

    n_split = compute_n_splits(dataset, split_all, mode, count)
    data = {}
    bodies = {}

    # split sessions according to n_split
    for s in dataset.sessions:
        if n_split[s] == 1:
            data[s] = dataset.get_session(s)
            bodies[s] = dataset.sess_bodies[s]
            continue

        if mode == "consecutive":
            parts = jnp.array_split(dataset.get_session(s), n_split[s])
            for i, arr in enumerate(parts):
                data[f"{s}.{i}"] = arr

        elif mode == "interleaved":
            # split into sections of size chunk_size * n_split
            chunks = jnp.array_split(
                dataset.get_session(s),
                jnp.ceil(len(dataset) / chunk_size / n_split[s]),
            )
            # select i'th chunk from each section and concatenate
            for i in range(n_split[s]):
                data[f"{s}.{i}"] = jnp.concatenate(
                    [
                        chunk[array_split_slice(len(chunk), n_split[s], i)]
                        for chunk in chunks
                    ]
                )

        for i in range(n_split[s]):
            bodies[f"{s}.{i}"] = dataset.sess_bodies[s]

    data, slices = stack_dict(data)
    ref = dataset.ref_session
    if n_split[ref] > 1:
        ref = f"{ref}.0"

    return dataset.with_sessions(data, slices, bodies, ref)


class UUIDGenerator(object):
    """Generate unique ids with prefix."""

    def __init__(self, prefix):
        self._i = 0
        self.uuids = set()
        self.prefix = prefix

    def add(self, uuid):
        self.uuids.add(uuid)

    def get(self):
        """Get a new unique id."""
        while True:
            uuid = f"{self.prefix}{self._i}"
            self._i += 1
            if uuid not in self.uuids:
                self.uuids.add(uuid)
                return uuid


def stack_dict(data):
    """Stack a dict of arrays into a single array.

    Parameters
    ----------
    data : dict[str, array]
        Keys are array names, values are arrays of shape (n_samples, ...).

    Returns
    -------
    data : array, shape (n_samples, ...)
        Concatenated data for all sessions.
    slices : dict[str, slice]
        Efficient index into `data` for each array by name.
    """
    data = list(data.items())
    # stack data
    stacked = jnp.concatenate([arr for name, arr in data], axis=0)
    # create slices
    slices = {}
    i = 0
    for name, arr in data:
        slices[name] = slice(i, i + arr.shape[0])
        i += arr.shape[0]
    return stacked, slices


def select_keypt_ixs(keypt_names, use_keypoints):
    """Select a subset of keypoints from a dataset.

    Parameters
    ----------
    keypt_names : list[str]
        Ordered list of keypoint names.
    use_keypoints : list[str]
        List of keypoints to select.

    Returns
    -------
    ixs : list[int]
        Indices of selected keypoints.
    """
    return [keypt_names.index(k) for k in use_keypoints]
