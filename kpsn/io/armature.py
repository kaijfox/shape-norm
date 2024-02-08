from typing import NamedTuple
import jax.numpy as jnp
from bidict import bidict


def _toposort(edges, root):
    """
    Parameters
    ----------
    edges : jnp.ndarray
        (n_edges, 2) array of edges.
    root : int
        Index of root node.

    Returns
    -------
    jnp.ndarray
        (n_edges, 2) array of edges, sorted by topological order.
    """
    new_bones = []

    def traverse_from(node):
        visited.add(int(node))
        for child in connected_to(node):
            if int(child) in visited:
                continue
            new_bones.append((child, node))
            traverse_from(child)

    visited = set()

    def connected_to(i):
        return jnp.concatenate(
            [
                edges[edges[:, 0] == i, 1],
                edges[edges[:, 1] == i, 0],
            ]
        )

    traverse_from(root)
    return jnp.array(new_bones)


class Armature(NamedTuple):
    keypoint_names: bidict  # <str>
    bones: jnp.ndarray  # <int>
    root: str

    @staticmethod
    def from_config(config):
        """
        Parameters
        ----------
        config : dict
            `dataset` section of config.
        """
        names = bidict(
            zip(range(len(config["use_keypoints"])), config["use_keypoints"])
        )
        parents = config["viz"]["armature"]
        root = [k for k, v in parents.items() if v is None][0]
        bones = jnp.array(
            [
                (names.inverse[kp], names.inverse[parent])
                for kp, parent in parents.items()
                if parent is not None
            ]
        )
        return Armature(names, _toposort(bones, names.inverse[root]), root)

    @property
    def keypt_by_name(self):
        return self.keypoint_names.inverse

    @property
    def n_kpts(self):
        return len(self.keypoint_names)

    def bone_name(self, i_bone, joiner="-"):
        child_name = self.keypoint_names[self.bones[i_bone, 0]]
        parent_name = self.keypoint_names[self.bones[i_bone, 1]]
        return f"{child_name}{joiner}{parent_name}"
