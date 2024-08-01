from typing import NamedTuple
from jaxtyping import Float, Array
import jax.numpy as jnp
from bidict import bidict
import numpy as np


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
    anterior: str
    posterior: str

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
        return Armature(
            names,
            _toposort(bones, names.inverse[root]),
            root,
            config["anterior"],
            config["posterior"],
        )

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


def bone_lengths(
    keypts: Float[Array, "*#K n_keypt"], armature: Armature
) -> Float[Array, "*#K n_bones"]:
    return jnp.stack(
        [
            jnp.lingalg.norm(
                keypts[..., armature.bones[i, 1], :]
                - keypts[..., armature.bones[i, 0], :],
                axis=-1,
            )
            for i in range(len(armature.bones))
        ],
        axis=-1,
    )


def construct_bones_transform(skeleton, root_keypt):
    n_kpts = len(skeleton) + 1
    u_to_x = np.zeros([n_kpts, n_kpts])
    x_to_u = np.zeros([n_kpts, n_kpts])
    u_to_x[root_keypt, root_keypt] = 1
    x_to_u[root_keypt, root_keypt] = 1
    # skeleton is topo sorted (thank youuuu)
    for child, parent in skeleton:
        x_to_u[child, parent] = -1
        x_to_u[child, child] = 1
        u_to_x[child] = u_to_x[parent]
        u_to_x[child, child] = 1
    bones_mask = np.ones(n_kpts, dtype=bool)
    bones_mask[root_keypt] = 0
    return {
        "u_to_x": jnp.array(u_to_x),
        "x_to_u": jnp.array(x_to_u),
        "root": root_keypt,
        "bone_mask": jnp.array(bones_mask),
    }


def bone_transform(keypts, transform_data):
    bones_and_root = transform_data["x_to_u"] @ keypts
    return (
        bones_and_root[..., transform_data["root"], :],
        bones_and_root[..., transform_data["bone_mask"], :],
    )


def join_with_root(bones, roots, transform_data):
    return np.insert(bones, transform_data["root"], roots, axis=-2)


def inverse_bone_transform(roots, bones, transform_data):
    if roots is None:
        roots = np.zeros(bones.shape[:-2] + (bones.shape[-1],))
    bones_and_root = join_with_root(bones, roots, transform_data)
    return transform_data["u_to_x"] @ bones_and_root
