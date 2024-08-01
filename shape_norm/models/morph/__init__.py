from . import lowrank_affine
from . import bone_length

morph_models = {
    lowrank_affine.type_name: lowrank_affine.model,
    bone_length.type_name: bone_length.model,
}
