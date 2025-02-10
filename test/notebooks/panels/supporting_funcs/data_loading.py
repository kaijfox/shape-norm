import jax.random as jr

jr.KeyArray = jr.PRNGKey
import shape_norm

from shape_norm.io import loaders
from shape_norm.models import joint
from shape_norm import config
from shape_norm.models import (
    instantiation,
    setup,
    pose,
    morph,
    util as model_util,
)
from shape_norm import fitting
from shape_norm.fitting import em
from shape_norm.io import alignment, features
from shape_norm.io.armature import Armature
from shape_norm.project.paths import Project
from shape_norm.fitting import methods
from shape_norm.fitting import scans
from shape_norm.pca import fit_with_center, CenteredPCA, PCAData
from shape_norm.io.dataset import PytreeDataset
from shape_norm.io.dataset_refactor import Dataset, SessionMetadata
from shape_norm.models.morph.lowrank_affine import LRAParams, model as lra_model
from shape_norm.viz import styles

from blscale_loader import loader, linear_skeletal as blscale_ls

from collections import defaultdict
from shape_norm.viz import util as vu
from scipy import spatial
import itertools as iit
import matplotlib.colors as mpl_col
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
from sklearn import metrics
import numpy as np
import re
from scipy.spatial import distance
import jax
import jax.numpy as jnp
from pprint import pprint
import numpy.linalg as la
from ruamel.yaml import YAML
import scipy.stats
from pathlib import Path
from cmap import Colormap
from bidict import bidict
import matplotlib as mpl
import joblib as jl
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import itertools as iit
import pandas as pd
import seaborn as sns

from .plotting import getc
from .keypoint_data import (
    mouse_data_armature_meta,
    full_ontogeny_data_aligned,
    blscale_dataset,
    blscale_dataset_aligned,
)
from .shapenorm_helpers import merge_morphs


def plotting_metadata(
    params_file,
):

    plot_meta = YAML(typ="safe").load(open(params_file, "r"))
    age_pal = {k: getc(c) for k, c in plot_meta["colors"]["age"].items()}
    sim_pal = {k: getc(c) for k, c in plot_meta["colors"]["sim_age"].items()}

    kp_order = []
    kp_colors = {}
    kp_colormap = Colormap(plot_meta["colors"]["keypoint"][0])
    for kp_color in plot_meta["colors"]["keypoint"][1:]:
        for kp in kp_color[1:]:
            kp_order.append(kp)
            kp_colors[kp] = kp_colormap(kp_color[0])

    use_kpts = [
        "shldr",
        "back",
        "hips",
        "t_base",
        "head",
        "l_ear",
        "r_ear",
        "nose",
        "lr_knee",
        "lr_foot",
        "rr_knee",
        "rr_foot",
        "lf_foot",
        "rf_foot",
    ]
    keypt_ix = lambda name: use_kpts.index(name)

    _bones = jnp.array(
        [
            [keypt_ix(c), keypt_ix(mouse_data_armature_meta()[1][c])]
            for c in use_kpts
            if mouse_data_armature_meta()[1][c] is not None
        ]
    )
    arms = Armature(
        keypoint_names=bidict(zip(range(14), use_kpts)),
        bones=blscale_ls.reroot(_bones, keypt_ix("hips")),
        root=keypt_ix("hips"),
        anterior=["shldr"],
        posterior=["hips"],
    )

    # --- Organize bone names and order
    bone_names = [
        f"{arms.keypoint_names[int(c)]}-{arms.keypoint_names[int(p)]}"
        for c, p in arms.bones
    ]
    short_names = bidict(plot_meta["armature"]["short_names"])
    bones_by_short = {v: bone_names.index(k) for k, v in short_names.items()}
    bone_ix_order = [
        bone_names.index(short_names.inverse[b])
        for b in plot_meta["armature"]["order"]
    ]
    bone_groups = plot_meta["armature"]["groups"]
    bone_group_names = plot_meta["armature"]["group_names"]

    # organize joint names and order
    joint_names = plot_meta["armature"]["joint_order"]
    joint_short_names = bidict(plot_meta["armature"]["joint_short_names"])
    joint_by_short = {v: i for i, v in enumerate(joint_names)}
    joint_kp_ixs = np.array(
        [
            [k for k in joint_short_names.inverse[j].split("-")]
            for j in joint_names
        ]
    )
    joint_kp_ixs = np.array(
        [
            [
                arms.keypoint_names.inverse[k]
                for k in joint_short_names.inverse[j].split("-")
            ]
            for j in joint_names
        ]
    )

    return (
        sim_pal,
        age_pal,
        kp_order,
        kp_colors,
        kp_colormap,
        use_kpts,
        keypt_ix,
        _bones,
        arms,
        bone_names,
        short_names,
        bones_by_short,
        bone_ix_order,
        bone_groups,
        bone_group_names,
        joint_names,
        joint_short_names,
        joint_by_short,
        joint_kp_ixs,
    )


def load_ontogeny_keypoint_data(data_dir, src_sess):
    data_dir = "/Users/kaifox/projects/mph/data_explore/data"
    ont_dataset, unaligned_dataset, align_inv = full_ontogeny_data_aligned(
        Path(data_dir),
        session_filter=lambda s: re.search(r"(\d+)wk", s).group(1)
        not in ["9", "12"],
        use_keypoints=[
            "shldr",
            "back",
            "hips",
            "t_base",
            "head",
            "l_ear",
            "r_ear",
            "nose",
            "lr_knee",
            "lr_foot",
            "rr_knee",
            "rr_foot",
            "lf_foot",
            "rf_foot",
        ],
    )
    obs_keypts = {s: ont_dataset.get_session(s) for s in ont_dataset.sessions}
    unaligned = {
        s: unaligned_dataset.get_session(s) for s in unaligned_dataset.sessions
    }
    sessions = ont_dataset.sessions
    src_keypts = obs_keypts[src_sess]

    # sessions = metadata['session_slice'].keys()
    # slices = metadata['session_slice']
    # src_sess = '24wk_m0'
    # obs_keypts = {s: gt_obs.keypts[slices[s]].reshape([-1, 14, 3]) for s in sessions}
    # unaligned = {s: inverse_saggital_align(obs_keypts[s], metadata['centroid'][s], metadata['rotation'][s]) for s in sessions}
    # src_kpts = obs_keypts[src_sess]

    age_groups = defaultdict(list)
    for s in sessions:
        age_groups[s.split("_")[0].rstrip("wk")].append(s)
    age_groups = dict(age_groups)
    ages = sorted(list(age_groups.keys()), key=lambda x: int(x))

    print("Age groups:")
    pprint(age_groups)

    return (
        ont_dataset,
        unaligned_dataset,
        align_inv,
        obs_keypts,
        unaligned,
        sessions,
        src_keypts,
        age_groups,
        ages,
    )


def ontogeny_data_ethograms(ethogram_path, params_file, mode="behavior"):
    plot_meta = YAML(typ="safe").load(open(params_file, "r"))
    bhv_labels = jl.load(ethogram_path)
    bhv_pal = {k: getc(c) for k, c in plot_meta["colors"]["bhv"].items()}
    bhv_keys = plot_meta["behavior"]["keys"]
    bhv_names = plot_meta["behavior"]["names"]
    bhv_order = list(bhv_keys.keys())

    if mode == "bouts":
        bhv_masks = {k: bhv_labels["masks"][n] for k, n in bhv_keys.items()}
        rear_mask = bhv_labels["masks"]["absolute_rear"]
        loco_mask = bhv_labels["masks"]["locomotion"]

    elif mode == "ethogram":
        bhv_masks = {
            k: {
                s: m[bhv_labels["behaviors"].index(v)]
                for s, m in bhv_labels["masks"].items()
            }
            for k, v in bhv_keys.items()
        }
        rear_mask = None
        loco_mask = None

    return (
        bhv_pal,
        bhv_keys,
        bhv_names,
        bhv_order,
        bhv_masks,
        rear_mask,
        loco_mask,
        bhv_labels,
    )


def load_blscale_data(scale_metadata_dir, source_data_dir, base, ref_session):
    # blscale_dir = Path("../../../../data_explore/testsets/blscale-dicts")
    # data_dir = Path("../../../../data_explore/data")
    blscale_dir = Path(scale_metadata_dir)
    data_dir = Path(source_data_dir)

    # base = "3"
    source_dict = (
        blscale_dir / "source-dicts" / "cohort-wise" / f"{base}wk-to-all.yaml"
    )
    # scale_dict = blscale_dir / "scale-dicts" / "cohort-wise" / f"{base}wk-to-all_ntt-norm.yaml"
    scale_dict = blscale_dir / "scale-dicts" / f"all-to-named-age_unnormed.yaml"

    project_config_kws, blscale_keypoints, blscale_scales = blscale_dataset(
        source_dict, scale_dict, data_dir, ref_session=ref_session
    )

    return (
        source_dict,
        scale_dict,
        blscale_keypoints,
        blscale_scales,
        project_config_kws,
    )


def load_blscale_dataset_aligned(
    scale_metadata_dir, source_data_dir, base, ref_session, use_kpts
):
    blscale_dir = Path(scale_metadata_dir)
    data_dir = Path(source_data_dir)

    source_dict = (
        blscale_dir / "source-dicts" / "cohort-wise" / f"{base}wk-to-all.yaml"
    )
    scale_dict = (
        blscale_dir
        / "scale-dicts"
        / "cohort-wise"
        / f"{base}wk-to-all_ntt-norm.yaml"
    )

    _, blscale_keypoints, blscale_scales = blscale_dataset(
        source_dict, scale_dict, data_dir, ref_session=ref_session
    )
    blscale_aligned, blscale_unaligned, blscale_align_inv = (
        blscale_dataset_aligned(
            source_dict,
            scale_dict,
            data_dir,
            ref_session=ref_session,
            use_keypoints=use_kpts,
        )
    )

    return (
        source_dict,
        scale_dict,
        blscale_keypoints,
        blscale_scales,
        blscale_aligned,
        blscale_unaligned,
        blscale_align_inv,
    )


def blscale_moseq_syllable_files(syllables_dir):

    syll_fmt = lambda s="": f"pop-results{s}-slds_fit.p"

    # apply-only
    syll_files = pd.DataFrame(
        sum(
            [
                [
                    dict(
                        sylls=syllables_dir
                        / f"3+24+72wk-to-all_{norm}"
                        / f"wk3+24+72-ref3_to{tgt}{run}"
                        / syll_fmt(f"-aseed{seed}"),
                        labels=None,
                        dataset=f"3+24+72.{tgt}{norm[0]}{short}-s{seed}",
                    )
                    for seed in range(1, 4)
                ]
                for tgt in [5, 7, 24, 52, 72]
                for (norm, run, short) in [
                    ("ntt-norm", "-mseed1", "-m1"),
                    ("ntt-norm", "-mseed2", "-m2"),
                ]
            ],
            [],
        )
    )

    for f in syll_files.sylls:
        if not f.exists():
            print(f"File {f} does not exist")

    return syll_files


def load_syllable_files(syll_files):
    onsets = []
    counts = []
    usages = []
    n_sylls = 100
    all_results = {}
    reget = lambda s, pat, g=1: m.group(g) if (m := re.match(pat, s)) else None
    for i, file in syll_files.iterrows():
        results = jl.load(file.sylls)
        all_results[file.dataset] = results
        labels = (
            jl.load(file.labels)["masks"] if file.labels is not None else {}
        )
        labels["none"] = {
            s: np.ones_like(r["syllable"]).astype(bool)
            for s, r in results.items()
        }

        for bhv in labels:
            for session, sess_results in results.items():
                meta = dict(
                    session=session,
                    bhv=bhv,
                    dataset=file.dataset,
                    age=reget(
                        session.split("-")[-1].split("_")[0], r"(.+)wk$", 1
                    ),
                    body=reget(
                        session.split("-")[-1].split("_")[1], r"(.+)bod$", 1
                    ),
                    animal_id=reget(
                        session.split("-")[-1].split("_")[-1], r"m(.+)$", 1
                    ),
                    condition=(
                        ""
                        if len(session.split("-")) == 0
                        else "-".join(session.split("-")[:-1])
                    ),
                )
                if session not in labels[bhv].keys():
                    print(f"No supervised labels for {session}, skipping.")
                    continue
                syllables = sess_results["syllable"][labels[bhv][session]]

                usages_data = (
                    pd.Series(syllables)
                    .value_counts()
                    .reindex(range(n_sylls), fill_value=0)
                    .rename("usage")
                )
                usages_data /= usages_data.sum()
                usages_data = (
                    pd.DataFrame(usages_data).reset_index().assign(**meta)
                )
                usages.append(usages_data)

                onset_ixs = np.where(np.diff(syllables))[0]
                onset_sylls = sess_results["syllable"][onset_ixs + 1]
                onset_df = pd.DataFrame(
                    dict(frame=onset_ixs[:-1], syllable=onset_sylls[:-1])
                )
                onset_df["duration"] = np.diff(onset_ixs)
                onsets.append(onset_df.assign(**meta))

                if onset_df.syllable.max() > n_sylls:
                    print(
                        f"Found syllable {onset_df.syllable.max()} > max label {n_sylls}"
                    )

                counts_data = onset_df.syllable.value_counts().reindex(
                    range(n_sylls), fill_value=0
                )
                counts.append(
                    pd.DataFrame(counts_data).reset_index().assign(**meta)
                )

    onsets = pd.concat(onsets)
    counts = pd.concat(counts)
    usages = pd.concat(usages)

    return onsets, counts, usages, all_results


def load_blscale_kpsn_model(
    project_dir,
    scan_name,
    model_name,
    params_file,
    source_data_path,
):
    plot_meta = YAML(typ="safe").load(open(params_file, "r"))

    assert project_dir.exists()
    project = Project(project_dir)

    ckpt = methods.load_fit(project.model(model_name))
    cfg = ckpt["config"]
    model = instantiation.get_model(cfg)
    arms = Armature.from_config(cfg["dataset"])
    params = ckpt["params"]

    # --- Organize bone names and order
    bone_names = [
        f"{arms.keypoint_names[int(c)]}-{arms.keypoint_names[int(p)]}"
        for c, p in arms.bones
    ]
    short_names = bidict(plot_meta["armature"]["short_names"])
    bone_ix_order = [
        bone_names.index(short_names.inverse[b])
        for b in plot_meta["armature"]["order"]
    ]
    bone_groups = plot_meta["armature"]["groups"]
    bone_group_names = plot_meta["armature"]["group_names"]

    if cfg["dataset"]["type"] == "arrays":
        raise NotImplementedError("array type project deprecated")
        dataset = loaders.arrays.from_arrays(dataset_keypoints, cfg["dataset"])
        scan_dataset, split_meta, align_meta = scans.prepare_scan_dataset(
            dataset, project, scan_name, return_session_inv=True
        )

    if cfg["dataset"]["type"] == "raw_npy":

        # Set location for keypoint data and calibration
        root = Path(cfg["dataset"]["root_path"])
        print(root)
        cfg["dataset"]["root_path"] = Path(source_data_path) / root.name
        cfg["calibration_file"] = project_dir.resolve() / "project.calib.p"

        # ignore any previously loaded blscale dataset
        dataset_keypoints = None

        # load dataset
        dataset = loaders.load_dataset(cfg["dataset"])
        scan_cfg = config.load_config(project.scan(scan_name) / "scan.yml")
        dataset_versions, split_meta, align_meta = scans.prepare_scan_dataset(
            dataset, cfg, return_session_inv=True, all_versions=True
        )
        scan_dataset = dataset_versions["train"]
        dataset_feats = dataset_versions["reduced"]
        # load dataset without subsampling
        dataset_full = loaders.load_dataset(
            cfg["dataset"], allow_subsample=False
        )
        scan_dataset_full, _, _ = scans.prepare_scan_dataset(dataset_full, cfg)
        dataset_versions_full, _, _ = scans.prepare_scan_dataset(
            dataset_full, cfg, return_session_inv=True, all_versions=True
        )
        scan_dataset_full = dataset_versions_full["train"]
        dataset_feats_full = dataset_versions_full["reduced"]

        merged_params = merge_morphs(
            params.morph,
            dataset.session_meta,
            scan_dataset.session_meta,
            split_meta[0],
        )

        # integrate blscale "uniform" into alignment scales
        blscale_metadata = YAML().load(
            cfg["dataset"]["root_path"] / "metadata.yml"
        )
        blscale_uniform = {
            s: blscale_metadata["scale_dict"][s]["uniform"]
            for s in dataset.sessions
        }
        align_meta["scale"] = [
            s / blscale_uniform[dataset.session_name(i)]
            for i, s in enumerate(align_meta["scale"])
        ]

    return (
        (project, ckpt, cfg, model, arms, params),
        (bone_names, short_names, bone_ix_order, bone_groups, bone_group_names),
        root,
        (
            dataset,
            scan_cfg,
            dataset_versions,
            split_meta,
            align_meta,
            scan_dataset,
            dataset_feats,
        ),
        (
            dataset_full,
            dataset_versions_full,
            scan_dataset_full,
            dataset_feats_full,
        ),
        (blscale_metadata, blscale_uniform),
        merged_params,
    )


def load_ont_kpsn_model(
    project_dir,
    scan_name,
    model_name,
    params_file,
    dataset_keypoints,
):
    assert project_dir.exists()
    project = Project(project_dir)

    plot_meta = YAML(typ="safe").load(open(params_file, "r"))

    ckpt = methods.load_fit(project.model(model_name))
    cfg = ckpt["config"]
    model = instantiation.get_model(cfg)
    arms = Armature.from_config(cfg["dataset"])
    params = ckpt["params"]

    # --- Organize bone names and order
    bone_names = [
        f"{arms.keypoint_names[int(c)]}-{arms.keypoint_names[int(p)]}"
        for c, p in arms.bones
    ]
    short_names = bidict(plot_meta["armature"]["short_names"])
    bone_ix_order = [
        bone_names.index(short_names.inverse[b])
        for b in plot_meta["armature"]["order"]
    ]
    bone_groups = plot_meta["armature"]["groups"]
    bone_group_names = plot_meta["armature"]["group_names"]

    # organize joint names and order
    joint_names = plot_meta["armature"]["joint_order"]
    joint_short_names = bidict(plot_meta["armature"]["joint_short_names"])
    # joint_kp_ixs = np.array(
    #     [
    #         [k for k in joint_short_names.inverse[j].split("-")]
    #         for j in joint_names
    #     ]
    # )
    joint_kp_ixs = np.array(
        [
            [
                arms.keypoint_names.inverse[k]
                for k in joint_short_names.inverse[j].split("-")
            ]
            for j in joint_names
        ]
    )

    if cfg["dataset"]["type"] == "arrays":
        # Organize dataset_keypoints as Dataset object
        dataset = loaders.arrays.from_arrays(dataset_keypoints, cfg["dataset"])
        dataset_versions, split_meta, align_meta = scans.prepare_scan_dataset(
            dataset, cfg, return_session_inv=True, all_versions=True
        )
        dataset_feats = dataset_versions["reduced"]
        scan_dataset = dataset_versions["train"]

        # Load full dataset without subsampling
        dataset_full = loaders.arrays.from_arrays(
            dataset_keypoints, cfg["dataset"], allow_subsample=False
        )
        dataset_versions_full, _, _ = scans.prepare_scan_dataset(
            dataset_full, cfg, return_session_inv=True, all_versions=True
        )
        dataset_feats_full = dataset_versions_full["reduced"]
        scan_dataset_full = dataset_versions_full["train"]

        # Generate final merged paramerters object
        merged_params = merge_morphs(
            params.morph,
            dataset.session_meta,
            scan_dataset.session_meta,
            split_meta[0],
        )

    if cfg["dataset"]["type"] == "raw_npy":
        raise ValueError

    return (
        (project, ckpt, cfg, model, arms, params),
        (bone_names, short_names, bone_ix_order, bone_groups, bone_group_names),
        (joint_names, joint_short_names, joint_kp_ixs),
        (
            dataset,
            dataset_versions,
            split_meta,
            align_meta,
            scan_dataset,
            dataset_feats,
        ),
        (
            dataset_full,
            dataset_versions_full,
            scan_dataset_full,
            dataset_feats_full,
        ),
        merged_params,
    )
