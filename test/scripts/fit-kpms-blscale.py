# -------------------------------------------------------  Setup  -------------

import numpy as np
import joblib as jl
import glob, os, re
from pathlib import Path
import keypoint_moseq as kpms

# -------------------------------------------------------  Parameters  --------

root_dir = Path("/n/groups/datta/kai/mph/moseq-explore")
blscale_dir = Path("/n/groups/datta/kai/mph/testsets/blscale-arrays")

src = "3"
tgt = ["52"]
dset_name = f"3wk-to-all_ntt-norm"
model_name = f"wk{src}_to{'-'.join(tgt)}"

parent_project_dir = root_dir / "kpms_projects" / "blscale-arrays"


# -------------------------------------------------------  Project def'n  -----


project_dir = parent_project_dir / dset_name / model_name

project_dir.mkdir(parents=True, exist_ok=True)
with open(str(project_dir / "source.txt"), "w") as f:
    f.write("kpms_blscale-arrays.ipynb")

data_dir = blscale_dir / dset_name
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)

print("Dataset:", data_dir)
print("Project:", project_dir)

config = lambda: kpms.load_config(project_dir)


# if parent_project_dir has not yet been initialized with a config.yml
bodyparts = (
    ["shldr", "back", "hips", "t_base", "t_tip"]
    + ["head", "l_ear", "r_ear", "nose"]
    + ["lr_knee", "lr_foot", "rr_knee", "rr_foot", "lf_foot", "rf_foot"]
)

skeleton_ixs = [0, 0, 1, 2, 3, 0, 5, 5, 5, 2, 9, 2, 11, 0, 0]
skeleton = [
    [bodyparts[start_ix], bodyparts[end_ix]]
    for start_ix, end_ix in enumerate(skeleton_ixs)
][1:]

kpms.setup_project(
    project_dir,
    overwrite=True,
    bodyparts=bodyparts,
    skeleton=skeleton,
    anterior_bodyparts=["hips"],
    posterior_bodyparts=["shldr"],
    use_bodyparts=(
        ["shldr", "back", "hips", "t_base", "head", "l_ear", "r_ear", "nose"]
    ),
)

# -------------------------------------------------------  Load data  ---------

# find files filter to target ages
filenames = {
    f[:-4]: data_dir / f for f in os.listdir(data_dir) if f.endswith(".npy")
}
if not (len(tgt) == 1 and tgt[0] == "-all"):
    for s in list(filenames.keys()):
        if re.search(r"(\d+)bod", s).group(1) not in [src] + tgt:
            filenames.pop(s)

# load arrays and format to kpms data strcuture
coords = {s: np.load(f) for s, f in filenames.items()}
confs = {s: np.ones_like(c[..., 0]) for s, c in coords.items()}

print("Sessions:", list(coords.keys()))
data, metadata = kpms.format_data(coords, confs, **config())


# -------------------------------------------------------  Fit KPMS  ----------

pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, None, pca_path=project_dir / "pca.p")
kpms.update_config(parent_project_dir, latent_dim=4)

model = kpms.init_model(data, pca=pca, **config())

num_ar_iters = 50
model_name = "arhmm_fit"
model, model_name = kpms.fit_model(
    model,
    data,
    metadata,
    project_dir,
    model_name=model_name,
    ar_only=True,
    num_iters=num_ar_iters,
)

# load model checkpoint
model_name = "arhmm_fit"
model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir, model_name, iteration=num_ar_iters
)

# modify kappa to maintain the desired syllable time-scale
model = kpms.update_hypparams(model, kappa=1e4)

# run fitting for an additional 1000 iters
new_model_name = "slds_fit"
model = kpms.fit_model(
    model,
    data,
    metadata,
    project_dir,
    new_model_name,
    ar_only=False,
    start_iter=current_iter,
    num_iters=current_iter + 1000,
    parallel_message_passing=True,
)[0]


# -------------------------------------------------------  Save results  -------

model_name = "slds_fit"
model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir, model_name, iteration=1050
)

results = kpms.extract_results(
    model, metadata, project_dir, model_name, save_results=False
)
jl.dump(results, f"{results_dir}/pop-results-{model_name}.p")

for seed in [1, 2, 3]:
    model_name = "slds_fit"
    run_name = f"seed{seed}"
    model, data, metadata, current_iter = kpms.load_checkpoint(
        project_dir, model_name, iteration=1050
    )
    pca = kpms.load_pca(project_dir)

    results, _ = kpms.apply_model(
        model,
        pca,
        data,
        metadata,
        project_dir,
        model_name,
        save_results=False,
        parallel_message_passing=True,
        seed=seed,
        **config(),
    )
    out_file = f"{project_dir}/pop-results-{run_name}-{model_name}.p"
    print(out_file)
    jl.dump(results, out_file)
