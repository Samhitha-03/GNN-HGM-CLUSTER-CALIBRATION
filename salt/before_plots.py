import glob
import os
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. Paths and verified tree configuration
raw_root_dir = "/nfs/mahler_2/jhubuserdirs/biswas/PileupStudies/dataset/"
root_files = sorted(glob.glob(os.path.join(raw_root_dir, "mc20*.root")))
tree_name = "ClusterTree"

if not root_files:
    raise FileNotFoundError("No files matched the 'mc20*.root' pattern.")

# 2. Full feature config
features_config = {
    "target_response_log": {
        "is_ratio": True, "is_event_level": False,
        "label": r"target response $\mathcal{R}_{\text{clus}}^{\text{EM}}$",
        "bins": 100, "log_x": True, "auto_range": True,
    },
    "target_response_linear": {
        "is_ratio": True, "is_event_level": False,
        "label": r"target response $\mathcal{R}_{\text{clus}}^{\text{EM}}$",
        "bins": 100, "log_x": False, "auto_range": True,
    },
    "clusterE": {
        "is_ratio": False, "is_event_level": False, "branch": "clusterE",
        "label": r"cluster signal $E_{EM}^{Clus}$ [GeV]",
        "bins": 100, "log_x": False, "auto_range": True,
    },
    "clusterEta": {
        "is_ratio": False, "is_event_level": False, "branch": "clusterEta",
        "label": r"cluster rapidity $y_{Clus}^{EM}$",
        "bins": 50, "range": (-2.5, 2.5), "log_x": False,
    },
    "cluster_SIGNIFICANCE": {
        "is_ratio": False, "is_event_level": False, "branch": "cluster_SIGNIFICANCE",
        "label": r"signal significance $\zeta_{clus}^{EM}$",
        "bins": 100, "log_x": False, "auto_range": True,
    },
    "cluster_time": {
        "is_ratio": False, "is_event_level": False, "branch": "cluster_time",
        "label": r"cluster time $t_{clus}$ [ns]",
        "bins": 100, "range": (-250, 100), "log_x": False,
    },
    "cluster_SECOND_TIME": {
        "is_ratio": False, "is_event_level": False, "branch": "cluster_SECOND_TIME",
        "label": r"cell-time variance $\mathrm{Var}_{clus}(t_{cell})\ [\mathrm{ns}^2]$",
        "bins": 100, "log_x": False, "auto_range": True,
    },
    "cluster_CENTER_LAMBDA": {
        "is_ratio": False, "is_event_level": False, "branch": "cluster_CENTER_LAMBDA",
        "label": r"depth in calorimeter $\lambda_{clus}$ [mm]",
        "bins": 100, "log_x": False, "auto_range": True,
    },
    "cluster_CENTER_MAG": {
        "is_ratio": False, "is_event_level": False, "branch": "cluster_CENTER_MAG",
        "label": r"distance from vertex $|\vec{c}_{clus}|$ [mm]",
        "bins": 100, "log_x": False, "auto_range": True,
    },
    "cluster_ENG_FRAC_EM": {
        "is_ratio": False, "is_event_level": False, "branch": "cluster_ENG_FRAC_EM",
        "label": r"signal fraction in EMC $f_{emc}$",
        "bins": 100, "range": (0.0, 1.0), "log_x": False,
    },
    "cluster_FIRST_ENG_DENS": {
        "is_ratio": False, "is_event_level": False, "branch": "cluster_FIRST_ENG_DENS",
        "label": r"signal density $\rho_{clus} = \langle \rho_{cell} \rangle$ [GeV/mm$^3$]",
        "bins": 100, "log_x": False, "auto_range": True,
    },
    "cluster_LONGITUDINAL": {
        "is_ratio": False, "is_event_level": False, "branch": "cluster_LONGITUDINAL",
        "label": r"longitudinal signal dispersion $\langle m^2_{long} \rangle$",
        "bins": 100, "range": (0.0, 1.0), "log_x": False,
    },
    "cluster_LATERAL": {
        "is_ratio": False, "is_event_level": False, "branch": "cluster_LATERAL",
        "label": r"lateral signal dispersion $\langle m^2_{lat} \rangle$",
        "bins": 100, "range": (0.0, 1.0), "log_x": False,
    },
    "cluster_PTD": {
        "is_ratio": False, "is_event_level": False, "branch": "cluster_PTD",
        "label": r"signal compactness $p_T^D$",
        "bins": 100, "range": (0.0, 1.0), "log_x": False,
    },
    "cluster_ISOLATION": {
        "is_ratio": False, "is_event_level": False, "branch": "cluster_ISOLATION",
        "label": r"cluster isolation $f_{iso}$",
        "bins": 100, "range": (0.0, 1.0), "log_x": False,
    },
    "nPrimVtx": {
        "is_ratio": False, "is_event_level": True, "branch": "nPrimVtx",
        "label": r"number of reconstructed primary vertices $NPV$",
        "bins": 80, "range": (0.5, 80.5), "log_x": False,
    },
    "avgMu": {
        "is_ratio": False, "is_event_level": True, "branch": "avgMu",
        "label": r"number of pile-up interactions $\mu$",
        "bins": 90, "range": (0.5, 90.5), "log_x": False,
    },
}

# 3. ATLAS-style layout
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'Liberation Sans'],
    'text.usetex': False,
    'xtick.top': True,
    'ytick.right': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.major.size': 8,
    'ytick.minor.size': 4,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})


def load_flat_values(cfg):
    """Load the full flattened, finite array for a given feature config."""
    if cfg["is_event_level"]:
        chunks = []
        for f in root_files:
            with uproot.open(f"{f}:{tree_name}") as tree:
                arr = tree[cfg["branch"]].array(library="np")
                chunks.append(arr)
        flat = np.concatenate(chunks)
    else:
        expressions = ["cluster_ENG_CALIB_TOT", "clusterE"] if cfg["is_ratio"] else [cfg["branch"]]
        chunks = []
        for batch in uproot.iterate(
            [f"{f}:{tree_name}" for f in root_files],
            expressions=expressions,
            step_size="150MB",
            library="ak",
        ):
            if cfg["is_ratio"]:
                calib_tot = ak.to_numpy(ak.flatten(batch["cluster_ENG_CALIB_TOT"], axis=None))
                clus_e = ak.to_numpy(ak.flatten(batch["clusterE"], axis=None))
                safe_mask = (calib_tot > 0.1) & (clus_e > 0)
                chunk = clus_e[safe_mask] / calib_tot[safe_mask]
            else:
                chunk = ak.to_numpy(ak.flatten(batch[cfg["branch"]], axis=None))
            chunks.append(chunk)
        flat = np.concatenate(chunks)

    flat = flat[np.isfinite(flat)]
    return flat


# 4. Main loop
os.makedirs("plots_before", exist_ok=True)
for config_name, cfg in features_config.items():
    print(f"Processing: {config_name}")
    flat_chunk = load_flat_values(cfg)

    if cfg.get("auto_range"):
        lo, hi = np.percentile(flat_chunk, [0.1, 99.9])
        if cfg["log_x"]:
            lo = max(lo, 1e-5)
        rng = (lo, hi)
    else:
        rng = cfg["range"]

    if cfg["log_x"]:
        bin_edges = np.logspace(np.log10(rng[0]), np.log10(rng[1]), cfg["bins"] + 1)
    else:
        bin_edges = np.linspace(rng[0], rng[1], cfg["bins"] + 1)

    hist_counts, _ = np.histogram(flat_chunk, bins=bin_edges)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.step(bin_edges[:-1], hist_counts, where='post', color='#2b7bba', linewidth=1.5)

    ax.set_yscale('log')
    if cfg["log_x"]:
        ax.set_xscale('log')

    ax.set_xlabel(cfg["label"], fontsize=14, loc='right')
    ax.set_ylabel("events" if cfg["is_event_level"] else "topo-clusters", fontsize=14, loc='top')

    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_ylim(bottom=0.5, top=np.max(hist_counts) * 4)

    ax.xaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs='auto') if cfg["log_x"] else ticker.AutoMinorLocator()
    )
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))

    plt.tight_layout()
    plt.savefig(f"plots_before/raw_dist_{config_name}.png", dpi=300)
    plt.show()
    plt.close()
