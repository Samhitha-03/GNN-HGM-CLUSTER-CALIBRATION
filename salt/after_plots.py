import glob
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_DIR = "/nfs/mahler_2/jhubuserdirs/gu_parackal/salt/salt/processed_dataset/mc20_v1/train/"
VAL_DIR   = "/nfs/mahler_2/jhubuserdirs/gu_parackal/salt/salt/processed_dataset/mc20_v1/val/"
OUT_DIR   = "plots_h5_after"
os.makedirs(OUT_DIR, exist_ok=True)

train_files = sorted(glob.glob(TRAIN_DIR + "*.h5"))
val_files   = sorted(glob.glob(VAL_DIR   + "*.h5"))
all_files   = train_files + val_files
print(f"Train files: {len(train_files)}  |  Val files: {len(val_files)}")



features_config = {
    "target_response": {
        "source": "clusters_raw", "is_ratio": True,
        "label": r"target response $\mathcal{R}_{\mathrm{clus}}^{\mathrm{EM}}$",
        "bins": 100,
        "x_range":(-4, 2.5), 
        "y_bottom":1e-5
    },
    "clusterE": {
        "source": "clusters", "is_ratio": False, "branch": "clusterE",
        "label": r"cluster signal $E_{\mathrm{EM}}^{\mathrm{Clus}}$ [GeV]",
        "bins": 100,
    },
    "clusterEta": {
        "source": "clusters", "is_ratio": False, "branch": "clusterEta",
        "label": r"cluster rapidity $y_{\mathrm{Clus}}^{\mathrm{EM}}$",
        "bins": 100,
    },
    "cluster_SIGNIFICANCE": {
        "source": "clusters", "is_ratio": False, "branch": "cluster_SIGNIFICANCE",
        "label": r"signal significance $\zeta_{\mathrm{clus}}^{\mathrm{EM}}$",
        "bins": 100,
    },
    "cluster_time": {
        "source": "clusters", "is_ratio": False, "branch": "cluster_time",
        "label": r"cluster time $t_{\mathrm{clus}}$ [ns]",
        "bins": 100,
    },
    "cluster_SECOND_TIME": {
        "source": "clusters", "is_ratio": False, "branch": "cluster_SECOND_TIME",
        "label": r"cell-time variance $\mathrm{Var}_{\mathrm{clus}}(t_{\mathrm{cell}})\ [\mathrm{ns}^2]$",
        "bins": 100,
    },
    "cluster_CENTER_LAMBDA": {
        "source": "clusters", "is_ratio": False, "branch": "cluster_CENTER_LAMBDA",
        "label": r"depth in calorimeter $\lambda_{\mathrm{clus}}$ [mm]",
        "bins": 100,
    },
    "cluster_CENTER_MAG": {
        "source": "clusters", "is_ratio": False, "branch": "cluster_CENTER_MAG",
        "label": r"distance from vertex $|\vec{c}_{\mathrm{clus}}|$ [mm]",
        "bins": 100,
    },
    "cluster_ENG_FRAC_EM_INCL": {
        "source": "clusters_raw", "is_ratio": False, "branch": "cluster_ENG_FRAC_EM_INCL",
        "label": r"signal fraction in EMC $f_{\mathrm{emc}}$",
        "bins": 100,
    },
    "cluster_FIRST_ENG_DENS": {
        "source": "clusters", "is_ratio": False, "branch": "cluster_FIRST_ENG_DENS",
        "label": r"signal density $\rho_{\mathrm{clus}} = \langle \rho_{\mathrm{cell}} \rangle$ [GeV/mm$^3$]",
        "bins": 100,
    },
    "cluster_LONGITUDINAL": {
        "source": "clusters", "is_ratio": False, "branch": "cluster_LONGITUDINAL",
        "label": r"longitudinal signal dispersion $\langle m^2_{\mathrm{long}} \rangle$",
        "bins": 100,
    },
    "cluster_LATERAL": {
        "source": "clusters", "is_ratio": False, "branch": "cluster_LATERAL",
        "label": r"lateral signal dispersion $\langle m^2_{\mathrm{lat}} \rangle$",
        "bins": 100,
    },
    "cluster_PTD": {
        "source": "clusters", "is_ratio": False, "branch": "cluster_PTD",
        "label": r"signal compactness $p_T D$",
        "bins": 100,
    },
    "cluster_ISOLATION": {
        "source": "clusters", "is_ratio": False, "branch": "cluster_ISOLATION",
        "label": r"cluster isolation $f_{\mathrm{iso}}$",
        "bins": 100,
    },
    "nPrimVtx": {
        "source": "jet", "is_ratio": False, "branch": "nPrimVtx",
        "label": r"number of reconstructed primary vertices $N_{\mathrm{PV}}$",
        "bins": 100,
    },
    "avgMu": {
        "source": "jet", "is_ratio": False, "branch": "avgMu",
        "label": r"number of pile-up interactions $\mu$",
        "bins": 100,
    },
}


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'Liberation Sans'],
    'text.usetex': False,
    'xtick.top': True, 'ytick.right': True,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 8, 'xtick.minor.size': 4,
    'ytick.major.size': 8, 'ytick.minor.size': 4,
    'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
})

# ── Load flat arrays ───────────────────────────────────────────────────────
def load_values(cfg, files):
    chunks = []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            if cfg["source"] == "jet":
                jet = f["jet"][:]
                chunk = jet[cfg["branch"]].astype(np.float32)
            elif cfg["is_ratio"]:
                raw    = f["clusters_raw"][:]
                valid  = raw["valid"].astype(bool)
                calib  = raw["cluster_ENG_CALIB_TOT"][valid].astype(np.float32)
                clus_e = raw["clusterE"][valid].astype(np.float32)
                safe   = (calib > 0.1) & (clus_e > 0)
                ratio  = clus_e[safe] / calib[safe]
                chunk  = np.log10(ratio[ratio > 0])
            else:
                data  = f[cfg["source"]][:]
                valid = data["valid"].astype(bool)
                chunk = data[cfg["branch"]][valid].astype(np.float32)
        chunks.append(chunk)
    flat = np.concatenate(chunks)
    return flat[np.isfinite(flat)]

# ── Main plotting loop ─────────────────────────────────────────────────────
for config_name, cfg in features_config.items():
    print(f"Processing: {config_name}")
    flat = load_values(cfg, all_files)

    lo, hi = np.percentile(flat, [0.1, 99.9])
    bin_edges = np.linspace(lo, hi, cfg["bins"] + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    counts, _ = np.histogram(flat, bins=bin_edges)
    density = counts / (flat.size * bin_width)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.step(bin_edges[:-1], density, where='post', color='#2b7bba', linewidth=1.5)

    ax.set_yscale('log')
    ax.set_xlabel(cfg["label"], fontsize=14, loc='right')
    ax.set_ylabel("normalized", fontsize=14, loc='top')
    ax.set_xlim(bin_edges[0], bin_edges[-1])

    nonzero = density[density > 0]
    ax.set_ylim(bottom=nonzero.min() * 0.3, top=nonzero.max() * 4)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"dist_after_{config_name}.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()
    print(f"  Saved -> {out_path}")
