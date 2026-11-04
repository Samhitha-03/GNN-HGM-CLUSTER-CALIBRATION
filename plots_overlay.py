import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. FILE PATHS
# ============================================================

STANDARD_FILE = (
   "/nfs/mahler_2/jhubuserdirs/gu_abraham/logs/"
   "gnn_regression_20260730-T004855/ckpts/"
   "epoch=019-val_loss=0.00077__test_vds.h5"
)

HGM_FILE = (
   "/nfs/mahler_2/jhubuserdirs/gu_abraham/logs/"
   "gnn_hgm_regression_20260802-T010103/ckpts/"
   "epoch=019-val_loss=-6.00700__test_vds.h5"
)

GMM_FILE = (
   "/nfs/mahler_2/jhubuserdirs/gu_parackal/salt/salt/logs/"
   "gnn_hgm_gmm_regression_20260803-T121938/ckpts/"
   "epoch=020-val_loss=-3.85937__test_vds.h5"
)




STANDARD_FIELD = "gnn_regression_cluster_ENG_CALIB_TOT"

HGM_FIELD = (
   "cluster_hgm_response_regression_cluster_ENG_CALIB_TOT"
)

GMM_FIELD = (
   "gnn_hgm_gmm_regression_cluster_ENG_CALIB_TOT_point_estimate"
)




def load_prediction(path, prediction_field):

   print(f"\nLoading:\n{path}")

   with h5py.File(path, "r") as f:

       d = f["clusters"]

       clusterE = d["clusterE"][:].astype(np.float64)

       truth = (
           d["cluster_ENG_CALIB_TOT"][:]
           .astype(np.float64)
       )

       pred = (
           d[prediction_field][:]
           .astype(np.float64)
       )

       mask = (
           d["mask"][:]
           .astype(bool)
       )

   return clusterE, truth, pred, mask




clusterE, truth, pred_standard, mask = load_prediction(
   STANDARD_FILE,
   STANDARD_FIELD
)

clusterE_hgm, truth_hgm, pred_hgm, mask_hgm = load_prediction(
   HGM_FILE,
   HGM_FIELD
)

clusterE_gmm, truth_gmm, pred_gmm, mask_gmm = load_prediction(
   GMM_FILE,
   GMM_FIELD
)




print("\n" + "=" * 70)
print("DATASET CONSISTENCY CHECK")
print("=" * 70)

print(
   "clusterE Standard/HGM identical:",
   np.array_equal(clusterE, clusterE_hgm)
)

print(
   "clusterE Standard/GMM identical:",
   np.array_equal(clusterE, clusterE_gmm)
)

print(
   "Truth Standard/HGM identical:",
   np.array_equal(truth, truth_hgm)
)

print(
   "Truth Standard/GMM identical:",
   np.array_equal(truth, truth_gmm)
)

print(
   "Mask Standard/HGM identical:",
   np.array_equal(mask, mask_hgm)
)

print(
   "Mask Standard/GMM identical:",
   np.array_equal(mask, mask_gmm)
)


valid = (
   (~mask)
   & np.isfinite(truth)
   & np.isfinite(pred_standard)
   & np.isfinite(pred_hgm)
   & np.isfinite(pred_gmm)
)

truth = truth[valid]

pred_standard = pred_standard[valid]
pred_hgm = pred_hgm[valid]
pred_gmm = pred_gmm[valid]


print("\n" + "=" * 70)
print("FINAL SAMPLE")
print("=" * 70)

print(f"Non-padded clusters used : {len(truth):,}")






diff_standard = pred_standard - truth
diff_hgm = pred_hgm - truth
diff_gmm = pred_gmm - truth




def print_metrics(name, truth, pred):

   diff = pred - truth

   bias = np.mean(diff)
   resolution = np.std(diff)
   rmse = np.sqrt(np.mean(diff**2))
   mae = np.mean(np.abs(diff))

   correlation = np.corrcoef(
       truth,
       pred
   )[0, 1]

   print("\n" + "=" * 60)
   print(name)
   print("=" * 60)

   print(f"Clusters          : {len(pred):,}")

   print("\nEnergy:")
   print(f"Bias              : {bias:.6f} GeV")
   print(f"Resolution (std)  : {resolution:.6f} GeV")
   print(f"RMSE              : {rmse:.6f} GeV")
   print(f"MAE               : {mae:.6f} GeV")
   print(f"Correlation       : {correlation:.8f}")


print_metrics(
   "STANDARD REGRESSION",
   truth,
   pred_standard
)

print_metrics(
   "HGM",
   truth,
   pred_hgm
)

print_metrics(
   "GMM K=3",
   truth,
   pred_gmm
)




all_diffs = np.concatenate([
   diff_standard,
   diff_hgm,
   diff_gmm
])


plot_low = np.percentile(all_diffs, 0.5)
plot_high = np.percentile(all_diffs, 99.5)

print("\nEnergy difference plotting range:")
print(f"{plot_low:.3f} to {plot_high:.3f} GeV")


plt.figure(figsize=(10, 7))


plt.hist(
   diff_standard,
   bins=250,
   range=(plot_low, plot_high),
   density=True,
   histtype="step",
   linewidth=2,
   label="Standard Regression"
)


plt.hist(
   diff_hgm,
   bins=250,
   range=(plot_low, plot_high),
   density=True,
   histtype="step",
   linewidth=2,
   label="HGM"
)


plt.hist(
   diff_gmm,
   bins=250,
   range=(plot_low, plot_high),
   density=True,
   histtype="step",
   linewidth=2,
   label="GMM K=3"
)


plt.axvline(
   0,
   linestyle="--",
   linewidth=1.5
)


plt.xlabel(
   r"$E_{\mathrm{pred}} - E_{\mathrm{truth}}$ [GeV]",
   fontsize=14
)

plt.ylabel(
   "Normalized entries",
   fontsize=14
)

plt.title(
   "Energy Prediction Residuals",
   fontsize=15
)

plt.legend(
   fontsize=12
)

plt.grid(
   True,
   alpha=0.25
)

plt.tight_layout()


plt.savefig(
   "Final_EnergyDifference_Overlay.png",
   dpi=300,
   bbox_inches="tight"
)

plt.close()



bins = np.array([
   0,
   10,
   20,
   30,
   40,
   50,
   75,
   100,
   150,
   200,
   300,
   400,
   500,
   700,
   1000
])


centres = []
counts = []

resolution_standard = []
resolution_hgm = []
resolution_gmm = []


for i in range(len(bins) - 1):

   selection = (
       (truth >= bins[i])
       &
       (truth < bins[i + 1])
   )

   n = np.sum(selection)

   # Only require enough statistics for the bin
   if n >= 30:

       centre = (
           bins[i] + bins[i + 1]
       ) / 2

       centres.append(centre)
       counts.append(n)

       resolution_standard.append(
           np.std(
               diff_standard[selection]
           )
       )

       resolution_hgm.append(
           np.std(
               diff_hgm[selection]
           )
       )

       resolution_gmm.append(
           np.std(
               diff_gmm[selection]
           )
       )


centres = np.array(centres)
counts = np.array(counts)

resolution_standard = np.array(
   resolution_standard
)

resolution_hgm = np.array(
   resolution_hgm
)

resolution_gmm = np.array(
   resolution_gmm
)




print("\n")
print("=" * 95)
print("BINNED ENERGY RESOLUTION")
print("=" * 95)

print(
   f"{'E truth':>10}"
   f"{'N':>14}"
   f"{'Standard':>16}"
   f"{'HGM':>16}"
   f"{'GMM K=3':>16}"
)


for c, n, s, h, g in zip(
   centres,
   counts,
   resolution_standard,
   resolution_hgm,
   resolution_gmm
):

   print(
       f"{c:10.1f}"
       f"{n:14d}"
       f"{s:16.6f}"
       f"{h:16.6f}"
       f"{g:16.6f}"
   )


# ============================================================
# 11. RESOLUTION PLOT
# ============================================================

plt.figure(
   figsize=(10, 7)
)


plt.plot(
   centres,
   resolution_standard,
   "o-",
   linewidth=2,
   markersize=6,
   label="Standard Regression"
)


plt.plot(
   centres,
   resolution_hgm,
   "o-",
   linewidth=2,
   markersize=6,
   label="HGM"
)


plt.plot(
   centres,
   resolution_gmm,
   "o-",
   linewidth=2,
   markersize=6,
   label="GMM K=3"
)


plt.xlabel(
   r"$E_{\mathrm{truth}}$ [GeV]",
   fontsize=14
)

plt.ylabel(
   r"$\sigma(E_{\mathrm{pred}}-E_{\mathrm{truth}})$ [GeV]",
   fontsize=14
)

plt.title(
   "Energy Resolution vs Truth Energy",
   fontsize=15
)


plt.legend(
   fontsize=12
)


plt.grid(
   True,
   alpha=0.25
)


plt.tight_layout()


plt.savefig(
   "Final_EnergyResolution.png",
   dpi=300,
   bbox_inches="tight"
)

plt.close()




ENERGY_SPLIT = 40.0

low_energy = (
   truth < ENERGY_SPLIT
)

high_energy = (
   truth >= ENERGY_SPLIT
)




