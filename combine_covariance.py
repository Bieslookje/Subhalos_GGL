"""
ggl_combine_covariance.py
=========================
Combines Gaussian (data-driven) and Jackknife covariance matrices:
  - Correlation structure from Gaussian covariance
  - Diagonal variances from Jackknife

Also produces diagnostic plots for each pair and the joint matrix.

Outputs
-------
cov_comb_nside{NSIDE}.npy   : {str((i,j)): (N_ELL, N_ELL)}
plots/cov_pair_{i}_{j}.png  : per-pair diagnostic
plots/cov_joint.png         : joint block-diagonal matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

#  Parameters
NSIDE       = 1024
VALID_PAIRS = [(i, j) for i in range(1, 6) for j in range(1, 5)]# if j > i]
os.makedirs("plots", exist_ok=True)

#  Load covariances
print("Loading covariances...", flush=True)

raw_gauss = np.load(f"cov_gaussian_full_nside{NSIDE}_v3.npy", allow_pickle=True).item()
raw_jk    = np.load(f"cov_jk_full_nside{NSIDE}.npy",       allow_pickle=True).item()

cov_gaussian_dict = {eval(k): v for k, v in raw_gauss.items()}
cov_jk_dict       = {eval(k): v for k, v in raw_jk.items()}

ells_binned = np.loadtxt("ells_binned.csv", delimiter=",")
N_ELL       = len(ells_binned)

#  Combine
print("Combining covariances...", flush=True)

cov_comb_dict = {}

for (i, j) in VALID_PAIRS:
    cov_gauss = cov_gaussian_dict[(i, j)]
    cov_jk    = cov_jk_dict[(i, j)]

    sigma_gauss = np.sqrt(np.diag(cov_gauss))
    sigma_jk    = np.sqrt(np.diag(cov_jk))

    corr_gauss  = cov_gauss / np.outer(sigma_gauss, sigma_gauss)
    cov_comb    = corr_gauss * np.outer(sigma_jk, sigma_jk)

    cov_comb_dict[(i, j)] = cov_comb
    print(f"  Pair ({i},{j}): diag range "
          f"[{np.diag(cov_comb).min():.2e}, {np.diag(cov_comb).max():.2e}]")

#  Save
np.save(f"cov_comb_full_nside{NSIDE}.npy",
        {str(k): v for k, v in cov_comb_dict.items()})
print("Saved cov_comb.", flush=True)

def to_corr(cov):
    """Convert covariance to correlation matrix."""
    sigma = np.sqrt(np.diag(cov))
    return cov / np.outer(sigma, sigma)

def plot_matrix(ax, mat, title, vmin=None, vmax=None, cmap="RdBu_r"):
    im = ax.imshow(mat, origin="upper", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im

#  Per-pair diagnostic plots
print("Plotting per-pair diagnostics...", flush=True)

for (i, j) in VALID_PAIRS:
    cov_gauss = cov_gaussian_dict[(i, j)]
    cov_jk    = cov_jk_dict[(i, j)]
    cov_comb  = cov_comb_dict[(i, j)]

    sigma_gauss = np.sqrt(np.diag(cov_gauss))
    sigma_jk    = np.sqrt(np.diag(cov_jk))
    sigma_comb  = np.sqrt(np.diag(cov_comb))

    corr_gauss = to_corr(cov_gauss)
    corr_jk    = to_corr(cov_jk)
    corr_comb  = to_corr(cov_comb)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Covariance diagnostics — lens {i}, source {j}", fontsize=12)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.4)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])

    plot_matrix(ax0, corr_gauss, "Corr (Gaussian)",  vmin=-1, vmax=1)
    plot_matrix(ax1, corr_jk,    "Corr (JK)",         vmin=-1, vmax=1)
    plot_matrix(ax2, corr_comb,  "Corr (Combined)",   vmin=-1, vmax=1)

    diff_corr = corr_comb - corr_gauss
    plot_matrix(ax3, diff_corr, "Corr: Comb - Gauss",
                vmin=-0.3, vmax=0.3, cmap="coolwarm")

    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2:])

    ax4.plot(ells_binned, sigma_gauss, label="Gaussian", lw=1.5, ls="--")
    ax4.plot(ells_binned, sigma_jk,    label="JK",        lw=1.5, ls=":")
    ax4.plot(ells_binned, sigma_comb,  label="Combined",  lw=2.0)
    ax4.set_xlabel(r"$\ell$")
    ax4.set_ylabel(r"$\sigma(C_\ell^{gE})$")
    ax4.set_title("Diagonal standard deviations")
    ax4.legend(fontsize=8)
    ax4.set_yscale("log")

    ratio_jk_gauss = sigma_jk / sigma_gauss
    ax5.axhline(1.0, color="k", lw=0.8, ls="--")
    ax5.plot(ells_binned, ratio_jk_gauss, lw=1.5, color="C1",
             label=r"$\sigma_{\rm JK}/\sigma_{\rm Gauss}$")
    ax5.set_xlabel(r"$\ell$")
    ax5.set_ylabel("Ratio")
    ax5.set_title(r"$\sigma_{\rm JK} / \sigma_{\rm Gauss}$")
    ax5.legend(fontsize=8)

    plt.savefig(f"plots/cov_pair_{i}_{j}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plots/cov_pair_{i}_{j}.png")

#  Joint block-diagonal matrix plot

print("Plotting joint covariance matrices...", flush=True)

def build_joint_corr(cov_dict, pairs):
    """Stack per-pair covariances into a joint block-diagonal correlation matrix."""
    blocks = [to_corr(cov_dict[p]) for p in pairs]
    sizes  = [b.shape[0] for b in blocks]
    N      = sum(sizes)
    joint  = np.zeros((N, N))
    idx = 0
    for b in blocks:
        n = b.shape[0]
        joint[idx:idx+n, idx:idx+n] = b
        idx += n
    return joint, sizes

def build_joint_sigma(cov_dict, pairs):
    return np.concatenate([np.sqrt(np.diag(cov_dict[p])) for p in VALID_PAIRS])

joint_gauss, sizes = build_joint_corr(cov_gaussian_dict, VALID_PAIRS)
joint_jk,    _     = build_joint_corr(cov_jk_dict,       VALID_PAIRS)
joint_comb,  _     = build_joint_corr(cov_comb_dict,     VALID_PAIRS)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Joint correlation matrices (block diagonal)", fontsize=12)

for ax, mat, title in zip(axes,
                           [joint_gauss, joint_jk, joint_comb],
                           ["Gaussian", "Jackknife", "Combined"]):
    im = ax.imshow(mat, origin="upper", aspect="auto",
                   vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    idx = 0
    for s in sizes[:-1]:
        idx += s
        ax.axhline(idx - 0.5, color="k", lw=0.5)
        ax.axvline(idx - 0.5, color="k", lw=0.5)

    tick_pos = []
    idx = 0
    for s, p in zip(sizes, VALID_PAIRS):
        tick_pos.append(idx + s // 2)
        idx += s
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"({p[0]},{p[1]})" for p in VALID_PAIRS], fontsize=7)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([f"({p[0]},{p[1]})" for p in VALID_PAIRS], fontsize=7)

plt.savefig("plots/cov_joint.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/cov_joint.png")

#  Joint sigma comparison
sigma_gauss_all = build_joint_sigma(cov_gaussian_dict, VALID_PAIRS)
sigma_jk_all    = build_joint_sigma(cov_jk_dict,       VALID_PAIRS)
sigma_comb_all  = build_joint_sigma(cov_comb_dict,     VALID_PAIRS)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle("Joint data vector: diagonal standard deviations", fontsize=12)

x = np.arange(len(sigma_gauss_all))
ax1.plot(x, sigma_gauss_all, lw=1.2, ls="--", label="Gaussian")
ax1.plot(x, sigma_jk_all,    lw=1.2, ls=":",  label="JK")
ax1.plot(x, sigma_comb_all,  lw=1.8,           label="Combined")
ax1.set_ylabel(r"$\sigma(C_\ell^{gE})$")
ax1.set_yscale("log")
ax1.legend(fontsize=9)

idx = 0
for s, p in zip(sizes, VALID_PAIRS):
    ax1.axvline(idx - 0.5, color="k", lw=0.5, ls="--", alpha=0.4)
    ax1.text(idx + s/2, ax1.get_ylim()[1], f"({p[0]},{p[1]})",
             ha="center", va="bottom", fontsize=7)
    idx += s

ax2.axhline(1.0, color="k", lw=0.8, ls="--")
ax2.plot(x, sigma_jk_all / sigma_gauss_all, lw=1.5,
         label=r"$\sigma_{\rm JK}/\sigma_{\rm Gauss}$")
ax2.set_ylabel("Ratio")
ax2.set_xlabel("Data vector index")
ax2.legend(fontsize=9)

idx = 0
for s in sizes[:-1]:
    ax2.axvline(idx + s - 0.5, color="k", lw=0.5, ls="--", alpha=0.4)
    idx += s

plt.savefig("plots/cov_sigma_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plots/cov_sigma_comparison.png")

print("\nDone.", flush=True)
