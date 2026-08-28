"""
ggl_jk_covariance.py
====================
JK covariance for C_ell^GE using NaMaster with a fixed workspace.
Patch assignment uses kmeans_radec with chunked find_nearest.

Key design:
- Workspace computed ONCE on the full mask, reused for all JK resamples
- Per JK resample: zero signal maps in patch k, keep mask fixed
- Decouple coupled C_ell with the fixed workspace

Outputs
-------
cls_jk_nside{NSIDE}.npy     : {str((i,j)): (N_JK, N_ELL)}
cov_jk_nside{NSIDE}.npy     : {str((i,j)): (N_ELL, N_ELL)}
cov_jk_inv_nside{NSIDE}.npy : {str((i,j)): (N_ELL, N_ELL)}  Hartlap-corrected
pcls_ge_nside{NSIDE}.csv    : E and B mode pseudo-Cls for each valid pair
kmeans_centers_njk{N_JK}.npy: patch centres for reproducibility
"""

from astropy.io import fits
import numpy as np
import healpy as hp
import pymaster as nmt
import gc
from kmeans_radec import KMeans, kmeans_sample

#  Parameters
NSIDE         = 1024
LMIN          = 8
LMAX          = 2048
N_BINS        = 32
N_JK          = 150
FIT_SUBSAMPLE = 100_000   # points used to fit k-means centres
CHUNK_SIZE    = 200_000   # points per find_nearest call

VALID_PAIRS = [(i, j) for i in range(1, 6) for j in range(1, 5)]# if j > i]
print(f"Valid (lens, source) pairs: {VALID_PAIRS}")

#  Binning scheme
sqrt_edges  = np.linspace(np.sqrt(LMIN), np.sqrt(LMAX), N_BINS + 1)
edges       = np.round(sqrt_edges ** 2).astype(int)
edges[-1]   = LMAX + 1
b           = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
ells_binned = b.get_effective_ells()
N_ELL       = len(ells_binned)

#  Load catalogs
print("Loading catalogs...", flush=True)
highdens        = fits.open('data/y3a2_gold2.2.1_redmagic_highdens.fits')[1].data
highlum         = fits.open('data/y3a2_gold2.2.1_redmagic_highlum_highz.fits')[1].data
random_highdens = fits.open('data/y3a2_gold2.2.1_redmagic_highdens_randoms.fits')[1].data
random_highlum  = fits.open('data/y3a2_gold2.2.1_redmagic_highlum_highz_randoms.fits')[1].data

#  Load mask
print("Building survey mask...", flush=True)
hdu_sel     = fits.open('data/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask.fits')
pixel_index = hdu_sel[1].data['HPIX']
mask_value  = hdu_sel[1].data['FRACGOOD']
mask_map    = np.zeros(hp.nside2npix(4096), dtype=np.float64)
mask_map[pixel_index] = mask_value
mask_sel    = hp.ud_grade(mask_map, NSIDE)

#  Shear response
R_shear     = np.array([0.7636, 0.7182, 0.6887, 0.6154])
R_selection = np.array([0.0046, 0.0083, 0.0126, 0.0145])
R           = R_shear + R_selection   # index 0 = source bin 1

#  Build NmtFields
print("Building NmtFields...", flush=True)

def build_density_field(lens_cat, ran_cat, mask, nside, lmax):
    npix   = hp.nside2npix(nside)
    ipix   = hp.ang2pix(nside, lens_cat['RA'], lens_cat['DEC'], lonlat=True)
    ipix_r = hp.ang2pix(nside, ran_cat['RA'],  ran_cat['DEC'],  lonlat=True)
    w_d    = np.asarray(lens_cat['WEIGHT'], dtype=float)
    w_r    = np.ones(len(ipix_r), dtype=float)
    n_data = np.bincount(ipix,   weights=w_d, minlength=npix)
    n_ran  = np.bincount(ipix_r, weights=w_r, minlength=npix)
    good         = (n_ran > 0) & (mask > 0)
    delta        = np.zeros(npix, dtype=float)
    delta[good]  = n_data[good] / n_ran[good] / np.mean(n_data[good] / n_ran[good]) - 1.0
    delta[~good] = -1.0
    return nmt.NmtField(mask, [delta], spin=0, lmax=lmax, lmax_mask=lmax)

density_fields = {}
shear_fields   = {}
shear_maps     = {}  

for i in range(1, 6):
    if i < 4:
        lens_bin = highdens[highdens['BIN_NUMBER'] == i]
        ran_bin  = random_highdens[random_highdens['BIN_NUMBER'] == i]
    else:
        lens_bin = highlum[highlum['BIN_NUMBER'] == i]
        ran_bin  = random_highlum[random_highlum['BIN_NUMBER'] == i]
    density_fields[i] = build_density_field(lens_bin, ran_bin, mask_sel, NSIDE, LMAX)

    if i < 5:
        mask_s = hp.ud_grade(hp.read_map(f'data/DES_shear_maps/mask_shear_z{i}.fits'), NSIDE)
        e1     = hp.ud_grade(hp.read_map(f'data/DES_shear_maps/shear_e1_z{i}.fits'),   NSIDE)
        e2     = hp.ud_grade(hp.read_map(f'data/DES_shear_maps/shear_e2_z{i}.fits'),   NSIDE)
        e1    *= -1
        shear_maps[i]   = (e1, e2, mask_s)
        shear_fields[i] = nmt.NmtField(mask_s, [e1 / R[i-1], e2 / R[i-1]],
                                        spin=2, lmax=LMAX, lmax_mask=LMAX)

print("NmtFields built.", flush=True)

#  Compute workspaces once, used for all JK resamples
print("Computing workspaces (once)...", flush=True)
workspaces = {}
for (i, j) in VALID_PAIRS:
    print(f"  Workspace ({i},{j})", flush=True)
    workspaces[(i,j)] = nmt.NmtWorkspace.from_fields(
        density_fields[i], shear_fields[j], b)
    gc.collect()

#  Full pseudo-Cls 
pcls_ge_dict = {}
for (i, j) in VALID_PAIRS:
    pcls_ge_dict[(i,j)] = workspaces[(i,j)].decouple_cell(
        nmt.compute_coupled_cell(density_fields[i], shear_fields[j]))  # (2, N_ELL)
    print(f"  Pair ({i},{j}): E-mode mean = "
          f"{pcls_ge_dict[(i,j)][0].mean():.3e}", flush=True)

#  Build JK patches with kmeans_radec
all_ra  = np.concatenate([highdens['RA'],  highlum['RA']])
all_dec = np.concatenate([highdens['DEC'], highlum['DEC']])

# Subsample for fitting 
rng   = np.random.default_rng(42)
idx   = rng.choice(len(all_ra), size=min(FIT_SUBSAMPLE, len(all_ra)), replace=False)
X_fit = np.column_stack([all_ra[idx], all_dec[idx]])
print(f"  Fitting k-means on {len(X_fit)} points...", flush=True)

km = kmeans_sample(X_fit, N_JK, maxiter=200, tol=1.0e-5)
if not km.converged:
    print("  WARNING: did not converge, running extra iterations...", flush=True)
    km.run(X_fit, maxiter=200)

print(f"  Converged: {km.converged}", flush=True)
sizes = np.bincount(km.labels)
print(f"  Cluster sizes: min={sizes.min()}, max={sizes.max()}, "
      f"mean={sizes.mean():.0f}", flush=True)

km_centers = km.centers   # (N_JK, 2)
np.save(f"kmeans_centers_njk{N_JK}.npy", km_centers)


def find_nearest_chunked(ra, dec, km_obj, chunk_size=CHUNK_SIZE):
    """Chunked find_nearest to avoid (Npoints, Ncentres) memory spike."""
    n      = len(ra)
    labels = np.empty(n, dtype=int)
    for start in range(0, n, chunk_size):
        end              = min(start + chunk_size, n)
        X_chunk          = np.column_stack([ra[start:end], dec[start:end]])
        labels[start:end] = km_obj.find_nearest(X_chunk)
    return labels


#  Assign patch IDs to all catalogs and shear map pixels
patch_ids_lens = {}
patch_ids_ran  = {}
for i in range(1, 6):
    if i < 4:
        lens_bin = highdens[highdens['BIN_NUMBER'] == i]
        ran_bin  = random_highdens[random_highdens['BIN_NUMBER'] == i]
    else:
        lens_bin = highlum[highlum['BIN_NUMBER'] == i]
        ran_bin  = random_highlum[random_highlum['BIN_NUMBER'] == i]
    print(f"  Lens bin {i}: {len(lens_bin)} galaxies, "
          f"{len(ran_bin)} randoms", flush=True)
    patch_ids_lens[i] = find_nearest_chunked(lens_bin['RA'], lens_bin['DEC'], km)
    patch_ids_ran[i]  = find_nearest_chunked(ran_bin['RA'],  ran_bin['DEC'],  km)

patch_ids_shear = {}
for j in range(1, 5):
    _, _, mask_s = shear_maps[j]
    good_pix     = np.where(mask_s > 0)[0]
    theta, phi   = hp.pix2ang(NSIDE, good_pix)
    ra_pix       = np.rad2deg(phi)
    dec_pix      = 90.0 - np.rad2deg(theta)
    print(f"  Shear bin {j}: {len(good_pix)} unmasked pixels", flush=True)
    labels                  = find_nearest_chunked(ra_pix, dec_pix, km)
    pix_patch_map           = np.full(hp.nside2npix(NSIDE), -1, dtype=int)
    pix_patch_map[good_pix] = labels
    patch_ids_shear[j]      = pix_patch_map

#  Building JK fields: signal zeroed in patch k
def build_density_delta_jk(lens_cat, ran_cat, mask_sel, nside, lmax,
                             pid_data, pid_ran, k):
    """Density field with patch k removed. Full mask kept for workspace compatibility."""
    npix   = hp.nside2npix(nside)
    keep_d = pid_data != k
    keep_r = pid_ran  != k

    ipix_d = hp.ang2pix(nside, lens_cat['RA'][keep_d],
                         lens_cat['DEC'][keep_d], lonlat=True)
    w_d    = np.asarray(lens_cat['WEIGHT'])[keep_d]
    ipix_r = hp.ang2pix(nside, ran_cat['RA'][keep_r],
                         ran_cat['DEC'][keep_r], lonlat=True)
    w_r    = np.ones(keep_r.sum())

    n_data = np.bincount(ipix_d, weights=w_d, minlength=npix)
    n_ran  = np.bincount(ipix_r, weights=w_r, minlength=npix)

    good        = (n_ran > 0) & (mask_sel > 0)

    delta       = np.zeros(npix, dtype=float)
    delta[good]  = n_data[good] / n_ran[good] / np.mean(n_data[good] / n_ran[good]) - 1.0
    # patch k pixels have no randoms -> delta stays 0, consistent with fixed mask
    return nmt.NmtField(mask_sel, [delta], spin=0, lmax=lmax, lmax_mask=lmax)


def build_shear_field_jk(e1_map, e2_map, mask_shear, R_j, lmax, pix_patch_map, k):
    """Shear field with patch k pixels zeroed. Full mask kept for workspace compatibility."""
    patch_k_pix = np.where(pix_patch_map == k)[0]
    e1 = e1_map.copy();  e1[patch_k_pix] = 0.0
    e2 = e2_map.copy();  e2[patch_k_pix] = 0.0
    return nmt.NmtField(mask_shear, [e1 / R_j, e2 / R_j],
                        spin=2, lmax=lmax, lmax_mask=lmax)

#  JK resampling loop: build fields -> compute_coupled_cell -> decouple
needed_lens   = sorted({i for i, j in VALID_PAIRS})
needed_source = sorted({j for i, j in VALID_PAIRS})
cls_jk        = {pair: np.zeros((N_JK, N_ELL)) for pair in VALID_PAIRS}

print("Running JK resampling...", flush=True)
for k in range(N_JK):
    print(f"  JK resample {k+1}/{N_JK}", flush=True)

    density_k = {}
    for i in needed_lens:
        if i < 4:
            lens_bin = highdens[highdens['BIN_NUMBER'] == i]
            ran_bin  = random_highdens[random_highdens['BIN_NUMBER'] == i]
        else:
            lens_bin = highlum[highlum['BIN_NUMBER'] == i]
            ran_bin  = random_highlum[random_highlum['BIN_NUMBER'] == i]
        density_k[i] = build_density_delta_jk(
            lens_bin, ran_bin, mask_sel, NSIDE, LMAX,
            patch_ids_lens[i], patch_ids_ran[i], k)

    shear_k = {}
    for j in needed_source:
        e1, e2, mask_s = shear_maps[j]
        shear_k[j] = build_shear_field_jk(
            e1, e2, mask_s, R[j-1], LMAX, patch_ids_shear[j], k)

    for (i, j) in VALID_PAIRS:
        ccl = nmt.compute_coupled_cell(density_k[i], shear_k[j])
        cls_jk[(i,j)][k, :] = workspaces[(i,j)].decouple_cell(ccl)[0]  # E-mode

    del density_k, shear_k
    gc.collect()

#  Hartlap correction
hartlap = (N_JK - N_ELL - 2) / (N_JK - 1)
print(f"\nHartlap H = {hartlap:.4f}  "
      f"(N_ell/N_JK = {N_ELL/N_JK:.3f}, "
      f"CI inflation ~{100*(1/hartlap - 1):.1f}%)", flush=True)

cov_jk     = {}
cov_jk_inv = {}
for pair, cl in cls_jk.items():
    mean             = cl.mean(axis=0)
    d                = cl - mean
    cov              = (N_JK - 1) / N_JK * (d.T @ d)
    cov_jk[pair]     = cov
    cov_jk_inv[pair] = hartlap * np.linalg.inv(cov)
    print(f"  Pair {pair}: diag range "
          f"[{np.diag(cov).min():.2e}, {np.diag(cov).max():.2e}]")

# Save covariance
def to_str_keys(d):
    return {str(k): v for k, v in d.items()}

np.save(f"cls_jk_full_nside{NSIDE}.npy",     to_str_keys(cls_jk))
np.save(f"cov_jk_full_nside{NSIDE}.npy",     to_str_keys(cov_jk))
np.save(f"cov_jk_inv_full_nside{NSIDE}.npy", to_str_keys(cov_jk_inv))

# Save pseudo-Cls
rows_ge = []
for (i, j) in VALID_PAIRS:
    for s in range(2):
        rows_ge.append([i, j, s] + pcls_ge_dict[(i,j)][s, :].tolist())
np.savetxt(f"pcls_ge_full_nside{NSIDE}.csv", np.array(rows_ge), delimiter=",", fmt="%.6e")

print("Done.", flush=True)
