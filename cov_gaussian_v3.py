"""
ggl_gaussian_covariance_data_driven.py
=======================================
Computes the Gaussian (NaMaster) covariance of C_ell^GE for
DES Y3 RedMaGiC x MetaCalibration.

Signal Cls are estimated directly from the data:
  - Coupled pseudo-Cls are measured from the fields
  - Decoupled, then interpolated back to unbinned ell for input to
    gaussian_covariance (which requires arrays of length lmax+1)

Number densities and shape noise read from the official DES Y3 2pt file:
  2pt_NG_final_2ptunblind_02_24_21_wnz_redmagic_covupdate.fits

Outputs
-------
cov_gaussian_nside{NSIDE}.npy : {str((i,j)): (N_ELL, N_ELL)}
"""

from astropy.io import fits
import numpy as np
import healpy as hp
import pymaster as nmt
from scipy.interpolate import interp1d
import gc

# ------------------------------------------------------------------ #
#  Parameters
# ------------------------------------------------------------------ #
NSIDE  = 1024
LMIN   = 8
LMAX   = 2048
N_BINS = 32

VALID_PAIRS = [(i, j) for i in range(1, 6) for j in range(1, 5)]# if j > i]
print(f"Valid (lens, source) pairs: {VALID_PAIRS}")

# ------------------------------------------------------------------ #
#  Binning
# ------------------------------------------------------------------ #
sqrt_edges  = np.linspace(np.sqrt(LMIN), np.sqrt(LMAX), N_BINS + 1)
edges       = np.round(sqrt_edges ** 2).astype(int)
edges[-1]   = LMAX + 1
b           = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
ells_binned = b.get_effective_ells()
N_ELL       = len(ells_binned)
print(f"N_ELL = {N_ELL}")

# ------------------------------------------------------------------ #
#  Load catalogs
# ------------------------------------------------------------------ #
print("Loading catalogs...", flush=True)
highdens        = fits.open('data/y3a2_gold2.2.1_redmagic_highdens.fits')[1].data
highlum         = fits.open('data/y3a2_gold2.2.1_redmagic_highlum_highz.fits')[1].data
random_highdens = fits.open('data/y3a2_gold2.2.1_redmagic_highdens_randoms.fits')[1].data
random_highlum  = fits.open('data/y3a2_gold2.2.1_redmagic_highlum_highz_randoms.fits')[1].data

# ------------------------------------------------------------------ #
#  Survey mask
# ------------------------------------------------------------------ #
print("Building survey mask...", flush=True)
hdu_sel     = fits.open('data/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask.fits')
pixel_index = hdu_sel[1].data['HPIX']
mask_value  = hdu_sel[1].data['FRACGOOD']
mask_map    = np.zeros(hp.nside2npix(4096), dtype=np.float64)
mask_map[pixel_index] = mask_value
mask_sel    = hp.ud_grade(mask_map, NSIDE)

# ------------------------------------------------------------------ #
#  Shear response
# ------------------------------------------------------------------ #
R_shear     = np.array([0.7636, 0.7182, 0.6887, 0.6154])
R_selection = np.array([0.0046, 0.0083, 0.0126, 0.0145])
R           = R_shear + R_selection

# ------------------------------------------------------------------ #
#  Number densities and shape noise from the DES Y3 2pt file
# ------------------------------------------------------------------ #
print("Reading n_gal and sigma_e from DES Y3 2pt file...", flush=True)

arcmin2_per_sr = (180.0 * 60.0 / np.pi) ** 2

hdudes        = fits.open('data/2pt_NG_final_2ptunblind_02_24_21_wnz_redmagic_covupdate.fits')
header_source = hdudes['nz_source'].header
header_lens   = hdudes['nz_lens'].header

ngal_source_dict = {}
sige_source_dict = {}
for j in range(1, 5):
    ngal_arcmin2         = header_source[f'NGAL_{j}']
    ngal_source_dict[j]  = ngal_arcmin2 * arcmin2_per_sr
    sige_source_dict[j]  = header_source[f'SIG_E_{j}']
    print(f"  Source bin {j}: n_gal={ngal_arcmin2:.4f} arcmin^-2 "
          f"= {ngal_source_dict[j]:.3e} sr^-1, "
          f"sigma_e={sige_source_dict[j]:.4f}")

ngal_lens_dict = {}
for i in range(1, 6):
    ngal_arcmin2 = header_lens[f'NGAL_{i}']
    ngal_lens_dict[i] = ngal_arcmin2 * arcmin2_per_sr
    print(f" Lens bin {i}: n_gal={ngal_arcmin2:.4f} arcmin^-2 = "
          f"n_gal={ngal_lens_dict[i]:.3e} sr^-1 "
          f"({ngal_lens_dict[i]/arcmin2_per_sr:.4f} arcmin^-2)")

hdudes.close()

# ------------------------------------------------------------------ #
#  Build full NmtFields
# ------------------------------------------------------------------ #
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
        shear_fields[i] = nmt.NmtField(mask_s, [e1 / R[i-1], e2 / R[i-1]],
                                        spin=2, lmax=LMAX, lmax_mask=LMAX)

print("NmtFields built.", flush=True)

# ------------------------------------------------------------------ #
#  Helper: decouple and interpolate back to unbinned ell
#  gaussian_covariance requires coupled Cls of length lmax+1,
#  but we want to use the decoupled (signal-level) estimates.
#  Strategy: decouple -> interpolate onto full ell grid -> use as
#  smooth signal input (noise added separately as flat spectrum).
# ------------------------------------------------------------------ #
ells_full = np.arange(LMAX + 1)

def decouple_and_interpolate(ccl, workspace, ells_binned, lmax,
                              fill_low=None, fill_high=None):
    """
    Decouple a coupled Cl, then interpolate back to a full unbinned
    ell array of length lmax+1.

    Parameters
    ----------
    ccl        : (n_cls, lmax+1) coupled pseudo-Cl array
    workspace  : NmtWorkspace used to decouple
    ells_binned: effective ell values of the bins
    lmax       : maximum ell
    fill_low   : fill value below ells_binned[0]  (default: first bin value)
    fill_high  : fill value above ells_binned[-1] (default: last bin value)

    Returns
    -------
    list of interpolated full-ell arrays, one per Cl component
    """
    decoupled = workspace.decouple_cell(ccl)   # (n_cls, N_ELL)
    out = []
    for cl_bin in decoupled:
        fv_low  = cl_bin[0]  if fill_low  is None else fill_low
        fv_high = cl_bin[-1] if fill_high is None else fill_high
        f = interp1d(ells_binned, cl_bin, kind='linear',
                     bounds_error=False,
                     fill_value=(fv_low, fv_high))
        out.append(f(ells_full))
    return out   # list of (lmax+1,) arrays

# ------------------------------------------------------------------ #
#  Gaussian covariance
# ------------------------------------------------------------------ #
print("Computing Gaussian covariance...", flush=True)

cov_gaussian = {}

for (i, j) in VALID_PAIRS:
    print(f"  Pair ({i},{j})", flush=True)

    # --- Workspaces ------------------------------------------------ #
    w_ge = nmt.NmtWorkspace.from_fields(density_fields[i], shear_fields[j],  b)
    w_gg = nmt.NmtWorkspace.from_fields(density_fields[i], density_fields[i], b)
    w_ee = nmt.NmtWorkspace.from_fields(shear_fields[j],   shear_fields[j],   b)

    # --- Coupled pseudo-Cls from data ------------------------------ #
    ccl_gg = nmt.compute_coupled_cell(density_fields[i], density_fields[i])  # (1, lmax+1)
    ccl_ge = nmt.compute_coupled_cell(density_fields[i], shear_fields[j])    # (2, lmax+1)
    ccl_ee = nmt.compute_coupled_cell(shear_fields[j],   shear_fields[j])    # (4, lmax+1)

    # --- Decouple and interpolate to full ell grid ----------------- #
    # gg: spin-0 x spin-0 -> 1 Cl
    cl_gg_full, = decouple_and_interpolate(ccl_gg, w_gg, ells_binned, LMAX)

    # ge: spin-0 x spin-2 -> [GE, GB]
    cl_ge_full, cl_gb_full = decouple_and_interpolate(ccl_ge, w_ge, ells_binned, LMAX)

    # ee: spin-2 x spin-2 -> [EE, EB, BE, BB]
    cl_ee_full, cl_eb_full, cl_be_full, cl_bb_full = \
        decouple_and_interpolate(ccl_ee, w_ee, ells_binned, LMAX)

    # --- Noise power spectra (flat, added to auto-spectra) --------- #
    nl_gg = 1.0 / ngal_lens_dict[i]
    nl_ee = sige_source_dict[j]**2 / ngal_source_dict[j]
    print(f"    nl_gg={nl_gg:.3e}  nl_ee={nl_ee:.3e}  "
          f"mean(cl_gg)={cl_gg_full.mean():.3e}  "
          f"mean(cl_ee)={cl_ee_full.mean():.3e}", flush=True)

    # Signal + noise
    cl_tt = cl_gg_full + nl_gg
    cl_te = cl_ge_full
    cl_tb = cl_gb_full
    cl_ee = cl_ee_full + nl_ee
    cl_eb = cl_eb_full
    cl_be = cl_be_full
    cl_bb = cl_bb_full + nl_ee

    # --- Covariance workspace -------------------------------------- #
    cw = nmt.NmtCovarianceWorkspace.from_fields(
        density_fields[i], shear_fields[j],
        density_fields[i], shear_fields[j])

    # --- Gaussian covariance — spins (0, 2, 0, 2) ----------------- #
    cov_g = nmt.gaussian_covariance(
        cw,
        0, 2, 0, 2,
        [cl_tt],
        [cl_te, cl_tb],
        [cl_te, cl_tb],
        [cl_ee, cl_eb, cl_be, cl_bb],
        w_ge, w_ge
    ).reshape(N_ELL, 2, N_ELL, 2)

    # E-mode block only
    cov_gaussian[(i, j)] = cov_g[:, 0, :, 0]

    print(f"    diag range: [{np.diag(cov_gaussian[(i,j)]).min():.2e}, "
          f"{np.diag(cov_gaussian[(i,j)]).max():.2e}]", flush=True)

    del w_ge, w_gg, w_ee, cw
    del ccl_gg, ccl_ge, ccl_ee
    gc.collect()

# ------------------------------------------------------------------ #
#  Save
# ------------------------------------------------------------------ #
print("Saving...", flush=True)

def to_str_keys(d):
    return {str(k): v for k, v in d.items()}

np.save(f"cov_gaussian_full_nside{NSIDE}_v3.npy", to_str_keys(cov_gaussian))

print(f"\n{'Pair':<10} {'diag_min':>12} {'diag_max':>12} {'cond':>12}")
print("-" * 50)
for pair, cov in cov_gaussian.items():
    d = np.diag(cov)
    print(f"{str(pair):<10} {d.min():>12.3e} {d.max():>12.3e} "
          f"{np.linalg.cond(cov):>12.1e}")

print("Done.", flush=True)
