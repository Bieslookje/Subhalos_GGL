"""

Outputs
-------
pcls_ge_nside{NSIDE}.csv    : E and B mode pseudo-Cls for each valid pair
"""

# ------------------------------------------------------------------ #
#  Imports
# ------------------------------------------------------------------ #
from astropy.io import fits
import numpy as np
import healpy as hp
import pymaster as nmt

# ------------------------------------------------------------------ #
#  Parameters
# ------------------------------------------------------------------ #
NSIDE         = 1024
LMIN          = 8
LMAX          = 2048
N_BINS        = 32

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
R           = R_shear + R_selection   # index 0 = source bin 1

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
#  Compute workspaces 
# ------------------------------------------------------------------ #
print("Computing workspaces (once)...", flush=True)
workspaces = {}
for (i, j) in VALID_PAIRS:
    print(f"  Workspace ({i},{j})", flush=True)
    workspaces[(i,j)] = nmt.NmtWorkspace.from_fields(
        density_fields[i], shear_fields[j], b)

# ------------------------------------------------------------------ #
#  Measure full pseudo-Cls using the fixed workspaces
# ------------------------------------------------------------------ #
print("Measuring full pseudo-Cls...", flush=True)
N_PAIRS = len(VALID_PAIRS)
pcls_ge = np.empty((N_PAIRS, 2, N_ELL))

for idx, (i, j) in enumerate(VALID_PAIRS):
    pcls_ge[idx] = workspaces[(i,j)].decouple_cell(
        nmt.compute_coupled_cell(density_fields[i], shear_fields[j]))  # (2, N_ELL)
    print(f"  Pair ({i},{j}): E-mode mean = "
          f"{pcls_ge[idx, 0].mean():.3e}", flush=True)

rows = []
for idx, (i, j) in enumerate(VALID_PAIRS):
    for s in range(2):
        rows.append([i, j, s] + pcls_ge[idx, s, :].tolist())
np.savetxt(f"pcls_ge_full_nside{NSIDE}.csv", np.array(rows), delimiter=",", fmt="%.6e")
