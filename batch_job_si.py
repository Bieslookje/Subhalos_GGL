import numpy as np
from scipy import special
from sashimi_si import *
hm = halo_model()
import os
from joblib import Parallel, delayed

def u_sidm_num(k, rhos, rs, rc, ct, beta=4.0, Nr=100):
    """
    Numerical Fourier transform of the SIDM density profile (mass-normalized).
    Returns shape (nk, nh).
    """
    k    = np.atleast_1d(k).astype(float)
    rhos = np.atleast_1d(rhos).astype(float)
    rs   = np.atleast_1d(rs).astype(float)
    rc   = np.atleast_1d(rc).astype(float)
    ct   = np.atleast_1d(ct).astype(float)

    nk = k.size
    nh = rs.size
    u  = np.empty((nk, nh), dtype=float)

    for j in range(nh):
        rt = ct[j] * rs[j]
        r  = np.logspace(-6, np.log10(rt), Nr)

        x  = r / rs[j]
        xc = rc[j] / rs[j]

        rho = rhos[j] / (
            ((x**beta + xc**beta) ** (1.0 / beta)) * (1.0 + x) ** 2
        )

        M_enc = np.trapezoid(4.0 * np.pi * r**2 * rho, r)

        kr   = np.outer(k, r)
        sinc = np.ones_like(kr)
        mask = kr != 0.0
        sinc[mask] = np.sin(kr[mask]) / kr[mask]

        integrand = 4.0 * np.pi * r[None, :]**2 * rho[None, :] * sinc
        u[:, j]   = np.trapezoid(integrand, r, axis=1) / M_enc

    if u.shape == (1, 1):
        return u[0, 0]
    return u


def _compute_one(M, z, k_grid, sigma0_m, w):
    """Compute subhalo params for a single (M, z) pair."""
    hm = halo_model()
    sh = subhalo_properties(sigma0_m=sigma0_m, w=w)

    ma200, z_acc,rsCDM_acc, rhosCDM_acc, rmaxCDM_acc, VmaxCDM_acc,rsSIDM_acc, rhosSIDM_acc, rcSIDM_acc, rmaxSIDM_acc, VmaxSIDM_acc,m_z0, rsCDM_z0, rhosCDM_z0, rmaxCDM_z0, VmaxCDM_z0,rsSIDM_z0, rhosSIDM_z0, rcSIDM_z0, rmaxSIDM_z0, VmaxSIDM_z0,ctCDM_z0, tt_ratio,weightCDM, weightSIDM,surviveCDM, surviveSIDM \
    = sh.subhalo_properties_calc(M0=M,redshift=z,M0_at_redshift=False)

    mask = weightSIDM > 0
    m    = m_z0[mask]
    rho  = rhosSIDM_z0[mask]
    rs   = rsSIDM_z0[mask]
    rc   = rcSIDM_z0[mask]
    ct   = ctCDM_z0[mask]
    wSIDM = weightSIDM[mask]   

    u = u_sidm_num(k_grid, rho, rs, rc, ct, Nr=100)  # shape (nk, Nsub)

    w1 = m * wSIDM        # shape (Nsub,)
    w2 = m**2 * wSIDM     # shape (Nsub,)

    Mz = hm.Mzi(M, z)

    I_k = (u * w1[None, :]).sum(axis=1) / Mz        # shape (nk,)
    J_k = (u**2 * w2[None, :]).sum(axis=1) / Mz**2  # shape (nk,)
    m_tot = np.sum(w1)

    return m_tot, I_k, J_k


def compute_subhalo_params_k_vect(filename, kmin, kmax, nk,
                                   Mmin, Mmax, nM,
                                   zmin, zmax, nz,
                                   sigma0_m, w,
                                   n_jobs=-1):
    k_grid = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    M_grid = np.logspace(np.log10(Mmin), np.log10(Mmax), nM)
    z_grid = np.linspace(zmin, zmax, nz)

    tasks = [
        (M_id, z_id, M, z)
        for M_id, M in enumerate(M_grid)
        for z_id, z in enumerate(z_grid)
    ]

    n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    print(f"Running {len(tasks)} tasks on {n_workers} workers", flush=True)

    m_mz  = np.empty((nM, nz), dtype=np.float32)
    I_kmz = np.empty((nk, nM, nz), dtype=np.float32)
    J_kmz = np.empty((nk, nM, nz), dtype=np.float32)

    # Chunked to avoid OOM
    chunk_size = n_workers * 4
    n_chunks   = -(-len(tasks) // chunk_size)  # ceiling division

    for chunk_idx, chunk_start in enumerate(range(0, len(tasks), chunk_size)):
        chunk = tasks[chunk_start : chunk_start + chunk_size]
        print(f"  Chunk {chunk_idx + 1}/{n_chunks}", flush=True)

        results = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")(
            delayed(_compute_one)(M, z, k_grid, sigma0_m, w)
            for (M_id, z_id, M, z) in chunk
        )

        for (M_id, z_id, _, _), (m_tot, I_k, J_k) in zip(chunk, results):
            m_mz[M_id, z_id]     = m_tot
            I_kmz[:, M_id, z_id] = I_k
            J_kmz[:, M_id, z_id] = J_k

    np.savez(
        filename,
        k_grid=k_grid, M_grid=M_grid, z_grid=z_grid,
        m_mz=m_mz, I_kmz=I_kmz, J_kmz=J_kmz
    )
    print(f"Saved to {filename}", flush=True)
