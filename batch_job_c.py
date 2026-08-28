import numpy as np
from scipy import special
from sashimi_c import *
from joblib import Parallel, delayed

def u_nfw(k, c, rs):
    k  = np.atleast_1d(k).astype(float)
    rs = np.atleast_1d(rs).astype(float)
    c  = np.atleast_1d(c).astype(float)

    if rs.shape != c.shape:
        raise ValueError("rs and c must have the same shape")

    x = k[:, None] * rs[None, :]
    denom = np.log(1.0 + c) - c / (1.0 + c)
    denom = denom[None, :]

    Si_x,  Ci_x  = special.sici(x)
    Si_xc, Ci_xc = special.sici((1.0 + c)[None, :] * x)

    u = (
        np.cos(x) * (Ci_xc - Ci_x)
        + np.sin(x) * (Si_xc - Si_x)
        - np.sin(c[None, :] * x) / ((1.0 + c)[None, :] * x)
    ) / denom

    if u.shape == (1, 1):
        return u[0, 0]
    return u

def _compute_one(M, z, k_grid):
    """Compute subhalo params for a single (M, z) pair."""
    obs = subhalo_observables(
        M0_per_Msun=M,
        redshift=z,
        M0_at_redshift=False
    )
    print(f"M={M:.2e}, z={z:.2f} → Nsub={len(obs.m0)}", flush=True)
    
    w1 = obs.m0 * obs.weight        # shape (Nsub,)
    w2 = obs.m0**2 * obs.weight     # shape (Nsub,)

    m = np.sum(w1)

    u_k_all = u_nfw(k_grid, obs.ct0, obs.rs0)  # shape (nk, Nsub)

    I_k = (u_k_all * w1[None, :]).sum(axis=1) / M          # shape (nk,)
    J_k = (w2[None, :] * u_k_all**2).sum(axis=1) / M**2   # shape (nk,)

    return m, I_k, J_k

def compute_subhalo_params_k_vect(filename,kmin, kmax, nk,
                                   Mmin, Mmax, nM,
                                   zmin, zmax, nz,
                                   n_jobs=-1):
    """
    Parallelized version of compute_subhalo_params_fast.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers. -1 uses all available CPUs.
        Set to e.g. 16 to match your SLURM --cpus-per-task.
    """

    k_grid = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    M_grid = np.logspace(np.log10(Mmin), np.log10(Mmax), nM)
    z_grid = np.linspace(zmin, zmax, nz)

    # Build flat list of all (M_id, z_id, M, z) tasks
    tasks = [
        (M_id, z_id, M, z)
        for M_id, M in enumerate(M_grid)
        for z_id, z in enumerate(z_grid)
    ]

    print(f"Running {len(tasks)} tasks with n_jobs={n_jobs}", flush=True)

    # Run in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_compute_one)(M, z, k_grid)
        for (M_id, z_id, M, z) in tasks
    )

    # Reassemble into output arrays
    m_mz  = np.empty((nM, nz), dtype=np.float32)
    I_kmz = np.empty((nk, nM, nz), dtype=np.float32)
    J_kmz = np.empty((nk, nM, nz), dtype=np.float32)

    for (M_id, z_id, M, z), (m, I_k, J_k) in zip(tasks, results):
        m_mz[M_id, z_id]     = m
        I_kmz[:, M_id, z_id] = I_k
        J_kmz[:, M_id, z_id] = J_k

    np.savez(
        filename,
        k_grid=k_grid,
        M_grid=M_grid,
        z_grid=z_grid,
        m_mz=m_mz,
        I_kmz=I_kmz,
        J_kmz=J_kmz
    )
