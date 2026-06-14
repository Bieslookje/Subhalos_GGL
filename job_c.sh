#!/bin/bash
#SBATCH --job-name=subhalo_pars_c
#SBATCH --output=subhalo_pars_c_%j.out
#SBATCH --error=subhalo_pars_c_%j.err
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=32G

module load 2025
module load SciPy-bundle/2025.06-gfbf-2025a

cd /home/sditvoorst/python_model

python - << 'EOF'
import os

from batch_job_c import compute_subhalo_params_k_vect

compute_subhalo_params_k_vect(
    filename='pars001.npz',
    kmin=1e-4,
    kmax=1e3,
    nk=128,
    Mmin=1e1,
    Mmax=1e18,
    nM=128,
    zmin=0,
    zmax=2,
    nz=128
)
EOF