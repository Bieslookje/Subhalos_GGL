# Subhalos_GGL
This page contains all the code that I developed for my thesis project on weak gravitational lensing by dark matter subhalos (https://scripties.uba.uva.nl/search?id=record_58493).

The main code halo_model_WL includes all the necessary functions to: (1) compute the matter power spectrum based a halo model formalism with or without the inclusion of subhalos and 'tweak' parameters, and (2) calculate the angular galaxy-galaxy lensing power spectrum for given tomographic lens and source samples. 

SASHIMI-C (for CDM subhalos) is applied in batch_job_c.py and SASHIMI-SIDM in batch_job_si.py, which are optimized to calculate the necessary subhalo properties efficiently for the input redshifts and halo masses.

The power spectrum estimates are calculated from galaxy position and ellipticity data using the NaMaster method in pcls.py.

The Gaussian correlation matrix is computed in cov_gaussian.py and the Jackknife covariance is computed in cov_jk.py.

Finally, the tweaks_MCMC notebooks contain the full MCMC tweak parameter estimation pipeline.

The most essential applications of the code are shown in tutorial.ipynb, including the comparison of the matter power spectra predicted by different models and likelihood calculations using DES data.
