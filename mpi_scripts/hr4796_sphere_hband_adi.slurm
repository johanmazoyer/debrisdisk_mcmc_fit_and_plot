#!/bin/bash
#SBATCH --job-name=hr47mcmc
#SBATCH --nodes=2 --ntasks-per-node=16
#SBATCH --time=7199
#SBATCH --partition=long
#SBATCH --mail-user=johan.mazoyer@obspm.fr

cd /obs/jmazoyer/GitProjects/my_projects/debrisdisk_mcmc_fit_and_plot/
mpiexec python diskfit_mcmc.py -p SPHERE_Hband_MCMC_ADI.yaml --mpi > /obs/jmazoyer/SPHERE_Hband_MCMC_anadisk_fixed.out

exit 0