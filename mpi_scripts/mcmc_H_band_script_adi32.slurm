#!/bin/bash
#SBATCH --job-name=32bA
#SBATCH --nodes=3 --ntasks-per-node=16
#SBATCH --time=23:59:00
#SBATCH --partition=medium
#SBATCH --mail-user=johan.mazoyer@obspm.fr

cd /obs/jmazoyer/GitProjects/my_projects/hr4796_mcmc_run_and_read/
mpiexec python diskfit_mcmc.py -p GPI_Hband_Hd32297_ADI.yaml --mpi > /obs/jmazoyer/GPI_Hband_MCMC_ADI32.out

exit 0