#!/bin/bash
#SBATCH --job-name=32bAb
#SBATCH --nodes=2 --ntasks-per-node=16
#SBATCH --time=23:59:00
#SBATCH --partition=medium
#SBATCH --mail-user=johan.mazoyer@obspm.fr

cd /obs/jmazoyer/GitProjects/my_projects/hr4796_mcmc_run_and_read/
mpiexec python diskfit_mcmc.py -p FakeHd32297bright_MCMC_ADI_bis.yaml --mpi > /obs/jmazoyer/GPI_Hband_MCMC_ADI32bb.out

exit 0