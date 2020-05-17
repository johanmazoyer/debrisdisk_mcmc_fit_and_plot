#!/bin/bash
#SBATCH --job-name=18bRt
#SBATCH --nodes=2 --ntasks-per-node=16
#SBATCH --time=23:59:00
#SBATCH --partition=medium
#SBATCH --mail-user=johan.mazoyer@obspm.fr

cd /obs/jmazoyer/GitProjects/my_projects/hr4796_mcmc_run_and_read/
mpiexec python diskfit_mcmc.py -p FakeHd181327bright_MCMC_RDI_ter.yaml --mpi > /obs/jmazoyer/GPI_Hband_MCMC_RDI18bt.out

exit 0