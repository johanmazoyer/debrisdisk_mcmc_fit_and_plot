#!/bin/bash
#SBATCH --job-name=32RDIb
#SBATCH --nodes=2 --ntasks-per-node=16
#SBATCH --time=23:59:00
#SBATCH --partition=medium
#SBATCH --mail-user=johan.mazoyer@obspm.fr

cd /obs/jmazoyer/GitProjects/my_projects/hr4796_mcmc_run_and_read/
mpiexec python diskfit_mcmc.py -p FakeHd32297faint_MCMC_RDI_bis.yaml --mpi > /obs/jmazoyer/GPI_Hband_MCMC_RDI_bis32.out

exit 0