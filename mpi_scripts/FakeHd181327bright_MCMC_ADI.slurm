#!/bin/bash
#SBATCH --job-name=181A
#SBATCH --nodes=3 --ntasks-per-node=16
#SBATCH --time=23:59:00
#SBATCH --partition=medium
#SBATCH --mail-user=johan.mazoyer@obspm.fr

cd /obs/jmazoyer/GitProjects/my_projects/hr4796_mcmc_run_and_read/
mpiexec python diskfit_mcmc.py -p FakeHd181327bright_fixspf_MCMC_ADI.yaml --mpi > /obs/jmazoyer/FakeHd181327bright_fixspf_MCMC_ADI.out

exit 0