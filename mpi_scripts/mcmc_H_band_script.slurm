#!/bin/bash
#SBATCH --job-name=Maz_mcmc_H
#SBATCH --nodes=2 --ntasks-per-node=16
#SBATCH --time=1
#SBATCH --partition=short
#SBATCH --mail-user=johan.mazoyer@obspm.fr
#SBATCH --mem=1gb

cd /obs/jmazoyer/GitProjects/my_projects/hr4796_mcmc_run_and_read/
mpiexec diskfit_mcmc.py > /obs/jmazoyer/GPI_Hband_MCMC.out


exit 0