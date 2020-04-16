#!/bin/bash
#SBATCH --job-name=Maz_mcmc_H
#SBATCH --nodes=2 --ntasks-per-node=16
#SBATCH --time=1
#SBATCH --partition=short
#SBATCH --mail-user=johan.mazoyer@obspm.fr
#SBATCH --mem=1gb

cd GitProjects/my_projects/hr4796_mcmc_run_and_read/
mpiexec python diskfit_mcmc.py GPI_Hband_MCMC.yaml > /obs/jmazoyer/GPI_Hband_MCMC.out


exit 0