#!/bin/bash
#SBATCH --job-name=Maz_H_ADI
#SBATCH --nodes=2 --ntasks-per-node=16
#SBATCH --time=23:59:00
#SBATCH --partition=medium
#SBATCH --mail-user=johan.mazoyer@obspm.fr
#SBATCH --mem=1gb

cd /obs/jmazoyer/GitProjects/my_projects/hr4796_mcmc_run_and_read/
mpiexec python diskfit_mcmc.py FakeHr4796brigth_MCMC_ADI.yaml > /obs/jmazoyer/GPI_Hband_MCMC_ADI.out

exit 0