#!/bin/bash
#SBATCH --job-name=Maz_mcmc_H
#SBATCH --nodes=2 --ntasks-per-node=16
#SBATCH --time=1
#SBATCH --partition=short
#SBATCH --mail-user=johan.mazoyer@obspm.fr
#SBATCH --mem=1gb

## Définir le répertoire scratch et recopier les fichiers nécessaires à l'exécution
SCRATCH=/scratch/$USER/run_${SLURM_JOBID}
cd ../../scratch/$USER/
srun --ntasks=$SLURM_JOB_NUM_NODES mkdir -p run_${SLURM_JOBID}
cd /scratch/$USER/run.${SLURM_JOBID}
# srun --ntasks=$SLURM_JOB_NUM_NODES cp /obs/$USER/GitProjects/my_projects/hr4796_mcmc_run_and_read/*.py .
# srun --ntasks=$SLURM_JOB_NUM_NODES cp -r /obs/$USER/GitProjects/my_projects/hr4796_mcmc_run_and_read/initialization_files .
# srun --ntasks=$SLURM_JOB_NUM_NODES cp -r /data/$USER/tycho/160318_H_Spec .

# mpiexec python diskfit_mcmc.py GPI_Hband_MCMC.yaml > GPI_Hband_MCMC.out

# srun --ntasks=$SLURM_JOB_NUM_NODES mv ./160318_H_Spec /data/$USER/tycho/160318_H_Spec
# cd ${SLURM_SUBMIT_DIR}
# mv ${SCRATCH}/GPI_Hband_MCMC.out .
##srun --ntasks=$SLURM_JOB_NUM_NODES rm -rf ${SCRATCH}


exit 0