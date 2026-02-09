#!/bin/bash
#SBATCH -A mch240091     # Allocation name
#SBATCH --nodes=1       # Total # of nodes 
#SBATCH --ntasks=2   # Total # of MPI tasks
#SBATCH --share
#SBATCH --time=5:00:00  # Total run time limit (hh:mm:ss)
#SBATCH -J Scripts    # Job name
#SBATCH -o log.o%j    # Name of stdout output file
#SBATCH -e log.e%j    # Name of stderr error file
#SBATCH -p standard # Queue (partition) name
#SBATCH --mail-user=lukwidmer@ucsb.edu
#SBATCH --mail-type=none

# Manage processing environment, load compilers and applications.
module purge
module load conda
conda activate hdf5

python3 extend_inicond_domain.py
