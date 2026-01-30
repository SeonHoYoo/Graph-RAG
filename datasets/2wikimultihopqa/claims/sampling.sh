#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=20000MB
#SBATCH --cpus-per-task=2
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

source /data2/hyewonjeon/.bashrc
source /data2/hyewonjeon/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck2

python sampling.py