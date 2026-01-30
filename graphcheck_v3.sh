#!/bin/bash
#SBATCH --nodes=1
#SBATCH --nodelist=n02
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

source /data2/hyewonjeon/.bashrc
source /data2/hyewonjeon/miniconda3/etc/profile.d/conda.sh
conda activate graphcheck

input_fname="train_sampled"
dataset="musique"
python -u graphcheck.py \
    --dataset ${dataset} \
    --input_filename ${input_fname}.json \
    --use_searchr1 \
    --searchr1_max_turns 5 \
    --use_total_search_results \
    --nudge_searchr1 \
    --graphcheck_filename ${input_fname}_v3.json
