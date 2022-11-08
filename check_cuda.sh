#!/bin/bash -l

#SBATCH --job-name="Job1"
#SBATCH --cpus-per-task=2                      # cpu here means threads 
#SBATCH --gpus=1                           # select 1 arbitrary gpu (Alternative you can specify gpu. For example, --gpus==rtx_2080_ti:1)
#SBATCH --ntasks=1                             # task is equivalent to number of processes. ntasks >=1 makes sense only if MPI is involved
#SBATCH --partition=full                       # partion full for gpu task; for cpu task --partition=cpu
#SBATCH --out=out.txt


source ~/miniconda3/etc/profile.d/conda.sh
conda activate cdialog

export KMER=6
export MODEL_PATH=~/models/dnabert_6
export DATA_PATH=~/data/LM_6_2000
export OUTPUT_PATH=~/train/LM_6_2000_3_epochs
export MAX_SEQ_LEN=2048
export BATCH_SIZE=1
export MODEL_NAME=dnalong
export PREPROCESS_THREADS=24

srun python3 check_cuda.py























