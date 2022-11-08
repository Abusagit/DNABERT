#!/bin/bash -l

#SBATCH --job-name="Job1"
#SBATCH --cpus-per-task=2                      # cpu here means threads 
#SBATCH --gpus=1                           # select 1 arbitrary gpu (Alternative you can specify gpu. For example, --gpus==rtx_2080_ti:1)
#SBATCH --ntasks=1                             # task is equivalent to number of processes. ntasks >=1 makes sense only if MPI is involved
#SBATCH --partition=full                       # partion full for gpu task; for cpu task --partition=cpu
#SBATCH --out=out.txt


source ~/miniconda3/etc/profile.d/conda.sh
conda activate test

export KMER=6
export MODEL_PATH=~/models/dnabert_6
export DATA_PATH=~/data/LM_6_2000
export OUTPUT_PATH=~/train/LM_6_2000_3_epochs_max_f1_checkpoint_2
export MAX_SEQ_LEN=2048
export BATCH_SIZE=1
export MODEL_NAME=dnalong
export PREPROCESS_THREADS=24
export GRAD_ACCUM_STEPS=4

cd examples
srun python3 run_finetune.py \
                    --model_type $MODEL_NAME \
                    --tokenizer_name=dna$KMER \
                    --model_name_or_path $MODEL_PATH \
                    --task_name dnaprom \
                    --do_train \
                    --do_eval \
                    --data_dir $DATA_PATH \
                    --max_seq_length $MAX_SEQ_LEN \
                    --per_gpu_eval_batch_size=$BATCH_SIZE \
                    --per_gpu_train_batch_size=$BATCH_SIZE \
                    --learning_rate 1e-4 \
                    --num_train_epochs 3.0 \
                    --output_dir $OUTPUT_PATH \
                    --evaluate_during_training \
                    --logging_steps 100 \
                    --save_steps 4000 \
                    --warmup_percent 0.1 \
                    --hidden_dropout_prob 0.1 \
                    --overwrite_output \
                    --weight_decay 0.01 \
                    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
                    --n_process $PREPROCESS_THREADS\
