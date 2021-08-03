#!/bin/bash

#$ -l rt_G.large=1
#$ -l h_rt=40:00:00
#$ -N glue-ex
#$ -o glue-ex.log
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4 cudnn/7.0/7.0.5 python/3.6/3.6.5
source glue/bin/activate
export CUDA_VISIBLE_DEVICES="0,1,2,3"

GLUE_DIR="."
#for TASK_NAME in "MNLI" "QNLI" "RTE" "WNLI"
for TASK_NAME in "CoLA" "SST2" "MRPC" "STSB" "MNLI" "QNLI" "RTE"
do
  for MODEL in $(ls extra);
  do
    python run_glue.py \
      --model_name_or_path extra/$MODEL/ \
      --task_name $TASK_NAME \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 8 \
      --learning_rate 2e-5 \
      --num_train_epochs 3 \
      --output_dir log/$MODEL'_'$TASK_NAME/
  done
done

