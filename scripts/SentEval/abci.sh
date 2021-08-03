#!/bin/bash

#$ -l rt_G.large=1
#$ -l h_rt=40:00:00
#$ -N senteval-ex
#$ -o senteval-ex.log
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4 cudnn/7.0/7.0.5 python/3.6/3.6.5
source ../../tacl/BuboQA/bert/bin/activate
export CUDA_VISIBLE_DEVICES="0,1,2,3"

for MODEL in $(cat runable_models.txt);
do
  python bert.py $MODEL
done

#for L in 2 4 6 8 10 12
  #do
  #for H in 128 256 512 768
  #do
  #  python bert.py $L','$H
  #done
#done

