#!/bin/bash

gpu_count=1 # Set this to the number of GPUs you have
gpu_id=0

while read ds size; do
    CUDA_VISIBLE_DEVICES=${gpu_id} nohup python3 main.py --pname CELS_${ds} \
        --jobs_per_task 10 \
        --samples_per_task ${size} \
        --dataset ${ds} > ${ds}.log 2>&1 &
    gpu_id=$(( (gpu_id + 1) % gpu_count ))
done < <(python3 get_test_sizes_cels.py)
