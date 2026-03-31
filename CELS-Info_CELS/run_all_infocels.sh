#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/Log_infocels"

gpu_count=1  # Set this to the number of GPUs you have
gpu_id=0     # Start with GPU 0
mkdir -p "${LOG_DIR}"
while read ds size; do
    # Skip empty lines or invalid data
    if [ -z "$ds" ] || [ -z "$size" ]; then
        continue
    fi

    CUDA_VISIBLE_DEVICES=${gpu_id} nohup python3 main_info.py --pname InfoCELS_${ds} \
        --jobs_per_task 10 \
        --samples_per_task ${size} \
        --dataset ${ds} > "${LOG_DIR}/${ds}.log" 2>&1 &

    echo "Starting job for dataset: ${ds} with ${size} instances on GPU ${gpu_id}"
    gpu_id=$(( (gpu_id + 1) % gpu_count ))
done < <(python3 get_test_sizes.py)
