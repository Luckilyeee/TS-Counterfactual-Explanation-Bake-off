#!/bin/bash

CUDA_VISIBLE_DEVICES=4 nohup python3 Glacier.py --datasets GunPoint BeetleFly BirdChicken Coffee ECG200 FaceFour Lightning2 Plane Trace Beef Car ArrowHead Lightning7 Worms --w-type unconstrained --output Glacier_unconstrained_all0.csv > Glacier_unconstrained0.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python3 Glacier.py --datasets GunPoint BeetleFly BirdChicken Coffee ECG200 FaceFour Lightning2 Plane Trace Beef Car ArrowHead Lightning7 Worms --w-type uniform --output Glacier_uniform_all0.csv > Glacier_uniform0.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python3 Glacier.py --datasets GunPoint BeetleFly BirdChicken Coffee ECG200 FaceFour Lightning2 Plane Trace Beef Car ArrowHead Lightning7 Worms --w-type local --output Glacier_local_all0.csv > Glacier_local0.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python3 Glacier.py --datasets GunPoint BeetleFly BirdChicken Coffee ECG200 FaceFour Lightning2 Plane Trace Beef Car ArrowHead Lightning7 Worms --w-type global --output Glacier_global_all0.csv > Glacier_global0.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python3 Glacier.py --datasets Chinatown CBF TwoLeadECG Computers OSULeaf SwedishLeaf --w-type unconstrained --output Glacier_unconstrained_all1.csv > Glacier_unconstrained1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python3 Glacier.py --datasets Chinatown CBF TwoLeadECG Computers OSULeaf SwedishLeaf --w-type uniform --output Glacier_uniform_all1.csv > Glacier_uniform_all1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python3 Glacier.py --datasets Chinatown CBF TwoLeadECG Computers OSULeaf SwedishLeaf --w-type local --output Glacier_local_all1.csv > Glacier_local_all1.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python3 Glacier.py --datasets Chinatown CBF TwoLeadECG Computers OSULeaf SwedishLeaf --w-type global --output Glacier_global_all1.csv > Glacier_global_all1.log 2>&1 &

