#!/bin/bash

# Run CIFAR10 experiment on ganomaly

declare -a arr=("mel" "nv" "bcc" "ak" "bkl" "df" "vasc" "scc")
for m in {0..2}
do
    echo "Manual Seed: $m"
    for i in "${arr[@]}";
    do
        echo "Running CIFAR. Anomaly Class: $i "
        python train.py --dataset cifar10 --isize 32 --niter 15 --anomaly_class $i --manualseed $m
    done
done
exit 0
