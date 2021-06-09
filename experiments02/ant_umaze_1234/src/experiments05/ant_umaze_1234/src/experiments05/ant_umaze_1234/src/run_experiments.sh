#!/usr/bin/env bash

for env_name in $1; do
    echo "=> Running environment ${env_name}"
    #for random_seed in 1234 2314 2345 1235; do
    for random_seed in 1234; do
        python main.py -c configs/algos/slbo_bm_200k.yml configs/env_tingwu/${env_name}.yml \
	    -s pc.bonus_scale=0.5 log_dir=./experiments05/${env_name}_${random_seed} seed=${random_seed}
    done
done
