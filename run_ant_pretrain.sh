#!/usr/bin/env bash
max_parallel=5
count=0
entropy_types=(latent state obs)
num_layers=(2)
network_hidden_sizes=(512)
clip_coefs=(0.25)
KNN_ks=(25)
ent_coefs=(0.002)
warmups=(40)
dm_batch_length=(50)
dreamer_epochs=(10)
dm_types=("contextual") # "base" "verybad_style"
for seed in {1..10}; do
    for entropy_type in "${entropy_types[@]}"; do
        for knn_k in "${KNN_ks[@]}"; do
            for ent_coef in "${ent_coefs[@]}"; do
                for warmup in "${warmups[@]}"; do
                    for dm in "${dm_types[@]}"; do
                        for dreamer_epoch in "${dreamer_epochs[@]}"; do
                            python3 ant_pretrain_unified.py \
                            --track                 "True" \
                            --seed                 "$seed" \
                            --entropy_type           "$entropy_type" \
                            --ent_coef              "$ent_coef" \
                            --vf_coef             0.0 \
                            --advantage_type       "running_avg" \
                            --knn_k                "$knn_k" \
                            --update_freq          5 \
                            --dm                    "${dm}" \
                            --dm_batch_length      50 \
                            --dreamer_epochs       "$dreamer_epoch" \
                            --warm_up               "$warmup" \
                            --save_dir             "runs_ant/${entropy_type}/16_rollouts_${dm}_knn${knn_k}_dm_batch_${dm_batch}_dreamer_epochs_${dreamer_epoch}/advantage_type_running_avg_ent_coef_${ent_coef}_warmup_${warmup}_noexc_nokl"&
                            ((count++))
                            if (( count % max_parallel == 0 )); then
                                wait
                            fi
                        done
                    done
                done
            done
        done
    done
done
wait