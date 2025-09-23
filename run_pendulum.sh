#!/usr/bin/env bash
max_parallel=4
count=0
entropy_type=(latent) #state obs
num_layers=(2)
network_hidden_sizes=(512)
clip_coefs=(0.25)
# noise_facors=(0.05)
KNN_ks=(5)
ent_coefs=(0.001)
uncertainty_coefs=(0.0 0.3)
for seed in {1..10}; do
    for layers in "${num_layers[@]}"; do
        for knn_k in "${KNN_ks[@]}"; do
            for ent_coef in "${ent_coefs[@]}"; do
                for entropy in "${entropy_type[@]}"; do
                    for uncertainty_coef in "${uncertainty_coefs[@]}"; do
                        # Skip if uncertainty_coef > 0 and entropy is not latent
                        if (( $(echo "$uncertainty_coef > 0" | bc -l) )) && [ "$entropy" != "latent" ]; then
                            continue
                        fi
                        python3 pendulum_pretrain_unified.py \
                        --track                 "True" \
                        --seed                 "$seed" \
                        --entropy_type         "$entropy" \
                        --num_layers           "$layers" \
                        --uncertainty_coef       "$uncertainty_coef" \
                        --ent_coef              "$ent_coef" \
                        --vf_coef             0.0 \
                        --advantage_type       "running_avg" \
                        --knn_k                "$knn_k" \
                        --update_freq          5 \
                        --dm                    "verybadnew" \
                        --save_dir             "runs_pendulum/${entropy}/16_rollouts_verybadnew_knn${knn_k}/advantage_type_running_avg_ent_coef_${ent_coef}_uncertainty_coef_${uncertainty_coef}"&
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
wait