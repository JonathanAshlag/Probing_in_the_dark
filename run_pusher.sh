#!/usr/bin/env bash
max_parallel=5
count=0
entropy_type=(latent)
num_layers=(2)
network_hidden_sizes=(512)
clip_coefs=(0.25)
# noise_facors=(0.05)
KNN_ks=(25)
uncertainty_coefs=(0.0)
ent_coefs=(0.005)
uncertainty_annealings=("cosine")
for entropy in "${entropy_type[@]}"; do
    for layers in "${num_layers[@]}"; do
        for uncertainty_annealing in "${uncertainty_annealings[@]}"; do
            for knn_k in "${KNN_ks[@]}"; do
                for ent_coef in "${ent_coefs[@]}"; do
                    for uncertainty_coef in "${uncertainty_coefs[@]}"; do
                        for seed in {6..10}; do
                            python3 pusher_pretrain_unified.py \
                            --track                 "True" \
                            --seed                 "$seed" \
                            --entropy_type         "$entropy" \
                            --num_layers           "$layers" \
                            --ent_coef              "$ent_coef" \
                            --vf_coef             0.0 \
                            --advantage_type       "running_avg" \
                            --knn_k                "$knn_k" \
                            --update_freq          5 \
                            --dm                    "verybad_const" \
                            --save_dir             "runs_pusher_walled_off/${entropy}/32_rollouts_verybad_const_knn${knn_k}/advantage_type_running_avg_ent_coef_${ent_coef}_nokl"&
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