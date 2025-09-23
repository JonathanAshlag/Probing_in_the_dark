#!/usr/bin/env bash
max_parallel=5
pretrains=(latent state obs random)
advantage_types=(gae)
learning_rates=(0.0003) # 0.0003 0.0001
clip_coefs=(0.25) #0.2
ent_coefs=(0.002) # 0.002 0.003 
num_envss=(4)
tasks=("escape" "jump")
count=0
for seed in {1..10}; do
  for num_envs in "${num_envss[@]}"; do
    for ent_coef in "${ent_coefs[@]}"; do
      for lr in "${learning_rates[@]}"; do
        for task in "${tasks[@]}"; do
          for pre in "${pretrains[@]}"; do
            if [ "$pre" == "random" ]; then
              init_type="random"
            else
              init_type="pretrained"
            fi
            echo "Running with seed: $seed, num_envs: $num_envs, pretrain: $pre, advantage type: $advantage_type, learning rate: $lr, ent coef: $ent_coef, task: $task"
            python3 ant_finetune_unified.py \
              --seed                 "$seed" \
              --pretrain_type        "$pre" \
              --vf_coef              0.5 \
              --num_steps            1000 \
              --path                 "runs_ant/${pre}/16_rollouts_contextual_knn25_dm_batch_50_dreamer_epochs_10/advantage_type_running_avg_ent_coef_0.002_warmup_40_noexc/PO_ant-v5__${seed}_entropy_type_${pre}_16_rollouts_contextual_knn25_dm_batch_50_dreamer_epochs_10_warmup_40_noexc" \
              --init_type           "$init_type" \
              --advantage_type      "gae" \
              --task                "$task" \
              --num_envs            "$num_envs" \
              --learning_rate       "$lr" \
              --save_dir          "runs_ant/finetuning/${task}/${pre}/PO_Pusher__${seed}_num_envs${num_envs}_contextual_lr_${lr}_ent_coef_${ent_coef}_${task}" \
              --ent_coef           "$ent_coef" &
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