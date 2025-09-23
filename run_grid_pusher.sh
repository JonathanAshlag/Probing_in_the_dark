#!/usr/bin/env bash
max_parallel=5
pretrains=(state obs latent random)
advantage_types=(gae)
learning_rates=(0.0003) # 0.0003 0.0001
clip_coefs=(0.3) #0.2
ent_coefs=(0.003) # 0.002 0.003 0.005
num_envss=(32)
tasks=("top_left" "pull_to_self" "hard_left")
count=0
for num_envs in "${num_envss[@]}"; do
  for ent_coef in "${ent_coefs[@]}"; do
    for lr in "${learning_rates[@]}"; do
      for task in "${tasks[@]}"; do
        for pre in "${pretrains[@]}"; do
          if [ "$pre" == "random" ]; then
            init_type="random"
            num_envs=8
          else
            init_type="pretrained"
          fi
          for seed in {1..10}; do
            echo "Running with seed: $seed, num_envs: $num_envs, pretrain: $pre, advantage type: $advantage_type, learning rate: $lr, ent coef: $ent_coef, task: $task"
            python3 pusher_finetune_unified.py \
              --seed                 "$seed" \
              --pretrain_type        "$pre" \
              --vf_coef              0.5 \
              --num_steps            100 \
              --path       "runs_pusher_walled_off/${pre}/32_rollouts_contextual_knn25/advantage_type_running_avg_uncertainty_0.0_ent_coef_0.005/PO_Pusher__${seed}_entropy_type_${pre}_0.05_contextual_KNN_25_alpha0.005_10mil"\
              --init_type           "$init_type" \
              --advantage_type      "gae" \
              --task                "$task" \
              --num_envs            "$num_envs" \
              --track            True \
              --learning_rate       "$lr" \
              --save_dir          "runs_pusher_walled_off/finetuning/${task}/${pre}/PO_Pusher__${seed}_num_envs${num_envs}_contextual_lr_${lr}_ent_coef_${ent_coef}" \
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

