# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from normalize import NormalizeObservation, NormalizeReward
from torch.distributions.normal import Normal
import argparse
import pathlib
import PROBE
import utils
import sys
import torch.nn.functional as F
from base_dm import DM , contextual_DM
from verybad_dm import verybad_DM
def parse_args():
    parser = argparse.ArgumentParser()
    # Algorithm specific arguments
    parser.add_argument("--exp_name", default=os.path.basename(__file__)[: -len(".py")], help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=67, help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=bool, default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=bool, default=False, help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb_project_name", default="", help="the wandb's project name")
    parser.add_argument("--wandb_entity", default="", help="the entity (team) of wandb's project")
    parser.add_argument("--capture_video", type=bool, default=False, help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save_model", type=bool, default=True, help="whether to save model into the `runs_new_reccurent/{run_name}` folder")
    parser.add_argument("--save_dir", type=str, default="runs/state/ant", help="the dir to save the model")
    parser.add_argument("--upload_model", type=bool, default=False, help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf_entity", default="", help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--env_id", default="PO_ant-v5", help="the id of the environment")
    parser.add_argument("--control_cost", type=int,default=0, help="coefficient of the control cost")
    parser.add_argument("--total_timesteps", type=int, default=10000000, help="total timesteps of the experiments")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=16, help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default=1000, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal_lr", type=bool, default=False, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=4, help="the K epochs to update the policy")
    parser.add_argument("--norm_adv", type=bool, default=False, help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.25, help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=bool, default=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--vf_coef", type=float, default=0.0, help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None, help="the target KL divergence threshold")
    parser.add_argument("--advantage_type", type=str, default="running_avg", help="what type of advantage_baseline to use, options are: 'running_avg', 'GAE', 'value_based'")
    parser.add_argument("--ent_coef", type=float, default=0.002, help="coefficient of the entropy")
    parser.add_argument("--anneal_alpha", type=bool, default=False, help="Toggles entropy coefficient annealing")
    parser.add_argument("--network_hidden_size", type=int, default=512, help="the size of the hidden layer in the network")
    parser.add_argument("--initial_epochs", type=int, default=400, help="the number of warm-up iterations (computed in runtime)")
    parser.add_argument("--dreamer_epochs", type=int, default=10, help="the number of dreamer epochs")
    parser.add_argument("--dreamer_batch_size", type=int, default=128, help="the dreamer batch size")
    parser.add_argument("--dreamer_window_length", type=int, default=30, help="the number of steps to use for the dreamer window length")
    parser.add_argument("--dm_batch_length", type=int, default=50, help="the number of steps to use for the dreamer batch length")
    parser.add_argument("--dm_hidden_dim", type=int, default=128, help="the hidden dimension of the dreamer model")
    parser.add_argument("--detailed_obs_logging", type=bool, default=False, help="whether to log the detailed observations")
    parser.add_argument("--entropy_sub_calc", type=int, default=10, help="the number of steps to calculate the entropy, should be a divisor of num_steps,reduces the memory usage")
    parser.add_argument("--warm_up", type=int, default=40, help="the number of warm-up iterations (computed in runtime)")
    parser.add_argument("--entropy_type", type=str, default="latent", help="the type of entropy to use, either 'latent' or 'state' or 'obs' ")
    parser.add_argument("--update_freq", type=int, default=5, help="the frequency of the dreamer training")
    parser.add_argument("--free_start", type=int, default=0, help="the number of steps to skip before starting the training")
    parser.add_argument("--irrelavent_obs_idx", type=list, default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], help="the indices of the ambivilant observations to use for the entropy calculation")
    parser.add_argument("--dm", type=str, default="contextual", help="the type of the dreamer to use, either 'verybad_style' or 'base' or 'contextual")
    parser.add_argument("--po_threshold", type=float, default=0.75, help="the factor of the noise to add to the actions")
    parser.add_argument("--include_cfrc_ext_in_observation", type=bool, default=False, help="whether to include the contact forces and torques in the observation")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers in the GRU for the dynamics model")
    parser.add_argument("--knn_k", type=int, default=25, help="the number of nearest neighbors to use in the KNN for the dynamics model")
    # parser.add_argument("--swicth", type=int, default=50, help="the number of steps to switch the entropy type from obs to latent")
    # to be filled in runtime
    parser.add_argument("--batch_size", type=int, default=0, help="the batch size (computed in runtime)")
    parser.add_argument("--minibatch_size", type=int, default=0, help="the mini-batch size (computed in runtime)")
    parser.add_argument("--num_iterations", type=int, default=0, help="the number of iterations (computed in runtime)")
    args = parser.parse_args()
    args.wandb_group = f"{args.entropy_type}_even_bigger_batch_{args.dm}"
    args.include_cfrc_ext_in_observation =False
    args.dreamer_episode_window = args.num_envs * max(args.dreamer_window_length,args.warm_up)
    return args

def make_env(env_id, idx, capture_video, run_name, gamma,horizon=1000,po_threshold=0.75,include_cfrc_ext_in_observation=False):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id,po_threshold=po_threshold,include_cfrc_ext_in_observation=include_cfrc_ext_in_observation)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env =  gym.wrappers.TimeLimit(env, max_episode_steps=horizon)  
        return env
    
    return thunk



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs,input_dim,hidden_dim=512):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod()+input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),#note we added another layer here
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod()+input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),#note we added another layer here
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0))


    def get_action_and_value(self, x,latent, action=None):
        latent=latent[-1] if latent.ndim == 3 else latent
        input = torch.cat([x,latent],dim=1)
        action_mean = self.actor_mean(input)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        value = self.critic(input)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value # value is not used in this agent, so we return 0


    
if __name__ == "__main__":
    args = parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # if args.uncertainty_coef != 0 and args.entropy_type == "latent":
    #     assert args.advantage_type == "running_avg"

        
        
    run_name = f"{args.env_id}__{args.seed}_entropy_type_{args.entropy_type}_16_rollouts_{args.dm}_knn{args.knn_k}_dm_batch_{args.dm_batch_length}_dreamer_epochs_{args.dreamer_epochs}_warmup_{args.warm_up}_noexc"
    if args.save_model and not os.path.exists(f"{args.save_dir}"):
            os.makedirs(f"{args.save_dir}", exist_ok=True)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=False,
        )
    writer = SummaryWriter(f"{args.save_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
   
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma,args.num_steps,args.po_threshold,args.include_cfrc_ext_in_observation) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


    if args.dm == "base":
        features_to_predict = None
        ireducible_error = None
        dm = DM(obs_space=envs.single_observation_space, act_space=envs.single_action_space.shape[0],prediction_size=envs.single_observation_space.shape[0],hidden_size=args.dm_hidden_dim ,device=device,
                episode_window=args.dreamer_episode_window, batch_length=args.dm_batch_length, lr=args.learning_rate,features_to_predict=features_to_predict,ireducible_error=ireducible_error,
                norm=False,num_layers=args.num_layers)
    elif args.dm == "contextual":
        features_to_predict = None
        ireducible_error = None
        dm = contextual_DM(obs_space=envs.single_observation_space, act_space=envs.single_action_space.shape[0],prediction_size=envs.single_observation_space.shape[0],hidden_size=args.dm_hidden_dim ,device=device,
                episode_window=args.dreamer_episode_window, batch_length=args.dm_batch_length, lr=args.learning_rate,features_to_predict=features_to_predict,ireducible_error=ireducible_error,
                norm=False,num_layers=args.num_layers)
    elif args.dm == "verybad_style":
        features_to_predict = None
        ireducible_error = None
        dm = verybad_DM(obs_space=envs.single_observation_space, act_space=envs.single_action_space.shape[0],prediction_size=envs.single_observation_space.shape[0],hidden_size=args.dm_hidden_dim ,device=device,
                episode_window=args.dreamer_episode_window, batch_length=args.dm_batch_length, lr=args.learning_rate,features_to_predict=features_to_predict,ireducible_error=ireducible_error,
                norm=False,num_layers=args.num_layers)
        
    dm_initial_lr =dm.get_training_data(args.num_steps)
    dm_lrnow = dm_initial_lr

    
    agent = Agent(envs=envs,input_dim=dm.latent_size,hidden_dim=args.network_hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    unnormed_obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    unnormed_true_states = torch.zeros((args.num_steps, args.num_envs,envs.envs[0].full_state_shape[0])).to(device)
    latents = torch.zeros((args.num_steps, args.num_envs, dm.latent_size)).to(device)
    firsts = torch.zeros((args.num_steps,args.num_envs)).to(device)
    firsts[0,:] = torch.ones(args.num_envs).to(device)
    dm.initiallize_stats(args.num_envs,args.num_steps)
    baseline =utils.MovingAvgRobust(k=20,tensor_shape=(args.num_steps,),device=device)#K IS CRUCIAL
    baseline.add(torch.zeros((args.num_steps,)).to(device))
  

    ##dreamer storage:
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs,reset_info  = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    not_first = torch.zeros(args.num_envs).to(device)    
    unnormed_true_state = torch.Tensor(np.stack(reset_info['state'])).to(device)
    unnormed_ob = torch.Tensor(np.stack(reset_info['unnormalized'])).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    ent_alpha= args.ent_coef
    warmup_finished = False

    for iteration in range(1, args.num_iterations + 1):
        if iteration ==args.warm_up:
            warmup_finished = True
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            dm_lrnow = frac * dm_initial_lr
            dm.update_learning_rate(dm_lrnow)

        if args.anneal_alpha:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            ent_alpha = frac * args.ent_coef
        
        # ALGO LOGIC: setup initial dreamer state
        latent, stats = dm.get_initial_state(args.num_envs)
        stats["true_state"] = unnormed_true_state
        stats["prev_obs"] = torch.zeros_like(unnormed_ob) 
        a_minus_1 = torch.zeros((1,args.num_envs, envs.single_action_space.shape[0])).to(device)  
        latent, stats=dm.step(unnormed_ob, a_minus_1.squeeze(0), not_first, latent, stats)  # initialize the dreamer state
        stats["prev_obs"] = unnormed_ob  # Now update prev_obs to current obs for next iteration

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            unnormed_obs[step] = unnormed_ob
            unnormed_true_states[step] = unnormed_true_state
            latents[step]=latent[-1]#only the last layer
            for key, value in stats.items():
                if key in dm.stats.keys():
                    dm.stats[key][step] = value
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value= agent.get_action_and_value(x=next_obs, latent=latent[-1], action=None)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            unnormed_ob = torch.Tensor(np.stack(infos['unnormalized'])).to(device)
            unnormed_true_state = torch.Tensor(np.stack(infos['state'])).to(device)
            stats["true_state"] = unnormed_true_state
            latent, stats = dm.step(unnormed_ob, action, not_first, latent, stats)
            stats["prev_obs"] = unnormed_ob

            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            #             writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            #             writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        dm_obs=torch.cat([unnormed_obs, unnormed_ob.unsqueeze(0)], dim=0)  # we want O_{0:t+1}
        a_minus_1 = torch.zeros((1,args.num_envs, envs.single_action_space.shape[0])).to(device)  # a_{-1} is zero
        dm_actions = torch.cat([a_minus_1, actions], dim=0)  # we want A_{-1:t}
        dm.update_train_data(dm_obs, dm_actions, firsts, args.num_envs,mean=infos['normalize_mean'].mean(),var=infos['normalize_var'].mean(),count=infos['normalize_count'].mean())#make sure it is unnomed as the normalization is within

        # compute the state entropies:
        if warmup_finished:
            state_entropies = torch.zeros((args.num_steps, args.num_envs)).to(device)
            latent_entropies = torch.zeros((args.num_steps, args.num_envs)).to(device)
            obs_entropies = torch.zeros((args.num_steps, args.num_envs)).to(device)
            for i in range(args.num_envs):
                obs_entropy = utils.compute_state_entropy(src_feats=unnormed_obs[args.free_start:,i,17:20], tgt_feats=unnormed_obs[args.free_start:,:,17:20], average_entropy=False,batch=True,calc_steps=args.entropy_sub_calc,k_fixed=args.knn_k)
                state_entropy = utils.compute_state_entropy(src_feats=unnormed_true_states[args.free_start:,i,17:20], tgt_feats=unnormed_true_states[args.free_start:,:,17:20], average_entropy=False,batch=True,calc_steps=args.entropy_sub_calc,k_fixed=args.knn_k)
                latent_entropy = utils.compute_state_entropy(src_feats=latents[args.free_start:,i], tgt_feats=latents[args.free_start:], average_entropy=False,batch=True,calc_steps=args.entropy_sub_calc,k_fixed=args.knn_k)#note that avg entropy is set to true
                
                obs_entropies[args.free_start:,i] = obs_entropy
                state_entropies[args.free_start:,i] = state_entropy
                latent_entropies[args.free_start:,i] = latent_entropy

            

            if args.entropy_type == "state":
                rewards = state_entropies
            elif args.entropy_type == "obs":
                rewards = obs_entropies
            elif args.entropy_type == "latent":
                rewards = latent_entropies 

  
            else:
                raise ValueError("entropy type not supported")
            

            # bootstrap value if not done
            with torch.no_grad():
                next_value = torch.zeros((args.num_envs,)).to(device)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                if args.advantage_type=='gae':#gae
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values
                elif args.advantage_type=='value_based':#value baseline
                    returns=torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                            if t == args.num_steps - 1:
                                nextnonterminal = 1.0 - next_done
                                next_return = next_value
                            else:
                                nextnonterminal = 1.0 - dones[t + 1]
                                next_return = returns[t + 1]
                            returns[t] = rewards[t] + 1 * nextnonterminal * next_return #could be args.gamma instaed of 1
                            advantages[t] = returns[t] - values[t]
                elif args.advantage_type == "running_avg":
                        pure_rewards = rewards.clone()#we will use the rewards as the baseline
                        rewards = rewards   
                        #no value function, just use the running average of the rewards:
                        returns = torch.zeros_like(rewards).to(device)
                        curr_baseline = baseline.get_avg()

                        returns_baseline = torch.zeros_like(rewards).to(device)
                        for t in reversed(range(args.num_steps)):
                            if t == args.num_steps - 1:
                                nextnonterminal = 1.0 - next_done
                                next_return = 0
                                next_baseline = 0
                            else:
                                nextnonterminal = 1.0 - dones[t + 1]
                                next_return = returns[t + 1]
                                next_baseline = returns_baseline[t + 1]
                            returns[t] = rewards[t] + 1 * nextnonterminal * next_return
                            returns_baseline[t] = curr_baseline[t] + 1 * nextnonterminal * next_baseline
                            advantages[t] = returns[t] - returns_baseline[t]
                        
                        #update the baseline:
                        baseline.add(pure_rewards.mean(dim=1))

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_latents = latents.reshape((-1,) + (dm.latent_size,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds],b_latents[mb_inds],b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            


        epochs =0
        if warmup_finished and iteration % args.update_freq == 0:
            epochs = utils.get_epochs(current_iter=iteration,args=args)
            dreamer_metrics = dm.train(
                    num_epochs=epochs,
                    batch_size=args.dreamer_batch_size,
                    num_steps=args.num_steps,
                )
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if warmup_finished:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("dreamer/learning_rate", dm_lrnow, global_step)
            writer.add_scalar("charts/encoder_epochs", epochs, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/latent_entropy",latent_entropies.cpu().mean().item(), global_step)
            writer.add_scalar("charts/state_entropy",state_entropies.cpu().mean().item(), global_step)
            writer.add_scalar("charts/objective",rewards.cpu().mean().item(), global_step)
            writer.add_scalar("charts/obs_entropy",obs_entropy.cpu().mean().item(), global_step)

            for key, value in dm.stats.items():
                writer.add_scalar(f"dynamics_model/{key}", value.cpu().mean().item(), global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # print dreamer training metrics
        if warmup_finished and iteration % args.update_freq == 0:
            for key, value in dreamer_metrics.items():
                #if key ends with loss, then it add it to writer:
                if key.endswith("loss") or key.endswith("ent"):
                    writer.add_scalar(f"dreamer/{key}", utils.list_mean(value), global_step)



    if args.save_model:
        model_path = f"{args.save_dir}/{run_name}/agent.pth"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        #save the normalization stats:
        normalize_count = infos['normalize_count']
        normalize_mean = infos['normalize_mean']
        normalize_var = infos['normalize_var']
        np.savez(model_path.replace(".pth", "_normalize.npz"), normalize_count=normalize_count, normalize_mean=normalize_mean, normalize_var=normalize_var)
        #save the dreamer model:
        torch.save(dm.dynamics_model.state_dict(), model_path.replace(".pth", "_dm.pth"))


    envs.close()
    writer.close()

