import torch
from torch import nn
import utils
import torch.nn.functional as F

class VeryBaddm(nn.Module):
    def __init__(self,input_dim: int,hidden_size:int,prediction_size: int,action_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.inital_hidden = nn.Parameter(torch.zeros((num_layers,1,hidden_size)))#initial hidden state
        self.observation_projection = nn.Sequential(

            nn.Linear(input_dim+2*hidden_size,2*hidden_size),
            nn.ELU(),
            nn.Linear(2*hidden_size,2*hidden_size),
            nn.ELU(),
            nn.Linear(2*hidden_size,2*hidden_size),
            nn.ELU()
        )
        self.reccurent = nn.GRU(input_size=2*hidden_size,hidden_size=hidden_size,batch_first=True,num_layers=self.num_layers)
        self.hidden_size = hidden_size
        self.action_projection =nn.Sequential(nn.Linear(action_dim,hidden_size),
                                                nn.ELU(),
                                                nn.Linear(hidden_size,hidden_size))
        
        self.latent_dynamics_model =nn.Sequential(
            nn.Linear(2*hidden_size,hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size,hidden_size)
            )
        self.obs_pred_projection = nn.Sequential(
            nn.Linear(input_dim,hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size,hidden_size)
        )
        

        self.prediction_size = prediction_size
        self.hidden_size = hidden_size


        self.reconstruction = nn.Sequential(
            nn.Linear(3*hidden_size,hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size,prediction_size))
        

    def embed(self,o_t,a_prev,h_prev):
        """
        o_t: Current observation [batch_size, input_dim]
        a_prev: Previous action [batch_size, action_dim]
        h_prev: Previous hidden state [num_layers, batch_size, hidden_dim]
        """
        #concatenate the observation and the previous action
        # gru_input = torch.cat([o_t,a_prev],dim=-1,).float()
        #project the input

        projected_action = self.action_projection(a_prev)
        context =torch.cat([h_prev[-1],projected_action],dim=-1) # [batch_size, hidden_dim+action_dim]
        projected_obs = self.observation_projection(torch.cat([o_t,context],dim=-1)).unsqueeze(1)
        
        #update the hidden state
        hids, h_t = self.reccurent(input=projected_obs,hx=h_prev)
        return h_t
    
    def embed_seq(self,obs,a_prevs,h_initial):
        """
        obs: sequence of observations [sequence_length,batch_size,input_dim]
        a_prevs: sequence of previous actions [sequence_length,batch_size,action_dim]
        h_initial: Initial hidden state [num_layers,batch_size,hidden_dim]
        """
        # gru_input = torch.cat([obs,a_prevs],dim=-1).float()#use O_{0:t-1} and a_{0:t-1} to get h_{0:t-1}
        #project the input
        seq_length, batch_size, action_dim = a_prevs.shape
        latents = torch.zeros((seq_length+1, batch_size, self.hidden_size)).to(obs.device)
        latents[0] = h_initial[-1]
        projected_actions = self.action_projection(a_prevs)
        for t in range(seq_length):
            context = torch.cat([h_initial[-1],projected_actions[t]],dim=-1)
            projected_obs = self.observation_projection(torch.cat([obs[t],context],dim=-1)).unsqueeze(1)
            #update the hidden state
            hids, h_t = self.reccurent(input=projected_obs,hx=h_initial)
            h_initial = h_t
            latents[t+1] = h_initial[-1]
        
        return latents    

        
class verybad_DM:
    def __init__(self,obs_space,hidden_size,act_space,prediction_size,device,episode_window,batch_length,lr,features_to_predict=None,ireducible_error=None,norm=False,num_layers=1):
    
        self.input_size = obs_space.shape[0]
        self.device = device
        if features_to_predict is not None:
            self.features_to_predict = features_to_predict
            prediction_size = len(features_to_predict)
        else:
            self.features_to_predict = None


        self.dynamics_model = VeryBaddm(input_dim=self.input_size,hidden_size= hidden_size,action_dim=act_space,prediction_size=prediction_size,num_layers=num_layers).to(self.device)
        self.num_layers = num_layers
        self.batch_length = batch_length
        self.optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=lr)
        self.latent_size = hidden_size
        self.ireducible_error = ireducible_error
        ####
        self.episode=0
        self.train_episodes={}
        self.episode_window=episode_window
        self.normalizer=utils.normalizer(device=self.device)
        self.norm = norm #whether to normalize the observations or not


        
    

    def initiallize_stats(self,num_envs,num_steps):
        self.stats={
            "pred_error": torch.zeros((num_steps)).to(self.device),
            "latent_consistensy_error": torch.zeros((num_steps)).to(self.device),
            "ireducible_error": torch.zeros((num_steps)).to(self.device),
        }
    
    def get_training_data(self, num_steps):
        self.dm_batch_length = min(num_steps, self.batch_length)
        dm_initial_lr = self.optimizer.param_groups[0]['lr']
        return dm_initial_lr
    
    def update_learning_rate(self,new_lr):
        self.optimizer.param_groups[0]['lr'] = new_lr

    def update_train_data(self,obs,actions,firsts,num_envs,mean,var,count):
        with torch.no_grad():
            for i in range(num_envs):
                self.train_episodes[self.episode] = {
                    "obs": obs[:,i],
                    "action": actions[:,i]}
                    # "is_first": firsts[:,i]}
                self.episode +=1
                self.episode = self.episode % self.episode_window

        #update the normalizer:
        self.normalizer.set_values(mean=mean,var=var,count=count)
    
    def get_initial_state(self,num_envs):
        hidden = self.dynamics_model.inital_hidden.expand(self.num_layers,num_envs,-1).contiguous().to(self.device)
        return hidden,{
            "latent_consistensy_error": torch.zeros(1),
            "pred_error": torch.zeros(1),
            "ireducible_error": torch.zeros(1),
            "uncertainty_error": torch.ones(1)
        }
    
    def step(self, obs, action,is_first, latent, stats):
        """
        obs: Current observation [batch_size, input_dim]
        action: past action [batch_size, action_dim]
        is_first: Whether the current step is the first step in the episode [batch_size]
        latent: Previous latent state [1,batch_size,hidden_dim]
        stats: Dictionary containing statistics for the current step
        """
        #embed the observation:
        
        if self.norm:
            obs = self.normalizer.normalize(obs)
        
        # Handle episode resets - when is_first is 1, use zero action for embedding
        # and reset the latent state for those environments
        is_first = is_first.view(-1)  # Ensure proper shape
        action_to_use = action.clone()
        
        if (is_first == 1).any():
            # Use zero actions for first steps (zeros out entire action vector for reset envs)
            reset_mask = (is_first == 1)
            action_to_use[reset_mask] = torch.zeros_like(action[reset_mask])
            
            # Reset latent states for environments that are starting new episodes
            latent[:, reset_mask] = self.dynamics_model.inital_hidden.expand(-1, reset_mask.sum(), -1)
            
        #predict next obs and next latent state:
        with torch.no_grad():

            
            #predict the next obs:
            embeded_action = self.dynamics_model.action_projection(action_to_use)
            embeded_obs = self.dynamics_model.obs_pred_projection(stats["prev_obs"])
            prediction_input = torch.cat([embeded_action,latent[-1],embeded_obs],dim=-1)
            next_obs_pred = self.dynamics_model.reconstruction(prediction_input)
            
            # Predict next latent from current latent and action (no observation needed)
            latent_pred_input = torch.cat([embeded_action, latent[-1]], dim=-1)
            next_latent_pred = self.dynamics_model.latent_dynamics_model(latent_pred_input)
            
            embed = self.dynamics_model.embed(o_t=obs,a_prev=action_to_use,h_prev=latent)
            #calculate the prediction error:
            if self.features_to_predict is not None:
                target_obs = obs[:,self.features_to_predict]
            else:
                target_obs = obs[:,:self.dynamics_model.prediction_size]
                
            pred_error =  torch.linalg.norm(next_obs_pred-target_obs).mean()
        
            #calculate the latent consistency error:
            latent_consistensy_error = F.mse_loss(next_latent_pred,embed[-1],reduction='none').mean()
            if self.ireducible_error is not None:
                #calculate the irreducible error:
                distance_to_puck=  torch.linalg.norm(stats["true_state"][:,14:17]-stats["true_state"][:,17:20]).mean()
                irreducible_error = self.ireducible_error*distance_to_puck
            else:
                irreducible_error = torch.tensor(0).to(self.device)
                
            #update stats:
            new_stats = {
                "latent_consistensy_error": 0,
                "pred_error": pred_error.cpu().item(),
                "uncertainty_error": 0,
                "ireducible_error": irreducible_error.cpu().item(),
            }
            if "true_state" in stats:
                new_stats["true_state"] = stats["true_state"]

        return embed,new_stats

    def train(self,num_epochs,batch_size,num_steps):
        sequence_length = num_steps #because we want to predict the next observation
        batch_size = batch_size
        sub_sample_size= 99
        #loss running average
        running_reconstruciton_loss = 0
        running_latent_dynamics_loss = 0
        for epoch in range(num_epochs):
            loss_terms = []
            latent = self.dynamics_model.inital_hidden.expand(self.num_layers,batch_size,-1).contiguous().to(self.device)
            obs, actions = utils.sample_traj(self.train_episodes, batch_size)
            if self.norm:
                obs = self.normalizer.normalize(obs)
            
            # Store all latents as we compute them sequentially
            history_sample_size = 16  # How many historical steps to predict for each timestep
            all_latents = []  # Will store latent at each timestep
            
            # First pass: compute all latents sequentially (since each depends on previous)
            for t in range(sequence_length):
                latent = self.dynamics_model.embed(o_t=obs[:,t],a_prev=actions[:,t],h_prev=latent)
                all_latents.append(latent[-1])  # Store the last layer of hidden state
            
            # Stack all latents for easier indexing
            all_latents = torch.stack(all_latents)  # [sequence_length, batch_size, hidden_size]
            
            # Now do double subsampling for prediction loss
            # Sample which timesteps to predict from
            timesteps_to_predict = torch.randperm(sequence_length)[:min(sub_sample_size, sequence_length)]
            
            for t in timesteps_to_predict:
                # For timestep t, we have latent l_t and want to predict transitions in history
                # Sample which historical transitions to predict (i < t)
                if t > 0:  # Can only predict history if t > 0
                    max_history = min(t, history_sample_size)  # Can predict from steps 0 to t-1
                    history_indices = torch.randperm(t)[:max_history]  # Sample from [0, t-1]
                    # Don't include t itself in history_indices since we're predicting from history
                    if len(history_indices) == 0:  # If no history samples, skip this timestep
                        continue
                    num_history_samples = len(history_indices)
                    
                    # Use l_t as context for ALL historical predictions
                    current_latent = all_latents[t]  # [batch_size, hidden_size] - this is l_t
                    
                    # Gather historical observations and actions
                    batch_obs = obs[:, history_indices]  # [batch_size, num_history_samples, obs_dim]
                    batch_obs = batch_obs.transpose(0, 1)  # [num_history_samples, batch_size, obs_dim]
                    batch_actions = actions[:, history_indices]  # [batch_size, num_history_samples, action_dim]
                    batch_actions = batch_actions.transpose(0, 1)  # [num_history_samples, batch_size, action_dim]
                    
                    # Expand l_t to match the number of history samples
                    batch_latents = current_latent.unsqueeze(0).expand(num_history_samples, -1, -1)  # [num_history_samples, batch_size, hidden_size]
                    # Reshape for batch processing
                    batch_latents_flat = batch_latents.reshape(-1, batch_latents.shape[-1])  # [max_history * batch_size, hidden_size]
                    batch_obs_flat = batch_obs.reshape(-1, batch_obs.shape[-1])  # [max_history * batch_size, obs_dim]
                    batch_actions_flat = batch_actions.reshape(-1, batch_actions.shape[-1])  # [max_history * batch_size, action_dim]
                    
                    # Get action projections for all history steps at once
                    projected_actions = self.dynamics_model.action_projection(batch_actions_flat)
                    projeted_obs = self.dynamics_model.obs_pred_projection(batch_obs_flat)
                    # Concatenate for prediction: l_t provides context for all predictions
                    prediction_input = torch.cat([
                        projected_actions,
                        batch_latents_flat,  # l_t repeated for each history step
                        projeted_obs  # o_i for each i in history_indices
                    ], dim=-1)
                    
                    # Predict next observations for all history steps at once
                    next_obs_pred = self.dynamics_model.reconstruction(prediction_input)
                    
                    # Predict only l_{t+1} from l_t and a_t (not for all history indices)
                    # if t < sequence_length - 1:  # Can only predict next latent if not at last timestep
                    #     # Use l_t and a_t to predict l_{t+1}
                    #     latent_pred_input = torch.cat([
                    #         self.dynamics_model.action_projection(actions[:, t]),  # a_t projected
                    #         current_latent  # l_t
                    #     ], dim=-1)
                    #     next_latent_pred = self.dynamics_model.latent_dynamics_model(latent_pred_input)
                        
                    #     # Target is l_{t+1}
                    #     target_latent = all_latents[t + 1]  # [batch_size, hidden_size]
                        
                    #     # Calculate latent prediction loss
                    #     latent_loss = F.mse_loss(next_latent_pred, target_latent)
                    # else:
                    #     latent_loss = torch.tensor(0.0).to(obs.device)
                    
                    # Get targets - ensure we don't go out of bounds
                    # history_indices are in [0, t-1], so history_indices + 1 are in [1, t]
                    # Since t < sequence_length, this should be safe, but let's be explicit
                    target_indices = history_indices + 1  # We want o_{i+1} for each i
                    # Ensure target_indices don't exceed sequence bounds
                    valid_mask = target_indices < sequence_length
                    if not valid_mask.all():
                        # Filter out invalid indices
                        history_indices = history_indices[valid_mask]
                        target_indices = target_indices[valid_mask]
                        num_history_samples = len(history_indices)
                        if num_history_samples == 0:
                            continue
                        # Re-gather data with valid indices only
                        batch_obs = obs[:, history_indices].transpose(0, 1)
                        batch_actions = actions[:, history_indices].transpose(0, 1)
                        batch_latents = current_latent.unsqueeze(0).expand(num_history_samples, -1, -1)
                        batch_latents_flat = batch_latents.reshape(-1, batch_latents.shape[-1])
                        batch_obs_flat = batch_obs.reshape(-1, batch_obs.shape[-1])
                        batch_actions_flat = batch_actions.reshape(-1, batch_actions.shape[-1])
                        projected_actions = self.dynamics_model.action_projection(batch_actions_flat)
                        projeted_obs = self.dynamics_model.obs_pred_projection(batch_obs_flat)
                        prediction_input = torch.cat([projected_actions, batch_latents_flat, projeted_obs], dim=-1)
                        next_obs_pred = self.dynamics_model.reconstruction(prediction_input)
                    
                    batch_targets = obs[:, target_indices]  # [batch_size, num_history_samples, obs_dim]
                    batch_targets = batch_targets.transpose(0, 1)  # [max_history, batch_size, obs_dim]
                    
                    if self.features_to_predict is not None:
                        batch_targets = batch_targets[:, :, self.features_to_predict]
                    else:
                        batch_targets = batch_targets[:, :, :self.dynamics_model.prediction_size]
                    
                    batch_targets_flat = batch_targets.reshape(-1, batch_targets.shape[-1])
                    
                    # Calculate reconstruction loss for all predictions at once
                    reconstruction_loss = F.mse_loss(next_obs_pred, batch_targets_flat)
                    
                    # Combine losses
                    total_step_loss = reconstruction_loss #+ latent_loss
                    loss_terms.append(total_step_loss)
            
            # Backward pass and optimization
            if len(loss_terms) > 0:
                total_loss = torch.stack(loss_terms).mean()
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), 1.0)
                self.optimizer.step()
                
                running_reconstruciton_loss += total_loss.item()
        
        #mean:
        running_reconstruciton_loss /= num_epochs
        metrics = {
            "reconstruction_loss": running_reconstruciton_loss,
        }
        return metrics
    
    def get_latents(self,obs,actions,initial_hidden=None,dones=None):
        """
        obs: sequence of observations [sequence_length,batch_size,input_dim]
        actions: sequence of actions [sequence_length,batch_size,action_dim] - the actual actions taken by agent
        initial_hidden: initial hidden states [num_layers,batch_size,hidden_dim] (optional)
        dones: episode termination flags [sequence_length,batch_size] (optional)
      
        gets obs and actions and predicts the latents for each timestep
        NOTE: actions contains the actual actions taken, but when is_first==1 we use zero action internally
        """
        if self.norm:
            obs = self.normalizer.normalize(obs)
        seq_length, batch_size, action_dim = actions.shape
        
        # Use provided initial hidden or default
        if initial_hidden is not None:
            hidden = initial_hidden.clone()
        else:
            # hidden = self.dynamics_model.inital_hidden.expand(self.num_layers,batch_size,-1).contiguous().to(self.device)
            raise ValueError("Initial hidden state must be provided for contextual DM.")
        
        # If dones are provided, handle episode resets properly
        
        latents = torch.zeros((seq_length, batch_size, self.dynamics_model.hidden_size)).to(self.device)
        
        for t in range(seq_length):
            # Determine the action to use for embedding
            if t == 0:
                # For the very first timestep, the action taken was based on the initial hidden state
                latents[t] = hidden[-1]
                # If dones from before this sequence indicate reset, use zero action
                # This would be handled by initial_hidden being from after reset
            else:
                # Check if step lead to completion (meaning we reset the environment)
                is_first = dones[t]  
                action_to_use = actions[t-1].clone()
                
                # For environments that just reset, use zero action instead of stored action
                if is_first.any():
                    reset_mask = is_first.bool()
                    action_to_use[reset_mask] = 0
                    # Reset hidden states for those environments
                    hidden[:, reset_mask] = self.dynamics_model.inital_hidden.expand(-1, reset_mask.sum(), -1)
            
                # Embed current observation with the appropriate action
                hidden = self.dynamics_model.embed(o_t=obs[t], a_prev=action_to_use, h_prev=hidden)
                latents[t] = hidden[-1]
        
        return latents
