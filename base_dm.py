import torch
from torch import nn
import utils
import torch.nn.functional as F
class dynamicsModel(nn.Module):

    def __init__(self,input_dim: int,hidden_size:int,prediction_size: int,action_dim: int, num_layers: int = 1) -> None:
        super().__init__()


                # Wider, deeper encoder with residual connections
        self.num_layers = num_layers
        self.initial_hidden = nn.Parameter(torch.zeros((num_layers, 1, hidden_size)))  # Learnable initial hidden state
        self.observation_projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512), 
            nn.ELU(),
            nn.Linear(512, hidden_size)
        )
        self.reccurent = nn.GRU(input_size=2*hidden_size,hidden_size=hidden_size,batch_first=True,num_layers=self.num_layers)
        self.hidden_size = hidden_size
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.ELU(),
            nn.Linear(256, hidden_size)
        )

        
        self.latent_dynamics_model =nn.Sequential(
            nn.Linear(2 * hidden_size, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, hidden_size)
        )

        self.prediction_size = prediction_size
        self.hidden_size = hidden_size


        self.reconstruction = nn.Sequential(
            nn.Linear(2 * hidden_size, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, prediction_size)
        )
        
    def predict(self,obs,actions,hidden):
        """
        obs: sequence of observations [sequence_length,batch_size,input_dim]
        actions: sequence of actions [sequence_length+1,batch_size,action_dim]
        hidden: the first hidden state [num_layers,batch_size,hidden_dim]
        gets obs and actions and predicts the next observation
        """
        projected_actions = self.action_projection(actions)#embed the actions a_{-1:t}
    
        projected_obs = self.observation_projection(obs)#project the obs O_{0:t}
        projected_input = torch.cat([projected_obs,projected_actions[:,:-1]],dim=-1)#use O_{0:t} and a_{-1:t-1} to get h_{0:t}
        #update the hidden state
        hiddens, h_t = self.reccurent(input=projected_input,hx=hidden)
        input = torch.cat([hiddens,projected_actions[:,1:]],dim=-1)#use h_{0:t} and a_{0:t} to predict O_{1:t+1}
        recunstructed = self.reconstruction(input)
        next_latent_pred = self.latent_dynamics_model(input[:-1])#use h_{0:t-1} and a_{0:t-1} to predict h_{1:t}
        return recunstructed,next_latent_pred,h_t,hiddens[1:]
    
    def signle_step_predict(self,hidden,action):
        """
        h_{t-1},a_{t-1} -> pred(h_t), pred(O_{t})
        hidden: [num_layers, batch_size, hidden_dim]
        """
        projected_action = self.action_projection(action)
        # Use the last layer's hidden state for prediction
        input = torch.cat([hidden[-1],projected_action],dim=-1)
        recunstructed = self.reconstruction(input)
        next_latent_pred = self.latent_dynamics_model(input)
        return recunstructed,next_latent_pred

    def embed(self,o_t,a_prev,h_prev):
        """
        o_t: Current observation [batch_size, input_dim]
        a_prev: Previous action [batch_size, action_dim]
        h_prev: Previous hidden state [num_layers, batch_size, hidden_dim]
        """
        #concatenate the observation and the previous action
        # gru_input = torch.cat([o_t,a_prev],dim=-1,).float()
        #project the input
        projected_obs = self.observation_projection(o_t).unsqueeze(1)
        projected_action = self.action_projection(a_prev).unsqueeze(1)
        projected_input = torch.cat([projected_obs,projected_action],dim=-1)
        #update the hidden state
        hids, h_t = self.reccurent(input=projected_input,hx=h_prev)
        return h_t
    
    # def embed_seq(self,obs,a_prevs,h_initial):
    #     """
    #     obs: sequence of observations [sequence_length,batch_size,input_dim]
    #     a_prevs: sequence of previous actions [sequence_length,batch_size,action_dim]
    #     h_initial: Initial hidden state [1,batch_size,hidden_dim]
    #     """
    #     # gru_input = torch.cat([obs,a_prevs],dim=-1).float()#use O_{0:t-1} and a_{0:t-1} to get h_{0:t-1}
    #     #project the input

    #     projected_obs = self.observation_projection(obs)
    #     projected_action = self.action_projection(a_prevs)
    #     projected_input = torch.cat([projected_obs,projected_action],dim=-1)
    #     #review the projected input shape:
    #     projected_input=projected_input.permute(1,0,2) # [sequence_length, batch_size, 128] -> [batch_size, 128, sequence_length]
    #     #update the hidden state
    #     hids, h_t = self.reccurent(input=projected_input,hx=h_initial)
    #     return hids
    




class DM:
    def __init__(self,obs_space,hidden_size,act_space,prediction_size,device,episode_window,batch_length,lr,features_to_predict=None,ireducible_error=None,norm=False,num_layers=1):
    
        self.input_size = obs_space.shape[0]
        self.device = device
        if features_to_predict is not None:
            self.features_to_predict = features_to_predict
            prediction_size = len(features_to_predict)
        else:
            self.features_to_predict = None


        self.dynamics_model = dynamicsModel(input_dim=self.input_size,hidden_size= hidden_size,action_dim=act_space,prediction_size=prediction_size,num_layers=num_layers).to(self.device)
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
        # Use the learned initial hidden state and expand it for all environments
        hidden = self.dynamics_model.initial_hidden.expand(self.num_layers, num_envs, -1).contiguous()
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
            latent[:, reset_mask] = self.dynamics_model.initial_hidden.expand(-1, reset_mask.sum(), -1)
            
        #predict next obs and next latent state:
        with torch.no_grad():

            next_obs_pred,next_latent_pred = self.dynamics_model.signle_step_predict(hidden=latent,action=action_to_use)
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
                "latent_consistensy_error": latent_consistensy_error.cpu().item(),
                "pred_error": pred_error.cpu().item(),
                "uncertainty_error": pred_error.cpu().item(),
                "ireducible_error": irreducible_error.cpu().item(),
            }

        return embed,new_stats

    def train(self,num_epochs,batch_size,num_steps):

        sequence_length = num_steps #+1 because we want to predict the next observation
        batch_size = batch_size
        #loss running average
        running_reconstruciton_loss = 0
        running_latent_dynamics_loss = 0
        for epoch in range(num_epochs):
            # Use the learned initial hidden state
            hidden = self.dynamics_model.initial_hidden.expand(self.num_layers, batch_size, -1).contiguous()
            obs, actions = utils.sample_traj(self.train_episodes, batch_size)
            if self.norm:
                obs = self.normalizer.normalize(obs)
            for start in range(0,sequence_length,self.dm_batch_length):
                end = min(start + self.dm_batch_length,sequence_length-1)
                chunk_obs = obs[:,start:end]#O_{0:t} 
                chunk_obs_reconstruct = obs[:,start+1:end+1]#(O_{0:t},a_{0:t}) -> O_{1:t+1}
                chunk_actions = actions[:,start:end+1]#a_{-1:t}
                reconstructed_obs ,next_latent_pred,hidden,hiddens = self.dynamics_model.predict(obs=chunk_obs,actions=chunk_actions,hidden=hidden)
                
                if self.features_to_predict is not None:
                    chunk_obs_reconstruct = chunk_obs_reconstruct[:,:,self.features_to_predict]
                else:
                    chunk_obs_reconstruct = chunk_obs_reconstruct[:,:,:self.dynamics_model.prediction_size]

                reconstruciton_loss = F.mse_loss(reconstructed_obs,chunk_obs_reconstruct)
                latent_dynamics_loss = F.mse_loss(next_latent_pred,hiddens.detach())
                representation_loss = F.mse_loss(next_latent_pred.detach(),hiddens)
                encoder_loss = reconstruciton_loss +latent_dynamics_loss+0.5*representation_loss

                #truncate the hidden state
                hidden = hidden.detach()
                # Backward pass and optimization
                self.optimizer.zero_grad()
                encoder_loss.backward()
                self.optimizer.step()

                #LOSS RUNNING average:
                running_reconstruciton_loss += reconstruciton_loss.item()
                running_latent_dynamics_loss += latent_dynamics_loss.item()
            
        
        #mean:
        running_reconstruciton_loss /= num_epochs
        running_latent_dynamics_loss /= num_epochs
        metrics = {
            "reconstruction_loss": running_reconstruciton_loss,
            "latent_dynamics_loss": running_latent_dynamics_loss
        }
        return metrics
    
    # def get_latents(self,obs,actions):
    #     """
    #     obs: sequence of observations [sequence_length,batch_size,input_dim]
    #     actions: sequence of actions [sequence_length,batch_size,action_dim]
    #   
    #     gets obs and actions and predicts the next observation
    #     """
    #     if self.norm:
    #         obs = self.normalizer.normalize(obs)
    #     seq_length, batch_size, _ = obs.shape
    #     latents =torch.zeros((seq_length, batch_size, self.dynamics_model.hidden_size)).to(self.device)
    #     hidden = torch.zeros((self.num_layers,batch_size, self.dynamics_model.hidden_size)).to(self.device)
    #     latents[0] = hidden[-1]
    #     hiddens= self.dynamics_model.embed_seq(obs=obs[1:], a_prevs=actions[:-1], h_initial=hidden)
    #     hiddens = hiddens.permute(1,0,2)
    #     latents[1:] = hiddens
    #     return latents


class contextual_dynamicsModel(nn.Module):

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

        self.prediction_size = prediction_size
        self.hidden_size = hidden_size


        self.reconstruction = nn.Sequential(
            nn.Linear(2*hidden_size,(hidden_size+action_dim)//2),
            nn.ELU(),
            nn.Linear((hidden_size+action_dim)//2,(hidden_size+action_dim)//2),
            nn.ELU(),
            nn.Linear((hidden_size+action_dim)//2,prediction_size))
        
    def predict(self,obs,actions,hidden):
        """
        obs: sequence of observations [sequence_length,batch_size,input_dim]
        actions: sequence of actions [sequence_length,batch_size,action_dim]
        hidden: the previous hidden state [num_layers,batch_size,hidden_dim]
        gets obs and actions and predicts the next observation
        """
        batch_size,sequence_length, _ = obs.shape
        hiddens = torch.zeros((batch_size,sequence_length, self.hidden_size)).to(obs.device)
        recunstructed_obs = torch.zeros((batch_size,sequence_length, self.prediction_size)).to(obs.device)#O_{1:t+1}
        predicted_latents = torch.zeros((batch_size,sequence_length, self.hidden_size)).to(obs.device)#h_{1:t+1}
        projected_actions = self.action_projection(actions)
        for t in range(sequence_length):
            input = torch.cat([hidden[-1],projected_actions[:,t]],dim=-1)#use h_{t} and a_{t} to predict O_{t+1}
            recunstructed_obs[:,t] = self.reconstruction(input)
            predicted_latents[:,t] = self.latent_dynamics_model(input)

            
            context = torch.cat([hidden[-1],projected_actions[:,t]],dim=-1) # [batch_size, hidden_dim+action_dim]
            projected_obs = self.observation_projection(torch.cat([obs[:,t],context],dim=-1)).unsqueeze(1)
            #update the hidden state
            _, hidden = self.reccurent(input=projected_obs,hx=hidden)
            hiddens[:,t] = hidden[-1]


        return recunstructed_obs, predicted_latents, hidden, hiddens

            
    
    
    def signle_step_predict(self,hidden,action):
        """
        hidden: [num_layers, batch_size, hidden_dim]
        """
        projected_action = self.action_projection(action)
        # Use the last layer's hidden state for prediction
        input = torch.cat([hidden[-1],projected_action],dim=-1)
        recunstructed = self.reconstruction(input)
        next_latent_pred = self.latent_dynamics_model(input)
        return recunstructed,next_latent_pred

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

        

    


class contextual_DM:
    def __init__(self,obs_space,hidden_size,act_space,prediction_size,device,episode_window,batch_length,lr,features_to_predict=None,ireducible_error=None,norm=False,num_layers=1):
    
        self.input_size = obs_space.shape[0]
        self.device = device
        if features_to_predict is not None:
            self.features_to_predict = features_to_predict
            prediction_size = len(features_to_predict)
        else:
            self.features_to_predict = None


        self.dynamics_model = contextual_dynamicsModel(input_dim=self.input_size,hidden_size= hidden_size,action_dim=act_space,prediction_size=prediction_size,num_layers=num_layers).to(self.device)
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
    def pred_next_obs(self,latent,action):
        with torch.no_grad():
            next_obs_pred,_ = self.dynamics_model.signle_step_predict(hidden=latent,action=action)
        return next_obs_pred
    
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

            next_obs_pred,next_latent_pred = self.dynamics_model.signle_step_predict(hidden=latent,action=action_to_use)
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
                "latent_consistensy_error": latent_consistensy_error.cpu().item(),
                "pred_error": pred_error.cpu().item(),
                "uncertainty_error": pred_error.cpu().item(),
                "ireducible_error": irreducible_error.cpu().item(),
            }
            if "true_state" in stats:
                new_stats["true_state"] = stats["true_state"]

        return embed,new_stats

    def train(self,num_epochs,batch_size,num_steps):

        sequence_length = num_steps #because we want to predict the next observation
        batch_size = batch_size
        #loss running average
        running_reconstruciton_loss = 0
        running_latent_dynamics_loss = 0
        for epoch in range(num_epochs):
            hidden = self.dynamics_model.inital_hidden.expand(self.num_layers,batch_size,-1).contiguous().to(self.device)
            obs, actions = utils.sample_traj(self.train_episodes, batch_size)
            if self.norm:
                obs = self.normalizer.normalize(obs)
            for start in range(0,sequence_length,self.dm_batch_length):
                end = min(start + self.dm_batch_length,sequence_length)
                chunk_obs = obs[:,start:end]#O_{0:t} 
                chunk_obs_reconstruct = obs[:,start:end]#(O_{0:t},a_{-1:t-1}) -> O_{0:t}
                chunk_actions = actions[:,start:end]#a_{-1:t-1}
                reconstructed_obs ,next_latent_pred,hidden,hiddens = self.dynamics_model.predict(obs=chunk_obs,actions=chunk_actions,hidden=hidden)
                
                if self.features_to_predict is not None:
                    chunk_obs_reconstruct = chunk_obs_reconstruct[:,:,self.features_to_predict]
                else:
                    chunk_obs_reconstruct = chunk_obs_reconstruct[:,:,:self.dynamics_model.prediction_size]

                reconstruciton_loss = F.mse_loss(reconstructed_obs,chunk_obs_reconstruct)
                latent_dynamics_loss = F.mse_loss(next_latent_pred,hiddens.detach())
                representation_loss = F.mse_loss(next_latent_pred.detach(),hiddens)
                encoder_loss = reconstruciton_loss +0.2*latent_dynamics_loss+0.1*representation_loss

                #truncate the hidden state
                hidden = hidden.detach()
                # Backward pass and optimization
                self.optimizer.zero_grad()
                encoder_loss.backward()
                self.optimizer.step()

                #LOSS RUNNING average:
                running_reconstruciton_loss += reconstruciton_loss.item()
                running_latent_dynamics_loss += latent_dynamics_loss.item()
            
        
        #mean:
        running_reconstruciton_loss /= num_epochs
        running_latent_dynamics_loss /= num_epochs
        metrics = {
            "reconstruction_loss": running_reconstruciton_loss,
            "latent_dynamics_loss": running_latent_dynamics_loss
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
    



