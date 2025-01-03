import torch
import torch.nn as nn
from torch.nn import functional as F
from visualization import visualize_attention

from models.GPT import GPT


################################################################################################################
# class MeDT
#
# We make embeddings of each input in the sequence. We then add position embeddings and feed embeddings
# to GPT transformer
# 
# Three different settings, 
#       BC: Behaviour Cloning
#       DT: Decision Transformer
#       MeDT: Medical Decision Transformer
#       
################################################################################################################
class MeDT(GPT):

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None, saps=None, divSaps=None, traj_len=None, is_visual=False):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)
        
        state_embeddings = self.state_emb(states.type(torch.float32))
        

        if actions is not None and self.model_type == 'DT': 

            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        
            my_pos_emb = torch.zeros(timesteps.shape[0] ,timesteps.shape[1]*3, self.config.n_embd).to(self.device)
            my_pos_emb[:,0::3,:] = timesteps
            my_pos_emb[:,1::3,:] = timesteps
            my_pos_emb[:,2::3,:] = timesteps   

        elif actions is None and self.model_type == 'DT': # only happens at very first timestep of evaluation

            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]

            my_pos_emb = torch.zeros(timesteps.shape[0] ,timesteps.shape[1]*2, self.config.n_embd).to(self.device)
            my_pos_emb[:,0::2,:] = timesteps
            my_pos_emb[:,1::2,:] = timesteps
                    

        elif actions is not None and self.model_type == 'MeDT': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
            
            card_embeddings = self.card_emb(divSaps[:,:,0].unsqueeze(-1).type(torch.float32))
            resp_embeddings = self.resp_emb(divSaps[:,:,1].unsqueeze(-1).type(torch.float32))
            neur_embeddings = self.neur_emb(divSaps[:,:,2].unsqueeze(-1).type(torch.float32))
            ren_embeddings = self.ren_emb(divSaps[:,:,3].unsqueeze(-1).type(torch.float32))
            hep_embeddings = self.hep_emb(divSaps[:,:,4].unsqueeze(-1).type(torch.float32))
            haem_embeddings = self.haem_emb(divSaps[:,:,5].unsqueeze(-1).type(torch.float32))
            oth_embeddings = self.oth_emb(divSaps[:,:,6].unsqueeze(-1).type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*10 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::10,:] = rtg_embeddings
            token_embeddings[:,1::10,:] = card_embeddings
            token_embeddings[:,2::10,:] = resp_embeddings
            token_embeddings[:,3::10,:] = neur_embeddings
            token_embeddings[:,4::10,:] = ren_embeddings
            token_embeddings[:,5::10,:] = hep_embeddings
            token_embeddings[:,6::10,:] = haem_embeddings  
            token_embeddings[:,7::10,:] = oth_embeddings                       
            token_embeddings[:,8::10,:] = state_embeddings
            token_embeddings[:,9::10,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        
            my_pos_emb = torch.zeros(timesteps.shape[0] ,states.shape[1]*10 - int(targets is None), self.config.n_embd).to(self.device)
            my_pos_emb[:,0::10,:] = timesteps
            my_pos_emb[:,1::10,:] = timesteps
            my_pos_emb[:,2::10,:] = timesteps                   
            my_pos_emb[:,3::10,:] = timesteps
            my_pos_emb[:,4::10,:] = timesteps
            my_pos_emb[:,5::10,:] = timesteps  
            my_pos_emb[:,6::10,:] = timesteps                   
            my_pos_emb[:,7::10,:] = timesteps
            my_pos_emb[:,8::10,:] = timesteps
            my_pos_emb[:,9::10,:] = timesteps[:,-states.shape[1] + int(targets is None):,:]


        elif actions is None and self.model_type == 'MeDT': # only happens at very first timestep of evaluation

            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            card_embeddings = self.card_emb(divSaps[:,:,0].unsqueeze(-1).type(torch.float32))
            resp_embeddings = self.resp_emb(divSaps[:,:,1].unsqueeze(-1).type(torch.float32))
            neur_embeddings = self.neur_emb(divSaps[:,:,2].unsqueeze(-1).type(torch.float32))
            ren_embeddings = self.ren_emb(divSaps[:,:,3].unsqueeze(-1).type(torch.float32))
            hep_embeddings = self.hep_emb(divSaps[:,:,4].unsqueeze(-1).type(torch.float32))
            haem_embeddings = self.haem_emb(divSaps[:,:,5].unsqueeze(-1).type(torch.float32))
            oth_embeddings = self.oth_emb(divSaps[:,:,6].unsqueeze(-1).type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*9 , self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::9,:] = rtg_embeddings
            token_embeddings[:,1::9,:] = card_embeddings
            token_embeddings[:,2::9,:] = resp_embeddings
            token_embeddings[:,3::9,:] = neur_embeddings
            token_embeddings[:,4::9,:] = ren_embeddings
            token_embeddings[:,5::9,:] = hep_embeddings
            token_embeddings[:,6::9,:] = haem_embeddings  
            token_embeddings[:,7::9,:] = oth_embeddings
            token_embeddings[:,8::9,:] = state_embeddings

            my_pos_emb = torch.zeros(timesteps.shape[0] ,timesteps.shape[1]*9, self.config.n_embd).to(self.device)
            my_pos_emb[:,0::9,:] = timesteps
            my_pos_emb[:,1::9,:] = timesteps
            my_pos_emb[:,2::9,:] = timesteps                   
            my_pos_emb[:,3::9,:] = timesteps
            my_pos_emb[:,4::9,:] = timesteps
            my_pos_emb[:,5::9,:] = timesteps  
            my_pos_emb[:,6::9,:] = timesteps                   
            my_pos_emb[:,7::9,:] = timesteps
            my_pos_emb[:,8::9,:] = timesteps
	

        elif actions is not None and self.model_type == 'BC':
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]

            my_pos_emb = torch.zeros(timesteps.shape[0] ,timesteps.shape[1]*2, self.config.n_embd).to(self.device)
            my_pos_emb[:,0::2,:] = timesteps
            my_pos_emb[:,1::2,:] = timesteps

        elif actions is None and self.model_type == 'BC': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings

            my_pos_emb = torch.zeros(token_embeddings.shape[0] ,token_embeddings.shape[1], token_embeddings.shape[2]).to(self.device)
            my_pos_emb[:,:,:] = timesteps


        else:
            raise NotImplementedError()

        # Add position embeddings
        position_embeddings =  self.pos_emb[:, :token_embeddings.shape[1], :] + my_pos_emb[:, :token_embeddings.shape[1], :]
        x = self.drop(token_embeddings) + position_embeddings
            
        for idx, block in enumerate(self.blocks):
            x, attn_score = block(x)
            if is_visual:
                visualize_attention(attn_score, idx)
                self.attn_score = attn_score

        x = self.ln_f(x)
        logits = self.head(x)

        action_loss = None

        if actions is not None and self.model_type == 'DT':
            logits = logits[:, 1::3, :] 
        elif actions is None and self.model_type == 'DT':
            logits = logits[:, 1:, :]            
        elif actions is not None and self.model_type == 'MeDT':
            logits = logits[:, 8::10, :]
        elif actions is None and self.model_type == 'MeDT':
            logits = logits[:, 8:, :]            
        elif actions is not None and self.model_type == 'BC':
            logits = logits[:, ::2, :] 
        elif actions is None and self.model_type == 'BC':
            logits = logits
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss        
        if targets is not None:
            action_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    
        return logits, action_loss, self.attn_score