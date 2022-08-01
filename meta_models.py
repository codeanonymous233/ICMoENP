
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from math import pi
from utils import get_act, gumbel_softmax, Multi_Attn, Context_Encoder, Softmax_Net


class NP(nn.Module):

    def __init__(self,args):
        super(NP,self).__init__()
        
        self.dim_x=args.dim_x
        self.dim_y=args.dim_y
        
        self.dim_h_lat=args.dim_h_lat 
        self.num_h_lat=args.num_h_lat 
        self.dim_lat=args.dim_lat 
        
        self.dim_h=args.dim_h 
        self.num_h=args.num_h
        self.act_type=args.act_type 
        self.amort_y=args.amort_y       

        self.emb_c_modules=[]
        self.emb_c_modules.append(nn.Linear(self.dim_x+self.dim_y,self.dim_h_lat))
        for i in range(self.num_h_lat):
            self.emb_c_modules.append(get_act(self.act_type))
            self.emb_c_modules.append(nn.Linear(self.dim_h_lat,self.dim_h_lat))
        self.emb_c_modules.append(get_act(self.act_type))
        self.context_net=nn.Sequential(*self.emb_c_modules)
        
        self.mu_net=nn.Linear(self.dim_h_lat, self.dim_lat) 
        self.logvar_net=nn.Linear(self.dim_h_lat, self.dim_lat) 
        
        self.dec_modules=[]
        self.dec_modules.append(nn.Linear(self.dim_x+self.dim_lat, self.dim_h))
        for i in range(self.num_h):
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_h))
        if self.amort_y:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, 2*self.dim_y)) 
        else:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_y))          
        self.dec_net=nn.Sequential(*self.dec_modules).cuda() 

    
    def get_context_idx(self,M):
        N = random.randint(1,M)
        idx = random.sample(range(0, M), N)
        idx = torch.tensor(idx).cuda()
        
        return idx
        
        
    def idx_to_data(self,data,sample_dim,idx):
        ind_data= torch.index_select(data, dim=sample_dim, index=idx)
        
        return ind_data
        
        
    def emb_aggregator(self,h_context,aggre_dim=1):
        h_aggre=torch.mean(h_context,dim=aggre_dim) 
        
        return h_aggre
    
    
    def reparameterization(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        
        return mu+eps*std

    
    def encoder(self,x_c,y_c,x_t,y_t):
        if self.training:
            memo_c,memo_t=torch.cat((x_c,y_c),dim=-1),torch.cat((x_t,y_t),dim=-1)
            memo_emb_c,memo_emb_t=self.context_net(memo_c),self.context_net(memo_t)
            
            h_c,h_t=self.emb_aggregator(memo_emb_c),self.emb_aggregator(memo_emb_t)
            mu_c,logvar_c=self.mu_net(h_c),self.logvar_net(h_c)
            mu_t,logvar_t=self.mu_net(h_t),self.logvar_net(h_t)
        else:
            memo_c=torch.cat((x_c,y_c),dim=-1)
            memo_emb_c=self.context_net(memo_c)
            
            h_c=self.emb_aggregator(memo_emb_c)
            mu_c,logvar_c=self.mu_net(h_c),self.logvar_net(h_c)
            mu_t,logvar_t=0,0
        
        return mu_c,logvar_c,mu_t,logvar_t
            
        
    def forward(self,x_c,y_c,x_t,y_t,x_pred,whether_acrobot=False,whether_image=True):
        mu_c,logvar_c,mu_t,logvar_t=self.encoder(x_c, y_c, x_t, y_t)
        
        if self.training:
            z_g=self.reparameterization(mu_t, logvar_t) 
        else:
            z_g=self.reparameterization(mu_c, logvar_c) 
        
        z_g_unsq=z_g.unsqueeze(1).expand(-1,x_pred.size(1),-1) 
        output=self.dec_net(torch.cat((x_pred,z_g_unsq),dim=-1)) 
        
        if whether_acrobot:
            if self.amort_y:
                output=torch.cat((torch.cos(output[...,0:1]),torch.sin(output[...,0:1]),
                                  torch.cos(output[...,1:2]),torch.sin(output[...,1:2]),
                                  4*pi*torch.tanh(output[...,2:3]),9*pi*torch.tanh(output[...,3:4]),output[...,4:]),axis=-1)                
                y_mean,y_std=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
                return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std
            else:
                y_pred=torch.cat((torch.cos(output[...,0:1]),torch.sin(output[...,0:1]),
                                  torch.cos(output[...,1:2]),torch.sin(output[...,1:2]),
                                  4*pi*torch.tanh(output[...,2:3]),9*pi*torch.tanh(output[...,3:4])),axis=-1)
                return mu_c,logvar_c,mu_t,logvar_t,y_pred 
        elif whether_image:
            if self.amort_y: 
                y_mean,y_std=F.sigmoid(output[...,:self.dim_y]),F.softplus(output[...,self.dim_y:])
                return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std
            else:
                y_pred=F.sigmoid(output)
                return mu_c,logvar_c,mu_t,logvar_t,y_pred         
        else:
            if self.amort_y:
                y_mean,y_std=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
                return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std
            else:
                y_pred=output
                return mu_c,logvar_c,mu_t,logvar_t,y_pred
    
    
    

class AttnNP(nn.Module):
    def __init__(self,args):
        super(AttnNP,self).__init__()
        
        self.dim_x=args.dim_x
        self.dim_y=args.dim_y
        
        self.dim_h_lat=args.dim_h_lat 
        self.num_h_lat=args.num_h_lat 
        self.dim_lat=args.dim_lat 
        self.num_head=args.num_head 
        self.dim_emb_x=args.dim_emb_x     
        
        self.dim_h=args.dim_h 
        self.num_h=args.num_h 
        self.act_type=args.act_type 
        self.amort_y=args.amort_y     

        self.emb_c_modules=[]
        self.emb_c_modules.append(nn.Linear(self.dim_x+self.dim_y,self.dim_h_lat))
        for i in range(self.num_h_lat):
            self.emb_c_modules.append(get_act(self.act_type))
            self.emb_c_modules.append(nn.Linear(self.dim_h_lat,self.dim_h_lat))
        self.emb_c_modules.append(get_act(self.act_type))
        self.context_net=nn.Sequential(*self.emb_c_modules)
        
        self.mu_net=nn.Linear(self.dim_h_lat, self.dim_lat) 
        self.logvar_net=nn.Linear(self.dim_h_lat, self.dim_lat) 
        
        self.dot_attn=Multi_Attn(self.dim_x, self.dim_x+self.dim_y, self.dim_emb_x, 
                                 self.dim_lat, self.num_head) 
        
        self.dec_modules=[]
        self.dec_modules.append(nn.Linear(self.dim_x+self.dim_lat+self.num_head*self.dim_lat, self.dim_h))        
        for i in range(self.num_h):
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_h))
        if self.amort_y:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, 2*self.dim_y)) 
        else:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_y))         
        self.dec_net=nn.Sequential(*self.dec_modules)
    
    
    
    def get_context_idx(self,M):
        N = random.randint(1,M)
        idx = random.sample(range(0, M), N)
        idx = torch.tensor(idx).cuda()
        
        return idx
        
        
    def idx_to_data(self,data,sample_dim,idx):
        ind_data= torch.index_select(data, dim=sample_dim, index=idx)
        
        return ind_data
        
        
    def emb_aggregator(self,h_context,aggre_dim=1):
        h_aggre=torch.mean(h_context,dim=aggre_dim) 
        
        return h_aggre
    
    
    def reparameterization(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        
        return mu+eps*std

    
    def encoder(self,x_c,y_c,x_t,y_t):
        if self.training:
            memo_c,memo_t=torch.cat((x_c,y_c),dim=-1),torch.cat((x_t,y_t),dim=-1)
            memo_emb_c,memo_emb_t=self.context_net(memo_c),self.context_net(memo_t)
            
            h_c,h_t=self.emb_aggregator(memo_emb_c),self.emb_aggregator(memo_emb_t)
            mu_c,logvar_c=self.mu_net(h_c),self.logvar_net(h_c)
            mu_t,logvar_t=self.mu_net(h_t),self.logvar_net(h_t)
            
        else:
            memo_c=torch.cat((x_c,y_c),dim=-1)
            memo_emb_c=self.context_net(memo_c)
            
            h_c=self.emb_aggregator(memo_emb_c)
            mu_c,logvar_c=self.mu_net(h_c),self.logvar_net(h_c)
            mu_t,logvar_t=0,0            
        
        return mu_c,logvar_c,mu_t,logvar_t
            
        
        
    def forward(self,x_c,y_c,x_t,y_t,x_pred,whether_acrobot=False,whether_image=True):
        initial_value=torch.cat((x_t,y_t),dim=-1)
        dot_attn_v,dot_attn_weight=self.dot_attn(x_t,x_pred,initial_value)
        
        mu_c,logvar_c,mu_t,logvar_t=self.encoder(x_c, y_c, x_t, y_t)
        
        if self.training:
            z_g=self.reparameterization(mu_t, logvar_t) 
        else:
            z_g=self.reparameterization(mu_c, logvar_c) 
            
        z_g_unsq=z_g.unsqueeze(1).expand(-1,x_pred.size(1),-1)
        c_merg=torch.cat((dot_attn_v,z_g_unsq),dim=-1)
        
        output=self.dec_net(torch.cat((x_pred,c_merg),dim=-1))
        if whether_acrobot:
            if self.amort_y:
                y_mean,y_std=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
                return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std
            else:
                y_pred=torch.cat((torch.cos(output[...,0:1]),torch.sin(output[...,0:1]),
                                      torch.cos(output[...,1:2]),torch.sin(output[...,1:2]),
                                      4*pi*torch.tanh(output[...,2:3]),9*pi*torch.tanh(output[...,3:4])),axis=-1)
                return mu_c,logvar_c,mu_t,logvar_t,y_pred 
        elif whether_image:
            if self.amort_y: 
                y_mean,y_std=F.sigmoid(output[...,:self.dim_y]),F.softplus(output[...,self.dim_y:])
                return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std
            else:
                y_pred=F.sigmoid(output)
                return mu_c,logvar_c,mu_t,logvar_t,y_pred         
        else:
            if self.amort_y:
                y_mean,y_std=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
                return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std
            else:
                y_pred=output
                return mu_c,logvar_c,mu_t,logvar_t,y_pred       
        
        
class MoE_NP(nn.Module):
    def __init__(self,args):
        super(MoE_NP,self).__init__()
        
        self.dim_x=args.dim_x
        self.dim_y=args.dim_y
        
        self.dim_h_lat=args.dim_h_lat 
        self.num_h_lat=args.num_h_lat 
        self.dim_lat=args.dim_lat 
        self.num_lat=args.num_lat 
        self.experts_in_gates=args.experts_in_gates 
        self.num_logit_layers=args.num_logit_layers 
        self.dim_logit_h=args.dim_logit_h 
        self.temperature=args.temperature 
        self.gumbel_max=args.gumbel_max 
        self.info_bottleneck=args.info_bottleneck
        
        self.dim_h=args.dim_h 
        self.num_h=args.num_h 
        self.act_type=args.act_type 
        self.amort_y=args.amort_y         

        self.expert_modules=nn.ModuleList([Context_Encoder(self.dim_x+self.dim_y, self.dim_h_lat, 
                                                           self.act_type, self.num_h_lat, self.dim_lat).cuda() 
                                           for i in range(self.num_lat)])
        
        if self.experts_in_gates:
            self.logit_net=Softmax_Net(self.dim_x+self.dim_lat, self.experts_in_gates,
                                       self.dim_logit_h, self.num_logit_layers,
                                       self.num_lat)
        else:
            self.logit_net=Softmax_Net(self.dim_x, self.experts_in_gates,
                                       self.dim_logit_h, self.num_logit_layers,
                                       self.num_lat)
        
        self.dec_modules=[]
        self.dec_modules.append(nn.Linear(self.dim_x+self.dim_lat, self.dim_h))
        for i in range(self.num_h):
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_h))
        if self.amort_y:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, 2*self.dim_y)) 
        else:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_y))        
        self.dec_net=nn.Sequential(*self.dec_modules)
    
    
    
    def get_context_idx(self,M):
        N = random.randint(1,M)
        idx = random.sample(range(0, M), N)
        idx = torch.tensor(idx).cuda()
        
        return idx
        
        
    def idx_to_data(self,data,sample_dim,idx):
        ind_data= torch.index_select(data, dim=sample_dim, index=idx)
        
        return ind_data
    
    
    def reparameterization(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        
        return mu+eps*std

    
    def encoder(self,x_c,y_c,x_t,y_t):
        if self.training:
            memo_c,memo_t=torch.cat((x_c,y_c),dim=-1),torch.cat((x_t,y_t),dim=-1)
            emb_c_list_mu=torch.cat([expert_module(memo_c)[0].unsqueeze(0) for expert_module 
                                     in self.expert_modules]) 
            emb_c_list_logvar=torch.cat([expert_module(memo_c)[1].unsqueeze(0) for expert_module 
                                     in self.expert_modules]) 
            emb_t_list_mu=torch.cat([expert_module(memo_t)[0].unsqueeze(0) for expert_module 
                                     in self.expert_modules])            
            emb_t_list_logvar=torch.cat([expert_module(memo_t)[1].unsqueeze(0) for expert_module 
                                     in self.expert_modules]) 
            
            emb_c_list_mu,emb_c_list_logvar=emb_c_list_mu.permute(1,0,2),emb_c_list_logvar.permute(1,0,2) 
            emb_t_list_mu,emb_t_list_logvar=emb_t_list_mu.permute(1,0,2),emb_t_list_logvar.permute(1,0,2) 
            
        else:
            memo_c=torch.cat((x_c,y_c),dim=-1)
            emb_c_list_mu=torch.cat([expert_module(memo_c)[0].unsqueeze(0) for expert_module 
                                    in self.expert_modules]) 
            emb_c_list_logvar=torch.cat([expert_module(memo_c)[1].unsqueeze(0) for expert_module 
                                        in self.expert_modules]) 
            emb_c_list_mu,emb_c_list_logvar=emb_c_list_mu.permute(1,0,2),emb_c_list_logvar.permute(1,0,2)
            
            emb_t_list_mu,emb_t_list_logvar=0,0
        
        return emb_c_list_mu,emb_c_list_logvar,emb_t_list_mu,emb_t_list_logvar
            
        
    def forward(self,x_c,y_c,x_t,y_t,x_pred,whether_acrobot=False,whether_image=True):
        mu_c,logvar_c,mu_t,logvar_t=self.encoder(x_c, y_c, x_t, y_t)
        
        if self.training:
            if self.info_bottleneck:
                z_experts=self.reparameterization(mu_t, logvar_t) 
            else:
                z_experts=mu_c
        else:
            if self.info_bottleneck:
                z_experts=self.reparameterization(mu_c, logvar_c)
            else:
                z_experts=mu_c            
        z_experts_unsq=z_experts.unsqueeze(1).expand(-1,x_pred.size()[1],-1,-1) 
        
        x_exp=x_pred.unsqueeze(2).expand(-1,-1,z_experts_unsq.size()[2],-1) 
        
        if self.experts_in_gates:
            x_z=torch.cat((x_exp,z_experts_unsq),dim=-1)
            alpha,y_hard=self.logit_net(x_z=x_z,temperature=self.temperature,gumbel_max=self.gumbel_max) 
        else:
            alpha,y_hard=self.logit_net(x_z=x_exp,temperature=self.temperature,gumbel_max=self.gumbel_max) 
        
        output=self.dec_net(torch.cat((x_exp,z_experts_unsq),dim=-1)) 
        
        if whether_acrobot:
            if self.amort_y:
                y_mean,y_std=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
                return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std
            else:
                y_pred=torch.cat((torch.cos(output[...,0:1]),torch.sin(output[...,0:1]),
                                      torch.cos(output[...,1:2]),torch.sin(output[...,1:2]),
                                      4*pi*torch.tanh(output[...,2:3]),9*pi*torch.tanh(output[...,3:4])),axis=-1)
                return mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha 
        elif whether_image:
            if self.amort_y: 
                y_mean,y_std=F.sigmoid(output[...,:self.dim_y]),F.softplus(output[...,self.dim_y:])
                return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std
            else:
                y_pred=F.sigmoid(output)
                return mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha         
        else:
            if self.amort_y:
                y_mean,y_std=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
                return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std,alpha
            else:
                y_pred=output
                return mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha 
            
    
    def conditional_predict(self,x_c,y_c,x_pred,whether_acrobot=False,whether_image=True):
        with torch.no_grad():
            mu_c,logvar_c,mu_t,logvar_t=self.encoder(x_c, y_c, x_c, y_c)
            z_experts=self.reparameterization(mu_c, logvar_c) 
            z_experts_unsq=z_experts.unsqueeze(1).expand(-1,x_pred.size()[1],-1,-1) 
        
            x_exp=x_pred.unsqueeze(2).expand(-1,-1,z_experts_unsq.size()[2],-1) 
            
            if self.experts_in_gates:
                x_z=torch.cat((x_exp,z_experts_unsq),dim=-1)
                alpha,y_hard=self.logit_net(x_z=x_z,temperature=self.temperature,gumbel_max=False) 
            else:
                alpha,y_hard=self.logit_net(x_z=x_exp,temperature=self.temperature,gumbel_max=False) 
                
            weighted_expert=torch.mul(y_hard,z_experts_unsq) 
            selected_z=torch.sum(weighted_expert,dim=-2)            
        
            output=self.dec_net(torch.cat((x_pred,selected_z),dim=-1)) 
        
            if whether_acrobot:
                if self.amort_y:
                    y_mean,y_std=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
                    return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std
                else:
                    y_pred=torch.cat((torch.cos(output[...,0:1]),torch.sin(output[...,0:1]),
                                      torch.cos(output[...,1:2]),torch.sin(output[...,1:2]),
                                      4*pi*torch.tanh(output[...,2:3]),9*pi*torch.tanh(output[...,3:4])),axis=-1)
                    return y_pred 
            elif whether_image:
                if self.amort_y: 
                    y_mean,y_std=F.sigmoid(output[...,:self.dim_y]),F.softplus(output[...,self.dim_y:])
                    return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std
                else:
                    y_pred=F.sigmoid(output)
                    return y_pred            
            else:
                if self.amort_y:
                    y_mean,y_std=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
                    return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std,alpha
                else:
                    y_pred=output
                    return y_pred            
            


class MoE_CondNP(nn.Module):
    def __init__(self,args):
        super(MoE_CondNP,self).__init__()
        
        self.dim_x=args.dim_x
        self.dim_y=args.dim_y
        
        self.dim_h_lat=args.dim_h_lat 
        self.num_h_lat=args.num_h_lat 
        self.dim_lat=args.dim_lat 
        self.num_lat=args.num_lat 
        self.experts_in_gates=args.experts_in_gates 
        self.num_logit_layers=args.num_logit_layers 
        self.dim_logit_h=args.dim_logit_h 
        self.temperature=args.temperature 
        self.gumbel_max=args.gumbel_max 
        self.info_bottleneck=args.info_bottleneck 
        
        self.dim_h=args.dim_h 
        self.num_h=args.num_h 
        self.act_type=args.act_type 
        self.amort_y=args.amort_y         

        self.expert_modules=nn.ModuleList([Context_Encoder(self.dim_x+self.dim_y, self.dim_h_lat, 
                                                           self.act_type, self.num_h_lat, self.dim_lat).cuda() 
                                           for i in range(self.num_lat)])
        
        if self.experts_in_gates:
            self.logit_net_post=Softmax_Net(self.dim_x+self.dim_y+self.dim_lat, self.experts_in_gates,
                                            self.dim_logit_h, self.num_logit_layers,
                                            self.num_lat)
            self.logit_net_prior=Softmax_Net(self.dim_x+self.dim_lat, self.experts_in_gates,
                                            self.dim_logit_h, self.num_logit_layers,
                                            self.num_lat)            
        else:
            self.logit_net_post=Softmax_Net(self.dim_x+self.dim_y, self.experts_in_gates,
                                            self.dim_logit_h, self.num_logit_layers,
                                            self.num_lat)
            self.logit_net_prior=Softmax_Net(self.dim_x, self.experts_in_gates,
                                             self.dim_logit_h, self.num_logit_layers,
                                             self.num_lat)            
        
        self.dec_modules=[]
        self.dec_modules.append(nn.Linear(self.dim_x+self.dim_lat, self.dim_h))
        for i in range(self.num_h):
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_h))
        if self.amort_y:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, 2*self.dim_y)) 
        else:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_y))           
        self.dec_net=nn.Sequential(*self.dec_modules)
    
    
    
    def get_context_idx(self,M):
        N = random.randint(1,M)
        idx = random.sample(range(0, M), N)
        idx = torch.tensor(idx).cuda()
        
        return idx
        
        
    def idx_to_data(self,data,sample_dim,idx):
        ind_data= torch.index_select(data, dim=sample_dim, index=idx)
        
        return ind_data
    
    
    def reparameterization(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        
        return mu+eps*std

    
    def encoder(self,x_c,y_c,x_t,y_t):
        if self.training:
            memo_c,memo_t=torch.cat((x_c,y_c),dim=-1),torch.cat((x_t,y_t),dim=-1)
            emb_c_list_mu=torch.cat([expert_module(memo_c)[0].unsqueeze(0) for expert_module 
                                     in self.expert_modules]) 
            emb_c_list_logvar=torch.cat([expert_module(memo_c)[1].unsqueeze(0) for expert_module 
                                     in self.expert_modules]) 
            emb_t_list_mu=torch.cat([expert_module(memo_t)[0].unsqueeze(0) for expert_module 
                                     in self.expert_modules])             
            emb_t_list_logvar=torch.cat([expert_module(memo_t)[1].unsqueeze(0) for expert_module 
                                     in self.expert_modules]) 
            
            emb_c_list_mu,emb_c_list_logvar=emb_c_list_mu.permute(1,0,2),emb_c_list_logvar.permute(1,0,2) 
            emb_t_list_mu,emb_t_list_logvar=emb_t_list_mu.permute(1,0,2),emb_t_list_logvar.permute(1,0,2) 
            
        else:
            memo_c=torch.cat((x_c,y_c),dim=-1)
            emb_c_list_mu=torch.cat([expert_module(memo_c)[0].unsqueeze(0) for expert_module 
                                    in self.expert_modules])
            emb_c_list_logvar=torch.cat([expert_module(memo_c)[1].unsqueeze(0) for expert_module 
                                        in self.expert_modules]) 
            emb_c_list_mu,emb_c_list_logvar=emb_c_list_mu.permute(1,0,2),emb_c_list_logvar.permute(1,0,2)
            
            emb_t_list_mu,emb_t_list_logvar=0,0
        
        return emb_c_list_mu,emb_c_list_logvar,emb_t_list_mu,emb_t_list_logvar
            
        
    def forward(self,x_c,y_c,x_t,y_t,x_pred,y_pred,whether_acrobot=False,whether_image=True):
        mu_c,logvar_c,mu_t,logvar_t=self.encoder(x_c, y_c, x_t, y_t)
        
        if self.training:
            if self.info_bottleneck:
                z_experts=self.reparameterization(mu_t,logvar_t) 
            else:
                z_experts=mu_c
        else:
            assert y_pred==None
            if self.info_bottleneck:
                z_experts=self.reparameterization(mu_c,logvar_c)
            else:
                z_experts=mu_c 
                
        z_experts_unsq=z_experts.unsqueeze(1).expand(-1,x_pred.size()[1],-1,-1) 
        
        x_exp=x_pred.unsqueeze(2).expand(-1,-1,z_experts_unsq.size()[2],-1) 
        
        if self.training:
            y_exp=y_pred.unsqueeze(2).expand(-1,-1,z_experts_unsq.size()[2],-1) 
            if self.experts_in_gates:
                xz_exp=torch.cat((x_exp,z_experts_unsq),dim=-1)
                xy_exp=torch.cat((x_exp,y_exp),dim=-1)
                xyz_exp=torch.cat((xy_exp,z_experts_unsq),dim=-1)
                alpha_post,y_hard_post=self.logit_net_post(x_z=xyz_exp,temperature=self.temperature,gumbel_max=self.gumbel_max) 
                alpha_prior,y_hard_prior=self.logit_net_prior(x_z=xz_exp,temperature=self.temperature,gumbel_max=self.gumbel_max) 
            else:
                xy_exp=torch.cat((x_exp,y_exp),dim=-1)
                alpha_post,y_hard_post=self.logit_net_post(x_z=xy_exp,temperature=self.temperature,gumbel_max=self.gumbel_max) 
                alpha_prior,y_hard_prior=self.logit_net_prior(x_z=x_exp,temperature=self.temperature,gumbel_max=self.gumbel_max)                 
        else:
            if self.experts_in_gates:
                xz_exp=torch.cat((x_exp,z_experts_unsq),dim=-1)
                alpha_post,y_hard_post=0,0
                alpha_prior,y_hard_prior=self.logit_net_prior(x_z=xz_exp,temperature=self.temperature,gumbel_max=self.gumbel_max) 
            else:
                alpha_post,y_hard_post=0,0
                alpha_prior,y_hard_prior=self.logit_net_prior(x_z=x_exp,temperature=self.temperature,gumbel_max=self.gumbel_max)              
        output=self.dec_net(torch.cat((x_exp,z_experts_unsq),dim=-1)) 
        
        if whether_acrobot:
            if self.amort_y:
                y_mean,y_std=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
                return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std,alpha_post,alpha_prior
            else:
                y_pred=torch.cat((torch.cos(output[...,0:1]),torch.sin(output[...,0:1]),
                                      torch.cos(output[...,1:2]),torch.sin(output[...,1:2]),
                                      4*pi*torch.tanh(output[...,2:3]),9*pi*torch.tanh(output[...,3:4])),axis=-1)
                return mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha_post,alpha_prior 
        elif whether_image:
            if self.amort_y: 
                y_mean,y_std=F.sigmoid(output[...,:self.dim_y]),F.softplus(output[...,self.dim_y:])
                return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std,alpha_post,alpha_prior
            else:
                y_pred=F.sigmoid(output)
                return mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha_post,alpha_prior         
        else:
            if self.amort_y:
                y_mean,y_std=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
                return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std,alpha_post,alpha_prior
            else:
                y_pred=output
                return mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha_post,alpha_prior 
            
    
    def conditional_predict(self,x_c,y_c,x_pred,whether_acrobot=False,whether_image=True):
        with torch.no_grad():
            mu_c,logvar_c,mu_t,logvar_t=self.encoder(x_c, y_c, x_c, y_c)
            z_experts=self.reparameterization(mu_c, logvar_c) 
            z_experts_unsq=z_experts.unsqueeze(1).expand(-1,x_pred.size()[1],-1,-1) 
        
            x_exp=x_pred.unsqueeze(2).expand(-1,-1,z_experts_unsq.size()[2],-1) 
            
            if self.experts_in_gates:
                x_z=torch.cat((x_exp,z_experts_unsq),dim=-1)
                alpha,y_hard=self.logit_net(x_z=x_z,temperature=self.temperature,gumbel_max=False) 
            else:
                alpha,y_hard=self.logit_net(x_z=x_exp,temperature=self.temperature,gumbel_max=False) 
                
            weighted_expert=torch.mul(y_hard,z_experts_unsq) 
            selected_z=torch.sum(weighted_expert,dim=-2)             
        
            output=self.dec_net(torch.cat((x_pred,selected_z),dim=-1)) 
        
            if whether_acrobot:
                if self.amort_y:
                    y_mean,y_std=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
                    return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std
                else:
                    y_pred=torch.cat((torch.cos(output[...,0:1]),torch.sin(output[...,0:1]),
                                      torch.cos(output[...,1:2]),torch.sin(output[...,1:2]),
                                      4*pi*torch.tanh(output[...,2:3]),9*pi*torch.tanh(output[...,3:4])),axis=-1)
                    return y_pred 
            elif whether_image:
                if self.amort_y: 
                    y_mean,y_std=F.sigmoid(output[...,:self.dim_y]),F.softplus(output[...,self.dim_y:])
                    return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std
                else:
                    y_pred=F.sigmoid(output)
                    return y_pred            
            else:
                if self.amort_y:
                    y_mean,y_std=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
                    return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_std,alpha
                else:
                    y_pred=output
                    return y_pred            
            

        
