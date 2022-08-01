
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

def get_act(act_type):
    if act_type=='ReLU':
        return nn.ReLU()
    elif act_type=='LeakyReLU':
        return nn.LeakyReLU()
    elif act_type=='ELU':
        return nn.ELU()
    elif act_type=='Sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError('Invalid argument for act_type')
    

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
       

    
class Multi_Attn(nn.Module):
    '''
    Multi-head attention and cross attention implementation.
    '''
    def __init__(self,dim_k_in,dim_v_in,dim_k,dim_v,num_head):
        super(Multi_Attn,self).__init__()
    
        self.dim_k_in=dim_k_in 
        self.dim_v_in=dim_v_in
        self.dim_k=dim_k 
        self.dim_v=dim_v 
        self.num_head=num_head 

        self.fc_k=nn.Linear(self.dim_k_in, self.num_head*self.dim_k,bias=False) 
        self.fc_q=nn.Linear(self.dim_k_in, self.num_head*self.dim_k,bias=False) 
        self.fc_v=nn.Linear(self.dim_v_in, self.num_head*self.dim_v,bias=False)         
    
    
    def emb_aggregator(self,h_context,aggre_dim):
        h_aggre=torch.mean(h_context,dim=aggre_dim)
        
        return h_aggre
    
    
    def forward(self,key,query,value):
        assert key.dim() == 3
        
        len_k, len_q, len_v = key.size(1), query.size(1), value.size(1)
           
        k=self.fc_k(key).view(key.size(0),len_k,self.num_head,self.dim_k)
        q=self.fc_q(query).view(key.size(0),len_q,self.num_head,self.dim_k)
        v=self.fc_v(value).view(key.size(0),len_v,self.num_head,self.dim_v)
        
        k,q,v=k.transpose(1,2),q.transpose(1,2),v.transpose(1,2)
        
        attn=torch.matmul(q/(self.dim_k)**0.5,k.transpose(2,3))
        attn=F.softmax(attn,dim=-1)
        
        attn_sq=attn.unsqueeze(-1) 
        v_sq_exp=v.unsqueeze(2).expand(-1,-1,len_q,-1,-1) 
        multi_attn_v=v_sq_exp.mul(attn_sq) 
        emb_attn_v=self.emb_aggregator(multi_attn_v,aggre_dim=3) 
        attn_v=emb_attn_v.transpose(1,2).contiguous().view(key.size(0),len_q,-1) 
            
        return attn_v,attn
    
    

class Context_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, act_type, num_layers, output_size):
        super(Context_Encoder, self).__init__()
        
        self.emb_c_modules=[]
        self.emb_c_modules.append(nn.Linear(input_size,hidden_size))
        for i in range(num_layers):
            self.emb_c_modules.append(get_act(act_type))
            self.emb_c_modules.append(nn.Linear(hidden_size,hidden_size))
        self.emb_c_modules.append(get_act(act_type))
        self.context_net=nn.Sequential(*self.emb_c_modules)
        
        self.mu_net=nn.Linear(hidden_size, output_size) 
        self.logvar_net=nn.Linear(hidden_size, output_size) 
    
    def forward(self,x,mean_dim=1):
        out=self.context_net(x)
        out=torch.mean(out,dim=mean_dim)
        mu, logvar=self.mu_net(out),self.logvar_net(out)
        
        return (mu,logvar)    
        


class Softmax_Net(nn.Module):
    def __init__(self,
                 dim_xz,
                 experts_in_gates,
                 dim_logit_h,
                 num_logit_layers,
                 num_experts):
        super().__init__()
        self.dim_xz=dim_xz
        self.experts_in_gates=experts_in_gates
        self.dim_logit_h=dim_logit_h
        self.num_logit_layers=num_logit_layers
        self.num_experts=num_experts

        self.logit_modules=[]
        if self.experts_in_gates:
            self.logit_modules.append(nn.Linear(self.dim_xz, self.dim_logit_h))
            for i in range(self.num_logit_layers):
                self.logit_modules.append(nn.ReLU())
                self.logit_modules.append(nn.Linear(self.dim_logit_h, self.dim_logit_h))
            self.logit_modules.append(nn.ReLU())
            self.logit_modules.append(nn.Linear(self.dim_logit_h, 1))
        else:
            self.logit_modules.append(nn.Linear(self.dim_xz, self.dim_logit_h))
            for i in range(self.num_logit_layers):
                self.logit_modules.append(nn.ReLU())
                self.logit_modules.append(nn.Linear(self.dim_logit_h, self.dim_logit_h))
            self.logit_modules.append(nn.ReLU())
            self.logit_modules.append(nn.Linear(self.dim_logit_h, self.num_experts))            
        self.logit_net=nn.Sequential(*self.logit_modules)

    def forward(self,x_z,temperature,gumbel_max=False,return_label=False):
        if self.experts_in_gates:
            logit_output=self.logit_net(x_z)
        else:
            x_z=torch.mean(x_z,dim=-2)
            logit_output=self.logit_net(x_z) 

        if not self.experts_in_gates:
            logit_output=logit_output.unsqueeze(-1) 

        if gumbel_max:
            logit_output=logit_output+sample_gumbel(logit_output.size())

        softmax_y=F.softmax(logit_output/temperature,dim=-2)

        softmax_y=softmax_y.squeeze(-1) 
        shape=softmax_y.size()
        _,ind=softmax_y.max(dim=-1) 
        y_hard=torch.zeros_like(softmax_y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard=y_hard.view(*shape) 
    
        y_hard=(y_hard-softmax_y).detach()+softmax_y
    
        softmax_y,y_hard=softmax_y.unsqueeze(-1),y_hard.unsqueeze(-1)
    
        return softmax_y, y_hard                
        

def sample_gumbel(shape,eps=1e-20,use_cuda=True):
    U=torch.rand(shape)
    if use_cuda:
        U=U.cuda()
        
    return -torch.log(-torch.log(U+eps)+eps)
    
    
def gumbel_softmax_sample(logits,temperature):
    y=logits+sample_gumbel(logits.size())
    
    return F.softmax(y/temperature,dim=-1)
    
    
def gumbel_softmax(logits,temperature,hard=False):
    y=gumbel_softmax_sample(logits,temperature)
    
    if not hard:
        return y

    shape=y.size()
    _,ind=y.max(dim=-1) 
    y_hard=torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard=y_hard.view(*shape) 
    
    y_hard=(y_hard-y).detach()+y
    
    return y_hard  



def softmax_to_hard_y(logits):
    logits=logits.squeeze(-1)
    shape=logits.size()
    _,ind=logits.max(dim=-1) 
    y_hard=torch.zeros_like(logits).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard=y_hard.view(*shape) 

    y_hard=(y_hard-logits).detach()+logits

    y_hard=y_hard.unsqueeze(-1) 

    return y_hard     
    





