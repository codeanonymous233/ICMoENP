
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from torch.distributions import MultivariateNormal


def kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    var_p = torch.exp(logvar_p)
    kld_sum = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / var_p \
        - 1.0 + logvar_p - logvar_q
    
    if mu_q.dim()==2:
        kl_div = (0.5 * torch.mean(kld_sum,dim=0)).sum() 
    elif mu_q.dim()==3:
        kl_div = (0.5 * torch.mean(kld_sum,dim=0)).sum() 
    elif mu_q.dim()==4:
        kl_div = (0.5 * torch.mean(kld_sum,dim=[0,1,2])).sum() 
        
    return kl_div


def kl_div_prior(z_mu,logvar):
    kld_sum=-0.5*torch.sum(1+logvar-z_mu.pow(2)-logvar.exp())
    if z_mu.dim()==2:
        kld=(torch.mean(kld_sum,dim=0)).sum() 
    elif z_mu.dim()==3:
        kld=(torch.mean(kld_sum,dim=[0])).sum() 
    elif z_mu.dim()==4:
        kld=(torch.mean(kld_sum,dim=[0,1,2])).sum() 
        
    return kld


def kl_div_cat(softmax_y, cat_dim):
    log_ratio = torch.log(softmax_y * cat_dim + 1e-20) 
    KLD = torch.sum(softmax_y * log_ratio, dim=-1).mean()    
    
    return KLD


def kl_div_cat_post_prior(softmax_y_post, softmax_y_prior):
    log_ratio = torch.log(softmax_y_post/(softmax_y_prior + 1e-20) + 1e-20) 
    KLD = torch.sum(softmax_y_post * log_ratio, dim=-1).mean()    
    
    return KLD

    
  

def mse_loss(y_pred,y):
    loss=F.mse_loss(y_pred,y)
    
    return loss


def contras_mse_loss(y_pred,y,z_g1,z_g2, beta=1.0):
    b_avg_mse = mse_loss(y_pred, y)
    loss = b_avg_mse + beta*contras_loss(z_g1, z_g2)
    
    return loss, b_avg_mse


def moe_mse_loss(y_pred,y,alpha,hard=True):
    y_expand=y.unsqueeze(2).expand(-1,-1,y_pred.size()[2],-1)
    
    if hard == False:
        weighted_y_pred=torch.mul(y_pred,alpha) 
        weighted_y_pred=torch.sum(weighted_y_pred,dim=-2) 
        loss=F.mse_loss(weighted_y_pred,y)
        
    else:
        shape = alpha.size()
        _, ind = alpha.max(dim=-1)
        alpha_hard = torch.zeros_like(alpha).view(-1, shape[-1])
        alpha_hard.scatter_(1, ind.view(-1, 1), 1)
        alpha_hard = alpha_hard.view(*shape)
        alpha_hard = (alpha_hard - alpha).detach() + alpha
        
        sq_loss=torch.mul(y_pred-y_expand,y_pred-y_expand)
        sum_loss=torch.mul(sq_loss,alpha_hard) 
        moe_mse=torch.sum(sum_loss,dim=-2)
        loss=moe_mse.mean()        
    
    return loss


def mse_kl_loss(y_pred,y,mu_q,logvar_q,mu_p,logvar_p,beta0):
    b_avg_mse=F.mse_loss(y_pred, y)
    kld=kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p)
    loss = b_avg_mse + (1.0/y.size()[1]) * beta0 * kld
    
    return loss,b_avg_mse,kld


def moe_mse_kl_loss(y_pred, y, mu_q, logvar_q, mu_p, logvar_p, 
                    alpha, cat_dim, hard=True, beta0=1.0, beta1=1.0):
    b_avg_mse=moe_mse_loss(y_pred,y,alpha,hard=hard)
    g_kld=kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p)
    c_kld=kl_div_cat(alpha, cat_dim)
    loss = b_avg_mse + (1.0/y.size()[1]) * beta0 * g_kld + beta1 * c_kld
    
    return loss,b_avg_mse,g_kld,c_kld


def moe_mse_condkl_loss(y_pred, y, mu_q, logvar_q, mu_p, logvar_p, 
                        alpha_post, alpha_prior, cat_dim, hard=True, beta0=1.0, beta1=1.0):
    b_avg_mse=moe_mse_loss(y_pred,y,alpha_post,hard=hard)
    g_kld=kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p)
    c_kld=kl_div_cat_post_prior(alpha_post, alpha_prior)
    loss = b_avg_mse + (1.0/y.size()[1]) * beta0 * g_kld + beta1 * c_kld
    
    return loss,b_avg_mse,g_kld,c_kld


def moe_mse_cat_kl_loss(y_pred, y, alpha, cat_dim, hard=True, beta1=1.0):
    b_avg_mse=moe_mse_loss(y_pred,y,alpha,hard=hard)
    c_kld=kl_div_cat(alpha, cat_dim)
    loss = b_avg_mse + beta1 * c_kld
    
    return loss,b_avg_mse,c_kld


def moe_mse_cat_condkl_loss(y_pred, y, alpha_post, alpha_prior, hard=True, beta1=1.0):
    b_avg_mse=moe_mse_loss(y_pred,y,alpha_post,hard=hard)
    c_kld=kl_div_cat_post_prior(alpha_post, alpha_prior)
    loss = b_avg_mse + beta1 * c_kld
    
    return loss,b_avg_mse,c_kld


def mse_kl_zeroprior_loss(y_pred, y, z_mu, logvar, beta=1.0):
    b_avg_mse=F.mse_loss(y_pred, y)
    kld=kl_div_prior(z_mu, logvar)
    loss = b_avg_mse + beta * kld
    
    return loss, b_avg_mse, kld



def nll_loss(y_mean,y_std,y,keep_dim=False):
    if y.size()[-1] == 1:
        pred_dist = Normal(y_mean, y_std)
    else:
        pred_dist = MultivariateNormal(y_mean, scale_tril=y_std.diag_embed())
    
    if keep_dim:
        log_likelihood = pred_dist.log_prob(y) 
    else:
        log_likelihood = (pred_dist.log_prob(y)).mean() 
    
    return -log_likelihood


def contras_nll_loss(y_mean,y_std,y,z_g1,z_g2, beta=1.0):
    b_avg_nll = nll_loss(y_mean, y_std, y)
    loss = b_avg_nll + beta*contras_loss(z_g1, z_g2)
    
    return loss, b_avg_nll


def nll_kl_loss(y_mean,y_std,y,mu_q,logvar_q,mu_p,logvar_p,beta0=1.0):
    b_avg_nll = nll_loss(y_mean,y_std,y)
    kld = kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p)
    loss = b_avg_nll + (1.0/y.size()[1]) * beta0 * kld
    
    return loss, b_avg_nll, kld
    

def moe_nll_loss(y_mean,y_std,y,alpha,hard):
    y_expand=y.unsqueeze(2).expand(-1,-1,y_mean.size()[2],-1)
    
    if hard == False:
        nll_list=[(nll_loss(y_mean[:,:,i,:], y_std[:,:,i,:], y_expand[:,:,i,:],keep_dim=True)).unsqueeze(2) for i in range(y_mean.size()[2])]
        nll_tensor=(torch.cat(nll_list,dim=2)).unsqueeze(-1) 
        sum_loss=torch.mul(nll_tensor,alpha) 

    else:
        shape = alpha.size()
        _, ind = alpha.max(dim=-1)
        alpha_hard = torch.zeros_like(alpha).view(-1, shape[-1])
        alpha_hard.scatter_(1, ind.view(-1, 1), 1)
        alpha_hard = alpha_hard.view(*shape)
        alpha_hard = (alpha_hard - alpha).detach() + alpha 
        
        nll_list=[(nll_loss(y_mean[:,:,i,:], y_std[:,:,i,:], y_expand[:,:,i,:])).unsqueeze(2) for i in range(y_mean.size()[2])]
        nll_tensor=(torch.cat(nll_list,dim=2)).unsqueeze(-1) 
        sum_loss=torch.mul(nll_tensor,alpha_hard) 
        
    moe_nll=torch.sum(sum_loss,dim=-2)
    loss=moe_nll.mean()
    
    return loss
    
    
def moe_nll_kl_loss(y_mean, y_std, y, mu_q, logvar_q, mu_p, logvar_p, 
                    alpha, cat_dim, hard=True, beta0=1.0, beta1=1.0):
    b_avg_nll=moe_nll_loss(y_mean,y_std,y,alpha,hard)
    g_kld=kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p)
    c_kld=kl_div_cat(alpha, cat_dim)
    loss = b_avg_nll + (1.0/y.size()[1]) * beta0 * g_kld + beta1 * c_kld
    
    return loss,b_avg_nll,g_kld,c_kld


def moe_nll_condkl_loss(y_mean, y_std, y, mu_q, logvar_q, mu_p, logvar_p, 
                        alpha_post, alpha_prior, hard=True, beta0=1.0, beta1=1.0):
    b_avg_nll=moe_nll_loss(y_mean,y_std,y,alpha,hard)
    g_kld=kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p)
    c_kld=kl_div_cat_post_prior(alpha_post, alpha_prior)
    loss = b_avg_nll + (1.0/y.size()[1]) * beta0 * g_kld + beta1 * c_kld
    
    return loss,b_avg_nll,g_kld,c_kld


def moe_nll_cat_kl_loss(y_mean,y_std,y,alpha,hard,beta1=1.0):
    b_avg_nll=moe_nll_loss(y_mean,y_std,y,alpha,hard)
    c_kld=kl_div_cat(alpha, cat_dim)
    loss = b_avg_nll + beta1 * c_kld
    
    return loss,b_avg_nll,c_kld


def moe_nll_cat_condkl_loss(y_mean,y_std,y,alpha_post, alpha_prior,hard,beta1=1.0):
    b_avg_nll=moe_nll_loss(y_mean,y_std,y,alpha_post,hard)
    c_kld=kl_div_cat_post_prior(alpha_post, alpha_prior)
    loss = b_avg_nll + beta1 * c_kld
    
    return loss,b_avg_nll,c_kld


def nll_kl_zeroprior_loss(y_mean, y_std, y, z_mu, logvar, beta=1.0):
    b_avg_nll = nll_loss(y_mean,y_std,y)
    b_avg_nll = b_avg_nll.mean()
    kld=kl_div_prior(z_mu, logvar)
    loss = b_avg_nll + beta * kld
    
    return loss, b_avg_nll, kld

 

    






