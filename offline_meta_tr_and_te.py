

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from image_utils.image_metadataset import *
from meta_loss import *




##########################################################################################################################
    # Training image completion models in single task and multi-task settings
##########################################################################################################################
    
    
def train_meta_net(epoch, meta_net, cat_dim, net_optim, train_loader,
                   check_lvm, whether_condition, tr_loss_fun, tr_hard, beta0, beta1):
 
    meta_net.train()
    epoch_train_loss, epoch_train_mse = [], [] 
    
    for batch_idx, (y_all, _) in enumerate(train_loader):
        batch_size = y_all.shape[0]
        y_all = y_all.permute(0,2,3,1).contiguous().view(batch_size, -1, 3).cuda()
        
        N = random.randint(1, 1024)  
        idx = get_context_idx(N) 
        idx_list = idx.tolist()
        idx_all = np.arange(1024).tolist()
        x_c = idx_to_x(idx, batch_size)
        y_c = idx_to_y(idx, y_all)
        idx_all_tensor = torch.tensor(idx_all,dtype=torch.long).cuda()
        x = idx_to_x(idx_all_tensor, batch_size).cuda()
        y = idx_to_y(idx_all_tensor, y_all).cuda()
        
        pred_idx = torch.tensor(list(set(idx_all)-set(idx_list)), dtype=torch.long).cuda() 
        x_t = idx_to_x(pred_idx, batch_size).cuda() 
        y_t = idx_to_y(pred_idx, y_all).cuda() 
        
        net_optim.zero_grad()
        
        if check_lvm == 'MoE_NP' :
            if whether_condition:
                mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha = meta_net(x_c,y_c,x,y,x_t)
                if meta_net.info_bottleneck:
                    loss, b_avg_mse, kld, c_kld = tr_loss_fun(y_pred,y_t,mu_c,logvar_c,mu_t,logvar_t,
                                                              alpha,cat_dim,tr_hard,beta0,beta1)
                else:
                    loss, b_avg_mse, c_kld = tr_loss_fun(y_pred, y_t, alpha, cat_dim, tr_hard, beta1) 
            else:
                mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha = meta_net(x_c,y_c,x,y,x)
                if meta_net.info_bottleneck:
                    loss, b_avg_mse, kld, c_kld = tr_loss_fun(y_pred,y,mu_c,logvar_c,mu_t,logvar_t,
                                                              alpha,cat_dim,tr_hard,beta0,beta1)                 
                else:
                    loss, b_avg_mse, c_kld = tr_loss_fun(y_pred, y, alpha, cat_dim, tr_hard, beta1)                    
        
        elif check_lvm == 'MoE_CondNP' :
            if whether_condition:
                mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha_post,alpha_prior = meta_net(x_c,y_c,x,y,x_t,y_t)
                if meta_net.info_bottleneck:
                    loss, b_avg_mse, kld, c_kld = tr_loss_fun(y_pred,y_t,mu_c,logvar_c,mu_t,logvar_t,
                                                              alpha_post,alpha_prior,
                                                              tr_hard,beta0,beta1)
                else:
                    loss, b_avg_mse, c_kld = tr_loss_fun(y_pred, y_t, alpha_post, alpha_prior, 
                                                         tr_hard, beta1) 
            else:
                mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha_post,alpha_prior = meta_net(x_c,y_c,x,y,x,y)
                if meta_net.info_bottleneck:
                    loss, b_avg_mse, kld, c_kld = tr_loss_fun(y_pred,y,mu_c,logvar_c,mu_t,logvar_t,
                                                              alpha_post,alpha_prior,
                                                              tr_hard,beta0,beta1)                 
                else:
                    loss, b_avg_mse, c_kld = tr_loss_fun(y_pred, y, alpha_post,alpha_prior, 
                                                         tr_hard, beta1)
                
        elif check_lvm == 'NP' or check_lvm == 'AttnNP':
            mu_c,logvar_c,mu_t,logvar_t,y_pred = meta_net(x_c,y_c,x,y,x)
            loss, b_avg_mse, kld = tr_loss_fun(y_pred,y,mu_c,logvar_c,mu_t,logvar_t,beta0) 
            
        else:
            raise NotImplementedError()           
            
        loss.backward()
        net_optim.step()

        epoch_train_loss.append(loss.data)
        if check_lvm == 'CNP':
            epoch_train_mse.append(loss.data)
        else:
            epoch_train_mse.append(b_avg_mse.data)

    avg_tr_mse = np.array(epoch_train_mse).sum() / len(train_loader)

    return avg_tr_mse   


def eval_meta_net(epoch, meta_net, eval_loader, 
                  check_lvm, whether_condition, te_loss_fun, te_hard):
    
    meta_net.eval()
    epoch_test_mse = []
    
    with torch.no_grad():
        for batch_idx, (y_all, _) in enumerate(eval_loader):
            batch_size = y_all.shape[0]
            y_all = y_all.permute(0,2,3,1).contiguous().view(batch_size, -1, 3).cuda()

            #N = random.randint(1, 1024) 
            N = 500
            idx = get_context_idx(N, order_pixels=True)
            idx_list = idx.tolist()
            idx_all = np.arange(1024).tolist()
            x_c = idx_to_x(idx, batch_size)
            y_c = idx_to_y(idx, y_all)
            idx_all_tensor = torch.tensor(idx_all,dtype=torch.long).cuda()
            x = idx_to_x(idx_all_tensor, batch_size).cuda()
            y = idx_to_y(idx_all_tensor, y_all).cuda()
            
            pred_idx = torch.tensor(list(set(idx_all)-set(idx_list)), dtype=torch.long).cuda() 
            x_t = idx_to_x(pred_idx, batch_size).cuda() 
            y_t = idx_to_y(pred_idx, y_all).cuda() 
            
            if check_lvm == 'MoE_NP' :
                if whether_condition:
                    mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha = meta_net(x_c,y_c,x_t,y_t,x_t)
                    b_avg_mse=te_loss_fun(y_pred,y_t,alpha,te_hard)
                else:
                    mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha = meta_net(x_c,y_c,x,y,x)
                    b_avg_mse=te_loss_fun(y_pred,y,alpha,te_hard)                     
            
            elif check_lvm == 'MoE_CondNP' :
                if whether_condition:
                    mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha_post,alpha_prior = meta_net(x_c,y_c,x_t,y_t,x_pred=x_t,y_pred=None)
                    b_avg_mse=te_loss_fun(y_pred,y_t,alpha_prior,te_hard)
                else:
                    mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha_post,alpha_prior = meta_net(x_c,y_c,x,y,x_pred=x,y_pred=None)
                    b_avg_mse=te_loss_fun(y_pred,y,alpha_prior,te_hard)
                
            elif check_lvm == 'NP' or check_lvm == 'AttnNP':
                mu_c,logvar_c,mu_t,logvar_t,y_pred = meta_net(x_c,y_c,x,y,x)
                b_avg_mse = te_loss_fun(y_pred,y)   
                
            else:
                raise NotImplementedError()
            
            epoch_test_mse.append(b_avg_mse)

    avg_te_mse = np.array(epoch_test_mse).sum() /len(eval_loader)

    return avg_te_mse


def run_tr_te(args, meta_net, cat_dim, net_optim, 
              train_loader, eval_loader, check_lvm, whether_condition, 
              tr_loss_fun, te_loss_fun, tr_hard, te_hard, beta0, beta1, writer):
    
    meta_tr_results, meta_te_results = [], []
    for epoch in range(1, args.epochs + 1):
        avg_tr_mse = train_meta_net(epoch, meta_net, cat_dim, net_optim,
                                    train_loader, check_lvm, whether_condition, tr_loss_fun, tr_hard, beta0,
                                    beta1)
        avg_te_mse = eval_meta_net(epoch, meta_net, eval_loader, 
                                   check_lvm, whether_condition, te_loss_fun, te_hard)
        
        meta_tr_results.append(avg_tr_mse)
        meta_te_results.append(avg_te_mse)
        torch.save(meta_net.state_dict(),'./runs_results_image/'+check_lvm+'/'+str(writer)+'/'+check_lvm+'.pth')
        
    meta_tr_results, meta_te_results = np.array(meta_tr_results), np.array(meta_te_results)
    
    return meta_tr_results, meta_te_results


