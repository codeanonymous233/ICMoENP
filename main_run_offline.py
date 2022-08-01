

import random
import numpy as np
import torch
import torch.optim as optim


from image_utils.image_metadataset import cifar10_metadataset
from offline_meta_tr_and_te import run_tr_te
from image_param_list import *
from meta_models import *
from meta_loss import *



def compute_result(param_list, meta_net, cat_dim, 
                   check_lvm, whether_condition, 
                   tr_loss_fun, te_loss_fun, 
                   tr_hard, te_hard, beta0, beta1, writer=1):

    train_loader,eval_loader=cifar10_metadataset()

    args,random,device=param_list()
    random.seed(args.seed)
    
    meta_net=meta_net(args).to(device)
    
    optimizer=optim.Adam(meta_net.parameters(), lr=5e-4)
    
    
    meta_tr_results, meta_te_results = run_tr_te(args=args, meta_net=meta_net, cat_dim=cat_dim, 
                                                 net_optim=optimizer, train_loader=train_loader, eval_loader=eval_loader, 
                                                 check_lvm=check_lvm, whether_condition=whether_condition, 
                                                 tr_loss_fun=tr_loss_fun, te_loss_fun=te_loss_fun, 
                                                 tr_hard=tr_hard, te_hard=te_hard, 
                                                 beta0=beta0, beta1=beta1, writer=writer)

    np.save('./runs_results_image/'+check_lvm+'/'+str(writer)+'/tr_loss_list', meta_tr_results)
    np.save('./runs_results_image/'+check_lvm+'/'+str(writer)+'/te_loss_list', meta_te_results)
    
    

train_model = 'MoE_NP'

if train_model == 'MoE_NP':    
    param_list=params_moe_np
    meta_net=MoE_NP
    cat_dim=2
    check_lvm='MoE_NP'
    whether_condition=False
    tr_loss_fun=moe_mse_cat_kl_loss
    te_loss_fun=moe_mse_loss
    tr_hard=False
    te_hard=False
    beta0=1.0
    beta1=1.0
if train_model == 'MoE_CondNP':    
    param_list=params_moe_condnp
    meta_net=MoE_CondNP
    cat_dim=2
    check_lvm='MoE_CondNP'
    whether_condition=False
    tr_loss_fun=moe_mse_cat_condkl_loss
    te_loss_fun=moe_mse_loss
    tr_hard=False
    te_hard=False
    beta0=1.0
    beta1=1.0
elif train_model == 'NP': 
    param_list=params_np
    meta_net=NP
    cat_dim=1
    check_lvm='NP'
    whether_condition=False
    tr_loss_fun=mse_kl_loss
    te_loss_fun=mse_loss
    tr_hard=False
    te_hard=False
    beta0=1.0
    beta1=1.0
elif train_model == 'AttnNP':
    param_list=params_attn_np
    meta_net=AttnNP
    cat_dim=1
    check_lvm='AttnNP'
    whether_condition=False
    tr_loss_fun=mse_kl_loss
    te_loss_fun=mse_loss
    tr_hard=False
    te_hard=False
    beta0=1.0
    beta1=1.0



compute_result(param_list, meta_net, cat_dim, 
               check_lvm, whether_condition, 
               tr_loss_fun, te_loss_fun, 
               tr_hard, te_hard, beta0, beta1, writer=1)
