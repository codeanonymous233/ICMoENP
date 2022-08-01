
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from image_utils.image_metadataset import *
from image_param_list import *
from meta_models import *
from meta_loss import mse_loss, mse_kl_loss, moe_mse_kl_loss, moe_mse_loss, moe_mse_cat_kl_loss



def meta_test(meta_net, eval_loader, num_c_points, order_pixels,
              check_lvm, whether_condition, 
              te_loss_fun, te_hard):
    
    meta_net.eval()
    epoch_test_mse = []
    
    with torch.no_grad():
        for batch_idx, (y_all, _) in enumerate(eval_loader):
            batch_size = y_all.shape[0]
            y_all = y_all.permute(0,2,3,1).contiguous().view(batch_size, -1, 3).cuda()

            N = num_c_points
            idx = get_context_idx(N, order_pixels=order_pixels) 
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
                    assert te_loss_fun == moe_mse_loss
                    b_avg_mse=te_loss_fun(y_pred,y_t,alpha,te_hard)
                else:
                    mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha = meta_net(x_c,y_c,x,y,x)
                    assert te_loss_fun == moe_mse_loss
                    b_avg_mse=te_loss_fun(y_pred,y,alpha,te_hard)                     
            
            elif check_lvm == 'MoE_CondNP' :
                if whether_condition:
                    mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha_post,alpha_prior = meta_net(x_c,y_c,x_t,y_t,x_pred=x_t,y_pred=None)
                    assert te_loss_fun == moe_mse_loss
                    b_avg_mse=te_loss_fun(y_pred,y_t,alpha_prior,te_hard)
                else:
                    mu_c,logvar_c,mu_t,logvar_t,y_pred,alpha_post,alpha_prior = meta_net(x_c,y_c,x,y,x_pred=x,y_pred=None)
                    assert te_loss_fun == moe_mse_loss
                    b_avg_mse=te_loss_fun(y_pred,y,alpha_prior,te_hard)                      
                
            elif check_lvm == 'NP' or check_lvm == 'AttnNP':
                mu_c,logvar_c,mu_t,logvar_t,y_pred = meta_net(x_c,y_c,x,y,x)
                assert te_loss_fun == mse_loss
                b_avg_mse = te_loss_fun(y_pred,y)   
                
            else:
                raise NotImplementedError()
            
            epoch_test_mse.append(b_avg_mse.detach().cpu().numpy())

    avg_te_mse = np.array(epoch_test_mse).sum() /len(eval_loader)
    avg_std=(np.array(epoch_test_mse)).std()
    print('====> Test set loss: {:.4f}'.format(avg_te_mse))
    print('====> Test set std: {:.4f}'.format(avg_std))

    return avg_te_mse    
    


train_loader,eval_loader=cifar10_metadataset()


train_model = 'MoE_NP'


##########################################################################################################################
    # pass arguments to models and load meta trained model
##########################################################################################################################

if train_model == 'MoE_NP':    
    args,random,device=params_moe_np()
    meta_net=MoE_NP
    check_lvm='MoE_NP'
    whether_condition=False
    te_loss_fun=moe_mse_loss
    te_hard=False
    meta_net=MoE_NP(args).cuda()  

elif train_model == 'MoE_CondNP':   
    args,random,device=params_moe_condnp()
    meta_net=MoE_CondNP
    check_lvm='MoE_CondNP'
    whether_condition=False
    te_loss_fun=moe_mse_loss
    te_hard=False
    meta_net=MoE_CondNP(args).cuda()      

elif train_model == 'NP': 
    args,random,device=params_np()
    meta_net=NP
    check_lvm='NP'
    whether_condition=False
    te_loss_fun=mse_loss
    te_hard=False
    meta_net=NP(args).cuda()
    
elif train_model == 'AttnNP': 
    args,random,device=params_attn_np()
    meta_net=AttnNP
    check_lvm='AttnNP'
    whether_condition=False
    te_loss_fun=mse_loss
    te_hard=False
    meta_net=AttnNP(args).cuda()

    
    
meta_net.load_state_dict(torch.load('./runs_results_image/MoE_NP/5/MoE_NP.pth'))



##########################################################################################################################
    # collect results
##########################################################################################################################


meta_test(meta_net=meta_net, eval_loader=eval_loader, num_c_points=10, order_pixels=False,
          check_lvm=check_lvm, whether_condition=whether_condition, 
          te_loss_fun=te_loss_fun, te_hard=te_hard)


'''
Collected Results for CIFAR10 are as follows:

MoE-CondNP-->
with 2 expert
#meta_net.load_state_dict(torch.load('./runs_results_image/MoE_CondNP/1/MoE_CondNP.pth'))
ordered case: number of context points-->avg mse(std)
10-->0.0534(0.0125)
100-->0.0482(0.0128)
200-->0.0458(0.0119)
500-->0.0338(0.0092)
800-->0.0212(0.0058)
1000-->0.0110(0.0026)
random cases: number of context points-->avg mse(std)
10-->0.0377(0.0108)
100-->0.0175(0.0043)
200-->0.0142(0.0034)
500-->0.0117(0.0028)
800-->0.0110(0.0026)
1000-->0.0107(0.0026)

'''



'''
meta_net.load_state_dict(torch.load('./runs_results_image/NP/1/NP.pth'))

NP-->
ordered case: number of context points-->avg mse(std)
10-->0.1087(0.0385)
100-->0.0860(0.0284)
200-->0.0676(0.0215)
500-->0.0427(0.0117)
800-->0.0297(0.0082)
1000-->0.0230(0.0056)
random case: number of context points-->avg mse(std)
10-->0.0464(0.0140)
100-->0.0255(0.0064)
200-->0.0240(0.0059)
500-->0.0231(0.0056)
800-->0.0229(0.0055)
1000-->0.0228(0.0055)

AttnNP-->
ordered case: number of context points-->avg mse(std)
10->0.0971(0.0318)
100->0.0600(0.0176)
200->0.0496(0.0139)
500->0.0299(0.0074)
800->0.0221(0.0056)
1000->0.0215(0.0054)
random cases: number of context points-->avg mse(std)
10-->0.0377(0.0111)
100-->0.0232(0.0059)
200-->0.0223(0.0056)
500-->0.0217(0.0055)
800-->0.0215(0.0055)
1000-->0.0215(0.0054)

'''


'''
MoE_NP with more experts, discrete posterior=discrete prior
random context [10, 200, 500, 800, 1000]

expert number 5
meta_net.load_state_dict(torch.load('./runs_results_image/MoE_NP/3/MoE_NP.pth'))
&0.0362 &0.0103 &0.0071 &0.0061 &0.0057

expert number 7
meta_net.load_state_dict(torch.load('./runs_results_image/MoE_NP/4/MoE_NP.pth'))
&0.0359 &0.0095 &0.0062 &0.0052 &0.0048

expert number 3
meta_net.load_state_dict(torch.load('./runs_results_image/MoE_NP/5/MoE_NP.pth'))
&0.0482 &0.0183 &0.0170 &0.0166 &0.0165

'''