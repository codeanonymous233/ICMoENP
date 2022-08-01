import argparse
import random
import torch

epochs=60


def params_np():
    
    parser_np = argparse.ArgumentParser(description='np')
    parser_np.add_argument('--no-cuda', action='store_true', default=False,
                           help='disables CUDA training')
    parser_np.add_argument('--seed', type=int, default=1, metavar='S',
                           help='random seed (default: 1)')
    parser_np.add_argument('--log-interval', type=int, default=1, metavar='N',
                           help='how many batches to wait before logging training status')
    parser_np.add_argument('--batch_size', type=int, default=8, metavar='N',
                           help='input batch size for training')
    parser_np.add_argument('--epochs', type=int, default=epochs, metavar='N',
                           help='number of epochs to train')
    
    parser_np.add_argument('--dim_x', type=int, default=2, metavar='N',
                           help='dimension of input')
    parser_np.add_argument('--dim_y', type=int, default=3, metavar='N',
                           help='dimension of output')
    
    parser_np.add_argument('--dim_h_lat', type=int, default=128, metavar='N',
                           help='dim of hidden units for encoders')
    parser_np.add_argument('--num_h_lat', type=int, default=3, metavar='N',
                           help='num of layers for encoders')
    parser_np.add_argument('--dim_lat', type=int, default=128, metavar='N',
                           help='dimension of z, the global latent variable') 
    
    parser_np.add_argument('--dim_h', type=int, default=128, metavar='N',
                           help='dim of hidden units for decoders')   
    parser_np.add_argument('--num_h', type=int, default=5, metavar='N') 
    parser_np.add_argument('--act_type', type=str, default='ReLU', metavar='N')   
    parser_np.add_argument('--amort_y', type=bool, default=False, metavar='N')
    
    args = parser_np.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    return args,random,device


def params_attn_np():
    
    parser_attn_np = argparse.ArgumentParser(description='attn_np')
    parser_attn_np.add_argument('--no-cuda', action='store_true', default=False,
                                help='disables CUDA training')
    parser_attn_np.add_argument('--seed', type=int, default=1, metavar='S',
                                help='random seed (default: 1)')
    parser_attn_np.add_argument('--log-interval', type=int, default=1, metavar='N',
                                help='how many batches to wait before logging training status')
    parser_attn_np.add_argument('--batch_size', type=int, default=8, metavar='N',
                                help='input batch size for training')
    parser_attn_np.add_argument('--epochs', type=int, default=epochs, metavar='N',
                                help='number of epochs to train')
    
    parser_attn_np.add_argument('--dim_x', type=int, default=2, metavar='N',
                                help='dimension of input')
    parser_attn_np.add_argument('--dim_y', type=int, default=3, metavar='N',
                                help='dimension of output')
    
    parser_attn_np.add_argument('--dim_h_lat', type=int, default=32, metavar='N',
                                help='dim of hidden units for encoders')
    parser_attn_np.add_argument('--num_h_lat', type=int, default=2, metavar='N',
                                help='num of layers for encoders')
    parser_attn_np.add_argument('--dim_lat', type=int, default=32, metavar='N',
                                help='dimension of z, the global latent variable')
    parser_attn_np.add_argument('--num_head', type=int, default=2, metavar='N',
                                help='num of heads for attention networks') 
    parser_attn_np.add_argument('--dim_emb_x', type=int, default=32, metavar='N',
                                help='num of heads for attention networks')    
    
    parser_attn_np.add_argument('--dim_h', type=int, default=128, metavar='N')   
    parser_attn_np.add_argument('--num_h', type=int, default=5, metavar='N') 
    parser_attn_np.add_argument('--act_type', type=str, default='ReLU', metavar='N')   
    parser_attn_np.add_argument('--amort_y', type=bool, default=False, metavar='N')
    
    args = parser_attn_np.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    return args,random,device




def params_moe_np():
    
    parser_moe_np = argparse.ArgumentParser(description='moe_nps')
    parser_moe_np.add_argument('--no-cuda', action='store_true', default=False,
                               help='disables CUDA training')
    parser_moe_np.add_argument('--seed', type=int, default=1, metavar='S',
                               help='random seed (default: 1)')
    parser_moe_np.add_argument('--log-interval', type=int, default=1, metavar='N',
                               help='how many batches to wait before logging training status')
    parser_moe_np.add_argument('--batch_size', type=int, default=32, metavar='N',
                               help='input batch size for training')
    parser_moe_np.add_argument('--epochs', type=int, default=epochs, metavar='N',
                               help='number of epochs to train') 
    parser_moe_np.add_argument('--dim_x', type=int, default=2, metavar='N',
                               help='dimension of input')
    parser_moe_np.add_argument('--dim_y', type=int, default=3, metavar='N',
                               help='dimension of output')
    
    parser_moe_np.add_argument('--dim_h_lat', type=int, default=128, metavar='N',
                               help='dim of hidden units for encoders')
    parser_moe_np.add_argument('--num_h_lat', type=int, default=3, metavar='N',
                               help='num of layers for encoders')
    parser_moe_np.add_argument('--dim_lat', type=int, default=128, metavar='N',
                               help='dimension of z, the global latent variable')
    
    parser_moe_np.add_argument('--num_lat', type=int, default=2, metavar='N',
                               help='num of latent variables')
    parser_moe_np.add_argument('--experts_in_gates', type=bool, default=False, metavar='N',
                               help='whether experts as input for encoders')
    parser_moe_np.add_argument('--dim_logit_h', type=int, default=32, metavar='N',
                               help='dim of hidden units for encoders') 
    parser_moe_np.add_argument('--num_logit_layers', type=int, default=2, metavar='N',
                               help='dim of hidden units for encoders')    
    parser_moe_np.add_argument('--temperature', type=int, default=0.1, metavar='N') 
    parser_moe_np.add_argument('--hard', type=bool, default=False, metavar='N')
    parser_moe_np.add_argument('--gumbel_max', type=bool, default=True, metavar='N')
    parser_moe_np.add_argument('--info_bottleneck', type=bool, default=False, metavar='N')    
    
    parser_moe_np.add_argument('--dim_h', type=int, default=128, metavar='N',
                               help='dim of hidden units for decoders')   
    parser_moe_np.add_argument('--num_h', type=int, default=5, metavar='N',
                               help='num of layers for decoders') 
    parser_moe_np.add_argument('--act_type', type=str, default='LeakyReLU', metavar='N',
                               help='type of activation units')   
    parser_moe_np.add_argument('--amort_y', type=bool, default=False, metavar='N')
    
    args = parser_moe_np.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    return args,random,device




def params_moe_condnp():
    
    parser_moe_condnp = argparse.ArgumentParser(description='moe_condnps')
    parser_moe_condnp.add_argument('--no-cuda', action='store_true', default=False,
                               help='disables CUDA training')
    parser_moe_condnp.add_argument('--seed', type=int, default=1, metavar='S',
                               help='random seed (default: 1)')
    parser_moe_condnp.add_argument('--log-interval', type=int, default=1, metavar='N',
                               help='how many batches to wait before logging training status')
    parser_moe_condnp.add_argument('--batch_size', type=int, default=8, metavar='N',
                               help='input batch size for training')
    parser_moe_condnp.add_argument('--epochs', type=int, default=epochs, metavar='N',
                               help='number of epochs to train')
    parser_moe_condnp.add_argument('--dim_x', type=int, default=2, metavar='N',
                               help='dimension of input')
    parser_moe_condnp.add_argument('--dim_y', type=int, default=3, metavar='N',
                               help='dimension of output')
    
    parser_moe_condnp.add_argument('--dim_h_lat', type=int, default=128, metavar='N',
                               help='dim of hidden units for encoders')
    parser_moe_condnp.add_argument('--num_h_lat', type=int, default=3, metavar='N',
                               help='num of layers for encoders')
    parser_moe_condnp.add_argument('--dim_lat', type=int, default=128, metavar='N',
                               help='dimension of z, the global latent variable')
    
    parser_moe_condnp.add_argument('--num_lat', type=int, default=2, metavar='N',
                               help='num of latent variables')
    parser_moe_condnp.add_argument('--experts_in_gates', type=bool, default=False, metavar='N',
                               help='whether experts as input for encoders')
    parser_moe_condnp.add_argument('--dim_logit_h', type=int, default=32, metavar='N',
                               help='dim of hidden units for encoders') 
    parser_moe_condnp.add_argument('--num_logit_layers', type=int, default=2, metavar='N',
                               help='dim of hidden units for encoders')    
    parser_moe_condnp.add_argument('--temperature', type=int, default=0.1, metavar='N') 
    parser_moe_condnp.add_argument('--hard', type=bool, default=False, metavar='N')
    parser_moe_condnp.add_argument('--gumbel_max', type=bool, default=True, metavar='N')
    parser_moe_condnp.add_argument('--info_bottleneck', type=bool, default=False, metavar='N')    
    
    parser_moe_condnp.add_argument('--dim_h', type=int, default=128, metavar='N')   
    parser_moe_condnp.add_argument('--num_h', type=int, default=5, metavar='N') 
    parser_moe_condnp.add_argument('--act_type', type=str, default='ReLU', metavar='N')   
    parser_moe_condnp.add_argument('--amort_y', type=bool, default=False, metavar='N')
    
    args = parser_moe_condnp.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    return args,random,device

