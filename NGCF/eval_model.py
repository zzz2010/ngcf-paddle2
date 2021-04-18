'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import paddorch as torch
import paddorch.optim as optim
import sys
from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time
from visualdl import LogWriter
from paddle import fluid
if __name__ == '__main__':
    if  args.gpu_id<0:
        place = fluid.CPUPlace( )
    else:
        place= fluid.CUDAPlace(args.gpu_id)
    with fluid.dygraph.guard(place=place):
        args.device = torch.device('0')
        torch.manual_seed(0)

        plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

        args.node_dropout = eval(args.node_dropout)
        args.mess_dropout = eval(args.mess_dropout)

        model = NGCF(data_generator.n_users,
                    data_generator.n_items,
                    norm_adj,
                    args).to(args.device)
        model_fn = args.weights_path
        model.load_state_dict(torch.load(model_fn))
        print("loaded model file:", model_fn)



        t0 = time()
        """
        *********************************************************
        Test.
        """


    
        import paddle


        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time()



        perf_str = ' recall@20=%.5f,ndcg=%.5f ' \
                'precision@20=%.5f, hit@20=%.5f' % \
                (   ret['recall'][0],ret['ndcg'][0],
                    ret['precision'][0],   ret['hit_ratio'][0] )
        print(perf_str)


