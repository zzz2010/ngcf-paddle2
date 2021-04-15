
import paddorch
from paddorch.convert_pretrain_model import load_pytorch_pretrain_model
from glob import glob

from paddle import fluid
import os
import numpy as np
import torch_NGCF

import paddorch

import time
import NGCF

from os.path import join
import paddle
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time

import joblib
if __name__ == '__main__':



    args.node_dropout_flag = 0
    input_weight_file = "../script_task_pretrained_models/ngcf-gowalla-10-64-1024-0.0001.pkl"
    paddorch.manual_seed(0)
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = "[0]"
    args.mess_dropout = "[0.0,0.0,0.0]"
    args.device = "cuda"
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    paddle_model = NGCF.NGCF(data_generator.n_users,
                             data_generator.n_items,
                             norm_adj,
                             args)

    tf_numpy_data=joblib.load("tf_numpy_data.joblib")
    users=tf_numpy_data['users']
    pos_items=tf_numpy_data['pos_items']
    neg_items  =tf_numpy_data['neg_items']
    print("num user",len(users))
    name_change=dict()
    name_change["item_embedding"]="item_emb"
    name_change["user_embedding"] = "user_emb"


    def load_tf_pretrain_model(paddle_model, tf_state_dict):
        from collections import OrderedDict
        '''
        paddle_model: dygraph layer object
        pytorch_state_dict: pytorch state_dict, assume in CPU device
        '''

        paddle_weight = paddle_model.state_dict()
        print("paddle num_params:", len(paddle_weight))
        print("torch num_params:", len(tf_state_dict))
        new_weight_dict = OrderedDict()
        for key in  name_change:
            tf_state_dict[name_change[key]]=tf_state_dict[key]
        torch_key_list = []
        for key in tf_state_dict.keys():
            if key not in  paddle_weight.keys():
                continue
            torch_key_list.append(key)

        for  paddle_key in paddle_weight.keys():
            torch_key=paddle_key
            print(torch_key, paddle_key, tf_state_dict[torch_key].shape, paddle_weight[paddle_key].shape)
            if len(tf_state_dict[torch_key].shape) == 0:
                continue
            ##handle all FC weight cases
            if ("fc" in torch_key and "weight" in torch_key) or (
                    len(tf_state_dict[torch_key].shape) == 2 and tf_state_dict[torch_key].shape[0] ==
                    tf_state_dict[torch_key].shape[1]):
                new_weight_dict[paddle_key] = tf_state_dict[torch_key].T.astype("float32")
            elif int(paddle_weight[paddle_key].shape[-1]) == int(tf_state_dict[torch_key].shape[-1]):
                new_weight_dict[paddle_key] = tf_state_dict[torch_key].astype("float32")
            else:
                new_weight_dict[paddle_key] = tf_state_dict[torch_key].T.astype("float32")
        paddle_model.set_dict(new_weight_dict)
        return paddle_model.state_dict()


    paddle_state_dict = load_tf_pretrain_model(paddle_model, tf_numpy_data)
    paddle_model.load_state_dict(paddle_state_dict)

    u_g_embeddings_paddle, pos_i_g_embeddings_paddle, neg_i_g_embeddings_paddle = paddle_model(users,
                                                                                               pos_items,
                                                                                               neg_items,
                                                                                               drop_flag=args.node_dropout_flag)

    trim_dim=256
    u_g_embeddings_tf=  tf_numpy_data['u_g_embeddings'][users][:,:trim_dim]
    # print(u_g_embeddings_paddle )

    # print("tf",u_g_embeddings_tf)
    print("forward output,max diff:",
          np.max(np.abs(u_g_embeddings_paddle.detach().numpy()[:,:trim_dim] -u_g_embeddings_tf)))
    print("FINAL paddle:", paddorch.mean(u_g_embeddings_paddle))

    print("FINAL tf:", np.mean(u_g_embeddings_tf))


