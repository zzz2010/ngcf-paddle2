
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


if __name__ == '__main__':
  import torch
  from paddle import fluid
  args.node_dropout_flag=0
  input_weight_file = "../script_task_pretrained_models/ngcf-gowalla-10-64-1024-0.0001.pkl"
  torch.manual_seed(0)
  paddorch.manual_seed(0)
  plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
  users, pos_items, neg_items = data_generator.sample()
  args.node_dropout="[0]"
  args.mess_dropout="[0.0,0.0,0.0]"
  args.device ="cuda"
  args.node_dropout = eval(args.node_dropout)
  args.mess_dropout = eval(args.mess_dropout)
  paddle_model = NGCF.NGCF(data_generator.n_users,
               data_generator.n_items,
               norm_adj,
               args)

  torch_model = torch_NGCF.NGCF(data_generator.n_users,
               data_generator.n_items,
               norm_adj,
               args).to("cuda")
  torch_state_dict=torch.load(input_weight_file)
  torch_model.load_state_dict(torch_state_dict)

  paddle_state_dict = load_pytorch_pretrain_model(paddle_model, torch_state_dict)
  paddle_model.load_state_dict(paddle_state_dict)


  u_g_embeddings_torch, pos_i_g_embeddings_torch, neg_i_g_embeddings_torch = torch_model(users,
                                                                 pos_items,
                                                                 neg_items,
                                                                 drop_flag=args.node_dropout_flag)
  batch_loss_torch, _, _ = torch_model.create_bpr_loss(u_g_embeddings_torch,
                                                                    pos_i_g_embeddings_torch,
                                                                    neg_i_g_embeddings_torch)

  batch_loss_torch.backward()


  from time import time
  u_g_embeddings_paddle, pos_i_g_embeddings_paddle, neg_i_g_embeddings_paddle = paddle_model(users,
                                                                 pos_items,
                                                                 neg_items,
                                                                 drop_flag=args.node_dropout_flag)
  start = time()
  batch_loss_paddle, _, _ = paddle_model.create_bpr_loss(u_g_embeddings_paddle,
                                                                    pos_i_g_embeddings_paddle,
                                                                    neg_i_g_embeddings_paddle)
  print("forward time:", time() - start)
  print("forward output,max diff:", np.max(np.abs(u_g_embeddings_paddle.detach().numpy() - batch_loss_paddle.detach().numpy())))

  start = time()
  batch_loss_paddle.backward()
  print("backward time:", time() - start)
  print("backward output,max diff:",
        np.max(np.abs(paddle_model.embedding_dict['user_emb'].gradient()- torch_model.embedding_dict['user_emb'].grad.cpu().numpy())))
