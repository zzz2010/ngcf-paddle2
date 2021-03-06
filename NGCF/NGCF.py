'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import paddorch as torch
import paddorch.nn as nn
import paddorch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size

        self.norm_adj = norm_adj

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()
        for name, param in self.embedding_dict.items():
            self.add_parameter( name, param) 

        for name, param in self.weight_dict.items():
            self.add_parameter( name, param) 

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
        self.sparse_norm_adj.stop_gradient=True

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_
        from collections import OrderedDict
        embedding_dict=OrderedDict()
        embedding_dict['item_emb']=nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)) )
        embedding_dict['user_emb']=nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size)) )


        # embedding_dict =  {
        #     'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
        #                                          self.emb_size)) ),
        #     'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
        #                                          self.emb_size)) )
        # }

        weight_dict = OrderedDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = torch.Tensor(x._indices())
        v = torch.Tensor(x._values())

        i = i[:, dropout_mask]
  
        v = v[dropout_mask]
        
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
 
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size
        #tensorflow code regloss is always 0, so not include here
        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, drop_flag=True):
        users=torch.LongTensor(users)
   
        pos_items=torch.LongTensor(pos_items)
        
        if len(neg_items)>0:
            neg_items=torch.LongTensor(neg_items)

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                     self.sparse_norm_adj._nnz()  ) if drop_flag else self.sparse_norm_adj
 
        A_hat.stop_gradient=True
        ##tensorflow by default will split the sparse matrix to 100 per chunck

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]


        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # transformed sum messages of neighbors.
            sum_embeddings = nn.LeakyReLU(negative_slope=0.2)(torch.matmul(side_embeddings,   self.weight_dict['W_gc_%d' % k] ,transpose_y=True )\
                                             + self.weight_dict['b_gc_%d' % k])


            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            ##move LeakyReLU up to be consistent with tensorflow code
            bi_embeddings = nn.LeakyReLU(negative_slope=0.2)(torch.matmul(bi_embeddings,  self.weight_dict['W_bi_%d' % k] ,transpose_y=True  )\
                                            + self.weight_dict['b_bi_%d' % k])

            # non-linear activation.


            ego_embeddings = sum_embeddings + bi_embeddings
            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user ]
        i_g_embeddings = all_embeddings[self.n_user: ]


        """
        *********************************************************
        look up.
        """
        u_g_embeddings = u_g_embeddings[users ]
        pos_i_g_embeddings = i_g_embeddings[pos_items ]
        if len(neg_items)>0:
            neg_i_g_embeddings = i_g_embeddings[neg_items ]
        else:
            neg_i_g_embeddings=None

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
