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
        model_file_prefix = f"ngcf-{args.dataset}-{len(args.layer_size)}-{args.embed_size}-{args.batch_size}-{args.lr}"
        if int(args.pretrain)!=0:
            import glob
            import os

            list_of_files = glob.glob(args.weights_path+"/%s*pdparams"%model_file_prefix) # * means all if need specific format then *.csv
            if len(list_of_files)>0:
                model_fn = max(list_of_files, key=os.path.getctime)
                model.load_state_dict(torch.load(model_fn))
                print("loaded model file:",model_fn)

        t0 = time()
        """
        *********************************************************
        Train.
        """
        test_log=LogWriter(logdir='log/%s_log' % ( args.dataset))
        cur_best_pre_0, stopping_step = 0, 0
    
        import paddle
        lr_scheduler= paddle.optimizer.lr.CosineAnnealingDecay(learning_rate= args.lr, T_max=100, eta_min=0.0001,verbose=True) 
        optimizer = optim.Adam(model.parameters(), lr=lr_scheduler)
        # optimizer = optim.Adam(model.parameters(), lr= args.lr)

        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        import tqdm
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss = 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            pbar=tqdm.trange(n_batch)
            for idx in pbar:
                t1_pre = time()
                users, pos_items, neg_items = data_generator.sample()
    
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                            pos_items,
                                                                            neg_items,
                                                                            drop_flag=args.node_dropout_flag)

                batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                                pos_i_g_embeddings,
                                                                                neg_i_g_embeddings)

                t1_end=time()
                # print(' backward time:' ,time()-t1_end)
                optimizer.zero_grad()
                batch_loss.backward()
                
                pbar.set_description("loss:%.2f , %.2f"%(batch_mf_loss, batch_emb_loss ))
                
                optimizer.minimize(batch_loss)

                loss += batch_loss.detach().numpy()
                mf_loss += batch_mf_loss.detach().numpy()
                emb_loss += batch_emb_loss.detach().numpy()
                del batch_loss,batch_mf_loss,batch_emb_loss,users, pos_items, neg_items,u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
            test_log.add_scalar(step=epoch,tag="train/mf_loss",value=float(mf_loss))
            test_log.add_scalar(step=epoch,tag="train/loss",value=float(loss))
            test_log.add_scalar(step=epoch,tag="train/emb_loss",value=float(emb_loss))

            if (epoch + 1) % 10 != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                        epoch, time() - t1, loss, mf_loss, emb_loss)
                    print(perf_str)
                continue

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            ret = test(model, users_to_test, drop_flag=False)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])

            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                        'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                        (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)

            test_log.add_scalar(step=epoch,tag="test/recall@20",value=float(ret['recall'][0]))
            test_log.add_scalar(step=epoch,tag="test/precision@20",value=float(ret['precision'][0]))
            test_log.add_scalar(step=epoch,tag="test/ndcg@20",value=float(ret['ndcg'][0]))
            test_log.add_scalar(step=epoch,tag="test/hit_ratio@20",value=float(ret['hit_ratio'][0]))

            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc', flag_step=5)

            # ###stop if the satisfy the criteria
            # if  "amazon" in args.dataset:
            #     if float(ret['recall'][0])>0.0337  and float(ret['ndcg'][0])>0.0261:
            #         break
            # if  "yelp" in args.dataset:
            #     if float(ret['recall'][0])>0.0579 and float(ret['ndcg'][0])>0.0477:
            #         break

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True:
                break

            # *********************************************************
            # save the user & item embeddings for pretraining.
            if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
                torch.save(model.state_dict(), args.weights_path + str(model_file_prefix) )
                print('save the weights in path: ', args.weights_path + str(model_file_prefix) )

        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        ndcgs = np.array(ndcg_loger)
        hit = np.array(hit_loger)

        best_rec_0 = max(recs[:, 0])
        idx = list(recs[:, 0]).index(best_rec_0)

        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                    (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                    '\t'.join(['%.5f' % r for r in pres[idx]]),
                    '\t'.join(['%.5f' % r for r in hit[idx]]),
                    '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
        print(final_perf)