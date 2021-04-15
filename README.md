
虽然是基于非官方的pytorch 版本改编
 
对比了非官方的pytorch和官方的tensorflow的代码， 有两个区别：
1. tf 用split， 就是把user 分为100人一组，分开计算再合拼，  torch 没有split。但我觉得最后结果一样
2. leakyRELU的位置不一样， tf是relu后相加， torch是加后再relu  

修改了leakyRELU的位置,最后我们的paddle 版本可以跟官方版本完全对齐

### 复现结果要点
1、使用的数据集、模型文件及完整的复现代码

- `数据集` ： 采用 Amazon book 和 Yelp2018 在github上面下载的
- `完整的复现代码` 在这个项目的folder：NGCF-Paddle (论文实现代码)，  paddorch (提供pytorch接口的paddle实现)

关于我写的torch接口代码请参考 [pytorch 转 paddle 心得](https://blog.csdn.net/weixin_48733317/article/details/108176827)
有兴趣了解的朋友可以看我在这个[视频](https://aistudio.baidu.com/aistudio/education/lessonvideo/698277)的Paddorch介绍（10分钟位置开始），
之前我用paddorch库复现了3个GAN类别的项目。


值得注意的是虽然说这个是NGCF的paddle版本，但你基本上看不到paddle api接口，因为都被我在paddorch库中重新封装了， 所以代码看起来就跟torch一样 


2、提供具体详细的说明文档(或notebook)，内容包括:

(1) 数据准备与预处理步骤

- 数据直接挂载和解压，没有其他预处理步骤
 

(2) 训练脚本/代码，最好包含训练一个epoch的运行日志

- 在下面的cell 中有从零开始训练的代码 ，因为算力卡不够，中途停了，后面从上checkpoint 开始继续训练
- `main.py` 是入口文件， 跟官方代码一样接口，参考[pytorch repo](https://github.com/huangtinglin/NGCF-PyTorch)


(3) 测试脚本/代码，必须包含评估得到最终精度的运行日志

- 原来的官方代码没有独立的测试脚本，测试是包含在training script里面，每10 epoch evaluation 一次，所以在训练的log中
看到 recall， precision， hit， ndcg 都是测试集的metric 分别对于k=20 和k=100
- 我单独写一个测试脚本 `test.py dataset model_file` 输出recall@20, NDCG@20 


(4) 最终精度，如精度相比源码有提升，需要说明精度提升用到的方法与技巧(不可更换网络主体结构，不可将测试集用于训练)

#### Amazon book Recall@20是X， 验收要求 0.0337
#### Amazon book NDCG@20是X， 验收要求 0.0261
#### Yelp2018 Recall@20是X， 验收要求 0.0579
#### Yelp2018 NDCG@20是X， 验收要求 0.0477

 



(5) 其它学员觉得需要说明的地方
- 记得安装paddorch库， `cd paddorch;pip install .`
- 关键点可以看我实现的paddorch.sparse.mm 函数

3、上传最终训练好的模型文件
- 在`NGCF-Paddle/NGCF/amazon_book_checkpoints` 和 `NGCF-Paddle/NGCF/yelp2018_checkpoints`

4、如评估结果保存在json文件中，可上传最终评估得到的json文件
没有生成json文件， 但可以下载visualdl`NGCF-Paddle/NGCF/log` 目录进行评价


 




=============================================================================================
# Neural Graph Collaborative Filtering
This is my PyTorch implementation for the paper:

>Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua (2019). Neural Graph Collaborative Filtering, [Paper in ACM DL](https://dl.acm.org/citation.cfm?doid=3331184.3331267) or [Paper in arXiv](https://arxiv.org/abs/1905.08108). In SIGIR'19, Paris, France, July 21-25, 2019.

The TensorFlow implementation can be found [here](<https://github.com/xiangwang1223/neural_graph_collaborative_filtering>).

## Introduction
My implementation mainly refers to the original TensorFlow implementation. It has the evaluation metrics as the original project. Here is the example of Gowalla dataset:

```
Best Iter=[38]@[32904.5]	recall=[0.15571	0.21793	0.26385	0.30103	0.33170], precision=[0.04763	0.03370	0.02744	0.02359	0.02088], hit=[0.53996	0.64559	0.70464	0.74546	0.77406], ndcg=[0.22752	0.26555	0.29044	0.30926	0.32406]
```

Hope it can help you!

## Environment Requirement
The code has been tested under Python 3.6.9. The required packages are as follows:
* pytorch == 1.3.1
* numpy == 1.18.1
* scipy == 1.3.2
* sklearn == 0.21.3

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the parser function in NGCF/utility/parser.py).
* Gowalla dataset
```
python main.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```

* Amazon-book dataset
```
python main.py --dataset amazon-book --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 200 --verbose 50 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```
## Supplement

* The parameter `negative_slope` of LeakyReLu was set to 0.2, since the default value of PyTorch and TensorFlow is different.
* If the arguement `node_dropout_flag` is set to 1, it will lead to higher calculational cost.