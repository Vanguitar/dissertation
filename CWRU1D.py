from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from src import metrics
from src.datasets import *

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from sklearn.manifold import TSNE

import cv2
from src.DCPN import DCPN
import argparse
import os
matplotlib.use('Agg')


# GPU设置
def GpuInit():

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    return session
# 参数设置
def parse_args():
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--dataset',default='cwru1D',choices=['cwru1D'])
    parser.add_argument('--n_clusters',default=4,type=int)
    parser.add_argument('--batch_size',default=512,type=int) #512改32
    parser.add_argument('--epochs',default=1,type=int)  #500改为1
    parser.add_argument('--cae_weights',
                        #default=None,
                        #default='results2/ytf/combinetrain_cae_model.h5',
                        help = 'This is argument must be given')
    parser.add_argument('--save_dir',default='results2/cwru')

    args = parser.parse_args()

    #print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == "__main__":

    args = parse_args()

    print(args)

    sess = GpuInit()

    x, y = load_CWRU1D(data_path = './2数据预处理/cwru/2原始数据/cwruN4')

    dc = DCPN(input_shape=(1024,1),n_clusters =4,datasets='cwru1D',x = x,y= y,
            pretrained = args.cae_weights,
            session = sess,
            lamda = 0,
            alpha = 1)
    dc.visulization(args.save_dir + '/embedding_1.svg', save_dir=args.save_dir)
    # 将x放入AE训练
    dc.pretrain(x,
                batch_size = args.batch_size,
                epochs = args.epochs,
                save_dir=args.save_dir)
    # 随机1000个。TSNE+GMM测试。
    dc.evaluate(flag_all=True)
    dc.visulization(args.save_dir + '/embedding_init.svg',save_dir = args.save_dir)


    # 将x TSNE+GMM后的acc。+定义triple loss
    dc.refineTrain(x,
                   batch_size = args.batch_size,
                   epochs = 5,
                   save_dir = args.save_dir,
                   second = True
                   )
    #  经过triple loss后的acc
    dc.evaluate(flag_all= True)
    dc.visulization(args.save_dir + '/embedding_refine.svg',save_dir = args.save_dir,flag = 1)
#将保存的 test.svg文件 用 visio 打开，此时就能查看此矢量图；然后选中该图，复制到word 中即可。或者直接图片插入到word里效果是一样的

















