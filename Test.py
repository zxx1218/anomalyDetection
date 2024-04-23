import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model.base_model import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob
from tqdm import tqdm
import argparse
import pdb
import warnings
import time
from getROC import save_roc_curve
warnings.filterwarnings("ignore") 

parser = argparse.ArgumentParser(description="MPN")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=4, help='length of the frame sequences')
parser.add_argument('--fdim', type=list, default=[128], help='channel dimension of the features')
parser.add_argument('--pdim', type=list, default=[128], help='channel dimension of the prototypes')
parser.add_argument('--psize', type=int, default=10, help='number of the prototypes')
parser.add_argument('--test_iter', type=int, default=1, help='channel of input images')
parser.add_argument('--K_hots', type=int, default=0, help='number of the K hots')
parser.add_argument('--alpha', type=float, default=0.5, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--num_workers_test', type=int, default=8, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='data/', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')

args = parser.parse_args()

torch.manual_seed(2020)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

# test_folder = args.dataset_path + args.dataset_type + "/testing/frames"   # 原始数据集
test_folder = r"/home/zxx/dqalpha/dataset/anotoimg"    # 自建视频帧图像路径

# Loading dataset
test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

test_size = len(test_dataset)

print("测试集长度:",test_size)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')


model = convAE(args.c, args.t_length, args.psize, args.fdim[0], args.pdim[0])
model.cuda()

dataset_type = args.dataset_type if args.dataset_type != 'SHTech' else 'shanghai'
# labels = np.load('./data/frame_labels_'+dataset_type+'.npy')    # 原始标签
labels = np.load('../dataset/mylabels.npy') # 自建数据集标签

if 'SHTech' in args.dataset_type or 'ped1' in args.dataset_type:
    labels = np.expand_dims(labels, 0)

print("所用标签:",labels.shape)

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))

for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])
    # print(videos[video_name])
    # print(video)

labels_list = []
anomaly_score_total_list = []
anomaly_score_ae_list = []
anomaly_score_mem_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

print('Evaluation of Version {0} on {1}'.format(args.model_dir.split('/')[-1], args.dataset_type))
# if 'ucf' in args.model_dir:
#     snapshot_dir = args.model_dir.replace(args.dataset_type,'UCF')
# else:
#     snapshot_dir = args.model_dir.replace(args.dataset_type,'SHTech')
snapshot_path = args.model_dir
psnr_dir = args.model_dir.replace('exp','results')

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    videos[video_name]['labels'] = labels[0][4+label_length:videos[video_name]['length']+label_length]
    labels_list = np.append(labels_list, labels[0][args.t_length+args.K_hots-1+label_length:videos[video_name]['length']+label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

print("labels_list:",labels_list)

# if not os.path.isdir(psnr_dir):
#     os.mkdir(psnr_dir)

ckpt = snapshot_path
ckpt_name = ckpt.split('_')[-1]
ckpt_id = int(ckpt.split('/')[-1].split('_')[-1][:-4])
# Loading the trained model
model = torch.load(ckpt)
if type(model) is dict:
    model = model['state_dict']
model.cuda()
model.eval()

# Setting for video anomaly detection
forward_time = AverageMeter()
video_num = 0
update_weights = None
imgs_k = []
k_iter = 0
anomaly_score_total_list.clear()
anomaly_score_ae_list.clear()
anomaly_score_mem_list.clear()
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    psnr_list[video_name].clear()
    feature_distance_list[video_name].clear()
pbar = tqdm(total=len(test_batch),
            bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]',)
with torch.no_grad():
    for k,(imgs) in enumerate(test_batch):
        hidden_state = None
        imgs = Variable(imgs).cuda()
        
        start_t = time.time()
        outputs, feas, _, _, _, fea_loss = model.forward(imgs[:,:3*4], update_weights, False)
        # print(outputs.shape)
        
        end_t = time.time()
        
        if k>=len(test_batch)//2:
            forward_time.update(end_t-start_t, 1)
        # import pdb;pdb.set_trace()
        # outputs = torch.cat(pred,1)
        mse_imgs = loss_func_mse((outputs[:]+1)/2, (imgs[:,-3:]+1)/2)

        mse_feas = fea_loss.mean(-1)
        
        mse_feas = mse_feas.reshape((-1,1,256,256))
        mse_imgs = mse_imgs.view((mse_imgs.shape[0],-1))
        mse_imgs = mse_imgs.mean(-1)
        mse_feas = mse_feas.view((mse_feas.shape[0],-1))
        mse_feas = mse_feas.mean(-1)
        # import pdb;pdb.set_trace()
        vid = video_num
        vdd = video_num if args.dataset_type != 'avenue' else 0
        for j in range(len(mse_imgs)):
            psnr_score = psnr(mse_imgs[j].item())
            fea_score = psnr(mse_feas[j].item())
            psnr_list[videos_list[vdd].split('/')[-1]].append(psnr_score)
            feature_distance_list[videos_list[vdd].split('/')[-1]].append(fea_score)
            k_iter += 1
            if k_iter == videos[videos_list[video_num].split('/')[-1]]['length']-args.t_length+1:
                video_num += 1
                update_weights = None
                k_iter = 0
                imgs_k = []
                hidden_state = None
            
        pbar.set_postfix({
                        'Epoch': '{0}'.format(ckpt_name),
                        'Vid': '{0}'.format(args.dataset_type+'_'+videos_list[vid].split('/')[-1]),
                        'AEScore': '{:.6f}'.format(psnr_score),
                        'MEScore': '{:.6f}'.format(fea_score),
                        'time': '{:.6f}({:.6f})'.format(end_t-start_t,forward_time.avg),
                        })
        pbar.update(1)

pbar.close()
forward_time.reset()
# Measuring the abnormality score and the AUC
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    template = calc(15, 2)  #滑动窗口长度为15，步幅为2 
    aa = filter(anomaly_score_list(psnr_list[video_name]), template, 15) # 对PSNR分数列表应用滑动窗口平均操作后得到的结果
    bb = filter(anomaly_score_list(feature_distance_list[video_name]), template, 15) # 对特征距离分数列表应用滑动窗口平均操作后得到的结果。
    temp = score_sum(aa, bb, args.alpha)
    anomaly_score_total_list += temp

    #如果anomaly_score_temp中的值大于阈值则将当前下标输出
    for index, value in enumerate(temp):
        if value > 0.6:
            print("frame: {} anomaly_score: {} video: {}".format(index, value, video_name))
    temp.clear()

anomaly_score_total = np.asarray(anomaly_score_total_list)


# for label, score in zip(labels_list, anomaly_score_total):    # 将labels_list中的值与anomaly_score_total中的值对应输出在控制台
#     print("Label:", label, " Anomaly Score:", score)


# 提取labels_list为1.0的anomaly_score_total中的值
anomaly_scores_label_1 = [score for label, score in zip(labels_list, anomaly_score_total) if label == 1.0]
# 提取labels_list为0.0的anomaly_score_total中的值
anomaly_scores_label_0 = [score for label, score in zip(labels_list, anomaly_score_total) if label == 0.0]
# print("为1.0的anomaly_score_total:",anomaly_scores_label_1)
# print("为0.0的anomaly_score_total:",anomaly_scores_label_0)
# print("labels_list:",labels_list)
# 计算labels_list为1.0对应的anomaly_score_total值的平均
average_label_1 = sum(anomaly_scores_label_1) / len(anomaly_scores_label_1)
# 计算labels_list为0.0对应的anomaly_score_total值的平均
average_label_0 = sum(anomaly_scores_label_0) / len(anomaly_scores_label_0) if len(anomaly_scores_label_0) != 0 else 0

print("平均值(label为1.0):", average_label_1)
print("平均值(label为0.0):", average_label_0)




print("anomaly_score_total:",anomaly_score_total.shape)
print("label_list:",labels_list.shape)
print("np.expand_dims(1-labels_list, 0):",np.expand_dims(1-labels_list, 0))
# accuracy_total = 100*AUC(anomaly_score_total, np.expand_dims(1-labels_list, 0))

print('The result of Version {0} Epoch {1} on {2}'.format(psnr_dir.split('/')[-1], ckpt_name, args.dataset_type))
# print('Total AUC: {:.4f}%'.format(accuracy_total))

# save_roc_curve(np.squeeze(anomaly_score_total), np.squeeze(np.expand_dims(1-labels_list, 0), axis=0), './') # 绘制roc曲线

print("anomaly_score_total:",anomaly_score_total)
print("labels_list:",labels_list)


# 实验
np.savetxt('score.txt', anomaly_score_total, fmt='%f')
np.savetxt('labels.txt', labels_list, fmt='%f')


