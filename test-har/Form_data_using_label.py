# -*- coding: utf-8 -*-

import os, argparse
import sys
import scipy
import timeit
import gzip
import torch

import numpy as np
from sys import stdout
import pickle as pkl
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.cluster import KMeans
import scipy.io as scio


def to_numpy(x):
	if x.is_cuda:
		return x.data.cpu().numpy()
	else:
		return x.data.numpy()

def cal_similar(data,K):
	data = data.cuda()
	N = data.shape[0]
	# print(data.shape)
	# assert 1 == 0
	similar_m = []
	for idx in range(N):
		dis = torch.sum(torch.pow(data-data[idx,:],2),dim=1)
		_, ind = dis.sort()
		# print(ind[0:K+1])
		# assert 1 == 0
		similar_m.append(ind[0:K+1].view(1,K+1).cpu())
		stdout.write('\r')    
		stdout.write("|index #{}".format(idx+1))
		stdout.flush()
	print(' ')

	similar_m = torch.cat(similar_m,dim=0)

	return similar_m

def cal_err(data,index):
	data = data.cuda()
	index = index.cuda()
	N = data.shape[0]
	err = []
	
	for idx in range(N):
		# print(torch.sum((index[data[idx,:]] != index[data[idx,0]])*1.0))
		temp = torch.sum((index[data[idx,:]] != index[data[idx,0]])*1.0).cpu()
		if temp>0:
			err.append(temp)
		stdout.write('\r')    
		stdout.write("|index #{}".format(idx+1))
		stdout.flush()
	print(' ')
	return err

def form_data(data, similar_m):
	K = similar_m.shape[1]
	data_aug = []
	for idx in range(K):
		data_s = data[similar_m[:,idx]]
		data_aug.append(torch.unsqueeze(data_s,-1))
		# torch.save(data_s,'mnist_dcn_s_{}.pkl'.format(idx+1))
	data_aug = torch.cat(data_aug,dim=-1)
	return data_aug
	







if __name__ == '__main__':
	path = 'data_ori/har/'
	data=scio.loadmat(path+'HAR.mat')
	X=data['X']
	X=X.astype('float32')
	Y=data['Y']-1
	X=X[:10200]
	Y=Y[:10200]
	image_train = X.astype('float32')
	label_train = Y.astype('int32').squeeze()
	
	K = 20

	new_data = []
	new_label = []
	for idx in range(6):
		index = np.where(label_train==idx)
		data_temp = image_train[index]
		similar_m = cal_similar(torch.from_numpy(data_temp),K)
		data_temp = form_data(torch.from_numpy(data_temp),similar_m)
		new_data.append(data_temp)
		new_label.append(torch.from_numpy(label_train[index]))

	image_train = torch.cat(new_data,dim=0)
	label_train = torch.cat(new_label,dim=0)
	index_new = np.arange(image_train.size(0))
	image_train = image_train[index_new,:,:]
	label_train = label_train[index_new]

	# image_train = np.concatenate(new_data, axis=0)
	# label_train = np.concatenate(new_label, axis=0)

	
	print(image_train.size())
	for idx in range(K+1):
		image_train_temp = to_numpy(image_train[:,:,idx])
		# print(image_train_temp.size())
		# label_train_temp = label_train[:,:,idx]
		# data_aug = {'image_train':to_numpy(image_train_temp), 'label_train':to_numpy(label_train_temp)}
		torch.save(image_train_temp,'data_NN/har/har_{}_all_real.pkl'.format(idx))

	torch.save(to_numpy(label_train),'data_NN/har/har_label_all_real.pkl')



