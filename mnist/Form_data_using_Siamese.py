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


def to_numpy(x):
	if x.is_cuda:
		return x.data.cpu().numpy()
	else:
		return x.data.numpy()

def cal_similar(data,K):
	data = data.cuda()
	N = data.shape[0]
	similar_m = []
	for idx in range(N):
		dis = torch.sum(torch.pow(data-data[idx,:],2),dim=1)
		_, ind = dis.sort()
		select = np.random.permutation(100)
		select1 = select[0:K] + 1
		select1 = select1.tolist()
		temp = []
		temp.append(0)
		temp.extend(select1)
		similar_m.append(ind[temp].view(1,K+1).cpu())
		stdout.write('\r')    
		stdout.write("|index #{}".format(idx+1))
		stdout.flush()

	similar_m = torch.cat(similar_m,dim=0)

	return similar_m

def cal_err(data,index):
	data = data.cuda()
	index = index.cuda()
	N = data.shape[0]
	err = 0
	for idx in range(N):
		err = err + torch.sum((index[data[idx,:]] != index[data[idx,0]])*1.0).cpu()
		stdout.write('\r')    
		stdout.write("|index #{}".format(idx+1))
		stdout.flush()
	return err

def form_data(data, similar_m):
	K = similar_m.shape[1]
	for idx in range(K):
		print(idx)
		data_s = data[similar_m[:,idx]]
		torch.save(to_numpy(data_s),'./data/mnist/mnist_{}_all_siamese.pkl'.format(idx))







if __name__ == '__main__':
	with gzip.open('../data/mnist/mnist_dcn.pkl.gz') as f:
		data = pkl.load(f)

	image_train = data['image_train']
	label_train = data['label_train']
	image_test = data['image_test']
	label_test = data['label_test']
	
	train_data = []
	train_data.append(image_train)
	train_data.append(image_test)
	
	train_label = []
	train_label.append(label_train)
	train_label.append(label_test)
	
	image_train = np.concatenate(train_data,axis=0)
	label_train = np.concatenate(train_label,axis=0)
	
	K = 40
	
	resume = 1

	if resume:
		similar_m = torch.load('similar_m.pkl')
	else:
		similar_m = cal_similar(torch.from_numpy(image_train),K)
		torch.save(similar_m,'similar_m.pkl')
	print(similar_m.size())
	form_data(torch.from_numpy(image_train),similar_m)
	torch.save(label_train,'./data/mnist/mnist_label_all_siamese.pkl')


