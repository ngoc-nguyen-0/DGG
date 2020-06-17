# -*- coding: utf-8 -*-

import os, argparse
import sys
import scipy
import timeit
import gzip
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

import numpy as np
from sys import stdout
import pickle as pkl
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.cluster import KMeans
import scipy.io as scio
import torchvision

def to_numpy(x):
	if x.is_cuda:
		return x.data.cpu().numpy()
	else:
		return x.data.numpy()

def cal_similar(data,K,random=False):
	data = data.cuda()
	N = data.shape[0]
	# print(data.shape)
	# assert 1 == 0
	similar_m = []
	for idx in range(N):
		dis = torch.sum(torch.pow(data-data[idx,:],2),dim=1)
		_, ind = dis.sort()
		if random == True:
			temp = []
			temp.append(np.array([0]))
			temp.append(np.random.permutation(data.size(0)-1)+1)
			temp = np.concatenate(temp,axis=0)
			ind = ind[temp]
			

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
	path = 'data_ori/STL10/'
	data=scio.loadmat(path+'train.mat')
	train_X = data['X']
	train_Y = data['y'].squeeze()

	path = 'data_ori/STL10/'
	data=scio.loadmat(path+'test.mat')
	test_X = data['X']
	test_Y = data['y'].squeeze()


	X = []
	Y = []
	X.append(train_X)
	X.append(test_X)
	Y.append(train_Y)
	Y.append(test_Y)
	X = np.concatenate(X,axis=0)
	Y = np.concatenate(Y,axis=0)
	X = np.reshape(X,(-1,3,96,96))
	X = np.transpose(X,(0,1,3,2))

	'''
	X = np.transpose(X,(0,3,2,1))
	plt.imshow(X[0,:,:,:])
	plt.show()
	'''
	image_train = X.astype('float32')/255
	image_train[:,0,:,:] = (image_train[:,0,:,:] - 0.485)/0.229
	image_train[:,1,:,:] = (image_train[:,1,:,:] - 0.456)/0.224
	image_train[:,2,:,:] = (image_train[:,2,:,:] - 0.406)/0.225
	label_train = Y.astype('float32')-1

	res50_model = torchvision.models.resnet50(pretrained=True)
	res50_conv = nn.Sequential(*list(res50_model.children())[:-2])
	res50_conv.eval()
	data = torch.from_numpy(image_train)
	dataloader = DataLoader(TensorDataset(data),batch_size=200,shuffle=False)
	res50_conv = res50_conv.cuda()
	total_output = []
	for batch_idx, batch in enumerate(dataloader):
		inputs = batch[0].cuda()
		output = res50_conv(inputs)
		total_output.append(output.data)
	total_output = torch.cat(total_output,dim=0)

	feature_train = torch.sum(torch.sum(total_output,dim=-1),dim=-1)/9
	

	image_train = to_numpy(feature_train)




	
	K = 20

	new_data = []
	new_label = []
	for idx in range(10):
		index = np.where(label_train==idx)
		data_temp = image_train[index]
		similar_m = cal_similar(torch.from_numpy(data_temp),K,random=True)
		data_temp = form_data(torch.from_numpy(data_temp),similar_m[:,:K+1])
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
		torch.save(image_train_temp,'data_NN/stl10/stl10_{}_all_real.pkl'.format(idx))

	torch.save(to_numpy(label_train),'data_NN/stl10/stl10_label_all_real.pkl')



