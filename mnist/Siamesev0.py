# -*- coding: utf-8 -*-

import os, argparse
import sys
import scipy
import timeit
import gzip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from sys import stdout
import pickle as pkl
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.cluster import KMeans
import torch.nn.functional as F
import math

from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import TensorDataset, DataLoader, Dataset

from sklearn import metrics, mixture

from torch.optim.optimizer import Optimizer
from sklearn.utils.linear_assignment_ import linear_assignment

import matplotlib.pyplot as plt





def to_numpy(x):
	if x.is_cuda:
		return x.data.cpu().numpy()
	else:
		return x.data.numpy()

class Siamese(nn.Module):
	def __init__(self,layer_sizes,activation=nn.ReLU(),w_init=None):
		super(Siamese,self).__init__()
		n_layer = len(layer_sizes)
		layers = []
		for idx in range(n_layer-1):
			layer = nn.Linear(layer_sizes[idx],layer_sizes[idx+1])
			if w_init is not None:
				layer.weight.data = torch.t(w_init[2*idx])
				layer.bias.data = w_init[2*idx+1]
			else:
				nn.init.xavier_normal_(layer.weight)

			layers.append(layer)
			if idx<n_layer-2:
				layers.append(activation)
		self.encoder = nn.Sequential(*layers)

	def forward(self,inputs):
		outputs = self.encoder(inputs)
		return outputs

def cal_similar(data,K):
	data = data.cuda()
	N = data.shape[0]
	similar_m = []
	for idx in range(N):
		dis = torch.sum(torch.pow(data-data[idx,:],2),dim=1)
		_, ind = dis.sort()
		similar_m.append(ind[0:K+1].view(1,K+1).cpu())
		stdout.write('\r')    
		stdout.write("| index #{}".format(idx+1))
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
		stdout.write("| index #{}".format(idx+1))
		stdout.flush()
	return err

def Form_data(data):
	K = 2
	similar_m = to_numpy(cal_similar(data,K))
	
	positive_list = []
	negative_list = []
	for idx in range(1,K+1):
		positive_list.extend(similar_m[:,idx])

		similar_v_n = np.random.permutation(data.size(0))
		negative_list.extend(similar_v_n)

	total_list = []
	total_list.extend(positive_list)
	total_list.extend(negative_list)
	data1 = []
	for idx  in range(2*K):
		data1.append(data)


	data1 = torch.cat(data1,dim=0)
	data2 = data[total_list,:]
	label = []
	label.append(torch.ones(len(positive_list)))
	label.append(torch.zeros(len(negative_list)))
	label = torch.cat(label,dim=0)

	return data1,data2,label
	



def train(siamese,optimizer,lr_scheduler,dataloader,epoch_num,use_gpu=torch.cuda.is_available()):

	if use_gpu:
		siamese = siamese.cuda()
		c = torch.tensor(3.0).cuda()
		th  = torch.tensor(0.0).view(1).cuda()
	else:
		c = torch.tensor(3.0)
		th  = torch.tensor(0.0).view(1)


	for epoch in range(epoch_num):
		lr_scheduler.step()
		Total_loss = []
		for batch_idx, batch in enumerate(dataloader):
			if use_gpu:
				inputs1 = Variable(batch[0].cuda())
				inputs2 = Variable(batch[1].cuda())
				label = batch[2].cuda()
				label = 2.0*label - 1
			else:
				inputs1 = Variable(batch[0])
				inputs2 = Variable(batch[1])
				label = batch[2]
				label = 2.0*label - 1

			latent1 = siamese(inputs1)
			latent2 = siamese(inputs2)
			distance = torch.sum(torch.pow(latent1-latent2,2),dim=1)
			loss = torch.sum(torch.max(c+torch.mul(label,distance),th))/inputs1.size(0)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			Total_loss.append(loss.item())

		Total_loss = np.mean(Total_loss)
		print('|Epoch:{} Total loss={:3f}'.format(epoch,Total_loss))

	return siamese


def random_select(data,K):
	N = data.size(0)
	M = data.size(1)
	outputs = []
	for idx in range(N):
		ind = np.random.permutation(M-1)
		ind = ind[:K] + 1
		temp = []
		temp.append(0)
		temp.extend(ind)
		outputs.append(data[idx,temp].view(1,-1))

	outputs = torch.cat(outputs,dim=0)
	
	return outputs


def cluster_acc(Y_pred, Y):
	assert Y_pred.size == Y.size
	D = max(Y_pred.max(), Y.max())+1
	w = np.zeros((D,D), dtype=np.int64)
	for i in range(Y_pred.size):
		w[Y_pred[i], Y[i]] += 1
	ind = linear_assignment(w.max() - w)
	return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w


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

	
	encoder_sizes = [784, 500, 500, 2000, 10]
	

	print('| Generate training data')

	reuse_data = 0

	if reuse_data == 0:
		data1, data2, label = Form_data(torch.from_numpy(image_train))
		torch.save(data1,'data_N1.pkl')
		torch.save(data2,'data_N2.pkl')
		torch.save(label,'data_label.pkl')
	else:
		data1 = torch.load('data_N1.pkl')
		data2 = torch.load('data_N2.pkl')
		label = torch.load('data_label.pkl')


	print("| Data shape: {}".format(data1.size()))
	print("| Number of positive pairs: {} Number of negative pairs: {}".format(torch.sum(label), label.size(0)-torch.sum(label)))
	print("| Completed.")

	resume = 0

	use_trainedweight = 1
	print("2. Construct the Siamese model")

	activation = nn.ReLU()

	if not resume:
		if use_trainedweight:
			w_init = []
			w_init.append(torch.tensor(np.loadtxt('../pretrained_weight/w_1.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('../pretrained_weight/b_1.out',delimiter=',').astype(np.float32)))
			
			w_init.append(torch.tensor(np.loadtxt('../pretrained_weight/w_2.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('../pretrained_weight/b_2.out',delimiter=',').astype(np.float32)))
			
			w_init.append(torch.tensor(np.loadtxt('../pretrained_weight/w_3.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('../pretrained_weight/b_3.out',delimiter=',').astype(np.float32)))
			
			w_init.append(torch.tensor(np.loadtxt('../pretrained_weight/w_4.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('../pretrained_weight/b_4.out',delimiter=',').astype(np.float32)))

			siamese = Siamese(encoder_sizes,activation,w_init=w_init)
		else:
			siamese = Siamese(encoder_sizes)
		

		BATCH_SIZE = 1000
		dataloader = DataLoader(TensorDataset(data1,data2,label),batch_size=BATCH_SIZE,shuffle=True)
		optimizer = optim.Adam(siamese.parameters(),lr=0.001,weight_decay=0.0001)
		lr_scheduler = StepLR(optimizer,step_size=70,gamma=0.1)
		print("2.2 Train the model")
		siamese = train(siamese,optimizer,lr_scheduler,dataloader,epoch_num=300)
		torch.save(siamese,'trained_siamese.pkl')
	else:
		print("| Loading trained siamese network")
		siamese = torch.load('trained_siamese.pkl')

	print("3. Test the network")
	siamese = siamese.cuda()
	siamese.eval()
	latent = siamese(torch.from_numpy(image_train).cuda())


	kmeans = KMeans(n_clusters=10,random_state=0).fit(to_numpy(latent))
	cls_index = kmeans.labels_
	mean = kmeans.cluster_centers_
	acc,_ = cluster_acc(cls_index,label_train)
	NMI = metrics.normalized_mutual_info_score(label_train, cls_index,average_method='arithmetic')
	print('| Kmeans ACC = {:6f} NMI = {:6f}'.format(acc,NMI))


	

	print("| Generate NN using the network")
	similar_m_latent = cal_similar(latent,100)
	similar_m_latent = random_select(similar_m_latent,40)
	torch.save(similar_m_latent,'similar_m.pkl')
	
