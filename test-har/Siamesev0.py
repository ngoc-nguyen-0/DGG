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
import scipy.io as scio

from sklearn.manifold import TSNE






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
				# layer.weight.data = w_init[2*idx]
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

def cal_similar(data,K,distance='Euclidean'):
	data = data.cuda()
	N = data.shape[0]
	# print(data.shape)
	# assert 1 == 0
	similar_m = []
	if distance == "cosine":
		normaler = torch.sqrt(torch.sum(torch.pow(data,2),dim=1))
	for idx in range(N):
		if distance == 'Euclidean':
			dis = torch.sum(torch.pow(data-data[idx,:],2),dim=1)
		elif distance == 'cosine':
			dis = 1 - torch.abs(torch.sum(torch.mul(data,data[idx,:].unsqueeze(0)),dim=1))/normaler/torch.norm(data[idx,:],2)
			# dis = 1 - torch.sum(torch.mul(data,data[idx,:].unsqueeze(0)),dim=1)/normaler/torch.norm(data[idx,:],2)

		_, ind = dis.sort()
		# print(ind[0:K+1])
		# assert 1 == 0
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
		# print(torch.sum((index[data[idx,:]] != index[data[idx,0]])*1.0))
		err = err + torch.sum((index[data[idx,:]] != index[data[idx,0]])*1.0).cpu()
		stdout.write('\r')    
		stdout.write("| index #{}".format(idx+1))
		stdout.flush()
	return err

def Form_data(data):
	K = 3
	similar_m = to_numpy(cal_similar(data,K,distance='cosine'))
	
	positive_list = []
	negative_list = []
	for idx in range(1,K+1):
		positive_list.extend(similar_m[:,idx])

		similar_v_n = np.random.permutation(data.size(0))
		'''
		while ((similar_v_n.view(-1,1) - similar_m) == 0).any():
			similar_v_n = np.random.permutation(data.size(0))
		'''
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

	siamese.train()
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

			# print(inputs1.size())

			latent1 = siamese(inputs1)
			latent2 = siamese(inputs2)
			distance = torch.sum(torch.pow(latent1-latent2,2),dim=1)
			# print(distance.type())
			# print(label.type())
			# print(c.type())
			# assert 1 == 0
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
		# ind = np.random.permutation(K)
		ind = ind[:K] + 1
		temp = []
		temp.append(0)
		temp.extend(ind)
		outputs.append(data[idx,temp].view(1,-1))

	outputs = torch.cat(outputs,dim=0)
	
	return outputs





if __name__ == '__main__':
	'''
	if dataset == 'mnist':
		path = path + 'mnist.pkl.gz'
		with gzip.open('../data/mnist/mnist_dcn.pkl.gz') as f:
			data = pkl.load(f)
		image_train = data['image_train']
		label_train = data['label_train']
		image_test = data['image_test']
		label_test = data['label_test']

		X = np.concatenate((image_train,image_test),axis=0)
		Y = np.concatenate((label_train,label_test),axis=0)
		X = (X+1)/2
		
	if dataset == 'reuters10k':
		data=scio.loadmat(path+'reuters10k.mat')
		X = data['X']
		Y = data['Y'].squeeze()
		
	if dataset == 'har':
		data=scio.loadmat(path+'HAR.mat')
		X=data['X']
		X=X.astype('float32')
		Y=data['Y']-1
		X=X[:10200]
		Y=Y[:10200]
	
	path = 'data_ori/reuters10k/'
	data=scio.loadmat(path+'reuters10k.mat')
	X = data['X']
	Y = data['Y'].squeeze()

	image_train = X.astype('float32')
	label_train = Y.astype('float32')
	'''
	path = 'data_ori/har/'
	data=scio.loadmat(path+'HAR.mat')
	X=data['X']
	X=X.astype('float32')
	Y=data['Y']-1
	X=X[:10200]
	Y=Y[:10200]
	image_train = X.astype('float32')
	label_train = Y.astype('float32')
	image_train = image_train/2
	encoder_sizes = [561, 500, 500, 2000, 10]
	# encoder_sizes = [784, 1024, 1024, 512, 10]
	

	print('| Generate training data')

	reuse_data = 1

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

	resume = 1

	use_trainedweight = 1
	print("2. Construct the Siamese model")

	activation = nn.ReLU()

	if not resume:
		if use_trainedweight:
			w_init = []
			
			w_init.append(torch.tensor(np.loadtxt('pretrained_weight/har/har_w_0.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('pretrained_weight/har/har_b_0.out',delimiter=',').astype(np.float32)))
			
			w_init.append(torch.tensor(np.loadtxt('pretrained_weight/har/har_w_1.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('pretrained_weight/har/har_b_1.out',delimiter=',').astype(np.float32)))
			
			w_init.append(torch.tensor(np.loadtxt('pretrained_weight/har/har_w_2.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('pretrained_weight/har/har_b_2.out',delimiter=',').astype(np.float32)))
			
			w_init.append(torch.tensor(np.loadtxt('pretrained_weight/har/har_w_3.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('pretrained_weight/har/har_b_3.out',delimiter=',').astype(np.float32)))
			for idx in range(len(w_init)):
				print(w_init[idx].size())
			siamese = Siamese(encoder_sizes,activation,w_init=w_init)
		else:
			siamese = Siamese(encoder_sizes)
		
		'''
		siamese = siamese.cuda()
		siamese.eval()
		latent = siamese(torch.from_numpy(image_train).cuda())
		similar_m_latent = cal_similar(latent,2)
		err1 = cal_err(similar_m_latent,torch.from_numpy(label_train))
		similar_m_ori = cal_similar(torch.from_numpy(image_train),2)
		err2 = cal_err(similar_m_ori,torch.from_numpy(label_train))
		print('| The number of inconsist NN in data space is {} The number of inconsist NN in latent space is {}'.format(err2,err1))

		assert 1 == 0
		'''

		BATCH_SIZE = 1000
		dataloader = DataLoader(TensorDataset(data1,data2,label),batch_size=BATCH_SIZE,shuffle=True)
		optimizer = optim.Adam(siamese.parameters(),lr=0.0005,weight_decay=0.0001)
		# optimizer = optim.RMSprop(siamese.parameters(),lr=0.001,weight_decay=0.0001)
		# optimizer = optim.RMSprop(siamese.parameters(),lr=0.001,weight_decay=0.0001)
		# optimizer = optim.SGD(siamese.parameters(),lr=0.001,momentum=0.9,weight_decay=0.0001)
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

	'''
	X_embedded = TSNE(n_components=2).fit_transform(latent.data)
	plt.figure()
	for k in range(6):
		plt.scatter((X_embedded[np.where(label_train==k),0].squeeze()),(X_embedded[np.where(label_train==k),1].squeeze()))
	plt.show()
	'''
	

	print("| Generate NN using the network")
	similar_m_latent = cal_similar(latent,100)
	similar_m_latent = random_select(similar_m_latent,40)
	torch.save(similar_m_latent,'similar_m.pkl')
	# ind = np.random.permutation(101)
	# similar_m_latent = similar_m_latent[:,ind[:21]]
	
	print("")
	print("| Evaluate the NN")
	err1 = cal_err(similar_m_latent,torch.from_numpy(label_train))
	print("")
	print(err1)
	
	print("| Generate NN using the original data")
	similar_m_ori = cal_similar(torch.from_numpy(image_train),20,distance='cosine')
	print("")
	print("| Evaluate the NN")
	err2 = cal_err(similar_m_ori,torch.from_numpy(label_train))
	print("")
	print("| Completed")
	print('| The number of inconsist NN in data space is {} The number of inconsist NN in latent space is {}'.format(err2,err1))

	print("| Generate NN using the original data")
	similar_m_ori = cal_similar(torch.from_numpy(image_train),20,distance='Euclidean')
	print("")
	print("| Evaluate the NN")
	err3 = cal_err(similar_m_ori,torch.from_numpy(label_train))
	print("")
	print("| Completed")
	print('| The numbers of inconsist NN in data space using Euclidean distance and cosine distance are respecitively {} and {}'.format(err3,err2))
	
	
