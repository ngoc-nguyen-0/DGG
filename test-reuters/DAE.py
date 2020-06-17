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
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment

import scipy.io as scio

import matplotlib.pyplot as plt



def to_numpy(x):
	if x.is_cuda:
		return x.data.cpu().numpy()
	else:
		return x.data.numpy()

def getcorrupt(data,noise_level):
	noise = torch.rand(data.size())
	mask = (noise > noise_level).float()
	if data.is_cuda:
		mask = mask.cuda()
	data_cor = torch.mul(mask,data)
	return data_cor

def layerwise_train(model,dataloader,optimizer,lr_scheduler,epoch_num=50,use_gpu=torch.cuda.is_available()):
	if use_gpu:
		model = model.cuda()
	
	model.train()
	sigma = 0.3
	for epochs in range(epoch_num):
		lr_scheduler.step()
		#lambda_sche.step()
		total_loss = []
		for batch_idx, batch in enumerate(dataloader):
			if use_gpu:
				inputs = Variable(batch[0].cuda())
			else:
				inputs = Variable(batch[0])
			
			  
			optimizer.zero_grad()
			# vis_recon = model(getcorrupt(inputs.data,sigma))
			vis_recon = model.as_DAE(inputs,sigma)
			loss = torch.sum(torch.pow(inputs-vis_recon,2))/inputs.size()[0]
			loss.backward()
			optimizer.step()
			
			total_loss.append(loss.item())
		
		stdout.write('\r')    
		stdout.write("| Epoch #{} Loss = {:.6f}".format(epochs+1, np.mean(total_loss)))
		stdout.flush()
	
	print("")
	return model

class AE_layer(nn.Module):
	def __init__(self,layer_type,n_vis,n_hid,activation=nn.ReLU(),dropout=0,w_init=None):
		super(AE_layer,self).__init__()
		encoder_layers = []
		layer = nn.Linear(n_vis,n_hid)
		if w_init is not None:
			# print(len(w_init))
			layer.weight.data = torch.t(w_init[0])
			layer.bias.data = w_init[1]
		
		else:
			# nn.init.xavier_uniform_(layer.weight)
			layer.weight.data = torch.randn(n_hid,n_vis)*1e-2
			layer.bias.data = torch.zeros(n_hid)
			
			
		
		encoder_layers.append(layer)
		if (layer_type == "First") or (layer_type == "Latent"):
			encoder_layers.append(activation)
		

		self.encoder_layers = nn.Sequential(*encoder_layers)
		
		decoder_layers = []
		layer = nn.Linear(n_hid,n_vis)
		if w_init is not None:
			layer.weight.data = torch.t(w_init[2])
			layer.bias.data = w_init[3]
		
		else:
			# nn.init.xavier_uniform_(layer.weight)
			layer.weight.data = torch.randn(n_vis,n_hid)*1e-2
			layer.bias.data = torch.zeros(n_vis)
			
		
		decoder_layers.append(layer)
		if (layer_type == "Last") or (layer_type == "Latent"):
			decoder_layers.append(activation)
			
		
		self.decoder_layers = nn.Sequential(*decoder_layers)

		
	def get_latent(self, inputs):
		latent = self.encoder_layers(inputs)
		return latent

	def as_DAE(self,inputs,sigma):
		latent = self.encoder_layers(getcorrupt(inputs,sigma))/(1-sigma)
		outputs = self.decoder_layers(getcorrupt(latent,sigma))/(1-sigma)

		return outputs
	
	def get_recon(self,inputs):
		recon = self.decoder_layers(inputs)
		return recon
		
	def forward(self,inputs):
		latent = self.encoder_layers(inputs)
		outputs = self.decoder_layers(latent)
		return outputs

class DAE(nn.Module):
	def __init__(self,layer_sizes,inputs,activation=nn.ReLU(),w_init=None):
		super(DAE,self).__init__()
		n_layer = len(layer_sizes)
		BATCH_SIZE = 100
		dropout = 0
		dataloader = DataLoader(TensorDataset(torch.from_numpy(inputs)),batch_size=BATCH_SIZE,shuffle=True)
		output_data = torch.from_numpy(inputs).cuda()
		model_set = []
		for idx in range(n_layer-1):
			if idx == 0:
				layer_type = "First"
				lr = 0.0001
			elif idx == n_layer-2:
				layer_type = "Last"
				lr = 0.0001
			else:
				layer_type = "Latent"
				lr = 0.0001

			print("| Generate the {}th layer".format(idx+1))
			if w_init is not None:
				# print(w_init[0].size())
				# print(len(w_init))
				layer_model = AE_layer(layer_type,layer_sizes[idx],layer_sizes[idx+1],activation,dropout,w_init[4*idx:4*idx+4]).cuda()
			else:
				layer_model = AE_layer(layer_type,layer_sizes[idx],layer_sizes[idx+1],activation,dropout)
				
				# optimizer = optim.SGD(layer_model.parameters(),lr=lr,momentum=0.9)
				optimizer = optim.Adam(layer_model.parameters(),lr=lr,weight_decay=0.0001)
				lr_scheduler = StepLR(optimizer, step_size=70, gamma=0.1)
				layer_model = layerwise_train(layer_model,dataloader,optimizer,lr_scheduler)
				
			model_set.append(layer_model)
			layer_model.eval()
			output_data = layer_model.get_latent(output_data)
			dataloader = DataLoader(TensorDataset(output_data),batch_size=BATCH_SIZE,shuffle=True)

		self.model_set = model_set

	def get_para(self):
		n_layer = len(self.model_set)
		para = []
		for idx in range(n_layer):
			para.extend(self.model_set[idx].parameters())

		return para

	def get_latent(self,inputs):
		n_layer = len(self.model_set)
		data = inputs
		for idx in range(n_layer-1):
			data = self.model_set[idx].get_latent(data)
		
		x_mean = self.model_set[-1].get_latent(data)

		return x_mean
		
	def get_recon(self,inputs):
		n_layer = len(self.model_set)
		data = inputs
		for idx in reversed(range(n_layer)):
			data = self.model_set[idx].get_recon(data)
		
		# recon = torch.sigmoid(data)
		# tt = nn.ReLU()
		# recon = tt(data)

		recon = data
		return recon   
	
					
	def forward(self):
		pass


def train(dae, optimizer,lr_scheduler,dataloader,epoch_num,use_gpu=torch.cuda.is_available()):

	if use_gpu:
		dae = dae.cuda()
	
	for epoch in range(epoch_num):
		lr_scheduler.step()
		Total_loss = []
		Recon_loss = []
		dae.train()
		for batch_idx, batch in enumerate(dataloader):
			
			if use_gpu:
				inputs = Variable(batch[0].cuda())
			else:
				inputs = Variable(batch[0])

			# x_mean = dae.get_latent(getcorrupt(inputs,0.3))
			x_mean = dae.get_latent(inputs)
			
			# 
			# x_re = dae.get_recon(getcorrupt(x_mean,0.1))
			x_re = dae.get_recon(x_mean)
			# loss = -torch.sum(torch.mul(inputs,torch.log(x_re+1e-10))+torch.mul(1-inputs,torch.log(1-x_re+1e-10)))
		
			loss = torch.sum(torch.pow(inputs-x_re,2))
			# loss = loss/inputs.size(0)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			Total_loss.append(loss.item())
			# Recon_loss.append(ELBO_rec.item())

		# print('|Epoch:{} Total loss={:3f} Recon_loss={:3f}'.format(epoch,np.mean(Total_loss),-np.mean(Recon_loss)))
		print('|Epoch:{} Total loss={:3f}'.format(epoch,np.mean(Total_loss)))

	return dae



def cluster_acc(Y_pred, Y):
	assert Y_pred.size == Y.size
	D = max(Y_pred.max(), Y.max())+1
	w = np.zeros((D,D), dtype=np.int64)
	for i in range(Y_pred.size):
		w[Y_pred[i], Y[i]] += 1
	# ind = linear_assignment(w.max() - w)
	row_ind, col_ind = linear_sum_assignment(w.max() - w)
	# print(ind)
	return w[row_ind,col_ind].sum()/Y_pred.size, w
	# return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w



if __name__ == '__main__':
	# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	parser = argparse.ArgumentParser(description="DeepVQ - CIFAR10")
	parser.add_argument("--dataset", "-d", required=True, help='Dataset')
	parser.add_argument("--resume", "-r", default='', help="Checkpoint file for resume training")
	parser.add_argument("--code_length", "-L", required=True, type=int, help="Hash code length")
	parser.add_argument("--batch_size", "-b", default=200, type=int, help="Batch size")
	parser.add_argument("--num_epochs", "-e", default=5, type=int, help="Number of epochs")
	parser.add_argument("--gamma", "-g", default=0.5, type=float, help="Penalty parameter")
	args = parser.parse_args()

	BATCH_SIZE = args.batch_size
	# default test case
	L = args.code_length
	n_cls = L
	dataset = args.dataset
	pretrain_epochs = 250
	num_epochs = args.num_epochs
	
	act_flag = 1
	if act_flag:
		activation = nn.ReLU()
		# activation = nn.LeakyReLU(0.1)
	else:
		activation = nn.Tanh()
	
	
	resume = args.resume

	# Configure the network
	dataset = "reuters10k"
	# dataset == "mnist"

	if dataset == "mnist":
		encoder_sizes = [784, 500, 500, 2000, L]
		classifer_sizes = [L,L]

	elif dataset == "reuters10k":
		encoder_sizes = [2000, 500, 500, 2000, 10]
		classifer_sizes = [10,L]

	elif dataset == "har":
		encoder_sizes = [561, 500, 500, 2000, L]
		classifer_sizes = [L,L]


	path = 'data_ori/reuters10k/'
	data=scio.loadmat(path+'reuters10k.mat')
	X = data['X']
	Y = data['Y'].squeeze()

	image_train = X.astype('float32')
	label_train = Y.astype('float32')


	'''
	image_train = []

	for idx in range(1):
		data_temp = torch.from_numpy(torch.load('data_NN/reuters10k/reuters10k_{}_all_siamese.pkl'.format(idx))).unsqueeze(2)
		image_train.append(data_temp)

	image_train = to_numpy(torch.cat(image_train,dim=-1))
	# image_train = (image_train + 1)/2

	label_train = torch.load('data_NN/reuters10k/reuters10k_label_all_siamese.pkl')
	label_train = label_train.astype('int32')
	'''


	print('| Training data')
	print("| Data shape: {}".format(image_train.shape))
	print("| Data range: {}/{}".format(image_train.min(), image_train.max()))


	print("| Completed.")

	resume = 0
	use_trainedweight = 0
	layer_wised = 0
	print("2. Construct the VAE model")
	
	if not resume:
		if use_trainedweight:
			w_init = []
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_w_0.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_b_0.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_w_7.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_b_7.out',delimiter=',').astype(np.float32)))

			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_w_1.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_b_1.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_w_6.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_b_6.out',delimiter=',').astype(np.float32)))

			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_w_2.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_b_2.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_w_5.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_b_5.out',delimiter=',').astype(np.float32)))

			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_w_3.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_b_3.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_w_4.out',delimiter=',').astype(np.float32)))
			w_init.append(torch.tensor(np.loadtxt('./pretrained_weight/reuters10k/reuters10k_b_4.out',delimiter=',').astype(np.float32)))
			vae = VAE(encoder_sizes,image_train,activation,w_init=w_init)
			torch.save(vae,'pretrained_vae.pkl')

		else:
			if layer_wised == 0:
				dae = DAE(encoder_sizes,image_train,activation)
				torch.save(dae,'layerwisetrained_dae.pkl')
			else:
				dae = torch.load('layerwisetrained_dae.pkl')
			# vae = VAE(encoder_sizes,image_train,activation)

			dae = dae.cuda()
			gmm = mixture.GaussianMixture(n_components=n_cls,n_init=5)
			x_mean= dae.get_latent(torch.from_numpy(image_train).cuda())
			# print(x_mean.size())
			label_pred = gmm.fit_predict(to_numpy(x_mean))
			acc,_ = cluster_acc(label_pred,label_train)
			print(acc)


			dataloader = DataLoader(TensorDataset(torch.from_numpy(image_train)),batch_size=BATCH_SIZE,shuffle=True)
			optimizer = optim.Adam(dae.get_para(),lr=0.001)
			# optimizer = optim.SGD(vae.get_para(),lr=0.00001,momentum=0.9)
			lr_scheduler = StepLR(optimizer,step_size=70,gamma=0.1)
			print("2.1 pretrain the VAE model")
			dae = train(dae,optimizer,lr_scheduler,dataloader,epoch_num=1)
			torch.save(dae,'pretrained_dae.pkl')
	if resume:
		print("|Load pretrained model: {}".format(resume))
		dae = torch.load('pretrained_dae.pkl')

	dae = dae.cuda()
	gmm = mixture.GaussianMixture(n_components=n_cls,n_init=5)
	x_mean= dae.get_latent(torch.from_numpy(image_train).cuda())
	# print(x_mean.size())
	label_pred = gmm.fit_predict(to_numpy(x_mean))
	acc,_ = cluster_acc(label_pred,label_train)
	print(acc)

	kmeans = KMeans(n_clusters=n_cls,random_state=0).fit(to_numpy(x_mean))
	cls_index = kmeans.labels_
	mean = kmeans.cluster_centers_
	acc,_ = cluster_acc(cls_index,label_train)
	NMI = metrics.normalized_mutual_info_score(label_train, cls_index,average_method='arithmetic')
	print('| Kmeans ACC = {:6f} NMI = {:6f}'.format(acc,NMI))

	