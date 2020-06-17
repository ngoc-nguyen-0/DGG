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

def layerwise_train(model,dataloader,optimizer,lr_scheduler,epoch_num=50,use_gpu=torch.cuda.is_available()):
	if use_gpu:
		model = model.cuda()
	
	model.train()
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
			vis_recon = model(inputs)
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
	def __init__(self,layer_type,n_vis,n_hid,activation=nn.ReLU(),w_init=None):
		super(AE_layer,self).__init__()
		encoder_layers = []
		layer = nn.Linear(n_vis,n_hid)
		if w_init is not None:
			layer.weight.data = torch.t(w_init[0])
			layer.bias.data = w_init[1]
		# else:
			# nn.init.xavier_uniform_(layer.weight)
			# layer.weight.data = torch.rand(n_hid,n_vis)*1e-2
			# layer.bias.data = torch.zeros(n_hid)

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
			nn.init.xavier_uniform_(layer.weight)
			layer.bias.data = torch.zeros(n_vis)
		
		decoder_layers.append(layer)
		if (layer_type == "Last") or (layer_type == "Latent"):
			decoder_layers.append(activation)
		
		self.decoder_layers = nn.Sequential(*decoder_layers)

		
	def get_latent(self, inputs):
		latent = self.encoder_layers(inputs)
		return latent
	
	def get_recon(self,inputs):
		recon = self.decoder_layers(inputs)
		return recon
		
	def forward(self,inputs):
		latent = self.encoder_layers(inputs)
		outputs = self.decoder_layers(latent)
		return outputs


class VAE(nn.Module):
	def __init__(self,layer_sizes,inputs,activation=nn.ReLU(),w_init=None):
		super(VAE,self).__init__()
		n_layer = len(layer_sizes)
		BATCH_SIZE = 100
		dataloader = DataLoader(TensorDataset(torch.from_numpy(inputs)),batch_size=BATCH_SIZE,shuffle=True)
		output_data = torch.from_numpy(inputs).cuda()
		model_set = []
		for idx in range(n_layer-1):
			if idx == 0:
				layer_type = "First"
				lr = 0.001
			elif idx == n_layer-2:
				layer_type = "Last"
				lr = 0.0001
			else:
				layer_type = "Latent"
				lr = 0.001

			print("| Generate the {}th layer".format(idx+1))
			if w_init is not None:
				layer_model = AE_layer(layer_type,layer_sizes[idx],layer_sizes[idx+1],activation,w_init[4*idx:4*idx+4]).cuda()
			else:
				layer_model = AE_layer(layer_type,layer_sizes[idx],layer_sizes[idx+1],activation)
				optimizer = optim.SGD(layer_model.parameters(),lr=lr,momentum=0.9)
				# optimizer = optim.Adam(layer_model.parameters(),lr=lr,weight_decay=0.0001)
				lr_scheduler = StepLR(optimizer, step_size=70, gamma=0.1)
				layer_model = layerwise_train(layer_model,dataloader,optimizer,lr_scheduler)
			
			model_set.append(layer_model)

			output_data = layer_model.get_latent(output_data)
			dataloader = DataLoader(TensorDataset(output_data),batch_size=BATCH_SIZE,shuffle=True)

		self.model_set = model_set
		'''
		temp = []
		temp.append(torch.ones(layer_sizes[-2],layer_sizes[-1])/500)
		temp.append(torch.zeros(layer_sizes[-1]))
		temp.append(torch.ones(layer_sizes[-2],layer_sizes[-1])/500)
		temp.append(torch.zeros(layer_sizes[-2]))
		self.var_layer = AE_layer("Last",layer_sizes[-2],layer_sizes[-1],activation,temp)
		'''
		self.var_layer = AE_layer("Last",layer_sizes[-2],layer_sizes[-1])
		
	def get_para(self):
		n_layer = len(self.model_set)
		para = []
		for idx in range(n_layer):
			para.extend(self.model_set[idx].parameters())

		para.extend(self.var_layer.parameters())

		return para

	def get_latent(self,inputs):
		n_layer = len(self.model_set)
		data = inputs
		for idx in range(n_layer-1):
			data = self.model_set[idx].get_latent(data)
		
		x_mean = self.model_set[-1].get_latent(data)
		x_logvar = self.var_layer.get_latent(data)

		return x_mean, x_logvar
		
	def get_recon(self,inputs):
		n_layer = len(self.model_set)
		data = inputs
		for idx in reversed(range(n_layer)):
			data = self.model_set[idx].get_recon(data)
		
		# recon = torch.sigmoid(data)
		# tt = nn.ReLU()
		recon = data
		return recon   
	
					
	def forward(self):
		pass



def gen_x(mean,std,J):
	x_samples = []
	v_size = mean.size()
	for idx in range(J):
		x_samples.append(mean + torch.mul(std,torch.randn(v_size).cuda()))

	return x_samples




class Classifer(nn.Module):
	def __init__(self,layer_sizes,activation=nn.ReLU()):
		super(Classifer,self).__init__()
		n_layer = len(layer_sizes)
		layers = []
		for idx in range(n_layer-1):
			layer = nn.Linear(layer_sizes[idx],layer_sizes[idx+1])
			
			# layer.weight.data = torch.randn(layer_sizes[idx+1],layer_sizes[idx])*1e-2

			nn.init.xavier_normal_(layer.weight)
			# nn.init.xavier_normal_(layer.bias.data)
			layer.bias.data = torch.zeros(layer_sizes[idx+1])
			# layer.bias.data = torch.randn(layer_sizes[idx+1])*1e-2
			
			layers.append(layer)
			if idx < n_layer-2:
				layers.append(activation)
			else:
				layers.append(nn.Softmax())


		self.model = nn.Sequential(*layers)

	def forward(self,inputs):
		output = self.model(inputs)

		return output


class GMM_Model(nn.Module):
	def __init__(self,N,K,mean=None,var=None,prior=None):
		super(GMM_Model,self).__init__()
		if mean is not None:
			self.mean = nn.Parameter(torch.from_numpy(mean).view(1,N,K))
			self.std = nn.Parameter(torch.sqrt(torch.from_numpy(var)).view(1,N,K))
		else:
			self.mean = nn.Parameter(torch.randn(1,N,K))
			self.std = nn.Parameter(torch.ones(1,N,K))
		
		self.N = N
		self.K = K

	def get_para(self):

		return self.mean, self.std


	def log_prob(self,data_mean,data_logvar,cond_prob,weight):
		term1 = torch.sum(-torch.log((self.std**2)*2*math.pi),dim=1)*0.5
		term2 = torch.sum(-torch.div(torch.pow(data_mean.view(-1,self.N,1)-self.mean,2)+torch.exp(data_logvar).view(-1,self.N,1),self.std**2),dim=1)*0.5
		prob = term2 + term1
		log_p1 = torch.sum(torch.mul(prob,cond_prob),dim=-1)
		log_p = torch.sum(torch.mul(log_p1,weight))

		return log_p

	def compute_prob(self,data):
		prob = torch.exp(torch.sum(-torch.log((self.std**2)*2*math.pi)-torch.div(torch.pow(data.view(-1,self.N,1)-self.mean,2),self.std**2),dim=1)*0.5)
		pc = torch.div(prob,(torch.sum(prob,dim=-1)).view(-1,1)+1e-10)		
		return pc

	def compute_entropy(self,inputs,weight):
		entropy1 = torch.sum(-torch.mul(inputs,torch.log(inputs+1e-10)),dim=-1)
		entropy = torch.sum(torch.mul(entropy1,weight))

		return entropy

	def forward(self):
		pass




def pretrain(vae, optimizer,lr_scheduler,dataloader,epoch_num,use_gpu=torch.cuda.is_available()):

	if use_gpu:
		vae = vae.cuda()
	
	J = 1
	for epoch in range(epoch_num):
		lr_scheduler.step()
		Total_loss = []
		Recon_loss = []
		vae.train()
		for batch_idx, batch in enumerate(dataloader):
			
			if use_gpu:
				inputs = Variable(batch[0].cuda())
			else:
				inputs = Variable(batch[0])

			x_mean, x_logvar = vae.get_latent(inputs)
			
			ELBO_rec = 0
			

			x_re = vae.get_recon(x_mean)
			# ELBO_rec = ELBO_rec + torch.sum(torch.mul(inputs,torch.log(x_re+1e-10))+torch.mul(1-inputs,torch.log(1-x_re+1e-10)))
			ELBO_rec = ELBO_rec - torch.sum(torch.pow(inputs-x_re,2))
			
			# ELBO = ELBO
			# ELBO_reg = - 0.5 * torch.sum(x_mean**2 + torch.exp(x_logvar) - x_logvar - 1)
			
			ELBO_reg = 0
			loss = -ELBO_rec - ELBO_reg
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			Total_loss.append(loss.item())
			Recon_loss.append(ELBO_rec.item())

		print('|Epoch:{} Total loss={:3f} Recon_loss={:3f}'.format(epoch,np.mean(Total_loss),-np.mean(Recon_loss)))

	return vae

def pretrain_classifer(vae,GMM,classifer,optimizer,lr_scheduler,dataloader,epoch_num,use_gpu=torch.cuda.is_available()):

	if use_gpu:
		vae = vae.cuda()
		classifer= classifer.cuda()
		GMM = GMM.cuda()
	
	for epoch in range(epoch_num):
		lr_scheduler.step()
		Total_loss = []
		vae.eval()
		classifer.train()
		label = []
		label_pred = []
		entropy_p = 0
		for batch_idx, batch in enumerate(dataloader):
			
			if use_gpu:
				inputs = Variable(batch[0].cuda())
				label.append(batch[1])
			else:
				inputs = Variable(batch[0])

			x_mean, _ = vae.get_latent(inputs)
			prob_target = GMM.compute_prob(x_mean).data

			cond_prob = classifer(x_mean)

			loss = -torch.sum(torch.mul(prob_target,torch.log(cond_prob)))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			Total_loss.append(loss.item())

			vae.eval()
			classifer.eval()
			x_mean,_ = vae.get_latent(inputs.data)
			pred_l = torch.max(classifer(x_mean),dim=-1)
			label_pred.append(pred_l[-1])
			entropy_p = entropy_p + torch.sum(torch.pow(cond_prob,2)).item()
			# print(pred_l[-1])
			# assert 1 == 0

		label = torch.cat(label,dim=0)
		label_pred = torch.cat(label_pred,dim=0)
		NMI = metrics.normalized_mutual_info_score(to_numpy(label), to_numpy(label_pred),average_method='arithmetic')
		# NMI = metrics.normalized_mutual_info_score(to_numpy(label), to_numpy(label_pred),average_method='max')
		print('|Epoch:{} Total loss={:3f} NMI={:5f} Entropy={:2f}'.format(epoch,np.mean(Total_loss),NMI,entropy_p))

	return classifer


def pretrain_classifer1(vae,classifer,optimizer,lr_scheduler,dataloader,epoch_num,use_gpu=torch.cuda.is_available()):

	if use_gpu:
		vae = vae.cuda()
		classifer= classifer.cuda()
	loss_f = nn.CrossEntropyLoss()
	for epoch in range(epoch_num):
		lr_scheduler.step()
		Total_loss = []
		vae.eval()
		classifer.train()
		label = []
		label_pred = []
		entropy_p = 0
		for batch_idx, batch in enumerate(dataloader):
			
			if use_gpu:
				x_mean, x_logvar = vae.get_latent(batch[0].cuda())
				x_samples = gen_x(x_mean,torch.exp(0.5*x_logvar),1)
				# inputs = Variable(x_mean)
				inputs = Variable(x_samples[0])
				label_train = Variable(batch[2].cuda())
				label.append(batch[1])
			else:
				x_mean, x_logvar = vae.get_latent(batch[0])
				# x_samples = gen_x(x_mean,torch.exp(0.5*x_logvar),1)
				inputs = Variable(x_mean)
				label_train = Variable(batch[2])
				label.append(batch[1])

			
			# x_samples = gen_x(x_mean,torch.ones(10).cuda(),1)

			cond_prob = classifer(inputs)
			# cond_prob = classifer(x_samples[0])

			loss = loss_f(cond_prob,label_train)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			Total_loss.append(loss.item())

			# vae.eval()
			classifer.eval()
			# x_mean,_ = vae.get_latent(batch[0].cuda())
			pred_l = torch.max(classifer(inputs.data),dim=-1)
			label_pred.append(pred_l[-1])
			entropy_p = entropy_p + torch.sum(torch.pow(cond_prob,2)).item()
			# print(pred_l[-1])
			# assert 1 == 0

		label = torch.cat(label,dim=0)
		label_pred = torch.cat(label_pred,dim=0)
		NMI = metrics.normalized_mutual_info_score(to_numpy(label), to_numpy(label_pred),average_method='arithmetic')
		# NMI = metrics.normalized_mutual_info_score(to_numpy(label), to_numpy(label_pred),average_method='max')
		print('|Epoch:{} Total loss={:3f} NMI={:5f} Entropy={:2f}'.format(epoch,np.mean(Total_loss),NMI,entropy_p))

	return classifer


def cluster_acc(Y_pred, Y):
	assert Y_pred.size == Y.size
	D = max(Y_pred.max(), Y.max())+1
	w = np.zeros((D,D), dtype=np.int64)
	for i in range(Y_pred.size):
		w[Y_pred[i], Y[i]] += 1
	ind = linear_assignment(w.max() - w)
	return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def compute_weight(inputs,sigma,similarity_type='Gauss',use_gpu=torch.cuda.is_available()):
	dist = torch.sum(torch.pow(inputs-inputs[:,:,0].unsqueeze(2),2),dim=1)/sigma
	if similarity_type == 'Gauss':
		Gauss_simi = torch.exp(-dist)
		Gauss_simi[:,0] = torch.sum(Gauss_simi[:,1:],dim=1)
		simi = torch.div(Gauss_simi,torch.sum(Gauss_simi,dim=1,keepdim=True))
	elif similarity_type == 'Student-t':
		t_simi = torch(1,1+dist)
		t_simi[:,0] = torch.sum(t_simi[:,1:],dim=1)
		simi = torch.div(t_simi,torch.sum(t_simi,dim=1,keepdim=True))
	elif similarity_type == 'inv-Guass':
		Gauss_simi = torch.exp(dist)
		Gauss_simi[:,0] = torch.sum(Gauss_simi[:,1:],dim=1)
		simi = torch.div(Gauss_simi,torch.sum(Gauss_simi,dim=1,keepdim=True))
		simi[:,0] = simi[:,0]
	elif similarity_type == 'No_inform':
		N = inputs.size(-1)
		simi = torch.ones(1,N)/(N-1)
		simi[0,0] = 1
		simi = torch.mul(torch.ones(inputs.size(0),1),simi)
		simi = torch.div(simi,torch.sum(simi,dim=1,keepdim=True))
	return simi.cuda()





def train(vae,classifer,GMM,optimizer,lr_scheduler,dataloader,epoch_num,use_gpu=torch.cuda.is_available()):
	
	J = 1

	if use_gpu:
		vae = vae.cuda()
		classifer= classifer.cuda()
		GMM = GMM.cuda()

	for epoch in range(epoch_num):
		lr_scheduler.step()
		Total_loss = []
		label = []
		label_pred = []
		entropy_p = 0
		Recon_loss = 0
		variance_t = []
		for batch_idx, batch in enumerate(dataloader):
			

			N_samples = batch[0].size(0)
			input_dim = batch[0].size(1)
			N_n = batch[0].size(2)

			temp = batch[0]
			temp1 = []
			temp2 = []
			for idx in range(N_n):
				temp1.append(temp[:,:,idx])
				temp2.append(temp[:,:,0])

			temp1 = torch.cat(temp1,dim=0)
			temp2 = torch.cat(temp2,dim=0)
			
			if use_gpu:
				inputs = Variable(temp1.cuda())
				target = Variable(temp2.cuda())
				label.append(batch[1])
			else:
				inputs = Variable(temp1)
				target = Variable(temp2)
				label.append(batch[1])

			
			sigma = 150
			if epoch < 1500:
				weight = compute_weight(batch[0],sigma,similarity_type='No_inform')
				# weight = compute_weight(batch[0],sigma,similarity_type='Gauss')

			else:
				weight = compute_weight(batch[0],sigma,similarity_type='inv-Guass')
			weight_temp = []
			for idx in range(N_n):
				weight_temp.append(weight[:,idx])
			weight = torch.cat(weight_temp).cuda()

			# Compute the prior of c
			
			
			vae.eval()
			classifer.eval()
			x_mean,_ = vae.get_latent(inputs)
			pc = classifer(x_mean).data
			# pc = 0.25*torch.ones(pc.size()).cuda()

			pc = torch.mul(pc,weight.view(-1,1))
			pc_temp = 0
			for idx in range(N_n):
				pc_temp = pc_temp + pc[idx*N_samples:(idx+1)*N_samples,:]
			pc_temp1 = []
			for idx  in range(N_n):
				pc_temp1.append(pc_temp)

			pc = torch.cat(pc_temp1,dim=0)
			
			# Begin training

			vae.train()
			classifer.train()
			GMM.train()



			loss = 0

			x_mean, x_logvar = vae.get_latent(inputs)
			J = 1
			x_samples = gen_x(x_mean,torch.exp(0.5*x_logvar),J)
			ELBO = 0
			for idx in range(J):
				
				x_re = vae.get_recon(x_samples[idx])
				
				# log_recon = torch.sum(torch.mul(target,torch.log(x_re+1e-10))+torch.mul(1-target,torch.log(1-x_re+1e-10)),dim=1)
				log_recon = -0.5*torch.sum(torch.pow(target-x_re,2),dim=1)
				
				ELBO = ELBO + 0.1*torch.sum(torch.mul(log_recon,weight))
				# ELBO = ELBO - torch.sum(torch.mul(cond_prob,torch.log(cond_prob)))
				
				# print(torch.mean(cond_prob).data)
			ELBO = ELBO/J

			# cond_prob = GMM.compute_prob(gen_x(x_mean,torch.exp(0.5*x_logvar),1)[0]).data
			# cond_prob = GMM.compute_prob(x_samples[idx]).data
			# cond_prob = GMM.compute_prob(x_mean).data
			cond_prob = classifer(x_samples[0])

			# print("Recon_loss={:3f}".format(ELBO.item()))
			ELBO = ELBO + GMM.log_prob(x_mean,x_logvar,cond_prob,weight)
			# print("Recon_loss+GMM={:3f}".format(ELBO.item()))

			ELBO = ELBO + GMM.compute_entropy(cond_prob,weight)
			# temp = torch.mul(torch.sum(x_logvar,dim=-1),weight[:,ii])
			# print(temp.size())
			ELBO = ELBO + 0.5*torch.sum(torch.mul(torch.sum(x_logvar,dim=-1),weight))

			ELBO = ELBO + torch.sum(torch.mul(torch.sum(torch.mul(cond_prob,torch.log(pc)),dim=-1),weight))
			# print("Recon_loss+GMM+Entropy={:3f}".format(ELBO.item()))
			loss = loss - ELBO/N_samples

			# loss = loss + 0*0.5*GMM.reg()


			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			Total_loss.append(loss.item())

			vae.eval()
			classifer.eval()
			x_mean,_ = vae.get_latent(inputs[:N_samples,:].data)
			pred_l = torch.max(classifer(x_mean),dim=-1)
			label_pred.append(pred_l[-1])
			entropy_p = entropy_p + torch.sum(torch.pow(cond_prob,2)).item()
			
			# print(pred_l[-1])
			# assert 1 == 0
			x_re = vae.get_recon(x_mean)
			# re_loss = F.binary_cross_entropy(x_re,inputs[:N_samples,:])
			re_loss = F.mse_loss(x_re,inputs[:N_samples,:])
			Recon_loss = Recon_loss + re_loss.item()
			temp = to_numpy(torch.mean(GMM.std.data**2))
			# print(temp)
			variance_t.append(temp)
			# print(variance_t)

		label = to_numpy(torch.cat(label,dim=0))
		label_pred = to_numpy(torch.cat(label_pred,dim=0))

		# print(label_pred.size(),label.size())

		# NMI = 0.0
		NMI = metrics.normalized_mutual_info_score(label, label_pred,average_method='arithmetic')
		# NMI = metrics.normalized_mutual_info_score(to_numpy(label), to_numpy(label_pred),average_method='max')
		# variance_t = np.concatenate(variance_t,axis=0)
		acc,_ = cluster_acc(label_pred,label)
		print('|Epoch:{} Total loss={:3f} NMI={:5f} ACC={:6f} Entropy={:2f} Reconstruction Loss={:6f} GMM variance={:6f}'.format(epoch,np.mean(Total_loss),NMI,acc,entropy_p,Recon_loss,np.mean(variance_t)))

	return vae,classifer,GMM



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

	if dataset == "mnist":
		encoder_sizes = [784, 500, 500, 2000, L]
		classifer_sizes = [L,L]

	elif dataset == "reuters10k":
		encoder_sizes = [2000, 500, 500, 2000, 10]
		classifer_sizes = [10,L,L]
		# classifer_sizes = [10,L]

	elif dataset == "har":
		encoder_sizes = [561, 500, 500, 2000, L]
		classifer_sizes = [L,L]


	# Load training data


	image_train = []

	for idx in range(21):
		data_temp = torch.from_numpy(torch.load('data_NN/reuters10k/reuters10k_{}_all_siamese.pkl'.format(idx))).unsqueeze(2)
		image_train.append(data_temp)

	image_train = to_numpy(torch.cat(image_train,dim=-1))
	# image_train = image_train/3

	label_train = torch.load('data_NN/reuters10k/reuters10k_label_all_siamese.pkl')
	label_train = label_train.astype('int32')


	print('| Training data')
	print("| Data shape: {}".format(image_train.shape))
	print("| Data range: {}/{}".format(image_train.min(), image_train.max()))
	print(sum((label_train==0)*1.0))
	print(sum((label_train==1)*1.0))
	print(sum((label_train==2)*1.0))
	print(sum((label_train==3)*1.0))


	print("| Completed.")

	use_trainedweight = 1
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
			vae = VAE(encoder_sizes,image_train[:,:,0],activation,w_init=w_init)
			torch.save(vae,'pretrained_vae.pkl')

		else:
			vae = VAE(encoder_sizes,image_train[:,:,0],activation,w_init=w_init)
			dataloader = DataLoader(TensorDataset(torch.from_numpy(image_train[:,:,0])),batch_size=BATCH_SIZE,shuffle=True)
			optimizer = optim.Adam(vae.get_para(),lr=0.001)
			# optimizer = optim.SGD(vae.get_para(),lr=0.00001,momentum=0.9)
			lr_scheduler = StepLR(optimizer,step_size=30,gamma=0.1)
			print("2.1 pretrain the VAE model")
			vae = pretrain(vae,optimizer,lr_scheduler,dataloader,epoch_num=100)
			torch.save(vae,'pretrained_vae.pkl')
	if resume:
		print("|Load pretrained model: {}".format(resume))
		vae = torch.load('pretrained_vae.pkl')
	
	# vae = torch.load('trained_vae.pkl')
	vae = vae.cuda()
	x_mean,_ = vae.get_latent(torch.from_numpy(image_train[:,:,0]).cuda())

	kmeans = KMeans(n_clusters=L,random_state=0).fit(to_numpy(x_mean))
	cls_index = kmeans.labels_
	mean = kmeans.cluster_centers_
	acc,_ = cluster_acc(cls_index,label_train)
	NMI = metrics.normalized_mutual_info_score(label_train, cls_index,average_method='arithmetic')
	print('| Kmeans ACC = {:6f} NMI = {:6f}'.format(acc,NMI))

	var = []
	mean = torch.from_numpy(mean).cuda()
	for idx in range(L):
		index = np.where(cls_index==idx)
		var_g = torch.sum((x_mean[index[0],:]-mean[idx,:])**2,dim=0,keepdim=True)/(len(index[0])-1)
		var.append(var_g)
	var = torch.cat(var,dim=0)
	mean = to_numpy(mean)
	var = to_numpy(var)
	mean = mean.astype(np.float32)
	var = var.astype(np.float32)
	# print(var)

	# assert 1 == 0



	GMM = GMM_Model(10,L,mean.transpose(),var.transpose())
	GMM = GMM.cuda()
	label_pred = torch.max(GMM.compute_prob(x_mean),dim=-1)
	NMI = metrics.normalized_mutual_info_score(label_train, to_numpy(label_pred[-1]),average_method='arithmetic')
	acc,_ = cluster_acc(to_numpy(label_pred[-1]),label_train)
	print('| GMM ACC = {:6f} NMI = {:6f}'.format(acc,NMI))








	# assert 1 == 0 
	# print(mean)
	# print(var)
	classifer = Classifer(classifer_sizes)

	dataloader = DataLoader(TensorDataset(torch.from_numpy(image_train[:,:,0]),
									      torch.from_numpy(label_train),
									      torch.from_numpy(cls_index.astype(np.long))),batch_size=BATCH_SIZE,shuffle=True)
	# optimizer = optim.Adam(classifer.parameters(),lr=0.0001,weight_decay=0.0001)
	# optimizer = optim.SGD(classifer.parameters(),lr=0.001,momentum=0.9,weight_decay=0.0001)
	# optimizer = optim.SGD(classifer.parameters(),lr=0.001,momentum=0.9)
	# optimizer = optim.SGD(classifer.parameters(),lr=0.01,momentum=0.9)
	optimizer = optim.Adam(classifer.parameters(),lr=0.001)
	lr_scheduler = StepLR(optimizer,step_size=300,gamma=0.1)
	print("2.1 pretrain the classifer")
	# classifer= pretrain_classifer(vae,GMM, classifer,optimizer,lr_scheduler,dataloader,epoch_num=10)
	classifer= pretrain_classifer1(vae,classifer,optimizer,lr_scheduler,dataloader,epoch_num=50)


	dataloader = DataLoader(TensorDataset(torch.from_numpy(image_train),
										  torch.from_numpy(label_train)),
									 batch_size=BATCH_SIZE,shuffle=True)
	# optimizer = optim.Adam(list(vae.get_para())+list(classifer.parameters())+list(GMM.get_para()),lr=0.002,weight_decay=0.0001)
	# optimizer = optim.Adam(list(vae.get_para())+list(classifer.parameters())+list(GMM.get_para()),lr=0.001)
	optimizer = optim.SGD(list(vae.get_para())+list(classifer.parameters())+list(GMM.get_para()),lr=0.002,momentum=0.9)
	# optimizer = optim.Adam(list(vae.get_para())+list(classifer.parameters())+list(GMM.get_para()),lr=0.0005)
	# optimizer = optim.RMSprop(list(vae.get_para())+list(classifer.parameters())+list(GMM.get_para()),lr=0.002,weight_decay=0.0001)
	lr_scheduler = StepLR(optimizer,step_size=10,gamma=0.9)
	print("2.2 Train the model")
	vae,classifer,GMM = train(vae,classifer,GMM,optimizer,lr_scheduler,dataloader,epoch_num=300)
	vae.eval()
	classifer.eval()
	x_mean,_ = vae.get_latent(torch.from_numpy(image_train[:,:,0]).cuda())
	pred_l = torch.max(classifer(x_mean),dim=-1)
	acc,_ = cluster_acc(to_numpy(pred_l[-1]),label_train)
	print('|ACC={:6f}'.format(acc))
	# torch.save(vae,'trained_vae.pkl')


