import hdf5storage
import numpy as np
import re
from mne.filter import filter_data, notch_filter
import time
import numpy as np
import random
import re

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import os

print ("")
print ("lib finish")
print ("")


################################################## nets
class _DenseLayer(nn.Sequential):
	def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, conv_bias, batch_norm):
		super(_DenseLayer, self).__init__()
		if batch_norm:
			self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
		# self.add_module('relu1', nn.ReLU()),
		self.add_module('elu1', nn.ELU()),
		self.add_module('conv1', nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=conv_bias)),
		if batch_norm:
			self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
		# self.add_module('relu2', nn.ReLU()),
		self.add_module('elu2', nn.ELU()),
		self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=conv_bias)),
		# self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=7, stride=1, padding=3, bias=conv_bias)),
		self.drop_rate = drop_rate

	def forward(self, x):
		# print("Dense Layer Input: ")
		# print(x.size())
		new_features = super(_DenseLayer, self).forward(x)
		# print("Dense Layer Output:")
		# print(new_features.size())
		if self.drop_rate > 0:
			new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
		return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
	def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, conv_bias, batch_norm):
		super(_DenseBlock, self).__init__()
		for i in range(num_layers):
			layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, conv_bias, batch_norm)
			self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
	def __init__(self, num_input_features, num_output_features, conv_bias, batch_norm):
		super(_Transition, self).__init__()
		if batch_norm:
			self.add_module('norm', nn.BatchNorm1d(num_input_features))
		# self.add_module('relu', nn.ReLU())
		self.add_module('elu', nn.ELU())
		self.add_module('conv', nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=conv_bias))
		self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))


class DenseNetEnconder(nn.Module):
	def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4, 4, 4, 4),  #block_config=(6, 12, 24, 48, 24, 20, 16),  #block_config=(6, 12, 24, 16),
				 in_channels=16, num_init_features=64, bn_size=4, drop_rate=0.2, conv_bias=True, batch_norm=False):

		super(DenseNetEnconder, self).__init__()

		# First convolution
		first_conv = OrderedDict([('conv0', nn.Conv1d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=conv_bias))])
		# first_conv = OrderedDict([('conv0', nn.Conv1d(in_channels, num_init_features, groups=in_channels, kernel_size=7, stride=2, padding=3, bias=conv_bias))])
		# first_conv = OrderedDict([('conv0', nn.Conv1d(in_channels, num_init_features, kernel_size=15, stride=2, padding=7, bias=conv_bias))])

		# first_conv = OrderedDict([
		# 	('conv0-depth', nn.Conv1d(in_channels, 32, groups=in_channels, kernel_size=7, stride=2, padding=3, bias=conv_bias)),
		# 	('conv0-point', nn.Conv1d(32, num_init_features, kernel_size=1, stride=1, bias=conv_bias)),
		# ])

		if batch_norm:
			first_conv['norm0'] = nn.BatchNorm1d(num_init_features)
		# first_conv['relu0'] = nn.ReLU()
		first_conv['elu0'] = nn.ELU()
		first_conv['pool0'] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

		self.densenet = nn.Sequential(first_conv)

		num_features = num_init_features
		for i, num_layers in enumerate(block_config):
			block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
								bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, conv_bias=conv_bias, batch_norm=batch_norm)
			self.densenet.add_module('denseblock%d' % (i + 1), block)
			num_features = num_features + num_layers * growth_rate
			if i != len(block_config) - 1:
				trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, conv_bias=conv_bias, batch_norm=batch_norm)
				self.densenet.add_module('transition%d' % (i + 1), trans)
				num_features = num_features // 2

		# Final batch norm
		if batch_norm:
			self.densenet.add_module('norm{}'.format(len(block_config) + 1), nn.BatchNorm1d(num_features))
		# self.features.add_module('norm5', BatchReNorm1d(num_features))

		self.densenet.add_module('relu{}'.format(len(block_config) + 1), nn.ReLU())
		self.densenet.add_module('pool{}'.format(len(block_config) + 1), nn.AvgPool1d(kernel_size=7, stride=3))  # stride originally 1

		self.num_features = num_features

		# Official init from torch repo.
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight.data)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def forward(self, x):
		features = self.densenet(x)
		# print("Final Output")
		# print(features.size())
		return features.view(features.size(0), -1)


class DenseNetClassifier(nn.Module):
	# def __init__(self, growth_rate=16, block_config=(3, 6, 12, 8),  #block_config=(6, 12, 24, 48, 24, 20, 16),  #block_config=(6, 12, 24, 16),
	# 			 in_channels=16, num_init_features=32, bn_size=2, drop_rate=0, conv_bias=False, drop_fc=0.5, num_classes=6):
	def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4, 4, 4, 4),
				 in_channels=16, num_init_features=64, bn_size=4, drop_rate=0.2, conv_bias=True, batch_norm=False, drop_fc=0.5, num_classes=6):

		super(DenseNetClassifier, self).__init__()

		self.features = DenseNetEnconder(growth_rate=growth_rate, block_config=block_config, in_channels=in_channels,
										 num_init_features=num_init_features, bn_size=bn_size, drop_rate=drop_rate,
										 conv_bias=conv_bias, batch_norm=batch_norm)

		# Linear layer
		self.classifier = nn.Sequential(
			nn.Dropout(p=drop_fc),
			nn.Linear(self.features.num_features, num_classes)
		)

		# Official init from torch repo.
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight.data)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def forward(self, x):
		features = self.features(x)
		out = self.classifier(features)
		return out, features

    
device = "cuda" if torch.cuda.is_available() else "cpu"

print ("device: ", device)
print ("")

model_cnn = torch.load("./Tools/model_1130.pt", map_location=torch.device('cpu'))
model_cnn.eval()


      
##################################################################################################################################      

print ("")
print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print ("read data")
print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print ("")    
    
######################################################################################



# file_name_array = np.loadtxt("file_path_24h.txt", delimiter = ',', dtype = np.str)

total_file_list = os.listdir("./Data/Raw/")

print ("len(total_file_list): ", len(total_file_list))
print ("")

T = len(total_file_list)

print ("T: ", T)
print ("")


for t in range(T):
    
    print ("***************************************")
    print ("t: ", t)
    print ("")
    
    
    file_name = total_file_list[t]
    save_name = file_name.rstrip(".mat")

    if os.path.isfile("./Data/iiic/"+ save_name + "_score.csv"):
        print('--alr done ' + save_name)

    else:
        print ("file_name: ", file_name)
        print ("")
        print ("save_name: ", save_name)
        print ("")

        path1 = "./Data/Raw/" + file_name
        print ("path1: ", path1)
        print ("")
    
        mat = hdf5storage.loadmat(path1)
        print (mat.keys())
        print ("")

        X = mat['data']
        print ("X.shape: ", X.shape)
        print ("")

        print ("************ Montage")
        print ("")

        X2 = X[[0,4,5,6, 11,15,16,17, 0,1,2,3, 11,12,13,14]] - X[[4,5,6,7, 15,16,17,18, 1,2,3,7, 12,13,14,18]]
        print ("X2.shape: ", X2.shape)

        print ("************ filtering")
        print ("")

        X2 = notch_filter(X2, 200, 60, n_jobs=-1, verbose='ERROR')
        X2 = filter_data(X2, 200, 0.5, 40, n_jobs=-1, verbose='ERROR') 

        print ("X2.shape: ", X2.shape)
        N = int(X2.shape[1]/400)

        print ("N: ", N)
        print ("")

        print ("************ reshaping")
        print ("")

        X3 = np.zeros((N-5,16,2000))

        for n in range(N-5):
            start_sn = n*400
            end_sn = start_sn + 2000
            x = X2[:,start_sn:end_sn]
            X3[n,:,:] = x

        print ("X3.shape: ", X3.shape)

        X = X3

        print ("X.shape: ", X.shape)
        print ("")
        print ("np.isnan(X).sum(): ", np.isnan(X).sum())
        print ("np.max(X): ", np.max(X))
        print ("np.min(X): ", np.min(X))
        print ("")

        X = np.where(X<=500, X, 500)
        X = np.where(X>=-500, X, -500)

        print ("X.shape: ", X.shape)
        print ("")
        print ("np.isnan(X).sum(): ", np.isnan(X).sum())
        print ("np.max(X): ", np.max(X))
        print ("np.min(X): ", np.min(X))
        print ("")

        X4 = X

        del X
        del X2
        del X3

        print ("X4.shape: ", X4.shape)
        print ("")


        ######################### evaluation
        batch_size = 1000
        def get_unlabeled_batch_list(X_train,batch_size):
            N = X_train.shape[0]
            sn_list = list(range(N))
            K = int(N/batch_size)
            X_list = list()
            end_sn = 0

            for k in range(K):
                start_sn = k*batch_size
                end_sn = start_sn + batch_size

                X = X_train[start_sn:end_sn,:,:]
                X_list.append(X)   
            if not end_sn == N:
                X = X_train[end_sn:N,:,:]
                X_list.append(X)   

            return (X_list)

        ################### scanning
        model_cnn.eval() #*
        # print ("unlabeled losses.avg: ", losses.avg)

        (X_batch_list) = get_unlabeled_batch_list(X4,batch_size)
        K = len(X_batch_list)

        print ("K: ", K)
        print ("")

        S_list = list() 
        V_list = list()

        for k in range(K):
            if k%100 == 0:
                print (k)

            X = X_batch_list[k]

            #print ("X.shape: ", X.shape)
            #print ("Y.shape: ", Y.shape)

            X = torch.from_numpy(X).float()
            X = X.to(device)

            output, v = model_cnn(X)

            S_list.append(output.detach().to('cpu'))
            V_list.append(v.detach().to('cpu'))

            del X
            del output
            del v


        #print ("unlabeled losses.avg: ", losses.avg)
        #print ("")

        S2 = torch.cat(S_list,dim=0)
        prob = F.softmax(S2, 1)
        unlabeled_score = prob.numpy()
    
        print ("")
        print ("unlabeled_score.shape: ", unlabeled_score.shape)
        print ("")

        V2 = torch.cat(V_list,dim=0)
        unlabeled_V = V2.numpy()

        print ("unlabeled_V.shape: ", unlabeled_V.shape)
        print ("")

        path3 = "./Data/iiic/"+ save_name + "_score.csv"
        np.savetxt(path3, unlabeled_score, delimiter=',')
    
        #path4 = "./Data/iiic/"+ save_name + "_vector.csv"
        #np.savetxt(path4, unlabeled_V, delimiter=',')

        print ("writing finish")
        print ("")

        del X4
        del X_batch_list
    
    print ("")
    print ("Done!")
    print ("")
    