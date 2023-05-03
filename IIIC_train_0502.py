
import os

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
from collections import OrderedDict

from sklearn.utils import class_weight

import itertools

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

print ("")
print ("done")
print ("")



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


    
    
class WeightedKLDivWithLogitsLoss(nn.KLDivLoss):
	def __init__(self, weight):
		super(WeightedKLDivWithLogitsLoss, self).__init__(size_average=None, reduce=None, reduction='none')
		self.register_buffer('weight', weight)

	def forward(self, input, target):
		# TODO: For KLDivLoss: input should 'log-probability' and target should be 'probability'
		# TODO: input for this method is logits, and target is probabilities
		batch_size = input.size(0)
		log_prob = F.log_softmax(input, 1)
		element_loss = super(WeightedKLDivWithLogitsLoss, self).forward(log_prob, target)

		sample_loss = torch.sum(element_loss, dim=1)
		sample_weight = torch.sum(target * self.weight, dim=1)

		weighted_loss = sample_loss*sample_weight
		# Average over mini-batch, not element-wise
		avg_loss = torch.sum(weighted_loss) / batch_size

		return avg_loss    
    
    
    
    
class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count        




def get_train_val_data(total_X, total_Y):
    N = total_X.shape[0]
    N1 = int(N*0.8)  # ratio for val
    N2 = N-N1
    #print (" N, N1, N2: ", N, N1, N2)
    sn_list = list(range(N))
    #print (sn_list[0:10])
    random.shuffle(sn_list)
    #print (sn_list[0:10])
    
    train_X = list()
    train_Y = list()    
    for n in range(N1):
        sn = sn_list[n]
        train_X.append(total_X[sn,:,:])
        train_Y.append(total_Y[sn,:])
    train_X = np.array(train_X)   
    train_Y = np.array(train_Y) 
    
    val_X = list()
    val_Y = list()    
    for n in range(N1,N):
        sn = sn_list[n]
        val_X.append(total_X[sn,:,:])
        val_Y.append(total_Y[sn,:])
    val_X = np.array(val_X)   
    val_Y = np.array(val_Y) 
    
    return train_X, train_Y, val_X, val_Y
    
    
def get_batch_sn_array(N, batch_size, epoch): 
    
    #print ("N: ", N)
    #print ("")
    
    total_sn_list = list()
    
    for n in range(N):
        
        total_sn_list.append(n)
        total_sn_list.append(-n)
        
    #print ("len(total_sn_list): ", len(total_sn_list))
    #print ("")
    
    #print (total_sn_list[0:10])
    
    
    seed_num = epoch
    
    #print ("seed_num: ", seed_num)
    #print ("")
    
    random.seed(seed_num)    
    
    random.shuffle(total_sn_list)
    
    
    result_list = list()
    
    N2 = int( len(total_sn_list)/batch_size )
    
    #print ("N2: ", N2)
    #print ("")
    
    for n in range(N2):
        
        start_sn = n*batch_size
        end_sn = (n+1)*batch_size
        
        result = list()
        
        for k in range(start_sn,end_sn):
            
            sn = total_sn_list[k]
            
            result.append(sn)
    
        result_list.append(result)
    
    result_array = np.array(result_list)

    return result_array


def get_batch_X_Y (train_X, train_Y, batch_sn_array):
    
    #print ("batch_sn_array.shape: ", batch_sn_array.shape)
    #print ("")
    
    #print (batch_sn_array)
    #print ("")
    
    K = batch_sn_array.shape[0]
    
    result_X = list()
    result_Y = list()
    
    for k in range(K):
        
        sn = batch_sn_array[k]
        
        if sn >= 0:
            
            sn = sn
            
            x = train_X[sn,:,:]
            
            y = train_Y[sn,:]
            
        else:
            
            sn = -sn
            
            x = np.array(train_X[sn,:,:])

            x2 = np.zeros((16, 2000))

            x2[0:4,:] = x[4:8,:]
            x2[4:8,:] = x[0:4,:]

            x2[8:12,:] = x[12:16,:]
            x2[12:16,:] = x[8:12,:]
            
            x = x2
            
            y = train_Y[sn,:]
    
        result_X.append(x)
        result_Y.append(y)
        
    result_X = np.array(result_X)
    result_Y = np.array(result_Y)

    #print ("result_X.shape: ", result_X.shape)
    #print ("result_Y.shape: ", result_Y.shape)
    #print ("")

    return (result_X,result_Y)



if __name__ == '__main__':
    
    
    
    train_all_X = np.load("../all_train_X.npy")
    train_all_Y = np.load("../all_train_Y2_hard.npy")

    print ("train_all_X.shape: ", train_all_X.shape)
    print ("train_all_Y.shape: ", train_all_Y.shape)
    print ("")
    
    
    train_10_X = np.load("../10_train_X.npy")
    train_10_Y = np.load("../10_train_Y2_hard.npy")

    print ("train_10_X.shape: ", train_10_X.shape)
    print ("train_10_Y.shape: ", train_10_Y.shape)
    print ("")
    
    train_X, train_Y, val_X, val_Y = get_train_val_data(train_10_X, train_10_Y)

    print ("train_X.shape: ", train_X.shape)
    print ("train_Y.shape: ", train_Y.shape)
    print ("")
    
    print ("val_X.shape: ", val_X.shape)
    print ("val_Y.shape: ", val_Y.shape)
    print ("")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print ("device: ", device)
    print ("")

    # model_cnn = torch.load("./previous_models/model_xxx.pt")
    model_cnn = DenseNetClassifier()

    model_cnn.to(device)

    #print (model_cnn)
    #print ("")
    
    train_label = np.argmax(train_Y,1)

    #print ( "Counter(train_label): ", Counter(train_label) )
    #print ("")

    train_W = class_weight.compute_class_weight('balanced',np.unique(train_label),train_label)

    #print ("train_W: ", train_W)
    #print ("")
    
    class_weight = train_W

    #print ("class_weight: ", class_weight)
    #print ("")

    class_weight_torch = torch.from_numpy(class_weight).float()

    criterion = WeightedKLDivWithLogitsLoss(class_weight_torch)

    criterion.to(device)

    #print ("criterion: ", criterion)
    #print ("")


    optimizer = optim.Adam(model_cnn.parameters(), lr=6.25*1e-5, betas = (0.9,0.999),eps = 1.0*1e-8, weight_decay=1.0*1e-3)

    #print ("optimizer: ", optimizer)
    #print ("")  
    
    batch_size = 32 
    
    time1 = time.time()    

    for epoch in range(20):

        print ("****************************************************************")
        print ("epoch: ", epoch)

        losses = AverageMeter()

        model_cnn.train()   


        total_batch_sn_array = get_batch_sn_array(train_X.shape[0], batch_size, epoch)

        print ("total_batch_sn_array.shape: ", total_batch_sn_array.shape)
        print ("")

        K = total_batch_sn_array.shape[0]

        print ("K: ", K)
        print ("")

        S_list = list()
        Y_list = list()

        for k in range(K):

            if k%100 == 0:
                print (k)

            batch_sn_array = total_batch_sn_array[k,:]
            (X,Y) = get_batch_X_Y(train_X, train_Y, batch_sn_array)

            X = X.astype("float64")
            Y = Y.astype("float64")

            X = torch.from_numpy(X).float()
            Y = torch.from_numpy(Y).float()

            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()

            output, _ = model_cnn(X)

            loss = criterion(output, Y)

            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            loss.backward()
            optimizer.step()

            losses.update(loss.item(), X.size(0))


            del X
            del Y
            del output


        #del X_batch_list
        #del Y_batch_list

        print ("losses.avg: ", losses.avg)
        print ("")


        time2 = time.time()  

        print("Duration: ", (time2-time1))
        print("")









