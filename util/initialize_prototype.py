import sys
sys.path.append('./util')
sys.path.append('./model')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2
import argparse
import os
import tensorflow as tf
from torchvision import models
from torch.utils.data import Dataset
from glob import glob
import json
import operator
import time
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import gc
from PIL import Image
from shutil import copyfile


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class cluster_dataset(Dataset):
    """ Data loader for processing images for deriving prototype with automatic
        clustering.
    """
	def __init__(self, img_dir, subj_name, transform=None):
		self.transform = transform
		self.init_data(img_dir,subj_name)

	def init_data(self, img_dir, subj_name):
		self.img = []
		self.img_id = []
		self.target = []
		for label in ['ASD','Ctrl']:
			files = glob(os.path.join(img_dir, label, '*.jpg'))
			for cur_img in files:
				cur_id = os.path.basename(cur_img).split('_')[0]
				if cur_id == subj_name:
					continue # excluding data for the test subject
				self.img.append(cur_img)
				self.img_id.append(os.path.basename(cur_img)[:-4])
				self.target.append(1 if label == 'ASD' else 0)

	def __getitem__(self,index):
		cur_img = self.img[index]
		cur_id = self.img_id[index]
		cur_img = Image.open(cur_img).convert('RGB')
		cur_label = self.target[index]
		if self.transform is not None:
			cur_img = self.transform(cur_img)

		return cur_img, cur_id, cur_label

	def __len__(self,):
		return len(self.img)


class proto_dataset(Dataset):
    """ Data loader for processing images for pre-processed prototype (after
        clustering as well as manual cleaning)
    """
	def __init__(self, img_dir, subj_name, transform=None):
		self.transform = transform
		self.init_data(img_dir, subj_name)

	def init_data(self, img_dir, subj_name):
		self.img = []
		files = glob(os.path.join(img_dir,'*.jpg'))

		for cur_img in files:
			cur_id = os.path.basename(cur_img).split('_')[0]
			if cur_id == subj_name:
				continue # excluding data for the test subject
			self.img.append(cur_img)


	def __getitem__(self,index):
		cur_img = self.img[index]
		cur_img = Image.open(cur_img).convert('RGB')
		if self.transform is not None:
			cur_img = self.transform(cur_img)

		return cur_img

	def __len__(self,):
		return len(self.img)


def find_prototype(subj_name,num_cluster):
    # deriving prototypes by clustering images based on their visual features
	probe_set = cluster_dataset('../autism_photo_taking', subj_name, transform)
	probe_loader = torch.utils.data.DataLoader(probe_set, batch_size=10, shuffle=False, num_workers=2)
	model = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1]).cuda()
	model.eval()

	asd_feat = []
	ctrl_feat = []
	start = time.time()
	for _, (img,img_id,labels) in enumerate(probe_loader):
		img = Variable(img).cuda()
		activation = model(img).squeeze(-1).squeeze(-1)
		activation = activation.data.cpu().numpy()

		for i in range(len(img)):
			if labels[i] == 1:
				asd_feat.append(activation[i])
			else:
				ctrl_feat.append(activation[i])

	gc.collect()
	torch.cuda.empty_cache()
	asd_feat = np.array(asd_feat)
	ctrl_feat = np.array(ctrl_feat)
	print('Finished extracting features, time spent %.2f seconds' %(time.time()-start))

	# find clusters for asd features
	kmeans = KMeans(n_clusters=num_cluster).fit(asd_feat)
	asd_labels = kmeans.labels_
	sil = silhouette_score(asd_feat,asd_labels,metric='euclidean')
	print('silhouette score for ASD clustering: %.3f' %sil)

	# find clusters for ctrl features
	kmeans = KMeans(n_clusters=num_cluster).fit(ctrl_feat)
	ctrl_labels = kmeans.labels_
	sil = silhouette_score(ctrl_feat,ctrl_labels,metric='euclidean')
	print('silhouette score for Control clustering: %.3f' %sil)

	# create prototype based on cluster centers
	asd_proptype = []
	ctrl_proptype = []

	for i in range(num_cluster):
		feat = asd_feat[asd_labels==i]
		asd_proptype.append(np.mean(feat,axis=0))
		feat = ctrl_feat[ctrl_labels==i]
		ctrl_proptype.append(np.mean(feat,axis=0))

	asd_proptype = np.array(asd_proptype).astype('float32')
	ctrl_proptype = np.array(ctrl_proptype).astype('float32')
	prototype = np.concatenate((asd_proptype,ctrl_proptype),axis=0)
	return prototype


def use_fix_prototype(subj_name):
    # computing the features for pre-defined prototypes
	prototype_dir = '../autism_photo_taking/prototypes'
	model = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1]).cuda()
	model.eval()
	prototype = []
	num_proto = 10
	for i in range(num_proto*2):
		cur_dir = os.path.join(prototype_dir,str(i+1))
		probe_set = proto_dataset(cur_dir,subj_name,transform)
		probe_loader = torch.utils.data.DataLoader(probe_set, batch_size=10, shuffle=False, num_workers=2)

		feat = []
		for _, img in enumerate(probe_loader):
			img = Variable(img).cuda()
			activation = model(img).squeeze(-1).squeeze(-1)
			activation = activation.data.cpu().numpy()

			feat.extend(activation)
		feat = np.array(feat).astype('float32')
		feat = (np.mean(feat,axis=0))
		prototype.append(feat)

	prototype = np.array(prototype)
	return prototype
