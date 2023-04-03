from PIL import Image
import os
import numpy as np
import torch.utils.data as data
import cv2
import gc
import torch
from glob import glob

class Dataset_category(data.Dataset):
	""" Dataloader for processing multiple images as inputs (for RNN or Transformer)
	"""
	def __init__(self,split,data,seq_len,transform,eval_sample=-1,eval_subj=None):
		self.split = split
		self.data = data
		self.eval_sample = eval_sample
		self.seq_len = seq_len
		self.transform = transform
		self.semantic_mask = np.load('./semantic_mask_super_non_overlap.npy',allow_pickle=True).item()
		self.idx2cat = {0:'indoor',1:'outdoor',2:'people'}
		self.idx2label = {0:'ctrl',1:'asd'}

		if self.split == 'val':
			# for a fair comparison, use a model to predict the scene categories
			self.predicted_category = np.load('predicted_scene.npy',allow_pickle=True).item()[eval_subj]
			# self.predicted_category = np.load('predicted_scene_ood.npy',allow_pickle=True).item()[eval_subj]

	def __getitem__(self,index):
		if self.split == 'train':
			data = self.data[list(self.data.keys())[index]]
		else:
			data = self.data

		img = data['img']
		target = data['label']

		if self.split == 'val':
			category = []
			for cur in img:
				cur = os.path.basename(cur)[:-4]
				if 'internet' in cur:
					category.append(self.predicted_category['_'.join(cur.split('_')[:2])])
				else:
					category.append(self.predicted_category[cur])
		else:
			category = data['category']

		cur_category = []
		img_id = []

		if len(img)>=self.seq_len:
			# if enough images for the current subject, sample without replacement
			cur_pool = np.random.choice(np.arange(len(img)),self.seq_len,replace=False)
		else:
			# otherwise, sample with replacement
			cur_pool = np.random.choice(np.arange(len(img)),self.seq_len,replace=True)

		for i,idx in enumerate(cur_pool):
			img_id.append(os.path.basename(img[idx])[:-4])
			cur_category.append(category[idx])
			cur_img = Image.open(img[idx]).convert('RGB')

			if self.transform is not None:
				cur_img = self.transform(cur_img)
			cur_img = cur_img.unsqueeze(0)

			# concatenating sequential images
			if i == 0:
				feat = cur_img
			else:
				feat = torch.cat((feat,cur_img),dim=0)

		cur_category = np.array(cur_category).astype('int')

		return feat, target, cur_category, img_id

	def __len__(self,):
		return len(self.data) if self.split == 'train' else self.eval_sample


class Dataset_single(data.Dataset):
	""" Dataloader for processing a single image as input (for CNN)
	"""
	def __init__(self,split,data,transform):
		self.split = split
		self.data = data
		self.transform = transform
		self.init_data()

	def init_data(self,):
		self.img = []
		self.target = []
		self.category = []
		if self.split == 'train':
			for subj in self.data:
				self.img.extend(self.data[subj]['img'])
				self.target.extend([self.data[subj]['label']]*len(self.data[subj]['img']))
				self.category.extend(self.data[subj]['category'])
		else:
			self.img = self.data['img']
			self.target = [self.data['label']]*len(self.data['img'])
			self.category = self.data['category']


	def __getitem__(self,index):
		img = self.img[index]
		target = self.target[index]
		category = self.category[index]
		img_id = os.path.basename(img)[:-4]
		img = Image.open(img).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)

		return img, target, category, img_id, category

	def __len__(self,):
		return len(self.img)
