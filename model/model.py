import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import numpy as np

class LSTM(nn.Module):
	"""
	Original LSTM model
	"""
	def __init__(self,embed_size=512):
		super(LSTM,self).__init__()
		self.input_x = nn.Linear(embed_size,embed_size,bias=True)
		self.forget_x = nn.Linear(embed_size,embed_size,bias=True)
		self.output_x = nn.Linear(embed_size,embed_size,bias=True)
		self.memory_x = nn.Linear(embed_size,embed_size,bias=True)
		self.input_h = nn.Linear(embed_size,embed_size,bias=True)
		self.forget_h = nn.Linear(embed_size,embed_size,bias=True)
		self.output_h = nn.Linear(embed_size,embed_size,bias=True)
		self.memory_h = nn.Linear(embed_size,embed_size,bias=True)

	def forward(self,x,state):
		h, c = state
		i = F.sigmoid(self.input_x(x) + self.input_h(h))
		f = F.sigmoid(self.forget_x(x) + self.forget_h(h))
		o = F.sigmoid(self.output_x(x) + self.output_h(h))

		g = F.tanh(self.memory_x(x) + self.memory_h(h))

		next_c = torch.mul(f,c) + torch.mul(i,g)
		h = torch.mul(o,next_c)
		state = (h,next_c)

		return state

class G_LSTM(nn.Module):
	"""
	LSTM implementation proposed by A. Graves (2013),
	it has more parameters compared to original LSTM
	"""
	def __init__(self,embed_size=512):
		super(G_LSTM,self).__init__()
		self.input_x = nn.Linear(embed_size,embed_size,bias=True)
		self.forget_x = nn.Linear(embed_size,embed_size,bias=True)
		self.output_x = nn.Linear(embed_size,embed_size,bias=True)
		self.memory_x = nn.Linear(embed_size,embed_size,bias=True)
		self.input_h = nn.Linear(embed_size,embed_size,bias=True)
		self.forget_h = nn.Linear(embed_size,embed_size,bias=True)
		self.output_h = nn.Linear(embed_size,embed_size,bias=True)
		self.memory_h = nn.Linear(embed_size,embed_size,bias=True)
		self.input_c = nn.Linear(embed_size,embed_size,bias=True)
		self.forget_c = nn.Linear(embed_size,embed_size,bias=True)
		self.output_c = nn.Linear(embed_size,embed_size,bias=True)

	def forward(self,x,state):
		h, c = state
		i = F.sigmoid(self.input_x(x) + self.input_h(h) + self.input_c(c))
		f = F.sigmoid(self.forget_x(x) + self.forget_h(h) + self.forget_c(c))
		g = F.tanh(self.memory_x(x) + self.memory_h(h))

		next_c = torch.mul(f,c) + torch.mul(i,g)
		o = F.sigmoid(self.output_x(x) + self.output_h(h) + self.output_c(next_c))
		h = torch.mul(o,next_c)
		state = (h,next_c)

		return state


class GRU(nn.Module):
	"""
	Gated Recurrent Unit without long-term memory
	"""
	def __init__(self,embed_size=512):
		super(GRU,self).__init__()
		self.update_x = nn.Linear(embed_size,embed_size,bias=True)
		self.update_h = nn.Linear(embed_size,embed_size,bias=True)
		self.reset_x = nn.Linear(embed_size,embed_size,bias=True)
		self.reset_h = nn.Linear(embed_size,embed_size,bias=True)
		self.memory_x = nn.Linear(embed_size,embed_size,bias=True)
		self.memory_h = nn.Linear(embed_size,embed_size,bias=True)

	def forward(self,x,state):
		z = F.sigmoid(self.update_x(x) + self.update_h(state))
		r = F.sigmoid(self.reset_x(x) + self.reset_h(state))
		mem = F.tanh(self.memory_x(x) + self.memory_h(torch.mul(r,state)))
		state = torch.mul(1-z,state) + torch.mul(z,mem)
		return state

class ASD_CNN(nn.Module):
	""" Prototypical ASD screening with CNN backbone
	"""
	def __init__(self, backend, hidden_size=512, use_prototype=False,
				init_prototype=None, num_prototype=10):
		super(ASD_CNN, self).__init__()
		# defining backend
		self.backend_name = backend
		if backend == 'resnet':
			resnet = models.resnet50(pretrained=True)
			self.init_resnet(resnet)
			input_size = 2048
		elif backend == 'vgg':
			vgg = models.vgg19(pretrained=True)
			self.init_vgg(vgg)
			input_size = 512
		else:
			assert 0, 'Backend not implemented'

		# without additional conv layer
		self.v_encoder = nn.Linear(input_size, hidden_size)

		self.decoder = nn.Linear(hidden_size,1,bias=True)
		self.category_embedding = nn.Embedding(num_embeddings=3, embedding_dim=hidden_size)

		self.use_prototype = use_prototype
		if use_prototype:
			self.num_prototype = num_prototype
			if init_prototype is None:
				self.prototype = nn.Parameter(torch.rand(
								2*self.num_prototype, input_size), requires_grad=True)
			else:
				self.prototype = nn.Parameter(torch.from_numpy(init_prototype),requires_grad=True)
			self.dist_decoder = nn.Linear(2*self.num_prototype, 1) # utilizing prototype to complement feature-based predictions
			self.gate = nn.Linear(input_size, 1)


		#fixing saliency module
		for para in self.backend.parameters():
			para.requires_grad = True # fixing pretrained modules or not

	def init_resnet(self,resnet):
		self.backend = nn.Sequential(*list(resnet.children())[:-2])

	def init_vgg(self,vgg):
		# self.backend = vgg.features
		self.backend = nn.Sequential(*list(vgg.features.children())[:-2]) # omitting the last Max Pooling


	def forward(self,x,category=None):
		x = self.backend(x)
		batch, c, w, h = x.shape
		if self.backend_name == 'resnet':
			# without additional conv layer
			v_feat = F.avg_pool2d(x,(7,7)) # Global Average Pooling for resnet
		else:
			# without additional conv layer
			v_feat = F.avg_pool2d(x,(14,14)) # Global Average Pooling for vgg19

		v_feat = v_feat.view(batch,-1)

		if category is not None:
			# initialize the categorical embedding
			category = self.category_embedding(category)
			# merge position embedding with visual features
			fuse_feat = torch.relu(self.v_encoder(v_feat))
			fuse_feat = fuse_feat + category
		else:
			fuse_feat = torch.relu(self.v_encoder(v_feat))

		output = F.sigmoid(self.decoder(F.dropout(fuse_feat,0.5)))

		if self.use_prototype:
			batch,feat = v_feat.shape
			prototype = self.prototype.unsqueeze(0).expand(batch,2*self.num_prototype,feat)

			# # euclidean distance
			dist = ((v_feat.unsqueeze(1)-prototype)**2).sum(-1)
			dist_feat = torch.log((dist+1)/(dist+1e-8))

			# using distance as features
			dist_pred = torch.sigmoid(self.dist_decoder(dist_feat.view(batch,-1)))

			# with adaptive gate
			gate = torch.sigmoid(self.gate(v_feat))
			output = output*gate + dist_pred*(1-gate)

			return output, v_feat, dist, x
		else:
			return output, v_feat, x

	def get_intermediate_grad(self,v_feat,category=None):
		if category is not None:
			# initialize the categorical embedding
			category = self.category_embedding(category)
			# merge position embedding with visual features
			fuse_feat = torch.relu(self.v_encoder(v_feat))
			fuse_feat = fuse_feat + category
		else:
			fuse_feat = torch.relu(self.v_encoder(v_feat))

		output = F.sigmoid(self.decoder(F.dropout(fuse_feat,0.5)))

		if self.use_prototype:
			batch,feat = v_feat.shape
			prototype = self.prototype.unsqueeze(0).expand(batch,2*self.num_prototype,feat)

			# # euclidean distance
			dist = ((v_feat.unsqueeze(1)-prototype)**2).sum(-1)
			dist_feat = torch.log((dist+1)/(dist+1e-8))

			# using distance as features
			dist_pred = torch.sigmoid(self.dist_decoder(dist_feat.view(batch,-1)))

			# with adaptive gate
			gate = torch.sigmoid(self.gate(v_feat))
			output = output*gate + dist_pred*(1-gate)

		return output


class ASD_RNN(nn.Module):
	def __init__(self,backend, seq_len, hidden_size=512, use_prototype=False,
				init_prototype=None, num_prototype=10):
		super(ASD_RNN,self).__init__()
		self.seq_len = seq_len
		# defining backend
		self.backend_name = backend
		if backend == 'resnet':
			resnet = models.resnet50(pretrained=True)
			self.init_resnet(resnet)
			input_size = 2048
		elif backend == 'vgg':
			vgg = models.vgg19(pretrained=True)
			self.init_vgg(vgg)
			input_size = 512
		else:
			assert 0, 'Backend not implemented'

		# without additional conv layer
		self.v_encoder = nn.Linear(input_size,hidden_size)

		self.rnn = LSTM(hidden_size)

		self.decoder = nn.Linear(hidden_size,1,bias=True)
		self.hidden_size = hidden_size
		self.category_embedding = nn.Embedding(num_embeddings=3,embedding_dim=hidden_size)

		self.use_prototype = use_prototype

		if use_prototype:
			self.num_prototype = num_prototype
			if init_prototype is None:
				self.prototype = nn.Parameter(torch.rand(
							2*self.num_prototype, input_size),requires_grad=True)
			else:
				self.prototype = nn.Parameter(torch.from_numpy(init_prototype),requires_grad=True)
			self.dist_decoder = nn.Linear(2*self.num_prototype*self.seq_len, 1) # utilizing prototype to complement feature-based predictions
			self.gate = nn.Linear(input_size,1)

		#fixing saliency module
		for para in self.backend.parameters():
			para.requires_grad = True # fixing pretrained modules or not

	def init_resnet(self,resnet):
		self.backend = nn.Sequential(*list(resnet.children())[:-2])

	def init_vgg(self,vgg):
		# self.backend = vgg.features
		self.backend = nn.Sequential(*list(vgg.features.children())[:-2]) # omitting the last Max Pooling

	def init_hidden(self,x): #initializing hidden state as all zero
		h = x.data.new().resize_as_(x.data).fill_(0)
		c = x.data.new().resize_as_(x.data).fill_(0)
		return (Variable(h),Variable(c))

	def forward(self,x,category=None):
		batch, seq, c, h, w = x.size()
		x = x.view(batch*seq,c,h,w)
		x = self.backend(x)
		if self.backend_name == 'resnet':
			v_feat = F.avg_pool2d(x,(7,7)) # Global Average Pooling for resnet
		else:
			v_feat = F.avg_pool2d(x,(14,14)) # Global Average Pooling for vgg19

		v_feat = v_feat.view(batch,seq,-1)

		if category is not None:
			# initialize the categorical embedding
			category = self.category_embedding(category)
			# merge position embedding with visual features
			fuse_feat = torch.relu(self.v_encoder(v_feat))
			fuse_feat = fuse_feat + category
		else:
			fuse_feat = torch.relu(self.v_encoder(v_feat))

		state = self.init_hidden(fuse_feat[:,0,:]) # initialize hidden state

		for i in range(self.seq_len):
			cur_x = fuse_feat[:,i,:].contiguous()
			state = self.rnn(cur_x,state)

		h, c = state
		output = F.sigmoid(self.decoder(F.dropout(h,p=0.5)))

		if self.use_prototype:

			batch,seq_len,feat = v_feat.shape
			prototype = self.prototype.unsqueeze(0).unsqueeze(1).expand(batch,seq_len,2*self.num_prototype,feat)

			# # euclidean distance
			dist = ((v_feat.unsqueeze(2)-prototype)**2).sum(-1)
			dist_feat = torch.log((dist+1)/(dist+1e-8))

			# using distance as features
			dist_pred = torch.sigmoid(self.dist_decoder(dist_feat.view(batch,-1)))

			# with adaptive gate
			gate = torch.sigmoid(self.gate(v_feat.mean(1)))
			output = output*gate + dist_pred*(1-gate)

			return output, v_feat, dist, x.view(batch,seq,-1,7,7)
		else:
			return output, v_feat, x.view(batch,seq,-1,7,7)

	def get_intermediate_grad(self,v_feat,category=None):

		if category is not None:
			# initialize the categorical embedding
			category = self.category_embedding(category)
			# merge position embedding with visual features
			fuse_feat = torch.relu(self.v_encoder(v_feat))
			fuse_feat = fuse_feat + category
		else:
			fuse_feat = torch.relu(self.v_encoder(v_feat))

		state = self.init_hidden(fuse_feat[:,0,:]) # initialize hidden state

		for i in range(self.seq_len):
			cur_x = fuse_feat[:,i,:].contiguous()
			state = self.rnn(cur_x,state)

		h, c = state
		output = F.sigmoid(self.decoder(F.dropout(h,p=0.5)))

		if self.use_prototype:

			batch,seq_len,feat = v_feat.shape
			prototype = self.prototype.unsqueeze(0).unsqueeze(1).expand(batch,seq_len,2*self.num_prototype,feat)

			# # euclidean distance
			dist = ((v_feat.unsqueeze(2)-prototype)**2).sum(-1)
			dist_feat = torch.log((dist+1)/(dist+1e-8))

			# using distance as features
			dist_pred = torch.sigmoid(self.dist_decoder(dist_feat.view(batch,-1)))

			# with adaptive gate
			gate = torch.sigmoid(self.gate(v_feat.mean(1)))
			output = output*gate + dist_pred*(1-gate)

		return output


class ASD_transformer(nn.Module):
	""" Prototypical ASD screening with Transformer backbone
	"""
	def __init__(self, backend, seq_len, hidden_size=512, num_prototype=12, init_prototype=None):
		super(ASD_transformer,self).__init__()
		self.seq_len = seq_len
		# defining backend
		self.backend_name = backend
		if backend == 'resnet':
			resnet = models.resnet50(pretrained=True)
			self.init_resnet(resnet)
			input_size = 2048
		elif backend == 'vgg':
			vgg = models.vgg19(pretrained=True)
			self.init_vgg(vgg)
			input_size = 512
		else:
			assert 0, 'Backend not implemented'

		self.num_prototype = num_prototype

		self.transformer_key = nn.Linear(hidden_size,hidden_size)
		self.transformer_query = nn.Linear(hidden_size,hidden_size)
		self.transformer_value = nn.Linear(hidden_size,hidden_size)

		self.decoder = nn.Linear(hidden_size,1,bias=True)
		self.hidden_size = hidden_size
		self.category_embedding = nn.Embedding(num_embeddings=3,embedding_dim=hidden_size)

		if init_prototype is None:
			self.prototype = nn.Parameter(torch.rand(2*self.num_prototype,input_size),requires_grad=False)
		else:
			self.prototype = nn.Parameter(torch.from_numpy(init_prototype))
		# self.dist2feat = nn.Linear(2*self.num_prototype+hidden_size,hidden_size)
		self.dist2feat = nn.Linear(2*self.num_prototype,hidden_size) # for pure distance-based inference

		#fixing feature extraction module
		for para in self.backend.parameters():
			para.requires_grad = True # fixing pretrained modules or not

	def init_resnet(self,resnet):
		self.backend = nn.Sequential(*list(resnet.children())[:-2])

	def init_vgg(self,vgg):
		self.backend = nn.Sequential(*list(vgg.features.children())[:-2]) # omitting the last Max Pooling

	def forward(self,x,category=None,):
		batch, seq, c, h, w = x.size()
		x = x.view(batch*seq,c,h,w)
		x = self.backend(x)
		if self.backend_name == 'resnet':
			# without additional conv layer
			x = F.avg_pool2d(x,(7,7)) # Global Average Pooling for resnet
		else:
			# without additional conv layer
			x = F.avg_pool2d(x,(14,14)) # Global Average Pooling for vgg19

		x = x.view(batch,seq,-1)

		# use the distance between fixed features and prototypes as features
		batch,seq_len,feat = x.shape
		prototype = self.prototype.unsqueeze(0).unsqueeze(1).expand(batch,seq_len,2*self.num_prototype,feat)

		# euclidean distance
		dist = ((x.unsqueeze(2)-prototype)**2).sum(-1)
		dist_feat = torch.log((dist+1)/(dist+1e-8))

		if category is not None:
			# initialize the categorical embedding
			category = self.category_embedding(category)
			fuse_feat = torch.relu(self.dist2feat(dist_feat))
			fuse_feat = fuse_feat + category
		else:
			# pure distance-based features
			fuse_feat = torch.relu(self.dist2feat(dist_feat))

		# self-attention with dot-product
		key = self.transformer_key(fuse_feat)
		query = self.transformer_query(fuse_feat)
		att = F.softmax(torch.bmm(query,key.transpose(1,2).contiguous())/np.sqrt(self.hidden_size),dim=-1) # start with single-head attention
		value = torch.bmm(att,fuse_feat)
		value = torch.relu(self.transformer_value(value))

		# with residual connection
		final_feat = fuse_feat + value
		output = F.sigmoid(self.decoder(F.dropout(final_feat.mean(1),0.5)))

		return output, x, att, dist


class scene_classifier(nn.Module):
	def __init__(self,num_scene):
		super(scene_classifier,self).__init__()
		self.backend = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
		self.decoder = nn.Linear(2048,num_scene,bias=True)

		#fixing saliency module
		for para in self.backend.parameters():
			para.requires_grad = True # fixing pretrained modules or not

	def forward(self,x):
		x = self.backend(x)
		batch, c, w, h = x.shape
		x = F.avg_pool2d(x,(7,7)) # Global Average Pooling for resnet
		x = x.view(batch,-1)

		output = F.softmax(self.decoder(F.dropout(x,0.2)),dim=-1)

		return output
