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
from model import ASD_CNN, ASD_RNN, ASD_transformer
from dataloader import Dataset_category, Dataset_single
from glob import glob
from initialize_prototype import find_prototype, use_fix_prototype
import operator
from sklearn import metrics

parser = argparse.ArgumentParser(description='Prototypical autism screening based on photo-taking data')
parser.add_argument('--backend',type=str,default='resnet',help='Backend for encoder')
parser.add_argument('--img_dir', type=str, default=None, help='Directory to images')
parser.add_argument('--eval_sample',type=int,default=10,help='Number of evaluation round for each subject')
parser.add_argument('--weights', type=str, default=None, help='Specififying the directory for the weights to be loaded (default: None)')
parser.add_argument('--lr',type=float,default=1e-4,help='specify learning rate')
parser.add_argument('--checkpoint_path',type=str,default=None,help='Directory for saving checkpoints')
parser.add_argument('--epoch',type=int,default=180,help='Specify maximum number of epoch')
parser.add_argument('--batch_size',type=int,default=12,help='Batch size')
parser.add_argument('--seq_len',type=int,default=14,help='Sequence length for RNN')
parser.add_argument('--embed_size',type=int,default=512,help='Embedding size for RNN')
parser.add_argument('--num_head',type=int,default=4,help='Number of heads for multi-head attention')
parser.add_argument('--clip',type=float,default=0.1,help='Gradient clipping')
parser.add_argument('--save_dir',type=str,default=None,help='Directory for saving the visualization results')
parser.add_argument('--alpha',type=float,default=0.01,help='Balance factor for positive prototype loss')
parser.add_argument('--beta',type=float,default=0.01,help='Balance factor for negative prototype loss')
parser.add_argument('--margin',type=float,default=0.8,help='margin for the prototype loss')
parser.add_argument('--n_proto',type=int,default=10,help='Number of prototypes')
parser.add_argument('--manual',type=bool,default=False,help='Using manually selected prototypes')
parser.add_argument('--model_type',type=str,default='cnn',help='Specifying the type of backbone, i.e., cnn, rnn, or transformer')

args = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def distance_loss(distance, target, multi_img=False):
    """ Objective function for aligning the visual features with
        prototypes corresponding to the correct labels (ASD or Control).
    """
    if multi_img:
        # multiple images are used as inputs (RNN or Transformer)
        batch, num_fix, proto = distance.shape
        distance = distance.view(batch, num_fix, 2, -1)
        pos_dist = distance[:, :, 0].min(-1)[0].mean(-1)
        neg_dist = distance[:, :, 1].min(-1)[0].mean(-1)
    else:
    	batch, proto = distance.shape
    	distance = distance.view(batch, 2, -1)
    	pos_dist = distance[:, 0].min(-1)[0]
    	neg_dist = distance[:, 1].min(-1)[0]

	pos_mask = torch.zeros(batch,).cuda()
	neg_mask = torch.zeros(batch,).cuda()
	# create label
	for i in range(batch):
		if target[i] == 1:
			pos_mask[i] = args.alpha
			neg_mask[i] = -args.beta
		else:
			pos_mask[i] = -args.beta
			neg_mask[i] = args.alpha

	loss = torch.relu(pos_mask*pos_dist + neg_mask*neg_dist+args.margin).mean()

	return loss

def cross_entropy(pred, target):
	loss = -(target*torch.log(torch.clamp(pred,min=1e-8,max=1))).sum(-1)
	return loss.mean()

def build_dict_category(path, ASD=True):
    """ Reorganizing in-domain data by subjects and class labels
    """
	category_dict = {'indoor':0,'outdoor':1,'people':2}
	category = glob(os.path.join(path,'*'))
	cur_dict = dict()

	for cur_cat in category:
		files = glob(os.path.join(cur_cat,'*.jpg'))
		for cur in files:
			if remove_proto is not None:
				cur_name = os.path.basename(cur)
				if cur_name in remove_pool:
					continue

			subject = os.path.basename(cur).split('_')[0]

			if subject not in cur_dict:
				cur_dict[subject] = dict()
				cur_dict[subject]['img'] = []
				cur_dict[subject]['label'] = 1 if ASD else 0
				cur_dict[subject]['category'] = []
			cur_dict[subject]['img'].append(cur)
			cur_dict[subject]['category'].append(
                                        category_dict[os.path.basename(cur_cat)])

	return cur_dict


def build_ood_dict(path):
    """ Reorganizing out-of-domain data by subjects and class labels
    """
	category_dict = {'indoor':0,'outdoor':1,'people':2}
	category = glob(os.path.join(path,'*'))
	cur_dict = dict()
	cur_dict['img'] = []
	cur_dict['category'] = []

	for cur_cat in category:
		files = glob(os.path.join(cur_cat,'*.jpg'))
		for cur in files:
			cur_dict['img'].append(cur)
			cur_dict['category'].append(category_dict[os.path.basename(cur_cat)])

	return cur_dict


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch):
    "adatively adjust lr based on epoch"
    if epoch <=30:
        lr = args.lr
    else :
        lr = args.lr * (0.5 ** (float(epoch-30) / 25))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # main experiment for in-domain evaluation
	tf_summary_writer = tf.summary.create_file_writer(args.checkpoint_path)
	pos_dict = build_dict_category(os.path.join(args.img_dir, 'ASD_category'))
	neg_dict = build_dict_category(os.path.join(args.img_dir, 'Ctrl_category'), ASD=False)
	nb_subj = len(pos_dict) + len(neg_dict)
	overall_acc = dict()

	if args.manual:
		print('Using manually selected prototypes')

    # specifying single or multiple images as input
    multi_img = True if args.model_type != 'cnn' else False

	# using leave-one-subject-out evaluation
	for i in range(nb_subj):
		# constructing training/evaluation split
		train_data = dict(list(pos_dict.items()) + list(neg_dict.items()))

		if i+1<=len(pos_dict):
			subj_name = list(pos_dict.keys())[i]
			eval_data = pos_dict[subj_name]
			del train_data[subj_name]
		else:
			subj_name = list(neg_dict.keys())[i-len(pos_dict)]
			eval_data = neg_dict[subj_name]
			del train_data[subj_name]

        if multi_img:
    		train_set = Dataset_category('train', train_data, args.seq_len, transform)
    		val_set = Dataset_category('val', eval_data, args.seq_len, transform, args.eval_sample, subj_name)
        else:
    		train_set = Dataset_single('train', train_data, transform)
    		val_set = Dataset_single('val', eval_data, transform)

		trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
		valloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

		init_prototype = None

		# compute initial prototype
		init_prototype = None
		if args.manual:
            # use manually filtered prototypes
			args.n_proto = 10
			init_prototype = use_fix_prototype(subj_name)
		else:
            # use prototypes derived solely with clustering
			init_prototype = find_prototype(subj_name, args.n_proto)

        # defining model architecture
        if args.model_type == 'cnn':
            model = ASD_CNN(args.backend, hidden_size=args.embed_size, use_prototype=True,
                            init_prototype=init_prototype, num_prototype=args.n_proto)
        elif args.model_type == 'rnn':
            model = ASD_RNN(args.backend, seq_len=args.seq_len, hidden_size=args.embed_size,
                        use_prototype=True, init_prototype=init_prototype, num_prototype=args.n_proto)
        elif args.model_type == 'transformer':
            model = ASD_transformer(args.backend, seq_len=args.seq_len, hidden_size=args.embed_size,
                                    num_prototype=args.n_proto, init_prototype=init_prototype)

		model = nn.DataParallel(model) # adding multiple GPU support
		model = model.cuda()
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5) # 5e-4

		def train(epoch, iteration):
            # training loop for a single epoch
			model.train()
			avg_loss = 0
			avg_dist_loss = 0

			for j, (img, target, category, img_id) in enumerate(trainloader):
				img, target, category = Variable(img), Variable(target.type(torch.FloatTensor)), Variable(category).long()
				target = target.unsqueeze(1) # for sigmoid
				img, target, category = img.cuda(), target.cuda(), category.cuda()
				optimizer.zero_grad()

                # inference and loss
				pred, feat, dist, x = model(img,category)
				ce_loss = F.binary_cross_entropy(pred,target)
				dist_loss = distance_loss(dist, target, multi_img)
				loss = ce_loss + dist_loss

				loss.backward()
				if args.clip != -1:
					clip_gradient(optimizer,args.clip)
				optimizer.step()

				avg_loss = (avg_loss*np.maximum(0,j) + ce_loss.data.cpu().numpy())/(j+1)
				avg_dist_loss = (avg_dist_loss*np.maximum(0,j) + dist_loss.data.cpu().numpy())/(j+1)

				with tf_summary_writer.as_default():
					tf.summary.scalar('training_loss_' + subj_name ,avg_loss,step=iteration)
					tf.summary.scalar('distance_loss_' + subj_name ,avg_dist_loss,step=iteration)

				iteration += 1

			return iteration

		def validation(iteration):
            # validation process
			model.eval()
			avg_pred = []

			for _, (img,target,category, img_id) in enumerate(valloader):
				img, target, category= Variable(img), Variable(target.type(torch.FloatTensor)), Variable(category).long()
				target = target.unsqueeze(1) # for sigmoid
				img, target, category = img.cuda(), target.cuda(), category.cuda()
				pred, feat, dist, x = model(img, category)
				pred = pred.data.cpu().numpy()
				target = target.data.cpu().numpy()[0,0]
				avg_pred.extend(pred)

			avg_pred = np.mean(avg_pred)
			avg_pred = avg_pred if target == 1 else (1-avg_pred)

			with tf_summary_writer.as_default():
				tf.summary.scalar('validation_acc_' + subj_name ,avg_pred,step=iteration)

			return avg_pred

		print('Start Leave-One-Subject Evaluation for Subject %s' %subj_name)
		iteration = 0
		best_acc = 0
		for epoch in range(args.epoch):
			adjust_lr(optimizer,epoch)
			iteration = train(epoch,iteration)
			eval_acc = validation(iteration)

			if eval_acc > best_acc and epoch>=args.epoch-3:
				best_acc = eval_acc
				torch.save(model.module.state_dict(),os.path.join(args.checkpoint_path,subj_name+'.pth'))

		overall_acc[subj_name] = dict()
		overall_acc[subj_name]['acc'] = eval_acc
		overall_acc[subj_name]['label'] = eval_data['label']

	np.save(os.path.join(args.checkpoint_path,'result'),overall_acc)

main()
