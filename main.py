import os, sys
import argparse
import json, bcolz
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import utils.config as config
import utils.data as data
import utils.utils as utils
import model.model as model


def run(net, loader, tracker, 
		optimizer, loss_criterion=None, train=False, prefix='', epoch=0):
	""" Run an epoch over the given loader """
	if train:
		net.train()
		tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
	else:
		net.eval()
		tracker_class, tracker_params = tracker.MeanMonitor, {}
		top_1_sub, top_1_rel, top_1_obj = [], [], []
		top_5_sub, top_5_rel, top_5_obj = [], [], []
		top_10_sub, top_10_rel, top_10_obj = [], [], []
		gt_sub, gt_rel, gt_obj = [], [], []


	loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)

	loss_t_all = tracker.track('{}_loss_all'.format(prefix), tracker_class(**tracker_params))
	# loss_t_sub = tracker.track('{}_loss_sub'.format(prefix), tracker_class(**tracker_params))
	# loss_t_rel = tracker.track('{}_loss_rel'.format(prefix), tracker_class(**tracker_params))
	# loss_t_obj = tracker.track('{}_loss_obj'.format(prefix), tracker_class(**tracker_params))

	acc_t_sub = tracker.track('{}_acc_sub'.format(prefix), tracker_class(**tracker_params))
	acc_t_rel = tracker.track('{}_acc_rel'.format(prefix), tracker_class(**tracker_params))
	acc_t_obj = tracker.track('{}_acc_obj'.format(prefix), tracker_class(**tracker_params))
	
	for v, q, rel, rel_sub, rel_rel, rel_obj, q_len in loader:
		v = v.cuda(async=True)
		q = q.cuda(async=True)
		rel_sub = rel_sub.cuda(async=True)
		rel_rel = rel_rel.cuda(async=True)
		rel_obj = rel_obj.cuda(async=True)
		q_len = q_len.cuda(async=True)
		sub_prob, rel_prob, obj_prob = net(v, q, rel_sub, rel_rel, rel_obj, q_len)
		
		loss_sub = loss_criterion(sub_prob, rel_sub)
		loss_rel = loss_criterion(rel_prob, rel_rel)
		loss_obj = loss_criterion(obj_prob, rel_obj)
		loss_all = config.lamda_sub*loss_sub + config.lamda_rel*loss_rel + config.lamda_obj*loss_obj

		acc_sub = utils.batch_accuracy(sub_prob, rel_sub).cpu()
		acc_rel = utils.batch_accuracy(rel_prob, rel_rel).cpu()
		acc_obj = utils.batch_accuracy(obj_prob, rel_obj).cpu()

		if train:
			optimizer.zero_grad()
			loss_all.backward()
			optimizer.step()
		else:
			# store information about evaluation of this minibatch
			top_1_sub.append(sub_prob.topk(1)[1])
			top_1_rel.append(rel_prob.topk(1)[1])
			top_1_obj.append(obj_prob.topk(1)[1])
			top_5_sub.append(sub_prob.topk(5)[1])
			top_5_rel.append(rel_prob.topk(5)[1])
			top_5_obj.append(obj_prob.topk(5)[1])
			top_10_sub.append(sub_prob.topk(10)[1])
			top_10_rel.append(rel_prob.topk(10)[1])
			top_10_obj.append(obj_prob.topk(10)[1])
			gt_sub.append(rel_sub.view(-1, 1))
			gt_rel.append(rel_rel.view(-1, 1))
			gt_obj.append(rel_obj.view(-1, 1))

		
		loss_t_all.append(loss_all.item())
		# loss_t_sub.append(loss_sub.item())
		# loss_t_rel.append(loss_rel.item())
		# loss_t_obj.append(loss_obj.item())

		acc_t_sub.append(acc_sub.mean())
		acc_t_rel.append(acc_rel.mean())
		acc_t_obj.append(acc_obj.mean())

		fmt = '{:.4f}'.format
		loader.set_postfix(
			loss_all=fmt(loss_t_all.mean.value),
			acc_sub=fmt(acc_t_sub.mean.value),
			acc_rel=fmt(acc_t_rel.mean.value),
			acc_obj=fmt(acc_t_obj.mean.value),
		)

	if not train:
		top_1_sub = torch.cat(top_1_sub, dim=0).cpu().numpy()
		top_1_rel = torch.cat(top_1_rel, dim=0).cpu().numpy()
		top_1_obj = torch.cat(top_1_obj, dim=0).cpu().numpy()
		top_5_sub = torch.cat(top_5_sub, dim=0).cpu().numpy()
		top_5_rel = torch.cat(top_5_rel, dim=0).cpu().numpy()
		top_5_obj = torch.cat(top_5_obj, dim=0).cpu().numpy()
		top_10_sub = torch.cat(top_10_sub, dim=0).cpu().numpy()
		top_10_rel = torch.cat(top_10_rel, dim=0).cpu().numpy()
		top_10_obj = torch.cat(top_10_obj, dim=0).cpu().numpy()
		gt_sub = torch.cat(gt_sub, dim=0).cpu().numpy()
		gt_rel = torch.cat(gt_rel, dim=0).cpu().numpy()
		gt_obj = torch.cat(gt_obj, dim=0).cpu().numpy()

		recall_1 = utils.recall(gt_sub, gt_rel, gt_obj,
								top_1_sub, top_1_rel, top_1_obj)
		recall_5 = utils.recall(gt_sub, gt_rel, gt_obj,
								top_5_sub, top_5_rel, top_5_obj)
		recall_10 = utils.recall(gt_sub, gt_rel, gt_obj,
								top_10_sub, top_10_rel, top_10_obj)		 
		return recall_1, recall_5, recall_10


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, help='saved and resumed file name')
	parser.add_argument('--resume', action='store_true', help='resumed flag')
	parser.add_argument('--gpu', default='0', help='the chosen gpu id')
	args = parser.parse_args()


	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	cudnn.benchmark = True

	########################################## ARGUMENT SETTING	 #################################
	if args.resume and not args.name:
		raise ValueError('Resuming requires file name!')
	name = args.name if args.name else datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
	if args.resume:
		target_name = name
		logs = torch.load(target_name)
	else: 
		target_name = os.path.join('logs', '{}'.format(name))
	print('will save to {}'.format(target_name))

	######################################### DATASET PREPARATION #################################
	with open(config.meta_data_path, 'r') as fd:
		meta_data = json.load(fd)
	train_loader = data.get_loader('train', meta_data)
	if config.train_set == 'train':
		val_loader = data.get_loader('val', meta_data)
	if config.train_set == 'train+val':
		val_loader = data.get_loader('test', meta_data)

	########################################## MODEL PREPARATION ##################################
	embeddings = bcolz.open(config.glove_path_filtered)[:]
	net = model.Net(embeddings).cuda()
	loss = nn.CrossEntropyLoss()
	optimizer = optim.Adam(
		[p for p in net.parameters() if p.requires_grad], 
		lr=config.initial_lr,
	)

	start_epoch = 0
	recall_10_val_best = 0.0
	if args.resume:
		net.load_state_dict(logs['model_state'])
		optimizer.load_state_dict(logs['optim_state'])
		start_epoch = logs['epoch']
		recall_10_val_best = logs['recall_10_val_best']

	tracker = utils.Tracker()
	best_epoch = start_epoch
	state = 'Valid' if config.train_set == 'train' else 'Test'
	for i in range(start_epoch, config.epochs):
		run(net, train_loader, tracker, optimizer, 
			loss_criterion=loss, train=True, prefix='train', epoch=i)

		results = {
			'epoch': i,
			'name': name,
			'model_state': net.state_dict(),
			'optim_state': optimizer.state_dict(),
		}

		if not config.train_set == 'all':
			r = run(net, val_loader, tracker, optimizer, 
				loss_criterion=loss, train=False, prefix='val', epoch=i)
			print("{} epoch {}: recall@1 is {:.4f}".format(state, i, r[0]), end=", ")
			print("recall@5 is {:.4f}, recall@10 is {:.4f}".format(r[1], r[2]))

			if r[2] > recall_10_val_best:
				recall_10_val_best = r[2]
				results['recall_10_val_best'] = recall_10_val_best
				best_epoch = i
				recall_1_val_best = r[0]
				recall_5_val_best = r[1]
				torch.save(results, target_name+'.pth')
	if not config.train_set == 'all':
		print("The best performance of {} is on epoch {}".format(state, best_epoch), end=": " )
		print("recall@1 is {:.4f}, recall@5 is {:.4f}, recall@10 is {:.4f}".format(
									recall_1_val_best, recall_5_val_best, recall_10_val_best))
	else:
		torch.save(results, target_name+'.pth')


if __name__ == '__main__':
	main()
