import os, sys
import argparse
import json, bcolz
from tqdm import tqdm

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import utils.config as config
import utils.vqa_data as data
import utils.utils as utils
import model.model as detector


def run(detector, loader, tracker, prefix='extract', top=None):
	""" Run an epoch over the given loader """
	facts, idxs, subs, objs, rels = [], [], [], [], []
	fmt = '{:.4f}'.format
	tracker_class, tracker_params = tracker.MeanMonitor, {}

	loader = tqdm(loader, desc='{}'.format(prefix), ncols=0)
	for idx, v, q, q_len in loader:
		v = v.cuda(async=True).squeeze(2)
		q = q.cuda(async=True)
		sub_prob, rel_prob, obj_prob = detector(v, q, 0, 0, 0, q_len)

		_, top_10_sub = sub_prob.topk(top)
		_, top_10_rel = rel_prob.topk(top)
		_, top_10_obj = obj_prob.topk(top)

		subs.append(top_10_sub.cpu())
		objs.append(top_10_obj.cpu())
		rels.append(top_10_rel.cpu())

		top_10_fact = torch.cat((top_10_sub,
			top_10_rel, top_10_obj), dim=1).view(-1, 3, top).transpose(1,2)
		facts.append(top_10_fact.cpu())
		idxs.append(idx.view(-1).clone())

	facts = torch.cat(facts, dim=0).numpy()
	idxs = torch.cat(idxs, dim=0).numpy()
	subs = torch.cat(subs, dim=0).numpy()
	objs = torch.cat(objs, dim=0).numpy()
	rels = torch.cat(rels, dim=0).numpy()

	return idxs, subs, rels, objs, facts


def save_to_disk(val_loader, result, split, vocabs):
	""" Save the extracted facts results into disk. """
	facts = {}

	sub_idx_to_str = {i: s for s, i in vocabs['subs'].items()}
	rel_idx_to_str = {i: s for s, i in vocabs['rels'].items()}
	obj_idx_to_str = {i: s for s, i in vocabs['objs'].items()}

	question_ids = val_loader.dataset.question_ids
	fact_file = '{}_{}_facts'.format(config.dataset, split)
	if config.version == 'v2':
		fact_file = 'v2_' + fact_file
	fact_file = os.path.join(config.fact_path, fact_file)

	with h5py.File('{}.h5'.format(fact_file), 'w') as f:
		subs = f.create_dataset('subs', data=result[1])
		rels = f.create_dataset('rels', data=result[2])
		objs = f.create_dataset('objs', data=result[3])

		qids = [question_ids[i] for i in result[0]]
		qids = f.create_dataset('qids', data=qids)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name',
		required=True,
		help='the name of the detector model, e.g., /logs/rvqa.pth')
	parser.add_argument('--gpu',
		default='0',
		help='the chosen gpu id')
	args = parser.parse_args()


	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	cudnn.benchmark = True

	with open(config.vocab_path, 'r') as fd:
		vocabs = json.load(fd)
	question_vocab = vocabs['question']

	embedding = bcolz.open(config.glove_path_filtered)[:]
	detector_logs = torch.load(args.name)
	detector_net = detector.Net(embedding).cuda()
	detector_net.load_state_dict(detector_logs['model_state'])
	detector_net.eval()
	print("loading model parameters...")

	if config.cp_data:
		splits = ['train', 'test']
	else:
		splits = ['train2014', 'val2014', 'test-dev2015', 'test2015']

	tracker = utils.Tracker()
	for split in splits:
		test = False if 'test' not in split or config.cp_data else True
		val_loader = data.get_loader(split, test=test, vocabs=question_vocab)
		facts = run(detector_net, val_loader, tracker, prefix=split, top=10)
		save_to_disk(val_loader, facts, split, vocabs)


if __name__ == '__main__':
	main()
