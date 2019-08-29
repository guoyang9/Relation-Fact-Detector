import os, re
import json, h5py

import torch
import torch.utils.data as data

import utils.config as config
import utils.utils as utils


def get_loader(split=None, test=False, vocabs=None):
	""" Returns a data loader for the desired split """
	image_path = config.rcnn_test_path if test else config.rcnn_trainval_path
	split = VQA(vocabs,	utils.path_for(split),	image_path)

	loader = torch.utils.data.DataLoader(
		split,
		batch_size=config.batch_size,
		pin_memory=True,
		num_workers=config.data_workers,
		collate_fn=collate_fn,
	)
	return loader

def collate_fn(batch):
	# put question lengths in descending order so that we can use packed sequences later
	batch.sort(key=lambda x: x[-1], reverse=True)
	return data.dataloader.default_collate(batch)


class VQA(data.Dataset):
	""" VQA dataset, open-ended """
	def __init__(self, question_vocab, questions_path, image_features_path):
		super(VQA, self).__init__()
		with open(questions_path, 'r') as fd:
			questions_json = json.load(fd)
			if not config.cp_data:
				questions_json = questions_json['questions']
		self.question_vocab = question_vocab

		# q
		self.question_ids = [q['question_id'] for q in questions_json]
		self.questions = list(prepare_questions(questions_json))
		self.questions = [self._encode_question(utils.tokenize_text(q)) for q in self.questions]

		# v
		self.image_features_path = image_features_path
		self.coco_id_to_index = self._create_coco_id_to_index()
		self.coco_ids = [q['image_id'] for q in questions_json]

	def _create_coco_id_to_index(self):
		""" Create a mapping from a COCO image id into the corresponding index into the h5 file """
		with h5py.File(self.image_features_path, 'r') as features_file:
			coco_ids = features_file['ids'][()]
		coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
		return coco_id_to_index

	def _encode_question(self, question):
		""" Turn a question into a vector of indices and a question length """
		vec = torch.zeros(config.max_question_len).long()
		for i, token in enumerate(question):
			if i < config.max_question_len:
				index = self.question_vocab.get(token, 0)
				vec[i] = index
		return vec, min(len(question), config.max_question_len)

	def _load_image(self, image_id):
		""" Load an image """
		if not hasattr(self, 'features_file'):
			# Loading the h5 file has to be done here and not in __init__ because when the DataLoader
			# forks for multiple works, every child would use the same file object and fail.
			# Having multiple readers using different file objects is fine though, so we just init in here.
			self.features_file = h5py.File(self.image_features_path, 'r')
		index = self.coco_id_to_index[image_id]
		img = self.features_file['features'][index]
		boxes = self.features_file['boxes'][index]
		return torch.from_numpy(img).unsqueeze(1), torch.from_numpy(boxes)

	def __getitem__(self, item):
		image_id = self.coco_ids[item]
		v, b = self._load_image(image_id)
		# since batches are re-ordered for PackedSequence's, the original question order is lost
		# we return `item` so that the order of (v, q, a) triples can be restored if desired
		# without shuffling in the dataloader, these will be in the order that they appear in the q and a json's.
		q, q_len = self.questions[item]
		return item, v, q, q_len

	def __len__(self):
		return len(self.questions)


# this is used for normalizing questions
_special_chars = re.compile('(\'+s)*[^a-z0-9- ]*')

def prepare_questions(questions, rvqa=False):
	""" Tokenize and normalize questions from a given question json in the usual VQA format. """
	if not rvqa:
		questions = [q['question'] for q in questions]
	for question in questions:
		question = question.lower()[:-1]
		question = _special_chars.sub('', question)
		question = re.sub(r'-+', ' ', question)
		yield question
