# training set 
train_set = 'train+val'
assert train_set in ['train', 'train+val', 'all']

merge = True # merge relations or not

# VQA dataset version
version = 'v1'
assert version in ['v1', 'v2']

# paths
main_path = '/raid/guoyangyang/vqa/'
if version == 'v1':
	qa_path = main_path+'vqa1.0/qa_path/'  # directory containing the question and annotation jsons from vqa v1.0 dataset
else:
	qa_path = main_path+'vqa2.0/qa_path/'  # directory containing the question and annotation jsons from vqa v2.0 dataset
rvqa_data_path = main_path + 'r-vqa/'
meta_data_path = rvqa_data_path + 'meta_data.json'
vocab_path = rvqa_data_path + 'vocabs.json'
bottom_up_path = main_path + 'VG100K'
image_features_path = main_path + 'visual-genome/VG.h5'
glove_path = main_path + 'word_embed/glove/'
glove_path_filtered = rvqa_data_path + 'glove_filter'

# vqa mscoco image features
rcnn_path = main_path + '/rcnn-data/'
bottom_up_trainval_path = main_path + 'rcnn-feature/trainval'  # directory containing the .tsv file(s) with bottom up features
bottom_up_test_path = main_path + 'rcnn-feature/test2015'  # directory containing the .tsv file(s) with bottom up features
rcnn_trainval_path = rcnn_path + 'genome-trainval.h5'  # path where preprocessed features from the trainval split are saved to and loaded from
rcnn_test_path = rcnn_path + 'genome-test.h5'  # path where preprocessed features from the test split are saved to and loaded from

# preprocess config
output_size = 100  # max number of object proposals per image
output_features = 2048  # number of features in each object proposal

# hyper parameters
max_question_len = 15
text_embed_size = 300
lamda_sub = 2.0 
lamda_rel = 0.8 
lamda_obj = 1.5 

# training config
epochs = 50
batch_size = 100
initial_lr = 3e-4
data_workers = 4
