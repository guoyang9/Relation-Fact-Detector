# relation-vqa
Re-implementation of the Relation Fact Detector for 'R-VQA: Learning Visual Relation Facts with Semantic Attention for Visual Question Answering'. You can also get the ```top k``` most related facts for VQA v1.0 and v2.0 datasets.

The paper was published on SIGKDD 2018 and can be downloaded at this [link](http://www.kdd.org/kdd2018/accepted-papers/view/r-vqa-learning-visual-relation-facts-with-semantic-attention-for-visual-que).

This repository focuses on the implementation of the Relation Fact Detector. There are some slightly differences between this repository and the original paper:
* The image feature used here is extracted from faster RCNN, instead of the region-based CNN models. However, you can easily change the input
with the later features.
* You can use pre-trained glove features to initialized word embeddings.
* I filtered the topk relations on the whole set, instead of the train+val splits.
* The samples with all the three elements (i.e., sub, rel, obj) are replaced with 'UNK' after the filtering will be removed.
* Other optimization methods (e.g., Adam), activation function (e.g., ReLU) and batch norm tricks are applied.

I greatly appreciate the first author Pan Lu for his help and the detailed reply to my questions!

## Performance
Models 				| Subject | Relation | Object | Recall@1 | Recall@5 | Recall@10
------------------- | ------- | -------- | ------ | -------- | -------- | ---------
R-VQA  				| 66.47   | 78.80    | 45.13  | 27.39    | 46.72    | 54.10
This Implementation | 73.67   | 76.68    | 60.87  | 40.52    | 73.92    | 83.63

## Prerequisites
	* pytorch==1.0.1  
	* nltk==3.4  
	* bcolz==1.2.1  
	* h5py==2.9.0

## Dataset
The R-VQA dataset can be downloaded at Pan Lu's [repository](https://github.com/lupantech/rvqa).

The VQA dataset can be downloaded at the
[official website](https://visualqa.org/download.html). This repository only implemented the model on VQA 1.0 and 2.0 datasets. If you want to
recover the results on COCO QA dataset, you need to write your own pytorch Dataset.

The pre-trained Glove features can be found on [glove website](https://nlp.stanford.edu/projects/glove/).

The mscoco bottom-up-attention image features have been released in this [repo](https://github.com/peteanderson80/bottom-up-attention).

I guess you may need the RCNN image feature of Visual Genome, please find the Baidu Netdisk link below with passcode ```oqi4```.
```https://pan.baidu.com/s/1J482wd3cdpYC40czyMBvtQ```.

## Runing Details of Relation Fact Detector
Put all the raw data in the right directory according to config.py.
1. Preprocess image features (Most times can skip this step and ask me for the h5 file).
	```
	python preprocess/preprocess-images.py
	```
2. Preprocess all the metadata. This will result all the needed meta json file and vocab files.
	```
	python preprocess/preprocess-meta.py
	```
3. Train the detector (Notice the train_set in config.py should be 'train').
	```
	python main.py --gpu=0 --name=rvqa
	```
4. Test the detector (Notice the train_set in config.py should be 'train+val').
	```
	python main.py --gpu=0 --name=rvqa
	```
## Extracting facts for VQA v1.0 and v2.0 datasets
1. Train the detector from scratch with all the given data (Notice the train_set in config.py should be 'all').
	```
	python main --gpu=0 --name=rvqa
	```
2. Extract facts.
	```
	python facts_vqa.py --gpu=0 --name=logs/rvqa
	```
