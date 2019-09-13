# IronyGeneration

## Requirements
- python==3
- pytorch==0.4
- pip3 install -r requirements


## Dataset
### 1. Download tweets
```python
nohup python download.py data/train_ironys_id.txt > download/train_ironys.txt
```
Replace train_ironys_id.txt with train_non_ironys_id.txt/dev_ironys_id.txt/dev_non_ironys_id.txt to download train_non_ironys/dev_ironys/dev_non_ironys

As some tweets are protected, the amount of obtained tweets may be less than that mentioned in paper.
### 2. Preprocess
```python
python prepare_data.py download/train_ironys.txt data/train_ironys.txt
```
Run similar commands to preprocess train_non_ironys, dev_ironys, and dev_non_ironys

## Train 
### 1. Build up directories
```
cd dumped; mkdir <test_name>; cd <test_name>
mkdir data; mkdir data/seq; mkdir data/whole
mkdir model
```
Place pre-trained models (dis.model, senti_dis.model, non_senti_dis.model) into the model directory

Place vocabulary (vocab) into the data directory

### 2. run main.py
Change the test_path parameter in main.py with your <test_name> and run
```python
nohup python main.py > dumped/<test_name>/train.log
```
Tap Ctrl+C to stop the training
