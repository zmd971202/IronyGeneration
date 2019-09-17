# IronyGeneration

The code and data of [A Neural Approach to Irony Generation](https://arxiv.org/abs/1909.06200)

## Requirements
- python==3
- pytorch==0.4
- pip3 install -r requirements


## Dataset
### 1. Download tweets
```python
mkdir download
nohup python download.py data/train_ironys_id.txt > download/train_ironys.txt
```
Replace train_ironys_id.txt with train_non_ironys_id.txt/test_ironys_id.txt/test_non_ironys_id.txt in the command to download train_non_ironys/test_ironys/test_non_ironys

As some tweets are protected, the amount of obtained tweets may be less than that mentioned in paper.
### 2. Preprocess
```python
python prepare_data.py download/train_ironys.txt data/train_ironys.txt
```
Change the file name in the command and run similar commands to preprocess train_non_ironys, test_ironys, and test_non_ironys

## Train 
### 1. Build up directories
```
cd dumped; mkdir <test_name>; cd <test_name>
mkdir data; mkdir data/seq; mkdir data/whole
mkdir model
```
Place pre-trained models (dis.model, senti_dis.model, non_senti_dis.model) into the model directory

Place vocabulary (vocab) into the data directory

### 2. Pre-training
Change the test_path parameter in main.py with your <test_name> and run
```python
nohup python main.py > dumped/<test_name>/train.log
```
Tap Ctrl+C to stop the training

### 3. RL training
Uncomment seq.load_state_dict to load pre-trained model and comment the pre-training part in main.py

Then run
```python
nohup python main.py > dumped/<test_name>/train.log
```
Tap Ctrl+C to stop the training

