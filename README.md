# Dialog Modeling
NTU Machine Learning Final Project, 2019 fall  
Kaggle competition: https://www.kaggle.com/c/ml2019fall-final-dialogue/overview

## Prerequisites
* Python 3.6
* wget

## Installation
1. Clone the repo
2. Install packages through pip
```
pip install -r requirements.txt
```

## Usage
### Download dataset
You can directly download the preprocessed data (recommended)
```
./download_preprocessed_data.sh
```
Or download raw data then run preprocess script
```
./download_dataset.sh
./preprocess_data.sh
```
Structured data that are used for training would be generated in **./struct_data** folder.
### Training
```
./train.sh <epoch> <batch size> <learning rate>
```
The fine-tuned BERT models would be saved in **./model** folder.  
Suggested parameters: epoch = 2, batch size = 4, learning rate = 1e-5
### Testing
```
./test.sh <output file path>
```
The output .csv file would be dumped in the specified path.  
***The above command will download and use the best model we trained automatically***. If you want to use a model you trained, please run
```
./test.sh <output file path> [model path]
```
