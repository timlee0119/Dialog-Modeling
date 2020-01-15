# Dialog Modeling
NTU Machine Learning Final Project, 2019 fall

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
```
./download_dataset.sh
```
Three files (**train.json, valid.json, test.json**) would be downloaded in **./data** folder.
### Data preprocessing
```
./preprocess_data.sh
```
This will generate structured data in **./struct_data** folder.
### Training
```
./train.sh
```
The best model would be saved in **./model** folder.
### Testing
```
./test.sh [output file path]
```
The output .csv file would be dumped in the specified path.
