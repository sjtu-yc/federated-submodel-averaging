# Federated Submodel Averaging
An implementation of federated submodel averaging (FedSubAvg).

## Data Process
taobao_data_process/: process raw Taobao log for centralized learning and split training set by user for federated learning. 

Notes: The full industrial alibaba dataset cannot be released due to the restriction of Alibaba. We show the training set of a sample user in taobao_data_process/user_data_sample. 

## Train and Evaluate
Run the following command for training and evaluating:
```shell
bash run_multiprocess.sh
```

The key implementation of FedSubAvg is in multiprocess_ps_functions.py.


## Environments

Linux  4.4.0

Python 3.6.0

Numpy 1.16.0

TensorFlow 1.14.0

FlatBuffers 1.11

Matplotlib 3.0.0

## Acknowledgement
We would like to sincerely thank Chaoyue Niu, Renjie Gu, and the edge-AI group in Taobao for their contributions to the project.