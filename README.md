# Federated Submodel Averaging
An implementation of federated submodel averaging (FedSubAvg).

## Data Process
taobao_data_process/: process raw Taobao log for centralized learning and split training set by user for federtaed learning. 

Notes: The full industrial alibaba dataset cannot be released due to the restriction of Alibaba. We show the training set of a sample user in user_data_sample. 

## Train:

```shell
bash run_multiprocess.sh
```


## Environments:

Linux  4.4.0

Python 3.6.0

Numpy 1.16.0

TensorFlow 1.14.0

FlatBuffers 1.11

Matplotlib 3.0.0

