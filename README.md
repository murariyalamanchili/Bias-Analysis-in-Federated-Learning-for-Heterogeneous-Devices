# Bias-Analysis-in-Federated-Learning-for-Heterogeneous-Devices

# Motivation

The proliferation of IoT devices in recent years, along with the rise in mobile device counts, has made edge computing techniques like federated learning feasible. These approaches will improve user experience, increase security, and reduce communication costs. However, bias is the most prevalent issue in machine learning applications, arising from a variety of sources such as edge data transportation, device heterogenity, model and client selection, etc. Thus, our motivation is to compare several edge computing methods such as Fed. Avg, Agnostic Fed, and TERM for heterogeneous sensor data in order for better fairness and privacy.

# Goal
Focus on model bias towards different groups due to variations in feature distribution


### How to run
```
python main.py 
```
options:
```
  --dataset {mnist,cifar10}          
  --federated_type {fedavg,afl,term}     
  --model {cnn,mlp}         
  --n_clients int            
  --global_epochs int    
  --local_epochs int
  --batch_size int
  --on_cuda {yes,no}
  --optimizer {sgd,adam}
  --lr float
  --iid {yes,no}
  --drfa_gamma float
```


### Docker setup

Required host copmputer environment
```
- OS: Ubuntu20.04
- CUDA 11.2
```

Docer setup
```
docker build -t agnostic_federated_learning .
docker run -it -v <host dir>:/app --gpus all agnostic_federated_learning
```


### Run experiment

#### IID

##### cifar10
```
# FedAvg
python main.py --federated_type fedavgy --dataset cifar10 --data_dist iid
# AFL
python main.py --federated_type afl --dataset cifar10 --data_dist iid
```
##### mnist
```
# FedAvg
python main.py --federated_type fedavg --dataset mnist --data_dist iid
# AFL
python main.py --federated_type afl --dataset mnist --data_dist iid
```

##### fashionmnist
```
# FedAvg
python main.py --federated_type fedavg --dataset fmnist --data_dist iid
# AFL
python main.py --federated_type afl --dataset fmnist --data_dist iid
```

#### From CSV
##### cifar10
```
# FedAvg
python main.py --federated_type fedavg --dataset cifar10 --data_dist from_csv --from_csv sample2
# AFL
python main.py --federated_type afl --dataset cifar10 --data_dist from_csv --from_csv sample2
```
##### mnist
```
# FedAvg
python main.py --federated_type fedavg --dataset mnist --data_dist from_csv --from_csv sample2
# AFL
python main.py --federated_type afl --dataset mnist --data_dist from_csv --from_csv sample2
```
##### fashionmnist
```
# FedAvg
python main.py --federated_type fedavg --dataset fmnist --data_dist from_csv --from_csv sample2
# AFL
python main.py --federated_type afl --dataset fmnist --data_dist from_csv --from_csv sample2
```
