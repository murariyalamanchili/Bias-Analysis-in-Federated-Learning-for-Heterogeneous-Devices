# Bias-Analysis-in-Federated-Learning-for-Heterogeneous-Devices

# Motivation

The proliferation of IoT devices in recent years, along with the rise in mobile device counts, has made edge computing techniques like federated learning feasible. These approaches will improve user experience, increase security, and reduce communication costs. However, bias is the most prevalent issue in machine learning applications, arising from a variety of sources such as edge data transportation, device heterogenity, model and client selection, etc. Thus, our motivation is to compare several edge computing methods such as Fed. Avg, Agnostic Fed, and TERM for heterogeneous sensor data in order for better fairness and privacy.

# Goal
Focus on model bias towards different groups due to variations in feature distribution

# Deliverables

  • Understand and implement different FL techniques – Federated averaging (FedAvg), Tilted Empirical Risk Minimization (TERM), and Agnostic FL (AFL) (code provided)
  
  • Utilize diverse datasets for training and assessment of two FL techniques (data provided)
  
  • Examine the enhancement in the variance of accuracy among individual client groups when employing various federated learning techniques.

# Software: 
Python, Laptop with CUDA-enabled GPU

# Team member responsibilities

Everyone of us will be reading a paper each and applies the respective algorithm to calculate accuracy for the given combined dataset and will be comparing with eachother's results and with the results in the reference papers given below.

Harinarayana Burra: Analysis of the paper with a given datasets which contains federated average algorithm[1].

Rupamythali Nimmakayala: Analysis of the paper with a given datasets which contains TERM algorithm[2].

Murari Yalamanchili: Analysis of the paper with a given datasets which contains Agnostic federated learning algorithm[3].

# Project Timeline

Oct 31, 2023: Initial understanding of research papers and study the different algorithm design for the particular dataset. Analyzing accuracy and other metrics 1 or 2 algorithms in the software by giving the first dataset.

Nov 15, 2023: Analzing accuracy and other metrics for remaining algorithms with the first dataset. Comparing eachother to understand the bias shown by algorithms . 

Nov 30, 2023: Comparing results obtained with their reference papers and analyzing algorithms with the second dataset.

Dec 12, 2023: Reporting final analysis and results.

# References:
  • Papers -  [1] Communication-Efficient Learning of Deep Networks from Decentralized Data (https://arxiv.org/abs/1602.05629); 
              [2] Tilted Empirical Risk Minimization (https://openreview.net/pdf?id=K5YasWXZT3O);
              [3] Agnostic Federated Learning (https://arxiv.org/pdf/1902.00146.pdf);
            
  • Code: FedAvg (https://github.com/alexbie98/fedavg); 
    TERM (https://github.com/litian96/TERM); 
   AFL (https://github.com/YuichiNAGAO/agnostic_federated_learning);
          
  • Datasets: CIFAR-10 (https://www.kaggle.com/c/cifar-10/data);
             FashionMNIST (https://github.com/zalandoresearch/fashion-mnist)

