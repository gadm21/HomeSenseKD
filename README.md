# HomeSenseKD

### This repository provides scripts for Federated Learning via Augmented Knowledge Distillation applied to WiFi sensing for smart home occupancy detection

WiFi sensing using Channel State Information (CSI) enables non-intrusive human activity recognition and occupancy monitoring in smart home environments without requiring wearable devices. However, training accurate deep learning models for WiFi-based sensing requires large and representative datasets, which raises privacy concerns when collecting data from multiple households. Federated Learning (FL) is a privacy-preserving approach for utilizing distributed data to train models without centralizing raw data, but it is limited to training homogeneous model architectures. In this work, we apply Federated Learning via Augmented Knowledge Distillation (FedAKD) to WiFi-based smart home sensing, enabling heterogeneous clients (e.g., different WiFi router models, households) to train personalized models while preserving privacy. FedAKD is evaluated on WiFi CSI datasets (HomeOccupancy and HomeHAR) and is shown to be more flexible and efficient than standard FL, with significant performance gains for clients and reduced communication overhead.

![federated learning][intro]


## Knowledge Distillation 


Knowledge Distillation (KD) is a technique to transfer knowledge from a trained model to a to-be-trained model. Unlike standard Federated Learning (FL) algorithms (FedAvg) which communicate model-dependent data (gradients or weights), KD can be used in the context of Federated Learning (FL) to distill knowledge among heterogeneous clients by communicating soft labels calculated using an un-labeled shared dataset. 

> Knowledge Distillation-based Federated Learning enables clients to independenlty design their learning models.

![Knowledge Distillation][KD]



## Federated Learning via Augmented Knowledge Distillation 
![Augmented Knowledge Distillation][FedAKD_timeline]
We push KD one step further by using an augmentation algorithm based on a server-controlled permutation and mixup augmentation [1] to distill knowledge more efficiently. 

```python

# Global round r of FedAKD starts here

# 1. Local training 
model.fit(local_data, local_labels, epochs = local_epochs) 

# 2. Receive alpha and beta from server 
alpha, beta = receive_metadata_from_server(global_round = r)

# 3. mixup augmentation 
np.random.seed(beta) # beta is used to set the seed to generate the same augmented version of public data across all nodes 
perm = np.random.permutation(len(pub_data))
aug_pub_data = alpha * pub_data + (1-alpha) * pub_data[perm, ...]


# 4. calculate (1) soft labels (2) performance on test data (prepare local knowledge) 
# A value indicating the performance is send to weight soft labels proportional to performance
local_soft_labels = model.predict(aug_pub_data) 
loss, acc = model.evaluate(test_data, test_labels) 


# 5. Send local knowledge, take some rest, then receive global knowledge  
send_to_server({'soft labels': local_soft_labels, 'performance': acc})
global_soft_labels = receive_labels_from_server() 

# 6. Digest knowledge 
model.fit(aug_pub_data, global_soft_labels) 

# Global round r of FedAKD ends here

```


## Datasets

We evaluate FedAKD on two WiFi CSI datasets for smart home sensing:

- **HomeOccupancy**: A WiFi CSI dataset for home occupancy detection with three classes (empty, sleep, work). The dataset contains CSI measurements collected from multiple WiFi access points across different household environments.

- **HomeHAR**: A WiFi CSI dataset for human activity recognition in home environments with seven activity classes (drink, eat, empty, sleep, smoke, watch, work). This dataset serves as public distillation data for knowledge transfer.

Both datasets are available on HuggingFace Hub:
- HomeOccupancy: https://huggingface.co/datasets/gadgadgad/HomeOccupancy
- HomeHAR: https://huggingface.co/datasets/gadgadgad/HomeHAR

## Usage

The main notebook `WS_FedAKD.ipynb` contains the complete implementation for:
- Loading and preprocessing WiFi CSI data from HuggingFace datasets
- Setting up federated learning with heterogeneous client models
- Running FedAKD iterations with augmented knowledge distillation
- Evaluating performance on test data

### Requirements

```bash
pip install datasets scipy tensorflow scikit-learn
```

### Running the experiments

Open the notebook in Jupyter or Google Colab and execute the cells sequentially. The notebook includes experiments for both IID and non-IID client data distributions.

## Results

We evaluate FedAKD on WiFi CSI datasets for smart home occupancy detection against standard Federated Learning and KD-based methods. The results demonstrate that FedAKD achieves superior performance in both IID and non-IID data distribution scenarios, with significant improvements in communication efficiency and model accuracy for heterogeneous clients.




## References 

[1] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412.

[2] Li, D., & Wang, J. (2019). Fedmd: Heterogenous federated learning via model distillation. arXiv preprint arXiv:1910.03581.


[intro]: https://github.com/gadm21/HomeSenseKD/blob/main/assets/intro.png
[KD]: https://github.com/gadm21/HomeSenseKD/blob/main/assets/KD_overview.png
[FedAKD_timeline]: https://github.com/gadm21/HomeSenseKD/blob/main/assets/FedAKD_timeline.png

