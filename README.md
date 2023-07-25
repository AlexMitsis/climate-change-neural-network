# Intelligent Systems Project: Regression with Neural Networks and Climate Change Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MpHNipkJbXBIJJ7dFO2uU7GykDMGVD6E?usp=sharing)
## Introduction

This project focuses on regression using neural networks to predict temperature changes based on various climate-related variables. We'll be using the Climate Change dataset for this purpose. The project involves loading and preparing the dataset, creating a custom Dataset object for data loading, defining neural network architectures, training the models, and evaluating the results.

## Loading and Preparation

We start by loading the required libraries and the climate change dataset. We'll also display the first few rows of the dataset and its dimensions.

```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the climate change dataset
dataframe = pd.read_csv('climate_change.csv')
print(dataframe.head(5))
print("Dimensions:", dataframe.shape)
```

## Creation of a Custom Dataset Object

Next, we create a custom Dataset object that will be used to feed the data to the DataLoader. We also define a transform object for data normalization.

```python
class ClimateDataset(torch.utils.data.Dataset):
    # ... (Dataset definition code)
    
# Min-Max Scaler class for data normalization
class MinMaxScaler():
    # ... (MinMaxScaler definition code)
```

## Creation of DataLoaders

We create DataLoader objects for the training and test sets using the ClimateDataset class.

```python
# Define batch size and transformation
batch_size = 500
transform = MinMaxScaler()

# Create train and test datasets and data loaders
train_dataset = ClimateDataset(csv_file='climate_change.csv', train=True, transform=transform)
test_dataset = ClimateDataset(csv_file='climate_change.csv', train=False, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
```

## Network Architecture

We define two neural network architectures: NeuralNetwork1 with a linear layer for linear regression, and NeuralNetwork2 with hidden layers and ReLU activation for non-linear regression.

```python
# Network defining code
class NeuralNetwork1(nn.Module):
    # ... (NeuralNetwork1 definition code)

class NeuralNetwork2(nn.Module):
    # ... (NeuralNetwork2 definition code)
```

## Training and Test Loop

We define functions for the training and test loops for both NeuralNetwork1 and NeuralNetwork2.

```python
def train_loop(dataloader, model, loss_fn, optimizer, current_epoch):
    # ... (Training loop code)

def test_loop(dataloader, model, loss_fn, current_epoch):
    # ... (Test loop code)
```

## Neural Network Training and Evaluation

Finally, we execute the training process for both NeuralNetwork1 and NeuralNetwork2 and evaluate their performance.

```python
# Seed for reproducibility (you can try different seeds)
torch.manual_seed(50)

# Choose the model to train (uncomment the model you want to use)
model = NeuralNetwork1()
# model = NeuralNetwork2()

# Loss function for regression
loss_fn = nn.MSELoss()
learning_rate = 1e-3

# Set the optimizer to use SGD (uncomment for Adam optimizer)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
epochs = 3000
epoch_step = 100
for t in range(epochs):
    if t == 0:
        print(f"Epoch {t + 1} - {t + epoch_step}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, t)
    if (t + 1) % 100 == 0:
        test_loop(test_dataloader, model, loss_fn, t)
    if (t + 1) % 100 == 0 and (t + 1) != epochs:
        print(f"Epoch {t + 1} - {t + 1 + epoch_step}\n-------------------------------")
print("Done!")
```

## Conclusion

In this project, we explored regression using neural networks and the climate change dataset. We defined and trained two neural network architectures (NeuralNetwork1 and NeuralNetwork2) for regression tasks. The results were evaluated using the R2 score and average loss. Feel free to experiment with different hyperparameters, network architectures, and optimization techniques to improve the model's performance.
