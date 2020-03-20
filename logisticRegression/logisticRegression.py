# Used often for classification, example:
# Outputs probability between email spam and not spam
#
# Linear Regression Vs logistic Regression
# LinR Multiplication:
#   Input(4)
#       Output: 16
#
# LogR Spam:
#   Input("YOU ARE THE WINNER OF 1 MILLION DOLLARS!")
#       Output: prob=0.8
#
#   Input("PayPal payment number:492054180")
#       Output: prob=0.3
#
# Steps to achieve this:
#   Step 1: Load Dataset
#   Step 2: Make Dataset Iterable
#   Step 3: Create Model Class
#   Step 4: Instantiate Model class
#   Step 5: Instantiate
#   Step 6: Instantiate
#   Step 7: Train Model

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.autograd
import matplotlib.pyplot as plt
import numpy as np

# STEP 1A: Loading MNIST Train Dataset
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# type(train_dataset[0])
# train_dataset[0][0].size() ## Input matrix (1, 28, 28)
# train_dataset[0][1] ## Label (5)

# STEP 1B: Loading MNIST Test Dataset
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# STEP 2: Make Dataset Iterable
batch_size = 100
n_iters = 3000

epochs = n_iters / len(train_dataset) / batch_size

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)










# Displaying the image
show_img = train_dataset[0][0].numpy().reshape(28, 28)
plt.imshow(show_img, cmap='gray')
plt.show()



