# Made by ElleVen, Federico Cassano
#
# logR is often used for classification, example:
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
# Aim of logR: reduce Cross Entropy Loss (with linR we want to reduce Mean Squared Error)
# Steps to achieve this:
#   Step 1: Load Dataset
#   Step 2: Make Dataset Iterable
#   Step 3: Create Model Class
#   Step 4: Instantiate Model Class
#   Step 5: Instantiate Loss Class
#   Step 6: Instantiate Optimizer Class
#   Step 7: Train Model

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.autograd
import matplotlib.pyplot as plt
from datetime import datetime

usingGPU = False
saveModel = False
loadModel = False


# STEP 3: Create model class, same as linear regression
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionModel, self).__init__()  # super() allows us to inherits everything from nn.module
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


# STEP 1A: Loading MNIST Train Dataset
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# type(train_dataset[0])
# train_dataset[0][0].size() ## Input matrix (1, 28, 28)
# train_dataset[0][1] ## Label (5)

# STEP 1B: Loading MNIST Test Dataset
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# STEP 2: Make Dataset Iterable
batch_size = 20
n_iters = 5000

epochs = int(n_iters / (len(train_dataset) / batch_size))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# shuffle=false because we are going to do only one forward pass

# STEP 4: Instantiate Model class
input_dim = 28 * 28  # train_dataset[0][0].size() = [1, 28, 28]
output_dim = 10

model = LogisticRegressionModel(input_dim, output_dim)
if torch.cuda.is_available() and usingGPU:
    model = model.cuda()

if loadModel:
    model.load_state_dict(torch.load('model.pkl'))

# STEP 5: Instantiate Loss Class
# With Linear Regression we used MSE, but with Logistic we use Cross Entropy Loss
# nn.CrossEntropyLoss() computes softmax and cross entropy
criterion = nn.CrossEntropyLoss()

# STEP 6: Instantiate Optimizer Class
# pretty much whats happening:
#   parameters = parameters - learning_rate * parameters_gradients
# it updates model's parameters every iteration
learning_rate = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# print(model.parameters())
# print(len(list(model.parameters())))
# print(list(model.parameters())[0].size())
# print(list(model.parameters())[1].size())

# STEP 7: Train Model
# Sub-steps:
#   1. Convert inputs/labels to grad tensors
#   2. Clear grad buffers
#   3. Get output given inputs
#   4. Get loss
#   5. Get grads w.r.t. parameters
#   6. Update parameters using grads
#   7. Repeat
i = 0
startTime = datetime.now()
for epoch in range(epochs):
    for j, (images, labels) in enumerate(train_loader):
        # Load images as tensor
        if torch.cuda.is_available() and usingGPU:
            images = torch.tensor(images.view(-1, 28 * 28))
            images = images.cuda()
            labels = torch.tensor(labels)
            labels = labels.cuda()
        else:
            images = torch.tensor(images.view(-1, 28 * 28))
            labels = torch.tensor(labels)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate loss: Softmax --> Cross Entropy Loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating params
        optimizer.step()

        i += 1
        if i % 100 == 0:
            # Calculate accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # load images to torch tensor
                if torch.cuda.is_available() and usingGPU:
                    images = torch.tensor(images.view(-1, 28 * 28))
                    images = images.cuda()
                else:
                    images = torch.tensor(images.view(-1, 28 * 28))

                # Forward pass only to get output/logits
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted.cpu() == labels.cpu()).sum()

            accuracy = 100 * correct / total
            print('Iteration: {}. Loss: {}. Accuracy: {}. \nPrediction: {}'.format(i, loss.item(), accuracy, predicted))

print("Labels:    ", labels)
print("Time elapsed:\n", datetime.now() - startTime)

if saveModel:
    # This saves only the parameters
    torch.save(model.state_dict(), 'model.pkl')

### Plot stuff ###
plt.clf()

# Plot true data that we know
plt.plot(labels.tolist(), "go", label="True data", alpha=0.5)

# Plot predicted data
plt.plot(predicted.tolist(), "-", label="Predictions", alpha=0.5)

# Legend and plot show
plt.legend(loc="best")
plt.show()  # shows plot