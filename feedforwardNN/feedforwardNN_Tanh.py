# ffNN is similar to LogisticRegression but it is able to input non-linear functions
# Sigmoid is the more stable but not as fast as the other
# TODO: Convert model to tanh, for now its a duplicate of sigmoid
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from datetime import datetime


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedforwardNeuralNetModel, self).__init__()  # super() allows us to inherits everything from nn.module
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-Linearity
        self.sigmoid = nn.Sigmoid()
        # Linear function again (Readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-Linearity
        out = self.sigmoid(out)
        # Linear function again (Readout)
        out = self.fc2(out)
        return out


train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

batch_size = 100
n_iters = 3000
epochs = int(n_iters / (len(train_dataset) / batch_size))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

input_dim = 28*28
hidden_dim = 100
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

i = 0
startTime = datetime.now()
for epoch in range(epochs):
    for j, (images, labels) in enumerate(train_loader):
        # Load images as tensor
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


