# ffNN is similar to LogisticRegression but it is able to input non-linear functions
# ReLU is the best (accuracy to iter rate) because of Hidden Layer
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
from datetime import datetime

usingGPU = False
saveModel = False
loadModel = False

# TODO: Figure out why gpu is slower than cpu

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedforwardNeuralNetModel, self).__init__()  # super() allows us to inherits everything from nn.module
        # Linear function 1 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Non-Linearity 1
        self.ReLU1 = nn.ReLU()

        # Linear function 2 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Non-Linearity 2
        self.ReLU2 = nn.ReLU()

        # Linear function 3 (Readout) 100 --> 10
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # you can put another hidden layer to make it more precise

    def forward(self, x):
        # Linear function
        out = self.fc1(x)

        # Non-Linearity 1
        out = self.ReLU1(out)

        # Linear function 2
        out = self.fc2(out)

        # Non-Linearity 2
        out = self.ReLU2(out)

        # Linear function 3 (Readout)
        out = self.fc3(out)
        return out


train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

batch_size = 100  # here if we increase we can really see the power of gpu against cpu
# 1000 bsize, 3000 iters : approx 2.24m GPU, approx 2.47m CPU
n_iters = 3000
epochs = int(n_iters / (len(train_dataset) / batch_size))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

input_dim = 28 * 28
hidden_dim = 100
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
if torch.cuda.is_available() and usingGPU:
    model = model.cuda()

if loadModel:
    model.load_state_dict(torch.load('relumodel.pkl'))

criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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