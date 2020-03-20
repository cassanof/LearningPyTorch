import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd
import torch.nn as nn
from datetime import datetime
usingGPU = True
saveModel = False
loadModel = False


# Creating a model class (True Equation: y = 2x + 1)
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()  # super() allows us to inherits everything from nn.module
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear = self.linear

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1
learning_rate = 0.01
epochs = 1000

model = LinearRegressionModel(input_dim, output_dim)
if torch.cuda.is_available() and usingGPU:
    model = model.cuda()

criterion = nn.MSELoss()  # MSE Stands for Mean Squared Error
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Linear regression
x_values = [i for i in range(11)]

x_train = np.array(x_values, dtype=np.float32)

# its 1D so lets make it 2D
x_train = x_train.reshape(-1, 1)

y_values = [2 * i + 1 for i in x_values]
# # slow way:
# y_values = []
# for i in x_values:
#     result = 2*1 +1
#     y_values.append(result)

# we make y_values numpy and 2D again
y_train = np.array(y_values, dtype=np.float32).reshape(-1, 1)

startTime = datetime.now()
for epoch in range(epochs):
    epoch += 1

    # Convert numpy array to tensor
    # inputs = torch.tensor(torch.from_numpy(x_train), requires_grad=True)
    # labels = torch.tensor(torch.from_numpy(y_train), requires_grad=True)
    inputs = torch.from_numpy(x_train)
    inputs.requires_grad = True
    if torch.cuda.is_available() and usingGPU:
        inputs = inputs.cuda()

    labels = torch.from_numpy(y_train)
    labels.requires_grad = True
    if torch.cuda.is_available() and usingGPU:
        labels = labels.cuda()

    # Clears gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward to get output
    outputs = model(inputs)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    print("epoch {}, loss {}".format(epoch, loss.data))

# Predicted values
model = model.cpu()
predicted = model(torch.from_numpy(x_train)).data.numpy()
print("Predicted values:\n", predicted)
print("Time elapsed:\n", datetime.now() - startTime)

if loadModel:
    model.load_state_dict(torch.load('model.pkl'))

if saveModel:
    # This saves only the parameters
    torch.save(model.state_dict(), 'model.pkl')


### Plot stuff ###
plt.clf()

# Plot true data that we know
plt.plot(x_train, y_train, "go", label="True data", alpha=0.5)

# Plot predicted data
plt.plot(x_train, predicted, "--", label="Predictions", alpha=0.5)

# Legend and plot show
plt.legend(loc="best")
plt.show()  # show plot
