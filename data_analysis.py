import torch

"""
linear regression formula:
    y = f(X[i], beta) + epsilon
"""

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create some data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Split into training, validation and test datasets
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Visualize the data
from matplotlib import pyplot as plt

def plot_predictions(train_data=X_train,
                    train_labels=y_train,
                    test_data=X_test,
                    test_labels=y_test,
                    predictions=None):
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    # Plot testing data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show legend
    plt.legend(prop={"size": 14})
    plt.show()

""" 
gradient descent model - torch.nn is the neural network module.
torch.optim contain torch optimizers. 
"""
from torch import nn
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                            requires_grad = True,
                                            dtype = torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                            requires_grad = True,
                                            dtype = torch.float))
    
    """ overrides forward method of nn.Module - reqired """    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

model_0 = LinearRegressionModel()
# Show the model's parameters:
print(f"{list(model_0.parameters())=}")
# Show the model's NAMED parameters
print(f"{list(model_0.state_dict())=}")

# Setup a loss function
loss_fn = nn.L1Loss()

# Setup an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# An epoch is one loop through the data
epochs = 200
# Set up training loop
for epoch in range(epochs):
    # Set model to training mode
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Set model to testing mode
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
    test_loss = loss_fn(test_pred, y_test)
    if epoch % 10 == 0:
        print(f"{epoch=} | {loss=} |{test_loss=}")

with torch.inference_mode():
    y_preds = model_0(X_test).detach().numpy()

plot_predictions(X_train, y_train, X_test, y_test, y_preds)
