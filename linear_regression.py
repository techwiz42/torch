"""
linear regression formula:
	y = f(X[i], beta + epsilon
"""
from multiprocessing import Process
from matplotlib import pyplot as plt
import torch
from torch import nn


RANDOM_SEED = 42

def create_dataset():
    """
	creates a set of 50 points and splits it 
	into a test set and a training set
    """
    # Create *known* parameters
    weight = 0.7
    bias = 0.3

    # Create some data
    start = 0
    end = 1
    step = 0.02
    tensor_x = torch.arange(start, end, step).unsqueeze(dim=1)
    tensor_y = weight * tensor_x + bias

    # Split into training, validation and test datasets
    train_split = int(0.8 * len(tensor_x))
    x_train, y_train = tensor_x[:train_split], tensor_y[:train_split]
    x_test, y_test = tensor_x[train_split:], tensor_y[train_split:]
    return x_train, y_train, x_test, y_test

def plot_predictions(train_data,
                    train_labels,
                    test_data,
                    test_labels,
                    predictions=None):
    """
	plots the test, train and predicted data
    """
    plt.figure(num=1, figsize=(10, 7))
    plt.suptitle("Training, Test and Predicted Data")
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    # Plot testing data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show legend
    plt.legend(prop={"size": 14})
    plt.show()

def plot_loss_values(epochs,
                    loss,
                    test_loss):
    """
	plots actual loss and test loss
    """
    plt.figure(num=2, figsize=(10,7))
    plt.suptitle("Training Loss vs Test Loss")
    plt.scatter(epochs, loss, c="g", label="Loss Values")
    plt.scatter(epochs, test_loss, c="b", label="Test Loss")
    plt.legend(prop={"size": 14})
    plt.show()

class LinearRegressionModel(nn.Module):
    """
	gradient descent model - torcn.nn is the neural network module.
	torch.optim contains torch optimizers.
    """

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                            requires_grad = True,
                                            dtype = torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                            requires_grad = True,
                                            dtype = torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        overrides forward method of nn.Module - required method
        """
        return self.weights * x + self.bias

def main():
    """
        Main entry point for program
    """
    torch.manual_seed(RANDOM_SEED)
    x_train, y_train, x_test, y_test = create_dataset()
    model_0 = LinearRegressionModel()

    # Setup a loss function
    loss_fn = nn.L1Loss()

    # Setup an optimizer
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

    # Tracking experiments
    epoch_count = []
    loss_values = []
    test_loss_values = []

    # An epoch is one loop through the data
    epochs = 200
    # Set up training loop - steps 0 - 5 are the guts of pytorch ML 
    for epoch in range(epochs):
        epoch_count.append(epoch)
        # 0. Set model to training mode
        model_0.train()
	# 1. train the model
        y_pred = model_0(x_train)
	# 2. calculate the loss
        loss = loss_fn(y_pred, y_train)
	# 3. Optimize
        optimizer.zero_grad()
	# 4. back propagate
        loss.backward()
	# 5. step the optimizer
        optimizer.step()

        # Set model to testing mode
        model_0.eval()
        with torch.inference_mode():
            test_pred = model_0(x_test)
        test_loss = loss_fn(test_pred, y_test)
        loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())

    with torch.inference_mode():
        y_preds = model_0(x_test).detach().numpy()


    # Plot predicted vs actual
    prediction_job = Process(
            target=plot_predictions,
            kwargs={"train_data":x_train,
                    "train_labels":y_train,
                    "test_data":x_test,
                    "test_labels":y_test,
                    "predictions":y_preds}
            )
    prediction_job.start()

    # Plot loss vs test loss
    loss_job = Process(
            target=plot_loss_values,
            kwargs={"epochs":epoch_count,
                    "loss": loss_values,
                    "test_loss": test_loss_values}
            )
    loss_job.start()

if __name__ == "__main__":
    main()
