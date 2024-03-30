""" Experimenting with image classifiers """
from timeit import default_timer as timer
import pandas as pd
from tqdm.auto import tqdm
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from torch import nn
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn

BATCH_SIZE = 32
RANDOM_SEED = 42
LEARNING_RATE = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FashionMNISTV0(nn.Module):
    """ Purely linear classifier """
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape,
                      out_features = hidden_units),
            nn.Linear(in_features = hidden_units,
                      out_features = output_shape)
        )

    def forward(self, x):
        """ Overrides nn.Module's forward method """
        return self.layer_stack(x)

class FashionMNISTV1(nn.Module):
    """ Classifier with ReLU """
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features = input_shape,
                    out_features = hidden_units),
          nn.ReLU(),
          nn.Linear(in_features = hidden_units,
                    out_features = output_shape),
          nn.ReLU()
        )
    def forward(self, x: torch.Tensor):
        """ Overrides nn.Module's forward method """
        return self.layer_stack(x)

class FashionMNISTV2(nn.Module):
    """ Convolutional Neural Network Classifier """
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=hidden_units*7*7,
                    out_features=output_shape)
       )

    def forward(self, x):
        """ Overrides nn.Module's forward method """
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x= self.classifier(x)
        return x

def get_data():
    """ Get data from FashionMNIST dataset """
    train_data = datasets.FashionMNIST(
        root = "data", #where to download to?
        train = True,
        download = True,
        transform = ToTensor(),
        target_transform = None
    )
    test_data = datasets.FashionMNIST(
        root = "data",
        train = False,
        download = True,
        transform = ToTensor(),
        target_transform = None
    )
    return train_data, test_data

def dataloader(dataset, batch_size=BATCH_SIZE, shuffle = True):
    """ Turns a dataset into a DataLoader with batches of data """
    return DataLoader(dataset = dataset,
                      batch_size = batch_size,
                      shuffle = shuffle)

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """ tracks the time required to train the model """
    total_time = end - start
    print(f"train time on {device=}: {total_time:3f} seconds")

def train_model(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                acc_fun,
                optimizer: torch.optim.Optimizer) -> float:
    # pylint: disable-msg=unused-variable
    """ trains the model based on training dataset """
    train_loss = 0
    train_acc = 0
    model.train()
    for batch, (x, y) in enumerate(data_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # 1. Forward pass
        y_pred = model(x)
        # 2. Calculate loss, accuracys
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += acc_fun(y_true = y,
                             y_pred = y_pred.argmax(dim=1))
        # 3. Optimize
        optimizer.zero_grad()
        # 4. back-propagate the loss
        loss.backward()
        # 5. optimizer step
        optimizer.step()
    train_acc /= len(data_loader)
    train_loss /= len(data_loader)
    print(f"train_loss: {train_loss:.5f} train_acc: {train_acc:.2f}%")
    return train_loss

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               acc_fun) -> dict:
    """ Evaluates the model based on test dataset """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # Make predictions
            y_pred = model(x)
            # Accumulate the loss and accuracy per batch
            loss += loss_fn(y_pred, y)
            acc += acc_fun(y_true = y,
                           y_pred = y_pred.argmax(dim=1))
        # Scale loss and acc to find average loss/accuracy per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
    print(f"test_loss: {loss:.5f} test_acc: {acc:.2f}%")
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}      

def main():
    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=unused-variable
    """ The main entry point for the image classifiers """
    train_data, test_data = get_data()
    class_names = train_data.classes
    train_dataloader = dataloader(train_data)
    test_dataloader = dataloader(test_data, shuffle = False)
    model0 = FashionMNISTV0(input_shape = 784, # 28 * 28 = 784
                            output_shape = 10,
                            hidden_units = 10).to(DEVICE)
    model1 = FashionMNISTV1(input_shape = 784,
                            output_shape = len(class_names),
                            hidden_units = 10).to(DEVICE)
    model2 = FashionMNISTV2(input_shape=1,
                            hidden_units = 10,
                            output_shape=len(class_names)).to(DEVICE)
    models = [model0, model1, model2]
    loss_fn = nn.CrossEntropyLoss()
    results = []
    for model in models:
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr = LEARNING_RATE)
        torch.cuda.manual_seed(RANDOM_SEED)
        train_time_start = timer()
        epochs = 3
        result = None
        for epoch in tqdm(range(epochs)):
            print(f"{epoch=} -----------------------------")
            #### Training
            train_model(model,
                        train_dataloader,
                        loss_fn,
                        accuracy_fn,
                        optimizer)
            #### Testing
            result = eval_model(model,
                                test_dataloader,
                                loss_fn,
                                accuracy_fn)
        train_time_end = timer()
        print_train_time(start=train_time_start,
                         end = train_time_end,
                         device=DEVICE)
        result.update({"train time": train_time_end - train_time_start})
        results.append(result)
    compare_results = pd.DataFrame(results)
    print(compare_results)

if __name__ == "__main__":
    main()
