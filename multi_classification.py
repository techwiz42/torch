""" learning to make an ML classifier """
from multiprocessing import Process
import matplotlib.pyplot as plt
# import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split as tts
# import pandas as pd
import torch
from torch import nn
import helper_functions as hfns

NUM_CLASSES = 4
NUM_FEATURES = 2
NUM_SAMPLES = 1000
RANDOM_STATE = 42

class BlobModel(nn.Module):
    """ multi-layered model for multi class characterization """
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features = input_features, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features = output_features))

    def forward(self, x):
        """ the mandatory forward method """
        return self.linear_layer_stack(x)

def create_data():
    # pylint: disable-msg=unbalanced-tuple-unpacking
    """ 
    create a toy dataset consisting of two sets of points
    arranged in a circle with differing distances from the
    center.
    """
    x_blob, y_blob = make_blobs(n_samples = NUM_SAMPLES,
                                n_features = NUM_FEATURES,
                                centers = NUM_CLASSES,
                                cluster_std = 1.5,
                                random_state = RANDOM_STATE)
    return x_blob, y_blob

def condition_and_split_data(x_blob, y_blob, device):
    """ turn array data into torch tensor and perform train/test split """
    x_tensor = torch.from_numpy(x_blob).type(torch.float).to(device)
    y_tensor = torch.from_numpy(y_blob).type(torch.LongTensor).to(device)
    x_train, x_test, y_train, y_test = tts(x_tensor,
                                         y_tensor,
                                         test_size=0.2,
                                         random_state=RANDOM_STATE)
    return x_train, x_test, y_train, y_test

def plot_data(x_blob, y_blob):
    """ plot data in a separate process """
    plt.title("Visualization of Raw Data")
    plt.scatter(x=x_blob[:,0],
                y=x_blob[:,1],
                c=y_blob,
                cmap=plt.colormaps['RdYlBu'])
    plt.show()

def accuracy_fn(y_true, y_pred):
    """ Calculate accuracy of predicted results """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

def plot_results(model, x_blob_train, y_blob_train, x_blob_test, y_blob_test):
    """ plot train and test results """
    plt.figure(figsize=(10, 7))
    plt.subplot(1,2,1)
    plt.title("Train")
    hfns.plot_decision_boundary(model, x_blob_train, y_blob_train)
    plt.subplot(1,2,2)
    plt.title("Test")
    hfns.plot_decision_boundary(model, x_blob_test, y_blob_test)
    plt.show()

def main():
    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=not-callable
    """ main entry point for script """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_blob, y_blob = create_data()
    # show plot of sample data
    Process(target=plot_data,
            kwargs={"x_blob": x_blob,
                    "y_blob": y_blob}).start()
    x_blob_train, x_blob_test, y_blob_train, y_blob_test = condition_and_split_data(x_blob,
                                                                y_blob, device)
    model = BlobModel(input_features=2,
                      output_features=4,
                      hidden_units=256).to(device)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr = 0.1)


    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)

    epochs = 100
    x_blob_train, x_blob_test = x_blob_train.to(device), x_blob_test.to(device)
    y_blob_train, y_blob_test = y_blob_train.to(device), y_blob_test.to(device)

    for epoch in range(epochs):
        model.train()

        y_logits = model(x_blob_train)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        loss = loss_fn(y_logits, y_blob_train)
        acc = accuracy_fn(y_true=y_blob_train,
                          y_pred=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(x_blob_test)
            #test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

            test_loss = loss_fn(test_logits, y_blob_test)
            test_acc = accuracy_fn(y_true = y_blob_train,
                                   y_pred = y_pred)
        if epoch % 10 == 0:
            print(f"{epoch=}, loss {loss:4f}, acc {acc:2f}, test_loss\
 {test_loss:4f}%, test_acc {test_acc:2f}%")

    #model.eval()
    #with torch.inference_mode():
        #logits = model(x_blob_test)
    #y_preds = torch.softmax(logits, dim=1).argmax(dim=1)
    Process(target=plot_results,
            kwargs={"model": model.to("cpu"),
                    "x_blob_train": x_blob_train.to("cpu"),
                    "y_blob_train": y_blob_train.to("cpu"),
                    "x_blob_test": x_blob_test.to("cpu"),
                    "y_blob_test": y_blob_test.to("cpu")}).start()

if __name__ == "__main__":
    main()
