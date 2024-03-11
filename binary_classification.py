""" learning to make an ML classifier """
from multiprocessing import Process
import matplotlib.pyplot as plt
# import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split as tts
# import pandas as pd
import torch
from torch import nn
from torch.nn import Sequential
import helper_functions as hfns

N_SAMPLES = 1000
RANDOM_STATE = 42

def create_data():
    """ 
    create a toy dataset consisting of two sets of points
    arranged in a circle with differing distances from the
    center.
    """
    n_samples = N_SAMPLES
    x_samples, y_samples = make_circles(n_samples,
        				      noise=0.03,
				      random_state=RANDOM_STATE)
    return x_samples, y_samples

def condition_and_split_data(x_samp, y_samp):
    """ turn array data into torch tensor and perform train/test split """
    x_tensor = torch.from_numpy(x_samp).type(torch.float)
    y_tensor = torch.from_numpy(y_samp).type(torch.float)
    x_train, x_test, y_train, y_test = tts(x_tensor,
                                         y_tensor,
                                         test_size=0.2,
                                         random_state=RANDOM_STATE)
    return x_train, x_test, y_train, y_test

def plot_data(x_samp, y_samp):
    """ plot data in a separate process """
    plt.title("Visualization of Raw Data")
    plt.scatter(x=x_samp[:,0],
                y=x_samp[:,1],
                c=y_samp,
                cmap=plt.colormaps['RdYlBu'])
    plt.show()

def accuracy_fn(y_true, y_pred):
    """ calculate accuracy of predictions """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

def main():
    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=not-callable
    """ main entry point for script """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_samples, y_samples = create_data()
    # show plot of sample data
    Process(target=plot_data,
            kwargs={"x_samp": x_samples,
                    "y_samp": y_samples}).start()
    x_train, x_test, y_train, y_test = condition_and_split_data(x_samples,
                                                                y_samples)
    model_0 = Sequential(
        nn.Linear(in_features=2, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=1)).to(device)
    # Setup loss fn
    loss_fn = nn.BCEWithLogitsLoss()
    # Setup optimizer
    optimizer = torch.optim.Adam(params=model_0.parameters(),
                                lr=0.01)
    model_0.eval()
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    epochs = 10
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    ### Training loop
    for epoch in range(epochs):
        ### Training mode
        model_0.train()
        ### 1. Forward pass
        y_logits = model_0(x_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        ### 2. Calculate loss/accuracy
        loss = loss_fn(y_logits,
                       y_train)
        acc = accuracy_fn(y_true=y_train,
                          y_pred=y_pred)
        ### 3. Optimizer zero grad
        optimizer.zero_grad()
        ### 4. Loss backward (backpropagate)
        loss.backward()
        ### 5. optimizer.step
        optimizer.step()
        ### Testing
        model_0.eval()
        with torch.inference_mode():
            # 1. forward pass
            test_logits = model_0(x_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            # 2. Calculate test loss/acc
            test_loss = loss_fn(test_logits,
                                y_test)
            test_acc = accuracy_fn(y_true=y_test,
                                   y_pred=test_pred)
            if epoch % 10 == 0:
                print(f"epoch {epoch}, loss: {loss:.5f}  acc: {acc:.2f}%, \
test_loss: {test_loss:.2f}, test_acc: {test_acc:.2f}%")
    Process(target=show_boundary,
            kwargs={"model": model_0.to("cpu"),
                    "x_train": x_train.to("cpu"),
                    "x_test": x_test.to("cpu"), 
                    "y_train": y_train.to("cpu"),
                    "y_test": y_test.to("cpu")}).start()

def show_boundary(model, x_train, x_test, y_train, y_test):
    """ plot the 'decision boundary' for test and train data """
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    hfns.plot_decision_boundary(model, x_train, y_train)
    plt.subplot(1,2,2)
    plt.title("Test")
    hfns.plot_decision_boundary(model, x_test, y_test)
    plt.show()

if __name__ == "__main__":
    main()
