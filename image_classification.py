import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from multiprocessing import Process
from helper_functions import accuracy_fn
from timeit import default_timer as timer
from tqdm.auto import tqdm

BATCH_SIZE = 32
RANDOM_SEED = 42
LEARNING_RATE = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FashionMNISTV0(nn.Module):
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
    return self.layer_stack(x)

class FashionMNISTV1(nn.Module):
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
    return self.layer_stack(x)

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
  return DataLoader(dataset = dataset,
                    batch_size = batch_size,
                    shuffle = shuffle)

def show_random_images(data, rows = 4, cols = 4, cmap = "gray"):
  torch.manual_seed(RANDOM_SEED)
  fig = plt.figure(figsize=(9,9))
  class_names = data.classes
  for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(data), size=[1]).item()
    img, label = data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap = cmap)
    plt.title(class_names[label])
    plt.axis(False)
  plt.show()

def show_an_image(data, labels, class_names,  cmap = "gray"):
  random_idx = torch.randint(0, len(data), size=[1]).item()
  img, label = data[random_idx], labels[random_idx]
  plt.imshow(img.squeeze(), cmap = cmap)
  title = class_names[label]
  plt.title(title)
  plt.axis(False)
  plt.show()

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
  total_time = end - start
  print(f"train time on {device=}: {total_time:3f} seconds")

def train_model(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer) -> float:
  train_loss = 0
  model.train()
  for batch, (x, y) in enumerate(data_loader):
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    # 1. Forward pass
    y_pred = model(x)
    # 2. Calculate loss
    loss = loss_fn(y_pred, y)
    train_loss += loss
    # 3. Optimize
    optimizer.zero_grad()
    # 4. back-propagate the loss
    loss.backward()
    # 5. optimizer step
    optimizer.step()
  train_loss /= len(data_loader)
  return train_loss

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn) -> dict:
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
      acc += accuracy_fn(y_true = y,
                         y_pred = y_pred.argmax(dim=1))
    # Scale loss and acc to find average loss/accuracy per batch
    loss /= len(data_loader)
    acc /= len(data_loader)
  return {"model_name": model.__class__.__name__,
          "model_loss": loss.item(),
          "model_acc": acc}      

def main():
  train_data, test_data = get_data()
  (image, label) = train_data[0]
  class_names = train_data.classes
  class_to_idx = train_data.class_to_idx
  Process(target = show_random_images,
          kwargs = {"data": train_data}).start()
  train_dataloader = dataloader(train_data)
  test_dataloader = dataloader(test_data, shuffle = False)
  train_features_batch, train_labels_batch = next(iter(train_dataloader))
  Process(target = show_an_image,
          kwargs = {"data": train_features_batch,
                    "labels": train_labels_batch,
                    "class_names": class_names}).start()
  model0 = FashionMNISTV0(input_shape = 784, # 28 * 28 = 784
                          output_shape = 10,
                          hidden_units = 10).to(DEVICE)
  model1 = FashionMNISTV1(input_shape = 784,
                          output_shape = 10,
                          hidden_units = len(class_names)).to(DEVICE)
  model = model1
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(params=model.parameters(),
                              lr = LEARNING_RATE)

  torch.manual_seed(RANDOM_SEED)
  train_time_start = timer()
  epochs = 3
  for epoch in tqdm(range(epochs)):
    print(f"{epoch=} -----------------------------")
    #### Training
    train_loss  = train_model(model,
                              train_dataloader,
                              loss_fn,
                              optimizer)
    #### Testing
    test_rslts = eval_model(model,
                            test_dataloader,
                            loss_fn,
                            accuracy_fn)
    model_name = test_rslts.get("model_name")
    test_loss = test_rslts.get("model_loss")
    test_acc = test_rslts.get("model_acc")
    print(f"{model_name=}, Train loss: {train_loss:4}, test loss {test_loss:4f}, test acc: {test_acc:4f}")
  train_time_end = timer()
  total_train_time = print_train_time(start=train_time_start,
                                      end = train_time_end,
                                      device=DEVICE)    
        



if __name__ == "__main__":
  main()
