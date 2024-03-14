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

LOSS_FN = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.SGD

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
                          hidden_units = 10)
  optimizer = OPTIMIZER(params=model0.parameters(),
                        lr = LEARNING_RATE)

  torch.manual_seed(RANDOM_SEED)
  train_time_start_on_cpu = timer()
  epochs = 3

  for epoch in tqdm(range(epochs)):
    train_loss = 0
    print(f"epoch {epoch} --------------")
    for batch, (x, y) in enumerate(train_dataloader): # x is data, y is labels
      model0.train() # Put model into training mode
      #1 forward pass
      y_pred = model0(x)
      #2 calculate loss
      loss = LOSS_FN(y_pred, y)
      train_loss+= loss
      #3 Optimize
      optimizer.zero_grad()
      #4 loss backward
      loss.backward()
      #5 optimizer step
      optimizer.step()
      if batch % 400 == 0:
        print(f"Looked at (batch * len(x)/{len(train_dataloader.dataset)} samples")
    # divide total train loss by length of train dataloader
    train_loss /= len(train_dataloader)

    #### Testing
    test_loss, test_acc = 0, 0
    #put model in evaluation mode
    model0.eval()
    with torch.inference_mode():
      for x_test, y_test in test_dataloader: # x is data, y is labels
        #1. forward pass
        test_pred = model0(x_test)
        #2. calculate loss
        test_loss += LOSS_FN(test_pred, y_test)
        #3. calculate accuracy
        test_acc += accuracy_fn(y_true = y_test, y_pred = test_pred.argmax(dim=1))
      # calculate test loss ave per batch
      test_loss /= len(test_dataloader)
      test_acc /= len(test_dataloader)

    print(f"Traain loss: {train_loss:4}, test loss {test_loss:4f}, test acc: {test_acc:4f}")
  train_time_end_on_cpu = timer()
  total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                              end = train_time_end_on_cpu,
                                              device=(next(model0.parameters())))    
        



if __name__ == "__main__":
  main()
