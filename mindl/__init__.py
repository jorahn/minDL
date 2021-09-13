__all__ = ['data']
import mindl.data

import torch
from torch import nn
from torch.utils.data import DataLoader

import tensorflow as tf
from tensorflow import keras

BACKENDS = ['pytorch', 'tensorflow'] # TODO: add flax

class Project(object):
    def __init__(self, backend, device):
        if not backend in BACKENDS:
            raise ValueError(f'Invalid backend {backend}. Must be in {str(BACKENDS)}')
        self.backend = backend
        self.device = device
    
    def get_data(self, train_data, train_labels, test_data=None, test_labels=None, shuffle=True, batch_size=64)
        if self.backend == 'pytorch':
            train_dl = DataLoader(zip(train_data, train_labels), batch_size=batch_size)
            if not test_data:
                return train_dl
            else:
                test_dl = DataLoader(zip(test_data, test_labels), batch_size=batch_size)
                return train_dl, test_dl
        if self.backend == 'tensorflow':
            train_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))
            if shuffle: train_ds = train_ds.shuffle()
            if not test_data:
                return train_ds.batch(batch_size)
            else:
                test_ds = tf.data.Dataset()
                return test_ds.batch(batch_size)
def make_data():

def get_fashion_mnist():
    # Download training data from open datasets.
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_data, test_data

def get_dl(train_data, test_data, batch_size=64):
    # Create data loaders.
    train_dl = DataLoader(train_data, batch_size=batch_size)
    test_dl = DataLoader(test_data, batch_size=batch_size)
    return train_dl, test_dl

def get_model(device):
    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    return model


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if batch % 100 == 0:
        #    loss, current = loss.item(), batch * len(X)
        #    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

def run(batch_size=64, learning_rate=1e-3, epochs=5, model=None, device=None):
    if not device:
        # Get cpu or gpu device for training.
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    train_data, test_data = get_fashion_mnist()
    train_dl, test_dl = get_dl(train_data, test_data, batch_size)
    
    if not model:
        model = get_model(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        train_loss = train(train_dl, model, loss_fn, optimizer, device)
        test_loss, test_acc = test(test_dl, model, loss_fn, device)
        print(f"Epoch {t+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy {test_acc:.3%}")
    #print("Done!")

    #torch.save(model.state_dict(), "model.pth")
    #print("Saved PyTorch Model State to model.pth")

    #classes = [
    #    "T-shirt/top",
    #    "Trouser",
    #    "Pullover",
    #    "Dress",
    #    "Coat",
    #    "Sandal",
    #    "Shirt",
    #    "Sneaker",
    #    "Bag",
    #    "Ankle boot",
    #]

    #model.eval()
    #x, y = test_data[0][0], test_data[0][1]
    #with torch.no_grad():
    #    pred = model(x)
    #    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    #    print(f'Predicted: "{predicted}", Actual: "{actual}"')
