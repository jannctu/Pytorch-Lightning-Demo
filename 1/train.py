from net import LitMNIST
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import os
# Init our model
mnist_model = LitMNIST()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(root = '../', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=32)

test_ds = MNIST(root = '../', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_ds, batch_size=32)

# Initialize a trainer
trainer = pl.Trainer(gpus=1, max_epochs=10, progress_bar_refresh_rate=20)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)
trainer.test(mnist_model, test_loader)