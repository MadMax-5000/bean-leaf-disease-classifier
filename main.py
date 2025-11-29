import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device is {device}")

train_df = pd.read_csv("/content/bean-leaf-lesions-classification/train.csv")
val_df = pd.read_csv("/content/bean-leaf-lesions-classification/val.csv")

train_df["image:FILE"] = "/content/bean-leaf-lesions-classification/" + train_df["image:FILE"]
val_df["image:FILE"] = "/content/bean-leaf-lesions-classification/" + val_df["image:FILE"]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

class AstraDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(dataframe["category"]).to(device)

    def __len__(self):
        return self.dataframe.shape[0]
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = Image.open(img_path)
        if self.transform:
            image = (self.transform(image)/255.0).to(device) # normalization images
        return image, label

train_dataset = AstraDataset(dataframe = train_df, transform = transform)
val_dataset = AstraDataset(dataframe = val_df, transform=transform)

LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 20

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size= BATCH_SIZE, shuffle=True)

GoogleNet = models.googlenet(weights="DEFAULT")

for param in GoogleNet.parameters():
    param.requires_grad = True

GoogleNet.fc

num_classes = len(train_df["category"].unique())
num_classes

GoogleNet.fc = torch.nn.Linear(GoogleNet.fc.in_features, num_classes)
GoogleNet.fc

GoogleNet.to(device)

loss_fun = nn.CrossEntropyLoss()
optimizer = Adam(GoogleNet.parameters(), lr=LR)

total_loss_train_plot = []
total_acc_train_plot = []

for epoch in range(EPOCHS):
    total_train_loss = 0
    total_train_acc = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        ouputs = GoogleNet(inputs)
        train_loss = loss_fun(ouputs, labels)
        total_train_loss += train_loss.item()
        train_loss.backward()
        train_acc = (torch.argmax(ouputs, axis=1) == labels).sum().item()
        total_train_acc += train_acc
        optimizer.step()
    total_loss_train_plot.append(round(total_train_loss/1000, 4))
    total_acc_train_plot.append(round(total_train_acc/train_dataset.__len__()* 100, 4))
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss{round(total_train_loss/1000, 4)} , Train Acc : {round(total_train_acc/train_dataset.__len__()* 100, 4)} %")

with torch.no_grad():
    total_acc_test = 0
    for inputs, labels in val_loader:
        prediction = GoogleNet(inputs)
        acc = (torch.argmax(prediction, axis=1) == labels).sum().item()
        total_acc_test += acc

print(total_acc_test/val_dataset.__len__()*100, 2)

GoogleNet = models.googlenet(weights="DEFAULT")

for param in GoogleNet.parameters():
    param.requires_grad = False

GoogleNet.fc = torch.nn.Linear(GoogleNet.fc.in_features, num_classes)
GoogleNet.fc.requires_grad_ = True
GoogleNet.to(device)

loss_fun = nn.CrossEntropyLoss()
optimizer = Adam(GoogleNet.parameters(), lr=LR)

total_loss_train_plot = []
total_acc_train_plot = []

for epoch in range(EPOCHS):
    total_train_loss = 0
    total_train_acc = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        ouputs = GoogleNet(inputs)
        train_loss = loss_fun(ouputs, labels)
        total_train_loss += train_loss.item()
        train_loss.backward()
        train_acc = (torch.argmax(ouputs, axis=1) == labels).sum().item()
        total_train_acc += train_acc
        optimizer.step()
    total_loss_train_plot.append(round(total_train_loss/1000, 4))
    total_acc_train_plot.append(round(total_train_acc/train_dataset.__len__()* 100, 4))
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss{round(total_train_loss/1000, 4)} , Train Acc : {round(total_train_acc/train_dataset.__len__()* 100, 4)} %")

with torch.no_grad():
    total_acc_test = 0
    for inputs, labels in val_loader:
        prediction = GoogleNet(inputs)
        acc = (torch.argmax(prediction, axis=1) == labels).sum().item()
        total_acc_test += acc

print(total_acc_test/val_dataset.__len__()*100, 2)
