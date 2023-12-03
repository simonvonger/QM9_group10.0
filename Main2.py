import os
import torch
import torch.nn as nn
import numpy
from Dataloader import DataLoaderQM9
from Model import PaiNN
from Training import Trainer
from Model import mse, mae
import matplotlib.pyplot as plt

# Define the Model in the global scope
Model = PaiNN(r_cut=5, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def saveModel(Model, path="./PaiNNModel.pth"):
    torch.save(Model.state_dict(), path)

def test():
    path = "PaiNNModel.pth"
    Model.load_state_dict(torch.load(path))
    Model.eval()  # Set the model to evaluation mode
    train_set = DataLoaderQM9(batch_size=2)
    for batch in enumerate(train_set):
        predictions = Model(batch)
        print(predictions)

def training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} will be used for training the PaiNN model")
    
    # Use the global Model in the training function
    optimizer = torch.optim.Adam(params=Model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    train_set = DataLoaderQM9(r_cut=5, batch_size=100)
    
    trainer = Trainer(
        Model=Model,
        loss=mse,
        target=2,
        optimizer=optimizer,
        Dataloader=train_set,
        scheduler=scheduler,
        device=device
    )
    trainer._train(num_epoch=100, early_stopping=30)
    trainer.plot_data()

if __name__ == "__main__":
    training()
    saveModel(Model)
    test()
