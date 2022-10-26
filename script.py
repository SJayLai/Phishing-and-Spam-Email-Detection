
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np

from evaluation import eval
from model import network
from datasets import PhishDataLoader


def get_model(n_inputs, hidden_units_1=300, hidden_units_2=100, n_outputs=2):
    print(['model building'])
    return network(n_inputs, hidden_units_1, hidden_units_2, n_outputs)


def save_model(model, filename):
    print(['save model'])
    torch.save(model.state_dict(), filename)

    
def train(model, train_loader, val_loader, epochs, optimizer, criterion, interval=100, save_epoch=10):
    print(['train'])
    iters = 0
    interval_loss = 0.0
    
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
       
            iters += 1

            optimizer.zero_grad()
         
            pred = model(data)
            loss = criterion(pred, target)

            interval_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            if iters % interval == 0:
                interval_loss /= interval
                print("Epoch %d, Itrs %d, Loss=%f" % (epoch + 1, iters, interval_loss))
                interval_loss = 0.0

        if (epoch + 1) % save_epoch == 0:
            eval(model, val_loader)
            save_model(model, f'ckpt_{epoch+1}.pth')
    

if __name__ == "__main__":

    batch_size = 1024
    learning_rate = 0.01
    epochs = 50

    tra_data, val_data = PhishDataLoader('data', split='train'), PhishDataLoader('data', split='val')

    train_loader = DataLoader(tra_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    model = get_model(tra_data.n_features)
    model.train()


    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()



    train(model, train_loader, val_loader, epochs, optimizer, criterion)



 