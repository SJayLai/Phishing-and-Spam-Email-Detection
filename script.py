
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import yaml

from evaluation import eval
from model import network
from datasets import PhishDataLoader



torch.manual_seed(12)
np.random.seed(12)

def get_model(n_inputs, hidden_units_1=300, hidden_units_2=100, n_outputs=2):
    print(['model building'])
    return network(n_inputs, hidden_units_1, hidden_units_2, n_outputs)


def save_model(model, filename):
    print(['save model'])
    torch.save(model.state_dict(), filename)

def config_loading(config_file, config=None):
    print(['config'])
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    return config 

    
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
            save_model(model, f'ckpt\\ckpt_{epoch+1}.pth')

def main():
    config_file = "config\\config.yaml"
    config = config_loading(config_file)

    batch_size = config["TRAIN_SETTING"]["BATCH_SIZE"]
    learning_rate = config["TRAIN_SETTING"]["LR"]
    epochs = config["TRAIN_SETTING"]["EPOCHS"]

    train_data, val_data = PhishDataLoader('data', split='train'), PhishDataLoader('data', split='val')
    train_loader, val_loader = DataLoader(train_data, batch_size=batch_size), DataLoader(val_data, batch_size=batch_size)

    model = get_model(train_data.n_features)
    model.train()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    train(model, train_loader, val_loader, epochs, optimizer, criterion)


if __name__ == "__main__":
    main()



 