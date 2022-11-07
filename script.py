
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import numpy as np
import json
import yaml
import math

from evaluation import eval
from model import DL_Model
from datasets import PhishDataLoader



torch.manual_seed(12)
np.random.seed(12)

def get_model(n_inputs, hidden_units_1=300, hidden_units_2=100, n_outputs=2):
    print(['model building'])
    return DL_Model(n_inputs, hidden_units_1, hidden_units_2, n_outputs)


def save_model(model, filename):
    print(['save model'])
    torch.save(model.state_dict(), filename)


def config_loading(config_file, config=None):
    print(['config'])
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    return config 

    
def train(model, train_loader, epochs, optimizer, criterion, interval=100, save_epoch=10):
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
            eval(model)
            save_model(model, f'ckpt\\ckpt_{epoch+1}.pth')

def main():
    config_file = "config\\config.yaml"
    config = config_loading(config_file)

    batch_size = config["TRAIN_SETTING"]["BATCH_SIZE"]
    learning_rate = config["TRAIN_SETTING"]["LR"]
    epochs = config["TRAIN_SETTING"]["EPOCHS"]

    train_data = PhishDataLoader('data', split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size)

    model = get_model(train_data.n_features)
    
    model.train()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    train(model, train_loader, epochs, optimizer, criterion)



 

 

if __name__ == "__main__":
    main()

# from sklearn.model_selection import train_test_split 
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# import joblib
# from model import ML_Model
# ml_model = ML_Model(config)
# model = ml_model()
# model.fit(train_X, train_y)
# save_name = "ckpt\\" + config["MODEL_NAME"]
# joblib.dump(model, save_name)

# test_y_predicted = model.predict(test_X)

# from sklearn.metrics import classification_report
# print(classification_report(test_y, test_y_predicted))

 