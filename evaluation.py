import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def eval(model):
    ['evaluation'] 
    csv_file = "data\\urls_feature_val.csv"
    inputs = pd.read_csv(csv_file).values.astype(np.float32)[:, 1:]
    data, target = inputs[:, :-1], inputs[:, -1:]
    s = StandardScaler()
    data = s.fit_transform(data)
  
    model.eval()
    with torch.no_grad():
        _, pred = model(torch.from_numpy(data)).max(dim=-1)
    print(classification_report(target, pred.numpy()))



