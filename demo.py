import joblib
import yaml
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from model import DL_Model

def config_loading(config_file, config=None):
    print(['config'])
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    return config


def demo(csv_file, config_file):
    config = config_loading(config_file)
    inputs = pd.read_csv(csv_file).values.astype(np.float32)[:, 1:]
    data, target = inputs[:, :-1], inputs[:, -1:]
    s = StandardScaler()
    data = s.fit_transform(data)

    res = np.zeros((len(data)), dtype=np.float32)

    test_model_list = config["TEST_MODEL_LIST"]
    for model_name in test_model_list:
        model = joblib.load('ckpt\\' + model_name)
        pred = model.predict(data)
        res += pred

    model = DL_Model(data.shape[-1])
    model.load_state_dict(torch.load("ckpt\\ckpt_70.pth"))
    model.eval()
    with torch.no_grad():
        _, pred = model(torch.from_numpy(data)).max(dim=-1)
    pred = pred.numpy()
    res += pred

    vote_res = (res > (len(test_model_list) + 1) // 2).astype(np.int32)
    print(classification_report(target, vote_res))


def main():
    config_file = "config\\config.yaml"
    csv_file = "data\\urls_feature_val.csv"
    demo(csv_file, config_file)



if __name__ == "__main__":
    main()