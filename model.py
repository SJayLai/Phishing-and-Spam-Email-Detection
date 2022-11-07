import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
class DL_Model(nn.Module):
    def __init__(self, n_inputs, hidden_units_1=300, hidden_units_2=100, n_outputs=2) -> None:
        super().__init__()
        self.pipeline1 = nn.Sequential(
            nn.Linear(n_inputs, hidden_units_1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units_1),
            nn.Linear(hidden_units_1, hidden_units_1),
            nn.BatchNorm1d(hidden_units_1),
           
        )
        self.pipeline2 = nn.Sequential(
            nn.Linear(hidden_units_1, hidden_units_2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units_2),
            nn.Linear(hidden_units_2, hidden_units_2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units_2),
            nn.Linear(hidden_units_2, hidden_units_1),
        )
        self.output_pipeline = nn.Sequential(nn.Linear(hidden_units_1, n_outputs))
        self.weight_init()
        
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                nn.init.uniform_(m.all_weights, -0.1, 0.1) 
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, data):
        out = self.pipeline1(data)
        out = self.pipeline2(out)
        out = self.output_pipeline(out)
        return out

class ML_Model():
    def __init__(self, config, n_estimators=100):
        self.n_estimators = n_estimators
        self.config = config
    def __call__(self):
        model_name = self.config["MODEL_NAME"]
        
        if model_name == "Random_Forest":
            print("using random forest")
            return RandomForestClassifier(n_estimators=self.n_estimators, 
                                          random_state=0)
        elif model_name == "LGBM":
            print("using LGBM")
            return LGBMClassifier(objective='binary', 
                                  learning_rate=0.05, 
                                  n_estimators=100, 
                                  random_state=0)
        elif model_name == "AdaBoost":
            print("using adaboost")
            return AdaBoostClassifier(n_estimators=self.n_estimators, 
                                      random_state=0)
        elif model_name == "XGBoost":
            print("using xgboost")
            return XGBClassifier(n_estimators=self.n_estimators, 
                                 random_state=0)

       
