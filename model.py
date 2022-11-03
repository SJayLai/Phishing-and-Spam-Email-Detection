import torch.nn as nn
class network(nn.Module):
    def __init__(self, n_inputs, hidden_units_1, hidden_units_2, n_outputs) -> None:
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

    def forward(self, data):
        out = self.pipeline1(data)
        out = self.pipeline2(out)
        out = self.output_pipeline(out)
        return out