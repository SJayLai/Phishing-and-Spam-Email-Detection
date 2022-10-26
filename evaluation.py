import torch
from sklearn.metrics import classification_report

def eval(model, val_loader):
    ['evaluation']
    model.eval()
    pred_result = []
    target_result = []
    for data, target in val_loader:
        with torch.no_grad():
            _, pred = model(data).max(dim=-1)
            _, targ = target.max(dim=-1)
            pred_result.append(pred.view(-1, 1))
            target_result.append(targ.view(-1, 1))

    pred_result = torch.cat(pred_result, dim=0).view(-1).numpy()
    target_result = torch.cat(target_result, dim=0).view(-1).numpy()
    print(classification_report(target_result, pred_result))