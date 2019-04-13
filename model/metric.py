import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

def roc_metric(output, target):
    with torch.no_grad():
        y_true = target.cpu().numpy()
        y_pred =  output.cpu().numpy()
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())
        return auc(fpr, tpr)

def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
