import numpy as np
import mygrad as mg


def CatCrossEntropy(y_pred, y_true):
    return -(mg.sum(y_true * mg.log(y_pred + 1e-8))) / len(y_pred)


def MeanSquaredError(y_pred, y_true):
    return mg.mean((y_pred - y_true) * (y_pred - y_true))


def MeanAbsError(y_pred, y_true):
    return mg.mean(mg.abs(y_pred - y_true))


def Kullback_Leibler(y_pred, y_true):
    return y_pred * mg.log(y_pred / y_true) + (1 - y_pred) * \
        mg.log((1 - y_pred) / (1 - y_true))
