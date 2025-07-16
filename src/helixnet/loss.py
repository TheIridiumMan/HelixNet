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


def softmax_crossentropy(y_pred, y_true):
    return mg.nnet.losses.softmax_crossentropy(y_pred.astype(np.int32), y_true.astype(np.int32))


def HuberLoss(y_pred, y_true, delta=1.0):
    error = y_true - y_pred
    is_small_error = mg.abs(error) <= delta

    squared_loss = 0.5 * (error**2)
    linear_loss = delta * (mg.abs(error) - 0.5 * delta)

    return mg.where(is_small_error, squared_loss, linear_loss)


def LogCoshLoss(y_pred, y_true):
    error = y_true - y_pred
    return mg.log(mg.cosh(error))


def FocalLoss(y_pred_sigmoid, y_true, alpha=0.25, gamma=2.0):
    p_t = y_pred_sigmoid * y_true + (1 - y_pred_sigmoid) * (1 - y_true)
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)

    bce = -mg.log(p_t)
    focal_modulator = (1 - p_t)**gamma

    return alpha_t * focal_modulator * bce


def HingeLoss(y_pred, y_true):
    # y_true should be -1 or 1
    return mg.maximum(0, 1 - y_true * y_pred)


def CosineSimilarityLoss(y_pred, y_true):
    # Normalize the vectors to unit length
    y_pred_norm = y_pred / mg.sqrt(mg.sum(y_pred**2, axis=-1, keepdims=True))
    y_true_norm = y_true / mg.sqrt(mg.sum(y_true**2, axis=-1, keepdims=True))

    # Calculate cosine similarity
    cosine_similarity = mg.sum(y_pred_norm * y_true_norm, axis=-1)

    # The loss is 1 minus the similarity
    return 1 - cosine_similarity


def BinaryCrossEntropy(y_pred, y_true):
    return -(y_true * mg.log(y_pred) + (1 - y_true) * mg.log(1 - y_pred))
