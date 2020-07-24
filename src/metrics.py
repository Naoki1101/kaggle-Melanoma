import numpy as np
from sklearn import metrics


# ===============
# RMSE
# ===============
def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


# ===============
# RMSLE
# ===============
def rmsle(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_log_error(y_true, y_pred))


# ===============
# MAE
# ===============
def mae(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)


# ===============
# MAPE
# ===============
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


# ===============
# AUC
# ===============
def auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)


# ===============
# Logloss
# ===============
def logloss(y_true, y_pred):
    return metrics.log_loss(y_true, y_pred)