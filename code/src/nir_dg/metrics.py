import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def score(task, y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if len(y) != len(prediction) or not np.isfinite(prediction).all():
        raise ValueError("Invalid predictions")
    if task == "regression":
        return {"rmse": float(np.sqrt(mean_squared_error(y, prediction))), "mae": float(mean_absolute_error(y, prediction)), "mse": float(mean_squared_error(y, prediction)), "r2": float(r2_score(y, prediction)) if len(y) > 1 else np.nan}
    if prediction.ndim != 1 or (prediction < 0).any() or (prediction > 1).any():
        raise ValueError("Binary classification expects probabilities in [0,1]")
    two_classes = len(np.unique(y)) == 2
    return {"roc_auc": float(roc_auc_score(y, prediction)) if two_classes else np.nan,
            "log_loss": float(log_loss(y, np.column_stack([1 - prediction, prediction]), labels=[0, 1])),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction >= .5)),
            "average_precision": float(average_precision_score(y, prediction)) if two_classes else np.nan}
