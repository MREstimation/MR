import numpy as np

# use a trained classifier for covariate shift weights
class CovShiftClassifierWeights:
    def __init__(self, model):
        self.model = model

    def compute_weights(self, X, X_test, y=None):
        y_pred_train = self.model.predict_proba(X)[:, 1]
        w = y_pred_train / (1 - y_pred_train)
        return w / np.sum(w)
