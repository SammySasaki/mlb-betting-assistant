import numpy as np


class PlattCalibratedPipeline:
    """Preprocessor → XGBoost → Platt sigmoid calibration, saveable with joblib.

    Keeps the three fitted components together so prediction_service can load
    a single file and call predict_proba(raw_DataFrame) without knowing the
    internal structure.
    """

    def __init__(self, preprocessor, xgb_clf, platt):
        self.preprocessor = preprocessor
        self.xgb_clf      = xgb_clf
        self.platt        = platt  # LogisticRegression fit on logit(raw_prob)

    def predict_proba(self, X_raw):
        X_enc = self.preprocessor.transform(X_raw)
        raw   = self.xgb_clf.predict_proba(X_enc)[:, 1]
        logit = np.log(raw.clip(1e-9, 1 - 1e-9) / (1 - raw.clip(1e-9, 1 - 1e-9)))
        cal   = self.platt.predict_proba(logit.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - cal, cal])
