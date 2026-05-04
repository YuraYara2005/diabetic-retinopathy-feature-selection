import warnings
# This stops Intel from spamming your terminal with the yellow Threading warnings
warnings.filterwarnings("ignore", message=".*'Threading' parallel backend is not supported.*")

from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier  # <-- FIXED: Changed to Classifier!
from sklearn.metrics import accuracy_score

def apply_feature_mask(X, mask):
    mask = np.array(mask)
    if np.sum(mask) == 0:
        return None
    return X[:, mask == 1]

class BaseModel(ABC):
    @abstractmethod
    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        pass

class SVMModel(BaseModel):
    def __init__(self):
        self.model = SVC(kernel='linear', C=1, cache_size=1000)

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        if X_train is None or X_val is None or X_train.shape[1] == 0:
            return 0
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_val)
        return accuracy_score(y_val, preds)

class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,  # Uses all cores for Random Forest natively!
            random_state=42
        )

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        if X_train is None or X_val is None or X_train.shape[1] == 0:
            return 0
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_val)
        return accuracy_score(y_val, preds)

class KNNModel(BaseModel):
    def __init__(self):
        # Utilizing the Classifier (Intel handles this MUCH faster)
        self.model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        if X_train is None or X_val is None or X_train.shape[1] == 0:
            return 0
        self.model.fit(X_train, y_train)
        # No more rounding needed!
        preds = self.model.predict(X_val)
        return accuracy_score(y_val, preds)

def get_model(name):
    name = name.lower()
    if name == "svm":
        return SVMModel()
    elif name in ["rf", "random_forest"]:
        return RandomForestModel()
    elif name == "knn":
        return KNNModel()
    else:
        raise ValueError(f"Unknown model: {name}")