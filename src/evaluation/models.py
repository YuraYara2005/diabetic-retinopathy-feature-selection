import numpy as np
from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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
        # تم إضافة cache_size لتسريع المعالجة في الأبعاد العالية
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
            n_jobs=-1,
            random_state=42  # لضمان استقرار نتائج الفيتنس
        )

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        if X_train is None or X_val is None or X_train.shape[1] == 0:
            return 0
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_val)
        return accuracy_score(y_val, preds)


class KNNModel(BaseModel):
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=5)

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        if X_train is None or X_val is None or X_train.shape[1] == 0:
            return 0
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_val)
        return accuracy_score(y_val, preds)


def get_model(name):
    name = name.upper()
    if name == "SVM":
        return SVMModel()
    elif name in ["RF", "RANDOM_FOREST"]:
        return RandomForestModel()
    elif name == "KNN":
        return KNNModel()
    else:
        raise ValueError("Unknown model")


# import numpy as np
# from abc import ABC, abstractmethod
#
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
#
#
# def apply_feature_mask(X, mask):
#     mask = np.array(mask)
#
#     if np.sum(mask) == 0:
#         return None
#
#     return X[:, mask == 1]
#
#
# class BaseModel(ABC):
#
#     @abstractmethod
#     def train_and_evaluate(self, X_train, y_train, X_val, y_val):
#         pass
#
#
# class SVMModel(BaseModel):
#     def __init__(self):
#         self.model = SVC(kernel='linear', C=1)
#
#     def train_and_evaluate(self, X_train, y_train, X_val, y_val):
#         if X_train is None or X_val is None:
#             return 0
#
#         self.model.fit(X_train, y_train)
#         preds = self.model.predict(X_val)
#
#         return accuracy_score(y_val, preds)
#
#
# class RandomForestModel(BaseModel):
#     def __init__(self):
#         self.model = RandomForestClassifier(
#             n_estimators=50,
#             max_depth=10,
#             n_jobs=-1
#         )
#
#     def train_and_evaluate(self, X_train, y_train, X_val, y_val):
#         if X_train is None or X_val is None:
#             return 0
#
#         self.model.fit(X_train, y_train)
#         preds = self.model.predict(X_val)
#
#         return accuracy_score(y_val, preds)
#
#
# class KNNModel(BaseModel):
#     def __init__(self):
#         self.model = KNeighborsClassifier(n_neighbors=5)
#
#     def train_and_evaluate(self, X_train, y_train, X_val, y_val):
#         if X_train is None or X_val is None:
#             return 0
#
#         self.model.fit(X_train, y_train)
#         preds = self.model.predict(X_val)
#
#         return accuracy_score(y_val, preds)
#
#
# def get_model(name):
#     name = name.upper()
#
#     if name == "SVM":
#         return SVMModel()
#     elif name in ["RF", "RANDOM_FOREST"]:
#         return RandomForestModel()
#     elif name == "KNN":
#         return KNNModel()
#     else:
#         raise ValueError("Unknown model")
