from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import sys
import pickle 
import os

from src.exception import CustomException
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param, scoring):
    report = {}
    trained_models = {}

    for name, model in models.items():
        grid = GridSearchCV(
            model,
            param[name],
            scoring=scoring,
            cv=3,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_test_prob = best_model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_test_prob)

        report[name] = score
        trained_models[name] = best_model

    return report, trained_models

def load_object(file_path):
    try:
        # if relative path, resolve from project root
        if not os.path.isabs(file_path):
            BASE_DIR = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            file_path = os.path.join(BASE_DIR, file_path)

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
