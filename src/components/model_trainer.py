import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test arrays")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Models for GridSearch (exclude CatBoost)
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                ),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(
                    class_weight="balanced"
                ),
                "Random Forest": RandomForestClassifier(
                    class_weight="balanced"
                ),
                "XGBoost": XGBClassifier(
                    eval_metric="logloss",
                    use_label_encoder=False
                ),
                "AdaBoost": AdaBoostClassifier()
            }

            params = {
                "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
                "KNN": {"n_neighbors": [3, 5, 7, 9]},
                "Decision Tree": {
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20]
                },
                "XGBoost": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1]
                }
            }

            logging.info("Running GridSearchCV for sklearn models")

            # IMPORTANT: evaluate_models now returns (scores, trained_models)
            model_report, trained_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
                scoring="roc_auc"
            )

            # Train CatBoost separately
            logging.info("Training CatBoost separately")

            cat_model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.1,
                loss_function="Logloss",
                verbose=False
            )
            cat_model.fit(X_train, y_train)

            cat_test_prob = cat_model.predict_proba(X_test)[:, 1]
            cat_auc = roc_auc_score(y_test, cat_test_prob)

            model_report["CatBoost"] = cat_auc

            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            if best_model_score < 0.7:
                raise CustomException("No sufficiently good model found")

            if best_model_name == "CatBoost":
                best_model = cat_model
            else:
                best_model = trained_models[best_model_name]

            logging.info(
                f"Best model selected: {best_model_name} | ROC AUC: {best_model_score}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
