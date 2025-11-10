# 1. 표준 라이브러리
import sys
from pathlib import Path
import platform
import time
import json
from os import PathLike
import joblib
from datetime import datetime
from typing import Optional
from dataclasses import dataclass


# 2. 서드파티 라이브러리 

# 2-1. 시각화
import plotly.express as px

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns

# 2-2. 
import shap
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, make_scorer, recall_score, 
    precision_score, f1_score, fbeta_score, average_precision_score, balanced_accuracy_score, precision_recall_fscore_support
)
from scipy.stats import uniform, randint
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator

sys.path.append(str(Path.cwd().parent))
from utils import DATA_DIR, MODEL_DIR


class ModelNotExistException(Exception):
    def __init__(self, *args):
        super().__init__(*args)

@dataclass
class ClassificationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    report: str
    
    def to_dict(self):
        return self.__dict__
    

class BinaryClassifierTester:
    def __init__(self, root: PathLike):
        self.root = Path(root)
    
    def execute(
        self, model: Optional[BaseEstimator], X_test: pd.DataFrame, y_test: pd.DataFrame, 
        name: Optional[str], ext: Optional[str] = 'pkl', 
        average: str = 'binary', threshold: float = 0.5):
        if not model and name and ext:
            model = self.load_model(name, ext)
        elif model:
            pass
        else:
            msg = '모델 또는 모델 경로(이름, 확장자)를 입력해주세요.'
            raise ModelNotExistException(msg)
        
        result = self.test(model, X_test, y_test, average, threshold)
        return result
        
    
    def load_model(self, name: str, ext: str = 'pkl') -> BaseEstimator:
        file_path = self.root / f'{name}.{ext}'
        if file_path.exists():
            loaded_model = joblib.load(file_path)
        else:
            msg = '해당 경로에 모델이 존재하지 않습니다.'
            raise ModelNotExistException(msg)
        
        return loaded_model
    
    def test(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.DataFrame, average: str = 'binary', threshold: float = 0.5):
        # 테스트셋 성능 평가
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_prob, average=average)

        report = classification_report(y_test, y_pred, digits=3)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        return ClassificationResult(
            accuracy=bal_acc,
            precision=precision,
            recall=recall,
            f1 = f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            report=report
        )
        
    def show_result(self, result: ClassificationResult):
        print("\nClassification Report (Test Set):")
        print(result.report)
        print(f"\nAccuracy (Test Set): {result.accuracy:.3f}")
        print(f"ROC-AUC (Test Set): {result.roc_auc:.3f}")
        print(f"PR-AUC (Test Set): {result.pr_auc:.3f}")
        print()
        # thr_list = np.round(np.arange(0.1, 1.0, 0.01), 2)
        # rows = []
        # for thr in thr_list:
        #     y_hat = (y_pred_proba_minority >= thr).astype(int)
        #     rows.append([
        #         thr,
        #         precision_score(y_test, y_hat, zero_division=0),
        #         recall_score(y_test, y_hat, zero_division=0),
        #         f1_score(y_test, y_hat, zero_division=0)
        #     ])

        # thr_table = pd.DataFrame(rows, columns=['threshold', 'precision', 'recall', 'f1'])
        