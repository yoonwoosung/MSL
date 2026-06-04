"""
학습 · 평가 · 튜닝 통합 모듈 (function2.py)

원래 train.py / evaluate.py / tune.py 세 모듈을 제출 양식(function2.py)에 맞춰
하나로 통합한 파일이다. 담당 기능은 다음과 같다.

  1) 학습      : RandomForest 최종 모델 학습 및 저장 (train_rf_model, save_model)
  2) 평가/시각화: 모델 비교 + 상세 지표/그림 저장 (compare_models, evaluate_and_visualize)
  3) 튜닝      : GridSearchCV 하이퍼파라미터 탐색 (tune_random_forest)

[왜 RandomForest를 최종 모델로 쓰는가]
  - 네트워크 트래픽 특성은 스케일/분포가 제각각이고 비선형 상호작용이 많은데,
    트리 기반 앙상블은 스케일 가정 없이 이를 잘 잡는다.
  - 여러 트리를 평균(bagging)하므로 단일 트리보다 분산이 작아 과적합에 강하다.
  - feature_importances_로 '어떤 행동 특성이 공격 판단에 기여했는가'를 설명할 수 있어
    누수 진단/검증(IP가 1등인지 확인)에도 유용하다.

[왜 accuracy만 보지 않는가]
  보안 탐지는 클래스 불균형이 흔하고 '공격을 놓치는 것(FN)'과 '정상을 공격으로
  오인(FP)'의 비용이 다르다. 그래서 Recall/Precision/F1과 혼동행렬을 함께 본다.

재현성: 모든 모델·분할·튜닝에 random_state=42 고정.
"""

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경에서도 png 저장 가능
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42


# ===========================================================================
# 1) 학습 (구 train.py)
# ===========================================================================
def train_rf_model(X_train, y_train, n_estimators=100, max_depth=15):
    """최종 RandomForest 모델을 학습해 반환한다."""
    print("모델 학습 시작 (RandomForest)")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, path="final_model.pkl"):
    """학습된 모델을 규격명(final_model.pkl)으로 저장한다."""
    joblib.dump(model, path)
    print(f"모델 저장 완료: {path}")


# ===========================================================================
# 2) 평가 / 시각화 (구 evaluate.py)
# ===========================================================================
def _build_candidates():
    """비교할 모델 후보. 전부 random_state=42로 고정(재현성)."""
    return {
        # 로지스틱 회귀: '로그 손실(log loss)'을 최소화하는 선형 분류기.
        # 빠르고 해석 쉬운 베이스라인.
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        # 단일 결정트리: 비선형은 잡지만 분산이 커서 과적합 경향(비교 기준점).
        "DecisionTree": DecisionTreeClassifier(
            max_depth=15, random_state=RANDOM_STATE
        ),
        # 랜덤포레스트: 트리 앙상블(bagging)로 분산을 줄인 최종 후보.
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def compare_models(X_train, y_train, X_test, y_test, result_dir="result"):
    """여러 모델을 학습/평가하고 훈련·테스트 성능을 비교표로 저장한다."""
    os.makedirs(result_dir, exist_ok=True)
    rows = []
    for name, model in _build_candidates().items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "train_accuracy": round(train_acc, 4),
                "test_accuracy": round(test_acc, 4),
                # 과적합 신호: train과 test의 간극
                "overfit_gap": round(train_acc - test_acc, 4),
                "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            }
        )
        print(f"[{name}] train={train_acc:.4f} / test={test_acc:.4f} (gap={train_acc - test_acc:.4f})")

    comp = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    out_csv = os.path.join(result_dir, "model_comparison.csv")
    comp.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"모델 비교표 저장: {out_csv}")
    return comp


def evaluate_and_visualize(model, X_test, y_test, X_train=None, y_train=None,
                           feature_names=None, class_names=None, result_dir="result"):
    """최종 모델의 상세 지표/그림을 result/ 에 저장한다."""
    os.makedirs(result_dir, exist_ok=True)
    print("테스트 데이터 예측 및 평가 시작...")
    y_pred = model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    # 1) metrics.txt (훈련/테스트 score 함께 -> 과적합 통제 확인)
    lines = []
    if X_train is not None and y_train is not None:
        train_acc = model.score(X_train, y_train)
        lines.append(f"Train Accuracy : {train_acc:.4f}")
        lines.append(f"Test  Accuracy : {test_acc:.4f}")
        lines.append(f"Overfit gap    : {train_acc - test_acc:.4f}")
    else:
        lines.append(f"Test Accuracy  : {test_acc:.4f}")
    lines.append("")
    lines.append("Precision : %.4f" % precision_score(y_test, y_pred, average="weighted", zero_division=0))
    lines.append("Recall    : %.4f" % recall_score(y_test, y_pred, average="weighted", zero_division=0))
    lines.append("F1-score  : %.4f" % f1_score(y_test, y_pred, average="weighted", zero_division=0))
    lines.append("")
    lines.append("[classification_report]")
    lines.append(report)

    metrics_path = os.path.join(result_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"지표 저장: {metrics_path}")
    print("\n".join(lines))

    # 2) prediction.csv
    pred_df = pd.DataFrame({"y_true": np.asarray(y_test), "y_pred": y_pred})
    pred_df.to_csv(os.path.join(result_dir, "prediction.csv"), index=False, encoding="utf-8-sig")

    # 3) Confusion Matrix png
    fig, ax = plt.subplots(figsize=(8, 8))
    disp_labels = class_names if class_names is not None else None
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap=plt.cm.Oranges, display_labels=disp_labels
    )
    ax.set_title("DDoS Detection - Confusion Matrix (after leakage fix)")
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close(fig)

    # 4) Feature Importance png (트리 기반 모델일 때만)
    if hasattr(model, "feature_importances_") and feature_names is not None:
        importances = model.feature_importances_
        idx = np.argsort(importances)[-15:]
        plt.figure(figsize=(12, 8))
        plt.title("Top 15 Features for DDoS Detection (no identifier leakage)")
        plt.barh(range(len(idx)), importances[idx], color="firebrick", align="center")
        plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "feature_importance.png"))
        plt.close()

    return test_acc


# ===========================================================================
# 3) 하이퍼파라미터 튜닝 (구 tune.py)
# ===========================================================================
def tune_random_forest(X_train, y_train, X_test=None, y_test=None, cv=3):
    """GridSearchCV로 RandomForest를 튜닝하고 best 모델/파라미터를 반환한다.

      - cv 교차검증으로 '훈련 데이터 안에서' 일반화 성능을 추정 -> 과적합 통제.
      - best_params_ 와 (훈련/테스트 score)를 함께 출력해 과적합 여부를 확인한다.
    """
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, None],
        "min_samples_split": [2, 5],
        # 리프 노드 최소 샘플 수 — 과적합 방지에 직접적으로 작용
        "min_samples_leaf": [1, 5],
        # 각 분할에서 고려할 특징 수 — RF가 DT보다 강한 핵심 이유
        "max_features": ["sqrt", "log2"],
    }

    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="f1_weighted",  # 불균형 고려해 f1 기준
        cv=cv,
        # Windows/Python 3.13에서 loky(프로세스 병렬) resource_tracker 버그를 피하기 위해
        # GridSearchCV는 단일 프로세스로 실행. 내부 RandomForest는 n_jobs=-1(스레드)로
        # 각 fit이 모든 코어를 사용하므로 속도 저하는 최소이며 결과는 동일하다.
        n_jobs=1,
        verbose=1,
    )
    print("--- GridSearchCV 튜닝 시작 ---")
    grid.fit(X_train, y_train)

    print(f"Best params : {grid.best_params_}")
    print(f"Best CV f1  : {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    # 과적합 통제 확인: 훈련 vs 테스트
    train_acc = best_model.score(X_train, y_train)
    print(f"Best 모델 Train Accuracy: {train_acc:.4f}")
    if X_test is not None and y_test is not None:
        test_acc = best_model.score(X_test, y_test)
        print(f"Best 모델 Test  Accuracy: {test_acc:.4f} (gap={train_acc - test_acc:.4f})")

    return best_model, grid.best_params_
