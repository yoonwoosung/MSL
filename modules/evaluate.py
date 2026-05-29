"""
평가 모듈 (evaluate.py)

담당:
  1) 모델 2~3개 비교 (LogisticRegression / DecisionTree / RandomForest)
  2) 훈련 score와 테스트 score를 함께 출력 -> 과적합 통제 확인
  3) Precision / Recall / F1 / Confusion Matrix 계산 후 result/ 폴더에 저장
     (metrics.txt, model_comparison.csv, confusion_matrix.png, feature_importance.png)

[왜 accuracy만 보지 않는가]
  보안 탐지(공격 vs 정상)는 클래스 불균형이 흔하고, '공격을 놓치는 것(FN)'과
  '정상을 공격으로 오인(FP)'의 비용이 다르다. 그래서 Recall/Precision/F1과
  혼동행렬을 함께 본다. accuracy 한 숫자는 불균형 데이터에서 쉽게 착시를 준다.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경에서도 png 저장 가능
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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


def _build_candidates():
    """비교할 모델 후보. 전부 random_state=42로 고정(재현성)."""
    return {
        # 로지스틱 회귀: '로그 손실(log loss)'을 최소화하는 선형 분류기.
        # 즉 학습 = 비용(손실)을 최소화하는 과정. 빠르고 해석 쉬운 베이스라인.
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
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
