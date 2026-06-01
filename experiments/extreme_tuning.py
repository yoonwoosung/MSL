"""
극단적 파라미터 탐색 실험 (experiments/extreme_tuning.py)

목적:
  GridSearchCV 탐색 범위를 극단적으로 확장하여
  "기본값이 최적임을 검증"하는 발표 근거를 확보한다.

기존 파일은 수정하지 않고 preprocessing.py를 import해서 재사용한다.
결과는 experiments/extreme_tuning_result.csv 에 저장된다.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

from modules.preprocessing import preprocess_data

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "ddos_dataset.csv")
RESULT_FILE = os.path.join(os.path.dirname(__file__), "extreme_tuning_result.csv")
RANDOM_STATE = 42

param_grid = {
    "n_estimators":      [10, 50, 100, 200],
    "max_depth":         [2, 5, 10, 20, None],
    "min_samples_split": [2, 5, 10, 20],
}

def main():
    print("=== 극단적 파라미터 탐색 실험 ===")
    print(f"탐색 조합 수: {4 * 5 * 4} × 3-fold = {4 * 5 * 4 * 3} fits\n")

    X_train, X_test, y_train, y_test, _, _ = preprocess_data(DATA_FILE)

    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)

    # 전체 결과 저장
    results = pd.DataFrame(grid.cv_results_)
    cols = [
        "param_n_estimators", "param_max_depth", "param_min_samples_split",
        "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score",
    ]
    results[cols].sort_values("rank_test_score").to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n전체 결과 저장: {RESULT_FILE}")

    # 최적 파라미터
    best = grid.best_params_
    best_cv_f1 = grid.best_score_
    best_model = grid.best_estimator_
    test_f1 = f1_score(y_test, best_model.predict(X_test), average="weighted")

    print(f"\n[최적 파라미터] {best}")
    print(f"Best CV F1  : {best_cv_f1:.4f}")
    print(f"Test F1     : {test_f1:.4f}")

    # 기본값과 비교
    default = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    default.fit(X_train, y_train)
    default_f1 = f1_score(y_test, default.predict(X_test), average="weighted")
    print(f"\n[기본값 Test F1] {default_f1:.4f}")
    print(f"[튜닝 후 Test F1] {test_f1:.4f}")
    print(f"[차이] {abs(test_f1 - default_f1):.4f}")

    # 극단 파라미터 성능 확인 (최악 조합)
    worst = results.sort_values("mean_test_score").iloc[0]
    print(f"\n[최악 조합]")
    print(f"  n_estimators={worst['param_n_estimators']}, "
          f"max_depth={worst['param_max_depth']}, "
          f"min_samples_split={worst['param_min_samples_split']}")
    print(f"  CV F1: {worst['mean_test_score']:.4f}")

if __name__ == "__main__":
    main()
