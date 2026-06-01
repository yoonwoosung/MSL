"""
하이퍼파라미터 튜닝 모듈 (tune.py)

GridSearchCV로 RandomForest의 하이퍼파라미터를 탐색한다.
  - cv 교차검증으로 '훈련 데이터 안에서' 일반화 성능을 추정 -> 과적합 통제.
  - best_params_ 와 (훈련/테스트 score)를 함께 출력해 과적합 여부를 확인한다.

재현성: estimator와 split 모두 random_state=42.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42


def tune_random_forest(X_train, y_train, X_test=None, y_test=None, cv=3):
    """GridSearchCV로 RandomForest를 튜닝하고 best 모델/파라미터를 반환한다."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, None],
        "min_samples_split": [2, 5],
        # Lecture 10: 리프 노드 최소 샘플 수 — 과적합 방지에 직접적으로 작용
        "min_samples_leaf": [1, 5],
        # Lecture 10: 각 분할에서 고려할 특징 수 — RF가 DT보다 강한 핵심 이유
        "max_features": ["sqrt", "log2"],
    }

    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="f1_weighted",  # 불균형 고려해 f1 기준
        cv=cv,
        n_jobs=-1,
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
