"""
학습 모듈 (train.py) — 학습만 담당한다(평가/시각화 제외).

[왜 RandomForest를 최종 모델로 쓰는가]
  - 네트워크 트래픽 특성은 스케일/분포가 제각각이고(byte 수, 시간, 비율 등),
    특성 간 비선형 상호작용이 많다. 트리 기반 앙상블은 이런 비선형/상호작용을
    스케일 가정 없이 잘 잡는다.
  - 여러 트리를 평균(bagging)하므로 단일 결정트리보다 분산이 작아 과적합에 강하다.
  - feature_importances_로 '어떤 행동 특성이 공격 판단에 기여했는가'를 설명할 수 있어,
    누수 진단/검증(IP가 1등인지 확인)에도 유용하다.
  비교용으로 evaluate.py에서 LogisticRegression / DecisionTree와 함께 평가한다.

재현성: 모든 모델·분할에 random_state=42 고정.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42


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
