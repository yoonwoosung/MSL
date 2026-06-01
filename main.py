"""
DDoS 탐지 파이프라인 진입점 (main.py)

흐름: 전처리(누수 차단) -> 모델 비교 -> (선택)튜닝 -> 최종 학습/저장 -> 평가/시각화
모든 단계 random_state=42 고정으로 재현 가능. 결과물은 result/ 폴더에 저장된다.
"""

import os
import joblib

from modules.preprocessing import preprocess_data
from modules.train import train_rf_model, save_model
from modules.evaluate import compare_models, evaluate_and_visualize
from modules.tune import tune_random_forest

# 실제 CSV 경로와 타깃 컬럼명 (코드/데이터 확인 결과 타깃 컬럼명은 'target')
DATA_FILE = "data/ddos_dataset.csv"
TARGET_COL = "target"
MODEL_SAVE_PATH = "final_model.pkl"
RESULT_DIR = "result"

# 튜닝(GridSearchCV)을 실행할지 여부. 데이터가 크면 시간이 오래 걸려 기본 False.
RUN_TUNING = True


def main():
    if not os.path.exists(DATA_FILE):
        print(f"에러: {DATA_FILE} 파일을 찾을 수 없습니다.")
        return

    # 1) 전처리 (식별자 drop으로 누수 차단 + train에만 scaler fit)
    X_train, X_test, y_train, y_test, feature_names, class_names = preprocess_data(
        DATA_FILE, target_col=TARGET_COL
    )

    # 2) 모델 비교 (LogisticRegression / DecisionTree / RandomForest)
    print("\n=== 모델 비교 ===")
    compare_models(X_train, y_train, X_test, y_test, result_dir=RESULT_DIR)

    # 3) 최종 모델 학습 또는 로드
    if os.path.exists(MODEL_SAVE_PATH):
        print("\n학습된 모델을 불러옵니다.")
        model = joblib.load(MODEL_SAVE_PATH)
    elif RUN_TUNING:
        print("\n=== GridSearchCV 튜닝 후 최종 모델 학습 ===")
        model, best_params = tune_random_forest(X_train, y_train, X_test, y_test)
        save_model(model, MODEL_SAVE_PATH)
    else:
        print("\n새로운 모델 학습을 시작합니다.")
        model = train_rf_model(X_train, y_train)
        save_model(model, MODEL_SAVE_PATH)

    # 4) 평가 및 시각화 (result/ 폴더에 저장)
    print("\n=== 최종 평가 ===")
    evaluate_and_visualize(
        model, X_test, y_test,
        X_train=X_train, y_train=y_train,
        feature_names=feature_names, class_names=class_names,
        result_dir=RESULT_DIR,
    )


if __name__ == "__main__":
    main()
