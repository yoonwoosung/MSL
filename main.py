import os
import joblib
from modules.function1 import preprocess_data
from modules.function2 import train_rf_model, evaluate_and_visualize

# 확인된 파일명과 경로
DATA_FILE = 'data/ddos_dataset.csv' 
MODEL_SAVE_PATH = 'kaggle_final_model.pkl'

def main():
    if not os.path.exists(DATA_FILE):
        print(f"에러: {DATA_FILE} 파일을 찾을 수 없습니다.")
        return

    # 1. 전처리
    X_train, X_test, y_train, y_test = preprocess_data(DATA_FILE)

    # 2. 학습 또는 로드
    if os.path.exists(MODEL_SAVE_PATH):
        print("\n학습된 모델을 불러옵니다.")
        model = joblib.load(MODEL_SAVE_PATH)
    else:
        print("\n새로운 모델 학습을 시작합니다. (멀티코어 활용)")
        model = train_rf_model(X_train, y_train)
        joblib.dump(model, MODEL_SAVE_PATH)
        print(f"학습 완료 및 '{MODEL_SAVE_PATH}' 저장 성공.")

    # 3. 결과 확인
    evaluate_and_visualize(model, X_test, y_test)

if __name__ == "__main__":
    main()