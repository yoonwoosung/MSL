import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

def train_rf_model(X_train, y_train):
    print("모델 학습 시작")
    # n_jobs=-1로 모든 CPU 코어를 사용합니다.
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_and_visualize(model, X_test, y_test):
    print("테스트 데이터 예측 및 평가 시작...")
    y_pred = model.predict(X_test)
    
    print(f"최종 Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\n[상세 보고서]\n", classification_report(y_test, y_pred))
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap=plt.cm.Oranges)
    ax.set_title('Kaggle DDoS Dataset Detection Result')
    plt.savefig('confusion_matrix_kaggle.png')
    plt.show()

    # 2. Feature Importance (상위 15개)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    
    plt.figure(figsize=(12, 8))
    plt.title('Kaggle Data: Top 15 Crucial Features for DDoS Detection')
    plt.barh(range(len(indices)), importances[indices], color='firebrick', align='center')
    plt.yticks(range(len(indices)), [X_test.columns[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance_kaggle.png')
    plt.show()