# DDoS 트래픽 탐지 머신러닝 프로젝트

네트워크 트래픽 데이터를 기반으로 DDoS 공격을 탐지하는 머신러닝 파이프라인입니다.
데이터 누수(Data Leakage)를 발견·제거하고, RandomForest 모델로 실제 일반화 성능을 검증하였습니다.

---

## 파일 구조

```
MSL/
├── main.py                        # 진입점: 전체 파이프라인 실행
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py           # 전처리 (식별자 drop, 범주형 인코딩, 스케일링)
│   ├── train.py                   # RandomForest 학습 및 모델 저장
│   ├── evaluate.py                # 모델 비교 및 성능 평가·시각화
│   └── tune.py                    # GridSearchCV 하이퍼파라미터 튜닝
├── data/
│   └── ddos_dataset.csv           # Kaggle DDoS 트래픽 데이터셋 (약 85만 행)
├── result/
│   ├── metrics.txt                # 정확도, Precision, Recall, F1
│   ├── confusion_matrix.png       # 혼동 행렬
│   ├── feature_importance.png     # 특성 중요도 Top 15
│   ├── model_comparison.csv       # 3개 모델 비교표
│   └── prediction.csv             # 테스트 세트 예측 결과
├── experiments/
│   └── extreme_tuning.py          # 극단적 파라미터 탐색 실험 (80조합)
├── final_model.pkl                # 학습된 최종 모델
├── requirements.txt               # 패키지 의존성
├── confusion_matrix_kaggle.png    # 누수 제거 전 혼동행렬 (비교용)
└── feature_importance_kaggle.png  # 누수 제거 전 특성 중요도 (비교용)
```

---

## 실행 방법

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 배치
Kaggle에서 데이터셋을 다운로드 후 `data/ddos_dataset.csv` 경로에 배치합니다.
- 데이터셋: https://www.kaggle.com/datasets/oktayrdeki/ddos-traffic-dataset

### 3. 실행
```bash
python3 main.py
```

### 4. 튜닝 포함 실행 (선택)
`main.py`에서 `RUN_TUNING = True`로 변경 후 실행합니다.
GridSearchCV 48조합 × 3-fold = 144 fits 수행 (시간 소요).

### 5. 결과 확인
`result/` 폴더에서 CSV, PNG, TXT 파일 확인
`final_model.pkl`에 학습된 모델 저장

---

## 핵심 내용: 데이터 누수 발견 및 제거

| 항목 | 누수 제거 전 | 누수 제거 후 |
|------|------------|------------|
| Test Accuracy | 100% (가짜) | 99.79% (진짜) |
| 특성 중요도 1위 | Source IP (0.44) | Packet Length (행동 특성) |
| 원인 | IP/Port가 숫자로 저장되어 학습에 포함 | 식별자 명시적 제거 + 범주형 인코딩 |

---

## 모델 비교 결과

| 모델 | Train Accuracy | Test Accuracy | Overfit gap | F1 |
|------|:--------------:|:-------------:|:-----------:|:---:|
| LogisticRegression | 0.9895 | 0.9895 | 0.0000 | 0.9895 |
| DecisionTree | 0.9984 | 0.9980 | 0.0004 | 0.9980 |
| **RandomForest** | **0.9984** | **0.9979** | **0.0006** | **0.9979** |

최종 모델: **RandomForest** (배깅 기반 앙상블, 과적합 방지, 특성 중요도 제공)

---

## 튜닝 결과

GridSearchCV로 5개 파라미터 48조합 탐색 및 극단적 탐색 80조합 추가 실험 수행.

- 최적 파라미터: `n_estimators=100, max_depth=10, max_features=sqrt, min_samples_leaf=1, min_samples_split=5`
- Best CV F1: **0.9981**
- 기본값 대비 성능 차이: **F1 0.0001** → 기본값이 이미 최적에 수렴

---

## 주요 변수

| 변수 | 설명 |
|------|------|
| X_train | 훈련 특성 (약 68만 행, 4개 특성) |
| X_test | 테스트 특성 (약 17만 행) |
| y_train | 훈련 레이블 (0: 정상, 1: DDoS) |
| y_test | 테스트 레이블 |
| feature_names | 학습에 사용된 특성명 리스트 |

---

## 의존성

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
joblib>=1.3
```

설치: `pip install -r requirements.txt`

---

## 재현성

모든 모듈에 `random_state=42` 통일 적용. 동일 환경에서 동일 결과 보장.
