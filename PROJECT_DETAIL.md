# MSL 프로젝트 상세 문서

> DDoS 탐지 머신러닝 파이프라인 — 데이터 누수 제거 및 모듈 구조 기반 구현

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 네트워크 트래픽 데이터 기반 DDoS 공격 탐지 |
| **핵심 문제** | 초기 구현의 데이터 누수(Data Leakage) 발견 및 해결 |
| **모델** | RandomForestClassifier (LogisticRegression, DecisionTree와 비교) |
| **핵심 메시지** | "100%는 누수의 증거, 진짜 성능은 95~99%" |

---

## 2. 디렉토리 구조

```
MSL/
├── main.py                        # 진입점: 전체 파이프라인 조율
├── modules/
│   ├── __init__.py                # 모듈 export
│   ├── preprocessing.py           # 데이터 전처리 + 누수 차단 (핵심)
│   ├── train.py                   # 모델 학습
│   ├── evaluate.py                # 평가 및 시각화
│   └── tune.py                    # 하이퍼파라미터 튜닝 (GridSearchCV)
├── requirements.txt               # 패키지 의존성
├── docs/
│   └── 발표대본.md                # 프로젝트 발표 가이드
├── confusion_matrix_kaggle.png    # 누수 제거 전 혼동행렬 (비교용)
├── feature_importance_kaggle.png  # 누수 제거 전 특성 중요도 (비교용)
└── result/                        # 실행 후 생성되는 출력 폴더
    ├── model_comparison.csv
    ├── metrics.txt
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── prediction.csv
```

---

## 3. 실행 흐름 (main.py)

```
데이터 로드 (data/ddos_dataset.csv)
  ↓
[preprocessing.py] preprocess_data()
  ├─ 식별자 컬럼 명시적 drop (IP, Port, Flow ID, Timestamp)
  ├─ 수치형 특성 선택
  ├─ inf / NaN 제거
  ├─ 레이블 인코딩
  ├─ train/test 분할 (80:20, stratify=y)
  └─ StandardScaler — train에만 fit, test는 transform만
  ↓
[evaluate.py] compare_models()
  ├─ LogisticRegression
  ├─ DecisionTree
  └─ RandomForest
  → result/model_comparison.csv 저장
  ↓
[train.py] train_rf_model()  또는  [tune.py] tune_random_forest()
  └─ final_model.pkl 저장
  ↓
[evaluate.py] evaluate_and_visualize()
  ├─ result/metrics.txt
  ├─ result/confusion_matrix.png
  ├─ result/feature_importance.png
  └─ result/prediction.csv
```

---

## 4. 주요 설정 변수 (main.py)

```python
DATA_FILE       = "data/ddos_dataset.csv"   # 입력 데이터 경로
TARGET_COL      = "target"                   # 레이블 컬럼명
MODEL_SAVE_PATH = "final_model.pkl"          # 모델 저장 경로
RUN_TUNING      = False                      # GridSearchCV 실행 여부
RESULT_DIR      = "result"                   # 출력 폴더
RANDOM_STATE    = 42                         # 전 모듈 공통 시드 (재현성)
```

---

## 5. 모듈별 상세 설명

### 5.1 preprocessing.py — 데이터 전처리 (핵심)

**문제**: 초기 구현에서 `select_dtypes(np.number)`로 모든 숫자형 컬럼을 특성으로 사용했는데, IP/Port가 숫자로 저장되어 있어 모델이 트래픽 행동이 아닌 공격자 IP를 암기 → 테스트 정확도 100%(가짜)

**해결**: 식별자 컬럼을 명시적으로 drop 후 수치형 특성 선택

```python
ID_COLUMNS = [
    "source ip", "src ip", "destination ip", "dst ip", "dest ip",
    "source port", "src port", "destination port", "dst port", "dest port",
    "flow id", "timestamp", "unnamed: 0"
]
```

**처리 순서** (순서가 중요함):
1. 컬럼명 정규화 (소문자 + 공백 제거)
2. 식별자 drop
3. 수치형 선택
4. inf 클리핑 → NaN 제거
5. 레이블 인코딩
6. train/test 분할 (`stratify=y`)
7. `scaler.fit_transform(X_train)` / `scaler.transform(X_test)`

**반환**: `(X_train, X_test, y_train, y_test, feature_names, class_names)`

---

### 5.2 train.py — 모델 학습

**선택 모델**: `RandomForestClassifier`

**선택 이유**:
- 네트워크 트래픽의 비선형 패턴 포착
- 특성 스케일에 무관 (byte, 시간, 비율이 혼재해도 무관)
- `feature_importances_`로 설명 가능
- Bagging으로 과적합 방지

```python
def train_rf_model(X_train, y_train, n_estimators=100, max_depth=15):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, path="final_model.pkl"):
    joblib.dump(model, path)
```

---

### 5.3 evaluate.py — 평가 및 시각화

#### compare_models()
3개 모델을 같은 데이터로 비교 후 CSV 저장

| 모델 | 특징 |
|------|------|
| LogisticRegression | 선형 베이스라인, 빠름 |
| DecisionTree | 비선형 포착, 과적합 경향 |
| RandomForest | 최적 후보, 분산 감소 |

**출력 지표**: `train_accuracy`, `test_accuracy`, `overfit_gap`, `precision`, `recall`, `f1`

#### evaluate_and_visualize()
최종 모델 상세 평가 및 시각화

**출력 파일**:
- `metrics.txt` — 훈련/테스트 정확도, precision/recall/F1
- `confusion_matrix.png` — 혼동행렬 히트맵
- `feature_importance.png` — Top 15 특성 중요도 바차트
- `prediction.csv` — 테스트 세트 예측값

**지표 선택 이유**:
- Recall (공격 감지율): 공격을 놓치는 것이 치명적
- Precision (오탐율): 정상을 공격으로 오인하는 비용
- F1: 위 둘의 조화평균
- Accuracy만 보는 것은 클래스 불균형 데이터에서 위험

---

### 5.4 tune.py — 하이퍼파라미터 튜닝

```python
param_grid = {
    "n_estimators":    [100, 200],
    "max_depth":       [10, 15, None],
    "min_samples_split": [2, 5],
}
# cv=3, scoring="f1_weighted"
```

- 교차검증(cv=3)으로 훈련 데이터 내에서 일반화 성능 추정
- `f1_weighted`를 기준 지표로 사용 (클래스 불균형 고려)
- 최적 파라미터로 재학습 후 저장

---

## 6. 데이터 누수 비교

| 항목 | 누수 제거 전 | 누수 제거 후 |
|------|------------|------------|
| Test Accuracy | 100% (가짜) | 95~99% (진짜) |
| 1위 특성 | Source IP (importance ≈ 0.44) | Flow Duration, Packet/Byte Rate |
| 모델 행동 | 공격자 IP 암기 | 트래픽 행동 패턴 학습 |
| 원인 | 수치형 컬럼 전체 포함 | 식별자 명시적 제거 |

비교 증거 파일:
- `confusion_matrix_kaggle.png` — 누수 제거 전
- `feature_importance_kaggle.png` — 누수 제거 전 (Source IP 1위)

---

## 7. 재현성 보장

모든 모듈에 `random_state=42` 동일 적용:

```python
train_test_split(..., random_state=42)
RandomForestClassifier(random_state=42)
LogisticRegression(random_state=42)
DecisionTreeClassifier(random_state=42)
GridSearchCV(..., cv=3)  # cv 자체는 시드 없이도 결정론적
```

---

## 8. 패키지 의존성

```
pandas>=2.0        # 데이터 처리
numpy>=1.24        # 수치 연산
scikit-learn>=1.3  # ML 모델 & 평가
matplotlib>=3.7    # 시각화
joblib>=1.3        # 모델 저장/로드
```

설치: `pip install -r requirements.txt`

---

## 9. 모듈 간 의존관계

```
main.py
  ├─ modules.preprocessing  → preprocess_data()
  ├─ modules.evaluate       → compare_models()
  ├─ modules.train          → train_rf_model(), save_model()
  ├─ modules.tune           → tune_random_forest()  (RUN_TUNING=True 시)
  └─ modules.evaluate       → evaluate_and_visualize()
```

**책임 분리 원칙**:
- `preprocessing.py` — 데이터 정제만
- `train.py` — 학습만
- `evaluate.py` — 평가/시각화만
- `tune.py` — 튜닝만

---

## 10. Git 히스토리

```
9695124  데이터 누수 제거 및 모듈 구조 개편   ← 최신 (누수 수정)
090ee3d  머신러닝 프로젝트 코드 업로드
56a355c  Merge branch 'main'
dd0bdf5  ml_ddos data .pkl파일 제외
aeaacfd  Update README.md
```

---

## 11. 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 데이터 배치
#    data/ddos_dataset.csv 위치에 데이터 파일 준비

# 3. 실행 (튜닝 없이)
python main.py

# 4. 튜닝 포함 실행
#    main.py에서 RUN_TUNING = True 로 변경 후 실행

# 5. 결과 확인
#    result/ 폴더에서 CSV, PNG, TXT 확인
#    final_model.pkl 로 학습된 모델 저장됨
```
