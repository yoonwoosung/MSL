# 발표 캡쳐 가이드

> 슬라이드 순서대로 필요한 캡쳐 항목, 파일 위치, 실행 방법 정리

---

## 슬라이드 1 — 표지

캡쳐 없음. PPT에 직접 작성.

---

## 슬라이드 2 — 문제 정의 & 전공 연계성

**① main.py 파이프라인 흐름 (코드 캡쳐)**
- `main.py` 상단 docstring + `TARGET_COL = "target"` 부분 IDE에서 열고 캡쳐

---

## 슬라이드 3 — 데이터 & EDA

**① 데이터셋 컬럼 / 크기 / 클래스 분포 (터미널 캡쳐)**
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/ddos_dataset.csv')
print('컬럼:', df.columns.tolist())
print('크기:', df.shape)
print(df['target'].value_counts())
"
```

**② ID_COLUMNS 리스트 (코드 캡쳐)**
- `modules/preprocessing.py` 27~35번째 줄 IDE에서 열고 캡쳐

---

## 슬라이드 4 — 전처리 & 데이터 누수 방지

**① 식별자 drop 함수 (코드 캡쳐)**
- `modules/preprocessing.py` `_drop_identifier_columns()` 함수 전체 캡쳐

**② 범주형 인코딩 추가 부분 (코드 캡쳐)**
- `modules/preprocessing.py` 3번 단계 주석 ~ LabelEncoder 적용 부분 캡쳐

**③ 스케일러 누수 방지 (코드 캡쳐)**
- `modules/preprocessing.py` 하단
- `scaler.fit_transform(X_train)` / `scaler.transform(X_test)` 두 줄 캡쳐

**④ 전처리 실행 로그 (터미널 캡쳐)**
```bash
python3 main.py
```
아래 출력 부분만 캡쳐:
```
[누수 방지] 식별자 컬럼 제거: ['Source IP', 'Dest IP', 'Source Port', 'Dest Port']
[범주형 인코딩] LabelEncoder 적용: ['Highest Layer', 'Transport Layer']
학습에 사용하는 특성 수: 4개
전처리 완료: 학습 (682068, 4), 테스트 (170517, 4)
```

---

## 슬라이드 5 — 모델 선택 이유 & 비교

**① 모델 후보 코드 (코드 캡쳐)**
- `modules/evaluate.py` `_build_candidates()` 함수 전체 캡쳐

**② 모델 비교 실행 로그 (터미널 캡쳐)**
`main.py` 실행 후 아래 부분 캡쳐:
```
[LogisticRegression] train=0.9895 / test=0.9895 (gap=-0.0000)
[DecisionTree]       train=0.9984 / test=0.9980 (gap=0.0004)
[RandomForest]       train=0.9984 / test=0.9979 (gap=0.0006)
```

**③ 모델 비교표 (터미널 캡쳐)**
```bash
python3 -c "import pandas as pd; print(pd.read_csv('result/model_comparison.csv').to_string(index=False))"
```

---

## 슬라이드 6 — 튜닝 & 과적합 통제

**① tune.py param_grid (코드 캡쳐)**
- `modules/tune.py` 19~29번째 줄 IDE에서 열고 캡쳐

**② 튜닝 실행 결과 (터미널 캡쳐)**
`main.py`에서 `RUN_TUNING = True` 확인 후:
```bash
rm -f final_model.pkl && python3 main.py
```
아래 출력 부분 캡쳐:
```
Fitting 3 folds for each of 48 candidates, totalling 144 fits
Best params : {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
Best CV f1  : 0.9981
Best 모델 Train Accuracy: 0.9984
Best 모델 Test  Accuracy: 0.9979 (gap=0.0005)
```

**③ 극단적 탐색 실행 결과 (터미널 캡쳐)**
```bash
python3 experiments/extreme_tuning.py
```
아래 출력 부분 캡쳐:
```
탐색 조합 수: 80 × 3-fold = 240 fits
[최적 파라미터] {'max_depth': 10, 'min_samples_split': 20, 'n_estimators': 10}
[기본값 Test F1] 0.9979
[튜닝 후 Test F1] 0.9980
[차이] 0.0001
[최악 조합] n_estimators=10, max_depth=2 / CV F1: 0.9632
```

**④ Overfit gap (파일 캡쳐)**
- `result/metrics.txt` 상단 3줄 캡쳐:
```
Train Accuracy : 0.9984
Test  Accuracy : 0.9979
Overfit gap    : 0.0005
```

---

## 슬라이드 7 — 평가 결과 (누수 수정 후)

**① 혼동행렬 Before / After (이미지 2장 나란히 PPT 배치)**
- Before: 루트의 `confusion_matrix_kaggle.png`
- After: `result/confusion_matrix.png`

**② Feature Importance Before / After (이미지 2장 나란히 PPT 배치)**
- Before: 루트의 `feature_importance_kaggle.png` — Source IP가 1위
- After: `result/feature_importance.png` — 행동 특성이 1위

**③ 최종 성능 지표 (파일 캡쳐)**
- `result/metrics.txt` 전체 내용 캡쳐

---

## 슬라이드 8 — 재현성 & 마무리

**① random_state=42 통일 (코드 캡쳐)**
- `modules/tune.py` 14번째 줄: `RANDOM_STATE = 42`
- `modules/evaluate.py` 36번째 줄: `RANDOM_STATE = 42`
- 두 줄 나란히 보이도록 IDE에서 캡쳐

**② 모델 저장 완료 로그 (터미널 캡쳐)**
`main.py` 실행 후 아래 출력 캡쳐:
```
모델 저장 완료: final_model.pkl
```

**③ 프로젝트 폴더 구조 (터미널 캡쳐)**
```bash
find . -not -path '*/data/*' -not -path '*/__pycache__/*' -not -name '*.pkl' -not -path '*/.git/*' | sort
```
