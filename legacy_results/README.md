# legacy_results — 누수 전 결과물 보존 아카이브

본 폴더는 **데이터 누수(leakage) 제거 *이전*의 결과물**과 발표 증거 자료를 영구 보존하기 위한 아카이브입니다.
정리일: **2026-06-04** / 브랜치: `develop` (이후 `main`에도 병합)

---

## 왜 보존하는가

본 프로젝트의 핵심 발표 메시지는 **"정확도 100%는 성능이 아니라 데이터 누수의 증거였다"** 입니다.

- 초기 모델은 테스트 정확도 **100%(1.0000)** 를 기록했습니다.
- 그러나 특성 중요도 1위가 **Source IP(식별자)** 였습니다 — 모델이 트래픽의 *행동*이 아니라
  *"이 IP가 공격자 명단에 있는가"* 를 암기한 것입니다.
- 식별자(IP/Port) 컬럼을 제거(`modules/preprocessing.py`의 `_drop_identifier_columns()`)한 뒤,
  **진짜 성능 99.79%** 를 확보했습니다.

따라서 누수 전 결과물은 발표의 **"가짜 100% → 진짜 99.79%"** 서사를 입증하는 결정적 증거이므로,
최종 정리 과정에서 삭제되지 않도록 별도 보존합니다.

---

## 폴더 구조

```
legacy_results/
├── README.md                          ← 본 문서
├── before_leakage_fix/                ← 누수 제거 전 결과물
│   ├── confusion_matrix_kaggle.png    ← 정확도 100% 혼동행렬 (대각선만 채워짐)
│   └── feature_importance_kaggle.png  ← Source IP가 1위인 특성 중요도 그래프
└── presentation_evidence/             ← 발표 PPT에 사용한 캡쳐 백업 (slide2~5)
    ├── slide2_문제정의/
    ├── slide3_EDA/
    ├── slide4_전처리/
    └── slide5_모델비교/
```

---

## 파일별 설명 및 발표 슬라이드 매핑

| 파일 | 누수 전/후 | 용도 |
|---|---|---|
| `before_leakage_fix/confusion_matrix_kaggle.png` | 누수 전 | 정확도 100% 혼동행렬 — "가짜 100%" 증거 |
| `before_leakage_fix/feature_importance_kaggle.png` | 누수 전 | Source IP 1위 — 누수 원인 증거 |
| `presentation_evidence/slide2~5/*` | 발표자료 | 문제정의 / EDA / 전처리 / 모델비교 캡쳐 백업 |

> **참고:** `kaggle_final_model.pkl`(누수 모델 파일)과 별도의 100% 정확도 로그(`.txt`)는
> 저장소에 추적되지 않습니다(`.pkl`은 `.gitignore`로 제외, 로그는 파일로 저장된 적 없음).
> 100% 정확도 자체는 위 `confusion_matrix_kaggle.png` 이미지에 그대로 담겨 있습니다.

---

## 주의

- **본 폴더는 교수님 제출 폴더(`project_1/`)에 포함하지 않습니다.** 제출본은 누수 제거 후의
  깨끗한 최종 결과물만 담습니다.
- 단, 본 아카이브는 `develop`·`main` 양쪽 git 히스토리에 영구 보존됩니다.
- 누수 *제거 후* 의 현재 결과물은 저장소 루트의 `result/` 폴더에 있습니다.
