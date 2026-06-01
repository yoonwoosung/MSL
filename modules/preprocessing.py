"""
전처리 모듈 (preprocessing.py)

핵심 책임:
  1) '식별자(identity)' 컬럼을 명시적으로 제거하여 데이터 누수(leakage)를 차단한다.
  2) 누수 없이(=train 통계만 사용해) 결측치/스케일링을 처리한다.

[왜 식별자를 지우는가]
  Source IP / Destination IP / Source/Destination Port 등은 트래픽의 '행동(behavior)'이
  아니라 '누가/어디서 보냈는가(identity)'를 나타낸다. DDoS 데이터는 한 공격 IP가
  수천 개의 row를 만들기 때문에, 같은 IP가 train/test 양쪽에 섞이면 모델은
  '트래픽이 공격처럼 생겼는가'가 아니라 '이 IP가 공격자 명단에 있는가'를 암기한다.
  이것이 혼동행렬 100% + feature importance 1등 Source IP(0.44)의 정체다.

  중요: 이 식별자들은 CSV에서 '숫자'로 저장되는 경우가 많다. 따라서
  select_dtypes(np.number)만으로는 절대 걸러지지 않는다 → 반드시 명시적으로 drop.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 학습에서 반드시 제외해야 하는 식별자/메타 컬럼.
# (CICFlowMeter 계열은 컬럼명에 앞뒤 공백이 붙는 경우가 있어, 공백 제거 후 비교한다.
#  Src/Dst 축약 표기, 대소문자 차이도 함께 흡수한다.)
ID_COLUMNS = [
    "source ip", "src ip",
    "destination ip", "dst ip", "dest ip",
    "source port", "src port",
    "destination port", "dst port", "dest port",
    "flow id",
    "timestamp",
    "unnamed: 0",
]


def _drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """식별자 컬럼을 (숫자/문자 여부와 무관하게) 명시적으로 제거한다."""
    # 컬럼명 정규화용 매핑: 소문자 + 양끝 공백 제거
    norm = {col: str(col).strip().lower() for col in df.columns}
    to_drop = [col for col, key in norm.items() if key in ID_COLUMNS]

    if to_drop:
        print(f"[누수 방지] 식별자 컬럼 제거: {to_drop}")
    else:
        print("[누수 방지] 데이터에서 식별자 컬럼을 찾지 못했습니다(이미 없거나 컬럼명이 다름).")
    return df.drop(columns=to_drop, errors="ignore")


def preprocess_data(file_path, target_col="target", test_size=0.2, random_state=42):
    """
    반환: X_train, X_test, y_train, y_test, feature_names, class_names

    처리 순서(누수 방지 관점에서 중요):
      식별자 drop  ->  범주형 인코딩  ->  수치형 선택  ->  inf/결측치 처리
      ->  레이블 인코딩  ->  train/test 분할(stratify)  ->  스케일러는 train에만 fit
    """
    print("--- 데이터 로딩 및 전처리 시작 ---")
    df = pd.read_csv(file_path)

    # 컬럼명 앞뒤 공백 제거(CICFlowMeter 호환)
    df.columns = [str(c).strip() for c in df.columns]

    if target_col not in df.columns:
        raise KeyError(
            f"타깃 컬럼 '{target_col}'을 찾을 수 없습니다. 실제 컬럼: {list(df.columns)[:20]} ..."
        )

    # 1) 정답(레이블) 정리: 공백 제거 + 문자열 통일
    df[target_col] = df[target_col].astype(str).str.strip()

    # 2) ★ 식별자 컬럼을 '먼저' 제거한다 (수치형 선택보다 앞서야 IP/Port가 새지 않는다)
    df = _drop_identifier_columns(df)

    # 3) 범주형 컬럼 인코딩 (Highest Layer, Transport Layer 등)
    #    수치형 선택 전에 처리해야 select_dtypes에서 포함된다.
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != target_col]
    if cat_cols:
        print(f"[범주형 인코딩] LabelEncoder 적용: {cat_cols}")
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # 4) 수치형 특성만 선택 (식별자를 이미 지운 뒤이므로 안전)
    feature_df = df.select_dtypes(include=[np.number]).copy()
    feature_df[target_col] = df[target_col]

    # 5) 무한대(inf) -> NaN -> 결측 행 제거
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.dropna(inplace=True)

    # 6) 레이블 인코딩 (정상/공격 등 -> 0,1,...)
    le = LabelEncoder()
    feature_df[target_col] = le.fit_transform(feature_df[target_col])
    print(f"인코딩 완료: {list(le.classes_)} -> {np.unique(feature_df[target_col])}")

    # 7) 특성/레이블 분리
    X = feature_df.drop(columns=[target_col])
    y = feature_df[target_col]
    feature_names = list(X.columns)
    print(f"학습에 사용하는 특성 수: {len(feature_names)}개")

    # 8) 분할 (재현성 random_state=42, 클래스 비율 유지 stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 9) 스케일링 — train에만 fit, test는 transform만 (누수 방지의 정석)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

    print(f"전처리 완료: 학습 {X_train.shape}, 테스트 {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names, list(le.classes_)
