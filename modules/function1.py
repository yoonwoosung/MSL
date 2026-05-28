import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(file_path):
    print(f"--- 데이터 로딩 및 전처리 시작 ---")
    # 전체 데이터 로드 (Ryzen 7900 + 64GB RAM 활용)
    df = pd.read_csv(file_path)
    
    # 확인된 컬럼명 'target' 사용
    target_col = 'target'
    
    # 1. 정답 데이터 정리 (공백 제거 및 대소문자 통합)
    df[target_col] = df[target_col].astype(str).str.strip()
    
    # 2. 숫자형 데이터만 추출 (Source IP, Dest IP 등 문자열 제외)
    # 머신러닝 모델은 수치 데이터만 학습 가능합니다.
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df[target_col] = df[target_col]

    # 3. 결측치 및 무한대(inf) 처리
    numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_df.dropna(inplace=True)

    # 4. 레이블 인코딩 (정상/공격을 0과 1로 변환)
    le = LabelEncoder()
    numeric_df[target_col] = le.fit_transform(numeric_df[target_col])
    print(f"인코딩 완료: {le.classes_} -> {np.unique(numeric_df[target_col])}")

    # 5. 특성과 레이블 분리
    X = numeric_df.drop(target_col, axis=1)
    y = numeric_df[target_col]
    
    # 6. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)

    print(f"전처리 완료: 학습 데이터 {X_train_final.shape}, 테스트 데이터 {X_test_final.shape}")
    return X_train_final, X_test_final, y_train, y_test