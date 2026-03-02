"""
preprocessor.py -- Data Cleaning, Encoding, Balancing & Scaling
===============================================================
Everything that transforms raw CSV rows into clean tensors lives here.
# kol 7aga bte7awel el CSV el kham le tensors nadeefa mawgooda hena

Design decisions:
  - We downsample the majority class to fix the insane 98:2 imbalance.
    (You *could* use SMOTE or class weights, but downsampling is simpler
    and keeps the experiment focused on the NN side.)
    # bne-downsample el majority class 3ashan ne7el moshkelet el 98:2 imbalance.
    # momken nesta5dem SMOTE aw class weights, bs el downsampling absat.

  - StandardScaler is fitted ONLY on the training split to prevent
    data leakage -- a common mistake that inflates test metrics.
    # el StandardScaler byet-fit 3al training BASS 3ashan nemna3
    # el data leakage -- 8alta sha2e3a btekhali el test metrics te3la men 8er 7a2.

  - All random operations use a fixed seed for reproducibility.
    # kol el 3amaliyyat el 3ashwa2eyya btesta5dem seed sabit 3ashan el nata2eg tetkrrar.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.utils          import resample


# ------------------------------------------------------------------
# 1. Downsample majority class to balance the dataset
# nenzil el majority class 3ashan newazzen el data
# ------------------------------------------------------------------
def downsample(df, random_state=42):
    """
    Brings the majority class (no failure) down to the same count
    as the minority class (failure).  This is brute-force but effective.
    # benenzel el majority class (mafesh failure) le nafs 3adad el minority class.
    # tare2a khashna bs sha8ala.

    After this, the model can't cheat by always predicting '0'.
    # ba3d keda el model mye2darsh ye8eshsh w ye-predict '0' 3ala tool.
    """
    majority = df[df["Machine failure"] == 0]
    minority = df[df["Machine failure"] == 1]

    majority_downsampled = resample(
        majority,
        replace=False,
        n_samples=len(minority),
        random_state=random_state,
    )

    df_balanced = pd.concat([majority_downsampled, minority])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"[preprocessor] Original size  : {len(df):,}")
    print(f"[preprocessor] Balanced size   : {len(df_balanced):,}")
    print(f"[preprocessor] Class counts:\n{df_balanced['Machine failure'].value_counts().to_string()}")

    return df_balanced


# ------------------------------------------------------------------
# 2. Encode categoricals + drop useless columns
# 7awel el categoricals le ar2am w esheel el columns elly malahash lazma
# ------------------------------------------------------------------
def encode_features(df):
    """
    - Drops 'id' and 'Product ID' (identifiers, not features).
    - Label-encodes any remaining categorical columns (just 'Type' in this dataset).
    Returns a clean, fully numeric DataFrame.
    # - biyesheel 'id' w 'Product ID' (identifiers, mesh features).
    # - biye3mel label encoding le ay categorical columns (hena 'Type' bass).
    # beyregga3 DataFrame ra2ami 100%.
    """
    df = df.copy()

    # These columns carry no predictive value -- they're just row identifiers
    # el columns di malahash ay 2eema tanabo2eyya -- mograrad identifiers
    cols_to_drop = ["id", "Product ID"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Encode anything that isn't already a number
    # 7awel ay 7aga mosh ra2am le ra2am
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"[preprocessor] Encoded '{col}': {list(le.classes_)}")

    return df, label_encoders


# ------------------------------------------------------------------
# 3. Split into X/y, then train/val, then scale
# e2sem le features w target, ba3den train w val, ba3den scale
# ------------------------------------------------------------------
def prepare_data(df, test_size=0.2, random_state=42):
    """
    Full pipeline in one call:
      1. Separate features (X) and target (y)
      2. Train/validation split (80/20 by default)
      3. StandardScaler fitted on train, applied to both
    # el pipeline kaml f call wa7da:
    #   1. efsel el features (X) 3an el target (y)
    #   2. e2sem le train w val (80/20 default)
    #   3. StandardScaler yet-fit 3al train, w yet-apply 3al etnen

    Returns numpy arrays ready for PyTorch tensors.
    # beyregga3 numpy arrays gehza lel PyTorch tensors.
    """
    target_col = "Machine failure"
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler on training data ONLY -- this is crucial to avoid leakage
    # el scaler yet-fit 3al training data BASS -- da mohimm awy 3ashan nemna3 leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    print(f"[preprocessor] Features shape : {X_train.shape[1]}")
    print(f"[preprocessor] Train samples  : {len(X_train)}")
    print(f"[preprocessor] Val samples    : {len(X_val)}")
    print(f"[preprocessor] Train pos rate : {y_train.mean():.2%}")
    print(f"[preprocessor] Val pos rate   : {y_val.mean():.2%}")

    return X_train, X_val, y_train, y_val, scaler