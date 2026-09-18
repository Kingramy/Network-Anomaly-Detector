"""
Loads network traffic data for the anomaly detector.

Supports:
  1. NSL-KDD (KDDTrain+.txt / .arff / .csv, and KDDTest+ variants)
  2. generate_synthetic_data() - a fallback so the pipeline always runs.
"""

import os
from io import StringIO
import numpy as np
import pandas as pd

NSL_KDD_PATH = os.path.join(os.path.dirname(__file__), "data", "KDDTrain+.txt")
LABEL_COLUMN = "is_anomaly"
ATTACK_TYPE_COLUMN = "attack_type"

NSL_KDD_FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

NSL_KDD_COLUMNS_43 = NSL_KDD_FEATURE_COLUMNS + ["label", "difficulty"]
NSL_KDD_COLUMNS_42 = NSL_KDD_FEATURE_COLUMNS + ["class"]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}

def generate_synthetic_data(n_normal: int = 2000, n_anomalies: int = 60, seed: int = 42, is_test: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    protocols = ["tcp", "udp", "icmp"]
    
    data = pd.DataFrame({
        "duration": rng.normal(30, 8, n_normal + n_anomalies).clip(1, None),
        "protocol_type": rng.choice(protocols, n_normal + n_anomalies),
        "service": ["http"] * (n_normal + n_anomalies),
        "flag": ["SF"] * (n_normal + n_anomalies),
        "src_bytes": rng.normal(500, 120, n_normal + n_anomalies).clip(0, None),
    })
    
    for col in NSL_KDD_FEATURE_COLUMNS:
        if col not in data.columns:
            data[col] = 0
            
    labels = [0] * n_normal + [1] * n_anomalies
    
    if is_test:
        attacks = ["normal"] * n_normal + ["mscan"] * (n_anomalies // 2) + ["snmpgetattack"] * (n_anomalies - n_anomalies // 2)
    else:
        attacks = ["normal"] * n_normal + ["neptune"] * (n_anomalies // 2) + ["guess_passwd"] * (n_anomalies - n_anomalies // 2)
    
    data[LABEL_COLUMN] = labels
    data[ATTACK_TYPE_COLUMN] = attacks
    return data.sample(frac=1, random_state=seed).reset_index(drop=True)

def load_nsl_kdd(path: str) -> pd.DataFrame:
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "@data":
            data_start = i + 1
            break
    data_lines = lines[data_start:]

    raw = pd.read_csv(StringIO("".join(data_lines)), header=None)
    n_cols = raw.shape[1]
    
    if n_cols == 43:
        raw.columns = NSL_KDD_COLUMNS_43
        attack_series = raw["label"].astype(str).str.strip().str.lower()
    elif n_cols == 42:
        raw.columns = NSL_KDD_COLUMNS_42
        attack_series = raw["class"].astype(str).str.strip().str.lower()
    else:
        raise ValueError(f"Unexpected NSL-KDD column count: {n_cols} (expected 42 or 43) in {path}")

    raw[ATTACK_TYPE_COLUMN] = attack_series
    raw[LABEL_COLUMN] = (attack_series != "normal").astype(int)
    
    keep = NSL_KDD_FEATURE_COLUMNS + [LABEL_COLUMN, ATTACK_TYPE_COLUMN]
    return raw[keep]

def _encode_categoricals(df: pd.DataFrame, train_columns=None) -> pd.DataFrame:
    """One-hot encodes categorical columns and aligns test data features to match training data."""
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS)
    
    if train_columns is not None:
        for col in train_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[train_columns + [LABEL_COLUMN, ATTACK_TYPE_COLUMN]]
        
    return df_encoded

def _find_nsl_kdd_file():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    candidates = [
        "KDDTrain+.txt", "KDDTrain+.csv", "KDDTrain+.arff",
        "KDDTrain+_20Percent.txt", "KDDTrain+_20Percent.arff",
        "nsl_kdd_dataset.csv", "NSL_KDD_Train.csv", "Train.txt", "Train.arff",
    ]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return NSL_KDD_PATH

def _find_nsl_kdd_test_file():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    candidates = [
        "KDDTest+.txt", "KDDTest+.csv", "KDDTest+.arff",
        "KDDTest-21.txt", "KDDTest-21.arff",
        "nsl_kdd_test.csv", "NSL_KDD_Test.csv", "Test.txt", "Test.arff",
    ]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return None

def load_data() -> pd.DataFrame:
    found_path = _find_nsl_kdd_file()
    if os.path.exists(found_path):
        print(f"Loading real data from {found_path}")
        df = load_nsl_kdd(found_path)
    else:
        print("No dataset found in data/ — using synthetic data instead.")
        df = generate_synthetic_data()
    return _encode_categoricals(df)

def load_test_data(train_feature_columns: list):
    path = _find_nsl_kdd_test_file()
    if path is None:
        print("No test dataset found in data/ — using synthetic test data instead.")
        df = generate_synthetic_data(n_normal=1000, n_anomalies=17, seed=101, is_test=True)
        return _encode_categoricals(df, train_feature_columns)
    print(f"Loading real TEST data from {path}")
    df = load_nsl_kdd(path)
    return _encode_categoricals(df, train_feature_columns)

def get_feature_columns(df: pd.DataFrame) -> list:
    return [col for col in df.columns if col not in [LABEL_COLUMN, ATTACK_TYPE_COLUMN]]

def get_attack_category(attack_name: str) -> str:
    return ATTACK_CATEGORY_MAP.get(str(attack_name).strip().lower(), "Unknown")

if __name__ == "__main__":
    df = load_data()
    feature_cols = get_feature_columns(df)
    print(df.head())
    print(f"\nLoaded {len(df)} rows.")
    print(f"Number of Feature columns: {len(feature_cols)}")
    print(df[LABEL_COLUMN].value_counts())