#!/usr/bin/env python
"""
prepare_tls_shards.py  –  run once to build demo shards (5 clients)

• parses first 5 *_ground_truth.csv in flows_anonymized/
• fits ONE global OneHotEncoder
• stores:
      tls_shards/X_000.npz … X_004.npz   (CSR feature matrices)
      tls_shards/y_000.npy … y_004.npy   (int-coded labels)
      tls_encoder.joblib                 (fitted encoder)
      tls_labels.npy                     (array of class names)
"""

import glob
import pathlib
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

# ── paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent
CSV_DIR    = BASE_DIR / "flows_anonymized"
OUT_DIR    = BASE_DIR / "tls_shards"
ENC_PATH   = BASE_DIR / "tls_encoder.joblib"
LABEL_PATH = BASE_DIR / "tls_labels.npy"

N_CLIENTS = 5            # we’ll generate exactly 5 shards/clients
K_CIPHER, K_GROUP, K_EXT, K_EXLEN = 8, 8, 100, 100
KEEP_COLS = [
    "TLS Client Version", "Client Cipher Suites", "TLS Extension Types",
    "TLS Extension Lengths", "TLS Elliptic Curves", "Ground Truth OS",
]

# ── small helpers ────────────────────────────────────────────────────────
def parse_hex_list(s: str):
    if pd.isna(s):
        return []
    clean = "".join(c for c in s if c in "0123456789abcdefABCDEF")
    return [clean[i:i + 4].lower() for i in range(0, len(clean), 4)]

def to_slots(series, k, pfx):
    return pd.DataFrame(
        series.apply(lambda lst: (lst + ["MISSING"] * k)[:k]).to_list(),
        columns=[f"{pfx}_{i}" for i in range(k)],
    )

# ── load a small subset of data ──────────────────────────────────────────
file_list = sorted(CSV_DIR.glob("*_ground_truth.csv"))[:5]
if len(file_list) < 5:
    raise RuntimeError("Fewer than 5 CSVs found – check flows_anonymized/")

print(f"📥  Reading {len(file_list)} CSVs …")
df = pd.concat(
    [pd.read_csv(p, usecols=KEEP_COLS, low_memory=False) for p in file_list]
).dropna().reset_index(drop=True)
print(f"   loaded {len(df):,} rows")

# ── feature engineering ---------------------------------------------------
print("🧹  Parsing hex lists …")
df["cipher_list"] = df["Client Cipher Suites"].apply(parse_hex_list)
df["group_list"]  = df["TLS Elliptic Curves"].apply(parse_hex_list)
df["ext_id_list"] = df["TLS Extension Types"].apply(parse_hex_list)
df["ext_len_list"]= df["TLS Extension Lengths"].apply(parse_hex_list)

X_raw = pd.concat(
    [
        to_slots(df["cipher_list"], K_CIPHER, "cipher"),
        to_slots(df["group_list"],  K_GROUP,  "group"),
        to_slots(df["ext_id_list"], K_EXT,    "extid"),
        to_slots(df["ext_len_list"],K_EXLEN,  "extlen"),
        df[["TLS Client Version"]],
    ],
    axis=1,
)
y_raw = df["Ground Truth OS"].astype(str).to_numpy()

# ── encode features -------------------------------------------------------
print("🧮  Fitting OneHotEncoder …")
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_enc = enc.fit_transform(X_raw)
joblib.dump(enc, ENC_PATH)

# ── encode labels ---------------------------------------------------------
classes, y_int = np.unique(y_raw, return_inverse=True)   # int32 labels
np.save(LABEL_PATH, classes)

# ── shard and save --------------------------------------------------------
print(f"✂️  Sharding into {N_CLIENTS} clients …")
idx = np.random.permutation(X_enc.shape[0])
splits = np.array_split(idx, N_CLIENTS)

OUT_DIR.mkdir(exist_ok=True)
for cid, sl in enumerate(splits):
    sp.save_npz(OUT_DIR / f"X_{cid:03d}.npz", X_enc[sl])
    np.save   (OUT_DIR / f"y_{cid:03d}.npy", y_int[sl])

print("✅  Done – shards and encoder saved to", OUT_DIR)