"""xgboost_quickstart.task  – lightweight loader for pre‑computed shards"""

from pathlib import Path
from logging import INFO
import numpy as np, scipy.sparse as sp, xgboost as xgb
from sklearn.model_selection import train_test_split
from flwr.common import log

BASE_DIR  = Path(__file__).resolve().parent.parent      # /flower_v1
SHARD_DIR = BASE_DIR / "tls_shards"
TEST_FRAC = 0.20

def load_data(partition_id: int, num_clients: int):
    """Return train/valid DMatrices for this client id."""
    X = sp.load_npz(SHARD_DIR / f"X_{partition_id:03d}.npz")
    y = np.load(SHARD_DIR / f"y_{partition_id:03d}.npy", allow_pickle=True)
    
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=42, stratify=y
    )

    log(INFO, f"[Client {partition_id}] rows: train {X_tr.shape[0]}  val {X_va.shape[0]}")
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)
    return dtrain, dvalid, X_tr.shape[0], X_va.shape[0]

# keep replace_keys unchanged if your codebase still uses it
def replace_keys(d, match="-", target="_"):
    return {k.replace(match,target): replace_keys(v,match,target) if isinstance(v,dict) else v
            for k,v in d.items()}
