"""Encrypted Echoes: Flower client implementation for federated TLS fingerprinting."""

import warnings
import os
import joblib

from flwr.common.context import Context
import xgboost as xgb
from flwr.client import Client, ClientApp
from flwr.common.config import unflatten_dict
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
)

from xgboost_quickstart.task import load_data, replace_keys

warnings.filterwarnings("ignore", category=UserWarning)

# Define Flower Client and client_fn
class FlowerClient(Client):
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params

    def _local_boost(self, bst_input):
        # Update trees based on local training data
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for server aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds()
            - self.num_local_round : bst_input.num_boosted_rounds()
        ]

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        # Save local model for inspection
        client_dir = f"client_models/client_{ins.config.get('partition_id', 'unknown')}"
        os.makedirs(client_dir, exist_ok=True)
        bst.save_model(f"{client_dir}/model_round_{global_round}.json")

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Add merror as an evaluation metric to measure accuracy
        if "eval_metric" in self.params:
            if isinstance(self.params["eval_metric"], list):
                if "merror" not in self.params["eval_metric"]:
                    self.params["eval_metric"].append("merror")
            elif self.params["eval_metric"] != "merror":
                self.params["eval_metric"] = ["mlogloss", "merror"]
        else:
            self.params["eval_metric"] = ["mlogloss", "merror"]
        
        # Set evaluation metrics in booster
        for metric in self.params["eval_metric"]:
            bst.set_param(f"eval_metric", metric)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        
        # Parse the results to get error rate (merror)
        result_parts = eval_results.split()
        error_rate = None
        
        # Find merror in results
        for part in result_parts:
            if "merror" in part:
                error_rate = float(part.split(":")[1])
                break
        
        # If we found error rate, calculate accuracy (1 - error_rate)
        if error_rate is not None:
            accuracy = 1.0 - error_rate
        else:
            # If merror not found, use the old method as fallback
            accuracy = float(eval_results.split("\t")[1].split(":")[1])
        
        # Get label encodings to compute other metrics
        os_labels_path = "preprocessed/os_labels.joblib"
        if os.path.exists(os_labels_path):
            os_labels = joblib.load(os_labels_path)
            metrics = {
                "accuracy": accuracy,
                "num_classes": len(os_labels)
            }
        else:
            metrics = {"accuracy": accuracy}

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=1.0 - accuracy,  # Use 1-accuracy as loss
            num_examples=self.num_val,
            metrics=metrics,
        )

def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(
        partition_id, num_partitions
    )

    cfg = replace_keys(unflatten_dict(context.run_config))
    num_local_round = cfg.get("local_epochs", 10)  # Default to 10 if not specified
    
    # Default XGBoost parameters if not provided in context
    default_params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "exact",
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    
    # Use provided params or default
    params = cfg.get("params", default_params)
    
    # Load label information to set num_class
    os_labels_path = "preprocessed/os_labels.joblib"
    if os.path.exists(os_labels_path):
        os_labels = joblib.load(os_labels_path)
        params["num_class"] = len(os_labels)
    
    # Return Client instance
    return FlowerClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
    )

# Flower ClientApp
app = ClientApp(
    client_fn,
)