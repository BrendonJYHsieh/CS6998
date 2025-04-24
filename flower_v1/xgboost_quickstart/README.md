# Privacy-Enhanced Federated Learning with XGBoost

This module implements privacy-enhanced federated learning using XGBoost and Flower. It includes differential privacy mechanisms and secure aggregation to protect client data during the federated learning process.

## Features

- Federated learning with XGBoost for classification tasks
- Differential privacy with configurable privacy budget (ε and δ)
- Support for both Laplace and Gaussian noise mechanisms
- Secure aggregation for additional protection during model updates
- Privacy accounting to track cumulative privacy budget usage
- Customizable privacy settings via configuration

## How Differential Privacy Works

The implementation uses two main privacy-enhancing technologies:

1. **Differential Privacy (DP)**: Adds calibrated noise to model updates to protect individual data points.
   - Configurable privacy budget (ε) controls the privacy-utility tradeoff
   - Lower ε values provide stronger privacy guarantees but may impact model accuracy
   - Supports both Laplace and Gaussian noise mechanisms

2. **Secure Aggregation**: Adds random masks to client models before sending them to the server.
   - Masks cancel out during aggregation but protect individual model updates
   - Provides protection against honest-but-curious server

## Usage

### Starting the Server

```bash
python -m flwr.server.start_server \
  --server-address=[::]:8080 \
  --config={"num-server-rounds": 10}
```

### Starting Clients with Custom Privacy Settings

```bash
python -m flwr.client.start_client \
  --server-address=[::]:8080 \
  --config={"dp_config": {"use_dp": true, "epsilon": 0.5, "mechanism": "gaussian", "use_secure_agg": true}}
```

### Privacy Configuration Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `use_dp` | Enable/disable differential privacy | `true` |
| `epsilon` | Privacy budget (lower = stronger privacy) | `1.0` |
| `delta` | Privacy failure probability | `1e-5` |
| `clip_norm` | Parameter clipping threshold | `5.0` |
| `mechanism` | Noise mechanism (`gaussian` or `laplace`) | `gaussian` |
| `use_secure_agg` | Enable/disable secure aggregation | `true` |

## Privacy-Utility Tradeoff

- ε ≤ 0.1: Very strong privacy, may significantly impact utility
- ε ≤ 0.5: Strong privacy, moderate impact on utility
- ε ≤ 2.0: Moderate privacy, smaller impact on utility
- ε ≤ 5.0: Weak privacy, minimal impact on utility
- ε > 5.0: Very weak privacy, negligible impact on utility

## Example: Adjusting Privacy Budget

For stronger privacy guarantees:

```bash
python -m flwr.client.start_client \
  --server-address=[::]:8080 \
  --config={"dp_config": {"epsilon": 0.5}}
```

For better utility (weaker privacy):

```bash
python -m flwr.client.start_client \
  --server-address=[::]:8080 \
  --config={"dp_config": {"epsilon": 5.0}}
```

## Privacy Report

After training completes, a privacy report is generated at `global_models/privacy_report.txt` containing:
- Total training rounds
- Final privacy budget expenditure (ε and δ)

## Advanced Usage

### Direct Python API

```python
from xgboost_quickstart.dp_utils import apply_dp_to_xgboost
from xgboost_quickstart.dp_config import merge_with_default_dp_config

# Define custom DP configuration
custom_dp_config = {
    "epsilon": 0.5,
    "mechanism": "gaussian"
}

# Merge with defaults
dp_config = merge_with_default_dp_config(custom_dp_config)

# Apply DP to an XGBoost model
private_model = apply_dp_to_xgboost(
    bst=my_xgboost_model,
    epsilon=dp_config["epsilon"],
    delta=dp_config["delta"],
    clip_norm=dp_config["clip_norm"],
    mechanism=dp_config["mechanism"]
) 