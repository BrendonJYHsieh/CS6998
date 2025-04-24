"""Encrypted Echoes: Flower server implementation for federated TLS fingerprinting."""

import os
from typing import Dict, List, Tuple, Union
import numpy as np

import xgboost as xgb
from flwr.common import Context, FitRes, Parameters, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedXgbBagging

from xgboost_quickstart.dp_utils import SecureAggregation

class PrivacyAwareFedXgbBagging(FedXgbBagging):
    """Extended FedXgbBagging strategy with privacy features"""
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_secure_agg = True  # Enable secure aggregation by default
        self.secure_agg_seed = np.random.randint(1, 1000000)  # Server-side seed
        self.privacy_accounting = {
            "total_epsilon": 0.0,
            "total_delta": 0.0,
            "num_rounds": 0
        }
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[int, FitRes]],
        failures: List[Union[Tuple[int, FitRes], BaseException]],
    ):
        """Aggregate model updates using secure aggregation if enabled."""
        # First perform standard aggregation from FedXgbBagging
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        # If secure aggregation is enabled, remove the mask from the aggregated model
        if self.use_secure_agg and aggregated_result is not None:
            # Access the parameters
            parameters = aggregated_result[0]
            if parameters.tensors:
                model_bytes = parameters.tensors[0]
                
                # Remove the secure mask from the aggregated model
                unmasked_model_bytes = SecureAggregation.remove_secure_agg_mask(
                    model_bytes, 
                    self.secure_agg_seed
                )
                
                # Update the parameters with the unmasked model
                parameters.tensors[0] = unmasked_model_bytes
        
        # Update privacy accounting based on client metrics
        if results:
            round_epsilon = 0.0
            round_delta = 0.0
            num_clients = len(results)
            
            for _, fit_res in results:
                client_metrics = fit_res.metrics
                if "privacy_budget_epsilon" in client_metrics:
                    round_epsilon += float(client_metrics["privacy_budget_epsilon"])
                if "privacy_budget_delta" in client_metrics:
                    round_delta += float(client_metrics["privacy_budget_delta"])
            
            # Average across clients
            if num_clients > 0:
                round_epsilon /= num_clients
                round_delta /= num_clients
                
                # Update cumulative privacy accounting
                self.privacy_accounting["total_epsilon"] += round_epsilon
                self.privacy_accounting["total_delta"] += round_delta
                self.privacy_accounting["num_rounds"] += 1
                
                print(f"Round {server_round} privacy metrics:")
                print(f"  - ε (epsilon): {round_epsilon}")
                print(f"  - δ (delta): {round_delta}")
                print(f"Cumulative privacy expenditure:")
                print(f"  - Total ε: {self.privacy_accounting['total_epsilon']}")
                print(f"  - Total δ: {self.privacy_accounting['total_delta']}")
        
        return aggregated_result

def evaluate_metrics_aggregation(eval_metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate evaluation metrics weighted by number of examples."""
    if not eval_metrics:
        return {}
    
    total_examples = sum(num_examples for num_examples, _ in eval_metrics)
    if total_examples == 0:
        return {}
    
    # Get all metric keys from all clients
    metric_keys = set()
    for _, metrics in eval_metrics:
        metric_keys.update(metrics.keys())
    
    # Aggregate each metric weighted by number of examples
    aggregated_metrics = {}
    for key in metric_keys:
        weighted_sum = 0.0
        examples_with_key = 0
        
        for num_examples, metrics in eval_metrics:
            if key in metrics:
                weighted_sum += metrics[key] * num_examples
                examples_with_key += num_examples
        
        if examples_with_key > 0:
            aggregated_metrics[key] = weighted_sum / examples_with_key
    
    return aggregated_metrics

def config_func(rnd: int) -> Dict[str, str]:
    """Return configuration for clients in the current round."""
    return {
        "global_round": str(rnd),
    }

def server_fn(context: Context) -> ServerAppComponents:
    """Configure and return the server components."""
    # Read from config
    num_rounds = int(context.run_config.get("num-server-rounds", "10"))
    fraction_fit = float(context.run_config.get("fraction-fit", "1.0"))
    fraction_evaluate = float(context.run_config.get("fraction-evaluate", "1.0"))
    min_fit_clients = int(context.run_config.get("min-fit-clients", "2"))
    min_evaluate_clients = int(context.run_config.get("min-evaluate-clients", "2"))
    min_available_clients = int(context.run_config.get("min-available-clients", "2"))
    
    # Create empty Parameters object
    parameters = Parameters(tensor_type="", tensors=[])
    
    # Define aggregation strategy
    strategy = PrivacyAwareFedXgbBagging(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=None,  # No need to aggregate fit metrics
    )
    
    # Configure the server
    config = ServerConfig(num_rounds=num_rounds)
    
    # Create a callback to save the final global model after training
    def save_final_model(server_round: int, results: List[Tuple[int, FitRes]]) -> None:
        """Save the final global model after training completes."""
        if server_round == num_rounds:
            print(f"Saving global model after round {server_round}...")
            try:
                # Get the latest global model parameters
                global_model_bytes = strategy.parameters.tensors[0]
                
                # Load the model
                bst = xgb.Booster()
                bst.load_model(bytearray(global_model_bytes))
                
                # Save the model
                os.makedirs("global_models", exist_ok=True)
                bst.save_model("global_models/final_federated_model.json")
                
                # Save privacy report
                with open("global_models/privacy_report.txt", "w") as f:
                    f.write("Privacy Report for Federated Learning\n")
                    f.write("====================================\n\n")
                    f.write(f"Total training rounds: {strategy.privacy_accounting['num_rounds']}\n")
                    f.write(f"Final privacy budget expenditure:\n")
                    f.write(f"  - Total ε (epsilon): {strategy.privacy_accounting['total_epsilon']}\n")
                    f.write(f"  - Total δ (delta): {strategy.privacy_accounting['total_delta']}\n")
                
                print("Global model and privacy report saved successfully!")
            except Exception as e:
                print(f"Error saving global model: {e}")
    
    strategy.on_fit_end = save_final_model
    
    return ServerAppComponents(strategy=strategy, config=config)

# Create and configure the Flower server
app = ServerApp(
    server_fn=server_fn,
)
