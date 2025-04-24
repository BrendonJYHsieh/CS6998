"""Differential Privacy utilities for federated XGBoost training"""

import numpy as np
import json
import xgboost as xgb
from typing import Dict, Any, Tuple, List, Optional


class DPNoiseAdder:
    """Adds differential privacy noise to model updates"""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        clip_norm: float = 5.0,
        noise_multiplier: Optional[float] = None,
        mechanism: str = "laplace"
    ):
        """
        Initialize the DP noise adder
        
        Args:
            epsilon: Privacy budget parameter
            delta: Privacy failure probability
            sensitivity: Maximum impact any single user can have on the model
            clip_norm: Gradient clipping threshold
            noise_multiplier: Custom noise multiplier (if None, calculated from epsilon)
            mechanism: "laplace" or "gaussian"
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.clip_norm = clip_norm
        self.mechanism = mechanism.lower()
        
        # Calculate noise multiplier if not provided
        if noise_multiplier is None:
            if self.mechanism == "laplace":
                self.noise_multiplier = self.sensitivity / self.epsilon
            else:  # gaussian
                self.noise_multiplier = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        else:
            self.noise_multiplier = noise_multiplier
    
    def add_noise_to_model(self, model_bytes: bytes) -> bytes:
        """
        Add DP noise to XGBoost model
        
        Args:
            model_bytes: Raw model bytes in JSON format
            
        Returns:
            Noisy model bytes in the same format
        """
        # Load model as JSON
        model_json = json.loads(model_bytes.decode('utf-8'))
        
        # Apply DP to tree weights and values
        if "learner" in model_json and "gradient_booster" in model_json["learner"]:
            gb = model_json["learner"]["gradient_booster"]
            
            if "model" in gb and "trees" in gb["model"]:
                trees = gb["model"]["trees"]
                
                for tree in trees:
                    # Apply noise to node values
                    if "nodes" in tree:
                        for node in tree["nodes"]:
                            if "leaf" in node:
                                # Clip leaf value
                                leaf_val = node["leaf"]
                                clipped_val = np.clip(leaf_val, -self.clip_norm, self.clip_norm)
                                
                                # Add appropriate noise
                                if self.mechanism == "laplace":
                                    noise = np.random.laplace(0, self.noise_multiplier)
                                else:  # gaussian
                                    noise = np.random.normal(0, self.noise_multiplier)
                                
                                # Update leaf value with noise
                                node["leaf"] = clipped_val + noise

        # Convert back to bytes
        return json.dumps(model_json).encode('utf-8')


class SecureAggregation:
    """Implements secure aggregation protocol for model averaging"""
    
    @staticmethod
    def apply_secure_agg_mask(model_bytes: bytes, mask_seed: int) -> Tuple[bytes, int]:
        """
        Apply a secure mask to model before sending
        
        Args:
            model_bytes: Raw model bytes
            mask_seed: Seed for random mask
            
        Returns:
            Masked model bytes and mask seed
        """
        # Load model as JSON
        model_json = json.loads(model_bytes.decode('utf-8'))
        
        # Set random seed for reproducibility
        np.random.seed(mask_seed)
        
        # Apply masks to tree weights and values
        if "learner" in model_json and "gradient_booster" in model_json["learner"]:
            gb = model_json["learner"]["gradient_booster"]
            
            if "model" in gb and "trees" in gb["model"]:
                trees = gb["model"]["trees"]
                
                for tree in trees:
                    # Apply mask to node values
                    if "nodes" in tree:
                        for node in tree["nodes"]:
                            if "leaf" in node:
                                # Generate mask with zero expectation
                                mask = np.random.normal(0, 1.0)
                                
                                # Apply mask
                                node["leaf"] = node["leaf"] + mask

        # Convert back to bytes
        return json.dumps(model_json).encode('utf-8'), mask_seed
    
    @staticmethod
    def remove_secure_agg_mask(model_bytes: bytes, mask_seed: int) -> bytes:
        """
        Remove the secure mask from a model after aggregation
        
        Args:
            model_bytes: Masked model bytes
            mask_seed: Seed used for masking
            
        Returns:
            Unmasked model bytes
        """
        # Load model as JSON
        model_json = json.loads(model_bytes.decode('utf-8'))
        
        # Set random seed for reproducibility to generate same masks
        np.random.seed(mask_seed)
        
        # Remove masks from tree weights and values
        if "learner" in model_json and "gradient_booster" in model_json["learner"]:
            gb = model_json["learner"]["gradient_booster"]
            
            if "model" in gb and "trees" in gb["model"]:
                trees = gb["model"]["trees"]
                
                for tree in trees:
                    # Remove mask from node values
                    if "nodes" in tree:
                        for node in tree["nodes"]:
                            if "leaf" in node:
                                # Generate same mask with zero expectation
                                mask = np.random.normal(0, 1.0)
                                
                                # Remove mask
                                node["leaf"] = node["leaf"] - mask

        # Convert back to bytes
        return json.dumps(model_json).encode('utf-8')


def apply_dp_to_xgboost(
    bst: xgb.Booster,
    epsilon: float = 1.0,
    delta: float = 1e-5, 
    clip_norm: float = 5.0,
    mechanism: str = "laplace"
) -> xgb.Booster:
    """
    Apply differential privacy to an XGBoost model
    
    Args:
        bst: XGBoost Booster object
        epsilon: Privacy parameter
        delta: Privacy failure probability 
        clip_norm: Gradient clipping threshold
        mechanism: "laplace" or "gaussian"
        
    Returns:
        XGBoost Booster with DP applied
    """
    # Create DP noise adder
    dp_adder = DPNoiseAdder(
        epsilon=epsilon,
        delta=delta,
        clip_norm=clip_norm,
        mechanism=mechanism
    )
    
    # Save model to raw bytes
    model_bytes = bst.save_raw("json")
    
    # Add DP noise
    noisy_model_bytes = dp_adder.add_noise_to_model(bytes(model_bytes))
    
    # Create new booster and load the noisy model
    noisy_bst = xgb.Booster(model_file=bytearray(noisy_model_bytes))
    
    return noisy_bst 