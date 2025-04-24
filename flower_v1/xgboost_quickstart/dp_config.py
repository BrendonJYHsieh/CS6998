"""Configuration for differential privacy in federated learning"""

import argparse
from typing import Dict, Any

def get_dp_config_parser() -> argparse.ArgumentParser:
    """
    Creates an argument parser for differential privacy configuration
    
    Returns:
        A configured argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Differential Privacy Configuration")
    
    # Differential Privacy parameters
    parser.add_argument("--use-dp", action="store_true", default=True,
                        help="Enable differential privacy (default: True)")
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Privacy budget epsilon (default: 1.0)")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Privacy failure probability delta (default: 1e-5)")
    parser.add_argument("--clip-norm", type=float, default=5.0,
                        help="Gradient clipping threshold (default: 5.0)")
    parser.add_argument("--mechanism", type=str, default="gaussian", choices=["gaussian", "laplace"],
                        help="Noise mechanism: gaussian or laplace (default: gaussian)")
    
    # Secure Aggregation
    parser.add_argument("--use-secure-agg", action="store_true", default=True,
                        help="Enable secure aggregation (default: True)")
    
    return parser

def dp_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Converts command line arguments to a DP configuration dictionary
    
    Args:
        args: Command line arguments parsed by argparse
        
    Returns:
        Dictionary of DP configuration options
    """
    return {
        "use_dp": args.use_dp,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "clip_norm": args.clip_norm,
        "mechanism": args.mechanism,
        "use_secure_agg": args.use_secure_agg
    }

def merge_with_default_dp_config(user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges user-provided configuration with default values
    
    Args:
        user_config: User-provided DP configuration
        
    Returns:
        Complete DP configuration with defaults filled in
    """
    default_config = {
        "use_dp": True,
        "epsilon": 1.0,
        "delta": 1e-5,
        "clip_norm": 5.0,
        "mechanism": "gaussian",
        "use_secure_agg": True
    }
    
    # Update default config with user values
    for key, value in user_config.items():
        if key in default_config:
            default_config[key] = value
    
    return default_config

def get_privacy_strength_description(config: Dict[str, Any]) -> str:
    """
    Returns a human-readable description of the privacy strength
    
    Args:
        config: DP configuration dictionary
        
    Returns:
        String description of privacy strength
    """
    if not config.get("use_dp", False):
        return "No differential privacy applied"
    
    epsilon = config.get("epsilon", 1.0)
    
    if epsilon <= 0.1:
        strength = "Very strong"
    elif epsilon <= 0.5:
        strength = "Strong"
    elif epsilon <= 2.0:
        strength = "Moderate"
    elif epsilon <= 5.0:
        strength = "Weak"
    else:
        strength = "Very weak"
    
    secure_agg = "with" if config.get("use_secure_agg", False) else "without"
    
    return f"{strength} privacy (ε={epsilon}) {secure_agg} secure aggregation"

def print_privacy_settings(config: Dict[str, Any]) -> None:
    """
    Prints a summary of the privacy settings
    
    Args:
        config: DP configuration dictionary
    """
    if not config.get("use_dp", False):
        print("Differential Privacy: DISABLED")
        return
    
    print("Differential Privacy Settings:")
    print(f"  - Status: ENABLED")
    print(f"  - Privacy budget (ε): {config.get('epsilon', 1.0)}")
    print(f"  - Privacy failure probability (δ): {config.get('delta', 1e-5)}")
    print(f"  - Parameter clipping threshold: {config.get('clip_norm', 5.0)}")
    print(f"  - Noise mechanism: {config.get('mechanism', 'gaussian').upper()}")
    print(f"  - Secure aggregation: {'ENABLED' if config.get('use_secure_agg', False) else 'DISABLED'}")
    print(f"  - Privacy strength: {get_privacy_strength_description(config)}") 