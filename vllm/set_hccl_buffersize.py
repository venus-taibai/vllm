#!/usr/bin/env python3

import os
import json
import sys
import math
import subprocess
from typing import Optional

def ExtractModelPathFromArgs(args):
    """Extract the --model parameter from command-line arguments."""
    for arg in args:
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
    print("Error: --model parameter is required.", file=sys.stderr)
    sys.exit(1)

def SetHcclBuffSizeEnv(model_dir) -> Optional[int]:
    """Set environment variables based on config.json and other parameters."""
    # Check if config.json exists
    config_file = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_file):
        print(f"Error: config.json not found in {model_dir}", file=sys.stderr)
        return None

    # Load config.json
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse config.json: {e}", file=sys.stderr)
        return None

    # Extract specific fields and store in variables
    hidden_size = config.get("hidden_size")
    n_routed_experts = config.get("n_routed_experts")
    num_experts_per_tok = config.get("num_experts_per_tok")

    # Check if the fields exist and print warnings if not
    if hidden_size is None:
        print("Warning: 'hidden_size' not found in config.json", file=sys.stderr)
    if n_routed_experts is None:
        print("Warning: 'n_routed_experts' not found in config.json", file=sys.stderr)
    if num_experts_per_tok is None:
        print("Warning: 'num_experts_per_tok' not found in config.json", file=sys.stderr)

    # Read BATCH_SIZE and EP_WORLD_SIZE from environment variables
    batch_size = os.getenv("BATCH_SIZE")
    ep_world_size = os.getenv("EP_WORLD_SIZE")

    if batch_size is None:
        print("Error: BATCH_SIZE is not set in the environment.", file=sys.stderr)
        return None
    if ep_world_size is None:
        print("Error: EP_WORLD_SIZE is not set in the environment.", file=sys.stderr)
        return None

    # Convert BATCH_SIZE and EP_WORLD_SIZE to integers
    try:
        batch_size = int(batch_size)
        ep_world_size = int(ep_world_size)
    except ValueError:
        print("Error: BATCH_SIZE and EP_WORLD_SIZE must be integers.", file=sys.stderr)
        return None

    # Calculate HCCL_BUFFSIZE
    if hidden_size is not None and n_routed_experts is not None and num_experts_per_tok is not None:
        K = num_experts_per_tok
        H = hidden_size
        RouterExpertNum = n_routed_experts
        BS = batch_size
        epWorldSize = ep_world_size
        sizeOfUint16 = 16/8
        hccl_buffsize = 2 * (
            BS * epWorldSize * min(RouterExpertNum / epWorldSize, K) * H * sizeOfUint16 / (1024 * 1024.0) + 2
        )
        hccl_buffsize = math.ceil(hccl_buffsize)  # Round up to the nearest integer

        
        # Set HCCL_BUFFSIZE as an environment variable
        # TODO: Only Decode Instance Needs the HCCL_BUFFSIZE
        # Need to distinguish between Decode, Prefill and Mix Instance.
        os.environ["HCCL_BUFFSIZE"] = str(hccl_buffsize)
        return hccl_buffsize
    return None

def main():
    # Get all command-line arguments except the script name
    args = sys.argv[1:]
    # Extract model path
    model_dir = ExtractModelPathFromArgs(args)
    hccl_buffsize = SetHcclBuffSizeEnv(model_dir)
    print(f"export HCCL_BUFFSIZE={hccl_buffsize}\n")

if __name__ == "__main__":
    main()