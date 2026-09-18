import time
import os
import torch
import torch.nn as nn
import numpy as np
import lightgbm as lgb

def load_mlp_from_state_dict(model_path):
    """
    Dynamically reconstructs the exact MLP architecture directly from
    the saved weights file without needing any class imports.
    """
    state_dict = torch.load(model_path, map_location='cpu')
    
    weights = [v for k, v in state_dict.items() if 'weight' in k]
    biases = [v for k, v in state_dict.items() if 'bias' in k]
    
    layers = []
    for i, (w, b) in enumerate(zip(weights, biases)):
        out_features, in_features = w.shape
        linear_layer = nn.Linear(in_features, out_features)
        linear_layer.weight.data.copy_(w)
        linear_layer.bias.data.copy_(b)
        layers.append(linear_layer)
        
        # Add ReLU activation between hidden layers
        if i < len(weights) - 1:
            layers.append(nn.ReLU())
            
    model = nn.Sequential(*layers)
    model.eval()
    
    num_features = weights[0].shape[1]
    return model, num_features

def run_benchmark():
    mlp_path = "results/model.pt"
    lgbm_path = "results/lgbm_model.txt"
    
    # -------------------------------------------------------------------------
    # 1. Benchmark MLP
    # -------------------------------------------------------------------------
    print("=== Evaluating MLP ===")
    mlp_size_kb = os.path.getsize(mlp_path) / 1024
    print(f"Model Storage Size: {mlp_size_kb:.2f} KB")
    
    mlp_model, feature_count = load_mlp_from_state_dict(mlp_path)
    print(f"Detected Input Features: {feature_count}")
    
    # Create test queries with the exact feature count (81)
    mlp_queries = torch.rand((1000, feature_count), dtype=torch.float32)
    
    # Warm-up pass
    with torch.no_grad():
        _ = mlp_model(mlp_queries[:10])
        
    # Measure inference time
    start_time = time.perf_counter()
    with torch.no_grad():
        for query in mlp_queries:
            _ = mlp_model(query.unsqueeze(0))
    end_time = time.perf_counter()
    
    mlp_latency_ms = ((end_time - start_time) / len(mlp_queries)) * 1000
    print(f"Average Inference Latency: {mlp_latency_ms:.4f} ms per query\n")
    
    # -------------------------------------------------------------------------
    # 2. Benchmark LightGBM
    # -------------------------------------------------------------------------
    print("=== Evaluating LightGBM ===")
    lgbm_size_kb = os.path.getsize(lgbm_path) / 1024
    print(f"Model Storage Size: {lgbm_size_kb:.2f} KB")
    
    lgb_model = lgb.Booster(model_file=lgbm_path)
    lgbm_queries = np.random.rand(1000, feature_count).astype(np.float32)
    
    # Warm-up pass
    _ = lgb_model.predict(lgbm_queries[:10])
    
    # Measure inference time
    start_time = time.perf_counter()
    for query in lgbm_queries:
        _ = lgb_model.predict(query.reshape(1, -1))
    end_time = time.perf_counter()
    
    lgbm_latency_ms = ((end_time - start_time) / len(lgbm_queries)) * 1000
    print(f"Average Inference Latency: {lgbm_latency_ms:.4f} ms per query\n")

if __name__ == "__main__":
    run_benchmark() 