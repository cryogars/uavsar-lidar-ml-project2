import os
import numpy as np
import xgboost as xgb
import torch
import rasterio
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import gc
import subprocess

# Define device
xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

# XGBoost parameters
xgb_params = {
    "sampling_method": "gradient_based",
    'objective': 'reg:squarederror',
    "min_child_weight": 30,
    'learning_rate': 0.3,
    'tree_method': 'hist',
    'booster': 'gbtree',
    'device': xgb_device,
    'max_depth': 0,
    "subsample": 1,
    "max_bin": 5096,
    "seed": 42
}

def cleanup_gpu():
    """Clean up GPU memory."""
    try:
        # Force garbage collection
        gc.collect()
        
        # CUDA cache clear if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Print GPU memory usage
        subprocess.run(["nvidia-smi"])
        
    except Exception as e:
        print(f"GPU cleanup warning: {str(e)}")

def load_data(base_path, site, year, resolution):
    """Load all required data for a specific site, year, and resolution."""
    data_path = os.path.join(base_path, year, site, f"{resolution}m")
    
    # Load all required rasters
    data = {}
    with rasterio.open(os.path.join(data_path, "LiDAR", "dem.tif")) as src:
        data['dem'] = src.read(1)
        data['transform'] = src.transform
        data['crs'] = src.crs
        nodata_mask = data['dem'] != src.nodata
    
    # Load other features based on model requirements
    features = {
        'wrapped_phase': 'UAVSAR/wrapped_phase.tif',
        'unwrapped_phase': 'UAVSAR/unwrapped_phase.tif',
        'inc_angle': 'UAVSAR/inc_angle.tif',
        'coherence': 'UAVSAR/coherence.tif',
        'amplitude': 'UAVSAR/amplitude.tif',
        'veg_height': 'LiDAR/veg_height.tif',
        'snow_depth': 'LiDAR/snow_depth.tif'
    }
    
    for feature, path in features.items():
        with rasterio.open(os.path.join(data_path, path)) as src:
            data[feature] = src.read(1)
            nodata_mask &= (data[feature] != src.nodata)
    
    # Create valid data mask
    valid_mask = nodata_mask & ~np.isnan(data['snow_depth']) & (data['snow_depth'] > 0)
    
    # Store indices for reconstruction
    data['valid_indices'] = np.where(valid_mask)
    
    return data, valid_mask

def prepare_features(data, model_type):
    """Prepare feature sets based on model type."""
    if model_type == 1:
        features = ['dem']
    elif model_type == 2:
        features = ['dem', 'wrapped_phase', 'unwrapped_phase', 'inc_angle', 
                   'coherence', 'amplitude']
    else:
        features = ['dem', 'wrapped_phase', 'unwrapped_phase', 'inc_angle', 
                   'coherence', 'amplitude', 'veg_height']
    
    X = np.column_stack([data[f][data['valid_indices']] for f in features])
    y = data['snow_depth'][data['valid_indices']]
    
    return X, y, features

def train_and_evaluate(X, y, train_idx, test_idx, veg_height=None):
    """Train model and evaluate performance."""
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
    dtest = xgb.DMatrix(X[test_idx], label=y[test_idx])
    
    # Train model
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Get predictions
    train_pred = model.predict(dtrain)
    test_pred = model.predict(dtest)
    
    # Calculate metrics
    metrics = {
        'train': calculate_metrics(y[train_idx], train_pred, 
                                 None if veg_height is None else veg_height[train_idx]),
        'test': calculate_metrics(y[test_idx], test_pred,
                                None if veg_height is None else veg_height[test_idx])
    }
    
    return model, train_pred, test_pred, metrics

def calculate_metrics(y_true, y_pred, veg_height=None):
    """Calculate performance metrics."""
    metrics = {
        'overall': {
            'rmse': float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            'mae': float(np.mean(np.abs(y_true - y_pred)))
        }
    }
    
    if veg_height is not None:
        # Calculate metrics for open and vegetated areas
        open_mask = veg_height < 0.5
        veg_mask = veg_height >= 0.5
        
        metrics['open'] = {
            'rmse': float(np.sqrt(np.mean((y_true[open_mask] - y_pred[open_mask]) ** 2))),
            'mae': float(np.mean(np.abs(y_true[open_mask] - y_pred[open_mask])))
        }
        
        metrics['vegetated'] = {
            'rmse': float(np.sqrt(np.mean((y_true[veg_mask] - y_pred[veg_mask]) ** 2))),
            'mae': float(np.mean(np.abs(y_true[veg_mask] - y_pred[veg_mask])))
        }
    
    return metrics

def run_model(base_path, results_path, site, year, model_type, resolution):
    """Run a specific model configuration and save results."""
    print(f"\nProcessing: Site={site}, Year={year}, Model={model_type}, Resolution={resolution}m")
    
    try:
        # Create results directory
        model_results_path = os.path.join(results_path, year, f"model{model_type}", f"{resolution}m", site)
        os.makedirs(model_results_path, exist_ok=True)
        
        # Load and prepare data
        data, valid_mask = load_data(base_path, site, year, resolution)
        X, y, features = prepare_features(data, model_type)
        
        # Split data
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.3, random_state=42
        )
        
        # Train and evaluate model
        model, train_pred, test_pred, metrics = train_and_evaluate(
            X, y, train_idx, test_idx,
            veg_height=data['veg_height'][data['valid_indices']] if model_type == 3 else None
        )
        
        # Prepare results
        results = {
            'metadata': {
                'site': site,
                'year': year,
                'model_type': model_type,
                'resolution': resolution,
                'features': features,
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'data': {
                'train_indices': train_idx.tolist(),
                'test_indices': test_idx.tolist(),
                'valid_indices': [coord.tolist() for coord in data['valid_indices']],
                'predictions': {
                    'train': train_pred.tolist(),
                    'test': test_pred.tolist()
                }
            },
            'metrics': metrics
        }
        
        # Save results
        results_file = os.path.join(model_results_path, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save model
        model_file = os.path.join(model_results_path, 'model.json')
        model.save_model(model_file)
        
        print(f"Results saved to: {model_results_path}")
        print("Metrics:", metrics)
        
    except Exception as e:
        print(f"Error processing {site} {year} model{model_type} {resolution}m: {str(e)}")
        raise
    
    finally:
        # Clean up GPU memory
        cleanup_gpu()

if __name__ == "__main__":
    # Base paths
    BASE_PATH = '/home/habeeb/insar_idaho/uavsar-lidar-ml-project2/data/Processed_Data'
    RESULTS_PATH = '/home/habeeb/insar_idaho/uavsar-lidar-ml-project2/results'
    
    # Define sites
    sites = ['Banner', 'Dry', 'Mores']
    
    # Example usage:
    # Run a specific configuration
    site = 'Banner'
    year = '2020'
    model_type = 1
    resolution = 3
    
    run_model(BASE_PATH, RESULTS_PATH, site, year, model_type, resolution)