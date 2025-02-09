import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import rasterio
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
import torch
xgb_device = ("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(dem_path, depth_path):
    """Prepare data from rasters, handling nodata values and NaN."""
    with rasterio.open(dem_path) as dem_src, rasterio.open(depth_path) as depth_src:
        dem_data = dem_src.read(1)
        depth_data = depth_src.read(1)
        
        # Create masks for valid data
        dem_mask = (dem_data != dem_src.nodata) & (~np.isnan(dem_data))
        depth_mask = (depth_data != depth_src.nodata) & (~np.isnan(depth_data))
        valid_mask = dem_mask & depth_mask
        
        # Get valid data
        X = dem_data[valid_mask].reshape(-1, 1)
        y = depth_data[valid_mask]
        
        # Store indices for reconstruction
        valid_idx = np.where(valid_mask)
        metadata = dem_src.meta.copy()
        
        print(f"Total pixels: {dem_data.size}")
        print(f"Valid pixels: {len(X)}")
        print(f"DEM range: {np.min(X):.2f} to {np.max(X):.2f}")
        print(f"Depth range: {np.min(y):.2f} to {np.max(y):.2f}")
        
    return X.astype(np.float32), y.astype(np.float32), valid_idx, metadata

def create_learning_curves(site='Dry', year='2020', resolutions=[3, 5, 10, 50, 100]):
    """Create learning curves using XGBoost native API."""
    curves_data = {}
    BASE_PATH = '/home/habeeb/insar_idaho/uavsar-lidar-ml-project2/data/Processed_Data'
    
    # Create results directory
    results_dir = os.path.join(BASE_PATH, 'Learning_Curves')
    os.makedirs(results_dir, exist_ok=True)
    
    # XGBoost parameters
    xgb_params = {
        "sampling_method": "gradient_based",
        'objective': 'reg:squarederror',
        "min_child_weight": 30,
        'learning_rate': 0.05,
        'tree_method': 'hist',
        'booster': 'gbtree',
        'device': xgb_device,
        'max_depth': 0,
        "subsample": 1,
        "max_bin":5096,
        "seed": 42
    }
    
    for resolution in resolutions:
        print(f"\nProcessing {resolution}m resolution")
        
        # Construct paths
        dem_path = os.path.join(BASE_PATH, year, site, f"{resolution}m", "LiDAR", "dem.tif")
        depth_path = os.path.join(BASE_PATH, year, site, f"{resolution}m", "LiDAR", "snow_depth.tif")
        
        # Get and prepare data
        X, y, valid_idx, metadata = prepare_data(dem_path, depth_path)
        
        # First split: 80% for training/validation, 20% held out for final testing
        train_val_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42
        )
        
        # Second split: Split remaining 80% into 70% training, 10% validation
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.125, random_state=42
        )
        
        # Create DMatrix objects for full datasets
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        
        dtrain_full = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Create range of training sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        val_scores = []
        
        print(f"\nTraining splits for {resolution}m:")
        print(f"Total samples: {len(X)}")
        print(f"Training samples: {len(X_train)} (70%)")
        print(f"Validation samples: {len(X_val)} (10%)")
        print(f"Test samples: {len(X_test)} (20%)")
        
        # Calculate learning curve points
        for train_size in train_sizes:
            print(f"\nProcessing training size: {train_size:.1%}")
            
            # Select subset of training data
            subset_size = int(len(X_train) * train_size)
            subset_idx = np.random.choice(len(X_train), subset_size, replace=False)
            X_train_subset = X_train[subset_idx]
            y_train_subset = y_train[subset_idx]
            
            # Create DMatrix for subset
            dtrain_subset = xgb.DMatrix(X_train_subset, label=y_train_subset)
            
            # Train model
            model = xgb.train(
                xgb_params,
                dtrain_subset,
                num_boost_round=1000,
                evals=[(dtrain_subset, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # Calculate scores
            train_pred = model.predict(dtrain_subset)
            val_pred = model.predict(dval)
            
            train_rmse = np.sqrt(np.mean((y_train_subset - train_pred) ** 2))
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            
            train_scores.append(train_rmse)
            val_scores.append(val_rmse)
            
            print(f"Train RMSE: {train_rmse:.3f}m")
            print(f"Validation RMSE: {val_rmse:.3f}m")
        
        curves_data[resolution] = {
            'train_sizes': train_sizes * len(X_train),
            'train_scores': train_scores,
            'val_scores': val_scores,
            'metadata': {
                'total_samples': len(X),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test)
            }
        }
        
        # Save test data indices for later use
        curves_data[resolution]['test_indices'] = test_idx.tolist()
    
    # Save curves data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(results_dir, f'learning_curves_{site}_{year}_{timestamp}.json')
    
    # Convert numpy arrays to lists for JSON serialization
    json_data = {
        res: {
            'train_sizes': data['train_sizes'].tolist(),
            'train_scores': data['train_scores'],
            'val_scores': data['val_scores'],
            'metadata': data['metadata'],
            'test_indices': data['test_indices']
        }
        for res, data in curves_data.items()
    }
    
    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    return curves_data

def plot_learning_curves(curves_data, site, year):
    """Plot learning curves for all resolutions."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    for i, (resolution, data) in enumerate(curves_data.items()):
        if i < len(axes):
            ax = axes[i]
            
            # Plot training and validation curves
            ax.plot(data['train_sizes'], data['train_scores'], 
                   label='Training', color='blue', marker='o')
            ax.plot(data['train_sizes'], data['val_scores'], 
                   label='Validation', color='red', marker='s')
            
            ax.set_title(f'{resolution}m Resolution')
            ax.set_xlabel('Number of Training Examples')
            ax.set_ylabel('RMSE (m)')
            ax.legend()
            ax.grid(True)
            
            # Add sample sizes to the plot
            ax.text(0.05, 0.95, 
                   f'Train: {data["metadata"]["train_samples"]}\n' +
                   f'Val: {data["metadata"]["val_samples"]}\n' +
                   f'Test: {data["metadata"]["test_samples"]}', 
                   transform=ax.transAxes, verticalalignment='top')
    
    plt.suptitle(f'Learning Curves - {site} {year}', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    results_dir = os.path.join('/home/habeeb/insat_part2/uavsar-lidar2/Data/Processed_Data', 'Learning_Curves')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(results_dir, f'learning_curves_{site}_{year}_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    return fig

if __name__ == "__main__":
    # Sites and years to process
    sites = ['Banner', 'Dry', 'Mores']
    years = ['2020', '2021']
    
    for site in sites:
        for year in years:
            # Skip Dry Creek 2021 as it doesn't exist
            if site == 'Dry' and year == '2021':
                continue
                
            print(f"\nProcessing {site} {year}")
            try:
                # Create learning curves
                curves_data = create_learning_curves(site=site, year=year)
                
                # Plot results
                fig = plot_learning_curves(curves_data, site, year)
                plt.close()
                
            except Exception as e:
                print(f"Error processing {site} {year}: {str(e)}")
                continue