import rasterio
import matplotlib.pyplot as plt
import numpy as np
from config import *

def print_product_stats(product_path, product_name):
    """Print detailed statistics and diagnostics for a product."""
    with rasterio.open(product_path) as src:
        data = src.read(1)
        print(f"\n{product_name}:")
        print(f"  Nodata value: {src.nodata}")
        print(f"  Data type: {data.dtype}")
        print(f"  Total pixels: {data.size}")
        print(f"  Nodata pixels: {np.sum(data == src.nodata)}")
        print(f"  Valid pixels: {np.sum(data != src.nodata)}")
        print(f"  Unique values: {len(np.unique(data))}")
        
        # Check for infinity and NaN
        print(f"  Contains inf: {np.any(np.isinf(data))}")
        print(f"  Contains NaN: {np.any(np.isnan(data))}")
        
        # Get valid data
        valid_mask = (data != src.nodata) & (~np.isinf(data)) & (~np.isnan(data))
        valid_data = data[valid_mask]
        
        if valid_data.size > 0:
            print(f"  Min: {np.min(valid_data):.4f}")
            print(f"  Max: {np.max(valid_data):.4f}")
            print(f"  Mean: {np.mean(valid_data):.4f}")
            print(f"  Std: {np.std(valid_data):.4f}")
            print(f"  Sample values: {valid_data.flatten()[:5]}")
        else:
            print("  No valid data found!")
            print(f"  Sample raw values: {data.flatten()[:5]}")

def visualize_products(site='Dry', year='2020', resolution=100, figsize=(15, 20)):
    """Visualize all products for given site, year, and resolution."""
    
    # Set up paths
    base_path = os.path.join(BASE_OUTPUT_PATH, year, site, f"{resolution}m")
    uavsar_dir = os.path.join(base_path, 'UAVSAR')
    lidar_dir = os.path.join(base_path, 'LiDAR')
    
    # Define products
    uavsar_products = ['amplitude', 'coherence', 'dem', 'wrapped_phase', 'unwrapped_phase', 'inc_angle']
    lidar_products = ['snow_depth', 'veg_height', 'dem']
    
    # Calculate number of rows needed (3 products per row)
    total_products = len(uavsar_products) + len(lidar_products)
    n_rows = (total_products + 2) // 3  # Round up division
    
    # Create figure
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    fig.suptitle(f'{site} {year} {resolution}m Resolution Products', fontsize=16, y=0.95)
    axes = axes.flatten()
    
    # Plot counter
    plot_idx = 0
    
    def plot_product(path, title, ax):
        with rasterio.open(path) as src:
            data = src.read(1)
            
            # Create mask for nodata, inf, and nan
            mask = (data == src.nodata) | np.isinf(data) | np.isnan(data)
            data = np.ma.masked_array(data, mask)
            
            # Different colormaps for different products
            cmap = 'viridis'
            if 'phase' in title.lower():
                cmap = 'hsv'  # Circular colormap for phase
            elif 'coherence' in title.lower():
                cmap = 'RdYlBu'  # Red-Yellow-Blue for coherence
            elif 'amplitude' in title.lower():
                cmap = 'gray'  # Grayscale for amplitude
            elif 'dem' in title.lower():
                cmap = 'terrain'  # Terrain colormap for DEM
            
            im = ax.imshow(data, cmap=cmap)
            plt.colorbar(im, ax=ax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Print diagnostics and plot UAVSAR products
    print("\nUAVSAR Products Diagnostics:")
    print("=" * 50)
    for product in uavsar_products:
        product_path = os.path.join(uavsar_dir, f"{product}.tif")
        if os.path.exists(product_path):
            print_product_stats(product_path, f"UAVSAR {product}")
            plot_product(product_path, f'UAVSAR {product}', axes[plot_idx])
            plot_idx += 1
    
    # Print diagnostics and plot LiDAR products
    print("\nLiDAR Products Diagnostics:")
    print("=" * 50)
    for product in lidar_products:
        product_path = os.path.join(lidar_dir, f"{product}.tif")
        if os.path.exists(product_path):
            print_product_stats(product_path, f"LiDAR {product}")
            plot_product(product_path, f'LiDAR {product}', axes[plot_idx])
            plot_idx += 1
    
    # Turn off any unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_products()