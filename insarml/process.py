import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
from datetime import datetime
import json
from config import *
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from affine import Affine

############################################################################
#                              UTILITY FUNCTIONS                             #
############################################################################

def cleanup_temp_files(directory):
    """Clean up any temporary files in the directory and its subdirectories."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.temp') or '.temp.' in file:
                temp_file = os.path.join(root, file)
                try:
                    os.remove(temp_file)
                    print(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    print(f"Failed to remove temporary file {temp_file}: {str(e)}")

def get_reference_parameters(ref_path, target_resolution):
    """Get reference grid parameters for alignment."""
    with rasterio.open(ref_path) as ref:
        bounds = ref.bounds
        transform = Affine.translation(bounds.left, bounds.top) * \
                   Affine.scale(target_resolution, -target_resolution)
        width = int((bounds.right - bounds.left) / target_resolution)
        height = int((bounds.top - bounds.bottom) / target_resolution)
        return bounds, transform, width, height

############################################################################
#                           PROCESSING FUNCTIONS                             #
############################################################################

def align_rasters(input_path, output_path, ref_bounds, ref_transform, ref_width, ref_height, 
                 target_crs, shapefile_gdf, product_type):
    """
    Align raster to reference grid with proper phase handling.
    """
    try:
        with rasterio.open(input_path) as src:
            # Handle wrapped phase conversion
            if product_type == 'wrapped_phase':
                # Read and convert complex data to phase angles
                data = src.read(1)
                data = np.angle(data)  # Convert to phase angles (-π to π)
                
                # Create temporary file for phase data
                temp_phase_path = output_path + '.phase.temp'
                kwargs_phase = src.meta.copy()
                kwargs_phase.update({
                    'dtype': 'float32',
                    'nodata': -9999
                })
                
                with rasterio.open(temp_phase_path, 'w', **kwargs_phase) as tmp:
                    tmp.write(data, 1)
                
                # Use the phase data as source
                src = rasterio.open(temp_phase_path)
            
            # Setup output parameters
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': ref_transform,
                'width': ref_width,
                'height': ref_height,
                'nodata': -9999,
                'dtype': 'float32'
            })
            
            # Create temporary file for reprojection
            temp_path = output_path + '.temp'
            with rasterio.open(temp_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=target_crs,
                    resampling=RESAMPLING_METHODS[product_type],
                    nodata=-9999
                )
            
            # Clean up phase temporary file if it exists
            if product_type == 'wrapped_phase':
                src.close()
                if os.path.exists(temp_phase_path):
                    os.remove(temp_phase_path)
            
            # Clip with shapefile
            with rasterio.open(temp_path) as dst:
                out_image, out_transform = mask(dst, shapefile_gdf.geometry, crop=True, 
                                             nodata=-9999, filled=True)
                kwargs = dst.meta.copy()
                kwargs.update({
                    'height': out_image.shape[1],
                    'width': out_image.shape[2],
                    'transform': out_transform,
                    'nodata': -9999
                })
                
                # Save final clipped raster
                with rasterio.open(output_path, 'w', **kwargs) as final:
                    final.write(out_image)
            
            # Clean up temporary reprojection file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return True
            
    except Exception as e:
        print(f"Error processing {product_type}: {str(e)}")
        return False

def process_site_year_resolution(args):
    """
    Process all products for a specific site, year, and resolution combination.
    """
    site, year, resolution = args
    print(f"\nProcessing {site} - {year} at {resolution}m resolution")
    
    try:
        # Setup output directories
        base_output_dir = os.path.join(BASE_OUTPUT_PATH, year, site, f"{resolution}m")
        lidar_dir = os.path.join(base_output_dir, 'LiDAR')
        uavsar_dir = os.path.join(base_output_dir, 'UAVSAR')
        validation_dir = os.path.join(base_output_dir, 'Validation')
        
        # Ensure directories exist
        for directory in [lidar_dir, uavsar_dir, validation_dir]:
            os.makedirs(directory, exist_ok=True)
        
        validation_results = {
            'site': site,
            'year': year,
            'resolution': resolution,
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'products': {}
        }
        
        # Get reference parameters from LiDAR DEM
        ref_bounds, ref_transform, ref_width, ref_height = get_reference_parameters(
            SITES[site]['dem'], resolution
        )
        
        # Get shapefile
        shapefile_gdf = gpd.read_file(SITES[site]['shapefile'])
        
        # Process UAVSAR products
        uavsar_products = UAVSAR_DATA[year]
        for product_type, input_path in uavsar_products.items():
            output_path = os.path.join(uavsar_dir, f"{product_type}.tif")
            print(f"Processing {site} {year} {resolution}m - UAVSAR {product_type}")
            
            success = align_rasters(
                input_path, output_path,
                ref_bounds, ref_transform, ref_width, ref_height,
                TARGET_CRS, shapefile_gdf, product_type
            )
            
            validation_results['products'][f"uavsar_{product_type}"] = {
                'status': 'success' if success else 'failed',
                'product_type': product_type,
                'resolution': resolution
            }
        
        # Process LiDAR products
        if year in LIDAR_DATA and site in LIDAR_DATA[year]:
            lidar_products = LIDAR_DATA[year][site]
            for product_type, input_path in lidar_products.items():
                output_path = os.path.join(lidar_dir, f"{product_type}.tif")
                print(f"Processing {site} {year} {resolution}m - LiDAR {product_type}")
                
                success = align_rasters(
                    input_path, output_path,
                    ref_bounds, ref_transform, ref_width, ref_height,
                    TARGET_CRS, shapefile_gdf, product_type
                )
                
                validation_results['products'][f"lidar_{product_type}"] = {
                    'status': 'success' if success else 'failed',
                    'product_type': product_type,
                    'resolution': resolution
                }
        
        # Process site DEM
        dem_output_path = os.path.join(lidar_dir, "dem.tif")
        print(f"Processing {site} {resolution}m - DEM")
        success = align_rasters(
            SITES[site]['dem'], dem_output_path,
            ref_bounds, ref_transform, ref_width, ref_height,
            TARGET_CRS, shapefile_gdf, 'dem'
        )
        
        validation_results['products']['lidar_dem'] = {
            'status': 'success' if success else 'failed',
            'product_type': 'dem',
            'resolution': resolution
        }
        
        validation_results['status'] = 'success'
        
    except Exception as e:
        print(f"Error processing {site} {year} {resolution}m: {str(e)}")
        validation_results['status'] = 'failed'
        validation_results['error'] = str(e)
    
    finally:
        # Clean up any remaining temporary files
        cleanup_temp_files(base_output_dir)
        
        # Save validation results
        validation_file = os.path.join(validation_dir, f"validation_report.json")
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=4)
    
    return validation_results

############################################################################
#                              MAIN EXECUTION                                #
############################################################################

def main():
    """
    Main processing function with parallel processing.
    """
    print("Starting data processing...")
    
    # Initial cleanup of any existing temp files
    cleanup_temp_files(BASE_OUTPUT_PATH)
    
    # Create processing tasks
    tasks = []
    for site in SITES:
        for year in SITES[site]['years']:
            for resolution in RESOLUTIONS:
                tasks.append((site, year, resolution))
    
    # Get number of CPU cores (leave one core free for system)
    n_cores = max(1, mp.cpu_count() - 1)
    print(f"Using {n_cores} CPU cores for processing")
    
    try:
        # Process tasks in parallel with progress bar
        with mp.Pool(n_cores) as pool:
            results = list(tqdm(pool.imap(process_site_year_resolution, tasks), 
                              total=len(tasks),
                              desc="Processing all combinations"))
        
        print("\nProcessing completed successfully!")
        
        # Summarize results
        print("\nProcessing Summary:")
        success_count = 0
        failed_count = 0
        for result in results:
            if result.get('status') == 'success':
                success_count += 1
            else:
                failed_count += 1
                print(f"Failed task: {result.get('error', 'Unknown error')}")
        
        print(f"Successfully processed: {success_count} combinations")
        print(f"Failed processing: {failed_count} combinations")
    
    finally:
        # Final cleanup
        print("\nPerforming final cleanup...")
        cleanup_temp_files(BASE_OUTPUT_PATH)

if __name__ == "__main__":
    main()