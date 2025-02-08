import rasterio
import os
import geopandas as gpd
import numpy as np
from config import *
import json
from datetime import datetime

def count_valid_pixels(raster_path):
    """Calculate area of valid pixels in km²."""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        valid_pixels = np.sum(data != src.nodata)
        pixel_area = abs(src.transform[0] * src.transform[4])
        return (valid_pixels * pixel_area) / 1_000_000  # Convert to km²

def check_products(site, year, resolution, output_dir):
    """Check all products for a given site, year, and resolution."""
    validation_results = {
        'site': site,
        'year': year,
        'resolution': resolution,
        'check_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'products': {},
        'consistency': {}
    }
    
    base_path = os.path.join(BASE_OUTPUT_PATH, year, site, f"{resolution}m")
    uavsar_dir = os.path.join(base_path, 'UAVSAR')
    lidar_dir = os.path.join(base_path, 'LiDAR')
    
    print(f"\nChecking {site} {year} {resolution}m products:")
    print("=" * 50)
    
    # Get shapefile area for comparison
    shapefile_path = SITES[site]['shapefile']
    gdf = gpd.read_file(shapefile_path)
    shapefile_area = gdf.geometry.area.sum() / 1_000_000  # Convert to km²
    validation_results['shapefile_area'] = shapefile_area
    
    # Initialize shape and area trackers
    all_shapes = set()
    all_areas = set()
    
    # Check UAVSAR products
    print("\nUAVSAR Products:")
    for product in ['amplitude', 'coherence', 'dem', 'wrapped_phase', 'unwrapped_phase', 'inc_angle']:
        product_path = os.path.join(uavsar_dir, f"{product}.tif")
        if os.path.exists(product_path):
            with rasterio.open(product_path) as src:
                area = count_valid_pixels(product_path)
                all_shapes.add(str(src.shape))
                all_areas.add(round(area, 4))
                
                validation_results['products'][f"uavsar_{product}"] = {
                    'shape': src.shape,
                    'crs': str(src.crs),
                    'area': round(area, 4),
                    'area_diff_percent': round(((area - shapefile_area) / shapefile_area) * 100, 2)
                }
                
                print(f"{product}:")
                print(f"  Shape: {src.shape}")
                print(f"  CRS: {src.crs}")
                print(f"  Area: {area:.4f} km²")
    
    # Check LiDAR products
    print("\nLiDAR Products:")
    for product in ['snow_depth', 'veg_height', 'dem']:
        product_path = os.path.join(lidar_dir, f"{product}.tif")
        if os.path.exists(product_path):
            with rasterio.open(product_path) as src:
                area = count_valid_pixels(product_path)
                all_shapes.add(str(src.shape))
                all_areas.add(round(area, 4))
                
                validation_results['products'][f"lidar_{product}"] = {
                    'shape': src.shape,
                    'crs': str(src.crs),
                    'area': round(area, 4),
                    'area_diff_percent': round(((area - shapefile_area) / shapefile_area) * 100, 2)
                }
                
                print(f"{product}:")
                print(f"  Shape: {src.shape}")
                print(f"  CRS: {src.crs}")
                print(f"  Area: {area:.4f} km²")
    
    # Check consistency
    validation_results['consistency'] = {
        'shapes_consistent': len(all_shapes) == 1,
        'areas_consistent': len(all_areas) == 1,
        'unique_shapes': list(all_shapes),
        'unique_areas': list(all_areas),
        'max_area_difference_percent': round(max([abs(a - shapefile_area) / shapefile_area * 100 for a in all_areas]), 2)
    }
    
    # Save validation results
    validation_file = os.path.join(output_dir, f"validation_{site}_{year}_{resolution}m.json")
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=4)
    
    return validation_results

def main():
    """Run validation checks for all combinations."""
    print("Starting automated validation checks...")
    
    # Create output directory for validation reports
    output_dir = os.path.join(BASE_OUTPUT_PATH, 'Validation_Reports')
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': []
    }
    
    # Process each combination
    for site in SITES:
        for year in SITES[site]['years']:
            for resolution in RESOLUTIONS:
                try:
                    result = check_products(site, year, resolution, output_dir)
                    summary['results'].append({
                        'site': site,
                        'year': year,
                        'resolution': resolution,
                        'shapes_consistent': result['consistency']['shapes_consistent'],
                        'areas_consistent': result['consistency']['areas_consistent'],
                        'max_area_difference_percent': result['consistency']['max_area_difference_percent']
                    })
                except Exception as e:
                    print(f"Error processing {site} {year} {resolution}m: {str(e)}")
    
    # Save overall summary
    summary_file = os.path.join(output_dir, 'validation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print final summary
    print("\nValidation Summary:")
    print("=" * 50)
    all_consistent = True
    for result in summary['results']:
        if not (result['shapes_consistent'] and result['areas_consistent']):
            all_consistent = False
        print(f"\n{result['site']} {result['year']} {result['resolution']}m:")
        print(f"  Shapes consistent: {result['shapes_consistent']}")
        print(f"  Areas consistent: {result['areas_consistent']}")
        print(f"  Max area difference: {result['max_area_difference_percent']}%")
    
    print("\nOverall Status:")
    print(f"All products consistent: {'Yes' if all_consistent else 'No'}")
    print(f"\nDetailed reports saved in: {output_dir}")

if __name__ == "__main__":
    main()