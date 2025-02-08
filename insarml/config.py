import os
from rasterio.warp import Resampling

# Base paths
BASE_INPUT_PATH = '/home/habeeb/insat_part2/uavsar-lidar2/Data'
BASE_OUTPUT_PATH = os.path.join(BASE_INPUT_PATH, 'Processed_Data')

# Resampling methods configuration
RESAMPLING_METHODS = {
    'amplitude': Resampling.bilinear,
    'coherence': Resampling.bilinear,
    'dem': Resampling.cubic,
    'wrapped_phase': Resampling.nearest,
    'unwrapped_phase': Resampling.nearest,
    'inc_angle': Resampling.bilinear,
    'snow_depth': Resampling.bilinear,
    'veg_height': Resampling.bilinear
}

# Target resolutions in meters
RESOLUTIONS = [3, 5, 10, 50, 100]

# Site configurations
SITES = {
    'Banner': {
        'shapefile': os.path.join(BASE_INPUT_PATH, 'Shape Files/Banner/banner_boundary.shp'),
        'dem': os.path.join(BASE_INPUT_PATH, 'Lidar/DEM/Banner Creek Summit/QSI_0.5M_PCDEM_USIDBN_20210917_20210917.tif'),
        'years': ['2020', '2021']
    },
    'Dry': {
        'shapefile': os.path.join(BASE_INPUT_PATH, 'Shape Files/Dry/dry_boundary.shp'),
        'dem': os.path.join(BASE_INPUT_PATH, 'Lidar/DEM/Dry Creek/QSI_0.5M_PCDEM_USIDDC_20210916_20210916.tif'),
        'years': ['2020']  # Dry Creek only has 2020 data
    },
    'Mores': {
        'shapefile': os.path.join(BASE_INPUT_PATH, 'Shape Files/Mores/mores_boundary.shp'),
        'dem': os.path.join(BASE_INPUT_PATH, 'Lidar/DEM/Mores Creek Summit/QSI_0.5M_PCDEM_USIDMC_20210917_20210917.tif'),
        'years': ['2020', '2021']
    }
}

# UAVSAR data paths
UAVSAR_DATA = {
    '2020': {
        'amplitude': os.path.join(BASE_INPUT_PATH, 'UAVSAR/2020/lowman_23205_20007-003_20011-003_0008d_s01_L090VV_01.amp2.grd.tiff'),
        'coherence': os.path.join(BASE_INPUT_PATH, 'UAVSAR/2020/lowman_23205_20007-003_20011-003_0008d_s01_L090VV_01.cor.grd.tiff'),
        'dem': os.path.join(BASE_INPUT_PATH, 'UAVSAR/2020/lowman_23205_20007-003_20011-003_0008d_s01_L090VV_01.hgt.grd.tiff'),
        'wrapped_phase': os.path.join(BASE_INPUT_PATH, 'UAVSAR/2020/lowman_23205_20007-003_20011-003_0008d_s01_L090VV_01.int.grd.tiff'),
        'unwrapped_phase': os.path.join(BASE_INPUT_PATH, 'UAVSAR/2020/lowman_23205_20007-003_20011-003_0008d_s01_L090VV_01.unw.grd.tiff'),
        'inc_angle': os.path.join(BASE_INPUT_PATH, 'UAVSAR/2020/lowman_23205_20011_003_200221_L090_CX_01.inc.tiff')
    },
    '2021': {
        'amplitude': os.path.join(BASE_INPUT_PATH, 'UAVSAR/2021/lowman_05208_21017-019_21019-019_0006d_s01_L090VV_01.amp2.grd.tiff'),
        'coherence': os.path.join(BASE_INPUT_PATH, 'UAVSAR/2021/lowman_05208_21017-019_21019-019_0006d_s01_L090VV_01.cor.grd.tiff'),
        'dem': os.path.join(BASE_INPUT_PATH, 'UAVSAR/2021/lowman_05208_21017-019_21019-019_0006d_s01_L090VV_01.hgt.grd.tiff'),
        'wrapped_phase': os.path.join(BASE_INPUT_PATH, 'UAVSAR/2021/lowman_05208_21017-019_21019-019_0006d_s01_L090VV_01.int.grd.tiff'),
        'unwrapped_phase': os.path.join(BASE_INPUT_PATH, 'UAVSAR/2021/lowman_05208_21017-019_21019-019_0006d_s01_L090VV_01.unw.grd.tiff'),
        'inc_angle': os.path.join(BASE_INPUT_PATH, 'UAVSAR/2021/lowman_05208_21019_019_210316_L090_CX_01.inc.tiff')
    }
}

# LiDAR data paths
LIDAR_DATA = {
    '2020': {
        'Banner': {
            'veg_height': os.path.join(BASE_INPUT_PATH, 'Lidar/2020/Banner Creek Summit/QSI_0.5M_PCVH_USIDBN_20200218_20200219.tif'),
            'snow_depth': os.path.join(BASE_INPUT_PATH, 'Lidar/2020/Banner Creek Summit/QSI_0.5M_PCSD_USIDBN_20200218_20200219.tif')
        },
        'Dry': {
            'veg_height': os.path.join(BASE_INPUT_PATH, 'Lidar/2020/Dry Creek/QSI_0.5M_PCVH_USIDDC_20200219_20200219.tif'),
            'snow_depth': os.path.join(BASE_INPUT_PATH, 'Lidar/2020/Dry Creek/QSI_0.5M_PCSD_USIDDC_20200219_20200219.tif')
        },
        'Mores': {
            'veg_height': os.path.join(BASE_INPUT_PATH, 'Lidar/2020/Mores Creek Summit/QSI_0.5M_PCVH_USIDMC_20200209_20200209.tif'),
            'snow_depth': os.path.join(BASE_INPUT_PATH, 'Lidar/2020/Mores Creek Summit/QSI_0.5M_PCSD_USIDMC_20200209_20200209.tif')
        }
    },
    '2021': {
        'Banner': {
            'veg_height': os.path.join(BASE_INPUT_PATH, 'Lidar/2021/Banner Creek Summit/QSI_0.5M_PCVH_USIDBN_20210315_20210315.tif'),
            'snow_depth': os.path.join(BASE_INPUT_PATH, 'Lidar/2021/Banner Creek Summit/QSI_0.5M_PCSD_USIDBN_20210315_20210315.tif')
        },
        'Mores': {
            'veg_height': os.path.join(BASE_INPUT_PATH, 'Lidar/2021/Mores Creek Summit/QSI_0.5M_PCVH_USIDMC_20210315_20210315.tif'),
            'snow_depth': os.path.join(BASE_INPUT_PATH, 'Lidar/2021/Mores Creek Summit/QSI_0.5M_PCSD_USIDMC_20210315_20210315.tif')
        }
    }
}

# Target CRS
TARGET_CRS = 'EPSG:6340'  # NAD83(2011) UTM zone 11N

# Create output directory structure
def create_directory_structure():
    """Create the complete directory structure for processed data."""
    for year in ['2020', '2021']:
        for site in SITES:
            # Skip if site doesn't have data for this year
            if year not in SITES[site]['years']:
                continue
                
            for resolution in RESOLUTIONS:
                # Create main resolution directory
                res_dir = os.path.join(BASE_OUTPUT_PATH, year, f"{site}", f"{resolution}m")
                
                # Create LiDAR and UAVSAR subdirectories
                lidar_dir = os.path.join(res_dir, 'LiDAR')
                uavsar_dir = os.path.join(res_dir, 'UAVSAR')
                validation_dir = os.path.join(res_dir, 'Validation')
                
                os.makedirs(lidar_dir, exist_ok=True)
                os.makedirs(uavsar_dir, exist_ok=True)
                os.makedirs(validation_dir, exist_ok=True)

if __name__ == "__main__":
    # Create directory structure
    create_directory_structure()
    print("Directory structure created successfully!")