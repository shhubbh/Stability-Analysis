import rasterio
from rasterio.enums import Resampling
import numpy as np

def resample_dem(input_dem_path, output_dem_path, scale_factor):
    # Open the source DEM
    with rasterio.open(input_dem_path) as src:
        # Calculate the new shape
        new_height = int(src.height / scale_factor)
        new_width = int(src.width / scale_factor)
        
        # Read the DEM data
        dem_data = src.read(1)
        
        # Resample DEM
        dem_resampled = src.read(
            1,
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        )

        # Calculate the new affine transform
        transform = src.transform * src.transform.scale(
            (src.width / dem_resampled.shape[-1]),
            (src.height / dem_resampled.shape[-2])
        )

        # Update metadata with new dimensions and transform
        profile = src.profile
        profile.update({
            'transform': transform,
            'width': dem_resampled.shape[-1],
            'height': dem_resampled.shape[-2]
        })

        # Write resampled DEM to new file
        with rasterio.open(output_dem_path, 'w', **profile) as dst:
            dst.write(dem_resampled, 1)

# Usage
input_dem_path =r"H:\Others\Backup from 1TB Anvita HDD\Orissa\Data\4_Tiringpahada\Slope_Stability_WS\code_ws\07-08-2024\Test_dems\total pit.tif"
output_dem_path = r"H:\Others\Backup from 1TB Anvita HDD\Orissa\Data\4_Tiringpahada\Slope_Stability_WS\code_ws\07-08-2024\Input_Dems\totalpit1.tif"
scale_factor = 55  # Reduces resolution by a factor of 2

resample_dem(input_dem_path, output_dem_path, scale_factor)
